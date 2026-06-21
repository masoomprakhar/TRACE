"""
Base class for the small binary crop classifiers (helmet, seatbelt).

Both are: crop in -> HOG+color features -> SVM -> binary label + confidence.
The only differences between helmet and seatbelt are the class names, the
CLAHE flag, and the confidence threshold — so they're thin subclasses
(see helmet_classifier.py, seatbelt_classifier.py) rather than copy-pasted
training scripts.

Why SVC over RandomForest here: HOG+color feature vectors are
high-dimensional and roughly linearly-ish separable after the kernel trick;
SVC with probability=True gives calibrated-enough confidence for the
needs_review routing. RandomForest is offered as a config swap if SVC
underperforms on your data — don't fight the model choice, just try both
and keep what scores better on the hard-conditions val split (guide
section 4).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.adapters.schema import ClassifierResult
from src.utils.features import extract_features

ClassifierBackend = Literal["svm", "random_forest"]


class BaseCropClassifier:
    """
    Not itself a Protocol implementer — subclasses fill in class_names,
    use_clahe, and review_threshold, then satisfy CropClassifierAdapter
    via the inherited `predict`.
    """

    class_names: tuple[str, ...] = ()       # e.g. ("no_helmet", "helmet")
    use_clahe: bool = False
    review_threshold: float = 0.55
    backend: ClassifierBackend = "svm"

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.clf = self._build_backend()
        self._fitted = False

    def _build_backend(self):
        if self.backend == "svm":
            return SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                        class_weight="balanced")
        elif self.backend == "random_forest":
            return RandomForestClassifier(n_estimators=300, max_depth=None,
                                            class_weight="balanced", n_jobs=-1)
        raise ValueError(f"Unknown backend: {self.backend}")

    # -- training -----------------------------------------------------

    def fit(self, crops: list[np.ndarray], labels: list[str]) -> None:
        """
        crops: list of BGR image crops (already cropped to the relevant
               ROI — head region for helmet, windshield region for seatbelt).
        labels: matching list of class name strings, must all be in
               self.class_names.
        """
        if not crops:
            raise ValueError("No training crops provided.")
        bad = set(labels) - set(self.class_names)
        if bad:
            raise ValueError(f"Labels {bad} not in expected classes {self.class_names}")

        X = np.stack([
            extract_features(c, use_clahe=self.use_clahe) for c in crops
        ])
        y = np.array(labels)

        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self._fitted = True

    # -- inference ------------------------------------------------------

    def predict(self, crop: np.ndarray) -> ClassifierResult:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} called before fit()/load(). "
                "Train it or load weights first."
            )
        feat = extract_features(crop, use_clahe=self.use_clahe).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)

        proba = self.clf.predict_proba(feat_scaled)[0]
        best_idx = int(np.argmax(proba))
        label = self.clf.classes_[best_idx]
        confidence = float(proba[best_idx])

        return ClassifierResult.make(
            cls_=label,
            confidence=confidence,
            threshold=self.review_threshold,
            source="sklearn",
        )

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "clf": self.clf,
                          "class_names": self.class_names}, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.scaler = payload["scaler"]
        self.clf = payload["clf"]
        self._fitted = True
