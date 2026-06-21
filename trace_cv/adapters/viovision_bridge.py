"""Bridge VioVision-trained sklearn classifiers into TRACE.

Loads helmet/seatbelt SVM pickles produced by viovision/scripts/train_crop_classifier.py
and reuses VioVision's HOG+color feature extraction at inference time.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from trace_cv.core.logging import get_logger

log = get_logger("viovision_bridge")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VIOVISION_SRC = _REPO_ROOT / "viovision" / "src"


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve model paths relative to the repository root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def _ensure_viovision_importable() -> None:
    src = str(_VIOVISION_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_extract_features():
    _ensure_viovision_importable()
    from utils.features import extract_features  # noqa: PLC0415

    return extract_features


class SklearnCropModel:
    """Sklearn SVM/RF crop classifier loaded from a VioVision .pkl checkpoint."""

    def __init__(
        self,
        weights: str,
        *,
        use_clahe: bool = False,
        no_class_keys: tuple[str, ...] = (),
        positive_class_keys: tuple[str, ...] = (),
    ):
        self.weights = str(resolve_repo_path(weights))
        self.use_clahe = use_clahe
        self.no_class_keys = no_class_keys
        self.positive_class_keys = positive_class_keys
        self._scaler = None
        self._clf = None
        self._class_names: tuple[str, ...] = ()
        self._tried = False

    def _ensure(self) -> None:
        if self._clf is not None or self._tried:
            return
        self._tried = True
        path = Path(self.weights)
        if not path.exists():
            log.warning("Sklearn weights not found: %s", path)
            return
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            self._scaler = payload["scaler"]
            self._clf = payload["clf"]
            self._class_names = tuple(payload.get("class_names", ()))
            log.info("Sklearn crop model loaded: %s", path)
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to load sklearn model %s: %s", path, exc)
            self._clf = None

    @property
    def available(self) -> bool:
        self._ensure()
        return self._clf is not None

    def _predict_label(self, region: np.ndarray) -> tuple[str, float]:
        self._ensure()
        if self._clf is None or region is None or region.size == 0:
            return "unknown", 0.0
        try:
            extract_features = _load_extract_features()
            feat = extract_features(region, use_clahe=self.use_clahe).reshape(1, -1)
            feat_scaled = self._scaler.transform(feat)
            proba = self._clf.predict_proba(feat_scaled)[0]
            best_idx = int(np.argmax(proba))
            label = str(self._clf.classes_[best_idx])
            return label, float(proba[best_idx])
        except Exception as exc:  # pragma: no cover
            log.warning("Sklearn inference failed: %s", exc)
            return "unknown", 0.0

    @staticmethod
    def _norm(name: str) -> str:
        return name.strip().lower().replace(" ", "_")

    def predict_helmet(self, region: np.ndarray) -> tuple[Optional[bool], float]:
        """Return (has_helmet, confidence). has_helmet is None when unsure."""
        label, conf = self._predict_label(region)
        name = self._norm(label)
        if any(k in name for k in self.no_class_keys):
            return False, conf
        if any(k in name for k in self.positive_class_keys) or "helmet" in name:
            if "no" in name and "helmet" in name:
                return False, conf
            return True, conf
        return None, conf

    def predict_seatbelt(self, region: np.ndarray) -> tuple[str, float]:
        """Return label in {belt, no_belt, occluded, unknown}."""
        label, conf = self._predict_label(region)
        name = self._norm(label)
        if any(k in name for k in ("occluded", "unknown", "unclear")):
            return "occluded", conf
        if any(k in name for k in self.no_class_keys):
            return "no_belt", conf
        if "belt" in name:
            return "belt", conf
        return "unknown", conf


def make_helmet_model(weights: str) -> SklearnCropModel:
    return SklearnCropModel(
        weights,
        use_clahe=False,
        no_class_keys=("no_helmet", "nohelmet", "without", "no-helmet", "head", "bare"),
        positive_class_keys=("helmet",),
    )


def make_seatbelt_model(weights: str) -> SklearnCropModel:
    return SklearnCropModel(
        weights,
        use_clahe=True,
        no_class_keys=("no_seatbelt", "noseatbelt", "without", "no-belt", "unbelted", "no_belt"),
        positive_class_keys=("seatbelt", "belt"),
    )


DEFAULT_VIOVISION_CLASS_MAP: dict[str, str] = {
    "two_wheeler": "motorcycle",
    "signal_light": "traffic_light",
    "license_plate": "license_plate",
    "windshield": "windshield",
    "car": "car",
    "person": "person",
}
