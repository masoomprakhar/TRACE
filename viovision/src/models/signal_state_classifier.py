"""
Signal-state classifier: red / yellow / green / unknown on cropped
traffic-light boxes.

Guide section 2.D explicitly offers two valid implementations and frames
the HSV heuristic as "a robust fast fallback," not a lesser option:
  - sklearn multiclass classifier (HOG + color hist -> SVC), trained on
    LISA/Bosch + your own crops.
  - HSV color heuristic: no training at all, just thresholded hue ranges
    on the brightest blob in the crop. Guide section 7: "Signal-state
    flaky at night? Fall back to the HSV color heuristic."

Both are implemented here behind one adapter-shaped class so the
orchestrator never needs to know which one is active. Toggle via
SignalStateClassifier(mode=...) or the pipeline config.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from src.adapters.schema import ClassifierResult
from src.models.base_crop_classifier import BaseCropClassifier

SignalMode = Literal["sklearn", "hsv_heuristic"]

# HSV hue ranges (OpenCV's H is 0-179). Tuned for typical LED traffic
# signal output; revisit against your own night/glare crops per guide
# section 4's hard-conditions subset.
_HUE_RANGES = {
    "red": [(0, 10), (170, 179)],   # red wraps around hue=0
    "yellow": [(15, 35)],
    "green": [(45, 90)],
}
_MIN_SATURATION = 80
_MIN_VALUE = 120


class SignalStateClassifierSklearn(BaseCropClassifier):
    # "unknown" is deliberately excluded here — it only arises at inference
    # from the HSV heuristic fallback (no training images carry that label).
    # Including it would cause train_test_split(stratify=) to crash with
    # "class has too few members" since it has 0 samples.
    class_names = ("red", "yellow", "green")
    use_clahe = False
    review_threshold = 0.55
    backend = "svm"


def hsv_heuristic_predict(crop_bgr: np.ndarray) -> ClassifierResult:
    """
    No-training fallback. Looks for the dominant bright, saturated hue in
    the crop and maps it to red/yellow/green. Returns 'unknown' with low
    confidence if nothing clears the saturation/value floor (this is the
    expected, honest outcome at night or in heavy glare — guide explicitly
    flags night/glare as the failure mode for this classifier, so 'unknown'
    routing to review is correct behavior, not an error case).
    """
    if crop_bgr.size == 0:
        return ClassifierResult.make("unknown", 0.0, threshold=0.55, source="hsv_heuristic")

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    bright_sat_mask = (s > _MIN_SATURATION) & (v > _MIN_VALUE)

    total_px = crop_bgr.shape[0] * crop_bgr.shape[1]
    best_color, best_frac = "unknown", 0.0

    for color, ranges in _HUE_RANGES.items():
        color_mask = np.zeros_like(bright_sat_mask)
        for lo, hi in ranges:
            color_mask |= (h >= lo) & (h <= hi)
        frac = float(np.sum(color_mask & bright_sat_mask)) / max(total_px, 1)
        if frac > best_frac:
            best_color, best_frac = color, frac

    # Fraction of matching pixels doubles as a crude confidence proxy.
    # Scale and clip so it lands in a sane [0, 1] range for the
    # needs_review comparison.
    confidence = min(1.0, best_frac * 8.0)
    if best_frac < 0.03:
        best_color, confidence = "unknown", 0.0

    return ClassifierResult.make(
        cls_=best_color, confidence=confidence, threshold=0.55, source="hsv_heuristic"
    )


class SignalStateClassifier:
    """
    Adapter-shaped wrapper. Satisfies CropClassifierAdapter regardless of
    which mode is active.
    """

    def __init__(self, mode: SignalMode = "hsv_heuristic") -> None:
        self.mode = mode
        self._sklearn_model: SignalStateClassifierSklearn | None = None
        if mode == "sklearn":
            self._sklearn_model = SignalStateClassifierSklearn()

    def fit(self, crops: list[np.ndarray], labels: list[str]) -> None:
        if self.mode != "sklearn":
            raise RuntimeError("fit() only applies in 'sklearn' mode.")
        self._sklearn_model.fit(crops, labels)

    def load(self, path: str) -> None:
        if self.mode != "sklearn":
            raise RuntimeError("load() only applies in 'sklearn' mode.")
        self._sklearn_model.load(path)

    def predict(self, crop: np.ndarray) -> ClassifierResult:
        if self.mode == "hsv_heuristic":
            return hsv_heuristic_predict(crop)
        return self._sklearn_model.predict(crop)
