"""Evaluation: classification metrics, detection mAP, OCR error rates, and
confidence calibration."""

from trace_cv.evaluation.calibration import (
    TemperatureScaler,
    expected_calibration_error,
)
from trace_cv.evaluation.metrics import (
    PRF,
    binary_prf,
    confusion_matrix,
    detection_map,
    multilabel_report,
    ocr_cer,
    ocr_exact_match,
)

__all__ = [
    "PRF",
    "TemperatureScaler",
    "expected_calibration_error",
    "binary_prf",
    "confusion_matrix",
    "detection_map",
    "multilabel_report",
    "ocr_cer",
    "ocr_exact_match",
]
