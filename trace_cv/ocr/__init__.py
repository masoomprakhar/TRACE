"""License-plate recognition: detection-agnostic OCR wrapper plus an
India-specific, format-aware corrector."""

from trace_cv.ocr.corrector import (
    correct_plate,
    format_plate,
    is_valid_plate,
    normalize_plate,
    plate_similarity,
)
from trace_cv.ocr.plate_ocr import PlateOCR

__all__ = [
    "PlateOCR",
    "correct_plate",
    "format_plate",
    "is_valid_plate",
    "normalize_plate",
    "plate_similarity",
]
