"""TrOCR adapter for VioVision OCR pipeline."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from src.adapters.schema import OCRResult
from src.utils.features import apply_clahe

INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")
REVIEW_CONFIDENCE_THRESHOLD = 0.6


def normalize_plate_text(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", raw).upper()


def matches_indian_plate_format(plate_text: str) -> bool:
    return bool(INDIAN_PLATE_REGEX.match(plate_text))


class TrOCRAdapter:
    """TrOCR plate reader — drop-in replacement for EasyOCRAdapter."""

    def __init__(self, model_path: str = "models/weights/trocr_plate", gpu: bool = False) -> None:
        self.model_path = model_path
        self.gpu = gpu
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            root = Path(__file__).resolve().parents[3]
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from trace_cv.ocr.trocr_backend import TrOCRPlateReader

            self._reader = TrOCRPlateReader(self.model_path, gpu=self.gpu)
        return self._reader

    def predict(self, plate_crop: np.ndarray) -> OCRResult:
        enhanced = apply_clahe(plate_crop)
        reader = self._get_reader()
        if not reader.available:
            return OCRResult(plate_text="", confidence=0.0, needs_review=True, raw_candidates=[])
        text, conf = reader.read(enhanced)
        norm = normalize_plate_text(text)
        valid = matches_indian_plate_format(norm) if norm else False
        needs_review = conf < REVIEW_CONFIDENCE_THRESHOLD or not valid
        return OCRResult(
            plate_text=norm,
            confidence=conf,
            needs_review=needs_review,
            raw_candidates=[norm] if norm else [],
        )
