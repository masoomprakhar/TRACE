"""
ANPR/OCR adapter. Guide section 2.E: do NOT train custom OCR.
Pipeline: YOLOv11 localizes plate -> CLAHE enhances -> OCR engine ->
Indian plate regex post-filter.

Default backend is fine-tuned TrOCR (``ocr_backend: trocr``). EasyOCR remains
available as a fallback when ``ocr_backend: easyocr``.

Install: pip install -r requirements-ml.txt  (transformers + easyocr)
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from src.adapters.schema import OCRResult
from src.utils.features import apply_clahe

INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")
REVIEW_CONFIDENCE_THRESHOLD = 0.6


def normalize_plate_text(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", raw).upper()


def matches_indian_plate_format(plate_text: str) -> bool:
    return bool(INDIAN_PLATE_REGEX.match(plate_text))


def get_ocr_adapter(
    backend: str = "trocr",
    *,
    model_path: str = "models/weights/trocr_plate",
    gpu: bool = False,
    lang: list[str] | None = None,
):
    """Factory: trocr (default) or easyocr."""
    backend = (backend or "trocr").lower()
    if backend == "trocr":
        from src.adapters.trocr_adapter import TrOCRAdapter

        return TrOCRAdapter(model_path=model_path, gpu=gpu)
    return EasyOCRAdapter(lang=lang)

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from src.adapters.schema import OCRResult
from src.utils.features import apply_clahe

INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")
REVIEW_CONFIDENCE_THRESHOLD = 0.6


def normalize_plate_text(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", raw).upper()


def matches_indian_plate_format(plate_text: str) -> bool:
    return bool(INDIAN_PLATE_REGEX.match(plate_text))


class EasyOCRAdapter:
    """
    Satisfies OCRAdapter using EasyOCR.
    Lazy-loads the reader on first predict() call so importing this
    module doesn't trigger a model download.
    """

    def __init__(self, lang: list[str] | None = None) -> None:
        self.lang = lang or ["en"]
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.lang, gpu=True, verbose=False)
        return self._reader

    def predict(self, plate_crop: np.ndarray) -> OCRResult:
        enhanced = apply_clahe(plate_crop)
        reader = self._get_reader()
        results = reader.readtext(enhanced, detail=1)

        candidates: list[str] = []
        confidences: list[float] = []
        for (_bbox, text, conf) in results:
            normalized = normalize_plate_text(text)
            if normalized:
                candidates.append(normalized)
                confidences.append(float(conf))

        if not candidates:
            return OCRResult(plate_text="", confidence=0.0,
                             needs_review=True, raw_candidates=[])

        best_idx = int(np.argmax(confidences))
        plate_text = candidates[best_idx]
        confidence = confidences[best_idx]
        format_ok = matches_indian_plate_format(plate_text)
        needs_review = (confidence < REVIEW_CONFIDENCE_THRESHOLD) or not format_ok

        return OCRResult(
            plate_text=plate_text,
            confidence=confidence,
            needs_review=needs_review,
            raw_candidates=candidates,
        )


class EnsembleOCRAdapter:
    """Confidence-voted ensemble across multiple OCR engine instances."""

    def __init__(self, engines: list) -> None:
        if not engines:
            raise ValueError("EnsembleOCRAdapter needs at least one engine.")
        self.engines = engines

    def predict(self, plate_crop: np.ndarray) -> OCRResult:
        results = [engine.predict(plate_crop) for engine in self.engines]
        valid = [r for r in results if r.plate_text]

        if not valid:
            return OCRResult(plate_text="", confidence=0.0,
                             needs_review=True, raw_candidates=[])

        vote_counts = Counter(r.plate_text for r in valid)
        top_text, top_count = vote_counts.most_common(1)[0]
        tied_texts = [t for t, c in vote_counts.items() if c == top_count]

        if len(tied_texts) > 1:
            best = max(valid, key=lambda r: r.confidence)
        else:
            matching = [r for r in valid if r.plate_text == top_text]
            best = max(matching, key=lambda r: r.confidence)

        format_ok = matches_indian_plate_format(best.plate_text)
        needs_review = (best.confidence < REVIEW_CONFIDENCE_THRESHOLD) or not format_ok

        return OCRResult(
            plate_text=best.plate_text,
            confidence=best.confidence,
            needs_review=needs_review,
            raw_candidates=[r.plate_text for r in valid],
        )


class PaddleOCRAdapter:
    """Stub — PaddleOCR 3.x is broken on most hardware. Use EasyOCRAdapter."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "PaddleOCR 3.x has an unimplemented PIR op on most hardware. "
            "Use EasyOCRAdapter instead: pip install easyocr"
        )
