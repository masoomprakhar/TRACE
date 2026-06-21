"""License-plate OCR: TrOCR (fine-tuned) or EasyOCR fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from trace_cv.core.logging import get_logger
from trace_cv.core.types import BBox, Plate
from trace_cv.ocr.corrector import correct_plate, format_plate

log = get_logger("ocr")

_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class PlateOCR:
    def __init__(
        self,
        langs: Optional[list[str]] = None,
        gpu: bool = False,
        *,
        backend: str = "easyocr",
        trocr_path: str = "models/weights/trocr_plate",
    ):
        self.langs = langs or ["en"]
        self.gpu = gpu
        self.backend = (backend or "easyocr").lower()
        self.trocr_path = trocr_path
        self._reader = None
        self._trocr = None
        self._tried = False

    def _ensure_reader(self):
        if self._tried:
            return
        self._tried = True
        if self.backend == "trocr":
            try:
                from trace_cv.ocr.trocr_backend import TrOCRPlateReader

                self._trocr = TrOCRPlateReader(self.trocr_path, gpu=self.gpu)
                if self._trocr.available:
                    log.info("TrOCR plate reader loaded from %s", self.trocr_path)
                    return
            except Exception as exc:  # pragma: no cover
                log.warning("TrOCR unavailable (%s); falling back to EasyOCR.", exc)
        try:
            import easyocr  # noqa: PLC0415

            self._reader = easyocr.Reader(self.langs, gpu=self.gpu, verbose=False)
            log.info("EasyOCR loaded (langs=%s, gpu=%s)", self.langs, self.gpu)
        except Exception as exc:  # pragma: no cover
            log.warning("EasyOCR unavailable (%s); plate text disabled.", exc)
            self._reader = None

    def _trocr_ready(self) -> bool:
        return self._trocr is not None and self._trocr.available

    @property
    def available(self) -> bool:
        self._ensure_reader()
        return self._trocr_ready() or self._reader is not None

    @staticmethod
    def _prep(crop: np.ndarray) -> np.ndarray:
        img = cv2.copyMakeBorder(
            crop, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        longest = max(img.shape[:2])
        if longest < 360:
            scale = 360.0 / longest
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        return img

    def _read_easyocr(self, plate_crop: np.ndarray) -> tuple[str, float]:
        results = self._reader.readtext(
            self._prep(plate_crop),
            detail=1,
            allowlist=_ALLOWLIST,
            paragraph=False,
        )
        if not results:
            return "", 0.0
        results.sort(key=lambda r: r[0][0][0])
        raw = "".join(r[1] for r in results)
        conf = float(np.mean([r[2] for r in results]))
        return raw, conf

    def read(self, plate_crop: np.ndarray, bbox: Optional[BBox] = None) -> Plate:
        self._ensure_reader()
        if plate_crop is None or plate_crop.size == 0:
            return Plate(bbox=bbox)
        if not self._trocr_ready() and self._reader is None:
            return Plate(bbox=bbox)

        try:
            if self._trocr_ready():
                raw, conf = self._trocr.read(self._prep(plate_crop))
            else:
                raw, conf = self._read_easyocr(plate_crop)
        except Exception as exc:  # pragma: no cover
            log.debug("OCR read failed: %s", exc)
            return Plate(bbox=bbox)

        if not raw:
            return Plate(bbox=bbox)

        corrected, valid = correct_plate(raw)
        text = format_plate(corrected) if corrected else None
        adj = conf * (1.0 if valid else 0.85)
        return Plate(
            text=text,
            confidence=round(adj, 4),
            bbox=bbox,
            raw_text=raw,
            valid_format=valid,
        )
