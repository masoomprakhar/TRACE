"""TrOCR-based Indian plate line reader for TRACE."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from trace_cv.adapters.viovision_bridge import resolve_repo_path
from trace_cv.core.logging import get_logger

log = get_logger("trocr")


class TrOCRPlateReader:
    def __init__(self, model_path: str, gpu: bool = False):
        self.model_path = str(resolve_repo_path(model_path))
        self.gpu = gpu
        self._processor = None
        self._model = None
        self._tried = False

    def _ensure(self) -> None:
        if self._model is not None or self._tried:
            return
        self._tried = True
        path = Path(self.model_path)
        if not path.exists():
            log.warning("TrOCR model path not found: %s", path)
            return
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            self._processor = TrOCRProcessor.from_pretrained(path)
            self._model = VisionEncoderDecoderModel.from_pretrained(path)
            if self.gpu:
                import torch

                self._model.to("cuda" if torch.cuda.is_available() else "cpu")
            self._model.eval()
            log.info("TrOCR loaded from %s", path)
        except Exception as exc:  # pragma: no cover
            log.warning("TrOCR load failed: %s", exc)
            self._model = None

    @property
    def available(self) -> bool:
        self._ensure()
        return self._model is not None

    def read(self, plate_crop: np.ndarray) -> tuple[str, float]:
        self._ensure()
        if self._model is None or plate_crop is None or plate_crop.size == 0:
            return "", 0.0
        try:
            import torch
            from PIL import Image

            rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).convert("RGB")
            pixel = self._processor(pil, return_tensors="pt").pixel_values
            device = next(self._model.parameters()).device
            pixel = pixel.to(device)
            with torch.no_grad():
                ids = self._model.generate(pixel, max_new_tokens=16)
            text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
            text = "".join(c for c in text.upper() if c.isalnum())
            return text, 0.85
        except Exception as exc:  # pragma: no cover
            log.debug("TrOCR infer failed: %s", exc)
            return "", 0.0
