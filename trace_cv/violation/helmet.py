"""Helmet non-compliance for two-wheeler riders.

Needs an optional helmet model (a YOLO detection or classification checkpoint
whose classes distinguish helmet vs no-helmet/head). If no model is
configured the module reports `available == False` and is skipped — it never
fabricates a violation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from trace_cv.core.logging import get_logger
from trace_cv.core.types import Violation, ViolationType
from trace_cv.detection.roi import crop, rider_head_roi
from trace_cv.violation.base import ViolationContext, ViolationModule

log = get_logger("helmet")

_NO_HELMET_KEYS = ("no_helmet", "nohelmet", "without", "no-helmet", "head", "bare")


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


class HelmetModel:
    """Lazy wrapper around an Ultralytics helmet checkpoint or VioVision sklearn .pkl."""

    def __init__(self, weights: str, device: str = "cpu", conf: float = 0.25):
        self.weights = weights
        self.device = device
        self.conf = conf
        self._model = None
        self._sklearn = None
        self._tried = False

    def _ensure(self):
        if self._model is not None or self._sklearn is not None or self._tried:
            return
        self._tried = True
        if str(self.weights).endswith(".pkl"):
            try:
                from trace_cv.adapters.viovision_bridge import make_helmet_model

                self._sklearn = make_helmet_model(self.weights)
                if self._sklearn.available:
                    log.info("Helmet sklearn model loaded: %s", self.weights)
            except Exception as exc:  # pragma: no cover
                log.warning("Helmet sklearn model unavailable (%s).", exc)
            return
        try:
            from ultralytics import YOLO  # noqa: PLC0415

            from trace_cv.adapters.viovision_bridge import resolve_repo_path

            self._model = YOLO(str(resolve_repo_path(self.weights)))
            log.info("Helmet model loaded: %s", self.weights)
        except Exception as exc:  # pragma: no cover
            log.warning("Helmet model unavailable (%s).", exc)
            self._model = None

    @property
    def available(self) -> bool:
        self._ensure()
        return self._model is not None or (
            self._sklearn is not None and self._sklearn.available
        )

    def predict(self, region: np.ndarray) -> tuple[Optional[bool], float]:
        """Return (has_helmet, confidence). has_helmet is None when unsure."""
        self._ensure()
        if self._sklearn is not None and self._sklearn.available:
            return self._sklearn.predict_helmet(region)
        if self._model is None or region is None or region.size == 0:
            return None, 0.0
        try:
            r = self._model(region, conf=self.conf, device=self.device, verbose=False)[0]
        except Exception:  # pragma: no cover
            return None, 0.0
        names = r.names

        # Classification head.
        if getattr(r, "probs", None) is not None:
            top = int(r.probs.top1)
            conf = float(r.probs.top1conf)
            name = _norm(names[top])
            if any(k in name for k in _NO_HELMET_KEYS):
                return False, conf
            if "helmet" in name:
                return True, conf
            return None, conf

        # Detection head.
        no_h, has_h = 0.0, 0.0
        for b in getattr(r, "boxes", []) or []:
            name = _norm(names[int(b.cls)])
            cf = float(b.conf[0])
            if any(k in name for k in _NO_HELMET_KEYS):
                no_h = max(no_h, cf)
            elif "helmet" in name:
                has_h = max(has_h, cf)
        if no_h > 0 and no_h >= has_h:
            return False, no_h
        if has_h > 0:
            return True, has_h
        return None, 0.0


class HelmetDetector(ViolationModule):
    type = ViolationType.NO_HELMET
    requires_model = True

    def __init__(self, model: Optional[HelmetModel] = None):
        self.model = model

    @property
    def available(self) -> bool:
        return self.model is not None and self.model.available

    def check(self, ctx: ViolationContext) -> list[Violation]:
        if not self.available:
            return []
        out: list[Violation] = []
        for d in ctx.detections:
            if not d.vehicle_class.is_two_wheeler:
                continue
            region = crop(ctx.frame, rider_head_roi(d.bbox))
            has_helmet, conf = self.model.predict(region)
            if has_helmet is False and conf >= ctx.thresholds.helmet_conf:
                out.append(
                    Violation(
                        type=self.type,
                        confidence=round(conf, 4),
                        bbox=d.bbox,
                        track_id=d.track_id,
                        vehicle_class=d.cls,
                    )
                )
        return out
