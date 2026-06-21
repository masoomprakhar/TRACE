"""Seatbelt non-compliance for four-wheeler drivers.

Like the helmet module, needs an optional model. The model is expected to
distinguish seatbelt / no_seatbelt / occluded — the 'occluded' class is what
prevents false positives from window glare or A-pillars (we never flag an
occluded driver).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from trace_cv.core.logging import get_logger
from trace_cv.core.types import Violation, ViolationType
from trace_cv.detection.roi import crop, driver_roi
from trace_cv.violation.base import ViolationContext, ViolationModule

log = get_logger("seatbelt")

_NO_BELT_KEYS = ("no_seatbelt", "noseatbelt", "without", "no-belt", "unbelted", "no_belt")
_OCCLUDED_KEYS = ("occluded", "unknown", "unclear")


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


class SeatbeltModel:
    """Lazy wrapper around an Ultralytics seatbelt checkpoint or VioVision sklearn .pkl."""

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
                from trace_cv.adapters.viovision_bridge import make_seatbelt_model

                self._sklearn = make_seatbelt_model(self.weights)
                if self._sklearn.available:
                    log.info("Seatbelt sklearn model loaded: %s", self.weights)
            except Exception as exc:  # pragma: no cover
                log.warning("Seatbelt sklearn model unavailable (%s).", exc)
            return
        try:
            from ultralytics import YOLO  # noqa: PLC0415

            from trace_cv.adapters.viovision_bridge import resolve_repo_path

            self._model = YOLO(str(resolve_repo_path(self.weights)))
            log.info("Seatbelt model loaded: %s", self.weights)
        except Exception as exc:  # pragma: no cover
            log.warning("Seatbelt model unavailable (%s).", exc)
            self._model = None

    @property
    def available(self) -> bool:
        self._ensure()
        return self._model is not None or (
            self._sklearn is not None and self._sklearn.available
        )

    def predict(self, region: np.ndarray) -> tuple[str, float]:
        """Return (label, confidence); label ∈ {belt, no_belt, occluded, unknown}."""
        self._ensure()
        if self._sklearn is not None and self._sklearn.available:
            return self._sklearn.predict_seatbelt(region)
        if self._model is None or region is None or region.size == 0:
            return "unknown", 0.0
        try:
            r = self._model(region, conf=self.conf, device=self.device, verbose=False)[0]
        except Exception:  # pragma: no cover
            return "unknown", 0.0
        names = r.names

        if getattr(r, "probs", None) is not None:
            top = int(r.probs.top1)
            conf = float(r.probs.top1conf)
            name = _norm(names[top])
            return _label(name), conf

        best_label, best_conf = "unknown", 0.0
        for b in getattr(r, "boxes", []) or []:
            name = _norm(names[int(b.cls)])
            cf = float(b.conf[0])
            if cf > best_conf:
                best_label, best_conf = _label(name), cf
        return best_label, best_conf


def _label(name: str) -> str:
    if any(k in name for k in _OCCLUDED_KEYS):
        return "occluded"
    if any(k in name for k in _NO_BELT_KEYS):
        return "no_belt"
    if "belt" in name:
        return "belt"
    return "unknown"


class SeatbeltDetector(ViolationModule):
    type = ViolationType.NO_SEATBELT
    requires_model = True

    def __init__(self, model: Optional[SeatbeltModel] = None):
        self.model = model

    @property
    def available(self) -> bool:
        return self.model is not None and self.model.available

    def check(self, ctx: ViolationContext) -> list[Violation]:
        if not self.available:
            return []
        out: list[Violation] = []
        for d in ctx.detections:
            if not d.vehicle_class.is_four_wheeler:
                continue
            region = crop(ctx.frame, driver_roi(d.bbox))
            label, conf = self.model.predict(region)
            # Never flag an occluded driver.
            if label == "no_belt" and conf >= ctx.thresholds.seatbelt_conf:
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
