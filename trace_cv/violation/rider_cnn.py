"""Rider-state CNN: multi-label no_helmet + triple_riding from motorcycle crops."""

from __future__ import annotations

from typing import Optional

from trace_cv.adapters.rider_cnn import RiderCNNModel
from trace_cv.core.types import Violation, ViolationType
from trace_cv.detection.roi import crop, motorcycle_rider_roi
from trace_cv.violation.base import ViolationContext, ViolationModule


class RiderCNNDetector(ViolationModule):
    """Emits NO_HELMET and/or TRIPLE_RIDING from one CNN forward pass."""

    type = ViolationType.NO_HELMET
    requires_model = True

    def __init__(self, model: Optional[RiderCNNModel] = None):
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
            region = crop(ctx.frame, motorcycle_rider_roi(d.bbox))
            preds = self.model.predict(region)

            no_h, no_conf = preds.get("no_helmet", (False, 0.0))
            if no_h and no_conf >= ctx.thresholds.helmet_conf:
                out.append(
                    Violation(
                        type=ViolationType.NO_HELMET,
                        confidence=round(no_conf, 4),
                        bbox=d.bbox,
                        track_id=d.track_id,
                        vehicle_class=d.cls,
                        detail={"source": "rider_cnn"},
                    )
                )

            triple, tri_conf = preds.get("triple_riding", (False, 0.0))
            if triple and tri_conf >= ctx.thresholds.helmet_conf:
                out.append(
                    Violation(
                        type=ViolationType.TRIPLE_RIDING,
                        confidence=round(tri_conf, 4),
                        bbox=d.bbox,
                        track_id=d.track_id,
                        vehicle_class=d.cls,
                        detail={"source": "rider_cnn"},
                    )
                )
        return out
