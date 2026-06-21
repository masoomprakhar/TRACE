"""Triple riding: three or more people on a two-wheeler.

Counts distinct persons that sit on/around a motorcycle's (upward-expanded)
box. With tracking, the same person keeps one ID across frames so they are
not double-counted.
"""

from __future__ import annotations

from trace_cv.core.types import (
    VehicleClass,
    Violation,
    ViolationType,
    bbox_contains_point,
    bbox_overlap_ratio,
)
from trace_cv.violation.base import ViolationContext, ViolationModule, expand_bbox


class TripleRidingDetector(ViolationModule):
    type = ViolationType.TRIPLE_RIDING

    def check(self, ctx: ViolationContext) -> list[Violation]:
        persons = [d for d in ctx.detections if d.vehicle_class == VehicleClass.PERSON]
        two_wheelers = [d for d in ctx.detections if d.vehicle_class.is_two_wheeler]
        min_riders = ctx.thresholds.triple_riding_min
        overlap_t = ctx.thresholds.rider_overlap

        out: list[Violation] = []
        for tw in two_wheelers:
            zone = expand_bbox(tw.bbox)
            riders = [
                p
                for p in persons
                if bbox_overlap_ratio(p.bbox, zone) >= overlap_t
                or bbox_contains_point(zone, p.center)
            ]
            n = len(riders)
            if n >= min_riders:
                avg_person = sum(p.confidence for p in riders) / n
                # More riders over the threshold => higher confidence.
                conf = min(0.99, 0.5 * (tw.confidence + avg_person) * (n / min_riders))
                out.append(
                    Violation(
                        type=self.type,
                        confidence=round(conf, 4),
                        bbox=tw.bbox,
                        track_id=tw.track_id,
                        vehicle_class=tw.cls,
                        detail={"riders": n},
                    )
                )
        return out
