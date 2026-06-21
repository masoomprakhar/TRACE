"""Illegal parking: a vehicle stays (nearly) stationary inside a no-parking
zone for longer than a threshold.

Stationarity is measured from track history — how many consecutive recent
frames the box stayed put (IoU above a threshold). Duration = frames / fps.
"""

from __future__ import annotations

from trace_cv.core.types import (
    Violation,
    ViolationType,
    bbox_center,
    bbox_iou,
    point_in_polygon,
)
from trace_cv.violation.base import ViolationContext, ViolationModule


class ParkingDetector(ViolationModule):
    type = ViolationType.ILLEGAL_PARKING

    def check(self, ctx: ViolationContext) -> list[Violation]:
        zones = ctx.scene.no_parking_zones
        if not zones:
            return []
        need_frames = ctx.thresholds.parking_seconds * ctx.fps

        out: list[Violation] = []
        for d in ctx.vehicles:
            center = bbox_center(d.bbox)
            zone = next(
                (z for z in zones if point_in_polygon(center, z.polygon)), None
            )
            if zone is None:
                continue

            hist = ctx.history(d.track_id)
            if len(hist) < 2:
                continue

            # Count consecutive recent frames where the box barely moved.
            stationary = 0
            for s in reversed(hist):
                if bbox_iou(d.bbox, s.bbox) >= ctx.thresholds.stationary_iou:
                    stationary += 1
                else:
                    break
            if stationary < need_frames:
                continue

            duration = stationary / max(ctx.fps, 1e-6)
            conf = min(0.97, 0.6 + 0.3 * d.confidence)
            out.append(
                Violation(
                    type=self.type,
                    confidence=round(conf, 4),
                    bbox=d.bbox,
                    track_id=d.track_id,
                    vehicle_class=d.cls,
                    detail={"zone": zone.name, "seconds": round(duration, 1)},
                )
            )
        return out
