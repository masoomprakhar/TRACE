"""Stop-line violation: a vehicle encroaches onto / crosses the stop line.

With tracking we detect the actual crossing (box bottom transitions past the
line between frames); without it we fall back to position within the stop-line
band. Vehicles that proceed *well* past the line while red are reported
separately as red-light violations.
"""

from __future__ import annotations

from trace_cv.core.types import Violation, ViolationType
from trace_cv.violation.base import ViolationContext, ViolationModule


class StopLineDetector(ViolationModule):
    type = ViolationType.STOP_LINE

    def check(self, ctx: ViolationContext) -> list[Violation]:
        stop = ctx.scene.stop_line
        if not stop.enabled or stop.y is None:
            return []
        y = stop.y
        band = max(20.0, 0.03 * ctx.height)

        out: list[Violation] = []
        for d in ctx.vehicles:
            bottom = d.bbox[3]
            hist = ctx.history(d.track_id)
            crossed = False
            if len(hist) >= 2:
                prev_bottom = hist[-2].bbox[3]
                crossed = prev_bottom <= y < bottom
            in_band = y <= bottom <= y + band
            if not (crossed or in_band):
                continue
            conf = min(0.99, d.confidence * (0.9 if crossed else 0.7))
            out.append(
                Violation(
                    type=self.type,
                    confidence=round(conf, 4),
                    bbox=d.bbox,
                    track_id=d.track_id,
                    vehicle_class=d.cls,
                    detail={"stop_line_y": y, "crossed": crossed},
                )
            )
        return out
