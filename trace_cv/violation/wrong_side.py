"""Wrong-side driving: a vehicle moves against the legal flow for its
carriageway.

Needs tracking (a motion vector over the track's history) and a one-time
per-camera calibration: the pixel column that divides the two carriageways
and the legal direction of travel on the right-hand side.
"""

from __future__ import annotations

from trace_cv.core.types import Violation, ViolationType, bbox_center
from trace_cv.violation.base import ViolationContext, ViolationModule

_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _direction(dx: float, dy: float, min_move: float) -> str | None:
    """Dominant motion direction, or None if movement is below threshold."""
    if (dx * dx + dy * dy) ** 0.5 < min_move:
        return None
    if abs(dy) >= abs(dx):
        return "down" if dy > 0 else "up"
    return "right" if dx > 0 else "left"


class WrongSideDetector(ViolationModule):
    type = ViolationType.WRONG_SIDE

    def check(self, ctx: ViolationContext) -> list[Violation]:
        lane = ctx.scene.lane
        if not lane.enabled or lane.divider_x is None:
            return []
        min_move = max(5.0, 0.01 * ctx.width)

        out: list[Violation] = []
        for d in ctx.vehicles:
            hist = ctx.history(d.track_id)
            if len(hist) < 2:
                continue
            x0, y0 = hist[0].center
            x1, y1 = hist[-1].center
            moving = _direction(x1 - x0, y1 - y0, min_move)
            if moving is None:
                continue

            cx = bbox_center(d.bbox)[0]
            on_right = cx >= lane.divider_x
            expected = (
                lane.correct_direction
                if on_right
                else _OPPOSITE.get(lane.correct_direction, lane.correct_direction)
            )
            if moving != expected:
                conf = min(0.95, 0.6 + 0.35 * d.confidence)
                out.append(
                    Violation(
                        type=self.type,
                        confidence=round(conf, 4),
                        bbox=d.bbox,
                        track_id=d.track_id,
                        vehicle_class=d.cls,
                        detail={
                            "moving": moving,
                            "expected": expected,
                            "side": "right" if on_right else "left",
                        },
                    )
                )
        return out
