"""Red-light running: a vehicle proceeds well past the stop line while the
signal is red.

Signal state comes from a fixed ROI (config) or, failing that, the largest
detected COCO 'traffic light'. Colour is classified in HSV — fast and robust,
no extra model required.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from trace_cv.core.types import (
    BBox,
    VehicleClass,
    Violation,
    ViolationType,
)
from trace_cv.detection.roi import crop
from trace_cv.violation.base import ViolationContext, ViolationModule


def classify_signal(signal_crop: np.ndarray) -> tuple[str, float]:
    """Return (state, confidence) where state ∈ {red, yellow, green, unknown}."""
    if signal_crop is None or signal_crop.size == 0:
        return "unknown", 0.0
    hsv = cv2.cvtColor(signal_crop, cv2.COLOR_BGR2HSV)
    masks = {
        "red": cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        | cv2.inRange(hsv, (170, 70, 50), (180, 255, 255)),
        "yellow": cv2.inRange(hsv, (15, 70, 50), (35, 255, 255)),
        "green": cv2.inRange(hsv, (40, 40, 50), (90, 255, 255)),
    }
    counts = {k: int(v.sum() // 255) for k, v in masks.items()}
    total = signal_crop.shape[0] * signal_crop.shape[1]
    state = max(counts, key=counts.get)
    lit = counts[state]
    if lit < 0.01 * total:
        return "unknown", 0.0
    conf = lit / (sum(counts.values()) + 1e-6)
    return state, float(conf)


class RedLightDetector(ViolationModule):
    type = ViolationType.RED_LIGHT

    def _signal_region(self, ctx: ViolationContext) -> Optional[BBox]:
        sig = ctx.scene.signal
        if sig.bbox:
            return tuple(sig.bbox)  # type: ignore[return-value]
        lights = [
            d for d in ctx.detections if d.vehicle_class == VehicleClass.TRAFFIC_LIGHT
        ]
        if not lights:
            return None
        # Largest / most confident traffic light.
        return max(lights, key=lambda d: d.confidence).bbox

    def check(self, ctx: ViolationContext) -> list[Violation]:
        if not ctx.scene.signal.enabled:
            return []
        region = self._signal_region(ctx)
        if region is None:
            return []
        state, sconf = classify_signal(crop(ctx.frame, region))
        if state != "red":
            return []

        stop = ctx.scene.stop_line
        if not stop.enabled or stop.y is None:
            return []
        y = stop.y
        band = max(20.0, 0.03 * ctx.height)

        out: list[Violation] = []
        for d in ctx.vehicles:
            # Bottom of the box is well past the stop line => ran the light.
            if d.bbox[3] > y + band:
                conf = min(0.99, 0.5 * sconf + 0.5 * d.confidence)
                out.append(
                    Violation(
                        type=self.type,
                        confidence=round(conf, 4),
                        bbox=d.bbox,
                        track_id=d.track_id,
                        vehicle_class=d.cls,
                        detail={"signal": state, "signal_conf": round(sconf, 3)},
                    )
                )
        return out
