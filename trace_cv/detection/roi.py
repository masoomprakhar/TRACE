"""Region-of-interest helpers: where on a vehicle to look for a helmet, a
seatbelt, or a license plate. Pure geometry + a clamped crop."""

from __future__ import annotations

import numpy as np

from trace_cv.core.types import BBox


def crop(img: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop with bounds clamping. Returns an empty array if degenerate."""
    h, w = img.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=img.dtype)
    return img[y1:y2, x1:x2]


def rider_head_roi(bbox: BBox, head_fraction: float = 0.35) -> BBox:
    """Top portion of a rider/motorcycle box — where a helmet would be."""
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2, y1 + (y2 - y1) * head_fraction)


def motorcycle_rider_roi(bbox: BBox, rider_fraction: float = 0.70) -> BBox:
    """Upper portion of a motorcycle box for CCTV rider-state CNN crops."""
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2, y1 + (y2 - y1) * rider_fraction)


def driver_roi(bbox: BBox) -> BBox:
    """Front-left quadrant of a car box — the driver's seat (right-hand
    drive). Top 70% of height, left 55% of width."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (x1, y1, x1 + 0.55 * w, y1 + 0.70 * h)


def plate_search_roi(bbox: BBox, two_wheeler: bool = False) -> BBox:
    """Lower band of a vehicle box where the plate usually sits."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    frac = 0.45 if two_wheeler else 0.4
    return (x1, y2 - h * frac, x2, y2)
