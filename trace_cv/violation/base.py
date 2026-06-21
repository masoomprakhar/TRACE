"""Shared scaffolding for violation modules: the per-frame context, the
track-history state, and the ViolationModule interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from trace_cv.core.config import SceneConfig, Thresholds
from trace_cv.core.types import (
    BBox,
    Detection,
    Point,
    VehicleClass,
    Violation,
    ViolationType,
    bbox_center,
)

if TYPE_CHECKING:  # avoid importing numpy at module load
    import numpy as np

VEHICLE_CLASSES = {
    VehicleClass.CAR,
    VehicleClass.MOTORCYCLE,
    VehicleClass.BUS,
    VehicleClass.TRUCK,
    VehicleClass.BICYCLE,
}


def is_vehicle(det: Detection) -> bool:
    return det.vehicle_class in VEHICLE_CLASSES


def expand_bbox(b: BBox, top: float = 0.6, side: float = 0.1) -> BBox:
    """Grow a box upward (and a little sideways) to capture riders who sit
    above a motorcycle's detected box."""
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    return (x1 - side * w, y1 - top * h, x2 + side * w, y2)


@dataclass
class TrackState:
    bbox: BBox
    frame_index: int

    @property
    def center(self) -> Point:
        return bbox_center(self.bbox)


@dataclass
class ViolationContext:
    frame: "np.ndarray"
    detections: list[Detection]
    scene: SceneConfig
    thresholds: Thresholds
    track_history: dict[int, list[TrackState]] = field(default_factory=dict)
    frame_index: int = 0
    fps: float = 15.0

    def history(self, track_id: Optional[int]) -> list[TrackState]:
        if track_id is None:
            return []
        return self.track_history.get(track_id, [])

    @property
    def vehicles(self) -> list[Detection]:
        return [d for d in self.detections if is_vehicle(d)]

    @property
    def height(self) -> int:
        return int(self.frame.shape[0]) if self.frame is not None else 720

    @property
    def width(self) -> int:
        return int(self.frame.shape[1]) if self.frame is not None else 1280


class ViolationModule(ABC):
    """Base class for all violation detectors."""

    type: ViolationType
    requires_model: bool = False

    @property
    def available(self) -> bool:
        """False if the module needs a model that isn't loaded. Modules that
        are unavailable are skipped silently — never emit fake violations."""
        return True

    @abstractmethod
    def check(self, ctx: ViolationContext) -> list[Violation]:
        ...
