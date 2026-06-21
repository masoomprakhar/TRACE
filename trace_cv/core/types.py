"""Shared data types and geometry helpers — the contract every TRACE
module depends on.

Deliberately dependency-light: no numpy / torch imports at module load, so
this can be imported (and unit-tested) without the ML stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

# Axis-aligned bounding box in pixel coordinates: (x1, y1, x2, y2).
BBox = tuple[float, float, float, float]
Point = tuple[float, float]


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class VehicleClass(str, Enum):
    """Road-user classes. Values align with COCO class names so detector
    output maps directly."""

    PERSON = "person"
    BICYCLE = "bicycle"
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"
    TRAFFIC_LIGHT = "traffic_light"
    OTHER = "other"

    @classmethod
    def from_name(cls, name: str) -> "VehicleClass":
        normalized = (name or "").strip().lower().replace(" ", "_")
        try:
            return cls(normalized)
        except ValueError:
            return cls.OTHER

    @property
    def is_two_wheeler(self) -> bool:
        return self in (VehicleClass.MOTORCYCLE, VehicleClass.BICYCLE)

    @property
    def is_four_wheeler(self) -> bool:
        return self in (VehicleClass.CAR, VehicleClass.BUS, VehicleClass.TRUCK)


class ViolationType(str, Enum):
    NO_HELMET = "no_helmet"
    NO_SEATBELT = "no_seatbelt"
    TRIPLE_RIDING = "triple_riding"
    WRONG_SIDE = "wrong_side"
    STOP_LINE = "stop_line"
    RED_LIGHT = "red_light"
    ILLEGAL_PARKING = "illegal_parking"

    @property
    def label(self) -> str:
        return _VIOLATION_LABELS[self]


_VIOLATION_LABELS = {
    ViolationType.NO_HELMET: "No Helmet",
    ViolationType.NO_SEATBELT: "No Seatbelt",
    ViolationType.TRIPLE_RIDING: "Triple Riding",
    ViolationType.WRONG_SIDE: "Wrong-Side Driving",
    ViolationType.STOP_LINE: "Stop-Line Violation",
    ViolationType.RED_LIGHT: "Red-Light Violation",
    ViolationType.ILLEGAL_PARKING: "Illegal Parking",
}


# --------------------------------------------------------------------------- #
# Geometry helpers (pure Python, no numpy)
# --------------------------------------------------------------------------- #
def bbox_area(b: BBox) -> float:
    w = max(0.0, b[2] - b[0])
    h = max(0.0, b[3] - b[1])
    return w * h


def bbox_center(b: BBox) -> Point:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_overlap_ratio(inner: BBox, outer: BBox) -> float:
    """area(inner ∩ outer) / area(inner) — how much of `inner` sits inside
    `outer`. Used to decide whether a person belongs to a vehicle."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a = bbox_area(inner)
    return inter / a if a > 0 else 0.0


def bbox_contains_point(b: BBox, p: Point) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def point_in_polygon(point: Point, polygon: list) -> bool:
    """Ray-casting point-in-polygon test. `polygon` is a list of (x, y)."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """A single detected object."""

    cls: str
    bbox: BBox
    confidence: float
    track_id: Optional[int] = None

    @property
    def vehicle_class(self) -> VehicleClass:
        return VehicleClass.from_name(self.cls)

    @property
    def center(self) -> Point:
        return bbox_center(self.bbox)

    def to_dict(self) -> dict:
        return {
            "cls": self.cls,
            "bbox": [round(v, 1) for v in self.bbox],
            "confidence": round(self.confidence, 4),
            "track_id": self.track_id,
        }


@dataclass
class Plate:
    """License-plate recognition result."""

    text: Optional[str] = None
    confidence: float = 0.0
    bbox: Optional[BBox] = None
    raw_text: Optional[str] = None        # OCR output before correction
    valid_format: bool = False            # passes Indian-plate format check

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox] if self.bbox else None,
            "raw_text": self.raw_text,
            "valid_format": self.valid_format,
        }


@dataclass
class Violation:
    """A detected traffic violation with a calibrated confidence score."""

    type: ViolationType
    confidence: float
    bbox: BBox
    track_id: Optional[int] = None
    vehicle_class: Optional[str] = None
    plate: Optional[Plate] = None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "label": self.type.label,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox],
            "track_id": self.track_id,
            "vehicle_class": self.vehicle_class,
            "plate": self.plate.to_dict() if self.plate else None,
            "detail": self.detail,
        }


@dataclass
class FrameResult:
    """Everything TRACE extracted from one frame/image."""

    detections: list[Detection] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    plates: dict[int, Plate] = field(default_factory=dict)  # track_id -> plate
    frame_index: int = 0
    timestamp: datetime = field(default_factory=utcnow)
    processing_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp.isoformat(),
            "processing_ms": round(self.processing_ms, 2),
            "detections": [d.to_dict() for d in self.detections],
            "violations": [v.to_dict() for v in self.violations],
            "plates": {str(k): v.to_dict() for k, v in self.plates.items()},
        }
