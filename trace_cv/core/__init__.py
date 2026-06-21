"""Core contracts shared across every TRACE module: data types, geometry
helpers, configuration, and logging."""

from trace_cv.core.types import (
    BBox,
    Detection,
    FrameResult,
    Plate,
    VehicleClass,
    Violation,
    ViolationType,
    bbox_area,
    bbox_center,
    bbox_contains_point,
    bbox_iou,
    bbox_overlap_ratio,
    point_in_polygon,
    utcnow,
)

__all__ = [
    "BBox",
    "Detection",
    "FrameResult",
    "Plate",
    "VehicleClass",
    "Violation",
    "ViolationType",
    "bbox_area",
    "bbox_center",
    "bbox_contains_point",
    "bbox_iou",
    "bbox_overlap_ratio",
    "point_in_polygon",
    "utcnow",
]
