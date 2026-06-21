"""Detection & tracking: YOLO detector (lazy), a dependency-light IOU
tracker, and ROI helpers for rider/driver/plate crops."""

from trace_cv.detection.detector import Detector
from trace_cv.detection.roi import (
    crop,
    driver_roi,
    plate_search_roi,
    rider_head_roi,
)
from trace_cv.detection.tracker import SimpleTracker

__all__ = [
    "Detector",
    "SimpleTracker",
    "crop",
    "driver_roi",
    "plate_search_roi",
    "rider_head_roi",
]
