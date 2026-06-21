"""
The four violations the guide says need NO trained model — pure geometry,
counting, or motion logic on top of generic detector output (section 1):

    triple_riding        -> counting persons per two_wheeler detection
    stop_line             -> geometry vs. a calibration polygon
    illegal_parking       -> geometry + track stillness over time
    wrong_side_driving    -> motion vector vs. calibration direction

These all consume Detection objects (or sequences of them across frames
for the stateful ones) and never touch a model adapter. Keeping them
separate from src/models/ on purpose: there is nothing here to train,
fine-tune, or swap to USE_MOCKS=false.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.adapters.schema import BBox, Detection


# ---------------------------------------------------------------------------
# Triple riding: count persons whose bbox overlaps a two_wheeler bbox
# ---------------------------------------------------------------------------

def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = a.width() * a.height()
    area_b = b.width() * b.height()
    return inter / float(area_a + area_b - inter)


def check_triple_riding(detections: list[Detection],
                          overlap_threshold: float = 0.15,
                          rider_limit: int = 2) -> list[Detection]:
    """
    Returns the list of two_wheeler detections carrying more riders than
    `rider_limit` (2 for standard Indian traffic rules). A 'rider' is any
    person detection whose box overlaps the two-wheeler box above
    overlap_threshold IoU — loose on purpose since riders are seated
    close together and full-body IoU with the vehicle box is naturally low.
    """
    two_wheelers = [d for d in detections if d.cls == "two_wheeler"]
    persons = [d for d in detections if d.cls == "person"]

    violators = []
    for tw in two_wheelers:
        rider_count = sum(1 for p in persons if _iou(tw.bbox, p.bbox) > overlap_threshold)
        if rider_count > rider_limit:
            violators.append(tw)
    return violators


# ---------------------------------------------------------------------------
# Stop-line: point-in-polygon test against a calibrated stop-line region
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationPolygon:
    """A camera-specific region, set up once during calibration (e.g. the
    area beyond a painted stop line). Points are pixel-space (x, y)."""
    points: list[tuple[int, int]]

    def contains(self, x: int, y: int) -> bool:
        """Standard ray-casting point-in-polygon test."""
        n = len(self.points)
        inside = False
        px, py = self.points[-1]
        for cx, cy in self.points:
            if ((cy > y) != (py > y)) and (
                x < (px - cx) * (y - cy) / ((py - cy) or 1e-9) + cx
            ):
                inside = not inside
            px, py = cx, cy
        return inside

    @staticmethod
    def bbox_centroid(bbox: BBox) -> tuple[int, int]:
        return ((bbox.x1 + bbox.x2) // 2, (bbox.y1 + bbox.y2) // 2)


def check_stop_line(detections: list[Detection], stop_zone: CalibrationPolygon,
                      signal_is_red: bool) -> list[Detection]:
    """
    A vehicle violates the stop line only if the signal is red AND its
    centroid sits inside the calibrated stop_zone polygon. signal_is_red
    comes from the signal-state classifier — this function is pure
    geometry and takes that as a given input, it does not call the
    classifier itself.
    """
    if not signal_is_red:
        return []
    vehicles = [d for d in detections if d.cls in ("car", "two_wheeler")]
    violators = []
    for v in vehicles:
        cx, cy = CalibrationPolygon.bbox_centroid(v.bbox)
        if stop_zone.contains(cx, cy):
            violators.append(v)
    return violators


# ---------------------------------------------------------------------------
# Illegal parking: track stillness over time in a no-parking zone
# ---------------------------------------------------------------------------

@dataclass
class TrackHistory:
    """Per-track centroid history, used to detect stillness over time.
    Caller is responsible for feeding consistent track_ids (e.g. from a
    tracker like ByteTrack running on top of the detector)."""
    centroids: list[tuple[int, int]] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)

    def update(self, centroid: tuple[int, int], frame_id: int) -> None:
        self.centroids.append(centroid)
        self.frame_ids.append(frame_id)

    def is_still(self, movement_threshold_px: float = 12.0,
                  min_frames: int = 10) -> bool:
        if len(self.centroids) < min_frames:
            return False
        recent = self.centroids[-min_frames:]
        xs = [c[0] for c in recent]
        ys = [c[1] for c in recent]
        spread = max(xs) - min(xs) + max(ys) - min(ys)
        return spread < movement_threshold_px

    def duration_frames(self) -> int:
        if not self.frame_ids:
            return 0
        return self.frame_ids[-1] - self.frame_ids[0]


class ParkingTracker:
    """
    Stateful tracker for illegal_parking: holds TrackHistory per track_id
    across frames and flags a track as illegally parked once it has been
    still inside a no-parking zone for longer than `still_frames_threshold`.
    """

    def __init__(self, no_parking_zone: CalibrationPolygon,
                 still_frames_threshold: int = 150) -> None:
        self.zone = no_parking_zone
        self.still_frames_threshold = still_frames_threshold
        self.histories: dict[int, TrackHistory] = {}

    def update(self, detections: list[Detection], frame_id: int) -> list[Detection]:
        violators = []
        for d in detections:
            if d.cls not in ("car", "two_wheeler") or d.track_id is None:
                continue
            cx, cy = CalibrationPolygon.bbox_centroid(d.bbox)
            if not self.zone.contains(cx, cy):
                self.histories.pop(d.track_id, None)
                continue

            history = self.histories.setdefault(d.track_id, TrackHistory())
            history.update((cx, cy), frame_id)

            if history.is_still() and history.duration_frames() >= self.still_frames_threshold:
                violators.append(d)
        return violators


# ---------------------------------------------------------------------------
# Wrong-side driving: motion vector vs. a calibrated allowed direction
# ---------------------------------------------------------------------------

def _unit_vector(v: tuple[float, float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 1e-6 else arr


def check_wrong_side(track_history: TrackHistory, allowed_direction: tuple[float, float],
                       angle_threshold_deg: float = 100.0,
                       min_frames: int = 8) -> bool:
    """
    Compares a track's net motion vector (first centroid -> latest
    centroid) against the calibrated allowed_direction for that lane.
    Returns True if the angle between them exceeds angle_threshold_deg
    (default 100 degrees — stricter than 90 to avoid flagging legitimate
    lane changes/turns as wrong-side).
    """
    if len(track_history.centroids) < min_frames:
        return False

    start = np.array(track_history.centroids[0], dtype=np.float32)
    end = np.array(track_history.centroids[-1], dtype=np.float32)
    motion = _unit_vector(tuple(end - start))
    allowed = _unit_vector(allowed_direction)

    if np.linalg.norm(motion) < 1e-6:
        return False  # not enough net movement to judge direction

    cos_angle = float(np.clip(np.dot(motion, allowed), -1.0, 1.0))
    angle_deg = np.degrees(np.arccos(cos_angle))
    return angle_deg > angle_threshold_deg
