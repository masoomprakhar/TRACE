"""A small, dependency-light multi-object tracker (greedy IoU matching).

Used when we drive frames ourselves or when Ultralytics' built-in ByteTrack
isn't available. Pure Python + the geometry helpers, so it is fully unit
testable without the ML stack. Assigns persistent track IDs in place.
"""

from __future__ import annotations

from trace_cv.core.types import Detection, bbox_iou


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: dict[int, dict] = {}  # id -> {bbox, cls, age}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Assign track_id to each detection (highest-confidence first) by
        greedy IoU match against live tracks; age out stale tracks."""
        used: set[int] = set()
        order = sorted(range(len(detections)), key=lambda i: -detections[i].confidence)

        for i in order:
            det = detections[i]
            best_iou, best_id = self.iou_threshold, None
            for tid, tr in self._tracks.items():
                if tid in used or tr["cls"] != det.cls:
                    continue
                iou = bbox_iou(det.bbox, tr["bbox"])
                if iou >= best_iou:
                    best_iou, best_id = iou, tid

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1

            det.track_id = best_id
            used.add(best_id)
            self._tracks[best_id] = {"bbox": det.bbox, "cls": det.cls, "age": 0}

        # Age & evict unmatched tracks.
        for tid in list(self._tracks):
            if tid not in used:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]

        return detections

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)
