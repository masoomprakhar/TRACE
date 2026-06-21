"""
Primary detector adapter — YOLOv11 via Ultralytics.

This is the ONE place PyTorch enters the codebase. The guide is explicit
that this backbone is non-negotiable (section 2.A): YOLOv11n is fine-tuned
from COCO-pretrained weights, not trained from scratch, and everything
downstream reads off its detections. There is no sklearn substitute for
a real-time multi-class object detector at this accuracy/speed point, so
this module is intentionally the exception to the "prefer sklearn" rule.

Everything else in src/models/ is sklearn and has zero torch dependency —
if Ultralytics/torch isn't installed, those modules still import and run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.adapters.schema import BBox, Detection, DetectorClass

# Map Ultralytics' integer class indices to our fixed DetectorClass
# literals. This MUST match the order classes were defined in traffic.yaml
# at training time — keep these two files in sync by hand, there's no
# automatic check for this.
CLASS_ID_TO_NAME: dict[int, DetectorClass] = {
    0: "car",
    1: "two_wheeler",
    2: "person",
    3: "license_plate",
    4: "windshield",
    5: "signal_light",
}


class YoloDetectorAdapter:
    """Satisfies DetectorAdapter. imgsz=960 default per guide section 2.A
    (small plates/distant vehicles need the higher resolution)."""

    def __init__(self, weights_path: str | Path, conf_threshold: float = 0.35,
                 imgsz: int = 960, use_sahi: bool = False) -> None:
        # Import deferred to construction time so simply importing this
        # module doesn't require torch/ultralytics to be installed —
        # only instantiating this specific adapter does.
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.use_sahi = use_sahi
        self.model = YOLO(str(self.weights_path))

        if use_sahi:
            # SAHI (slicing-aided hyper inference) per guide section 2.A:
            # boosts small-object recall on high-res CCTV frames at
            # inference time with no retraining required.
            from sahi import AutoDetectionModel
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(self.weights_path),
                confidence_threshold=conf_threshold,
            )

    def predict(self, frame: np.ndarray) -> list[Detection]:
        if self.use_sahi:
            return self._predict_sahi(frame)
        return self._predict_direct(frame)

    def track(self, frame: np.ndarray, persist: bool = True) -> list[Detection]:
        """
        Same as predict(), but runs Ultralytics' built-in ByteTrack and
        populates Detection.track_id. Required for the stateful no-model
        violations (illegal_parking, wrong_side_driving in
        src/utils/geometry_violations.py) — both need a consistent
        track_id across frames, which plain predict() does not provide.

        persist=True keeps tracker state alive across calls on this same
        adapter instance, which is what you want for a continuous video
        stream from one camera. If you're juggling multiple camera feeds
        through one adapter, persist=False per source switch instead, or
        run one adapter instance per camera (simplest, recommended).

        Not available in SAHI mode — ByteTrack expects a single
        model.track() call per frame with internal state, which SAHI's
        tiled inference doesn't integrate with out of the box.
        """
        if self.use_sahi:
            raise NotImplementedError(
                "track() isn't supported with use_sahi=True. Use predict() "
                "with SAHI for single-frame detection, or disable SAHI if "
                "you need tracking."
            )

        results = self.model.track(
            frame, imgsz=self.imgsz, conf=self.conf_threshold,
            persist=persist, tracker="bytetrack.yaml", verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            # boxes.id is None for any box ByteTrack didn't assign a track
            # to yet (e.g. first frame before confirmation) — those boxes
            # still come through predict()-style with track_id=None rather
            # than being dropped, so detector-only consumers don't lose them.
            track_ids = boxes.id
            for i, box in enumerate(boxes):
                cls_id = int(box.cls.item())
                cls_name = CLASS_ID_TO_NAME.get(cls_id)
                if cls_name is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                track_id = int(track_ids[i].item()) if track_ids is not None else None
                detections.append(Detection(
                    cls=cls_name,
                    bbox=BBox(x1, y1, x2, y2),
                    confidence=float(box.conf.item()),
                    track_id=track_id,
                ))
        return detections

    def _predict_direct(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf_threshold, verbose=False
        )
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                cls_name = CLASS_ID_TO_NAME.get(cls_id)
                if cls_name is None:
                    continue  # unknown class id, skip rather than crash
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection(
                    cls=cls_name,
                    bbox=BBox(x1, y1, x2, y2),
                    confidence=float(box.conf.item()),
                ))
        return detections

    def _predict_sahi(self, frame: np.ndarray) -> list[Detection]:
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            frame, self.sahi_model,
            slice_height=512, slice_width=512,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        )
        detections: list[Detection] = []
        for pred in result.object_prediction_list:
            cls_name = CLASS_ID_TO_NAME.get(pred.category.id)
            if cls_name is None:
                continue
            bbox = pred.bbox.to_xyxy()
            detections.append(Detection(
                cls=cls_name,
                bbox=BBox(*map(int, bbox)),
                confidence=float(pred.score.value),
            ))
        return detections
