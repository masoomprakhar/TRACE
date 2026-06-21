"""TracePipeline — the end-to-end orchestrator.

preprocess -> detect (+track) -> violation engine -> plate OCR -> evidence
-> persistence. Heavy models are lazy, so constructing the pipeline is cheap
and it runs (degrading gracefully) even before the ML stack is installed.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from trace_cv.adapters.roboflow_plate import RoboflowPlateDetector
from trace_cv.core.config import Settings, load_settings
from trace_cv.core.logging import get_logger
from trace_cv.core.types import Detection, Violation
from trace_cv.detection.detector import Detector
from trace_cv.detection.roi import crop, plate_search_roi
from trace_cv.detection.tracker import SimpleTracker
from trace_cv.evidence.builder import EvidenceBuilder
from trace_cv.ocr.plate_ocr import PlateOCR
from trace_cv.preprocessing.pipeline import AdaptivePreprocessor
from trace_cv.storage.db import Repository
from trace_cv.violation.engine import ViolationEngine
from trace_cv.violation.helmet import HelmetDetector
from trace_cv.violation.rider_cnn import RiderCNNDetector
from trace_cv.violation.seatbelt import SeatbeltDetector

log = get_logger("pipeline")


class TracePipeline:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or load_settings()
        t = self.settings.thresholds

        self.pre = AdaptivePreprocessor(
            blur_thresh=100.0, dark_thresh=70.0, haze_thresh=0.5, contrast_thresh=0.22
        )
        self.detector = Detector(
            self.settings.models.detector,
            device=self.settings.device,
            conf=t.detection_conf,
            iou=t.nms_iou,
            imgsz=self.settings.models.detector_imgsz,
            class_map=self.settings.models.detector_class_map or None,
            backend=self.settings.models.detector_backend,
        )
        plate_keep = {"license_plate"} if self.settings.models.detector_backend == "viovision" else set()
        plate_backend = (self.settings.models.plate_backend or "yolo").lower()
        if plate_backend == "roboflow":
            self.plate_detector = RoboflowPlateDetector(
                workspace=self.settings.models.roboflow_workspace,
                workflow_id=self.settings.models.roboflow_workflow_id,
                workflow_classes=self.settings.models.roboflow_plate_classes,
            )
        elif self.settings.models.plate:
            self.plate_detector = Detector(
                self.settings.models.plate,
                device=self.settings.device,
                conf=t.detection_conf,
                imgsz=self.settings.models.detector_imgsz,
                keep=plate_keep,
                class_map=self.settings.models.detector_class_map or None,
                backend=self.settings.models.detector_backend,
            )
        else:
            self.plate_detector = None
        self.plate_backend = plate_backend
        self.engine = ViolationEngine.from_settings(self.settings)
        self.ocr = PlateOCR(
            langs=self.settings.models.ocr_langs,
            gpu=self.settings.device == "cuda",
            backend=self.settings.models.ocr_backend,
            trocr_path=self.settings.models.trocr_model_path,
            roboflow_workspace=self.settings.models.roboflow_workspace,
            roboflow_ocr_model_id=self.settings.models.roboflow_ocr_model_id,
            roboflow_ocr_workflow_id=self.settings.models.roboflow_ocr_workflow_id,
            roboflow_ocr_workflow_classes=self.settings.models.roboflow_ocr_workflow_classes,
        )
        self.builder = EvidenceBuilder(self.settings.storage_dir)
        self.repo = Repository(self.settings.db_url)
        self.frame_index = 0
        self.tracker = SimpleTracker()

    def reset_live_session(self) -> None:
        """Clear tracking / temporal state when starting a new live feed."""
        self.frame_index = 0
        self.tracker = SimpleTracker()
        self.engine.reset()

    # -- introspection ------------------------------------------------------
    def model_status(self) -> dict:
        helmet = any(
            (isinstance(m, HelmetDetector) or isinstance(m, RiderCNNDetector))
            and m.available
            for m in self.engine.modules
        )
        seatbelt = any(
            isinstance(m, SeatbeltDetector) and m.available for m in self.engine.modules
        )
        return {
            "detector": self.detector.available,
            "ocr": self.ocr.available,
            "helmet": helmet,
            "seatbelt": seatbelt,
            "plate": self.plate_detector.available if self.plate_detector else False,
        }

    # -- plate OCR ----------------------------------------------------------
    def _attach_plates(self, frame: np.ndarray, violations: list[Violation]) -> None:
        if not self.ocr.available:
            return
        cache: dict = {}
        for v in violations:
            key = v.track_id if v.track_id is not None else tuple(round(c) for c in v.bbox)
            if key in cache:
                v.plate = cache[key]
                continue
            two_wheeler = (v.vehicle_class or "") in ("motorcycle", "bicycle")
            roi = plate_search_roi(v.bbox, two_wheeler=two_wheeler)
            region = crop(frame, roi)
            if self.plate_detector and self.plate_detector.available and region.size:
                dets = self.plate_detector.detect(region)
                if dets:
                    best = max(dets, key=lambda d: d.confidence)
                    region = crop(region, best.bbox)
            plate = self.ocr.read(region, bbox=roi) if region.size else None
            cache[key] = plate
            v.plate = plate

    # -- single image -------------------------------------------------------
    def process_image(
        self,
        image: np.ndarray,
        *,
        location: str = "Camera-01",
        persist: bool = True,
        use_tracking: bool = False,
        frame_index: Optional[int] = None,
    ) -> dict:
        t0 = time.perf_counter()
        fi = frame_index if frame_index is not None else self.frame_index

        enhanced, quality = self.pre.process(image)

        if use_tracking and self.detector.available:
            detections: list[Detection] = self.detector.track(enhanced)
        else:
            detections = self.detector.detect(enhanced)
            if use_tracking:  # detector gave no IDs; assign our own
                detections = self.tracker.update(detections)

        violations = self.engine.run(
            enhanced, detections, frame_index=fi, fps=self.settings.scene.fps
        )
        self._attach_plates(enhanced, violations)

        processing_ms = (time.perf_counter() - t0) * 1000.0
        evidence = self.builder.build(
            image,
            detections,
            violations,
            location=location,
            processing_ms=processing_ms,
            quality=quality.to_dict(),
        )
        if persist and evidence["records"]:
            self.repo.add_records(evidence["records"])

        self.frame_index = fi + 1
        return {
            "id": evidence["event_id"],
            "timestamp": evidence["timestamp"].isoformat(),
            "annotated_url": f"/api/events/{evidence['event_id']}/evidence",
            "annotated": evidence["annotated"],
            "evidence_path": evidence["evidence_path"],
            "processing_ms": round(processing_ms, 2),
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "quality": quality.to_dict(),
            "detections": [d.to_dict() for d in detections],
            "violations": [v.to_dict() for v in violations],
            "plates": [
                v.plate.to_dict() for v in violations if v.plate and v.plate.text
            ],
            "records": evidence["records"],
        }

    # -- video --------------------------------------------------------------
    def process_video(
        self,
        path: str,
        *,
        location: str = "Camera-01",
        sample_every: int = 1,
        max_frames: Optional[int] = None,
        persist: bool = True,
    ) -> dict:
        import cv2  # local: only needed for video decode

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or self.settings.scene.fps
        self.engine.scene.fps = fps
        self.engine.reset()

        n_frames = 0
        n_violations = 0
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % sample_every == 0:
                    res = self.process_image(
                        frame,
                        location=location,
                        persist=persist,
                        use_tracking=True,
                        frame_index=idx,
                    )
                    n_frames += 1
                    n_violations += len(res["violations"])
                idx += 1
                if max_frames and n_frames >= max_frames:
                    break
        finally:
            cap.release()
        return {
            "frames_processed": n_frames,
            "violations": n_violations,
            "fps": round(fps, 2),
        }
