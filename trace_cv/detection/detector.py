"""YOLO detector wrapper (Ultralytics).

Imported lazily: the package loads without torch/ultralytics installed.
Default weights are COCO, which already cover person, bicycle, car,
motorcycle, bus, truck and traffic light — enough to drive triple-riding,
red-light, stop-line, parking and wrong-side detection out of the box.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from trace_cv.core.logging import get_logger
from trace_cv.core.types import Detection

log = get_logger("detector")

# COCO names we keep (normalized: spaces -> underscores).
_KEEP = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic_light"}
# VioVision fine-tuned detector classes (before remap).
_VIOVISION_KEEP = {"person", "car", "two_wheeler", "license_plate", "windshield", "signal_light"}


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


class Detector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: str = "cpu",
        conf: float = 0.35,
        iou: float = 0.45,
        keep: Optional[set[str]] = None,
        *,
        imgsz: int = 640,
        class_map: Optional[dict[str, str]] = None,
        backend: str = "coco",
    ):
        from trace_cv.adapters.viovision_bridge import (
            DEFAULT_VIOVISION_CLASS_MAP,
            resolve_repo_path,
        )

        self.weights = str(resolve_repo_path(weights))
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.class_map = class_map or {}
        self.backend = backend
        if keep is not None:
            self.keep = keep
        elif backend == "viovision":
            self.keep = _VIOVISION_KEEP
        else:
            self.keep = _KEEP
        if backend == "viovision" and not self.class_map:
            self.class_map = dict(DEFAULT_VIOVISION_CLASS_MAP)
        self._model = None
        self._tried = False

    # -- lazy loader --------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None or self._tried:
            return
        self._tried = True
        weights = self.weights
        from pathlib import Path  # noqa: PLC0415

        if not Path(weights).exists():
            if self.backend == "viovision":
                log.warning(
                    "Custom detector weights missing (%s); falling back to yolov8n.pt (COCO).",
                    weights,
                )
                self.weights = "yolov8n.pt"
                self.backend = "coco"
                self.keep = _KEEP
                self.class_map = {}
                weights = self.weights
            else:
                log.warning("Detector weights not found: %s", weights)
        try:
            from ultralytics import YOLO  # noqa: PLC0415

            self._model = YOLO(weights)
            log.info("YOLO loaded: %s (device=%s)", self.weights, self.device)
        except Exception as exc:  # pragma: no cover - depends on env
            log.warning("Ultralytics unavailable (%s); detection disabled.", exc)
            self._model = None

    @property
    def available(self) -> bool:
        self._ensure_model()
        return self._model is not None

    # -- parsing ------------------------------------------------------------
    def _parse(self, result) -> list[Detection]:
        out: list[Detection] = []
        names = result.names
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return out
        for b in boxes:
            raw_name = _norm(names[int(b.cls)])
            if self.keep and raw_name not in self.keep:
                continue
            name = self.class_map.get(raw_name, raw_name) if self.class_map else raw_name
            xyxy = b.xyxy[0].tolist()
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            tid = int(b.id[0]) if getattr(b, "id", None) is not None else None
            out.append(
                Detection(cls=name, bbox=tuple(xyxy), confidence=conf, track_id=tid)
            )
        return out

    # -- inference ----------------------------------------------------------
    def detect(self, img: np.ndarray) -> list[Detection]:
        """Single-image detection (no tracking)."""
        self._ensure_model()
        if self._model is None:
            return []
        result = self._model(
            img,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        return self._parse(result)

    def track(self, img: np.ndarray, persist: bool = True) -> list[Detection]:
        """Detection + ByteTrack IDs across frames (video mode)."""
        self._ensure_model()
        if self._model is None:
            return []
        result = self._model.track(
            img,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            persist=persist,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]
        return self._parse(result)
