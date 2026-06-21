"""Roboflow workflow-based license plate detector for TRACE."""

from __future__ import annotations

import numpy as np

from trace_cv.adapters.roboflow_common import (
    RoboflowClient,
    bbox_from_prediction,
    class_matches,
    collect_predictions,
    parse_class_filter,
)
from trace_cv.core.logging import get_logger
from trace_cv.core.types import Detection

log = get_logger("roboflow_plate")


class RoboflowPlateDetector:
    """Detect license plates via a Roboflow workflow (e.g. general-segmentation-api-2)."""

    def __init__(
        self,
        *,
        workspace: str = "prakhar-parkar",
        workflow_id: str = "general-segmentation-api",
        workflow_classes: str = "license_plate",
    ):
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.workflow_classes = workflow_classes
        self._allowed = parse_class_filter(workflow_classes)
        self._client = RoboflowClient()

    @property
    def available(self) -> bool:
        return self._client.available

    def _parse_workflow_output(self, result, img_w: int, img_h: int) -> list[Detection]:
        out: list[Detection] = []
        for pred in collect_predictions(result):
            label = str(pred.get("class") or pred.get("class_name") or pred.get("label") or "")
            if not class_matches(label, self._allowed):
                continue
            bbox = bbox_from_prediction(pred, img_w, img_h)
            if not bbox:
                continue
            conf = float(pred.get("confidence") or pred.get("score") or 0.5)
            out.append(Detection(cls="license_plate", bbox=bbox, confidence=conf))
        return out

    def detect(self, img: np.ndarray) -> list[Detection]:
        if not self.available or img is None or img.size == 0:
            return []
        h, w = img.shape[:2]
        try:
            result = self._client.run_workflow_image(
                img,
                workspace=self.workspace,
                workflow_id=self.workflow_id,
                parameters={"classes": self.workflow_classes},
            )
            return self._parse_workflow_output(result, w, h)
        except Exception as exc:  # pragma: no cover
            log.warning("Roboflow plate workflow failed: %s", exc)
            return []
