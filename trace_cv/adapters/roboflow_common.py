"""Shared Roboflow InferenceHTTPClient helpers for TRACE adapters."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from trace_cv.core.logging import get_logger

log = get_logger("roboflow")


def norm_class(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


class RoboflowClient:
    """Lazy wrapper around inference_sdk.InferenceHTTPClient."""

    def __init__(
        self,
        *,
        api_url: str = "https://serverless.roboflow.com",
        api_key: Optional[str] = None,
    ):
        self.api_url = api_url
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        self._client = None
        self._tried = False

    def _ensure(self) -> None:
        if self._client is not None or self._tried:
            return
        self._tried = True
        if not self.api_key:
            log.warning("ROBOFLOW_API_KEY not set; Roboflow inference disabled.")
            return
        try:
            from inference_sdk import InferenceHTTPClient  # noqa: PLC0415

            self._client = InferenceHTTPClient(
                api_url=self.api_url,
                api_key=self.api_key,
            )
            log.info("Roboflow inference client ready (%s)", self.api_url)
        except Exception as exc:  # pragma: no cover
            log.warning("inference-sdk unavailable (%s).", exc)
            self._client = None

    @property
    def available(self) -> bool:
        self._ensure()
        return self._client is not None

    def infer_image(self, img: np.ndarray, model_id: str) -> Any:
        self._ensure()
        if self._client is None or img is None or img.size == 0:
            return {}
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            cv2.imwrite(tmp.name, img)
            return self._client.infer(tmp.name, model_id=model_id)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def run_workflow(
        self,
        image_path: str,
        *,
        workspace: str,
        workflow_id: str,
        parameters: Optional[dict] = None,
    ) -> Any:
        self._ensure()
        if self._client is None:
            return {}
        return self._client.run_workflow(
            workspace_name=workspace,
            workflow_id=workflow_id,
            images={"image": image_path},
            parameters=parameters or {},
            use_cache=True,
        )

    def run_workflow_image(
        self,
        img: np.ndarray,
        *,
        workspace: str,
        workflow_id: str,
        parameters: Optional[dict] = None,
    ) -> Any:
        self._ensure()
        if self._client is None or img is None or img.size == 0:
            return {}
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            cv2.imwrite(tmp.name, img)
            return self.run_workflow(
                tmp.name,
                workspace=workspace,
                workflow_id=workflow_id,
                parameters=parameters,
            )
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def parse_class_filter(classes: str | list | None) -> set[str]:
    """Normalize Roboflow workflow ``classes`` parameter to a match set."""
    if not classes:
        return set()
    if isinstance(classes, str):
        parts = [p.strip() for p in classes.replace(";", ",").split(",") if p.strip()]
    else:
        parts = [str(c).strip() for c in classes if str(c).strip()]
    return {norm_class(p) for p in parts}


def class_matches(label: str, allowed: set[str]) -> bool:
    if not allowed:
        return True
    name = norm_class(label)
    return name in allowed or any(a in name or name in a for a in allowed)


def bbox_from_prediction(pred: dict, img_w: int, img_h: int) -> Optional[tuple[float, float, float, float]]:
    """Convert a Roboflow prediction dict to pixel xyxy bbox."""
    if "x" in pred and "y" in pred and "width" in pred and "height" in pred:
        x, y = float(pred["x"]), float(pred["y"])
        w, h = float(pred["width"]), float(pred["height"])
        if x <= 1 and y <= 1 and w <= 1 and h <= 1:
            return x * img_w, y * img_h, (x + w) * img_w, (y + h) * img_h
        if w < img_w and h < img_h:
            return x - w / 2, y - h / 2, x + w / 2, y + h / 2
        return x, y, x + w, y + h
    if "bbox" in pred:
        b = pred["bbox"]
        if isinstance(b, dict):
            return (
                float(b.get("x1", b.get("left", 0))),
                float(b.get("y1", b.get("top", 0))),
                float(b.get("x2", b.get("right", 0))),
                float(b.get("y2", b.get("bottom", 0))),
            )
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            return tuple(float(v) for v in b[:4])  # type: ignore[return-value]
    return None


def collect_predictions(result: Any) -> list[dict]:
    """Flatten Roboflow infer/workflow JSON into a list of prediction dicts."""
    found: list[dict] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "class" in node or "class_name" in node or "label" in node:
                if any(k in node for k in ("confidence", "score", "x", "width", "bbox")):
                    found.append(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(result)
    return found
