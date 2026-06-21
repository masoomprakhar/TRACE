"""Roboflow character OCR via hosted model or segmentation workflow."""

from __future__ import annotations

import re

import numpy as np

from trace_cv.adapters.roboflow_common import (
    RoboflowClient,
    bbox_from_prediction,
    collect_predictions,
    get_roboflow_client,
    parse_class_filter,
)
from trace_cv.core.logging import get_logger

log = get_logger("roboflow_ocr")

_CHAR_RE = re.compile(r"^[A-Z0-9\-]$")


def _norm_char(label: str) -> str:
    ch = label.strip().upper().replace(" ", "")
    if ch in ("-", "DASH", "HYPHEN"):
        return "-"
    return ch


def predictions_to_chars(
    preds: list[dict],
    img_w: int,
    img_h: int,
    *,
    allowed: set[str] | None = None,
) -> list[dict]:
    chars: list[dict] = []
    for pred in preds:
        raw = str(pred.get("class") or pred.get("class_name") or pred.get("label") or "")
        ch = _norm_char(raw)
        if not _CHAR_RE.match(ch):
            continue
        if allowed:
            from trace_cv.adapters.roboflow_common import class_matches

            if not class_matches(raw, allowed) and not class_matches(ch, allowed):
                continue
        bbox = bbox_from_prediction(pred, img_w, img_h)
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        conf = float(pred.get("confidence") or pred.get("score") or 0.5)
        chars.append(
            {
                "ch": ch,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "cx": (x1 + x2) / 2,
                "conf": conf,
            }
        )
    chars.sort(key=lambda c: c["cx"])
    return chars


def chars_to_text(chars: list[dict]) -> str:
    return "".join(c["ch"] for c in chars)


class RoboflowCharOCR:
    """Read plate text from character boxes (infer model or workflow)."""

    def __init__(
        self,
        *,
        workspace: str = "prakhar-parkar",
        model_id: str | None = "ocr-character-cgtzm/4",
        workflow_id: str | None = None,
        workflow_classes: str = "-, 0, 1",
    ):
        self.workspace = workspace
        self.model_id = model_id
        self.workflow_id = workflow_id
        self.workflow_classes = workflow_classes
        self._allowed = parse_class_filter(workflow_classes)
        self._client = get_roboflow_client()

    @property
    def available(self) -> bool:
        return self._client.available and bool(self.model_id or self.workflow_id)

    def _infer_chars(self, img: np.ndarray) -> list[dict]:
        h, w = img.shape[:2]
        if self.workflow_id:
            result = self._client.run_workflow_image(
                img,
                workspace=self.workspace,
                workflow_id=self.workflow_id,
                parameters={"classes": self.workflow_classes},
            )
        elif self.model_id:
            result = self._client.infer_image(img, self.model_id)
        else:
            return []
        return predictions_to_chars(
            collect_predictions(result),
            w,
            h,
            allowed=self._allowed if self.workflow_id else None,
        )

    def read(self, plate_crop: np.ndarray) -> tuple[str, float]:
        if not self.available or plate_crop is None or plate_crop.size == 0:
            return "", 0.0
        try:
            chars = self._infer_chars(plate_crop)
            if not chars:
                return "", 0.0
            text = chars_to_text(chars)
            conf = float(np.mean([c["conf"] for c in chars]))
            return text, conf
        except Exception as exc:  # pragma: no cover
            log.warning("Roboflow OCR failed: %s", exc)
            return "", 0.0
