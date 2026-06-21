"""Roboflow workflow-based helmet detection for TRACE.

Uses ``run_workflow`` with ``classes: helmet, no_helmet`` (or configured).
Falls back to ``infer(model_id=...)`` when no workflow_id is set.
API key from ROBOFLOW_API_KEY only — never hardcode credentials.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from trace_cv.adapters.roboflow_common import (
    collect_predictions,
    get_roboflow_client,
    norm_class,
)
from trace_cv.core.logging import get_logger

log = get_logger("roboflow_helmet")

_NO_HELMET_KEYS = ("no_helmet", "nohelmet", "without", "no-helmet", "head", "bare", "no helmet")
_HELMET_KEYS = ("helmet", "with_helmet", "with helmet")


class RoboflowHelmetModel:
    """Helmet / no-helmet via Roboflow workflow or hosted model infer."""

    def __init__(
        self,
        *,
        workspace: str = "prakhar-parkar",
        workflow_id: str | None = None,
        workflow_classes: str = "helmet, no_helmet",
        model_id: str | None = "helmet-gj8do/2",
        conf: float = 0.25,
    ):
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.workflow_classes = workflow_classes
        self.model_id = model_id
        self.conf = conf
        self._client = get_roboflow_client()

    @property
    def available(self) -> bool:
        return self._client.available

    def _label_to_bool(self, label: str) -> Optional[bool]:
        name = norm_class(label)
        if any(k in name for k in _NO_HELMET_KEYS):
            return False
        if any(k in name for k in _HELMET_KEYS):
            if "no" in name and "helmet" in name:
                return False
            return True
        return None

    def _predict_from_preds(self, preds: list[dict]) -> tuple[Optional[bool], float]:
        if not preds:
            return None, 0.0
        best_no, best_yes = 0.0, 0.0
        best_label, best_unknown = None, 0.0
        for p in preds:
            label = str(p.get("class") or p.get("class_name") or p.get("label") or "")
            cf = float(p.get("confidence") or p.get("score") or 0.0)
            if cf < self.conf:
                continue
            verdict = self._label_to_bool(label)
            if verdict is False:
                best_no = max(best_no, cf)
            elif verdict is True:
                best_yes = max(best_yes, cf)
            elif cf > best_unknown:
                best_unknown, best_label = cf, label
        if best_no > 0 and best_no >= best_yes:
            return False, best_no
        if best_yes > 0:
            return True, best_yes
        if best_label:
            return self._label_to_bool(best_label), best_unknown
        return None, 0.0

    def predict(self, region: np.ndarray) -> tuple[Optional[bool], float]:
        """Return (has_helmet, confidence). has_helmet is None when unsure."""
        if not self.available or region is None or region.size == 0:
            return None, 0.0
        try:
            if self.workflow_id:
                result = self._client.run_workflow_image(
                    region,
                    workspace=self.workspace,
                    workflow_id=self.workflow_id,
                    parameters={"classes": self.workflow_classes},
                )
            elif self.model_id:
                result = self._client.infer_image(region, self.model_id)
            else:
                return None, 0.0
            return self._predict_from_preds(collect_predictions(result))
        except Exception as exc:  # pragma: no cover
            log.warning("Roboflow helmet failed: %s", exc)
            return None, 0.0
