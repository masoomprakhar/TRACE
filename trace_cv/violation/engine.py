"""Violation engine: orchestrates every violation module with shared
tracking context, and maintains per-track history for the temporal checks
(wrong-side motion, parking duration, stop-line crossing)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from trace_cv.core.config import ModelPaths, SceneConfig, Settings, Thresholds
from trace_cv.core.logging import get_logger
from trace_cv.core.types import Detection, Violation
from trace_cv.violation.base import TrackState, ViolationContext, ViolationModule
from trace_cv.violation.helmet import HelmetDetector, HelmetModel
from trace_cv.violation.parking import ParkingDetector
from trace_cv.violation.red_light import RedLightDetector
from trace_cv.violation.seatbelt import SeatbeltDetector, SeatbeltModel
from trace_cv.violation.stop_line import StopLineDetector
from trace_cv.violation.rider_cnn import RiderCNNDetector
from trace_cv.violation.triple_riding import TripleRidingDetector
from trace_cv.violation.wrong_side import WrongSideDetector

log = get_logger("engine")


class ViolationEngine:
    def __init__(
        self,
        scene: SceneConfig,
        thresholds: Thresholds,
        helmet_model: Optional[HelmetModel] = None,
        seatbelt_model: Optional[SeatbeltModel] = None,
        rider_cnn_model=None,
        use_rider_cnn: bool = False,
        history_len: int = 90,
    ):
        self.scene = scene
        self.thresholds = thresholds
        self.history_len = history_len
        self.track_history: dict[int, list[TrackState]] = {}
        self._pending: dict[tuple, int] = {}
        self.use_rider_cnn = use_rider_cnn

        triple_mod: ViolationModule = TripleRidingDetector()
        helmet_mod: ViolationModule = HelmetDetector(helmet_model)
        modules: list[ViolationModule] = [
            RedLightDetector(),
            StopLineDetector(),
            WrongSideDetector(),
            ParkingDetector(),
            SeatbeltDetector(seatbelt_model),
        ]
        if use_rider_cnn and rider_cnn_model is not None:
            modules.insert(0, RiderCNNDetector(rider_cnn_model))
        else:
            modules.insert(0, triple_mod)
            modules.append(helmet_mod)

        self.modules = modules

    # -- construction from Settings ----------------------------------------
    @classmethod
    def from_settings(cls, settings: Settings) -> "ViolationEngine":
        models: ModelPaths = settings.models
        rider_backend = (models.rider_backend or "svm").lower()
        use_rider_cnn = rider_backend == "cnn"
        rider_cnn = None
        helmet = None

        if use_rider_cnn:
            from trace_cv.adapters.rider_cnn import RiderCNNModel

            rider_cnn = RiderCNNModel(
                models.rider_cnn_weights,
                device=settings.device,
            )
        else:
            helmet_backend = (models.helmet_backend or "local").lower()
            if helmet_backend == "roboflow":
                from trace_cv.adapters.roboflow_helmet import RoboflowHelmetModel

                helmet = RoboflowHelmetModel(
                    workspace=models.roboflow_workspace,
                    workflow_id=models.roboflow_helmet_workflow_id or None,
                    workflow_classes=models.roboflow_helmet_classes,
                    model_id=models.roboflow_helmet_model_id,
                )
            elif models.helmet:
                helmet = HelmetModel(models.helmet, settings.device)

        seatbelt = (
            SeatbeltModel(models.seatbelt, settings.device) if models.seatbelt else None
        )
        return cls(
            settings.scene,
            settings.thresholds,
            helmet,
            seatbelt,
            rider_cnn_model=rider_cnn,
            use_rider_cnn=use_rider_cnn and rider_cnn is not None and rider_cnn.available,
        )

    # -- history ------------------------------------------------------------
    def _update_history(self, detections: list[Detection], frame_index: int) -> None:
        for d in detections:
            if d.track_id is None:
                continue
            states = self.track_history.setdefault(d.track_id, [])
            states.append(TrackState(bbox=d.bbox, frame_index=frame_index))
            if len(states) > self.history_len:
                del states[0]

    def reset(self) -> None:
        self.track_history.clear()
        self._pending.clear()

    def _confirm(self, violations: list[Violation], frame_index: int) -> list[Violation]:
        """Require violations to persist for confirm_frames (video + tracking only)."""
        need = self.thresholds.confirm_frames
        if need <= 1:
            return violations
        if not any(v.track_id is not None for v in violations):
            return violations
        confirmed: list[Violation] = []
        seen: set[tuple] = set()
        for v in violations:
            key = (v.type.value, v.track_id, tuple(round(c) for c in v.bbox))
            seen.add(key)
            self._pending[key] = self._pending.get(key, 0) + 1
            if self._pending[key] >= need:
                confirmed.append(v)
        # Drop stale keys not seen this frame.
        stale = [k for k in self._pending if k not in seen]
        for k in stale:
            del self._pending[k]
        return confirmed

    # -- run ----------------------------------------------------------------
    def run(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_index: int = 0,
        fps: Optional[float] = None,
    ) -> list[Violation]:
        self._update_history(detections, frame_index)
        ctx = ViolationContext(
            frame=frame,
            detections=detections,
            scene=self.scene,
            thresholds=self.thresholds,
            track_history=self.track_history,
            frame_index=frame_index,
            fps=fps or self.scene.fps,
        )
        violations: list[Violation] = []
        for module in self.modules:
            try:
                violations.extend(module.check(ctx))
            except Exception as exc:  # one module failing must not kill the frame
                log.warning("module %s failed: %s", module.type.value, exc)
        return self._confirm(violations, frame_index)

    @property
    def active_modules(self) -> list[str]:
        return [m.type.value for m in self.modules if m.available]
