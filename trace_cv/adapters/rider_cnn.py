"""Multi-label rider state CNN (no_helmet + triple_riding) for CCTV crops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from trace_cv.adapters.viovision_bridge import resolve_repo_path
from trace_cv.core.logging import get_logger

log = get_logger("rider_cnn")

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_THRESH = {"no_helmet": 0.5, "triple_riding": 0.5}


class RiderCNNModel:
    """Torch CNN with sigmoid outputs for multi-label rider violations."""

    def __init__(
        self,
        weights: str,
        device: str = "cpu",
        thresholds: Optional[dict[str, float]] = None,
    ):
        self.weights = str(resolve_repo_path(weights))
        self.device = device
        self.thresholds = dict(_DEFAULT_THRESH)
        if thresholds:
            self.thresholds.update(thresholds)
        self._model = None
        self._meta: dict = {}
        self._tried = False

    def _ensure(self) -> None:
        if self._model is not None or self._tried:
            return
        self._tried = True
        path = Path(self.weights)
        if not path.exists():
            log.warning("Rider CNN weights not found: %s", path)
            return
        try:
            import torch
            from torchvision import models
            from torch import nn

            payload = torch.load(path, map_location=self.device, weights_only=False)
            self._meta = {
                "backbone": payload.get("backbone", "mobilenet_v3_small"),
                "imgsz": int(payload.get("imgsz", 224)),
            }
            if "thresholds" in payload:
                self.thresholds.update(payload["thresholds"])
            thresh_file = path.parent / "rider_multilabel_thresholds.json"
            if thresh_file.exists():
                self.thresholds.update(json.loads(thresh_file.read_text()))

            imgsz = self._meta["imgsz"]
            if self._meta["backbone"] == "efficientnet_b0":
                base = models.efficientnet_b0(weights=None)
                in_features = base.classifier[1].in_features
                base.classifier = nn.Identity()
            else:
                base = models.mobilenet_v3_small(weights=None)
                in_features = base.classifier[0].in_features
                base.classifier = nn.Identity()

            class RiderNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = base
                    self.head = nn.Linear(in_features, 2)

                def forward(self, x):
                    return self.head(self.backbone(x))

            self._model = RiderNet()
            self._model.load_state_dict(payload["model"])
            self._model.to(self.device)
            self._model.eval()
            self._imgsz = imgsz
            log.info("Rider CNN loaded: %s", path)
        except Exception as exc:  # pragma: no cover
            log.warning("Rider CNN unavailable (%s)", exc)
            self._model = None

    @property
    def available(self) -> bool:
        self._ensure()
        return self._model is not None

    def _preprocess(self, region: np.ndarray) -> "torch.Tensor":
        import torch
        from torchvision import transforms

        if region is None or region.size == 0:
            region = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self._imgsz, self._imgsz)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return tf(rgb).unsqueeze(0).to(self.device)

    def predict(self, region: np.ndarray) -> dict[str, tuple[bool, float]]:
        """Return per-label (active, confidence) for no_helmet and triple_riding."""
        self._ensure()
        out = {
            "no_helmet": (False, 0.0),
            "triple_riding": (False, 0.0),
        }
        if self._model is None or region is None or region.size == 0:
            return out
        try:
            import torch

            with torch.no_grad():
                logits = self._model(self._preprocess(region))
                probs = torch.sigmoid(logits)[0].cpu().numpy()
            labels = ("no_helmet", "triple_riding")
            for i, name in enumerate(labels):
                conf = float(probs[i])
                active = conf >= self.thresholds.get(name, 0.5)
                out[name] = (active, conf)
        except Exception as exc:  # pragma: no cover
            log.warning("Rider CNN infer failed: %s", exc)
        return out
