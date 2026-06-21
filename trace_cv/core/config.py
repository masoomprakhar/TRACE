"""Configuration: scene geometry, thresholds, model paths.

Plain dataclasses + a YAML loader (no pydantic dependency here, so config
loads without the web stack). Per-camera scene geometry is what turns
geometric detections into violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass
class Thresholds:
    detection_conf: float = 0.35
    nms_iou: float = 0.45
    helmet_conf: float = 0.60
    seatbelt_conf: float = 0.65
    triple_riding_min: int = 3
    confirm_frames: int = 3
    parking_seconds: float = 30.0
    stationary_iou: float = 0.90
    rider_overlap: float = 0.20


@dataclass
class StopLine:
    enabled: bool = False
    y: Optional[int] = None


@dataclass
class LaneConfig:
    enabled: bool = False
    divider_x: Optional[int] = None
    correct_direction: str = "down"  # legal motion on the RIGHT of the divider


@dataclass
class SignalROI:
    enabled: bool = True
    bbox: Optional[list] = None  # [x1, y1, x2, y2]; None -> auto-detect


@dataclass
class Zone:
    name: str = "zone"
    polygon: list = field(default_factory=list)  # [[x, y], ...]


@dataclass
class SceneConfig:
    fps: float = 15.0
    stop_line: StopLine = field(default_factory=StopLine)
    lane: LaneConfig = field(default_factory=LaneConfig)
    signal: SignalROI = field(default_factory=SignalROI)
    no_parking_zones: list = field(default_factory=list)  # list[Zone]


@dataclass
class ModelPaths:
    detector: str = "yolov8n.pt"
    helmet: Optional[str] = None
    seatbelt: Optional[str] = None
    plate: Optional[str] = None
    ocr_langs: list = field(default_factory=lambda: ["en"])
    # Plate localization: "yolo" (local weights), "roboflow" (workflow API), or null ROI fallback.
    plate_backend: str = "yolo"
    roboflow_workspace: str = "prakhar-parkar"
    roboflow_workflow_id: str = "general-segmentation-api"
    roboflow_plate_classes: str = "license_plate"
    # Helmet: "local" (pkl/pt), "roboflow" workflow, or legacy infer model_id.
    helmet_backend: str = "local"
    roboflow_helmet_workflow_id: str = "general-segmentation-api"
    roboflow_helmet_classes: str = "helmet, no_helmet"
    roboflow_helmet_model_id: str | None = "helmet-gj8do/2"
    # Rider state: cnn (multi-label), svm/helmet path, or roboflow helmet workflow.
    rider_backend: str = "cnn"
    rider_cnn_weights: str = "models/weights/rider_multilabel_cnn.pt"
    # OCR: trocr (fine-tuned) or easyocr.
    ocr_backend: str = "trocr"
    trocr_model_path: str = "models/weights/trocr_plate"
    # VioVision integration: custom YOLO class remap + inference size.
    detector_imgsz: int = 640
    detector_backend: str = "coco"  # "coco" | "viovision"
    detector_class_map: dict = field(default_factory=dict)


@dataclass
class Settings:
    device: str = "cpu"
    storage_dir: str = "data/output"
    db_url: str = "sqlite:///data/trace.db"
    thresholds: Thresholds = field(default_factory=Thresholds)
    scene: SceneConfig = field(default_factory=SceneConfig)
    models: ModelPaths = field(default_factory=ModelPaths)


def _build(dc_type, data: Any):
    """Recursively build a dataclass from a (possibly partial) dict, keeping
    defaults for anything missing. Lists of dataclasses are handled for
    no_parking_zones explicitly by the caller."""
    if data is None:
        return dc_type()
    if not isinstance(data, dict):
        return data
    kwargs = {}
    for f in fields(dc_type):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _build(f.type, value)
        else:
            kwargs[f.name] = value
    return dc_type(**kwargs)


def load_settings(path: Optional[str | Path] = None) -> Settings:
    """Load settings from YAML, falling back to packaged defaults. Unknown
    keys are ignored; missing keys keep their dataclass defaults."""
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    raw: dict = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}

    settings = Settings(
        device=raw.get("device", "cpu"),
        storage_dir=raw.get("storage_dir", "data/output"),
        db_url=raw.get("db_url", "sqlite:///data/trace.db"),
        thresholds=_build(Thresholds, raw.get("thresholds")),
        models=_build(ModelPaths, raw.get("models")),
        scene=_build_scene(raw.get("scene")),
    )
    return settings


def _build_scene(data: Optional[dict]) -> SceneConfig:
    if not data:
        return SceneConfig()
    scene = SceneConfig(
        fps=data.get("fps", 15.0),
        stop_line=_build(StopLine, data.get("stop_line")),
        lane=_build(LaneConfig, data.get("lane")),
        signal=_build(SignalROI, data.get("signal")),
    )
    zones = data.get("no_parking_zones") or []
    scene.no_parking_zones = [
        Zone(name=z.get("name", f"zone{i}"), polygon=z.get("polygon", []))
        for i, z in enumerate(zones)
        if isinstance(z, dict)
    ]
    return scene
