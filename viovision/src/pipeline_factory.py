"""
Builds the live set of adapters for one pipeline run, real or mocked,
purely from configs/pipeline.yaml. This is the file that makes
"swapping USE_MOCKS=false is a drop-in" literally true: nothing else in
the codebase imports a concrete adapter class directly except this
factory and the per-model training scripts.

Usage:
    from src.pipeline_factory import build_pipeline
    pipeline = build_pipeline("configs/pipeline.yaml")
    detections = pipeline.detector.predict(frame)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.adapters.mocks import (
    MockCropClassifierAdapter,
    MockDetectorAdapter,
    MockOCRAdapter,
    MockVLMAdjudicatorAdapter,
)
from src.adapters.schema import (
    CropClassifierAdapter,
    DetectorAdapter,
    OCRAdapter,
    VLMAdjudicatorAdapter,
)


@dataclass
class Pipeline:
    detector: DetectorAdapter
    helmet: CropClassifierAdapter
    seatbelt: CropClassifierAdapter
    signal: CropClassifierAdapter
    ocr: OCRAdapter
    vlm: VLMAdjudicatorAdapter
    config: dict[str, Any]


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_pipeline(config_path: str | Path = "configs/pipeline.yaml",
                     anthropic_api_key: str | None = None) -> Pipeline:
    cfg = _load_config(config_path)
    mocks = cfg["use_mocks"]
    weights = cfg["weights"]
    thresholds = cfg["review_thresholds"]

    # -- detector ---------------------------------------------------------
    if mocks["detector"]:
        detector: DetectorAdapter = MockDetectorAdapter()
    else:
        from src.models.yolo_detector import YoloDetectorAdapter
        detector = YoloDetectorAdapter(
            weights_path=weights["detector"],
            conf_threshold=cfg["detector"]["conf_threshold"],
            imgsz=cfg["detector"]["imgsz"],
            use_sahi=cfg["detector"]["use_sahi"],
        )

    # -- helmet -------------------------------------------------------------
    if mocks["helmet"]:
        helmet: CropClassifierAdapter = MockCropClassifierAdapter(
            class_names=("no_helmet", "helmet"), review_threshold=thresholds["helmet"]
        )
    else:
        from src.models.helmet_classifier import HelmetClassifier
        helmet = HelmetClassifier()
        helmet.load(weights["helmet"])

    # -- seatbelt -------------------------------------------------------------
    if mocks["seatbelt"]:
        seatbelt: CropClassifierAdapter = MockCropClassifierAdapter(
            class_names=("no_seatbelt", "seatbelt"), review_threshold=thresholds["seatbelt"]
        )
    else:
        from src.models.seatbelt_classifier import SeatbeltClassifier
        seatbelt = SeatbeltClassifier()
        seatbelt.load(weights["seatbelt"])

    # -- signal -------------------------------------------------------------
    if mocks["signal"]:
        signal: CropClassifierAdapter = MockCropClassifierAdapter(
            class_names=("red", "yellow", "green", "unknown"),
            review_threshold=thresholds["signal"],
        )
    else:
        from src.models.signal_state_classifier import SignalStateClassifier
        signal = SignalStateClassifier(mode=cfg["signal"]["mode"])
        if cfg["signal"]["mode"] == "sklearn":
            signal.load(weights["signal"])

    # -- ocr -------------------------------------------------------------
    if mocks["ocr"]:
        ocr: OCRAdapter = MockOCRAdapter()
    else:
        from src.adapters.ocr_adapter import EasyOCRAdapter
        ocr = EasyOCRAdapter()

    # -- vlm -------------------------------------------------------------
    if mocks["vlm"]:
        vlm: VLMAdjudicatorAdapter = MockVLMAdjudicatorAdapter()
    else:
        if not anthropic_api_key:
            raise ValueError("anthropic_api_key required when use_mocks.vlm is false.")
        from src.adapters.vlm_adapter import ClaudeVLMAdjudicatorAdapter
        vlm = ClaudeVLMAdjudicatorAdapter(api_key=anthropic_api_key)

    return Pipeline(
        detector=detector, helmet=helmet, seatbelt=seatbelt, signal=signal,
        ocr=ocr, vlm=vlm, config=cfg,
    )
