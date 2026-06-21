"""Load config/roboflow_models.yaml for training & dataset scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "config" / "roboflow_models.yaml"


def load(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_PATH
    if not cfg_path.exists():
        return {}
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    ws = os.environ.get("ROBOFLOW_WORKSPACE") or raw.get("workspace", "prakhar-parkar")
    raw["workspace"] = ws
    if os.environ.get("ROBOFLOW_WORKFLOW_ID"):
        raw.setdefault("workflows", {}).setdefault("segmentation", {})["id"] = os.environ[
            "ROBOFLOW_WORKFLOW_ID"
        ]
    if os.environ.get("ROBOFLOW_OCR_MODEL_ID"):
        raw.setdefault("models", {})["ocr_character"] = os.environ["ROBOFLOW_OCR_MODEL_ID"]
    return raw


def api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""
