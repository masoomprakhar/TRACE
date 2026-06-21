"""Labeled evaluation dataset builder and loader.

Builds a holdout set of junction scenes with ground-truth violation labels
and optional detection boxes. Used by scripts/run_real_eval.py.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from trace_cv.core.types import ViolationType
from trace_cv.demo import _VEHICLE_COLORS, make_synthetic_scene

EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"
IMAGES_DIR = EVAL_DIR / "images"
MANIFEST_PATH = EVAL_DIR / "manifest.json"

# Scene templates: vehicle type, bbox ratios (x1,y1,x2,y2 as fractions of W,H),
# persons for triple-riding, expected violation types.
_TEMPLATES = [
    {
        "id": "triple_moto",
        "vehicle": "motorcycle",
        "bbox_frac": (0.55, 0.45, 0.72, 0.78),
        "persons_frac": [(0.58, 0.42, 0.62, 0.55), (0.63, 0.40, 0.67, 0.53), (0.68, 0.42, 0.72, 0.55)],
        "violations": ["triple_riding"],
    },
    {
        "id": "red_light_car",
        "vehicle": "car",
        "bbox_frac": (0.52, 0.55, 0.68, 0.88),
        "violations": ["red_light", "stop_line"],
        "signal_red": True,
    },
    {
        "id": "parked_car",
        "vehicle": "car",
        "bbox_frac": (0.08, 0.52, 0.18, 0.82),
        "violations": ["illegal_parking"],
    },
    {
        "id": "wrong_side_bike",
        "vehicle": "motorcycle",
        "bbox_frac": (0.35, 0.50, 0.48, 0.75),
        "violations": ["wrong_side"],
        "motion": "up",
    },
    {
        "id": "clean_car",
        "vehicle": "car",
        "bbox_frac": (0.58, 0.35, 0.72, 0.58),
        "violations": [],
    },
]


def _frac_bbox(w: int, h: int, frac: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = frac
    return [x1 * w, y1 * h, x2 * w, y2 * h]


def _draw_person(img: np.ndarray, bbox) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (80, 180, 255), -1)


def _draw_vehicle(img: np.ndarray, bbox, color) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), 2)


def build_eval_set(n_per_template: int = 4, seed: int = 42) -> Path:
    """Generate labeled eval images + manifest.json."""
    rng = random.Random(seed)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    for template in _TEMPLATES:
        for i in range(n_per_template):
            frame = make_synthetic_scene(rng=rng)
            h, w = frame.shape[:2]
            bbox = _frac_bbox(w, h, template["bbox_frac"])
            vt = template["vehicle"]
            _draw_vehicle(frame, bbox, _VEHICLE_COLORS.get(vt, (100, 100, 100)))

            detections = [
                {"cls": vt, "bbox": [round(v, 1) for v in bbox], "confidence": 0.9}
            ]
            for pfrac in template.get("persons_frac", []):
                pb = _frac_bbox(w, h, pfrac)
                _draw_person(frame, pb)
                detections.append(
                    {"cls": "person", "bbox": [round(v, 1) for v in pb], "confidence": 0.85}
                )

            # Draw a red signal lamp in the signal ROI region for red-light cases.
            if template.get("signal_red"):
                sx1, sy1, sx2, sy2 = _frac_bbox(w, h, (700 / 960, 24 / 540, 760 / 960, 70 / 540))
                cv2.circle(
                    frame,
                    (int((sx1 + sx2) / 2), int((sy1 + sy2) / 2)),
                    12,
                    (0, 0, 220),
                    -1,
                )
                detections.append(
                    {
                        "cls": "traffic_light",
                        "bbox": [round(sx1, 1), round(sy1, 1), round(sx2, 1), round(sy2, 1)],
                        "confidence": 0.88,
                    }
                )

            image_id = f"{template['id']}_{i:02d}"
            image_path = IMAGES_DIR / f"{image_id}.jpg"
            cv2.imwrite(str(image_path), frame)

            entries.append(
                {
                    "id": image_id,
                    "image": str(image_path.relative_to(EVAL_DIR.parent.parent)),
                    "width": w,
                    "height": h,
                    "vehicle": vt,
                    "violations": list(template["violations"]),
                    "detections_gt": detections,
                    "detail": {
                        "template": template["id"],
                        "signal_red": template.get("signal_red", False),
                    },
                }
            )

    manifest = {
        "version": 1,
        "n_samples": len(entries),
        "violation_labels": [v.value for v in ViolationType],
        "detection_labels": ["car", "motorcycle", "person", "traffic_light"],
        "samples": entries,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return MANIFEST_PATH


def load_manifest(path: Optional[Path] = None) -> dict:
    p = path or MANIFEST_PATH
    if not p.exists():
        build_eval_set()
    return json.loads(p.read_text())


def manifest_samples(manifest: dict) -> list[dict]:
    return manifest.get("samples", [])
