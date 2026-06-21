"""Demo utilities: synthetic scene generation, database seeding, and an
evaluation showcase.

These let you populate the dashboard and exercise the full evidence/storage
path WITHOUT any ML models or real footage — invaluable for a demo and for
CI. Nothing here fabricates results into the real detection path; it only
seeds illustrative records explicitly.
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Optional

import cv2
import numpy as np

from trace_cv.core.config import Settings, load_settings
from trace_cv.core.types import Detection, Plate, Violation, ViolationType, utcnow
from trace_cv.evidence.builder import EvidenceBuilder
from trace_cv.evaluation.metrics import (
    detection_map,
    multilabel_report,
    ocr_cer,
    ocr_exact_match,
)
from trace_cv.ocr.corrector import format_plate
from trace_cv.storage.db import Repository

_STATES = ["MH", "DL", "KA", "TN", "UP", "GJ", "RJ", "WB", "MP", "HR"]
_SERIES = ["AB", "Cd", "BZ", "CA", "XY", "GK", "PL", "QR"]

_VEHICLE_COLORS = {
    "car": (90, 140, 200),
    "motorcycle": (70, 70, 220),
    "truck": (110, 170, 110),
    "bus": (60, 190, 220),
}

# Plausible violation menus per vehicle type.
_MENU = {
    "motorcycle": [
        ViolationType.NO_HELMET,
        ViolationType.TRIPLE_RIDING,
        ViolationType.RED_LIGHT,
        ViolationType.WRONG_SIDE,
    ],
    "car": [
        ViolationType.NO_SEATBELT,
        ViolationType.RED_LIGHT,
        ViolationType.STOP_LINE,
        ViolationType.ILLEGAL_PARKING,
    ],
    "truck": [ViolationType.STOP_LINE, ViolationType.ILLEGAL_PARKING, ViolationType.WRONG_SIDE],
    "bus": [ViolationType.STOP_LINE, ViolationType.RED_LIGHT],
}


def random_plate(rng: random.Random) -> str:
    s = rng.choice(_STATES)
    rto = rng.randint(1, 35)
    series = rng.choice(_SERIES).upper()
    num = rng.randint(1, 9999)
    return format_plate(f"{s}{rto:02d}{series}{num:04d}")


def make_synthetic_scene(
    w: int = 960, h: int = 540, rng: Optional[random.Random] = None
) -> np.ndarray:
    rng = rng or random.Random()
    img = np.full((h, w, 3), (62, 62, 68), np.uint8)  # asphalt
    cv2.rectangle(img, (0, 0), (w, int(h * 0.18)), (150, 130, 110), -1)  # sky/buildings
    # Lane edges + dashed centre line.
    cv2.line(img, (int(w * 0.12), h), (int(w * 0.42), int(h * 0.2)), (220, 220, 220), 2)
    cv2.line(img, (int(w * 0.88), h), (int(w * 0.58), int(h * 0.2)), (220, 220, 220), 2)
    cy0 = int(h * 0.2)
    for y in range(cy0, h, 40):
        cv2.line(img, (w // 2, y), (w // 2, min(y + 20, h)), (200, 200, 80), 2)
    return img


def _draw_vehicle(img: np.ndarray, bbox, color) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), 2)
    # windshield strip
    wy = y1 + int((y2 - y1) * 0.18)
    cv2.rectangle(img, (x1 + 6, wy), (x2 - 6, wy + int((y2 - y1) * 0.28)),
                  (210, 230, 240), -1)


def seed_demo(
    settings: Optional[Settings] = None, n: int = 40, seed: int = 7
) -> int:
    """Seed the database + evidence store with `n` illustrative events."""
    settings = settings or load_settings()
    rng = random.Random(seed)
    repo = Repository(settings.db_url)
    builder = EvidenceBuilder(settings.storage_dir)
    vehicles = list(_VEHICLE_COLORS.keys())

    total = 0
    for _ in range(n):
        frame = make_synthetic_scene(rng=rng)
        h, w = frame.shape[:2]
        vt = rng.choice(vehicles)
        bw = rng.randint(120, 200)
        bh = rng.randint(90, 150)
        x1 = rng.randint(40, w - bw - 40)
        y1 = rng.randint(int(h * 0.3), h - bh - 30)
        bbox = (x1, y1, x1 + bw, y1 + bh)
        _draw_vehicle(frame, bbox, _VEHICLE_COLORS[vt])

        det = Detection(cls=vt, bbox=bbox, confidence=round(rng.uniform(0.6, 0.95), 3))
        menu = _MENU[vt]
        chosen = rng.sample(menu, k=rng.randint(1, min(2, len(menu))))
        plate = Plate(
            text=random_plate(rng),
            confidence=round(rng.uniform(0.6, 0.95), 3),
            valid_format=True,
        )
        violations = [
            Violation(
                type=t,
                confidence=round(rng.uniform(0.65, 0.97), 3),
                bbox=bbox,
                vehicle_class=vt,
                plate=plate,
                detail={"riders": 3} if t == ViolationType.TRIPLE_RIDING else {},
            )
            for t in chosen
        ]
        ts = utcnow() - timedelta(
            hours=rng.uniform(0, 24), minutes=rng.uniform(0, 59)
        )
        evidence = builder.build(
            frame,
            [det],
            violations,
            location=f"Camera-{rng.randint(1, 6):02d}",
            processing_ms=round(rng.uniform(35, 120), 1),
            timestamp=ts,
        )
        total += repo.add_records(evidence["records"])
    return total


def run_demo_eval() -> dict:
    """Showcase the evaluation harness on small synthetic data."""
    labels = ["car", "motorcycle", "person"]
    gts = [
        [("car", (10, 10, 50, 50)), ("person", (60, 60, 80, 100))],
        [("motorcycle", (0, 0, 40, 40))],
    ]
    preds = [
        [("car", (12, 11, 52, 51), 0.92), ("person", (60, 62, 80, 101), 0.81)],
        [("motorcycle", (2, 2, 42, 42), 0.74), ("car", (100, 100, 140, 140), 0.3)],
    ]
    det = detection_map(preds, gts, labels)

    vtypes = ["no_helmet", "triple_riding", "red_light", "no_seatbelt"]
    y_true = [{"no_helmet", "triple_riding"}, {"red_light"}, {"no_seatbelt"}, set()]
    y_pred = [{"no_helmet", "triple_riding"}, {"red_light"}, set(), {"red_light"}]
    cls = multilabel_report(y_true, y_pred, vtypes)

    ocr_preds = ["MH 01 AB 1234", "DL 5C A 1234", "KA03MG2255"]
    ocr_gts = ["MH01AB1234", "DL05CA1234", "KA03MG2255"]
    ocr = {
        "exact_match": round(ocr_exact_match(ocr_preds, ocr_gts), 4),
        "mean_cer": round(
            sum(ocr_cer(p, g) for p, g in zip(ocr_preds, ocr_gts)) / len(ocr_gts), 4
        ),
    }
    return {"detection": det, "violation_classification": cls, "ocr": ocr}
