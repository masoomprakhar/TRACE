#!/usr/bin/env python3
"""Prepare model weights so TRACE runs out of the box.

What this does:
  * Triggers the YOLO COCO detector download (yolov8n.pt) so vehicle / person /
    traffic-light detection works offline after the first run — this is the
    zero-key default path (config/default.yaml).
  * Reports which OPTIONAL weights referenced by the configs are present vs
    missing, and prints the exact command to train each one.

What this deliberately does NOT do: fabricate or download project-specific
weights (seatbelt / helmet-SVM / TrOCR / VioVision YOLO) — those are trained on
your data. Missing optional weights degrade honestly: the corresponding module
is skipped, never faked. See README and scripts/train_all.py.

Usage:
  python scripts/fetch_weights.py            # fetch COCO + report status
  python scripts/fetch_weights.py --no-yolo  # status only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (config key, path it points to, how to produce it)
OPTIONAL_WEIGHTS = [
    ("rider_cnn", "models/weights/rider_multilabel_cnn.pt",
     "python scripts/train_all.py  (rider CNN stage)  — ships in repo"),
    ("seatbelt", "models/weights/seatbelt.pt",
     "python training/seatbelt/train.py  (or VioVision seatbelt_svm.pkl)"),
    ("helmet_svm", "viovision/models/weights/helmet_svm.pkl",
     "VioVision pipeline  (viovision/PIPELINE.md)"),
    ("trocr", "models/weights/trocr_plate",
     "python scripts/train_all.py  (TrOCR stage)  — or use ocr_backend: easyocr"),
    ("viovision_yolo", "viovision/runs/detect/runs/train/viovision_yolo11n/weights/best.pt",
     "python scripts/train_all.py --device cuda  (YOLO stage)"),
]


def fetch_yolo() -> bool:
    """Ask Ultralytics to fetch the small COCO detector (auto-cached)."""
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"  ultralytics not installed ({exc}).")
        print("  -> pip install -r requirements-ml.txt   then re-run.")
        return False
    try:
        YOLO("yolov8n.pt")  # downloads + caches on first call
        print("  OK: yolov8n.pt ready (COCO detector for the zero-key default).")
        return True
    except Exception as exc:
        print(f"  Could not fetch yolov8n.pt ({exc}). Check network access.")
        return False


def report_status() -> None:
    print("\nOptional weights referenced by configs:")
    for name, rel, how in OPTIONAL_WEIGHTS:
        present = (ROOT / rel).exists()
        mark = "present " if present else "MISSING "
        print(f"  [{mark}] {name:14} {rel}")
        if not present:
            print(f"             produce: {how}")
    print(
        "\nMissing optional weights are fine: that module is skipped (no faked "
        "violations). The zero-key path needs only yolov8n.pt + EasyOCR."
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch/verify TRACE model weights")
    p.add_argument("--no-yolo", action="store_true", help="skip COCO download")
    args = p.parse_args()

    print("=" * 60)
    print("TRACE — weight preparation")
    print("=" * 60)
    if not args.no_yolo:
        print("\nFetching COCO detector (zero-key default):")
        fetch_yolo()
    report_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
