#!/usr/bin/env python3
"""Prepare all TRACE training datasets (Roboflow download + local packs).

Covers problem-statement models:
  - YOLO detector (motorcycle, plate, traffic light)
  - Rider multi-label CNN (helmet / triple riding)
  - Seatbelt classifier
  - TrOCR plate line OCR (Roboflow char boxes + inference labels)

Prerequisites:
  export ROBOFLOW_API_KEY=...   # see .env.example

Usage:
  python scripts/prepare_all_datasets.py
  python scripts/prepare_all_datasets.py --skip-download --skip-inference
  python scripts/prepare_all_datasets.py --quick-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.load_roboflow_config import api_key  # noqa: E402


def run(cmd: list[str], desc: str, *, required: bool = True) -> bool:
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    print(" ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        msg = f"FAILED: {desc} (exit {r.returncode})"
        if required:
            print(msg, file=sys.stderr)
        else:
            print(f"WARN: {msg}", file=sys.stderr)
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare all TRACE training datasets")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-rider", action="store_true")
    p.add_argument("--skip-plate-lines", action="store_true")
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--skip-seatbelt", action="store_true")
    p.add_argument("--skip-yolo-pack", action="store_true")
    p.add_argument("--quick-only", action="store_true", help="Only prepare_quick_train_pack")
    p.add_argument("--max-eval-images", type=int, default=80)
    args = p.parse_args()

    py = sys.executable
    ok = True

    if args.quick_only:
        return 0 if run([py, "scripts/prepare_quick_train_pack.py"], "Quick YOLO/rider pack") else 1

    if not args.skip_download:
        if not api_key():
            print("ROBOFLOW_API_KEY not set — skipping Roboflow downloads.", file=sys.stderr)
        else:
            ok &= run(
                [
                    py,
                    "scripts/roboflow_download_eval.py",
                    "--kind",
                    "both",
                    "--max-images",
                    str(args.max_eval_images),
                ],
                "Download helmet + plate eval sets from Roboflow",
                required=False,
            )

    if not args.skip_rider:
        ok &= run(
            [py, "scripts/build_rider_multilabel_dataset.py", "--synthetic"],
            "Rider multi-label dataset (helmet / triple)",
            required=False,
        )

    if not args.skip_plate_lines:
        ok &= run(
            [py, "scripts/build_plate_line_dataset.py"],
            "Plate line OCR dataset (Roboflow char YOLO)",
            required=False,
        )

    if not args.skip_inference:
        if api_key():
            ok &= run(
                [
                    py,
                    "scripts/build_ocr_from_roboflow_inference.py",
                    "--backend",
                    "model",
                    "--merge",
                    "--max-images",
                    str(args.max_eval_images),
                ],
                "OCR line labels via ocr-character-cgtzm/4 inference",
                required=False,
            )
            ok &= run(
                [
                    py,
                    "scripts/build_ocr_from_roboflow_inference.py",
                    "--backend",
                    "workflow",
                    "--merge",
                    "--max-images",
                    str(max(20, args.max_eval_images // 2)),
                ],
                "OCR line labels via general-segmentation-api-4 workflow",
                required=False,
            )
        else:
            print("Skip inference OCR labeling (no API key).", file=sys.stderr)

    if not args.skip_seatbelt:
        ok &= run(
            [py, "viovision/scripts/prepare_seatbelt_karan_dataset.py"],
            "Seatbelt cls dataset (Karan Panja Roboflow)",
            required=False,
        )

    if not args.skip_yolo_pack:
        ok &= run(
            [py, "scripts/prepare_quick_train_pack.py"],
            "Merged YOLO + eval rider crops pack",
            required=False,
        )

    print("\n--- Dataset preparation complete ---")
    print("  config registry: config/roboflow_models.yaml")
    print("  YOLO pack:       data/datasets/quick_viovision/data.yaml")
    print("  Rider CNN:       data/datasets/rider_multilabel/")
    print("  Seatbelt:        data/datasets/seatbelt_cls/")
    print("  OCR lines:       data/ocr/lines/manifest.json")
    print("\nNext: python scripts/train_all.py --device cuda")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
