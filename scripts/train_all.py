#!/usr/bin/env python3
"""Full TRACE training pipeline for all problem-statement models.

Stages:
  1. prepare_all_datasets.py  — Roboflow downloads + local packs + inference OCR labels
  2. YOLO detector            — training/train_detector.py
  3. Rider multi-label CNN    — training/train_rider_multilabel_cnn.py
  4. Seatbelt classifier      — training/train_seatbelt.py
  5. TrOCR plate OCR          — training/train_trocr_plate.py
  6. Evaluation               — scripts/run_full_eval.py

Geometric violations (wrong-side, stop-line, red-light, parking) use scene config +
tracking — no separate model training; calibrate via scripts/calibrate_camera.py.

Usage:
  export ROBOFLOW_API_KEY=...
  python scripts/train_all.py --device cuda --epochs 30
  python scripts/train_all.py --prepare-only
  python scripts/train_all.py --skip-prepare --skip-trocr
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEIGHTS = ROOT / "models" / "weights"
YOLO_DATA = ROOT / "data" / "datasets" / "quick_viovision" / "data.yaml"
RIDER_DATA = ROOT / "data" / "datasets" / "rider_multilabel"
OCR_DATA = ROOT / "data" / "ocr" / "lines"
SEATBELT_DATA = ROOT / "data" / "datasets" / "seatbelt_cls"
VIIO_BEST = (
    ROOT / "viovision" / "runs" / "detect" / "runs" / "train" / "viovision_yolo11n" / "weights" / "best.pt"
)
EVAL_SUMMARY = ROOT / "data" / "eval" / "eval-summary.json"


def run(cmd: list[str], desc: str) -> bool:
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    print(" ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"FAILED: {desc} (exit {r.returncode})", file=sys.stderr)
        return False
    return True


def copy_yolo_best() -> None:
    candidates = sorted(
        (ROOT / "runs" / "detect").glob("**/quick_eval_boost*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates += sorted(
        (ROOT / "viovision" / "runs").glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for src in candidates:
        if src.resolve() == VIIO_BEST.resolve():
            continue
        VIIO_BEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, VIIO_BEST)
        print(f"Copied detector weights: {src} -> {VIIO_BEST}")
        return


def train_yolo(device: str, epochs: int, batch: int) -> bool:
    if not YOLO_DATA.exists():
        print("Skip YOLO: run prepare_all_datasets.py first", file=sys.stderr)
        return False
    ok = run(
        [
            sys.executable,
            str(ROOT / "training" / "train_detector.py"),
            "--data",
            str(YOLO_DATA),
            "--epochs",
            str(epochs),
            "--batch",
            str(batch),
            "--device",
            device,
            "--name",
            "quick_eval_boost",
        ],
        "YOLO detector (motorcycle / plate / signal)",
    )
    if ok:
        copy_yolo_best()
    return ok


def train_rider_cnn(device: str, epochs: int, batch: int) -> bool:
    labels = RIDER_DATA / "labels.csv"
    if not labels.exists():
        print("Skip rider CNN: missing labels.csv", file=sys.stderr)
        return False
    ok = run(
        [
            sys.executable,
            str(ROOT / "training" / "train_rider_multilabel_cnn.py"),
            "--data",
            str(RIDER_DATA),
            "--epochs",
            str(epochs),
            "--batch",
            str(batch),
            "--device",
            device,
        ],
        "Rider multi-label CNN (helmet / passengers)",
    )
    return ok


def train_seatbelt(device: str, epochs: int) -> bool:
    if not SEATBELT_DATA.exists():
        print("Skip seatbelt: missing dataset dir", file=sys.stderr)
        return False
    return run(
        [
            sys.executable,
            str(ROOT / "training" / "train_seatbelt.py"),
            "--data",
            str(SEATBELT_DATA),
            "--epochs",
            str(epochs),
            "--device",
            device,
        ],
        "Seatbelt classifier",
    )


def train_trocr(device: str, epochs: int) -> bool:
    manifest = OCR_DATA / "manifest.json"
    if not manifest.exists():
        print("Skip TrOCR: missing OCR manifest", file=sys.stderr)
        return False
    return run(
        [
            sys.executable,
            str(ROOT / "training" / "train_trocr_plate.py"),
            "--data",
            str(OCR_DATA.relative_to(ROOT)),
            "--epochs",
            str(epochs),
            "--batch",
            "4",
            "--device",
            device,
        ],
        "TrOCR plate line OCR",
    )


def run_eval() -> bool:
    report = ROOT / "data" / "eval" / "REPORT-train-all.txt"
    ok = run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_full_eval.py"),
            "--config",
            "config/roboflow.yaml",
            "--report",
            str(report.relative_to(ROOT)),
        ],
        "Full evaluation on eval manifest",
    )
    if ok and report.exists():
        _write_eval_summary(report)
    return ok


def _write_eval_summary(report_path: Path) -> None:
    """Parse REPORT text into eval-summary.json for the dashboard."""
    text = report_path.read_text()
    summary: dict = {"source": str(report_path.name), "metrics": {}}
    for line in text.splitlines():
        line = line.strip()
        if "mAP@0.5" in line and "motorcycle" not in line.lower():
            try:
                summary["metrics"]["detection_map50"] = float(line.split()[-1])
            except ValueError:
                pass
        if "motorcycle" in line.lower() and "ap" in line.lower():
            try:
                summary["metrics"]["motorcycle_ap50"] = float(line.split()[-1])
            except ValueError:
                pass
        if "no_helmet" in line.lower() and "f1" in line.lower():
            try:
                summary["metrics"]["no_helmet_f1"] = float(line.split()[-1])
            except ValueError:
                pass
        if "exact_match" in line.lower() or "exact match" in line.lower():
            try:
                summary["metrics"]["ocr_exact_match"] = float(line.split()[-1])
            except ValueError:
                pass
    EVAL_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    EVAL_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"Eval summary -> {EVAL_SUMMARY}")


def main() -> int:
    p = argparse.ArgumentParser(description="Train all TRACE models (full pipeline)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--skip-yolo", action="store_true")
    p.add_argument("--skip-rider", action="store_true")
    p.add_argument("--skip-seatbelt", action="store_true")
    p.add_argument("--skip-trocr", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    if args.prepare_only or not args.skip_prepare:
        prep = [sys.executable, str(ROOT / "scripts" / "prepare_all_datasets.py")]
        if not run(prep, "Dataset preparation (Roboflow + local packs)"):
            return 1
        if args.prepare_only:
            return 0

    results: dict[str, bool] = {}
    if not args.skip_yolo:
        results["yolo"] = train_yolo(args.device, args.epochs, args.batch)
    if not args.skip_rider:
        results["rider_cnn"] = train_rider_cnn(args.device, max(args.epochs, 20), args.batch)
    if not args.skip_seatbelt:
        results["seatbelt"] = train_seatbelt(args.device, max(args.epochs, 12))
    if not args.skip_trocr:
        results["trocr"] = train_trocr(args.device, min(args.epochs, 12))

    print("\n--- Training summary ---")
    for name, passed in results.items():
        print(f"  {name}: {'OK' if passed else 'FAILED/SKIPPED'}")

    if not args.skip_eval:
        run_eval()

    print("\n--- Product config ---")
    print("  Runtime:  export TRACE_CONFIG=config/roboflow.yaml && python -m trace_cv.cli serve")
    print("  Weights:  models/weights/ + viovision/.../best.pt")
    print("  Registry: config/roboflow_models.yaml")

    if not results:
        return 0
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
