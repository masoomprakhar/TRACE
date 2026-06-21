#!/usr/bin/env python3
"""Quick-train all four model gaps on small local datasets (CPU-friendly).

Prerequisites:
  python scripts/prepare_quick_train_pack.py

Usage:
  python scripts/run_quick_train.py
  python scripts/run_quick_train.py --device cuda --epochs 30
  python scripts/run_quick_train.py --skip-yolo --skip-trocr
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
VIIO_BEST = ROOT / "viovision" / "runs" / "detect" / "runs" / "train" / "viovision_yolo11n" / "weights" / "best.pt"


def run(cmd: list[str], desc: str) -> bool:
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    print(" ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"FAILED: {desc} (exit {r.returncode})", file=sys.stderr)
        return False
    return True


def train_yolo(device: str, epochs: int, batch: int) -> bool:
    if not YOLO_DATA.exists():
        print("Skip YOLO: run prepare_quick_train_pack.py first", file=sys.stderr)
        return False
    base = "yolo11n.pt"
    ok = run(
        [
            sys.executable,
            str(ROOT / "training" / "train_detector.py"),
            "--data",
            str(YOLO_DATA),
            "--weights",
            base,
            "--epochs",
            str(max(epochs, 20)),
            "--imgsz",
            "640",
            "--batch",
            str(batch),
            "--device",
            device,
            "--name",
            "quick_eval_boost",
            "--project",
            "runs/trace",
        ],
        "YOLO detector (two_wheeler + plate)",
    )
    candidates = sorted(
        (ROOT / "runs" / "detect" / "runs" / "trace").glob("quick_eval_boost*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) + [
        ROOT / "runs" / "trace" / "quick_eval_boost" / "weights" / "best.pt",
        WEIGHTS / "vehicle_detector.pt",
    ]
    src_pt = next((p for p in candidates if p.exists()), None)
    if src_pt:
        dest_dir = VIIO_BEST.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        if src_pt.resolve() != VIIO_BEST.resolve():
            shutil.copy2(src_pt, VIIO_BEST)
        if src_pt.resolve() != (WEIGHTS / "vehicle_detector.pt").resolve():
            shutil.copy2(src_pt, WEIGHTS / "vehicle_detector.pt")
        print(f"Detector weights ready -> {VIIO_BEST}")
        return True
    return ok


def train_rider_cnn(device: str, epochs: int, batch: int) -> bool:
    if not (RIDER_DATA / "labels.csv").exists():
        print("Skip rider CNN: no labels.csv", file=sys.stderr)
        return False
    return run(
        [
            sys.executable,
            str(ROOT / "training" / "train_rider_multilabel_cnn.py"),
            "--data",
            str(RIDER_DATA.relative_to(ROOT)),
            "--epochs",
            str(epochs),
            "--batch",
            str(batch),
            "--device",
            device,
            "--name",
            "rider_multilabel_cnn.pt",
        ],
        "Rider multi-label CNN",
    )


def train_seatbelt(device: str, epochs: int) -> bool:
    if not (SEATBELT_DATA / "train").exists():
        print("Skip seatbelt: no seatbelt_cls/", file=sys.stderr)
        return False
    return run(
        [
            sys.executable,
            str(ROOT / "training" / "train_seatbelt.py"),
            "--mode",
            "cls",
            "--data",
            str(SEATBELT_DATA.relative_to(ROOT)),
            "--epochs",
            str(epochs),
            "--imgsz",
            "128",
            "--batch",
            "16",
            "--device",
            device,
            "--name",
            "seatbelt_quick",
        ],
        "Seatbelt classifier",
    )


def train_trocr(device: str, epochs: int) -> bool:
    if not (OCR_DATA / "manifest.json").exists():
        print("Skip TrOCR: no OCR manifest", file=sys.stderr)
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
        "TrOCR plate OCR",
    )


def run_eval() -> bool:
    # Write to the single canonical report (data/eval/REPORT.txt) so there is
    # exactly one source of truth regardless of which pipeline produced it.
    return run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_full_eval.py"),
            "--config",
            "config/roboflow.yaml",
        ],
        "Full evaluation",
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Quick train all models on small packs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--skip-yolo", action="store_true")
    p.add_argument("--skip-rider", action="store_true")
    p.add_argument("--skip-seatbelt", action="store_true")
    p.add_argument("--skip-trocr", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()

    if args.prepare_only:
        return subprocess.call([sys.executable, str(ROOT / "scripts" / "prepare_quick_train_pack.py")])

    prep = ROOT / "scripts" / "prepare_quick_train_pack.py"
    if not (RIDER_DATA / "labels.csv").exists() or not YOLO_DATA.exists():
        print("Preparing datasets…")
        subprocess.check_call([sys.executable, str(prep)])

    results = {}
    if not args.skip_yolo:
        results["yolo"] = train_yolo(args.device, args.epochs, args.batch)
    if not args.skip_rider:
        results["rider"] = train_rider_cnn(args.device, max(args.epochs, 20), args.batch)
    if not args.skip_seatbelt:
        results["seatbelt"] = train_seatbelt(args.device, max(args.epochs, 12))
    if not args.skip_trocr:
        results["trocr"] = train_trocr(args.device, min(args.epochs, 8))

    print("\n--- Training summary ---")
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAILED/SKIPPED'}")

    if not args.skip_eval:
        run_eval()

    report = ROOT / "data" / "eval" / "REPORT.txt"
    results = ROOT / "data" / "eval" / "results.json"
    if results.exists():
        from trace_cv.evaluation.summary_export import write_eval_summary
        import json
        write_eval_summary(json.loads(results.read_text()), ROOT / "data" / "eval" / "eval-summary.json")
    if report.exists():
        print(f"\nReport: {report}")

    return 0 if not results or all(v for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
