#!/usr/bin/env python3
"""Seatbelt-only TRACE evaluation — P/R/F1 + negative FP rate.

Uses data/eval/seatbelt_manifest.json (from download_seatbelt_karan_eval.py).

Usage:
  python scripts/download_seatbelt_karan_eval.py
  python scripts/eval_seatbelt.py --config config/viovision.yaml
  python scripts/eval_seatbelt.py --gt-detections
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "data" / "eval" / "seatbelt_manifest.json"
REPORT = ROOT / "data" / "eval" / "REPORT-seatbelt.txt"


def _metrics(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    acc = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc, "tp": tp, "fp": fp, "fn": fn}


def run_seatbelt_eval(config: str | None, use_gt: bool) -> dict:
    import os

    import cv2

    from trace_cv.core.config import load_settings
    from trace_cv.core.types import Detection
    from trace_cv.pipeline import TracePipeline

    settings = load_settings(config or os.environ.get("TRACE_CONFIG"))
    pipe = TracePipeline(settings)

    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing {MANIFEST}. Run scripts/download_seatbelt_karan_eval.py")

    data = json.loads(MANIFEST.read_text())
    samples = data.get("samples", [])

    tp = fp = fn = 0
    neg_fp = 0
    neg_total = 0

    for s in samples:
        img_rel = s.get("image", "")
        img_path = ROOT / "data" / "eval" / img_rel
        if not img_path.exists():
            img_path = ROOT / img_rel
        if not img_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        gt = set(s.get("violations") or [])
        is_negative = len(gt) == 0 and "negative" in str(s.get("detail", {})).lower()

        if use_gt:
            h, w = frame.shape[:2]
            dets = [
                Detection(
                    cls="car",
                    confidence=0.99,
                    bbox=(0.0, 0.0, float(w), float(h)),
                )
            ]
            from trace_cv.violation.base import ViolationContext
            from trace_cv.violation.seatbelt import SeatbeltDetector

            seat_mod = next(
                (m for m in pipe.engine.modules if m.__class__.__name__ == "SeatbeltDetector"),
                None,
            )
            if seat_mod is None:
                pred = set()
            else:
                ctx = ViolationContext(
                    frame=frame,
                    detections=dets,
                    thresholds=settings.thresholds,
                    scene=settings.scene,
                    frame_index=0,
                )
                pred = {v.type.value for v in seat_mod.check(ctx)}
        else:
            result = pipe.process_image(frame, location="seatbelt-eval", persist=False)
            pred = {v.get("type") for v in result.get("violations", [])}

        pred_no = "no_seatbelt" in pred
        gt_no = "no_seatbelt" in gt

        if is_negative:
            neg_total += 1
            if pred_no:
                neg_fp += 1
            continue

        if gt_no and pred_no:
            tp += 1
        elif gt_no and not pred_no:
            fn += 1
        elif not gt_no and pred_no:
            fp += 1

    m = _metrics(tp, fp, fn)
    fp_rate = neg_fp / neg_total if neg_total else 0.0
    return {
        "config": config,
        "n_samples": len(samples),
        "use_gt_detections": use_gt,
        "model_status": pipe.model_status(),
        "no_seatbelt": m,
        "negative_fp_rate": fp_rate,
        "negative_fp": neg_fp,
        "negative_total": neg_total,
    }


def print_report(results: dict) -> str:
    lines = [
        "=" * 60,
        "SEATBELT EVALUATION REPORT",
        "=" * 60,
        f"  Config         : {results.get('config')}",
        f"  Samples        : {results.get('n_samples')}",
        f"  GT detections  : {results.get('use_gt_detections')}",
        f"  Model status   : {json.dumps(results.get('model_status', {}))}",
        "",
        "no_seatbelt metrics:",
    ]
    m = results.get("no_seatbelt", {})
    lines += [
        f"  Precision : {m.get('precision', 0):.4f}",
        f"  Recall    : {m.get('recall', 0):.4f}",
        f"  F1-score  : {m.get('f1', 0):.4f}",
        f"  TP/FP/FN  : {m.get('tp', 0)}/{m.get('fp', 0)}/{m.get('fn', 0)}",
        "",
        f"Negative FP rate (bike/plate): {results.get('negative_fp_rate', 0):.4f} "
        f"({results.get('negative_fp', 0)}/{results.get('negative_total', 0)})",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Seatbelt eval report")
    p.add_argument("--config", default="config/viovision.yaml")
    p.add_argument("--gt-detections", action="store_true")
    p.add_argument("--out", default=str(ROOT / "data" / "eval" / "seatbelt_results.json"))
    p.add_argument("--report", default=str(REPORT))
    args = p.parse_args()

    try:
        results = run_seatbelt_eval(args.config, args.gt_detections)
    except Exception as exc:
        print(f"eval failed: {exc}", file=sys.stderr)
        return 1

    report = print_report(results)
    print(report)
    Path(args.out).write_text(json.dumps(results, indent=2))
    Path(args.report).write_text(report + "\n")
    print(f"\nReport saved to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
