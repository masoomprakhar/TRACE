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


def run_seatbelt_eval(config: str | None, via_pipeline: bool) -> dict:
    """Evaluate the seatbelt classifier on the cropped driver dataset.

    The Karan seat-belt dataset is *cropped driver/windshield images*, so the
    correct, isolating evaluation runs the seatbelt model directly on each
    whole image. (Running the full detection pipeline on these crops finds no
    "car" box, so the four-wheeler gate filters everything out and the model
    never runs — that is exactly the 0/0/4 result the old default produced.)
    Pass ``--via-pipeline`` to instead test end-to-end on full street scenes.

    A sample is a positive if its GT contains ``no_seatbelt``; an empty GT is a
    negative (belt worn / not applicable) and feeds the false-positive rate.
    """
    import os

    import cv2

    from trace_cv.core.config import load_settings
    from trace_cv.pipeline import TracePipeline
    from trace_cv.violation.seatbelt import SeatbeltDetector

    settings = load_settings(config or os.environ.get("TRACE_CONFIG"))
    pipe = TracePipeline(settings)

    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing {MANIFEST}. Run scripts/download_seatbelt_karan_eval.py")

    data = json.loads(MANIFEST.read_text())
    samples = data.get("samples", [])

    seat_mod = next(
        (m for m in pipe.engine.modules if isinstance(m, SeatbeltDetector)), None
    )
    model = getattr(seat_mod, "model", None)
    model_ready = model is not None and getattr(model, "available", False)
    if not model_ready and not via_pipeline:
        # Be honest rather than silently scoring 0/0/N against a missing model.
        return {
            "config": config,
            "n_samples": len(samples),
            "mode": "classifier-direct",
            "model_status": pipe.model_status(),
            "error": (
                "Seatbelt model not loaded (no weights). Provide a seatbelt "
                "checkpoint in the config to populate real numbers."
            ),
            "no_seatbelt": _metrics(0, 0, 0),
            "negative_fp_rate": 0.0,
            "negative_fp": 0,
            "negative_total": 0,
        }

    thr = settings.thresholds.seatbelt_conf
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
        gt_no = "no_seatbelt" in gt
        is_negative = len(gt) == 0

        if via_pipeline:
            result = pipe.process_image(frame, location="seatbelt-eval", persist=False)
            pred = {v.get("type") for v in result.get("violations", [])}
            pred_no = "no_seatbelt" in pred
        else:
            # Classifier-direct: the cropped image IS the driver region.
            label, conf = model.predict(frame)
            pred_no = label == "no_belt" and conf >= thr

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
        "mode": "pipeline" if via_pipeline else "classifier-direct",
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
        f"  Mode           : {results.get('mode')}",
        f"  Model status   : {json.dumps(results.get('model_status', {}))}",
    ]
    if results.get("error"):
        lines += ["", f"  NOTE: {results['error']}"]
    lines += [
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
    p.add_argument(
        "--via-pipeline",
        action="store_true",
        help="Run end-to-end on full scenes instead of the classifier-direct "
        "default (the dataset is cropped driver images, so direct is correct).",
    )
    p.add_argument("--out", default=str(ROOT / "data" / "eval" / "seatbelt_results.json"))
    p.add_argument("--report", default=str(REPORT))
    args = p.parse_args()

    try:
        results = run_seatbelt_eval(args.config, args.via_pipeline)
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
