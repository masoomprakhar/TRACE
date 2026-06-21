#!/usr/bin/env python3
"""Full TRACE evaluation — detection mAP, violation P/R/F1, OCR, efficiency.

Produces a concept-note-ready report with Accuracy, Precision, Recall,
F1-score, and mAP as required by the problem statement.

Prerequisites:
  pip install -r requirements-ml.txt
  export PYTHONPATH=$PWD
  export ROBOFLOW_API_KEY=...          # if using config/roboflow.yaml
  export TRACE_CONFIG=config/roboflow.yaml

Prepare labeled data (see docs/EVALUATION.md):
  data/eval/images/     — real traffic photos
  data/eval/manifest.json — ground truth labels

Usage:
  # Full pipeline eval (detection + violations + OCR)
  python scripts/run_full_eval.py --config config/roboflow.yaml

  # Violation-logic only (uses GT boxes from manifest — isolates rules/models)
  python scripts/run_full_eval.py --config config/roboflow.yaml --gt-detections

  # Rebuild synthetic set (smoke test only — replace with real images for submission)
  python scripts/run_full_eval.py --rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a (no ground truth)"


def _print_detection(det: dict) -> None:
    print(f"  mAP@0.5      : {_fmt(det.get('map50'))}")
    print(f"  mAP@0.5:0.95 : {_fmt(det.get('map5095'))}")
    n_eval = det.get("n_classes_evaluated")
    if n_eval is not None:
        print(f"  Classes evaluated (have GT or predictions): {n_eval}")
    per = det.get("per_class_ap50") or {}
    support = det.get("per_class_support") or {}
    if per:
        print("  Per-class AP@0.5  (n_gt / n_pred):")
        for cls, ap in per.items():
            s = support.get(cls, {})
            print(
                f"    {cls:<16} {ap:.4f}   "
                f"({s.get('n_gt', '?')} / {s.get('n_pred', '?')})"
            )
    # Flag any class that was excluded so a reader sees WHY mAP moved.
    excluded = [c for c, s in support.items() if s.get("n_gt", 0) == 0 and s.get("n_pred", 0) == 0]
    if excluded:
        print(f"  Excluded (absent from GT and predictions): {', '.join(excluded)}")


def _print_violations(vc: dict) -> None:
    if "note" in vc:
        print(f"  ({vc['note']})")
        return
    macro = vc.get("macro", {})
    micro = vc.get("micro", {})
    print("  Macro averages:")
    print(f"    Precision : {macro.get('precision', 0):.4f}")
    print(f"    Recall    : {macro.get('recall', 0):.4f}")
    print(f"    F1-score  : {macro.get('f1', 0):.4f}")
    print("  Micro averages:")
    print(f"    Precision : {micro.get('precision', 0):.4f}")
    print(f"    Recall    : {micro.get('recall', 0):.4f}")
    print(f"    F1-score  : {micro.get('f1', 0):.4f}")
    print(f"    Accuracy  : {micro.get('accuracy', 0):.4f}")
    print("  Per violation type:")
    for label, m in (vc.get("per_label") or {}).items():
        print(
            f"    {label:<18} P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  Acc={m['accuracy']:.3f}"
        )


def _print_ocr(ocr: dict) -> None:
    em = ocr.get("exact_match")
    cer = ocr.get("mean_cer")
    if em is None and cer is None:
        print(f"  ({ocr.get('note', 'no plate ground truth in manifest')})")
        return
    print(f"  Exact match  : {em:.4f}" if em is not None else "  Exact match  : n/a")
    print(f"  Mean CER     : {cer:.4f}" if cer is not None else "  Mean CER     : n/a")


def _run_benchmark(config: str | None, image: str | None, iters: int) -> dict:
    import statistics
    import time

    import cv2

    from trace_cv.core.config import load_settings
    from trace_cv.demo import make_synthetic_scene
    from trace_cv.pipeline import TracePipeline

    settings = load_settings(config)
    pipe = TracePipeline(settings)
    if image:
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(image)
    else:
        img = make_synthetic_scene()

    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        pipe.process_image(img, location="Benchmark", persist=False)
        times.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(times)
    return {
        "iterations": iters,
        "mean_ms_per_frame": round(mean_ms, 2),
        "fps": round(1000.0 / mean_ms, 2) if mean_ms else 0.0,
        "p95_ms": round(sorted(times)[int(0.95 * len(times)) - 1], 2),
    }


def print_report(results: dict, bench: dict | None) -> None:
    _banner("TRACE EVALUATION REPORT")
    print(f"  Config         : {results.get('config')}")
    print(f"  Samples        : {results.get('n_samples')}")
    print(f"  GT detections  : {results.get('use_gt_detections')}")
    print(f"  Model status   : {json.dumps(results.get('model_status', {}))}")

    _banner("1. DETECTION — mAP (object localization)")
    _print_detection(results.get("detection", {}))

    _banner("2. VIOLATION CLASSIFICATION — P / R / F1 / Accuracy")
    print(f"  Exact-set accuracy : {results.get('exact_match_accuracy', 0):.4f}")
    _print_violations(results.get("violation_classification", {}))

    _banner("3. LICENSE PLATE OCR")
    _print_ocr(results.get("ocr", {}))

    if bench:
        _banner("4. COMPUTATIONAL EFFICIENCY")
        print(f"  Mean latency   : {bench['mean_ms_per_frame']} ms/frame")
        print(f"  P95 latency    : {bench['p95_ms']} ms")
        print(f"  Throughput     : {bench['fps']} FPS")

    plate_det = results.get("plate_detection") or {}
    if plate_det.get("map50") is not None:
        _banner("PLATE DETECTION (Roboflow workflow)")
        _print_detection(plate_det)

    _banner("CONCEPT NOTE — copy these numbers")
    det = results.get("detection", {})
    vc = results.get("violation_classification", {})
    macro = vc.get("macro", {}) if isinstance(vc, dict) else {}
    def _cell(x):
        return x if isinstance(x, (int, float)) else "n/a"

    print(
        f"| Detection mAP@0.5 | {_cell(det.get('map50'))} |\n"
        f"| Detection mAP@0.5:0.95 | {_cell(det.get('map5095'))} |\n"
        f"| Detection classes evaluated | {det.get('n_classes_evaluated', 'n/a')} |\n"
        f"| Violation macro-F1 | {_cell(macro.get('f1'))} |\n"
        f"| Violation macro-Precision | {_cell(macro.get('precision'))} |\n"
        f"| Violation macro-Recall | {_cell(macro.get('recall'))} |\n"
        f"| End-to-end latency | {bench['mean_ms_per_frame'] if bench else 'n/a'} ms |"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Full TRACE evaluation report")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="data/eval/results.json")
    p.add_argument("--report", default="data/eval/REPORT.txt")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--gt-detections", action="store_true")
    p.add_argument("--skip-benchmark", action="store_true")
    p.add_argument("--benchmark-image")
    p.add_argument("--benchmark-iters", type=int, default=20)
    args = p.parse_args()

    import os

    from trace_cv.evaluation.dataset import build_eval_set
    from trace_cv.evaluation.runner import run_eval

    config = args.config or os.environ.get("TRACE_CONFIG")

    manifest_path = ROOT / "data" / "eval" / "manifest.json"
    if args.rebuild or not manifest_path.exists():
        build_eval_set()
        print("Built synthetic eval set (replace with real images for submission).")

    if manifest_path.exists():
        n = len(json.loads(manifest_path.read_text()).get("samples", []))
        if n < 10:
            print(
                "WARNING: fewer than 10 eval images. Add real labeled photos to "
                "data/eval/ for credible mAP/F1 numbers.\n"
            )

    results = run_eval(
        config,
        ROOT / args.out,
        use_gt_detections=args.gt_detections,
    )

    bench = None
    if not args.skip_benchmark:
        try:
            bench = _run_benchmark(config, args.benchmark_image, args.benchmark_iters)
            results["efficiency"] = bench
            (ROOT / args.out).write_text(json.dumps(results, indent=2))
        except Exception as exc:
            print(f"Benchmark skipped: {exc}", file=sys.stderr)

    # Capture report to file and stdout
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(results, bench)
    report_text = buf.getvalue()
    print(report_text)
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    print(f"\nReport saved to {report_path}")
    print(f"JSON saved to  {ROOT / args.out}")

    from trace_cv.evaluation.summary_export import write_eval_summary

    summary_path = ROOT / "data" / "eval" / "eval-summary.json"
    write_eval_summary(results, summary_path)
    print(f"Dashboard summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
