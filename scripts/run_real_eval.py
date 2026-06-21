#!/usr/bin/env python3
"""Run TRACE on a labeled holdout set and report real metrics.

Usage:
  export PYTHONPATH=$PWD
  export TRACE_CONFIG=config/viovision.yaml
  python scripts/run_real_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_cv.evaluation.dataset import build_eval_set
from trace_cv.evaluation.runner import run_eval


def main() -> int:
    p = argparse.ArgumentParser(description="Real TRACE evaluation on labeled holdout set")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="data/eval/results.json")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--gt-detections", action="store_true")
    p.add_argument("--persist", action="store_true")
    args = p.parse_args()

    if args.rebuild or not (ROOT / "data" / "eval" / "manifest.json").exists():
        path = build_eval_set()
        print(f"built eval set: {path}")

    if args.build_only:
        return 0

    import os

    config = args.config or os.environ.get("TRACE_CONFIG")
    results = run_eval(
        config, ROOT / args.out, persist=args.persist, use_gt_detections=args.gt_detections
    )
    print(json.dumps({k: v for k, v in results.items() if k != "samples"}, indent=2))
    print(f"\nWrote per-sample details to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
