#!/usr/bin/env python3
"""Run the TRACE evaluation showcase (detection mAP, violation P/R/F1, OCR
CER / exact-match) on small synthetic data.

In a real evaluation you would pass predictions and ground truth from your
annotated test split into trace_cv.evaluation.metrics directly; this script
demonstrates the harness and output format.

Usage:  python scripts/run_eval.py
"""

import json

from trace_cv.demo import run_demo_eval

if __name__ == "__main__":
    print(json.dumps(run_demo_eval(), indent=2))
