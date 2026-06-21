#!/usr/bin/env python3
"""Computational-efficiency benchmark for TRACE.

Reports per-stage latency (preprocess / detect / violation engine) and overall
throughput (FPS) — the efficiency + scalability metric the brief asks for.
Falls back to a synthetic scene if no image is given; uses real models if the
ML extras are installed.

Usage:
  python scripts/benchmark.py [--image PATH] [-n 30]
"""

import argparse
import statistics
import time

import cv2

from trace_cv.core.config import load_settings
from trace_cv.demo import make_synthetic_scene
from trace_cv.pipeline import TracePipeline


def _time(fn) -> tuple[object, float]:
    t = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("-n", "--iters", type=int, default=30)
    ap.add_argument("--config")
    args = ap.parse_args()

    settings = load_settings(args.config)
    pipe = TracePipeline(settings)
    print("device:", settings.device, "| models:", pipe.model_status())

    img = cv2.imread(args.image) if args.image else make_synthetic_scene()
    if img is None:
        print(f"error: could not read {args.image}")
        return 1
    h, w = img.shape[:2]
    print(f"image: {w}x{h}  | iterations: {args.iters}\n")

    # Warm-up (model load / first-call JIT).
    enh, _ = pipe.pre.process(img)
    pipe.detector.detect(enh)

    pre_t, det_t, vio_t = [], [], []
    for _ in range(args.iters):
        (enhanced, _q), dt_pre = _time(lambda: pipe.pre.process(img))
        dets, dt_det = _time(lambda e=enhanced: pipe.detector.detect(e))
        _, dt_vio = _time(lambda e=enhanced, d=dets: pipe.engine.run(e, d))
        pre_t.append(dt_pre)
        det_t.append(dt_det)
        vio_t.append(dt_vio)

    def row(name, xs):
        print(f"  {name:<16} mean {statistics.mean(xs):6.1f} ms   "
              f"median {statistics.median(xs):6.1f} ms")

    row("preprocess", pre_t)
    row("detect", det_t)
    row("violation engine", vio_t)
    total = [a + b + c for a, b, c in zip(pre_t, det_t, vio_t)]
    row("end-to-end", total)
    print(f"\nthroughput: {1000.0 / statistics.mean(total):.1f} FPS "
          f"({'GPU' if settings.device == 'cuda' else 'CPU'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
