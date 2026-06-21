#!/usr/bin/env python3
"""Real YOLO end-to-end demo for TRACE.

Runs the full pipeline on a real image with real detection (requires the ML
extras: `pip install -r requirements-ml.txt`). Defaults to Ultralytics'
bundled street photo so it works with zero setup.

To make the violation/evidence path visible on a still image, it positions a
stop line on the lowest detected vehicle (clearly flagged below) — on real
deployments the stop line / signal / zones come from per-camera calibration in
config/default.yaml instead.

Usage:
  python scripts/demo_real.py [IMAGE_PATH]
"""

import sys
from collections import Counter

import cv2

from trace_cv.core.config import load_settings
from trace_cv.detection.detector import Detector
from trace_cv.pipeline import TracePipeline


def _default_image() -> str:
    from ultralytics.utils import ASSETS

    return str(ASSETS / "bus.jpg")


def main() -> int:
    img_path = sys.argv[1] if len(sys.argv) > 1 else _default_image()
    img = cv2.imread(img_path)
    if img is None:
        print(f"error: could not read {img_path}", file=sys.stderr)
        return 1
    h, w = img.shape[:2]
    print(f"image: {img_path} ({w}x{h})")

    settings = load_settings()

    # Pass 1 — real detection.
    det = Detector(settings.models.detector, conf=settings.thresholds.detection_conf)
    dets = det.detect(img)
    if not det.available:
        print("Detector unavailable — install ML extras: "
              "pip install -r requirements-ml.txt", file=sys.stderr)
        return 2
    print(f"\nreal detections: {len(dets)}  {dict(Counter(d.cls for d in dets))}")
    for d in sorted(dets, key=lambda d: -d.confidence)[:8]:
        print(f"  - {d.cls:<12} {int(d.confidence*100)}%  {[round(x) for x in d.bbox]}")

    # DEMO-ONLY: place a stop line on the lowest detected vehicle so the
    # geometry engine fires on these real detections.
    vehicles = [
        d for d in dets
        if d.vehicle_class.is_four_wheeler or d.vehicle_class.is_two_wheeler
    ]
    if vehicles:
        bottom = max(d.bbox[3] for d in vehicles)
        settings.scene.stop_line.enabled = True
        settings.scene.stop_line.y = int(bottom) - 10
        settings.scene.signal.enabled = False
        print(f"\n[demo] stop line placed at y={settings.scene.stop_line.y}")

    # Pass 2 — full pipeline (detect -> violations -> OCR -> evidence -> store).
    pipe = TracePipeline(settings)
    print("models:", pipe.model_status())
    res = pipe.process_image(img, location="Demo")
    print(f"\nprocessing: {res['processing_ms']} ms | "
          f"preprocessing: {res['quality']['applied'] or 'none'}")
    print(f"violations: {len(res['violations'])}")
    for v in res["violations"]:
        plate = f"  plate={v['plate']['text']}" if v.get("plate") else ""
        print(f"  - {v['label']} {int(v['confidence']*100)}%{plate}")
    print(f"\nevidence image: {res['evidence_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
