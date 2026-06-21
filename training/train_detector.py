#!/usr/bin/env python3
"""Fine-tune a YOLO vehicle + person detector for TRACE.

The runtime ships on COCO weights (yolov8n.pt) which already cover person,
bicycle, car, motorcycle, bus, truck and traffic light. Fine-tune here when
you want a detector adapted to Indian traffic scenes — e.g. on the IDD /
India Driving Dataset exported in YOLO format (see training/README.md and
training/datasets/vehicle.example.yaml).

The class set is up to your dataset; TRACE's Detector keeps the COCO-style
names person/bicycle/car/motorcycle/bus/truck/traffic_light, so keep those
names if you want the rest of the pipeline (triple-riding, parking, ...) to
work unchanged. The best checkpoint is copied to models/weights/ and can be
wired in via `models.detector` in config/default.yaml.

Example:
  python training/train_detector.py \\
      --data training/datasets/vehicle.example.yaml \\
      --weights yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device 0

Requires the ML extras:  pip install -r requirements-ml.txt
"""

from __future__ import annotations

import argparse

from _common import ensure_weights_dir, load_yolo, publish_best, rel


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a YOLO vehicle/person detector for TRACE.",
    )
    p.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    p.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="base weights (yolov8n.pt; try yolov10n.pt for YOLOv10)",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="", help="'' (auto), 'cpu', '0', '0,1'")
    p.add_argument("--project", default="runs/trace", help="Ultralytics project dir")
    p.add_argument("--name", default="vehicle_detector", help="run name")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    YOLO = load_yolo()
    ensure_weights_dir()

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        # Traffic-scene augmentation: mosaic + mild geometric/colour jitter.
        # NO vertical flip — vehicles/people are never upside down on a road.
        mosaic=1.0,
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    dest = publish_best(results.save_dir, "vehicle_detector.pt")
    print("\n" + "=" * 70)
    if dest:
        print(f"Best weights copied to: {rel(dest)}")
        print("\nNext steps:")
        print("  1. Point the runtime at the fine-tuned detector in")
        print("     config/default.yaml:")
        print(f"         models:\n           detector: {rel(dest)}")
        print("  2. Sanity-check it:")
        print("         python -m trace_cv.cli detect path/to/traffic.jpg")
    else:
        print(f"Training finished but best.pt was not found under {results.save_dir}.")
    print("=" * 70)
    return 0 if dest else 1


if __name__ == "__main__":
    raise SystemExit(main())
