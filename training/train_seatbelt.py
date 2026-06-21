#!/usr/bin/env python3
"""Train the optional TRACE seatbelt model (classifier or detector).

The runtime interpreter (trace_cv/violation/seatbelt.py, SeatbeltModel) keys
off class *names*: a name containing belt -> BELT; any of
no_seatbelt/noseatbelt/without/no-belt/unbelted/no_belt -> NO BELT; any of
occluded/unknown/unclear -> OCCLUDED. The 'occluded' class is what prevents
false positives from window glare / A-pillars — TRACE never flags an
occluded driver — so train all THREE classes: belt, no_belt, occluded.

Two modes:
  cls (default)  Ultralytics YOLO classification (yolov8n-cls.pt) on an
                 ImageFolder-style dataset:
                     <data>/train/belt/*.jpg
                     <data>/train/no_belt/*.jpg
                     <data>/train/occluded/*.jpg
                     <data>/val/...   (val/ recommended)
  det            object detection with classes belt/no_belt/occluded; --data
                 is an Ultralytics dataset YAML (names: [belt, no_belt, occluded]).

Source data: Roboflow Universe seatbelt datasets exported in the matching
format. See training/README.md. Crop to the driver window before training so
inputs match the runtime ROI (trace_cv/detection/roi.py: driver_roi).

The best checkpoint is copied to models/weights/seatbelt.pt.

Example (classification):
  python training/train_seatbelt.py --mode cls \\
      --data data/datasets/seatbelt_cls --epochs 30 --imgsz 224 --device 0

Example (detection):
  python training/train_seatbelt.py --mode det \\
      --data training/datasets/seatbelt.yaml --epochs 50 --imgsz 640

Requires the ML extras:  pip install -r requirements-ml.txt
"""

from __future__ import annotations

import argparse

from _common import ensure_weights_dir, load_yolo, publish_best, rel


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the TRACE seatbelt model (3-class: belt/no_belt/occluded).",
    )
    p.add_argument("--mode", choices=["cls", "det"], default="cls")
    p.add_argument(
        "--data",
        required=True,
        help="cls: dataset folder (belt/, no_belt/, occluded/); det: dataset YAML",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="base weights (default: yolov8n-cls.pt for cls, yolov8n.pt for det)",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="image size (default: 224 for cls, 640 for det)",
    )
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="", help="'' (auto), 'cpu', '0', '0,1'")
    p.add_argument("--project", default="runs/trace")
    p.add_argument("--name", default="seatbelt")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    YOLO = load_yolo()
    ensure_weights_dir()

    weights = args.weights or (
        "yolov8n-cls.pt" if args.mode == "cls" else "yolov8n.pt"
    )
    imgsz = args.imgsz or (224 if args.mode == "cls" else 640)

    model = YOLO(weights)
    if args.mode == "cls":
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            fliplr=0.5,
            flipud=0.0,
        )
    else:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            mosaic=1.0,
            fliplr=0.5,
            flipud=0.0,
            degrees=5.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
        )

    dest = publish_best(results.save_dir, "seatbelt.pt")
    print("\n" + "=" * 70)
    if dest:
        print(f"Best weights copied to: {rel(dest)}")
        print("\nWire it into config/default.yaml:")
        print(f"    models.seatbelt: {rel(dest)}")
        print("\nThe SeatbeltDetector module activates automatically once set.")
    else:
        print(f"Training finished but best.pt was not found under {results.save_dir}.")
    print("=" * 70)
    return 0 if dest else 1


if __name__ == "__main__":
    raise SystemExit(main())
