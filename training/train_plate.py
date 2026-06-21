#!/usr/bin/env python3
"""Train the optional TRACE license-plate detector.

The runtime uses this as a plain localizer: trace_cv/detection/detector.py
builds a Detector with keep=set() for the plate model, so it accepts ANY
class — it only needs to draw a tight box around the plate, which is then
cropped and passed to EasyOCR + the Indian-plate corrector. Train a single
`license_plate` class (detection only; classification makes no sense here).

Source data: Roboflow Universe license-plate datasets exported in YOLOv8
format. Copy training/datasets/plate.example.yaml, edit the paths, and keep
names: [license_plate]. See training/README.md.

The best checkpoint is copied to models/weights/plate.pt.

Example:
  python training/train_plate.py \\
      --data training/datasets/plate.example.yaml \\
      --weights yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device 0

Requires the ML extras:  pip install -r requirements-ml.txt
"""

from __future__ import annotations

import argparse

from _common import ensure_weights_dir, load_yolo, publish_best, rel


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a single-class (license_plate) YOLO detector for TRACE.",
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
    p.add_argument("--project", default="runs/trace")
    p.add_argument("--name", default="plate")
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
        # Plates are small + rectangular: mosaic helps, keep geometry gentle
        # and never flip vertically (text orientation matters downstream).
        mosaic=1.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=3.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    dest = publish_best(results.save_dir, "plate.pt")
    print("\n" + "=" * 70)
    if dest:
        print(f"Best weights copied to: {rel(dest)}")
        print("\nWire it into config/default.yaml:")
        print(f"    models.plate: {rel(dest)}")
        print("\nDetected plates are cropped and read by EasyOCR + the corrector.")
    else:
        print(f"Training finished but best.pt was not found under {results.save_dir}.")
    print("=" * 70)
    return 0 if dest else 1


if __name__ == "__main__":
    raise SystemExit(main())
