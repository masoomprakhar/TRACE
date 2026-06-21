#!/usr/bin/env python3
"""Train the optional TRACE helmet model (classifier or detector).

The runtime interpreter (trace_cv/violation/helmet.py, HelmetModel) accepts
either a classification or a detection checkpoint and keys off class *names*:
a name containing helmet -> HELMET; a name containing any of
no_helmet/nohelmet/without/no-helmet/head/bare -> NO HELMET. So the only
hard requirement is that your classes are named `helmet` and `no_helmet`.

Two modes:
  cls (default)  Ultralytics YOLO classification (yolov8n-cls.pt) on an
                 ImageFolder-style dataset:
                     <data>/train/helmet/*.jpg
                     <data>/train/no_helmet/*.jpg
                     <data>/val/helmet/*.jpg   (val/ recommended)
                     <data>/val/no_helmet/*.jpg
  det            object detection with classes helmet/no_helmet; --data is an
                 Ultralytics dataset YAML (names: [helmet, no_helmet]).

Source data: Roboflow Universe helmet datasets exported in the matching
format (Folder for cls, YOLOv8 for det). See training/README.md. Augment
rare cases (night, triple-riding) with training/augment.py first.

The best checkpoint is copied to models/weights/helmet.pt.

Example (classification):
  python training/train_helmet.py --mode cls \\
      --data data/datasets/helmet_cls --epochs 30 --imgsz 224 --device 0

Example (detection):
  python training/train_helmet.py --mode det \\
      --data training/datasets/helmet.yaml --epochs 50 --imgsz 640

Requires the ML extras:  pip install -r requirements-ml.txt
"""

from __future__ import annotations

import argparse

from _common import ensure_weights_dir, load_yolo, publish_best, rel


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the TRACE helmet model (cls or det).",
    )
    p.add_argument("--mode", choices=["cls", "det"], default="cls")
    p.add_argument(
        "--data",
        required=True,
        help="cls: dataset folder (helmet/, no_helmet/); det: dataset YAML",
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
    p.add_argument("--name", default="helmet")
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
        # Classification: light flips only; classes are helmet / no_helmet.
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

    dest = publish_best(results.save_dir, "helmet.pt")
    print("\n" + "=" * 70)
    if dest:
        print(f"Best weights copied to: {rel(dest)}")
        print("\nWire it into config/default.yaml:")
        print(f"    models.helmet: {rel(dest)}")
        print("\nThe HelmetDetector module activates automatically once set.")
    else:
        print(f"Training finished but best.pt was not found under {results.save_dir}.")
    print("=" * 70)
    return 0 if dest else 1


if __name__ == "__main__":
    raise SystemExit(main())
