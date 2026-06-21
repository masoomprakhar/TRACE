#!/usr/bin/env python3
"""Generate a calibrated TRACE camera config from a reference frame.

Derives stop-line, lane divider, signal ROI, and no-parking polygons from
image dimensions using the junction layout ratios in trace_cv.demo. Override
any value manually after generation.

Usage:
  python scripts/calibrate_camera.py --image data/samples/junction-01-reference.jpg
  python scripts/calibrate_camera.py --width 1920 --height 1080 --out config/my-cam.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def geometry_for_frame(width: int, height: int, camera_name: str = "Camera-01") -> dict:
    """Map junction layout ratios (960×540 reference) to any resolution."""
    w, h = width, height
    return {
        "device": "cpu",
        "storage_dir": "data/output",
        "db_url": "sqlite:///data/trace.db",
        "models": {
            "detector": "viovision/runs/detect/runs/train/viovision_yolo11n/weights/best.pt",
            "helmet": "viovision/models/weights/helmet_svm.pkl",
            "seatbelt": "viovision/models/weights/seatbelt_svm.pkl",
            "plate": "viovision/runs/detect/runs/train/viovision_yolo11n/weights/best.pt",
            "ocr_langs": ["en"],
            "detector_imgsz": 640,
            "detector_backend": "viovision",
            "detector_class_map": {
                "two_wheeler": "motorcycle",
                "signal_light": "traffic_light",
                "license_plate": "license_plate",
                "windshield": "windshield",
            },
        },
        "thresholds": {
            "detection_conf": 0.35,
            "nms_iou": 0.45,
            "helmet_conf": 0.55,
            "seatbelt_conf": 0.65,
            "triple_riding_min": 3,
            "confirm_frames": 3,
            "parking_seconds": 30.0,
            "stationary_iou": 0.90,
            "rider_overlap": 0.20,
        },
        "scene": {
            "fps": 15.0,
            "stop_line": {"enabled": True, "y": int(h * 380 / 540)},
            "lane": {
                "enabled": True,
                "divider_x": w // 2,
                "correct_direction": "down",
            },
            "signal": {
                "enabled": True,
                "bbox": [
                    int(w * 700 / 960),
                    int(h * 24 / 540),
                    int(w * 790 / 960),
                    int(h * 108 / 540),
                ],
            },
            "no_parking_zones": [
                {
                    "name": "Left Curb Bay",
                    "polygon": [
                        [int(w * 40 / 960), int(h * 280 / 540)],
                        [int(w * 200 / 960), int(h * 280 / 540)],
                        [int(w * 200 / 960), int(h * 520 / 540)],
                        [int(w * 40 / 960), int(h * 520 / 540)],
                    ],
                }
            ],
        },
        "_meta": {
            "camera": camera_name,
            "reference_size": [width, height],
            "calibrated_by": "scripts/calibrate_camera.py",
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Calibrate TRACE camera scene geometry")
    p.add_argument("--image", help="Reference frame (reads width/height)")
    p.add_argument("--width", type=int, help="Frame width if no image given")
    p.add_argument("--height", type=int, help="Frame height if no image given")
    p.add_argument(
        "--out",
        default="config/camera-junction-01.yaml",
        help="Output YAML path (default: config/camera-junction-01.yaml)",
    )
    p.add_argument("--camera", default="Junction-01", help="Camera label for metadata")
    p.add_argument(
        "--write-reference",
        action="store_true",
        help="Also write a synthetic reference frame to data/samples/",
    )
    args = p.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"error: cannot read {args.image}", file=sys.stderr)
            return 1
        h, w = img.shape[:2]
    elif args.width and args.height:
        w, h = args.width, args.height
    else:
        from trace_cv.demo import make_synthetic_scene

        ref_path = ROOT / "data" / "samples" / "junction-01-reference.jpg"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        scene = make_synthetic_scene()
        cv2.imwrite(str(ref_path), scene)
        h, w = scene.shape[:2]
        print(f"wrote reference frame: {ref_path} ({w}×{h})")
        if not args.write_reference:
            args.write_reference = True

    cfg = geometry_for_frame(w, h, camera_name=args.camera)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    print(f"calibrated config written: {out}")
    print(f"  stop_line.y      = {cfg['scene']['stop_line']['y']}")
    print(f"  lane.divider_x   = {cfg['scene']['lane']['divider_x']}")
    print(f"  signal.bbox      = {cfg['scene']['signal']['bbox']}")
    print(f"  no_parking zones = {len(cfg['scene']['no_parking_zones'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
