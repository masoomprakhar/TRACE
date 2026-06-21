#!/usr/bin/env python3
"""Build multi-label rider dataset for CCTV helmet + triple-riding CNN.

Sources:
  - Khadatkar & Wasule (Roboflow): helmet-and-no-helmet-rider-detection
  - HF vivekvar/cctv-datasets merged_v3 YOLO export

Output:
  data/datasets/rider_multilabel/
    images/{split}/*.jpg
    labels.csv  (path, no_helmet, triple_riding, split)

Usage:
  export ROBOFLOW_API_KEY=...
  python scripts/build_rider_multilabel_dataset.py
  python scripts/build_rider_multilabel_dataset.py --skip-khadatkar
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "datasets" / "rider_multilabel"
IMAGES = OUT / "images"

MOTO_KEYS = ("motorcycle", "bike", "two_wheeler", "scooter", "moped", "motor")
PERSON_KEYS = ("person", "rider", "driver", "pedestrian")
HELMET_OK = ("with helmet", "with_helmet", "helmet")
HELMET_NO = ("without helmet", "without_helmet", "no_helmet", "no helmet", "nohelmet")


def _norm(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")


def _is_moto(name: str) -> bool:
    n = _norm(name)
    return any(k in n for k in MOTO_KEYS)


def _is_person(name: str) -> bool:
    n = _norm(name)
    return any(k in n for k in PERSON_KEYS)


def _is_helmet_ok(name: str) -> bool:
    n = _norm(name)
    if any(x in n for x in HELMET_NO):
        return False
    return any(x in n for x in HELMET_OK) or n == "helmet"


def _is_helmet_no(name: str) -> bool:
    n = _norm(name)
    return any(x in n for x in HELMET_NO) or ("no" in n and "helmet" in n)


def yolo_to_xyxy(parts: list[str], w: int, h: int) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = map(float, parts[1:5])
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return x1, y1, x2, y2


def bbox_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def load_class_names(yaml_path: Path) -> dict[int, str]:
    if not yaml_path.exists():
        return {}
    raw = yaml.safe_load(yaml_path.read_text()) or {}
    names = raw.get("names") or {}
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def parse_yolo_label_file(
    label_path: Path, img_w: int, img_h: int, id_to_name: dict[int, str]
) -> list[dict]:
    boxes: list[dict] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        name = id_to_name.get(cls_id, f"class_{cls_id}")
        xyxy = yolo_to_xyxy(parts, img_w, img_h)
        boxes.append({"name": name, "bbox": xyxy})
    return boxes


def rider_fraction_crop(img, bbox: tuple[float, float, float, float], frac: float = 0.70):
    x1, y1, x2, y2 = bbox
    y2_crop = y1 + (y2 - y1) * frac
    h, w = img.shape[:2]
    cx1, cy1 = max(0, int(x1)), max(0, int(y1))
    cx2, cy2 = min(w, int(x2)), min(h, int(y2_crop))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return img[cy1:cy2, cx1:cx2]


def labels_for_moto(
    moto_bbox: tuple[float, float, float, float],
    all_boxes: list[dict],
    *,
    triple_min: int = 3,
    overlap: float = 0.2,
) -> tuple[int, int]:
    from trace_cv.violation.base import expand_bbox

    zone = expand_bbox(moto_bbox)
    riders = [
        b
        for b in all_boxes
        if _is_person(b["name"]) and bbox_iou(b["bbox"], zone) >= overlap
    ]
    triple = 1 if len(riders) >= triple_min else 0

    no_helmet = 0
    for b in all_boxes:
        if _is_helmet_no(b["name"]) and bbox_iou(b["bbox"], moto_bbox) >= overlap:
            no_helmet = 1
            break
    if not no_helmet:
        helmet_ok = any(
            _is_helmet_ok(b["name"]) and bbox_iou(b["bbox"], moto_bbox) >= overlap
            for b in all_boxes
        )
        if not helmet_ok and riders:
            for b in all_boxes:
                if _is_helmet_no(b["name"]):
                    no_helmet = 1
                    break

    return no_helmet, triple


def process_yolo_split(
    dataset_dir: Path,
    split: str,
    id_to_name: dict[int, str],
    rows: list[dict],
    prefix: str,
    *,
    triple_min: int,
) -> int:
    img_root = dataset_dir / split / "images"
    lbl_root = dataset_dir / split / "labels"
    if not img_root.exists():
        img_root = dataset_dir / "images"
        lbl_root = dataset_dir / "labels"
    if not img_root.exists():
        return 0

    out_split = IMAGES / split
    out_split.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in sorted(img_root.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl = lbl_root / f"{img_path.stem}.txt"
        boxes = parse_yolo_label_file(lbl, w, h, id_to_name)
        motos = [b for b in boxes if _is_moto(b["name"])]
        if not motos:
            continue
        for i, moto in enumerate(motos):
            crop = rider_fraction_crop(img, moto["bbox"])
            if crop is None or crop.size == 0:
                continue
            no_h, triple = labels_for_moto(
                moto["bbox"], boxes, triple_min=triple_min
            )
            name = f"{prefix}_{split}_{img_path.stem}_m{i}.jpg"
            dest = out_split / name
            cv2.imwrite(str(dest), crop)
            rows.append(
                {
                    "path": str(dest.relative_to(OUT)),
                    "no_helmet": no_h,
                    "triple_riding": triple,
                    "split": split,
                }
            )
            n += 1
    return n


def download_khadatkar(raw_dir: Path) -> Path | None:
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        print("skip Khadatkar: ROBOFLOW_API_KEY not set", file=sys.stderr)
        return None
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=key)
        proj = rf.workspace("gw-khadatkar-and-sv-wasule").project(
            "helmet-and-no-helmet-rider-detection"
        )
        ver = proj.version(5)
        ver.export("yolov8")
        ds = ver.download("yolov8", location=str(raw_dir), overwrite=True)
        return Path(ds.location) if hasattr(ds, "location") else raw_dir
    except Exception as exc:
        print(f"Khadatkar download failed: {exc}", file=sys.stderr)
        return None


def download_hf_cctv(dest: Path) -> bool:
    if (dest / "merged_v3" / "data.yaml").exists():
        return True
    try:
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                "vivekvar/cctv-datasets",
                "--local-dir",
                str(dest),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except Exception as exc:
        print(f"HF cctv download failed: {exc}", file=sys.stderr)
        return False


def build_synthetic(out_root: Path, rows: list[dict], n_per_split: int = 80) -> int:
    """Minimal synthetic motorcycle-rider crops for pipeline smoke tests."""
    rng = np.random.default_rng(42)
    images = out_root / "images"
    n = 0
    for split in ("train", "valid", "test"):
        out_split = images / split
        out_split.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_split):
            img = rng.integers(40, 200, (180, 140, 3), dtype=np.uint8)
            no_h = int(rng.random() < 0.35)
            triple = int(rng.random() < 0.15)
            name = f"syn_{split}_{i:04d}.jpg"
            dest = out_split / name
            cv2.imwrite(str(dest), img)
            rows.append(
                {
                    "path": str(dest.relative_to(out_root)),
                    "no_helmet": no_h,
                    "triple_riding": triple,
                    "split": split,
                }
            )
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description="Build rider multi-label CCTV dataset")
    p.add_argument("--out", default="data/datasets/rider_multilabel")
    p.add_argument("--skip-khadatkar", action="store_true")
    p.add_argument("--skip-hf", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="Add synthetic crops if downloads fail")
    p.add_argument("--triple-min", type=int, default=3)
    args = p.parse_args()

    global OUT, IMAGES
    OUT = Path(args.out)
    IMAGES = OUT / "images"
    if OUT.exists():
        shutil.rmtree(OUT)
    IMAGES.mkdir(parents=True)

    rows: list[dict] = []
    total = 0

    if not args.skip_khadatkar:
        raw = ROOT / "data" / "raw" / "khadatkar_helmet"
        ds_dir = download_khadatkar(raw)
        if ds_dir and ds_dir.exists():
            names = load_class_names(ds_dir / "data.yaml")
            for split in ("train", "valid", "test"):
                total += process_yolo_split(
                    ds_dir, split, names, rows, "kh", triple_min=args.triple_min
                )

    if not args.skip_hf:
        hf_root = ROOT / "data" / "datasets" / "cctv_hf"
        if download_hf_cctv(hf_root):
            merged = hf_root / "merged_v3"
            if merged.exists():
                names = load_class_names(merged / "data.yaml")
                for split in ("train", "valid", "test"):
                    total += process_yolo_split(
                        merged, split, names, rows, "cctv", triple_min=args.triple_min
                    )

    if not rows and args.synthetic:
        total += build_synthetic(OUT, rows)

    if not rows:
        print("No samples built — download datasets, use --synthetic, or check API key.", file=sys.stderr)
        return 1

    csv_path = OUT / "labels.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "no_helmet", "triple_riding", "split"])
        w.writeheader()
        w.writerows(rows)

    print(f"Built {total} crops -> {OUT}")
    print(f"  no_helmet positives: {sum(r['no_helmet'] for r in rows)}")
    print(f"  triple_riding positives: {sum(r['triple_riding'] for r in rows)}")
    print(f"  labels: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
