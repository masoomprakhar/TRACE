#!/usr/bin/env python3
"""Build binary seatbelt classification dataset from car windshield crops.

Sources (both via Roboflow, CCTV/traffic-surveillance angle):
  - seatbelttraining/seatbelt-detection-lb1ec  v3  (~3489 images)
  - dataset-9xayt/seatbelt-0lhjh               v1  (~3025 images)

Output:
  data/datasets/seatbelt_cls/
    images/{split}/*.jpg
    labels.csv  (path, no_seatbelt, split)

Usage:
  export ROBOFLOW_API_KEY=...
  python scripts/build_seatbelt_dataset.py
  python scripts/build_seatbelt_dataset.py --skip-ds1 --skip-ds2   # process existing
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "datasets" / "seatbelt_cls"
IMAGES = OUT / "images"

# ── class-name normalisation ─────────────────────────────────────────────────
SEATBELT_OK = (
    "seatbelt", "with_seatbelt", "with-seatbelt", "belted",
    "seat_belt", "seat-belt", "belt_on", "person-seatbelt",
    "person_seatbelt", "person-with-seatbelt",
)
SEATBELT_NO = (
    "no_seatbelt", "no-seatbelt", "noseatbelt", "without_seatbelt",
    "without-seatbelt", "unbelted", "no_seat_belt", "no-seat-belt",
    "person-noseatbelt", "person_noseatbelt", "person-without-seatbelt",
    "no seat-belt detected", "no seat belt",
)
PERSON_KEYS = ("person", "driver", "occupant", "rider")
WINDOW_KEYS = ("windshield", "window", "car_window", "car-window",
               "front_window", "front-window")


def _norm(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")


def _is_seatbelt_ok(name: str) -> bool:
    n = _norm(name)
    if any(_norm(x) in n for x in SEATBELT_NO):
        return False
    return any(_norm(x) in n for x in SEATBELT_OK)


def _is_seatbelt_no(name: str) -> bool:
    n = _norm(name)
    return any(_norm(x) in n for x in SEATBELT_NO) or (
        ("no" in n or "without" in n or "un" in n) and "belt" in n
    )


def _is_person(name: str) -> bool:
    n = _norm(name)
    return any(k in n for k in PERSON_KEYS)


def _is_window(name: str) -> bool:
    n = _norm(name)
    return any(k in n for k in WINDOW_KEYS)


# ── YOLO helpers ──────────────────────────────────────────────────────────────

def yolo_to_xyxy(parts: list[str], w: int, h: int) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = map(float, parts[1:5])
    return (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h


def bbox_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua else 0.0


def load_class_names(yaml_path: Path) -> dict[int, str]:
    if not yaml_path.exists():
        return {}
    raw = yaml.safe_load(yaml_path.read_text()) or {}
    names = raw.get("names") or {}
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def parse_label_file(lbl: Path, w: int, h: int, id_to_name: dict[int, str]) -> list[dict]:
    boxes: list[dict] = []
    if not lbl.exists():
        return boxes
    for line in lbl.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        boxes.append({
            "name": id_to_name.get(cls_id, f"class_{cls_id}"),
            "bbox": yolo_to_xyxy(parts, w, h),
        })
    return boxes


def safe_crop(img: np.ndarray, bbox: tuple) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    H, W = img.shape[:2]
    cx1, cy1 = max(0, int(x1)), max(0, int(y1))
    cx2, cy2 = min(W, int(x2)), min(H, int(y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = img[cy1:cy2, cx1:cx2]
    return crop if crop.size > 0 else None


# ── core processing ───────────────────────────────────────────────────────────

def process_split(
    dataset_dir: Path,
    split: str,
    id_to_name: dict[int, str],
    rows: list[dict],
    prefix: str,
) -> int:
    img_root = dataset_dir / split / "images"
    lbl_root = dataset_dir / split / "labels"
    if not img_root.exists():
        img_root = dataset_dir / "images"
        lbl_root = dataset_dir / "labels"
    if not img_root.exists():
        return 0

    names_list    = list(id_to_name.values())
    has_sb_cls    = any(_is_seatbelt_ok(n) or _is_seatbelt_no(n) for n in names_list)
    has_person    = any(_is_person(n) for n in names_list)
    has_window    = any(_is_window(n) for n in names_list)

    print(
        f"[{prefix}/{split}] classes={names_list} "
        f"has_sb={has_sb_cls} has_person={has_person} has_window={has_window}"
    )

    out_split = IMAGES / split
    out_split.mkdir(parents=True, exist_ok=True)
    n = 0

    for img_path in sorted(img_root.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = parse_label_file(lbl_root / f"{img_path.stem}.txt", W, H, id_to_name)
        if not boxes:
            continue

        if has_sb_cls:
            # Each seatbelt/no-seatbelt box is the crop — mirrors the CCTV
            # windshield-crop approach: the annotation IS the region of interest
            sb_boxes = [b for b in boxes if _is_seatbelt_ok(b["name"]) or _is_seatbelt_no(b["name"])]
            if not sb_boxes:
                continue
            for i, sb in enumerate(sb_boxes):
                crop = safe_crop(img, sb["bbox"])
                if crop is None:
                    continue
                no_sb = 1 if _is_seatbelt_no(sb["name"]) else 0
                fname = f"{prefix}_{split}_{img_path.stem}_sb{i}.jpg"
                dest  = out_split / fname
                cv2.imwrite(str(dest), crop)
                rows.append({"path": str(dest.relative_to(OUT)), "no_seatbelt": no_sb, "split": split})
                n += 1

        elif has_window:
            # Windshield region crop; label via overlap with seatbelt annotations
            window_boxes = [b for b in boxes if _is_window(b["name"])]
            sb_no_boxes  = [b for b in boxes if _is_seatbelt_no(b["name"])]
            sb_ok_boxes  = [b for b in boxes if _is_seatbelt_ok(b["name"])]
            for i, wb in enumerate(window_boxes):
                crop = safe_crop(img, wb["bbox"])
                if crop is None:
                    continue
                has_no = any(bbox_iou(wb["bbox"], b["bbox"]) >= 0.1 for b in sb_no_boxes)
                has_ok = any(bbox_iou(wb["bbox"], b["bbox"]) >= 0.1 for b in sb_ok_boxes)
                if not has_no and not has_ok:
                    continue
                fname = f"{prefix}_{split}_{img_path.stem}_w{i}.jpg"
                dest  = out_split / fname
                cv2.imwrite(str(dest), crop)
                rows.append({"path": str(dest.relative_to(OUT)), "no_seatbelt": 1 if has_no else 0, "split": split})
                n += 1

        elif has_person:
            # Person box crop; label via overlap with seatbelt annotations
            person_boxes = [b for b in boxes if _is_person(b["name"])]
            sb_no_boxes  = [b for b in boxes if _is_seatbelt_no(b["name"])]
            sb_ok_boxes  = [b for b in boxes if _is_seatbelt_ok(b["name"])]
            for i, pb in enumerate(person_boxes):
                crop = safe_crop(img, pb["bbox"])
                if crop is None:
                    continue
                has_no = any(bbox_iou(pb["bbox"], b["bbox"]) >= 0.15 for b in sb_no_boxes)
                has_ok = any(bbox_iou(pb["bbox"], b["bbox"]) >= 0.15 for b in sb_ok_boxes)
                if not has_no and not has_ok:
                    continue
                fname = f"{prefix}_{split}_{img_path.stem}_p{i}.jpg"
                dest  = out_split / fname
                cv2.imwrite(str(dest), crop)
                rows.append({"path": str(dest.relative_to(OUT)), "no_seatbelt": 1 if has_no else 0, "split": split})
                n += 1

        else:
            # Fallback: whole image; label from class names present in frame
            sb_no = any(_is_seatbelt_no(b["name"]) for b in boxes)
            sb_ok = any(_is_seatbelt_ok(b["name"]) for b in boxes)
            if not sb_no and not sb_ok:
                continue
            fname = f"{prefix}_{split}_{img_path.stem}_full.jpg"
            dest  = out_split / fname
            cv2.imwrite(str(dest), img)
            rows.append({"path": str(dest.relative_to(OUT)), "no_seatbelt": 1 if sb_no else 0, "split": split})
            n += 1

    print(f"[{prefix}/{split}] -> {n} crops")
    return n


# ── downloads ─────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def download_ds1(raw_dir: Path) -> Path | None:
    """seatbelttraining-7yh0f / seatbelt-detection-lb1ec v3 — ~3489 traffic images."""
    key = _get_api_key()
    if not key:
        print("skip DS1: ROBOFLOW_API_KEY not set", file=sys.stderr)
        return None
    try:
        from roboflow import Roboflow
        rf   = Roboflow(api_key=key)
        proj = rf.workspace("seatbelttraining-7yh0f").project("seatbelt-detection-lb1ec")
        ver  = proj.version(3)
        ver.export("yolov8")
        ds   = ver.download("yolov8", location=str(raw_dir), overwrite=True)
        return Path(ds.location) if hasattr(ds, "location") else raw_dir
    except Exception as exc:
        print(f"DS1 download failed: {exc}", file=sys.stderr)
        return None


def download_ds2(raw_dir: Path) -> Path | None:
    """dataset-9xayt / seatbelt-0lhjh v1 — ~3025 traffic images."""
    key = _get_api_key()
    if not key:
        print("skip DS2: ROBOFLOW_API_KEY not set", file=sys.stderr)
        return None
    try:
        from roboflow import Roboflow
        rf   = Roboflow(api_key=key)
        proj = rf.workspace("dataset-9xayt").project("seatbelt-0lhjh")
        ver  = proj.version(1)
        ver.export("yolov8")
        ds   = ver.download("yolov8", location=str(raw_dir), overwrite=True)
        return Path(ds.location) if hasattr(ds, "location") else raw_dir
    except Exception as exc:
        print(f"DS2 download failed: {exc}", file=sys.stderr)
        return None


def build_synthetic(out_root: Path, rows: list[dict], n_per_split: int = 80) -> int:
    rng = np.random.default_rng(42)
    n   = 0
    for split in ("train", "valid", "test"):
        (IMAGES / split).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_split):
            img   = rng.integers(40, 200, (224, 224, 3), dtype=np.uint8)
            no_sb = int(rng.random() < 0.4)
            fname = f"syn_{split}_{i:04d}.jpg"
            dest  = IMAGES / split / fname
            cv2.imwrite(str(dest), img)
            rows.append({"path": str(dest.relative_to(out_root)), "no_seatbelt": no_sb, "split": split})
            n += 1
    return n


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Build seatbelt classification dataset")
    p.add_argument("--out",       default="data/datasets/seatbelt_cls")
    p.add_argument("--skip-ds1",  action="store_true", help="Skip seatbelttraining DS1 download")
    p.add_argument("--skip-ds2",  action="store_true", help="Skip dataset-9xayt DS2 download")
    p.add_argument("--synthetic", action="store_true", help="Add synthetic crops if downloads fail")
    args = p.parse_args()

    global OUT, IMAGES
    OUT    = Path(args.out)
    IMAGES = OUT / "images"
    if OUT.exists():
        shutil.rmtree(OUT)
    IMAGES.mkdir(parents=True)

    rows:  list[dict] = []
    total: int        = 0

    # DS1
    if not args.skip_ds1:
        raw1   = ROOT / "data" / "raw" / "seatbelt_ds1"
        ds1dir = download_ds1(raw1)
    else:
        raw1   = ROOT / "data" / "raw" / "seatbelt_ds1"
        ds1dir = raw1 if raw1.exists() else None

    if ds1dir and ds1dir.exists():
        names = load_class_names(ds1dir / "data.yaml")
        for split in ("train", "valid", "test"):
            total += process_split(ds1dir, split, names, rows, "sb1")

    # DS2
    if not args.skip_ds2:
        raw2   = ROOT / "data" / "raw" / "seatbelt_ds2"
        ds2dir = download_ds2(raw2)
    else:
        raw2   = ROOT / "data" / "raw" / "seatbelt_ds2"
        ds2dir = raw2 if raw2.exists() else None

    if ds2dir and ds2dir.exists():
        names = load_class_names(ds2dir / "data.yaml")
        for split in ("train", "valid", "test"):
            total += process_split(ds2dir, split, names, rows, "sb2")

    if not rows and args.synthetic:
        total += build_synthetic(OUT, rows)

    if not rows:
        print(
            "No samples built — set ROBOFLOW_API_KEY, use --synthetic, or check dataset dirs.",
            file=sys.stderr,
        )
        return 1

    csv_path = OUT / "labels.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "no_seatbelt", "split"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nBuilt {total} crops -> {OUT}")
    print(f"  no_seatbelt positives : {sum(r['no_seatbelt'] for r in rows)}")
    print(f"  seatbelt    positives : {sum(1 - r['no_seatbelt'] for r in rows)}")
    print(f"  labels                : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
