#!/usr/bin/env python3
"""Prepare Karan Panja seatbelt classification dataset (belt vs no_belt).

Source: https://universe.roboflow.com/karan-panja/seat-belt-detection-uhqwa
Classes mapped:
  person-seatbelt, seatbelt -> belt/
  person-noseatbelt        -> no_belt/
  Not Clear                -> excluded

Output (ImageFolder for train_seatbelt.py --mode cls):
  data/datasets/seatbelt_cls/{train,val}/{belt,no_belt}/*.jpg

Usage:
  export ROBOFLOW_API_KEY=...
  python viovision/scripts/prepare_seatbelt_karan_dataset.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "datasets" / "seatbelt_cls"
RAW = ROOT / "data" / "raw" / "seatbelt_karan"

WORKSPACE = "karan-panja"
PROJECT = "seat-belt-detection-uhqwa"
VERSION = 1

BELT_NAMES = {"person-seatbelt", "seatbelt", "person_seatbelt"}
NO_BELT_NAMES = {"person-noseatbelt", "person_noseatbelt", "no_seatbelt", "noseatbelt"}
SKIP_NAMES = {"not clear", "not_clear", "unclear", "unknown"}

MIN_CROP = 32


def _api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def load_names(yaml_path: Path) -> dict[int, str]:
    raw = yaml.safe_load(yaml_path.read_text()) or {}
    names = raw.get("names") or {}
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def map_class(name: str) -> str | None:
    n = name.strip().lower().replace("_", "-")
    if n in SKIP_NAMES or "not clear" in n:
        return None
    if name in BELT_NAMES or n in {x.lower() for x in BELT_NAMES}:
        return "belt"
    if name in NO_BELT_NAMES or n in {x.lower().replace("_", "-") for x in NO_BELT_NAMES}:
        return "no_belt"
    if "noseatbelt" in n or "no-seatbelt" in n or "no_seatbelt" in n:
        return "no_belt"
    if "seatbelt" in n and "no" not in n:
        return "belt"
    return None


def yolo_crop(parts: list[str], w: int, h: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = map(float, parts[1:5])
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def download_raw(dest: Path) -> Path | None:
    key = _api_key()
    if not key:
        print("ROBOFLOW_API_KEY not set", file=sys.stderr)
        return None
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=key)
        proj = rf.workspace(WORKSPACE).project(PROJECT)
        ver = proj.version(VERSION)
        ver.export("yolov8")
        ds = ver.download("yolov8", location=str(dest), overwrite=True)
        return Path(ds.location) if hasattr(ds, "location") else dest
    except Exception as exc:
        print(f"Karan Panja download failed: {exc}", file=sys.stderr)
        return None


def process_split(ds_dir: Path, split: str, dst_split: str, counters: dict[str, int]) -> None:
    img_dir = ds_dir / split / "images"
    lbl_dir = ds_dir / split / "labels"
    if not img_dir.exists():
        return
    id_to_name = load_names(ds_dir / "data.yaml")
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = id_to_name.get(int(parts[0]), "")
            target = map_class(cls)
            if target is None:
                continue
            x1, y1, x2, y2 = yolo_crop(parts, w, h)
            if (x2 - x1) < MIN_CROP or (y2 - y1) < MIN_CROP:
                continue
            crop = img[y1:y2, x1:x2]
            out_dir = OUT / dst_split / target
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = counters.get(target, 0)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_{idx:04d}.jpg"), crop)
            counters[target] = idx + 1


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare Karan Panja seatbelt cls dataset")
    p.add_argument("--out", default="data/datasets/seatbelt_cls")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    global OUT
    OUT = Path(args.out)
    if OUT.exists():
        shutil.rmtree(OUT)

    ds_dir = RAW
    if not args.skip_download:
        ds = download_raw(RAW)
        if ds:
            ds_dir = ds

    if not (ds_dir / "data.yaml").exists():
        print("Dataset not found. Set ROBOFLOW_API_KEY and retry.", file=sys.stderr)
        return 1

    counters: dict[str, int] = {}
    split_map = {"train": "train", "valid": "val", "test": "val"}
    for src, dst in split_map.items():
        process_split(ds_dir, src, dst, counters)

    belt = counters.get("belt", 0)
    nob = counters.get("no_belt", 0)
    print(f"Seatbelt cls dataset -> {OUT}")
    print(f"  belt: {belt}")
    print(f"  no_belt: {nob}")
    if belt == 0 and nob == 0:
        print("No crops — check class names in data.yaml", file=sys.stderr)
        return 1
    print("\nTrain:")
    print(f"  python training/train_seatbelt.py --mode cls --data {OUT} --epochs 30 --device cpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
