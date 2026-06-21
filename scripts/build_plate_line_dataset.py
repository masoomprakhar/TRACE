#!/usr/bin/env python3
"""Build line-level Indian plate OCR dataset for TrOCR fine-tuning.

Sources (priority):
  1. Roboflow plate-ocr char boxes (prakhar-parkar/plate-ocr-3clbn-b2ojn)
  2. Optional Kaggle indian-license-plates-with-labels (line text)
  3. Rendered synthetic plates (fallback / augmentation)

Output:
  data/ocr/lines/{train,val,test}/*.jpg
  data/ocr/lines/manifest.json  — [{image, plate_text, split}]

Usage:
  export ROBOFLOW_API_KEY=...
  python scripts/build_plate_line_dataset.py
  python scripts/build_plate_line_dataset.py --synthetic-only --n-synthetic 400
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ocr" / "lines"
MANIFEST = OUT / "manifest.json"

PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")
CHAR_RE = re.compile(r"^[A-Z0-9]$")


def _api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def load_names(yaml_path: Path) -> dict[int, str]:
    if not yaml_path.exists():
        return {}
    raw = yaml.safe_load(yaml_path.read_text()) or {}
    names = raw.get("names") or {}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def yolo_char_boxes(label_path: Path, w: int, h: int, id_to_name: dict[int, str]) -> list[dict]:
    chars: list[dict] = []
    if not label_path.exists():
        return chars
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        ch = id_to_name.get(cls_id, "").strip().upper()
        if not CHAR_RE.match(ch):
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        chars.append({"ch": ch, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx * w})
    chars.sort(key=lambda c: c["cx"])
    return chars


def assemble_text(chars: list[dict]) -> str:
    return "".join(c["ch"] for c in chars)


def plate_line_crop(img: np.ndarray, chars: list[dict], pad: int = 8) -> np.ndarray | None:
    if not chars:
        return None
    h, w = img.shape[:2]
    x1 = max(0, min(c["x1"] for c in chars) - pad)
    y1 = max(0, min(c["y1"] for c in chars) - pad)
    x2 = min(w, max(c["x2"] for c in chars) + pad)
    y2 = min(h, max(c["y2"] for c in chars) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def download_plate_ocr(raw_dir: Path) -> Path | None:
    key = _api_key()
    if not key:
        print("skip plate-ocr Roboflow: ROBOFLOW_API_KEY not set", file=sys.stderr)
        return None
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=key)
        proj = rf.workspace("prakhar-parkar").project("plate-ocr-3clbn-b2ojn")
        ver = proj.version(1)
        ver.export("yolov8")
        ds = ver.download("yolov8", location=str(raw_dir), overwrite=True)
        return Path(ds.location) if hasattr(ds, "location") else raw_dir
    except Exception as exc:
        print(f"plate-ocr download failed: {exc}", file=sys.stderr)
        return None


def process_yolo_dataset(ds_dir: Path, split_map: dict[str, str], entries: list[dict]) -> int:
    names = load_names(ds_dir / "data.yaml")
    n = 0
    for src_split, dst_split in split_map.items():
        img_dir = ds_dir / src_split / "images"
        lbl_dir = ds_dir / src_split / "labels"
        if not img_dir.exists():
            continue
        out_dir = OUT / dst_split
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            chars = yolo_char_boxes(lbl_dir / f"{img_path.stem}.txt", w, h, names)
            text = assemble_text(chars)
            if len(text) < 6:
                continue
            crop = plate_line_crop(img, chars)
            if crop is None or crop.size == 0:
                continue
            name = f"rf_{dst_split}_{img_path.stem}.jpg"
            dest = out_dir / name
            cv2.imwrite(str(dest), crop)
            entries.append({"image": str(dest.relative_to(OUT)), "plate_text": text, "split": dst_split})
            n += 1
    return n


def render_plate(text: str, w: int = 320, h: int = 96, rng: random.Random | None = None) -> np.ndarray:
    rng = rng or random.Random()
    bg = (rng.randint(235, 252), rng.randint(235, 252), rng.randint(230, 245))
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.rectangle(img, (4, 4), (w - 4, h - 4), (12, 12, 12), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0 + rng.uniform(-0.1, 0.15)
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = max(8, (w - tw) // 2)
    y = (h + th) // 2
    cv2.putText(img, text, (x, y), font, scale, (8, 8, 8), thick, cv2.LINE_AA)
    if rng.random() < 0.3:
        noise = np.random.randint(-18, 18, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _format_plate(raw: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    return s if PLATE_RE.match(s) else ""


def build_synthetic(n_train: int, n_val: int, n_test: int, entries: list[dict]) -> int:
    states = ["MH", "KA", "DL", "TN", "GJ", "UP", "RJ", "WB"]
    rng = random.Random(42)
    n = 0
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        out_dir = OUT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            st = rng.choice(states)
            dist = rng.randint(1, 99)
            series = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(rng.randint(1, 3)))
            num = rng.randint(1000, 9999)
            text = _format_plate(f"{st}{dist:02d}{series}{num}")
            if not text:
                continue
            disp = f"{st} {dist:02d} {series} {num}"
            crop = render_plate(disp, rng=rng)
            name = f"syn_{split}_{i:04d}.jpg"
            dest = out_dir / name
            cv2.imwrite(str(dest), crop)
            entries.append({"image": str(dest.relative_to(OUT)), "plate_text": text, "split": split})
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description="Build plate line OCR dataset for TrOCR")
    p.add_argument("--out", default="data/ocr/lines")
    p.add_argument("--skip-roboflow", action="store_true")
    p.add_argument("--synthetic-only", action="store_true")
    p.add_argument("--n-synthetic", type=int, default=300)
    args = p.parse_args()

    global OUT, MANIFEST
    OUT = Path(args.out)
    MANIFEST = OUT / "manifest.json"
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    entries: list[dict] = []
    total = 0

    if not args.synthetic_only and not args.skip_roboflow:
        raw = ROOT / "data" / "raw" / "plate_ocr_rf"
        ds = download_plate_ocr(raw)
        if ds and ds.exists():
            total += process_yolo_dataset(
                ds,
                {"train": "train", "valid": "val", "test": "test"},
                entries,
            )

    if not entries or args.synthetic_only:
        n = args.n_synthetic
        total += build_synthetic(int(n * 0.7), int(n * 0.15), int(n * 0.15), entries)

    if not entries:
        print("No plate line samples built.", file=sys.stderr)
        return 1

    MANIFEST.write_text(json.dumps({"samples": entries}, indent=2))
    print(f"Built {total} line crops -> {OUT}")
    print(f"  manifest: {MANIFEST} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
