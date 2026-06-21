#!/usr/bin/env python3
"""Build small training packs from local eval data for quick score boosts.

Sources (no large downloads required):
  - data/eval/manifest.json  -> YOLO detector mini-set + rider CNN crops
  - data/eval/dl_helmet/     -> extra rider CNN crops (helmet / no_helmet boxes)
  - data/datasets/seatbelt_cls/ (already prepared)

Outputs:
  data/datasets/quick_viovision/     YOLO (two_wheeler, license_plate, …)
  data/datasets/rider_multilabel/   labels.csv + images/
  data/ocr/lines/                   via build_plate_line_dataset --synthetic-only

Usage:
  python scripts/prepare_quick_train_pack.py
  python scripts/prepare_quick_train_pack.py --ocr-synthetic 200
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "eval" / "manifest.json"
OUT_YOLO = ROOT / "data" / "datasets" / "quick_viovision"
OUT_RIDER = ROOT / "data" / "datasets" / "rider_multilabel"
SEATBELT_MANIFEST = ROOT / "data" / "eval" / "seatbelt_manifest.json"
OUT_SEATBELT = ROOT / "data" / "datasets" / "seatbelt_cls"
DL_HELMET = ROOT / "data" / "eval" / "dl_helmet"

# VioVision traffic.yaml class indices
CLS_MAP = {
    "car": 0,
    "motorcycle": 1,
    "two_wheeler": 1,
    "person": 2,
    "license_plate": 3,
    "windshield": 4,
    "signal_light": 5,
    "bus": 0,
    "truck": 0,
}


def xyxy_to_yolo(x1, y1, x2, y2, w, h) -> tuple[float, float, float, float]:
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    return cx / w, cy / h, bw / w, bh / h


def prepare_yolo_from_dl_helmet(split: str, rng: random.Random, max_images: int = 80) -> list[tuple[str, list[str]]]:
    """Return (stem, yolo_lines) from dl_helmet for motorcycle-heavy augmentation."""
    names = {0: "helmet", 1: "no_helmet"}
    yaml_path = DL_HELMET / "data.yaml"
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        ns = raw.get("names") or names
        if isinstance(ns, list):
            names = {i: n for i, n in enumerate(ns)}

    out: list[tuple[Path, str, list[str]]] = []
    img_dir = DL_HELMET / split / "images"
    lbl_dir = DL_HELMET / split / "labels"
    if not img_dir.exists():
        return out
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng.shuffle(images)
    for img_path in images[:max_images]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        helmet_boxes: list[tuple[float, float, float, float]] = []
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                helmet_boxes.append((x1, y1, x2, y2))
        if helmet_boxes:
            x1 = min(b[0] for b in helmet_boxes)
            y1 = min(b[1] for b in helmet_boxes)
            x2 = max(b[2] for b in helmet_boxes)
            y2 = max(b[3] for b in helmet_boxes)
            pad_x, pad_y = (x2 - x1) * 0.35, (y2 - y1) * 0.8
            mx1, my1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            mx2, my2 = min(w, x2 + pad_x), min(h, y2 + pad_y * 0.2)
        else:
            margin = 0.05
            mx1, my1, mx2, my2 = w * margin, h * margin, w * (1 - margin), h * (1 - margin)
        lines = []
        cx, cy, bw, bh = xyxy_to_yolo(mx1, my1, mx2, my2, w, h)
        lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        for x1, y1, x2, y2 in helmet_boxes:
            pcx, pcy, pbw, pbh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            lines.append(f"2 {pcx:.6f} {pcy:.6f} {pbw:.6f} {pbh:.6f}")
        out.append((img_path, f"dl_{split}_{img_path.stem}", lines))
    return out


def prepare_yolo_from_manifest(samples: list[dict], val_ratio: float = 0.2) -> int:
    """Export manifest GT boxes to VioVision-style YOLO splits."""
    if OUT_YOLO.exists():
        shutil.rmtree(OUT_YOLO)
    for split in ("train", "valid"):
        (OUT_YOLO / "images" / split).mkdir(parents=True)
        (OUT_YOLO / "labels" / split).mkdir(parents=True)

    rng = random.Random(42)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_ids = {s["id"] for s in shuffled[:n_val]}

    n = 0
    for s in samples:
        img_path = ROOT / s["image"]
        if not img_path.exists():
            continue
        w, h = int(s.get("width", 0)), int(s.get("height", 0))
        if w <= 0 or h <= 0:
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]
        dets = s.get("detections_gt") or []
        if not dets:
            continue
        split = "valid" if s["id"] in val_ids else "train"
        stem = s["id"].replace("/", "_")[:80]
        dst_img = OUT_YOLO / "images" / split / f"{stem}.jpg"
        shutil.copy2(img_path, dst_img)
        lines = []
        for d in dets:
            cls = d.get("cls", "")
            cid = CLS_MAP.get(cls)
            if cid is None:
                continue
            bb = d.get("bbox") or []
            if len(bb) < 4:
                continue
            x1, y1, x2, y2 = map(float, bb[:4])
            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            continue
        lbl = OUT_YOLO / "labels" / split / f"{stem}.txt"
        lbl.write_text("\n".join(lines) + "\n")
        n += 1
        # Oversample motorcycle-heavy frames in train (plates dominate otherwise).
        if split == "train" and any(l.startswith("1 ") for l in lines):
            for rep in range(5):
                rep_stem = f"{stem}_m{rep}"
                shutil.copy2(dst_img, OUT_YOLO / "images" / split / f"{rep_stem}.jpg")
                (OUT_YOLO / "labels" / split / f"{rep_stem}.txt").write_text("\n".join(lines) + "\n")
                n += 1

    # Augment with dl_helmet rider crops (real Indian traffic).
    rng = random.Random(42)
    for split, our_split in (("train", "train"), ("valid", "valid")):
        for img_path, stem, lines in prepare_yolo_from_dl_helmet(split, rng, max_images=60 if our_split == "train" else 15):
            dst_img = OUT_YOLO / "images" / our_split / f"{stem}.jpg"
            shutil.copy2(img_path, dst_img)
            (OUT_YOLO / "labels" / our_split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n += 1

    yaml_path = OUT_YOLO / "data.yaml"
    yaml_path.write_text(
        yaml.dump(
            {
                "path": str(OUT_YOLO.resolve()),
                "train": "images/train",
                "val": "images/valid",
                "names": {
                    0: "car",
                    1: "two_wheeler",
                    2: "person",
                    3: "license_plate",
                    4: "windshield",
                    5: "signal_light",
                },
            },
            default_flow_style=False,
        )
    )
    return n


def crop_moto(img, bbox, frac: float = 0.72):
    x1, y1, x2, y2 = map(float, bbox[:4])
    y2c = y1 + (y2 - y1) * frac
    h, w = img.shape[:2]
    cx1, cy1 = max(0, int(x1)), max(0, int(y1))
    cx2, cy2 = min(w, int(x2)), min(h, int(y2c))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return img[cy1:cy2, cx1:cx2]


def prepare_rider_from_manifest(samples: list[dict], rows: list[dict], val_ratio: float = 0.2) -> int:
    rng = random.Random(42)
    n = 0
    img_root = OUT_RIDER / "images"
    for split in ("train", "valid", "test"):
        (img_root / split).mkdir(parents=True, exist_ok=True)

    for s in samples:
        img_path = ROOT / s["image"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        motos = [d for d in (s.get("detections_gt") or []) if d.get("cls") == "motorcycle"]
        if not motos:
            continue
        viols = set(s.get("violations") or [])
        no_h = 1 if "no_helmet" in viols else 0
        triple = 1 if "triple_riding" in viols else 0
        split = "test" if rng.random() < val_ratio else "train"
        for i, m in enumerate(motos):
            crop = crop_moto(img, m["bbox"])
            if crop is None or crop.size == 0:
                continue
            name = f"mf_{s['id'][:40]}_{i}.jpg".replace("/", "_")
            rel = f"images/{split}/{name}"
            cv2.imwrite(str(OUT_RIDER / rel), crop)
            rows.append(
                {"path": rel, "no_helmet": no_h, "triple_riding": triple, "split": split}
            )
            n += 1
    return n


def prepare_rider_from_dl_helmet(rows: list[dict], max_per_split: int = 120) -> int:
    """Helmet dataset: class 0=helmet, 1=no_helmet in Roboflow export."""
    names = {0: "helmet", 1: "no_helmet"}
    yaml_path = DL_HELMET / "data.yaml"
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        ns = raw.get("names") or names
        if isinstance(ns, list):
            names = {i: n for i, n in enumerate(ns)}

    n = 0
    img_root = OUT_RIDER / "images"
    for roboflow_split, our_split in (("train", "train"), ("valid", "valid"), ("test", "test")):
        img_dir = DL_HELMET / roboflow_split / "images"
        lbl_dir = DL_HELMET / roboflow_split / "labels"
        if not img_dir.exists():
            continue
        count = 0
        for img_path in sorted(img_dir.iterdir()):
            if count >= max_per_split:
                break
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            lbl = lbl_dir / f"{img_path.stem}.txt"
            if not lbl.exists():
                continue
            has_no, has_ok = False, False
            best_box = None
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                cname = names.get(cid, "").lower()
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                pad = 20
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                if "no" in cname and "helmet" in cname:
                    has_no = True
                    best_box = (x1, y1, x2, y2)
                elif "helmet" in cname:
                    has_ok = True
                    if best_box is None:
                        best_box = (x1, y1, x2, y2)
            if best_box is None:
                crop = cv2.resize(img, (224, 224))
            else:
                x1, y1, x2, y2 = best_box
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
            no_h = 1 if has_no else 0
            name = f"dl_{roboflow_split}_{img_path.stem}.jpg"
            rel = f"images/{our_split}/{name}"
            cv2.imwrite(str(OUT_RIDER / rel), crop)
            rows.append(
                {
                    "path": rel,
                    "no_helmet": no_h,
                    "triple_riding": 0,
                    "split": our_split,
                }
            )
            n += 1
            count += 1
    return n


def prepare_seatbelt_from_eval(val_ratio: float = 0.2) -> tuple[int, int, int]:
    """Build ImageFolder seatbelt pack from data/eval/seatbelt_manifest.json."""
    if not SEATBELT_MANIFEST.exists():
        return 0, 0, 0
    if OUT_SEATBELT.exists():
        shutil.rmtree(OUT_SEATBELT)
    rng = random.Random(42)
    samples = json.loads(SEATBELT_MANIFEST.read_text()).get("samples") or []
    rng.shuffle(samples)
    n_val = max(3, int(len(samples) * val_ratio))
    by_cls: dict[str, list[dict]] = {"belt": [], "no_belt": [], "occluded": []}
    for s in samples:
        note = (s.get("detail") or {}).get("note", "")
        viols = set(s.get("violations") or [])
        if "motorcycle" in note.lower() or "negative" in note.lower():
            cls = "occluded"
        elif "no_seatbelt" in viols:
            cls = "no_belt"
        else:
            cls = "belt"
        by_cls[cls].append(s)
    val_set: list[dict] = []
    for items in by_cls.values():
        if items:
            val_set.append(items[0])
    for s in samples:
        if len(val_set) >= n_val:
            break
        if s not in val_set:
            val_set.append(s)
    val_keys = {s.get("image") for s in val_set}
    counts = {"belt": 0, "no_belt": 0, "occluded": 0}
    for s in samples:
        img_rel = s.get("image", "")
        img_path = ROOT / "data" / "eval" / img_rel
        if not img_path.exists():
            img_path = ROOT / img_rel
        if not img_path.exists():
            continue
        split = "val" if s.get("image") in val_keys else "train"
        note = (s.get("detail") or {}).get("note", "")
        viols = set(s.get("violations") or [])
        if "motorcycle" in note.lower() or "negative" in note.lower():
            cls = "occluded"
        elif "no_seatbelt" in viols:
            cls = "no_belt"
        else:
            cls = "belt"
        dest_dir = OUT_SEATBELT / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(img_rel).name
        shutil.copy2(img_path, dest_dir / stem)
        counts[cls] += 1
    return counts["belt"], counts["no_belt"], counts["occluded"]


def write_rider_csv(rows: list[dict]) -> Path:
    OUT_RIDER.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_RIDER / "labels.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "no_helmet", "triple_riding", "split"])
        w.writeheader()
        w.writerows(rows)
    return csv_path


def run_ocr_synthetic(n: int) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_plate_line_dataset.py"),
        "--synthetic-only",
        "--n-synthetic",
        str(n),
    ]
    r = subprocess.run(cmd, cwd=str(ROOT))
    return r.returncode


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare small training packs from local eval data")
    p.add_argument("--ocr-synthetic", type=int, default=250, help="Synthetic OCR line count")
    p.add_argument("--skip-ocr", action="store_true")
    args = p.parse_args()

    if not MANIFEST.exists():
        print(f"Missing {MANIFEST}", file=sys.stderr)
        return 1

    samples = json.loads(MANIFEST.read_text()).get("samples") or []
    print(f"Manifest samples: {len(samples)}")

    ny = prepare_yolo_from_manifest(samples)
    print(f"YOLO quick pack: {ny} images -> {OUT_YOLO}")

    if OUT_RIDER.exists():
        shutil.rmtree(OUT_RIDER)
    rows: list[dict] = []
    nr1 = prepare_rider_from_manifest(samples, rows)
    nr2 = prepare_rider_from_dl_helmet(rows)
    csv_path = write_rider_csv(rows)
    print(f"Rider CNN: {nr1 + nr2} crops -> {csv_path}")
    print(f"  no_helmet+: {sum(int(r['no_helmet']) for r in rows)}")
    print(f"  triple+: {sum(int(r['triple_riding']) for r in rows)}")

    seatbelt_dir = ROOT / "data" / "datasets" / "seatbelt_cls"
    nb, nn, no = prepare_seatbelt_from_eval()
    if nb + nn + no:
        print(f"Seatbelt cls: {nb} belt, {nn} no_belt, {no} occluded -> {seatbelt_dir}")
    elif seatbelt_dir.exists():
        n_belt = len(list((seatbelt_dir / "train" / "belt").glob("*.jpg")))
        n_nb = len(list((seatbelt_dir / "train" / "no_belt").glob("*.jpg")))
        print(f"Seatbelt cls ready: {n_belt} belt, {n_nb} no_belt (train/)")
    else:
        print("Seatbelt: no eval images found")

    if not args.skip_ocr:
        print(f"Building {args.ocr_synthetic} synthetic OCR lines…")
        run_ocr_synthetic(args.ocr_synthetic)

    print("\nNext: python scripts/run_quick_train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
