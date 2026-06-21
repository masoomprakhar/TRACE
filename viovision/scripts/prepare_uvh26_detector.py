"""
Prepare primary detector fine-tune data from:
  UVH-26 — Urban Vision Hackathon Dataset (IISc, Bengaluru CCTV)
  https://huggingface.co/datasets/iisc-aim/UVH-26

Download from HuggingFace first:
    pip install huggingface_hub
    huggingface-cli download iisc-aim/UVH-26 --repo-type dataset --local-dir ./uvh26/

If the download was interrupted (connection reset, segfault, etc.) and you
only have a partial download — e.g. only UVH-26-Train/images/000/ — this
script handles it gracefully. Images referenced in the JSON but not present
on disk are silently skipped. Only what you actually have gets processed.
Pass --partial-ok to suppress the missing-Val-dir warning in that case.

Expected layout after full download:
    uvh26/
    ├── UVH-26-Train/
    │   ├── images/000/*.png, images/001/*.png ...
    │   ├── UVH-26-MV-Train.json     ← use this (majority voting)
    │   └── UVH-26-ST-Train.json
    └── UVH-26-Val/
        ├── images/000/*.png ...
        ├── UVH-26-MV-Val.json
        └── UVH-26-ST-Val.json

Partial download layout (what you likely have after a dropped connection):
    uvh26/
    └── UVH-26-Train/
        ├── images/000/*.png        ← only this subfolder downloaded
        ├── UVH-26-MV-Train.json   ← JSON is small, likely fully downloaded
        └── UVH-26-ST-Train.json

In the partial case the script auto-splits UVH-26-Train 80/20 into
data/splits/train/ and data/splits/valid/ using only the images on disk.

Annotations are COCO JSON format (not YOLO .txt).
This script converts them to YOLO format and remaps the 14 UVH-26
classes to VioVision's 3 base detector classes:

  UVH-26 id  Class              → VioVision class  (id)
  ─────────────────────────────────────────────────────
  1          Hatchback          → car               (0)
  2          Sedan              → car               (0)
  3          SUV                → car               (0)
  4          MUV                → car               (0)
  5          Bus                → car               (0)
  6          Truck              → car               (0)
  7          Three-wheeler      → two_wheeler        (1)
  8          Two-wheeler        → two_wheeler        (1)
  9          LCV                → car               (0)
  10         Mini-bus           → car               (0)
  11         Tempo-traveller    → car               (0)
  12         Bicycle            → two_wheeler        (1)
  13         Van                → car               (0)
  14         Other              → DISCARD

Usage:
    # Full download — dry-run first
    python scripts/prepare_uvh26_detector.py --uvh-dir ./uvh26/ --dry-run
    python scripts/prepare_uvh26_detector.py --uvh-dir ./uvh26/

    # Partial download (interrupted) — auto-splits train folder into train/val
    python scripts/prepare_uvh26_detector.py --uvh-dir ./uvh26/ --partial-ok --dry-run
    python scripts/prepare_uvh26_detector.py --uvh-dir ./uvh26/ --partial-ok
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Class remap: UVH-26 COCO category_id (1-indexed) → VioVision class id
# None = discard
# ---------------------------------------------------------------------------
UVH_ID_TO_VIOVISION: dict[int, int | None] = {
    1:  0,   # Hatchback    → car
    2:  0,   # Sedan        → car
    3:  0,   # SUV          → car
    4:  0,   # MUV          → car
    5:  0,   # Bus          → car
    6:  0,   # Truck        → car
    7:  1,   # Three-wheeler→ two_wheeler
    8:  1,   # Two-wheeler  → two_wheeler
    9:  0,   # LCV          → car
    10: 0,   # Mini-bus     → car
    11: 0,   # Tempo-traveller → car
    12: 1,   # Bicycle      → two_wheeler
    13: 0,   # Van          → car
    14: None, # Other       → discard
}

CLASS_NAMES = {0: "car", 1: "two_wheeler", 2: "person"}

ROOT        = Path(__file__).resolve().parents[1]
OUT_BASE    = ROOT / "data" / "splits"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_image(image_dir: Path, filename: str) -> Path | None:
    """
    UVH-26 images are nested in subfolders (images/000/, images/001/, ...).
    Search all subdirs for the filename.
    """
    # Fast path: flat
    direct = image_dir / filename
    if direct.exists():
        return direct
    # Search one level deep (the actual layout)
    for subdir in image_dir.iterdir():
        if subdir.is_dir():
            candidate = subdir / filename
            if candidate.exists():
                return candidate
    return None


def coco_bbox_to_yolo(x: float, y: float, w: float, h: float,
                       img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """COCO [x_min, y_min, width, height] → YOLO [cx, cy, bw, bh] normalised."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    bw = w / img_w
    bh = h / img_h
    return cx, cy, bw, bh


def process_split(uvh_dir: Path, split_folder: str, json_filename: str,
                   out_split: str, dry_run: bool) -> dict:
    split_dir  = uvh_dir / split_folder
    json_path  = split_dir / json_filename
    image_dir  = split_dir / "images"

    if not json_path.exists():
        print(f"  [WARN] {json_path} not found, skipping split.")
        return {}
    if not image_dir.is_dir():
        print(f"  [WARN] {image_dir} not found, skipping split.")
        return {}

    print(f"\nLoading {json_path} ...")
    with open(json_path) as f:
        coco = json.load(f)

    # Build lookup: image_id → {filename, width, height}
    id_to_img: dict[int, dict] = {
        img["id"]: img for img in coco["images"]
    }

    # Group annotations by image_id
    anns_by_img: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    out_img_dir = OUT_BASE / out_split / "images"
    out_lbl_dir = OUT_BASE / out_split / "labels"
    if not dry_run:
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

    counters = {"images": 0, "boxes": 0, "skipped_no_anns": 0,
                "skipped_img_missing": 0, "discarded_boxes": 0}
    per_class: dict[int, int] = {0: 0, 1: 0, 2: 0}

    for img_id, img_info in id_to_img.items():
        filename = img_info["file_name"]
        img_w    = img_info["width"]
        img_h    = img_info["height"]

        anns = anns_by_img.get(img_id, [])
        if not anns:
            counters["skipped_no_anns"] += 1
            continue

        # Convert annotations
        yolo_lines: list[str] = []
        for ann in anns:
            cat_id   = ann["category_id"]
            dst_id   = UVH_ID_TO_VIOVISION.get(cat_id)
            if dst_id is None:
                counters["discarded_boxes"] += 1
                continue
            x, y, w, h = ann["bbox"]
            if w < 2 or h < 2:
                continue   # degenerate box
            cx, cy, bw, bh = coco_bbox_to_yolo(x, y, w, h, img_w, img_h)
            # Clamp to [0,1] — some COCO annotations slightly exceed bounds
            cx  = max(0.0, min(1.0, cx))
            cy  = max(0.0, min(1.0, cy))
            bw  = max(0.0, min(1.0, bw))
            bh  = max(0.0, min(1.0, bh))
            yolo_lines.append(f"{dst_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            per_class[dst_id] = per_class.get(dst_id, 0) + 1
            counters["boxes"] += 1

        if not yolo_lines:
            counters["skipped_no_anns"] += 1
            continue

        # Find the actual image file (nested in subfolders)
        img_path = find_image(image_dir, filename)
        if img_path is None:
            counters["skipped_img_missing"] += 1
            continue

        if not dry_run:
            stem     = img_path.stem
            out_name = f"uvh_{stem}{img_path.suffix}"
            dst_img  = out_img_dir / out_name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            lbl_path = out_lbl_dir / f"uvh_{stem}.txt"
            lbl_path.write_text("\n".join(yolo_lines) + "\n")

        counters["images"] += 1

    counters["per_class"] = per_class
    return counters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert UVH-26 COCO JSON → YOLO format for VioVision."
    )
    parser.add_argument("--uvh-dir", required=True, type=Path,
                        help="Path to the downloaded UVH-26 root (contains "
                             "UVH-26-Train/ and optionally UVH-26-Val/ subdirs).")
    parser.add_argument("--annotation", choices=["MV", "ST"], default="MV",
                        help="Which consensus annotation to use: MV (majority "
                             "voting, default) or ST (STAPLE EM-based).")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of train images to use as val when "
                             "UVH-26-Val/ is missing (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partial-ok", action="store_true",
                        help="Suppress warnings about missing image subfolders "
                             "and Val split. Use when download was interrupted.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts and remap without writing any files.")
    args = parser.parse_args()

    uvh_dir: Path = args.uvh_dir.resolve()
    if not uvh_dir.is_dir():
        print(f"ERROR: {uvh_dir} is not a directory.")
        sys.exit(1)

    train_dir = uvh_dir / "UVH-26-Train"
    val_dir   = uvh_dir / "UVH-26-Val"

    if not train_dir.is_dir():
        print(f"ERROR: UVH-26-Train/ not found under {uvh_dir}")
        print(f"  Run: huggingface-cli download iisc-aim/UVH-26 "
              f"--repo-type dataset --local-dir {uvh_dir}")
        sys.exit(1)

    has_val = val_dir.is_dir() and any(val_dir.rglob("*.json"))
    if not has_val and not args.partial_ok:
        print(f"[WARN] UVH-26-Val/ not found or empty — download was likely "
              f"interrupted. Will auto-split UVH-26-Train {int((1-args.val_split)*100)}"
              f"/{int(args.val_split*100)} into train/val.")
        print(f"  Pass --partial-ok to suppress this warning.\n")

    # Print effective remap
    print("Effective class remap (UVH-26 → VioVision):")
    print(f"  {'ID':>3}  {'UVH-26 class':<20}  VioVision class")
    print(f"  {'─'*3}  {'─'*20}  {'─'*20}")
    uvh_names = {
        1:"Hatchback", 2:"Sedan", 3:"SUV", 4:"MUV", 5:"Bus", 6:"Truck",
        7:"Three-wheeler", 8:"Two-wheeler", 9:"LCV", 10:"Mini-bus",
        11:"Tempo-traveller", 12:"Bicycle", 13:"Van", 14:"Other"
    }
    for uid, name in uvh_names.items():
        dst = UVH_ID_TO_VIOVISION[uid]
        dst_label = CLASS_NAMES.get(dst, "DISCARD") if dst is not None else "DISCARD"
        print(f"  {uid:>3}  {name:<20}  {dst_label}")
    print()

    if args.dry_run:
        print("DRY RUN — no files will be written.\n")

    total = {"images": 0, "boxes": 0, "skipped_no_anns": 0,
             "skipped_img_missing": 0, "discarded_boxes": 0,
             "per_class": {0: 0, 1: 0, 2: 0}}

    if has_val:
        # Full download — process train and val splits independently
        splits = [
            ("UVH-26-Train", f"UVH-26-{args.annotation}-Train.json", "train"),
            ("UVH-26-Val",   f"UVH-26-{args.annotation}-Val.json",   "valid"),
        ]
        for folder, json_file, out_split in splits:
            print(f"Processing {folder} → data/splits/{out_split}/ ...")
            counts = process_split(uvh_dir, folder, json_file, out_split, args.dry_run)
            if not counts:
                continue
            _accumulate(total, counts)
            _print_split_summary(counts)

    else:
        # Partial download — load train JSON, filter to images on disk,
        # then 80/20 split before writing
        json_path = train_dir / f"UVH-26-{args.annotation}-Train.json"
        if not json_path.exists():
            print(f"ERROR: {json_path} not found. The JSON files are small and "
                  f"should have downloaded even on a partial run. "
                  f"Try re-running the huggingface-cli download command — it "
                  f"resumes from where it left off.")
            sys.exit(1)

        image_dir = train_dir / "data"
        print(f"Loading {json_path} ...")
        with open(json_path) as f:
            coco = json.load(f)

        id_to_img: dict[int, dict] = {img["id"]: img for img in coco["images"]}
        anns_by_img: dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_img[ann["image_id"]].append(ann)

        # Build (img_id, img_path, yolo_lines) for images actually on disk
        usable: list[tuple[int, Path, list[str]]] = []
        discarded_boxes = 0
        for img_id, img_info in id_to_img.items():
            img_path = find_image(image_dir, img_info["file_name"])
            if img_path is None:
                continue  # not downloaded — silently skip
            anns = anns_by_img.get(img_id, [])
            yolo_lines, disc = _convert_anns(anns, img_info["width"],
                                              img_info["height"])
            discarded_boxes += disc
            if yolo_lines:
                usable.append((img_id, img_path, yolo_lines))

        print(f"  Images on disk with valid annotations: {len(usable)}")
        print(f"  Discarded boxes (Other class): {discarded_boxes}")
        if not args.partial_ok:
            full_count = len(id_to_img)
            print(f"  Images in JSON but not on disk: "
                  f"{full_count - len(usable)} "
                  f"(expected — only partial download)")

        if len(usable) < 10:
            print(f"ERROR: only {len(usable)} usable images found. "
                  f"Check that UVH-26-Train/images/000/ exists and contains .png files.")
            sys.exit(1)

        # 80/20 split
        random.seed(args.seed)
        random.shuffle(usable)
        n_val   = max(1, int(len(usable) * args.val_split))
        val_set   = usable[:n_val]
        train_set = usable[n_val:]
        print(f"  Auto-split: {len(train_set)} train / {len(val_set)} val\n")

        for out_split, subset in [("train", train_set), ("valid", val_set)]:
            out_img_dir = OUT_BASE / out_split / "images"
            out_lbl_dir = OUT_BASE / out_split / "labels"
            if not args.dry_run:
                out_img_dir.mkdir(parents=True, exist_ok=True)
                out_lbl_dir.mkdir(parents=True, exist_ok=True)

            per_class: dict[int, int] = {0: 0, 1: 0, 2: 0}
            boxes = 0
            print(f"Writing {out_split}: {len(subset)} images ...")
            for _, img_path, yolo_lines in subset:
                if not args.dry_run:
                    out_name = f"uvh_{img_path.stem}{img_path.suffix}"
                    dst = out_img_dir / out_name
                    if not dst.exists():
                        shutil.copy2(img_path, dst)
                    (out_lbl_dir / f"uvh_{img_path.stem}.txt").write_text(
                        "\n".join(yolo_lines) + "\n"
                    )
                for line in yolo_lines:
                    cid = int(line.split()[0])
                    per_class[cid] = per_class.get(cid, 0) + 1
                    boxes += 1

            total["images"] += len(subset)
            total["boxes"]  += boxes
            total["discarded_boxes"] += discarded_boxes
            for cid in (0, 1, 2):
                total["per_class"][cid] += per_class.get(cid, 0)

    print(f"\n=== UVH-26 totals ===")
    print(f"  Images written : {total['images']}")
    print(f"  Boxes written  : {total['boxes']}")
    print(f"  Discarded boxes (class 'Other'): {total['discarded_boxes']}")
    print(f"  Per class:")
    for cid, name in CLASS_NAMES.items():
        n = total["per_class"].get(cid, 0)
        print(f"    [{cid}] {name:<14}: {n:,} boxes")

    if not args.dry_run:
        print(f"\n  Output at: {OUT_BASE}/{{train,valid}}/")
        print(f"\n  Next: python scripts/merge_detector_datasets.py "
              f"--extra data/splits/windshield_finetune")
    else:
        print(f"\n  Remove --dry-run to write files.")


def _convert_anns(anns: list, img_w: int, img_h: int
                   ) -> tuple[list[str], int]:
    """Convert COCO annotations to YOLO lines. Returns (lines, discarded_count)."""
    lines: list[str] = []
    discarded = 0
    for ann in anns:
        dst_id = UVH_ID_TO_VIOVISION.get(ann["category_id"])
        if dst_id is None:
            discarded += 1
            continue
        x, y, w, h = ann["bbox"]
        if w < 2 or h < 2:
            continue
        cx = max(0.0, min(1.0, (x + w / 2) / img_w))
        cy = max(0.0, min(1.0, (y + h / 2) / img_h))
        bw = max(0.0, min(1.0, w / img_w))
        bh = max(0.0, min(1.0, h / img_h))
        lines.append(f"{dst_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, discarded


def _accumulate(total: dict, counts: dict) -> None:
    for k in ("images", "boxes", "skipped_no_anns",
               "skipped_img_missing", "discarded_boxes"):
        total[k] += counts.get(k, 0)
    for cid in (0, 1, 2):
        total["per_class"][cid] += counts.get("per_class", {}).get(cid, 0)


def _print_split_summary(counts: dict) -> None:
    print(f"  images written : {counts.get('images', 0)}")
    print(f"  boxes written  : {counts.get('boxes', 0)}")
    skipped = counts.get('skipped_no_anns', 0) + counts.get('skipped_img_missing', 0)
    print(f"  skipped        : {skipped} "
          f"(no valid anns or image not on disk)\n")


if __name__ == "__main__":
    main()
