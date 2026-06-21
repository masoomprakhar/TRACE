"""
Merge multiple YOLO-format dataset sources into the canonical
data/splits/{train,valid,test}/ layout that traffic.yaml points at,
then print a class-balance report.

Run this AFTER all individual prepare_*.py scripts to produce the final
unified fine-tune dataset for train_yolo.py.

Also does a hard-conditions tagging pass if you have images with known
hard-condition filenames (night_, rain_, glare_ prefixes) — prints a
separate mAP tracking reminder so you don't forget to run eval on that
subset (guide section 4).

Usage:
    # Merge windshield fine-tune data into the main splits (UVH-26 already
    # wrote directly to data/splits/, so only the windshield supplement needs
    # explicit merging):
    python scripts/merge_detector_datasets.py \
        --extra data/splits/windshield_finetune

    # Merge multiple extras at once:
    python scripts/merge_detector_datasets.py \
        --extra data/splits/windshield_finetune data/splits/plate_finetune
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
MAIN_SPLIT = ROOT / "data" / "splits"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS     = ("train", "valid", "test")

CLASS_NAMES = {0: "car", 1: "two_wheeler", 2: "person",
               3: "license_plate", 4: "windshield", 5: "signal_light"}

HARD_PREFIXES = ("night_", "rain_", "glare_", "fog_", "dark_")


def count_boxes(split_dir: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    lbl_dir = split_dir / "labels"
    if not lbl_dir.is_dir():
        return counts
    for lbl in lbl_dir.iterdir():
        if lbl.suffix != ".txt":
            continue
        for line in lbl.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                cid = int(parts[0])
                counts[cid] = counts.get(cid, 0) + 1
    return counts


def count_images(split_dir: Path) -> int:
    img_dir = split_dir / "images"
    if not img_dir.is_dir():
        return 0
    return sum(1 for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def count_hard_images(split_dir: Path) -> int:
    img_dir = split_dir / "images"
    if not img_dir.is_dir():
        return 0
    return sum(1 for f in img_dir.iterdir()
               if f.suffix.lower() in IMAGE_EXTS
               and any(f.name.startswith(p) for p in HARD_PREFIXES))


def merge_extra_into_main(extra_dir: Path, dry_run: bool) -> dict[str, int]:
    """Copy images+labels from extra split dirs into the main splits."""
    merged = {"images": 0, "labels": 0, "skipped_exist": 0}
    for split in SPLITS:
        src_img = extra_dir / split / "images"
        src_lbl = extra_dir / split / "labels"
        dst_img = MAIN_SPLIT / split / "images"
        dst_lbl = MAIN_SPLIT / split / "labels"
        if not src_img.is_dir():
            continue
        for src_f in src_img.iterdir():
            if src_f.suffix.lower() not in IMAGE_EXTS:
                continue
            dst_f = dst_img / src_f.name
            if dst_f.exists():
                merged["skipped_exist"] += 1
                continue
            if not dry_run:
                shutil.copy2(src_f, dst_f)
            merged["images"] += 1

        if src_lbl.is_dir():
            for src_f in src_lbl.iterdir():
                if src_f.suffix != ".txt":
                    continue
                dst_f = dst_lbl / src_f.name
                if not dst_f.exists() and not dry_run:
                    shutil.copy2(src_f, dst_f)
                merged["labels"] += 1

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge supplementary YOLO datasets into the main splits "
                    "and print a class-balance report."
    )
    parser.add_argument("--extra", nargs="*", type=Path, default=[],
                        help="Extra dataset dirs to merge (each must have "
                             "train/valid/test subdirs with images/ + labels/).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only, do not copy files.")
    args = parser.parse_args()

    # -- Merge extra sources --------------------------------------------------
    for extra in args.extra:
        if not extra.is_dir():
            print(f"[WARN] --extra path not found: {extra}, skipping.")
            continue
        print(f"\nMerging {extra} into {MAIN_SPLIT} ...")
        result = merge_extra_into_main(extra, args.dry_run)
        print(f"  Copied {result['images']} images, {result['labels']} labels "
              f"(skipped {result['skipped_exist']} already-existing).")

    if args.dry_run:
        print("\nDRY RUN — no files were written.\n")

    # -- Class balance report -------------------------------------------------
    print("\n=== Class balance report ===")
    print(f"  Split dirs: {MAIN_SPLIT}\n")

    total_counts: dict[int, int] = {}
    total_hard = 0

    for split in SPLITS:
        split_dir = MAIN_SPLIT / split
        n_img   = count_images(split_dir)
        n_hard  = count_hard_images(split_dir)
        counts  = count_boxes(split_dir)
        total_hard += n_hard
        for cid, n in counts.items():
            total_counts[cid] = total_counts.get(cid, 0) + n

        print(f"  [{split:5s}]  {n_img:5d} images  "
              f"({n_hard} tagged hard-condition: "
              f"{100*n_hard/max(n_img,1):.0f}%)")
        max_in_split = max(counts.values()) if counts else 1
        for cid in range(6):
            n = counts.get(cid, 0)
            name = CLASS_NAMES.get(cid, f"class_{cid}")
            bar  = "█" * min(30, n // max(1, max_in_split // 30))
            print(f"    [{cid}] {name:14s}: {n:6d}  {bar}")

    print(f"\n  All splits combined:")
    max_n = max(total_counts.values()) if total_counts else 1
    for cid in range(6):
        n    = total_counts.get(cid, 0)
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        bar  = "█" * min(40, n // max(1, max_n // 40))
        flag = "  ← ⚠ LOW" if n < 200 else ""
        print(f"    [{cid}] {name:14s}: {n:6d}  {bar}{flag}")

    print(f"\n  Hard-condition images (night/rain/glare/fog prefix): {total_hard}")
    if total_hard < 100:
        print(f"  ⚠  Guide section 4: hold out a DELIBERATE hard-conditions "
              f"subset and report metrics on it separately. Currently only "
              f"{total_hard} images are tagged. Rename hard-condition images "
              f"with a night_/rain_/glare_ prefix so this counter picks them up.")

    # Warn on missing classes
    missing = [CLASS_NAMES[i] for i in range(6) if total_counts.get(i, 0) == 0]
    if missing:
        print(f"\n  ⚠  Classes with 0 boxes (not yet in fine-tune data): {missing}")
        print(f"     license_plate: run prepare_uvh26_detector.py or add a plate "
              f"detection Roboflow dataset.")
        print(f"     windshield:    run prepare_seatbelt_and_windshield_dataset.py.")
        print(f"     signal_light:  add a traffic-light Roboflow dataset "
              f"(LISA/Bosch crops, re-annotated with class id 5).")

    print(f"\n  When class balance looks reasonable, run:")
    print(f"    python scripts/train_yolo.py --data configs/traffic.yaml")


if __name__ == "__main__":
    main()
