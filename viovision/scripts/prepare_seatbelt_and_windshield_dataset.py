"""
Prepare windshield + seatbelt data from:
  seat_belt and mobile — aiactive
  https://universe.roboflow.com/aiactive20092009-gmail-com/seat_belt-and-mobile

This dataset has THREE classes: mobile, seatbelt, windshield.

This script serves TWO purposes in the pipeline, which is why it's worth
running alongside prepare_seatbelt_dataset.py:

  PURPOSE 1 — Seatbelt classifier crops (merges with seatbelt_raw output):
    windshield + seatbelt boxes in the SAME image mean you get matched
    "here is the windshield ROI, and inside it there IS a seatbelt visible"
    pairs. Seatbelt crops go to data/annotations/seatbelt/seatbelt/.
    Windshield regions that contain NO seatbelt box become no_seatbelt crops
    (these are true negatives, not synthesised — much higher quality than
    the random-region negatives from prepare_seatbelt_dataset.py).

  PURPOSE 2 — Windshield detector fine-tune data:
    Windshield boxes are re-formatted as YOLO annotations and written to
    data/splits/windshield_finetune/{train,valid,test}/
    to be merged into the primary YOLOv11 fine-tune split for the
    `windshield` class (class id 4 in traffic.yaml).

  mobile boxes are discarded entirely.

Usage:
    python scripts/prepare_seatbelt_and_windshield_dataset.py --api-key YOUR_KEY

Run AFTER prepare_seatbelt_dataset.py — this script APPENDS to the
seatbelt annotation dirs rather than wiping them.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------------
WORKSPACE   = "aiactive20092009-gmail-com"
PROJECT     = "seat_belt-and-mobile"
VERSION     = 1
RF_FORMAT   = "yolov11"

# Source class names as they appear in the downloaded data.yaml.
# Adjust if Roboflow exports them differently (check printed id_to_name).
SEATBELT_NAMES   = {"seatbelt", "seat_belt", "Seatbelt", "seat belt"}
WINDSHIELD_NAMES = {"windshield", "Windshield", "wind_shield"}
DISCARD_NAMES    = {"mobile", "Mobile", "phone"}

MIN_CROP_PX = 32

ROOT                = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR        = ROOT / "data" / "raw" / "seatbelt_mobile_raw"
SEATBELT_ANNOT_DIR  = ROOT / "data" / "annotations" / "seatbelt"
WINDSHIELD_YOLO_DIR = ROOT / "data" / "splits" / "windshield_finetune"
IMAGE_EXTS          = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Class id for windshield in traffic.yaml (must match configs/traffic.yaml)
WINDSHIELD_CLASS_ID = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_data_yaml(yaml_path: Path) -> dict[int, str]:
    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    merged = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def yolo_to_pixel(cx: float, cy: float, bw: float, bh: float,
                   W: int, H: int) -> tuple[int, int, int, int]:
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    aa = (ax2 - ax1) * (ay2 - ay1)
    ba = (bx2 - bx1) * (by2 - by1)
    return inter / max(aa + ba - inter, 1)


def iter_split(split_dir: Path):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.is_dir():
        return
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            yield img_path, label_path


def process_image(img_path: Path, label_path: Path,
                   id_to_name: dict[int, str],
                   split_name: str,
                   counters: dict[str, int]) -> list[str]:
    """
    Returns YOLO-format lines for windshield boxes (for the detector
    fine-tune split). Saves seatbelt/no_seatbelt crops as a side-effect.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    H, W = img.shape[:2]

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    # Separate boxes by semantic role
    seatbelt_boxes: list[tuple[int, int, int, int]] = []
    windshield_boxes: list[tuple[int, int, int, int]] = []
    windshield_yolo_lines: list[str] = []

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        name = id_to_name.get(cls_id, "")

        if name in DISCARD_NAMES:
            continue

        x1, y1, x2, y2 = yolo_to_pixel(*map(float, parts[1:5]), W, H)
        if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
            continue

        if name in SEATBELT_NAMES:
            seatbelt_boxes.append((x1, y1, x2, y2))
        elif name in WINDSHIELD_NAMES:
            windshield_boxes.append((x1, y1, x2, y2))
            # Re-emit as windshield class for the detector fine-tune data
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            windshield_yolo_lines.append(
                f"{WINDSHIELD_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )

    # -- Purpose 1: seatbelt classifier crops ---------------------------------

    # Positive: seatbelt boxes
    pos_dir = SEATBELT_ANNOT_DIR / "seatbelt"
    pos_dir.mkdir(parents=True, exist_ok=True)
    for x1, y1, x2, y2 in seatbelt_boxes:
        crop = img[y1:y2, x1:x2]   # raw — CLAHE applied by classifier at feature time
        idx = counters.get("seatbelt", 0)
        cv2.imwrite(str(pos_dir / f"sbm_{img_path.stem}_{idx:04d}.jpg"), crop)
        counters["seatbelt"] = idx + 1

    # True negative: windshield boxes that have NO seatbelt overlap
    # These are far higher quality than synthesised negatives because the
    # annotator explicitly labelled this windshield and chose NOT to add a
    # seatbelt box inside it.
    neg_dir = SEATBELT_ANNOT_DIR / "no_seatbelt"
    neg_dir.mkdir(parents=True, exist_ok=True)
    for wx1, wy1, wx2, wy2 in windshield_boxes:
        # Skip this windshield box if a seatbelt box overlaps it significantly
        has_seatbelt = any(iou((wx1, wy1, wx2, wy2), sb) > 0.15
                            for sb in seatbelt_boxes)
        if has_seatbelt:
            continue
        crop = img[wy1:wy2, wx1:wx2]   # raw — CLAHE applied by classifier at feature time
        idx = counters.get("no_seatbelt", 0)
        cv2.imwrite(str(neg_dir / f"sbm_{img_path.stem}_ws_{idx:04d}.jpg"), crop)
        counters["no_seatbelt"] = idx + 1

    return windshield_yolo_lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare seatbelt + windshield data from aiactive dataset."
    )
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--version", type=int, default=VERSION)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if args.skip_download and DOWNLOAD_DIR.exists():
        print(f"Skipping download, using {DOWNLOAD_DIR}")
    else:
        print(f"Downloading {WORKSPACE}/{PROJECT} v{args.version} ...")
        from roboflow import Roboflow
        rf = Roboflow(api_key=args.api_key)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        project.version(args.version).download(
            RF_FORMAT, location=str(DOWNLOAD_DIR), overwrite=True
        )
        print(f"Downloaded to {DOWNLOAD_DIR}")

    yaml_candidates = list(DOWNLOAD_DIR.rglob("data.yaml"))
    if not yaml_candidates:
        print("ERROR: data.yaml not found.")
        sys.exit(1)
    id_to_name = parse_data_yaml(yaml_candidates[0])
    print(f"Source classes: {id_to_name}")

    # Validate we can recognise the class names
    found_seatbelt  = any(n in SEATBELT_NAMES   for n in id_to_name.values())
    found_windshield = any(n in WINDSHIELD_NAMES for n in id_to_name.values())
    if not found_seatbelt:
        print(f"[WARN] No seatbelt class found. Saw: {list(id_to_name.values())}. "
              f"Update SEATBELT_NAMES if needed.")
    if not found_windshield:
        print(f"[WARN] No windshield class found. Saw: {list(id_to_name.values())}. "
              f"Update WINDSHIELD_NAMES if needed.")

    # Set up windshield YOLO fine-tune dirs
    for split in ("train", "valid", "test"):
        (WINDSHIELD_YOLO_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (WINDSHIELD_YOLO_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {}

    for split_name in ("train", "valid", "test"):
        pairs = list(iter_split(DOWNLOAD_DIR / split_name))
        print(f"\nProcessing '{split_name}': {len(pairs)} images ...")

        ws_img_out = WINDSHIELD_YOLO_DIR / split_name / "images"
        ws_lbl_out = WINDSHIELD_YOLO_DIR / split_name / "labels"

        for img_path, label_path in pairs:
            ws_lines = process_image(img_path, label_path, id_to_name,
                                      split_name, counters)
            if ws_lines:
                # Copy image + write remapped label for windshield fine-tune
                import shutil as _sh
                _sh.copy2(img_path, ws_img_out / img_path.name)
                lbl_out = ws_lbl_out / (img_path.stem + ".txt")
                lbl_out.write_text("\n".join(ws_lines) + "\n")
                counters["windshield_images"] = counters.get("windshield_images", 0) + 1

    print("\n=== Results ===")
    for k, v in sorted(counters.items()):
        print(f"  {k:25s}: {v}")
    print(f"\n  Seatbelt crops appended to: {SEATBELT_ANNOT_DIR}")
    print(f"  Windshield YOLO data at:    {WINDSHIELD_YOLO_DIR}")
    print(f"\n  NOTE: the no_seatbelt crops here are TRUE negatives (windshield")
    print(f"  regions with no seatbelt annotation) — higher quality than the")
    print(f"  synthesised negatives from prepare_seatbelt_dataset.py.")
    print(f"\n  Merge windshield_finetune/ into your main YOLO splits to add")
    print(f"  the `windshield` class to the primary detector.")


if __name__ == "__main__":
    main()
