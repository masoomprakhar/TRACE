"""
Prepare helmet crop dataset from:
  Helmet and no helmet rider detection — Khadatkar & Wasule
  https://universe.roboflow.com/gw-khadatkar-and-sv-wasule/helmet-and-no-helmet-rider-detection

What this script does:
  1. Downloads the dataset via the Roboflow SDK (YOLOv11 format).
  2. Reads each image + its YOLO annotation file.
  3. Crops every bounding box.
  4. Remaps source class names → VioVision schema:
       "With Helmet"    -> helmet
       "Without Helmet" -> no_helmet
       "License Plate"  -> DISCARDED (not needed for the classifier)
  5. Saves crops into:
       data/annotations/helmet/helmet/
       data/annotations/helmet/no_helmet/

Output is exactly what train_crop_classifier.py --model helmet expects.

Usage:
    python scripts/prepare_helmet_dataset.py --api-key YOUR_KEY

Get your free Roboflow API key at https://app.roboflow.com/settings/api
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------------
WORKSPACE  = "gw-khadatkar-and-sv-wasule"
PROJECT    = "helmet-and-no-helmet-rider-detection"
VERSION    = 5          # Roboflow dataset version (not model version)
RF_FORMAT  = "yolov11"  # gives us TXT annotations + data.yaml

# Class names exactly as they appear in the downloaded data.yaml / label files.
# Map to VioVision schema strings; None = discard.
CLASS_REMAP: dict[str, str | None] = {
    "With Helmet":    "helmet",
    "Without Helmet": "no_helmet",
    "licence":        None,   # crops go to classifier; YOLO boxes go to detector splits
}

# VioVision detector class id for license_plate (must match traffic.yaml)
LICENSE_PLATE_CLASS_ID = 3

# Minimum crop dimension — skip boxes that are too small to be useful
MIN_CROP_PX = 20

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "data" / "raw" / "helmet_raw"
OUT_DIR      = ROOT / "data" / "annotations" / "helmet"
SPLITS_DIR   = ROOT / "data" / "splits"   # detector fine-tune output

IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_data_yaml(yaml_path: Path) -> dict[int, str]:
    """Return {class_id: class_name} from a Roboflow-exported data.yaml."""
    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    raise ValueError(f"Unexpected 'names' format in {yaml_path}: {type(names)}")


def iter_split(split_dir: Path):
    """Yield (image_path, label_path) pairs from a train/valid/test split dir."""
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


def extract_plates_for_detector(img_path: Path, label_path: Path,
                                 id_to_name: dict[int, str],
                                 out_split: str,
                                 plate_counters: dict[str, int]) -> None:
    """
    Write license_plate boxes from the helmet dataset into the detector
    fine-tune splits as YOLO .txt labels (class id 3 = license_plate).
    Images are copied with a 'hpl_' prefix to avoid collisions.
    """
    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    plate_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        if id_to_name.get(cls_id) != "licence":
            continue
        # Keep coordinates as-is (already normalised YOLO format),
        # just swap to VioVision's license_plate class id
        plate_lines.append(
            f"{LICENSE_PLATE_CLASS_ID} {' '.join(parts[1:5])}"
        )

    if not plate_lines:
        return

    out_img_dir = SPLITS_DIR / out_split / "images"
    out_lbl_dir = SPLITS_DIR / out_split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"hpl_{img_path.name}"
    dst_img  = out_img_dir / out_name
    if not dst_img.exists():
        import shutil as _sh
        _sh.copy2(img_path, dst_img)

    lbl_out = out_lbl_dir / f"hpl_{img_path.stem}.txt"
    lbl_out.write_text("\n".join(plate_lines) + "\n")
    plate_counters[out_split] = plate_counters.get(out_split, 0) + len(plate_lines)


def crop_and_save(img_path: Path, label_path: Path,
                  id_to_name: dict[int, str],
                  counters: dict[str, int]) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARN] Could not read {img_path}, skipping.")
        return
    h, w = img.shape[:2]

    with open(label_path) as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])

        source_name = id_to_name.get(cls_id)
        if source_name is None:
            continue

        target_class = CLASS_REMAP.get(source_name)
        if target_class is None:
            continue  # discard

        # YOLO normalised → pixel
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
            continue

        crop = img[y1:y2, x1:x2]
        out_dir = OUT_DIR / target_class
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = counters.get(target_class, 0)
        out_path = out_dir / f"{img_path.stem}_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), crop)
        counters[target_class] = idx + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare helmet crop dataset from Roboflow."
    )
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--version", type=int, default=VERSION,
                        help=f"Dataset version to download (default: {VERSION})")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if data already in data/raw/helmet_raw/")
    args = parser.parse_args()

    # -- 1. Download ----------------------------------------------------------
    if args.skip_download and DOWNLOAD_DIR.exists():
        print(f"Skipping download, using existing data at {DOWNLOAD_DIR}")
    else:
        print(f"Downloading {WORKSPACE}/{PROJECT} v{args.version} ...")
        from roboflow import Roboflow
        rf = Roboflow(api_key=args.api_key)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        dataset = project.version(args.version).download(
            RF_FORMAT, location=str(DOWNLOAD_DIR), overwrite=True
        )
        print(f"Downloaded to {DOWNLOAD_DIR}")

    # -- 2. Find data.yaml and parse class names ------------------------------
    yaml_candidates = list(DOWNLOAD_DIR.rglob("data.yaml"))
    if not yaml_candidates:
        print("ERROR: data.yaml not found after download. Check DOWNLOAD_DIR.")
        sys.exit(1)
    data_yaml = yaml_candidates[0]
    id_to_name = parse_data_yaml(data_yaml)
    print(f"Classes in downloaded dataset: {id_to_name}")

    # Validate our remap covers all source classes
    unmapped = [n for n in id_to_name.values() if n not in CLASS_REMAP]
    if unmapped:
        print(f"  [WARN] These source classes have no remap entry and will be "
              f"DISCARDED: {unmapped}")

    # -- 3. Wipe and recreate output dirs ------------------------------------
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for cls in [v for v in CLASS_REMAP.values() if v is not None]:
        (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    # -- 4. Crop all splits + extract plate boxes for detector -----------------
    counters: dict[str, int] = {}
    plate_counters: dict[str, int] = {}

    # Map Roboflow split name → detector split name
    split_map = {"train": "train", "valid": "valid", "test": "test"}

    for split_name in ("train", "valid", "test"):
        split_dir = DOWNLOAD_DIR / split_name
        pairs = list(iter_split(split_dir))
        print(f"\nProcessing split '{split_name}': {len(pairs)} images ...")
        for img_path, label_path in pairs:
            crop_and_save(img_path, label_path, id_to_name, counters)
            extract_plates_for_detector(img_path, label_path, id_to_name,
                                         split_map[split_name], plate_counters)

    # -- 5. Report ------------------------------------------------------------
    print("\n=== Helmet classifier crops ===")
    total = 0
    for cls in [v for v in CLASS_REMAP.values() if v is not None]:
        n = counters.get(cls, 0)
        total += n
        print(f"  {cls:15s}: {n} crops -> {OUT_DIR / cls}")
    print(f"  {'TOTAL':15s}: {total} crops")

    print("\n=== License plate boxes → detector splits ===")
    plate_total = sum(plate_counters.values())
    for split_name, n in sorted(plate_counters.items()):
        print(f"  {split_name:8s}: {n} boxes written to data/splits/{split_name}/")
    print(f"  {'TOTAL':8s}: {plate_total} license_plate boxes")

    if counters.get("helmet", 0) == 0 or counters.get("no_helmet", 0) == 0:
        print("\n[WARN] One classifier class has 0 crops — check CLASS_REMAP matches "
              "the actual class names in the downloaded data.yaml above.")
    if plate_total == 0:
        print("\n[WARN] 0 license_plate boxes extracted. Check that 'License Plate' "
              "exists as a class name in the downloaded data.yaml.")
    else:
        print(f"\nReady to train helmet classifier:")
        print(f"  python scripts/train_crop_classifier.py --model helmet "
              f"--data-dir {OUT_DIR} --out models/weights/helmet_svm.pkl")
        print(f"\nRe-run merge_detector_datasets.py to see updated license_plate count.")


if __name__ == "__main__":
    main()
