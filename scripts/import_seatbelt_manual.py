#!/usr/bin/env python3
"""Import manually labeled seatbelt images into TRACE train + eval sets.

Expects a labels.json next to the source images (or pass --labels).

labels.json format:
{
  "samples": [
    {
      "file": "image3525_png.jpg",
      "cls": "belt",
      "note": "driver wearing belt, drinking"
    },
    {
      "file": "xyzabc (24).jpeg",
      "cls": "no_belt",
      "note": "night interior, no strap visible"
    }
  ]
}

Outputs:
  data/raw/seatbelt_manual/images/     — normalized copies
  data/raw/seatbelt_manual/labels.json
  data/datasets/seatbelt_cls/          — driver-roi crops merged into train/val
  data/eval/images/seatbelt/manual/    — full frames for eval
  data/eval/seatbelt_manifest.json     — merged (manual + existing negatives)

Usage:
  python scripts/import_seatbelt_manual.py /Users/prakharsingh/Downloads/seatbelt
  python scripts/import_seatbelt_manual.py --rebuild-manifest-only
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_cv.detection.roi import crop, driver_roi  # noqa: E402

RAW = ROOT / "data" / "raw" / "seatbelt_manual"
RAW_IMAGES = RAW / "images"
LABELS_PATH = RAW / "labels.json"
CLS_ROOT = ROOT / "data" / "datasets" / "seatbelt_cls"
EVAL_IMAGES = ROOT / "data" / "eval" / "images" / "seatbelt" / "manual"
MANIFEST = ROOT / "data" / "eval" / "seatbelt_manifest.json"
EVAL = ROOT / "data" / "eval"

# Human labels for /Users/prakharsingh/Downloads/seatbelt (edit labels.json to change).
DEFAULT_LABELS = {
    "samples": [
        {"file": "2024-02-seatbelt-camera-qld.jpg", "cls": "no_belt", "note": "driver, no strap"},
        {"file": "image3525_png.jpg", "cls": "belt", "note": "strap visible across chest"},
        {"file": "image3535_png.jpg", "cls": "belt", "note": "strap visible"},
        {"file": "seatbelt.webp", "cls": "belt", "note": "buckling, strap across torso"},
        {"file": "images (3).jpeg", "cls": "belt", "note": "black strap on red shirt"},
        {"file": "images (5).jpeg", "cls": "no_belt", "note": "through windshield, no strap"},
        {"file": "images (6).jpeg", "cls": "belt", "note": "both occupants fastening belts"},
        {"file": "images (7).jpeg", "cls": "belt", "note": "buckling grey belt"},
        {"file": "images (8).jpeg", "cls": "belt", "note": "woman buckling"},
        {"file": "images (9).jpeg", "cls": "belt", "note": "buckling close-up"},
        {"file": "xyzabc (24).jpeg", "cls": "no_belt", "note": "night, strap hanging unused"},
        {"file": "xyzabc (150).jpeg", "cls": "no_belt", "note": "night interior"},
        {"file": "xyzabc (170).jpeg", "cls": "no_belt", "note": "night, looking down"},
        {"file": "xyzabc (215).jpeg", "cls": "no_belt", "note": "night, on phone, no strap"},
        {"file": "xyzabc (242).jpeg", "cls": "no_belt", "note": "hoodie, no strap"},
        {"file": "xyzabc (253).jpeg", "cls": "no_belt", "note": "strap retracted, not worn"},
        {
            "file": "videonetics-ai-enabled-seat-belt-detection-technology-920x533.jpg",
            "cls": "no_belt",
            "note": "CCTV front view, driver no visible strap",
        },
        {
            "file": "50556f60-82c0-44d5-9f1b-e651908c06dd_6f7b0e0a.webp",
            "cls": "belt",
            "note": "bus driver wearing belt",
        },
    ]
}


def _slug(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_").lower()
    return stem[:80] or "img"


def _read_image(path: Path) -> tuple[any, str]:
    """Load image; convert webp to jpg array."""
    img = cv2.imread(str(path))
    if img is not None:
        return img, ".jpg"
    # OpenCV may lack webp — try PIL fallback
    try:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        import numpy as np

        arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        return arr, ".jpg"
    except Exception:
        return None, ".jpg"


def _driver_crop(frame) -> any:
    h, w = frame.shape[:2]
    roi = driver_roi((0.0, 0.0, float(w), float(h)))
    region = crop(frame, roi)
    return region if region is not None and region.size else frame


def import_source(src_dir: Path, labels: dict, val_ratio: float, seed: int) -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    RAW_IMAGES.mkdir(parents=True, exist_ok=True)
    EVAL_IMAGES.mkdir(parents=True, exist_ok=True)

    LABELS_PATH.write_text(json.dumps(labels, indent=2))

    rng = random.Random(seed)
    samples = labels.get("samples", [])
    imported: list[dict] = []

    for i, row in enumerate(samples, start=1):
        fname = row.get("file", "")
        cls = row.get("cls", "").strip()
        if cls not in {"belt", "no_belt", "occluded"}:
            print(f"skip {fname}: invalid cls={cls!r}", file=sys.stderr)
            continue
        src = src_dir / fname
        if not src.exists():
            print(f"skip missing: {src}", file=sys.stderr)
            continue

        frame, ext = _read_image(src)
        if frame is None:
            print(f"skip unreadable: {src}", file=sys.stderr)
            continue

        safe = f"manual_{i:02d}_{_slug(fname)}{ext}"
        raw_path = RAW_IMAGES / safe
        eval_path = EVAL_IMAGES / safe
        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(str(eval_path), frame)

        crop_img = _driver_crop(frame)
        imported.append(
            {
                "cls": cls,
                "safe": safe,
                "crop": crop_img,
                "eval_rel": str(eval_path.relative_to(EVAL)),
                "violations": ["no_seatbelt"] if cls == "no_belt" else [],
                "note": row.get("note", ""),
            }
        )

    if not imported:
        print("No images imported.", file=sys.stderr)
        return 1

    # Merge crops into seatbelt_cls (keep existing Karan data, add manual_* files)
    for split in ("train", "val"):
        for c in ("belt", "no_belt", "occluded"):
            (CLS_ROOT / split / c).mkdir(parents=True, exist_ok=True)

    rng.shuffle(imported)
    n_val = max(1, int(round(len(imported) * val_ratio)))
    val_set = {x["safe"] for x in imported[:n_val]}

    added = {"train": 0, "val": 0}
    for item in imported:
        split = "val" if item["safe"] in val_set else "train"
        out = CLS_ROOT / split / item["cls"] / item["safe"]
        cv2.imwrite(str(out), item["crop"])
        added[split] += 1

    # Rebuild manifest: manual samples + preserve negatives from existing manifest
    manual_manifest = []
    for item in imported:
        manual_manifest.append(
            {
                "image": item["eval_rel"],
                "eval_kind": "seatbelt_violation",
                "violations": item["violations"],
                "detail": {
                    "source": "seatbelt_manual",
                    "cls": item["cls"],
                    "note": item["note"],
                },
            }
        )

    existing = []
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text()).get("samples", [])
    negatives = [
        s
        for s in existing
        if not s.get("violations")
        and "negative" in str((s.get("detail") or {}).get("note", "")).lower()
    ]
    merged = manual_manifest + negatives
    MANIFEST.write_text(json.dumps({"samples": merged}, indent=2))

    print(f"Imported {len(imported)} images -> {RAW_IMAGES}")
    print(f"  belt:     {sum(1 for x in imported if x['cls'] == 'belt')}")
    print(f"  no_belt:  {sum(1 for x in imported if x['cls'] == 'no_belt')}")
    print(f"  crops -> seatbelt_cls train={added['train']} val={added['val']}")
    print(f"  manifest: {MANIFEST} ({len(merged)} samples, {len(manual_manifest)} manual)")
    print(f"  labels:   {LABELS_PATH}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Import manual seatbelt images + labels")
    p.add_argument(
        "source",
        nargs="?",
        default="/Users/prakharsingh/Downloads/seatbelt",
        help="folder with raw images",
    )
    p.add_argument("--labels", default=None, help="path to labels.json (optional)")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rebuild-manifest-only",
        action="store_true",
        help="re-merge manifest from data/raw/seatbelt_manual/labels.json",
    )
    args = p.parse_args()

    if args.rebuild_manifest_only:
        if not LABELS_PATH.exists():
            print(f"Missing {LABELS_PATH}", file=sys.stderr)
            return 1
        labels = json.loads(LABELS_PATH.read_text())
        return import_source(RAW_IMAGES.parent.parent / "Downloads" / "seatbelt", labels, args.val_ratio, args.seed)

    labels_path = Path(args.labels) if args.labels else None
    if labels_path and labels_path.exists():
        labels = json.loads(labels_path.read_text())
    elif LABELS_PATH.exists():
        labels = json.loads(LABELS_PATH.read_text())
    else:
        labels = DEFAULT_LABELS

    return import_source(Path(args.source), labels, args.val_ratio, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
