#!/usr/bin/env python3
"""Build TrOCR line crops by labeling images with Roboflow character OCR.

Uses InferenceHTTPClient:
  - model_id ocr-character-cgtzm/4  (default), or
  - workflow general-segmentation-api-4 with classes "-, 0, 1"

Output merges into data/ocr/lines/ for train_trocr_plate.py.

Usage:
  export ROBOFLOW_API_KEY=...
  python scripts/build_ocr_from_roboflow_inference.py
  python scripts/build_ocr_from_roboflow_inference.py --backend workflow --merge
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_plate_line_dataset import (  # noqa: E402
    MANIFEST,
    OUT,
    assemble_text,
    plate_line_crop,
)
from scripts.load_roboflow_config import api_key, load as load_rf_config  # noqa: E402
from trace_cv.adapters.roboflow_ocr import RoboflowCharOCR, predictions_to_chars  # noqa: E402
from trace_cv.adapters.roboflow_common import collect_predictions  # noqa: E402

PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def discover_image_dirs(cfg: dict) -> list[Path]:
    dirs: list[Path] = [
        ROOT / "data" / "eval" / "roboflow_raw" / "plate" / "train" / "images",
        ROOT / "data" / "eval" / "roboflow_raw" / "plate" / "valid" / "images",
        ROOT / "data" / "raw" / "plate_ocr_rf" / "train" / "images",
        ROOT / "data" / "raw" / "plate_ocr_rf" / "valid" / "images",
        ROOT / "data" / "eval" / "images",
    ]
    seen: list[Path] = []
    for d in dirs:
        if d.exists() and d not in seen:
            seen.append(d)
    return seen


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return list(data.get("samples") or [])
    return list(data)


def save_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"samples": entries}, indent=2))


def label_image(
    reader: RoboflowCharOCR,
    img_path: Path,
    split: str,
    entries: list[dict],
    *,
    min_chars: int = 4,
) -> int:
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    text, conf = reader.read(img)
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(text) < min_chars or conf < 0.25:
        return 0
    h, w = img.shape[:2]
    # Re-run for per-character boxes (needed for line crop).
    client = reader._client  # noqa: SLF001
    if reader.workflow_id:
        result = client.run_workflow_image(
            img,
            workspace=reader.workspace,
            workflow_id=reader.workflow_id,
            parameters={"classes": reader.workflow_classes},
        )
    else:
        result = client.infer_image(img, reader.model_id or "")
    chars = predictions_to_chars(collect_predictions(result), w, h)
    if len(chars) < min_chars:
        return 0
    crop = plate_line_crop(img, chars)
    if crop is None:
        return 0
    assembled = assemble_text(chars)
    plate_text = re.sub(r"[^A-Z0-9]", "", assembled.upper()) or text
    if not PLATE_RE.match(plate_text) and len(plate_text) < min_chars:
        return 0

    out_split = split if split in ("train", "val", "test") else "train"
    out_dir = OUT / out_split
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"rf_{img_path.stem}.jpg"
    out_path = out_dir / name
    cv2.imwrite(str(out_path), crop)
    entries.append(
        {
            "image": str(out_path.relative_to(ROOT)),
            "plate_text": plate_text,
            "split": out_split,
            "source": "roboflow_inference",
            "confidence": round(conf, 4),
        }
    )
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description="Build OCR lines via Roboflow inference")
    p.add_argument("--out", default="data/ocr/lines")
    p.add_argument("--backend", choices=("model", "workflow", "both"), default="model")
    p.add_argument("--image-dir", action="append", default=[])
    p.add_argument("--merge", action="store_true", help="Append to existing manifest")
    p.add_argument("--max-images", type=int, default=500)
    p.add_argument("--train-ratio", type=float, default=0.75)
    p.add_argument("--val-ratio", type=float, default=0.15)
    args = p.parse_args()

    global OUT, MANIFEST  # noqa: PLW0603
    OUT = ROOT / args.out
    MANIFEST = OUT / "manifest.json"
    OUT.mkdir(parents=True, exist_ok=True)

    if not api_key():
        print("ROBOFLOW_API_KEY required (set in .env or environment).", file=sys.stderr)
        return 1

    cfg = load_rf_config()
    wf = (cfg.get("workflows") or {}).get("segmentation") or {}
    models = cfg.get("models") or {}
    workspace = cfg.get("workspace", "prakhar-parkar")

    readers: list[RoboflowCharOCR] = []
    if args.backend in ("model", "both"):
        readers.append(
            RoboflowCharOCR(
                workspace=workspace,
                model_id=models.get("ocr_character", "ocr-character-cgtzm/4"),
                workflow_id=None,
            )
        )
    if args.backend in ("workflow", "both"):
        readers.append(
            RoboflowCharOCR(
                workspace=workspace,
                model_id=None,
                workflow_id=wf.get("id", "general-segmentation-api-4"),
                workflow_classes=wf.get("ocr_char_classes", "-, 0, 1"),
            )
        )

    if not any(r.available for r in readers):
        print("Roboflow inference client unavailable (pip install inference-sdk).", file=sys.stderr)
        return 1

    entries = load_manifest(MANIFEST) if args.merge else []
    existing_stems = {Path(e.get("image", "")).stem for e in entries}

    image_paths: list[Path] = []
    for d in args.image_dir:
        image_paths.extend(iter_images(ROOT / d))
    if not image_paths:
        for d in discover_image_dirs(cfg):
            image_paths.extend(iter_images(d))
    image_paths = list(dict.fromkeys(image_paths))[: args.max_images]
    if not image_paths:
        print("No source images found. Run roboflow_download_eval.py first.", file=sys.stderr)
        return 1

    random.shuffle(image_paths)
    n = len(image_paths)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    added = 0
    for i, img_path in enumerate(image_paths):
        if img_path.stem in existing_stems:
            continue
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        for reader in readers:
            before = len(entries)
            added += label_image(reader, img_path, split, entries)
            if len(entries) > before:
                break

    if not entries:
        print("No OCR lines produced from inference.", file=sys.stderr)
        return 1

    save_manifest(MANIFEST, entries)
    print(f"Roboflow inference OCR: +{added} crops -> {OUT} ({len(entries)} total in manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
