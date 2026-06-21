#!/usr/bin/env python3
"""Import IDD Lite (IIIT Hyderabad) into TRACE eval set.

Extracts idd-lite.tar.gz, converts semantic masks to detection bounding boxes,
copies images to data/eval/images/, and writes data/eval/manifest.json.

Usage:
  python scripts/import_idd_lite.py
  python scripts/import_idd_lite.py --tar ~/Downloads/idd-lite.tar.gz
  python scripts/import_idd_lite.py --split val --max-images 100
  python scripts/import_idd_lite.py --extract-only

Then evaluate:
  export TRACE_CONFIG=config/viovision.yaml
  export PYTHONPATH=$PWD
  python scripts/run_full_eval.py --config config/viovision.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_cv.evaluation.idd_lite import (
    DEFAULT_ROOT,
    build_manifest,
    extract_archive,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Import IDD Lite into TRACE eval")
    p.add_argument(
        "--tar",
        default=str(Path.home() / "Downloads" / "idd-lite.tar.gz"),
        help="Path to idd-lite.tar.gz",
    )
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--no-copy", action="store_true", help="Build manifest without copying images")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="Extracted idd20k_lite directory")
    args = p.parse_args()

    root = Path(args.root)
    tar = Path(args.tar)

    if not root.exists():
        alt = Path.home() / "Downloads" / "idd-lite (1).tar.gz"
        if not tar.exists() and alt.exists():
            tar = alt
        if not tar.exists():
            print(f"error: IDD Lite not found. Provide --tar or extract to {root}", file=sys.stderr)
            return 1
        print(f"Extracting {tar} ...")
        root = extract_archive(tar)
        print(f"Extracted to {root}")

    if args.extract_only:
        pairs = len(list((root / "leftImg8bit" / args.split).rglob("*_image.jpg")))
        print(f"Ready: {root} ({args.split} images: {pairs})")
        return 0

    manifest = build_manifest(
        root,
        split=args.split,
        max_images=args.max_images,
        copy_images=not args.no_copy,
    )
    n = manifest["n_samples"]
    with_dets = sum(1 for s in manifest["samples"] if s["detections_gt"])
    print(f"IDD Lite import complete: {n} samples ({with_dets} with detections)")
    print(f"  images  -> data/eval/images/")
    print(f"  manifest-> data/eval/manifest.json")
    print(f"\nNext:")
    print(f"  export TRACE_CONFIG=config/viovision.yaml PYTHONPATH=$PWD")
    print(f"  python scripts/run_full_eval.py --config config/viovision.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
