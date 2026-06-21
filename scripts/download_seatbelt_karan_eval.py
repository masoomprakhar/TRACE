#!/usr/bin/env python3
"""Download Karan Panja seatbelt eval set + negative motorcycle/plate images.

Builds data/eval/seatbelt_manifest.json with eval_kind: seatbelt_violation.

Usage:
  export ROBOFLOW_API_KEY=...
  python scripts/download_seatbelt_karan_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
IMAGES = EVAL / "images" / "seatbelt"
MANIFEST = EVAL / "seatbelt_manifest.json"

WORKSPACE = "karan-panja"
PROJECT = "seat-belt-detection-uhqwa"
VERSION = 1

NO_BELT_RE = re.compile(r"no[\s_-]?seatbelt|person-noseatbelt|noseatbelt", re.I)
SKIP_RE = re.compile(r"not[\s_-]?clear", re.I)


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


def violations_from_labels(label_path: Path, id_to_name: dict[int, str]) -> list[str]:
    if not label_path.exists():
        return []
    vios: list[str] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = id_to_name.get(int(parts[0]), "")
        if SKIP_RE.search(cls):
            continue
        if NO_BELT_RE.search(cls) or "noseatbelt" in cls.lower().replace("_", ""):
            vios.append("no_seatbelt")
    return list(dict.fromkeys(vios))


def download_test(raw: Path) -> Path | None:
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
        ds = ver.download("yolov8", location=str(raw), overwrite=True)
        return Path(ds.location) if hasattr(ds, "location") else raw
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return None


def copy_negatives(samples: list[dict], max_neg: int = 30) -> None:
    """Copy motorcycle/plate images from main eval manifest as FP negatives."""
    main = EVAL / "manifest.json"
    if not main.exists():
        return
    data = json.loads(main.read_text())
    neg = 0
    for s in data.get("samples", []):
        if neg >= max_neg:
            break
        detail = s.get("detail") or {}
        vios = s.get("violations") or detail.get("violations") or []
        plate = detail.get("plate_text")
        fname = s.get("image") or s.get("path")
        if not fname:
            continue
        is_plate = bool(plate) or "plate" in str(s.get("eval_kind", "")).lower()
        is_helmet = "helmet" in str(s.get("eval_kind", "")).lower() or "no_helmet" in vios
        if not (is_plate or is_helmet):
            continue
        src = EVAL / "images" / Path(fname).name
        if not src.exists():
            src = ROOT / fname
        if not src.exists():
            continue
        dest_name = f"neg_{Path(fname).name}"
        dest = IMAGES / dest_name
        shutil.copy2(src, dest)
        samples.append(
            {
                "image": str(dest.relative_to(EVAL)),
                "eval_kind": "seatbelt_violation",
                "violations": [],
                "detail": {"note": "negative motorcycle/plate"},
            }
        )
        neg += 1
    print(f"  added {neg} negative samples")


def main() -> int:
    p = argparse.ArgumentParser(description="Build seatbelt eval manifest")
    p.add_argument("--max-test", type=int, default=80)
    args = p.parse_args()

    raw = ROOT / "data" / "raw" / "seatbelt_karan_eval"
    ds = download_test(raw)
    if not ds or not (ds / "data.yaml").exists():
        print("Could not download Karan Panja test set.", file=sys.stderr)
        return 1

    if IMAGES.exists():
        shutil.rmtree(IMAGES)
    IMAGES.mkdir(parents=True)

    id_to_name = load_names(ds / "data.yaml")
    samples: list[dict] = []
    test_img = ds / "test" / "images"
    test_lbl = ds / "test" / "labels"
    if not test_img.exists():
        test_img = ds / "valid" / "images"
        test_lbl = ds / "valid" / "labels"

    count = 0
    for img_path in sorted(test_img.iterdir()):
        if count >= args.max_test:
            break
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = test_lbl / f"{img_path.stem}.txt"
        vios = violations_from_labels(lbl, id_to_name)
        # skip ambiguous Not Clear-only frames
        if lbl.exists():
            classes = [id_to_name.get(int(l.split()[0]), "") for l in lbl.read_text().splitlines() if l.strip()]
            if classes and all(SKIP_RE.search(c) for c in classes):
                continue
        dest = IMAGES / img_path.name
        shutil.copy2(img_path, dest)
        samples.append(
            {
                "image": str(dest.relative_to(EVAL)),
                "eval_kind": "seatbelt_violation",
                "violations": vios,
                "detail": {},
            }
        )
        count += 1

    copy_negatives(samples)
    MANIFEST.write_text(json.dumps({"samples": samples}, indent=2))
    print(f"Seatbelt eval manifest: {MANIFEST} ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
