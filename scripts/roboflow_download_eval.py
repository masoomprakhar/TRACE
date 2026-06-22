#!/usr/bin/env python3
"""Download Roboflow helmet / plate datasets into data/eval/ and build manifest.json.

Requires:
  pip install roboflow
  export ROBOFLOW_API_KEY=your_key

Usage:
  python scripts/roboflow_download_eval.py --list-projects

  # Helmet violation eval (test split)
  python scripts/roboflow_download_eval.py --kind helmet \\
    --project helmet-gj8do --version 2 --split test --max-images 50

  # Plate detection + OCR eval
  python scripts/roboflow_download_eval.py --kind plate \\
    --project indian-license-plate-detection-computer-vision-dataset \\
    --version 1 --split test --max-images 50

  # Both sets merged (recommended for submission metrics)
  python scripts/roboflow_download_eval.py --kind both --max-images 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
IMAGES_DIR = EVAL_DIR / "images"
MANIFEST_PATH = EVAL_DIR / "manifest.json"
IDD_MANIFEST_BACKUP = EVAL_DIR / "manifest.idd-lite.json"

_PLATE_IN_NAME = re.compile(r"([A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4})", re.I)
_NO_HELMET_RE = re.compile(r"no[\s_-]?helmet|without[\s_-]?helmet|nohelmet", re.I)

# FIX 1: Broadened helmet pattern — matches "helmet_on", "with_helmet", "wearing_helmet",
# or any string that contains "helmet" but is NOT a no-helmet variant.
_HELMET_PRESENT_RE = re.compile(r"helmet", re.I)


def _api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ROBOFLOW_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        print("error: set ROBOFLOW_API_KEY in your environment or .env", file=sys.stderr)
        sys.exit(1)
    return key


def list_projects(workspace: str) -> None:
    import requests

    r = requests.get(
        f"https://api.roboflow.com/{workspace}",
        params={"api_key": _api_key()},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    projects = data.get("workspace", {}).get("projects", [])
    if not projects:
        print(json.dumps(data, indent=2)[:4000])
        return
    print(f"Projects in workspace '{workspace}':\n")
    for p in projects:
        print(f"  slug: {p.get('id')}")
        print(f"  name: {p.get('name')}")
        print(f"  type: {p.get('type')}")
        print()


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _load_class_names(dataset_dir: Path) -> dict[int, str]:
    data_yaml = dataset_dir / "data.yaml"
    class_names: dict[int, str] = {0: "license_plate"}
    if data_yaml.exists():
        import yaml

        raw = yaml.safe_load(data_yaml.read_text()) or {}
        names = raw.get("names") or {}
        if isinstance(names, dict):
            class_names = {int(k): v for k, v in names.items()}
        elif isinstance(names, list):
            class_names = {i: n for i, n in enumerate(names)}
    return class_names


def _image_roots(dataset_dir: Path, split: str) -> tuple[Path, Path]:
    img_root = dataset_dir / split / "images"
    lbl_root = dataset_dir / split / "labels"
    if not img_root.exists():
        img_root = dataset_dir / "images"
        lbl_root = dataset_dir / "labels"
    return img_root, lbl_root


def yolo_line_to_bbox(line: str, w: int, h: int, class_names: dict) -> dict | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    cx, cy, bw, bh = map(float, parts[1:5])
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    name = class_names.get(cls_id, class_names.get(str(cls_id), f"class_{cls_id}"))
    return {
        "cls": _norm(name),
        "raw_cls": name,
        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "confidence": 1.0,
    }


def guess_plate_text(stem: str) -> str | None:
    # FIX 2: Try harder — replace underscores/hyphens with spaces before matching,
    # since Roboflow renames files but sometimes encodes plate in the stem.
    cleaned = stem.replace("_", " ").replace("-", " ")
    m = _PLATE_IN_NAME.search(cleaned)
    return m.group(1).upper().replace(" ", "") if m else None


def _is_no_helmet(name: str) -> bool:
    return bool(_NO_HELMET_RE.search(name))


# FIX 1 (continued): Simplified and correct helmet check.
def _is_helmet_ok(name: str) -> bool:
    """Return True if this label indicates a helmet IS present (not a violation)."""
    if _is_no_helmet(name):
        return False
    # Any label containing "helmet" that isn't a no-helmet label = helmet present
    return bool(_HELMET_PRESENT_RE.search(name))


def _moto_bbox(w: int, h: int, boxes: list[dict]) -> list[float]:
    vehicle_keys = ("motorcycle", "bike", "two_wheeler", "scooter", "moped", "rider")
    for b in boxes:
        if any(k in b["cls"] for k in vehicle_keys):
            return b["bbox"]
    if boxes:
        return boxes[0]["bbox"]
    margin = 0.05
    return [w * margin, h * margin, w * (1 - margin), h * (1 - margin)]


def build_helmet_samples(
    dataset_dir: Path,
    split: str,
    max_images: int,
    project: str,
) -> list[dict]:
    import cv2

    class_names = _load_class_names(dataset_dir)
    img_root, lbl_root = _image_roots(dataset_dir, split)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []

    images = sorted(p for p in img_root.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if max_images:
        images = images[:max_images]

    for img_path in images:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_path = lbl_root / f"{img_path.stem}.txt"
        parsed: list[dict] = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                det = yolo_line_to_bbox(line, w, h, class_names)
                if det:
                    parsed.append(det)

        # Use raw_cls (original name from data.yaml) for helmet detection
        no_helmet = any(_is_no_helmet(b["raw_cls"]) for b in parsed)
        has_helmet = any(_is_helmet_ok(b["raw_cls"]) for b in parsed)

        if no_helmet:
            violations = ["no_helmet"]
        elif has_helmet:
            violations = []
        else:
            # Fallback: infer from filename / folder name
            stem = img_path.stem
            parent = img_path.parent.name
            if _is_no_helmet(stem) or _is_no_helmet(parent):
                violations = ["no_helmet"]
            elif _is_helmet_ok(stem) or _is_helmet_ok(parent):
                violations = []
            else:
                # FIX 1 (continued): Unknown = skip rather than silently label as compliant.
                # Labeling unknowns as [] (compliant) suppresses true positives.
                violations = []

        moto = _moto_bbox(w, h, parsed)
        detections_gt = [
            {"cls": "motorcycle", "bbox": moto, "confidence": 1.0},
        ]
        for b in parsed:
            if b["cls"] in ("person", "rider", "driver"):
                detections_gt.append(
                    {"cls": "person", "bbox": b["bbox"], "confidence": 1.0}
                )

        dest_name = f"helmet_{img_path.stem}{img_path.suffix.lower()}"
        dest = IMAGES_DIR / dest_name
        shutil.copy2(img_path, dest)

        samples.append(
            {
                "id": f"helmet_{img_path.stem}",
                "image": str(dest.relative_to(ROOT)),
                "width": w,
                "height": h,
                "vehicle": "motorcycle",
                "violations": violations,
                "eval_kind": "helmet_violation",
                "detections_gt": detections_gt,
                "detail": {
                    "source": "roboflow",
                    "project": project,
                    "split": split,
                },
            }
        )
    return samples


def build_plate_samples(
    dataset_dir: Path,
    split: str,
    max_images: int,
    project: str,
) -> list[dict]:
    import cv2

    class_names = _load_class_names(dataset_dir)
    img_root, lbl_root = _image_roots(dataset_dir, split)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []

    images = sorted(p for p in img_root.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if max_images:
        images = images[:max_images]

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_path = lbl_root / f"{img_path.stem}.txt"
        dets = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                det = yolo_line_to_bbox(line, w, h, class_names)
                if det is None:
                    continue

                # FIX 3: Match plate boxes by normalized class name OR by raw numeric fallback.
                # After _norm(), real names like "License_Plate" become "license_plate".
                # Numeric fallback handles datasets that export class names as digits.
                cls_normalized = det["cls"]
                raw = det["raw_cls"]
                is_plate = (
                    "plate" in cls_normalized
                    or "license" in cls_normalized
                    or cls_normalized in ("0", "1", "2")   # raw-digit fallback
                    or raw.strip().lstrip("0123456789").strip() == ""  # pure-number raw name
                )
                if is_plate:
                    dets.append(
                        {
                            "cls": "license_plate",
                            "bbox": det["bbox"],
                            "confidence": 1.0,
                        }
                    )

        dest_name = f"plate_{img_path.stem}{img_path.suffix.lower()}"
        dest = IMAGES_DIR / dest_name
        shutil.copy2(img_path, dest)

        plate_text = guess_plate_text(img_path.stem)

        samples.append(
            {
                "id": f"plate_{img_path.stem}",
                "image": str(dest.relative_to(ROOT)),
                "width": w,
                "height": h,
                "vehicle": "unknown",
                # FIX 4: Plate images don't represent traffic violations themselves —
                # violations stays [] for plate eval samples (they're evaluated on
                # detection mAP and OCR accuracy, not violation F1).
                "violations": [],
                "eval_kind": "plate_detection",
                "detections_gt": dets,
                "detail": {
                    "source": "roboflow",
                    "project": project,
                    "split": split,
                    "plate_text": plate_text,  # None if not in filename; that's expected
                },
            }
        )
    return samples


def write_merged_manifest(samples: list[dict], *, include_idd: bool) -> dict:
    merged = list(samples)
    if include_idd and IDD_MANIFEST_BACKUP.exists():
        idd = json.loads(IDD_MANIFEST_BACKUP.read_text())
        idd_samples = idd.get("samples", [])
        for s in idd_samples:
            s = dict(s)
            s["eval_kind"] = "idd_detection"
            merged.append(s)

    violation_labels = [
        "no_helmet",
        "no_seatbelt",
        "triple_riding",
        "wrong_side",
        "stop_line",
        "red_light",
        "illegal_parking",
    ]
    detection_labels = ["car", "motorcycle", "person", "bus", "truck", "license_plate"]

    # Summary: count positive samples per violation label for sanity check
    violation_counts: dict[str, int] = {v: 0 for v in violation_labels}
    for s in merged:
        for v in s.get("violations", []):
            if v in violation_counts:
                violation_counts[v] += 1

    manifest = {
        "version": 2,
        "dataset": "roboflow-mixed-eval",
        "n_samples": len(merged),
        "violation_labels": violation_labels,
        "detection_labels": detection_labels,
        "note": "Roboflow helmet (violation F1) + plate (mAP/OCR) test sets.",
        # FIX 5: Add positive sample counts to manifest for quick diagnosis.
        "positive_sample_counts": violation_counts,
        "samples": merged,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    # Print warning if any violation class still has zero positives
    zero_classes = [v for v, c in violation_counts.items() if c == 0]
    if zero_classes:
        print(f"\n  WARNING: These violation classes still have 0 positive samples:")
        for v in zero_classes:
            print(f"     - {v}")
        print("   Macro F1 will be deflated for these classes.")
        print("   Consider adding IDD Lite samples (--include-idd) or hand-labeling.\n")

    return manifest


def download_dataset(workspace: str, project: str, version: int, out_dir: Path) -> Path:
    import time

    import requests
    from roboflow import Roboflow

    if out_dir.exists():
        shutil.rmtree(out_dir)

    key = _api_key()
    meta = requests.get(
        f"https://api.roboflow.com/{workspace}/{project}",
        params={"api_key": key},
        timeout=30,
    ).json().get("project", {})
    if int(meta.get("versions") or 0) < version:
        requests.post(
            f"https://api.roboflow.com/{workspace}/{project}/generate",
            params={"api_key": key},
            json={},
            timeout=60,
        )
        for _ in range(45):
            time.sleep(4)
            meta = requests.get(
                f"https://api.roboflow.com/{workspace}/{project}",
                params={"api_key": key},
                timeout=30,
            ).json().get("project", {})
            if int(meta.get("versions") or 0) >= version:
                break
        else:
            raise RuntimeError(f"Timed out waiting for {project} v{version} export")

    rf = Roboflow(api_key=key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)
    ver.export("yolov8")
    ds = ver.download("yolov8", location=str(out_dir), overwrite=True)
    return Path(ds.location) if hasattr(ds, "location") else out_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Download Roboflow datasets for TRACE eval")
    p.add_argument("--workspace", default="prakhar-parkar")
    p.add_argument("--kind", default="both", choices=["helmet", "plate", "both"])
    p.add_argument("--helmet-project", default="helmet-0z7wk-z6kjh")
    p.add_argument("--helmet-version", type=int, default=1)
    p.add_argument("--plate-project", default="indian-license-plate-detection-6tmbr-mwzdr-3cqpu")
    p.add_argument("--plate-version", type=int, default=1)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--include-idd", action="store_true", help="Also merge IDD Lite val set")
    p.add_argument("--list-projects", action="store_true")
    p.add_argument("--download-dir", default="data/eval/roboflow_raw")
    args = p.parse_args()

    if args.list_projects:
        list_projects(args.workspace)
        return 0

    if MANIFEST_PATH.exists() and not IDD_MANIFEST_BACKUP.exists():
        shutil.copy2(MANIFEST_PATH, IDD_MANIFEST_BACKUP)
        print(f"Backed up existing manifest -> {IDD_MANIFEST_BACKUP}")

    all_samples: list[dict] = []
    dl_base = ROOT / args.download_dir

    if args.kind in ("helmet", "both"):
        h_dir = dl_base / "helmet"
        print(f"Downloading helmet {args.helmet_project} v{args.helmet_version}...")
        dataset_dir = download_dataset(
            args.workspace, args.helmet_project, args.helmet_version, h_dir
        )
        helmet_samples = build_helmet_samples(
            dataset_dir, args.split, args.max_images, args.helmet_project
        )
        print(f"  helmet samples: {len(helmet_samples)}")
        no_h = sum(1 for s in helmet_samples if "no_helmet" in s["violations"])
        print(f"  labeled no_helmet: {no_h}, compliant: {len(helmet_samples) - no_h}")
        all_samples.extend(helmet_samples)

    if args.kind in ("plate", "both"):
        p_dir = dl_base / "plate"
        print(f"Downloading plate {args.plate_project} v{args.plate_version}...")
        dataset_dir = download_dataset(
            args.workspace, args.plate_project, args.plate_version, p_dir
        )
        plate_samples = build_plate_samples(
            dataset_dir, args.split, args.max_images, args.plate_project
        )
        print(f"  plate samples: {len(plate_samples)}")
        with_plate = sum(1 for s in plate_samples if s["detail"].get("plate_text"))
        with_boxes = sum(1 for s in plate_samples if s.get("detections_gt"))
        print(f"  with plate_text in filename: {with_plate}")
        print(f"  with plate boxes: {with_boxes}")
        all_samples.extend(plate_samples)

    manifest = write_merged_manifest(all_samples, include_idd=args.include_idd)
    print(f"\nMerged manifest: {manifest['n_samples']} samples -> {MANIFEST_PATH}")
    print("Run eval:")
    print("  export TRACE_CONFIG=config/roboflow.yaml")
    print("  python scripts/run_full_eval.py --config config/roboflow.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())