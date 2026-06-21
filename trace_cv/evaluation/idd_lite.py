"""Import IIIT Hyderabad IDD Lite (idd20k_lite) into TRACE eval format.

IDD Lite uses 7 semantic classes in ``*_label.png`` (grayscale trainIds 0–6):
  0 drivable, 1 non-drivable, 2 living-thing, 3 vehicles,
  4 roadside, 5 construction, 6 vegetation/sky

We convert class masks to bounding boxes for detection mAP evaluation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from trace_cv.core.types import ViolationType

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = _REPO / "data" / "datasets" / "idd20k_lite"
EVAL_IMAGES = _REPO / "data" / "eval" / "images"
MANIFEST_PATH = _REPO / "data" / "eval" / "manifest.json"

# IDD Lite level-1 trainIds -> TRACE detection class
_PERSON_ID = 2
_VEHICLE_ID = 3

DETECTION_LABELS = ["car", "motorcycle", "person", "bus", "truck"]


def _vehicle_cls(w: int, h: int) -> str:
    """Heuristic: wide blobs → car, narrow → motorcycle (IDD lumps all in class 3)."""
    if h <= 0:
        return "car"
    ratio = w / h
    if ratio < 0.85 and h > 40:
        return "motorcycle"
    if ratio > 2.2 and h > 30:
        return "truck"
    return "car"


def mask_to_detections(
    label: np.ndarray,
    *,
    min_area: int = 250,
) -> list[dict]:
    """Extract detection GT boxes from an IDD Lite semantic label mask."""
    out: list[dict] = []
    for class_id in (_PERSON_ID, _VEHICLE_ID):
        binary = (label == class_id).astype(np.uint8)
        if binary.sum() == 0:
            continue
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            if class_id == _PERSON_ID:
                cls = "person"
            else:
                cls = _vehicle_cls(int(w), int(h))
            out.append(
                {
                    "cls": cls,
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "confidence": 1.0,
                }
            )
    return out


def _find_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    img_root = root / "leftImg8bit" / split
    lbl_root = root / "gtFine" / split
    if not img_root.exists() or not lbl_root.exists():
        return pairs
    for img in sorted(img_root.rglob("*_image.jpg")):
        stem = img.name.replace("_image.jpg", "")
        lbl = lbl_root / img.parent.name / f"{stem}_label.png"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def build_manifest(
    root: Path | None = None,
    *,
    split: str = "val",
    max_images: int | None = None,
    copy_images: bool = True,
    out_manifest: Path | None = None,
) -> dict:
    root = root or DEFAULT_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"IDD Lite not found at {root}. Extract idd-lite.tar.gz:\n"
            f"  mkdir -p data/datasets && tar -xzf ~/Downloads/idd-lite.tar.gz -C data/datasets/"
        )

    pairs = _find_pairs(root, split)
    if max_images:
        pairs = pairs[:max_images]

    if copy_images:
        EVAL_IMAGES.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for img_path, lbl_path in pairs:
        image = cv2.imread(str(img_path))
        label = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
        if image is None or label is None:
            continue
        h, w = image.shape[:2]
        dets = mask_to_detections(label)

        sample_id = f"idd_{split}_{img_path.parent.name}_{img_path.stem.replace('_image','')}"
        rel_image = f"data/eval/images/{sample_id}.jpg"
        dest = _REPO / rel_image
        if copy_images:
            shutil.copy2(img_path, dest)

        samples.append(
            {
                "id": sample_id,
                "image": rel_image,
                "width": w,
                "height": h,
                "vehicle": "unknown",
                "violations": [],
                "detections_gt": dets,
                "detail": {
                    "source": "idd-lite",
                    "split": split,
                    "sequence": img_path.parent.name,
                    "original": str(img_path.relative_to(root)),
                },
            }
        )

    manifest = {
        "version": 1,
        "dataset": "idd-lite-iith",
        "split": split,
        "n_samples": len(samples),
        "violation_labels": [v.value for v in ViolationType],
        "detection_labels": DETECTION_LABELS,
        "note": (
            "IDD Lite provides semantic masks only (no violation labels). "
            "violations[] is empty; use for detection mAP and pipeline preprocessing eval. "
            "Combine with Roboflow helmet/plate sets for violation F1."
        ),
        "samples": samples,
    }

    out = out_manifest or MANIFEST_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    return manifest


def extract_archive(tar_path: Path, dest: Path | None = None) -> Path:
    """Extract idd-lite.tar.gz to data/datasets/."""
    import tarfile

    dest = dest or DEFAULT_ROOT.parent
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest)
    root = dest / "idd20k_lite"
    if not root.exists():
        raise FileNotFoundError(f"Expected idd20k_lite/ inside {tar_path}")
    return root
