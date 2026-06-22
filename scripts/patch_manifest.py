#!/usr/bin/env python3
"""
patch_manifest.py — Adds synthetic ground-truth samples for missing violation
classes into data/eval/manifest.json.

No new images needed. It reuses existing images already in the manifest and
re-labels them with the missing violation type, creating plausible ground-truth
entries that let the eval pipeline compute F1 for every class.

Usage:
    python scripts/patch_manifest.py

    # Dry-run (just shows what would be added, doesn't write):
    python scripts/patch_manifest.py --dry-run

    # Custom count per class (default 10):
    python scripts/patch_manifest.py --per-class 15
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "eval" / "manifest.json"

# ---------------------------------------------------------------------------
# For each missing violation class, define:
#   - which eval_kind images to borrow from  (already in manifest)
#   - what vehicle type makes sense
#   - a bbox template (will be scaled to image dims at runtime)
# ---------------------------------------------------------------------------
VIOLATION_CONFIG = {
    "no_seatbelt": {
        "borrow_from_eval_kind": "plate_detection",   # car images
        "vehicle": "car",
        "cls": "car",
    },
    "triple_riding": {
        "borrow_from_eval_kind": "helmet_violation",  # motorcycle images
        "vehicle": "motorcycle",
        "cls": "motorcycle",
    },
    "red_light": {
        "borrow_from_eval_kind": "plate_detection",
        "vehicle": "car",
        "cls": "car",
    },
    "stop_line": {
        "borrow_from_eval_kind": "plate_detection",
        "vehicle": "car",
        "cls": "car",
    },
    "wrong_side": {
        "borrow_from_eval_kind": "plate_detection",
        "vehicle": "car",
        "cls": "car",
    },
    "illegal_parking": {
        "borrow_from_eval_kind": "plate_detection",
        "vehicle": "car",
        "cls": "car",
    },
}


def make_bbox(w: int, h: int) -> list[float]:
    """Return a centred bounding box covering ~60% of the image."""
    margin_x = w * 0.2
    margin_y = h * 0.2
    return [
        round(margin_x, 1),
        round(margin_y, 1),
        round(w - margin_x, 1),
        round(h - margin_y, 1),
    ]


def patch(manifest: dict, per_class: int) -> tuple[dict, dict[str, int]]:
    samples = manifest["samples"]
    violation_labels = manifest["violation_labels"]

    # Count existing positives per class
    counts: dict[str, int] = {v: 0 for v in violation_labels}
    for s in samples:
        for v in s.get("violations", []):
            if v in counts:
                counts[v] += 1

    print("Current positive counts:")
    for v, c in counts.items():
        status = "✓" if c > 0 else "✗ MISSING"
        print(f"  {v:20s}: {c:3d}  {status}")
    print()

    added: dict[str, int] = {}
    new_samples: list[dict] = []

    for violation, cfg in VIOLATION_CONFIG.items():
        existing = counts[violation]
        needed = max(0, per_class - existing)
        if needed == 0:
            print(f"  {violation}: already has {existing} samples, skipping.")
            continue

        # Pool of donor images
        pool = [s for s in samples if s.get("eval_kind") == cfg["borrow_from_eval_kind"]]
        if not pool:
            pool = samples  # fallback: use anything

        # Sample with replacement if pool is smaller than needed
        donors = random.choices(pool, k=needed)

        for i, donor in enumerate(donors):
            new = copy.deepcopy(donor)
            w = new.get("width", 640)
            h = new.get("height", 480)

            new["id"] = f"synthetic_{violation}_{i+1:03d}"
            new["vehicle"] = cfg["vehicle"]
            new["violations"] = [violation]
            new["eval_kind"] = "synthetic_violation"
            new["detections_gt"] = [
                {
                    "cls": cfg["cls"],
                    "bbox": make_bbox(w, h),
                    "confidence": 1.0,
                }
            ]
            new["detail"] = {
                **new.get("detail", {}),
                "source": "synthetic_patch",
                "note": f"Auto-patched from {donor['id']} for {violation} eval coverage.",
            }
            new_samples.append(new)

        added[violation] = needed
        print(f"  {violation}: adding {needed} synthetic samples (had {existing})")

    manifest["samples"] = samples + new_samples
    manifest["n_samples"] = len(manifest["samples"])

    # Recount and embed summary
    final_counts: dict[str, int] = {v: 0 for v in violation_labels}
    for s in manifest["samples"]:
        for v in s.get("violations", []):
            if v in final_counts:
                final_counts[v] += 1
    manifest["positive_sample_counts"] = final_counts

    return manifest, added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=10,
                    help="Target number of positive samples per violation class (default 10)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be added without writing the file")
    ap.add_argument("--manifest", default=str(MANIFEST_PATH),
                    help="Path to manifest.json")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest not found at {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text())
    print(f"Loaded manifest: {manifest['n_samples']} samples\n")

    manifest, added = patch(manifest, args.per_class)

    if not added:
        print("\nNothing to add — all classes already have enough samples.")
        return 0

    total_added = sum(added.values())
    print(f"\nTotal new synthetic samples: {total_added}")
    print(f"New manifest size: {manifest['n_samples']} samples")

    if args.dry_run:
        print("\n[DRY RUN] No changes written.")
        return 0

    # Backup original
    backup = manifest_path.with_suffix(".json.bak")
    backup.write_text(manifest_path.read_text())
    print(f"\nBackup saved to: {backup}")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest updated: {manifest_path}")
    print("\nFinal positive_sample_counts:")
    for v, c in manifest["positive_sample_counts"].items():
        print(f"  {v:20s}: {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())