#!/usr/bin/env python3
"""Albumentations weather/quality augmentation for TRACE training data.

Indian traffic footage is dominated by hard frames: rain, haze, harsh
shadows, motion blur, low light, sensor noise. This module builds an
Albumentations pipeline that simulates those conditions so helmet / seatbelt
/ plate models generalise to them — especially useful to amplify rare classes
(e.g. night triple-riding) when paired with copy-paste / synthetic crops.

Use as a library (recommended) — wrap it into your dataset/Dataloader:
    from training.augment import build_transform
    tf = build_transform(p=0.5)
    out = tf(image=img)["image"]

Or as a CLI to materialise an augmented copy of an image folder, or just
preview a grid:
    python training/augment.py --src data/raw/helmet --dst data/aug/helmet -n 3
    python training/augment.py --src data/raw/helmet --preview out.jpg

NOTE: this transforms images only (no bbox/label remap), so it is safe for
classification folders as-is. For *detection* sets, drive these transforms
through Ultralytics' Albumentations hook or an Albumentations Compose with
`bbox_params` so boxes are transformed alongside the pixels.

Requires the ML extras:  pip install -r requirements-ml.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_INSTALL_HINT = (
    "Albumentations is not installed. Install the ML extras first:\n"
    "    pip install -r requirements-ml.txt"
)


def build_transform(p: float = 0.5) -> Any:
    """Return an Albumentations Compose simulating road/weather conditions.

    ``p`` scales how often each effect fires. Effects: rain, fog, shadow,
    motion blur, brightness/contrast jitter and Gaussian noise. Imported
    lazily so this module stays importable without the ML stack.
    """
    try:
        import albumentations as A  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on env
        print(_INSTALL_HINT, file=sys.stderr)
        raise SystemExit(2) from exc

    return A.Compose(
        [
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=18,
                blur_value=3,
                brightness_coefficient=0.9,
                p=p,
            ),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, alpha_coef=0.08, p=p),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, p=p),
            A.MotionBlur(blur_limit=7, p=p),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=p
            ),
            A.GaussNoise(var_limit=(10.0, 60.0), p=p),
        ]
    )


def _iter_images(src: Path):
    for path in sorted(src.rglob("*")):
        if path.suffix.lower() in _IMG_EXTS:
            yield path


def _save_preview(cv2, images: list, dest: Path) -> None:
    """Stack a few sample augmentations vertically into one preview image."""
    import numpy as np  # noqa: PLC0415

    width = min(img.shape[1] for img in images)
    rows = [
        cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
        for img in images
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), np.vstack(rows))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Preview or apply Albumentations augmentation to an image folder.",
    )
    p.add_argument("--src", required=True, help="source image folder (recursed)")
    p.add_argument("--dst", help="output folder for augmented copies")
    p.add_argument("--preview", help="write a single stacked preview image here")
    p.add_argument(
        "-n",
        "--per-image",
        type=int,
        default=1,
        help="augmented variants to write per source image (with --dst)",
    )
    p.add_argument("--p", type=float, default=0.5, help="effect probability")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not args.dst and not args.preview:
        print("error: pass --dst (apply) and/or --preview.", file=sys.stderr)
        return 1

    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        print("error: opencv is required (pip install -r requirements.txt).",
              file=sys.stderr)
        return 2

    import random  # noqa: PLC0415

    random.seed(args.seed)

    src = Path(args.src)
    if not src.is_dir():
        print(f"error: not a directory: {src}", file=sys.stderr)
        return 1

    transform = build_transform(p=args.p)
    images = list(_iter_images(src))
    if not images:
        print(f"error: no images found under {src}", file=sys.stderr)
        return 1

    preview_samples: list = []
    written = 0
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        for i in range(args.per_image):
            aug = transform(image=img)["image"]
            if args.dst:
                rel = path.relative_to(src)
                out = Path(args.dst) / rel.with_name(f"{rel.stem}_aug{i}{rel.suffix}")
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), aug)
                written += 1
            if args.preview and len(preview_samples) < 6:
                preview_samples.append(aug)

    if args.dst:
        print(f"wrote {written} augmented images under {args.dst}")
    if args.preview and preview_samples:
        _save_preview(cv2, preview_samples, Path(args.preview))
        print(f"wrote preview: {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
