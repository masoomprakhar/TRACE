#!/usr/bin/env python3
"""Generate a SYNTHETIC helmet/no_helmet classification dataset.

This is a no-dataset smoke-test path: it lets you exercise the full helmet
training pipeline (and produce a loadable models/weights/helmet.pt) without
downloading anything. A model trained on this synthetic data classifies
synthetic rider-head crops well but will NOT generalize to real photos — for
production, point train_helmet.py at a real Roboflow/Kaggle helmet dataset
(same command, real images).

Output (ImageFolder layout expected by train_helmet.py --mode cls):
  <out>/train/helmet/*.jpg   <out>/train/no_helmet/*.jpg
  <out>/val/helmet/*.jpg     <out>/val/no_helmet/*.jpg

Usage:
  python training/make_synthetic_helmet.py --out data/datasets/helmet_cls \\
      --n-train 350 --n-val 80
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

_SKIN = [(180, 160, 150), (150, 130, 120), (120, 100, 90), (90, 70, 60), (200, 180, 170)]
_HAIR = [(20, 20, 20), (35, 25, 20), (60, 45, 35), (25, 25, 30)]
_HELMET = [(40, 40, 220), (220, 120, 40), (240, 240, 240), (30, 30, 30),
           (40, 200, 220), (60, 180, 60)]


def _bg(size: int, rng: random.Random) -> np.ndarray:
    g = rng.randint(55, 150)
    base = np.full((size, size, 3), (g, g, g + rng.randint(-10, 10)), np.uint8)
    noise = (np.random.randn(size, size, 3) * 9).astype(np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def render(helmet: bool, rng: random.Random, size: int = 160) -> np.ndarray:
    img = _bg(size, rng)
    cx = size // 2 + rng.randint(-12, 12)
    cy = size // 2 + rng.randint(-4, 14)
    rw = rng.randint(38, 52)
    rh = rng.randint(48, 64)
    skin = rng.choice(_SKIN)

    cv2.ellipse(img, (cx, cy), (rw, rh), 0, 0, 360, skin, -1)            # head
    ey = cy + rng.randint(0, 8)
    cv2.circle(img, (cx - rw // 3, ey), 3, (25, 25, 25), -1)             # eyes
    cv2.circle(img, (cx + rw // 3, ey), 3, (25, 25, 25), -1)

    if helmet:
        col = rng.choice(_HELMET)
        cv2.ellipse(img, (cx, cy - rh // 6), (rw + 6, rh - 4), 0, 180, 360, col, -1)
        cv2.line(img, (cx - rw - 4, cy - rh // 6), (cx + rw + 4, cy - rh // 6),
                 (20, 20, 20), 2)                                        # visor line
        cv2.ellipse(img, (cx - rw // 3, cy - rh // 2), (7, 13), 0, 0, 360,
                    (255, 255, 255), -1)                                 # specular
        cv2.line(img, (cx - rw, cy + rh // 3), (cx, cy + rh), (25, 25, 25), 2)
        cv2.line(img, (cx + rw, cy + rh // 3), (cx, cy + rh), (25, 25, 25), 2)
    else:
        hair = rng.choice(_HAIR)
        cv2.ellipse(img, (cx, cy - rh // 5), (rw, rh - rh // 4), 0, 180, 360, hair, -1)
        for _ in range(rng.randint(12, 26)):                            # strands
            x = cx + rng.randint(-rw, rw)
            cv2.line(img, (x, cy - rh), (x + rng.randint(-4, 4), cy - rh // 3),
                     hair, 1)

    if rng.random() < 0.6:
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3),
                                  beta=rng.randint(-25, 25))
    if rng.random() < 0.5:
        m = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-12, 12), 1.0)
        img = cv2.warpAffine(img, m, (size, size), borderMode=cv2.BORDER_REPLICATE)
    return img


def _emit(root: Path, split: str, cls: str, n: int, helmet: bool, rng):
    d = root / split / cls
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(d / f"{cls}_{i:04d}.jpg"), render(helmet, rng))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Synthetic helmet dataset generator")
    p.add_argument("--out", default="data/datasets/helmet_cls")
    p.add_argument("--n-train", type=int, default=350)
    p.add_argument("--n-val", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    root = Path(args.out)
    _emit(root, "train", "helmet", args.n_train, True, rng)
    _emit(root, "train", "no_helmet", args.n_train, False, rng)
    _emit(root, "val", "helmet", args.n_val, True, rng)
    _emit(root, "val", "no_helmet", args.n_val, False, rng)
    total = 2 * (args.n_train + args.n_val)
    print(f"wrote {total} images to {root}/ (train+val, helmet/no_helmet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
