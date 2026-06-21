#!/usr/bin/env python3
"""Generate a SYNTHETIC seatbelt classification dataset (belt/no_belt/occluded).

Same purpose and caveat as make_synthetic_helmet.py: a no-dataset smoke-test
path to exercise the seatbelt training pipeline and produce a loadable
models/weights/seatbelt.pt. Trained on synthetic driver-window crops it will
NOT generalize to real photos — use a real Roboflow seatbelt dataset for
production. The `occluded` class is what teaches the model to abstain on glare
/ blocked views (TRACE never flags an occluded driver).

Output (ImageFolder layout expected by train_seatbelt.py --mode cls):
  <out>/{train,val}/{belt,no_belt,occluded}/*.jpg

Usage:
  python training/make_synthetic_seatbelt.py --out data/datasets/seatbelt_cls \\
      --n-train 320 --n-val 70
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

_SEAT = [(45, 45, 55), (60, 55, 50), (80, 75, 70), (35, 40, 60), (55, 50, 48)]
_SHIRT = [(60, 60, 180), (180, 140, 60), (70, 140, 70), (200, 200, 200), (50, 50, 50)]
_SKIN = [(180, 160, 150), (150, 130, 120), (120, 100, 90), (200, 180, 170)]
_STRAP = [(25, 25, 25), (40, 40, 45), (20, 30, 60)]


def _bg(size: int, rng: random.Random) -> np.ndarray:
    seat = rng.choice(_SEAT)
    img = np.full((size, size, 3), seat, np.uint8)
    noise = (np.random.randn(size, size, 3) * 8).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # window pillar on one side
    if rng.random() < 0.6:
        x = rng.choice([0, size - rng.randint(12, 26)])
        cv2.rectangle(img, (x, 0), (x + rng.randint(12, 26), size), (20, 20, 25), -1)
    return img


def render(label: str, rng: random.Random, size: int = 160) -> np.ndarray:
    img = _bg(size, rng)
    cx = size // 2 + rng.randint(-12, 12)
    top = int(size * 0.34) + rng.randint(-6, 6)
    shirt = rng.choice(_SHIRT)

    # torso (shoulders -> bottom) and head
    cv2.rectangle(img, (cx - 42, top + 8), (cx + 42, size), shirt, -1)
    cv2.ellipse(img, (cx, top + 6), (44, 26), 0, 180, 360, shirt, -1)  # shoulders
    cv2.ellipse(img, (cx, top - 18), (20, 24), 0, 0, 360, rng.choice(_SKIN), -1)

    if label == "belt":
        p1 = (cx - 34, top + 4)
        p2 = (cx + 28, size - 4)
        col = rng.choice(_STRAP)
        cv2.line(img, p1, p2, col, rng.randint(7, 10))
        cv2.line(img, p1, p2, (200, 200, 200), 1)  # stitch highlight
    elif label == "no_belt":
        pass  # torso only
    else:  # occluded — wash out the chest with glare / reflection
        overlay = img.copy()
        gx = cx + rng.randint(-20, 20)
        cv2.ellipse(overlay, (gx, int(size * 0.6)),
                    (rng.randint(55, 80), rng.randint(45, 70)), 0, 0, 360,
                    (255, 255, 255), -1)
        for _ in range(rng.randint(2, 4)):  # vertical reflection streaks
            sx = rng.randint(20, size - 20)
            cv2.line(overlay, (sx, 0), (sx + rng.randint(-10, 10), size),
                     (245, 245, 245), rng.randint(6, 14))
        cv2.addWeighted(overlay, rng.uniform(0.5, 0.7), img,
                        rng.uniform(0.3, 0.5), 0, img)

    if rng.random() < 0.6:
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.75, 1.25),
                                  beta=rng.randint(-20, 20))
    return img


def _emit(root: Path, split: str, cls: str, n: int, rng: random.Random):
    d = root / split / cls
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(d / f"{cls}_{i:04d}.jpg"), render(cls, rng))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Synthetic seatbelt dataset generator")
    p.add_argument("--out", default="data/datasets/seatbelt_cls")
    p.add_argument("--n-train", type=int, default=320)
    p.add_argument("--n-val", type=int, default=70)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    root = Path(args.out)
    for cls in ("belt", "no_belt", "occluded"):
        _emit(root, "train", cls, args.n_train, rng)
        _emit(root, "val", cls, args.n_val, rng)
    total = 3 * (args.n_train + args.n_val)
    print(f"wrote {total} images to {root}/ (belt/no_belt/occluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
