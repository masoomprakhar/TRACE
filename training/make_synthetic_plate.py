"""Generate synthetic Indian license-plate crops in YOLO detection format.

Output layout (--out directory):
  images/train/*.jpg   images/val/*.jpg
  labels/train/*.txt   labels/val/*.txt   (0 cx cy w h  normalised)
  dataset.yaml

Smoke-test usage:
  python training/make_synthetic_plate.py --out data/datasets/plate_det
  python training/train_plate.py --data data/datasets/plate_det/dataset.yaml \
      --weights yolov8n.pt --epochs 12 --imgsz 320 --batch 16 --device cpu
"""
from __future__ import annotations

import argparse
import random
import textwrap
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── Indian plate rendering ────────────────────────────────────────────────────

_STATES = ["MH", "DL", "KA", "TN", "UP", "GJ", "RJ", "WB", "MP", "AP"]
_ALPHA = "ABCDEFGHJKLMNPRSTUVWXYZ"  # avoid I, O, Q (look like digits)
_DIGITS = "0123456789"


def _rand_plate_text(rng: random.Random) -> str:
    state = rng.choice(_STATES)
    dist = f"{rng.randint(1, 99):02d}"
    series = "".join(rng.choices(_ALPHA, k=rng.randint(1, 3)))
    num = f"{rng.randint(1, 9999):04d}"
    return f"{state}{dist}{series}{num}"


def render_plate(rng: random.Random) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return (scene_img 320×240, plate_xyxy) with a synthetic Indian plate."""
    W, H = 320, 240

    # ── background: blurry vehicle body colour ──────────────────────────────
    bg_hue = rng.randint(0, 179)
    bg = np.full((H, W, 3), 128, np.uint8)
    bg[:] = cv2.cvtColor(
        np.array([[[bg_hue, rng.randint(80, 200), rng.randint(100, 220)]]], np.uint8),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    # add a soft gradient + noise so detector can't cheat on solid colour
    grad = np.linspace(0.7, 1.15, W, dtype=np.float32)
    bg = np.clip(bg * grad[np.newaxis, :, np.newaxis], 0, 255).astype(np.uint8)
    noise = rng.randint(-18, 18)
    bg = np.clip(bg.astype(np.int16) + rng.randint(-12, 12), 0, 255).astype(np.uint8)

    # ── plate position: lower-centre of vehicle crop ─────────────────────────
    pw = rng.randint(110, 160)  # plate width
    ph = rng.randint(30, 46)    # plate height
    cx = rng.randint(W // 4, 3 * W // 4)
    cy = rng.randint(int(0.52 * H), int(0.82 * H))
    x1 = max(4, cx - pw // 2)
    y1 = max(4, cy - ph // 2)
    x2 = min(W - 4, x1 + pw)
    y2 = min(H - 4, y1 + ph)
    pw, ph = x2 - x1, y2 - y1

    # ── draw plate ────────────────────────────────────────────────────────────
    white_plate = rng.random() > 0.3  # white plate (private) vs yellow (commercial)
    plate_bg = (255, 255, 255) if white_plate else (0, 220, 255)
    plate_txt = (0, 0, 0) if white_plate else (0, 0, 0)

    cv2.rectangle(bg, (x1, y1), (x2, y2), (50, 50, 50), 2)  # border shadow
    cv2.rectangle(bg, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), plate_bg, -1)

    text = _rand_plate_text(rng)
    font = cv2.FONT_HERSHEY_DUPLEX
    fs = ph * 0.024
    thickness = max(1, int(ph * 0.055))
    (tw, th), _ = cv2.getTextSize(text, font, fs, thickness)
    # shrink font until text fits
    while tw > pw - 6 and fs > 0.3:
        fs *= 0.9
        (tw, th), _ = cv2.getTextSize(text, font, fs, thickness)
    tx = x1 + (pw - tw) // 2
    ty = y1 + (ph + th) // 2 - 2
    cv2.putText(bg, text, (tx, ty), font, fs, plate_txt, thickness, cv2.LINE_AA)

    # thin blue strip at top (Indian plate indicator, private)
    if white_plate:
        cv2.rectangle(bg, (x1 + 2, y1 + 2), (x2 - 2, y1 + 5), (180, 80, 0), -1)

    # ── random mild augmentations ─────────────────────────────────────────────
    if rng.random() > 0.5:
        alpha = rng.uniform(0.75, 1.25)
        bg = np.clip(bg.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    if rng.random() > 0.6:
        bg = cv2.GaussianBlur(bg, (3, 3), 0)

    return bg, (x1, y1, x2, y2)


# ── dataset generation ────────────────────────────────────────────────────────

def generate(out_dir: Path, n_train: int = 300, n_val: int = 60,
             seed: int = 42) -> Path:
    rng = random.Random(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, (x1, y1, x2, y2) = render_plate(rng)
            H, W = img.shape[:2]
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            stem = f"plate_{i:05d}"
            cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out_dir / "labels" / split / f"{stem}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            )

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml.dump({
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["license_plate"],
    }, default_flow_style=False))

    print(f"[make_synthetic_plate] {n_train} train + {n_val} val images → {out_dir}")
    print(f"  dataset YAML: {yaml_path}")
    return yaml_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    ap = argparse.ArgumentParser(
        description="Generate synthetic Indian-plate YOLO detection dataset.")
    ap.add_argument("--out", default="data/datasets/plate_det",
                    help="Output directory for the dataset")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(Path(args.out), args.n_train, args.n_val, args.seed)


if __name__ == "__main__":
    _cli()
