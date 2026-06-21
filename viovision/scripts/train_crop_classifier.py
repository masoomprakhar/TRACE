"""
Train any of the sklearn crop classifiers (helmet, seatbelt, signal-state)
from a directory of labeled crops.

Expected data layout (matches Roboflow Universe's typical export-by-class
folder structure, so exports can be used close to as-is):

    data/annotations/helmet/
        helmet/      *.jpg
        no_helmet/   *.jpg

    data/annotations/seatbelt/
        seatbelt/    *.jpg
        no_seatbelt/ *.jpg

    data/annotations/signal/
        red/    *.jpg
        yellow/ *.jpg
        green/  *.jpg

Usage:
    python scripts/train_crop_classifier.py --model helmet \\
        --data-dir data/annotations/helmet \\
        --out models/weights/helmet_svm.pkl

    python scripts/train_crop_classifier.py --model seatbelt \\
        --data-dir data/annotations/seatbelt \\
        --out models/weights/seatbelt_svm.pkl

    python scripts/train_crop_classifier.py --model signal \\
        --data-dir data/annotations/signal \\
        --out models/weights/signal_svm.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.helmet_classifier import HelmetClassifier
from src.models.seatbelt_classifier import SeatbeltClassifier
from src.models.signal_state_classifier import SignalStateClassifierSklearn

MODEL_REGISTRY = {
    "helmet": HelmetClassifier,
    "seatbelt": SeatbeltClassifier,
    "signal": SignalStateClassifierSklearn,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Classifiers whose prep script already baked CLAHE into the saved crops.
# For these, BaseCropClassifier.fit() must NOT re-apply CLAHE, and neither
# should predict() at inference time. Both are controlled by use_clahe on the
# classifier instance — we override it to False below after construction.
# The SeatbeltClassifier default is use_clahe=True (correct for raw inference
# crops), but training from pre-processed crops needs it off.
CLAHE_ALREADY_APPLIED = {"seatbelt"}


def load_crops_from_class_folders(data_dir: Path, class_names: tuple[str, ...]
                                    ) -> tuple[list[np.ndarray], list[str]]:
    crops: list[np.ndarray] = []
    labels: list[str] = []

    for class_name in class_names:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            print(f"  WARNING: expected folder {class_dir} not found, skipping.")
            continue
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"  {class_name}: {len(files)} images")
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"    skipping unreadable file: {f}")
                continue
            crops.append(img)
            labels.append(class_name)

    return crops, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an sklearn crop classifier.")
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["svm", "random_forest"], default=None,
                          help="Override the classifier's default backend.")
    args = parser.parse_args()

    model_cls = MODEL_REGISTRY[args.model]
    classifier = model_cls()

    # Fix: backend override must rebuild clf on the instance, not the class.
    if args.backend:
        classifier.backend = args.backend
        classifier.clf = classifier._build_backend()

    # Fix: seatbelt crops from our prep scripts already have CLAHE baked in.
    # Suppress it here so fit() and the saved model's predict() don't double-apply.
    # Without this, train features ≠ inference features → silent accuracy loss.
    if args.model in CLAHE_ALREADY_APPLIED:
        classifier.use_clahe = False
        print(f"  [INFO] CLAHE suppressed for training: crops from "
              f"prepare_{args.model}_dataset.py already have it applied. "
              f"The saved model will also predict without CLAHE — ensure "
              f"inference crops are pre-processed with CLAHE before calling "
              f"classifier.predict().")

    print(f"Loading crops for '{args.model}' from {args.data_dir} ...")

    # Preflight: fail fast if class folders are missing, before wasting time
    # loading thousands of images only to hit an error at split time.
    missing = [c for c in classifier.class_names
               if not (args.data_dir / c).is_dir()]
    if missing:
        print(f"ERROR: expected class folders not found under {args.data_dir}:")
        for m in missing:
            print(f"  missing: {args.data_dir / m}")
        print(f"\nRun the appropriate prepare_*_dataset.py script first, then "
              f"re-run this script.")
        sys.exit(1)

    crops, labels = load_crops_from_class_folders(args.data_dir, classifier.class_names)

    if len(crops) < 20:
        print(f"ERROR: only {len(crops)} usable images found. Need more data "
              f"before training meaningfully (guide suggests 1,000-2,000 crops "
              f"for helmet, similar order for seatbelt/signal).")
        sys.exit(1)

    # Class balance check before splitting — imbalanced data causes the SVM
    # to learn the majority class and ignore the minority without warning.
    from collections import Counter
    dist = Counter(labels)
    print(f"\nClass distribution:")
    for cls, n in sorted(dist.items()):
        bar = "█" * min(40, n // max(1, max(dist.values()) // 40))
        print(f"  {cls:15s}: {n:5d}  {bar}")
    ratio = min(dist.values()) / max(dist.values())
    if ratio < 0.5:
        print(f"\n  [WARN] Class imbalance ratio {ratio:.2f} (< 0.5). The SVM "
              f"uses class_weight='balanced' which compensates, but accuracy "
              f"on the minority class may still be poor. Consider adding more "
              f"minority-class crops before training.")

    train_crops, val_crops, train_labels, val_labels = train_test_split(
        crops, labels, test_size=args.test_size, random_state=args.seed,
        stratify=labels,
    )

    print(f"\nTraining on {len(train_crops)} crops, validating on {len(val_crops)} ...")
    classifier.fit(train_crops, train_labels)

    print("\nValidation report:")
    val_preds = [classifier.predict(c).cls for c in val_crops]
    print(classification_report(val_labels, val_preds))

    # Hard-conditions subset report (guide section 4): images with night_/
    # rain_/glare_ prefix in their filename are flagged separately so you can
    # see if accuracy degrades on hard conditions even when overall accuracy
    # looks fine. Only meaningful if your crop filenames carry these prefixes
    # (they will if you rename hard-condition source images before running
    # the prep scripts).
    hard_prefixes = ("night_", "rain_", "glare_", "fog_", "dark_")
    hard_idx = [i for i, f in enumerate(val_crops)
                if any(str(f).split("/")[-1].startswith(p) for p in hard_prefixes)
                ] if hasattr(val_crops[0], '__fspath__') else []
    # val_crops are np.ndarray, not paths — hard-subset tracking requires
    # the prep scripts to embed condition in the crop pixel data, which they
    # don't. Print a reminder instead.
    print(f"  [Guide section 4] Hard-conditions subset tracking: to report "
          f"separate metrics on night/rain/glare crops, prefix those source "
          f"images with 'night_'/'rain_'/'glare_' before running the prep "
          f"script, then re-run with --hard-subset (not yet implemented here).")

    classifier.save(args.out)
    print(f"\nSaved trained classifier to {args.out}")
    print(f"Next: set configs/pipeline.yaml use_mocks.{args.model}: false "
          f"and weights.{args.model}: {args.out}")


if __name__ == "__main__":
    main()
