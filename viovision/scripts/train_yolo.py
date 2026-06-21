"""
Fine-tune YOLOv11n on the traffic dataset. Thin wrapper around the
Ultralytics training call so hyperparameters live in one reviewable place
instead of being buried in a shell command.

Guide section 2.A recipe this implements:
    yolo detect train model=yolo11n.pt data=traffic.yaml epochs=80 \\
        imgsz=960 batch=16

Usage:
    python scripts/train_yolo.py --data configs/traffic.yaml --epochs 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_CLASSES = {0: "car", 1: "two_wheeler", 2: "person",
                    3: "license_plate", 4: "windshield", 5: "signal_light"}


def preflight_check(data_yaml: Path) -> None:
    """
    Validate data/splits structure and class schema before spending GPU time.
    Fails fast with a clear message rather than dying 10 minutes into training.
    """
    import yaml
    if not data_yaml.exists():
        print(f"ERROR: data yaml not found: {data_yaml}")
        sys.exit(1)

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    yaml_root = Path(cfg.get("path", data_yaml.parent.parent))
    if not yaml_root.is_absolute():
        yaml_root = data_yaml.parent / yaml_root

    errors: list[str] = []

    # Check split directories exist and are non-empty
    for split_key in ("train", "val"):
        rel = cfg.get(split_key, "")
        split_img_dir = yaml_root / rel
        if not split_img_dir.is_dir():
            errors.append(f"Missing split dir: {split_img_dir}")
            continue
        imgs = [f for f in split_img_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        if not imgs:
            errors.append(f"Split dir is empty: {split_img_dir}")
        else:
            lbl_dir = split_img_dir.parent.parent / "labels" / split_img_dir.name
            if not lbl_dir.is_dir():
                # Try sibling labels/ dir (common Roboflow layout)
                lbl_dir = split_img_dir.parent / "labels"
            if not lbl_dir.is_dir():
                errors.append(f"No labels dir found alongside {split_img_dir}")
            else:
                lbls = list(lbl_dir.glob("*.txt"))
                print(f"  [{split_key:5s}] {len(imgs):5d} images, "
                      f"{len(lbls):5d} labels in {split_img_dir}")

    # Check class names match expected schema
    names = cfg.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    for cid, expected in REQUIRED_CLASSES.items():
        actual = names.get(cid, names.get(str(cid)))
        if actual is None:
            errors.append(f"Class id {cid} ({expected}) missing from data.yaml names")
        elif actual != expected:
            errors.append(f"Class id {cid}: expected '{expected}', got '{actual}'. "
                          f"Fix traffic.yaml or re-run prepare scripts with correct "
                          f"class ordering.")

    if errors:
        print("\nPreflight FAILED — fix these before training:")
        for e in errors:
            print(f"  ✗ {e}")
        print("\nRun scripts/merge_detector_datasets.py for a class balance report.")
        sys.exit(1)

    print("Preflight passed.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11n on traffic.yaml")
    parser.add_argument("--data", type=Path, default=Path("configs/traffic.yaml"))
    parser.add_argument("--base-weights", default="yolo11n.pt",
                        help="Start from COCO-pretrained weights, never from scratch "
                             "(guide section 2.A).")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960,
                        help="960+ matters for small plates/distant vehicles. "
                             "Drop to 640 only if GPU time runs out (guide section 7).")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0,
                        help="Freeze early backbone layers for N epochs if your "
                             "fine-tuning set is small (guide section 2.A), then "
                             "unfreeze for the remainder. Implemented as two "
                             "separate model.train() calls — NOT resume=True, "
                             "which does not support unfreezing mid-run.")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="viovision_yolo11n")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip data directory validation (use if you know "
                             "the splits are correct and want to skip the check).")
    args = parser.parse_args()

    if not args.skip_preflight:
        print("Running preflight checks ...")
        preflight_check(args.data)

    from ultralytics import YOLO

    base_train_kwargs = dict(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        # Mosaic + mixup stay on (YOLO defaults, guide section 5).
    )

    if args.freeze_backbone_epochs > 0:
        remaining_epochs = args.epochs - args.freeze_backbone_epochs
        if remaining_epochs <= 0:
            print(f"ERROR: --freeze-backbone-epochs ({args.freeze_backbone_epochs}) "
                  f"must be less than --epochs ({args.epochs}).")
            sys.exit(1)

        # Phase 1: frozen backbone warmup.
        # freeze=list(range(10)) locks the first 10 layers of YOLOv11n,
        # covering most of the early backbone feature extractor.
        print(f"Phase 1: {args.freeze_backbone_epochs} epochs with backbone frozen ...")
        phase1_name = args.name + "_phase1"
        model = YOLO(args.base_weights)
        model.train(
            name=phase1_name,
            epochs=args.freeze_backbone_epochs,
            freeze=list(range(10)),
            **base_train_kwargs,
        )

        # Phase 2: full fine-tune from phase 1's best weights.
        # Load best.pt from the completed phase 1 run — NOT resume=True.
        # resume=True continues the same frozen run; loading best.pt starts
        # a fresh unfrozen run from the best checkpoint, which is correct.
        phase1_weights = (Path(args.project) / phase1_name / "weights" / "best.pt")
        if not phase1_weights.exists():
            # Fallback to last.pt if best.pt wasn't saved (early stopping off)
            phase1_weights = phase1_weights.parent / "last.pt"
        if not phase1_weights.exists():
            print(f"ERROR: phase 1 weights not found at {phase1_weights}. "
                  f"Check {args.project}/{phase1_name}/weights/")
            sys.exit(1)

        print(f"\nPhase 2: {remaining_epochs} epochs unfrozen from {phase1_weights} ...")
        model2 = YOLO(str(phase1_weights))
        model2.train(
            name=args.name,
            epochs=remaining_epochs,
            # No freeze= here — all layers trainable
            **base_train_kwargs,
        )
        print(f"\nDone. Final weights under {args.project}/{args.name}/weights/")

    else:
        # Single-phase training — straightforward fine-tune.
        model = YOLO(args.base_weights)
        model.train(name=args.name, epochs=args.epochs, **base_train_kwargs)
        print(f"\nDone. Weights and logs under {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
