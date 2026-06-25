#!/usr/bin/env python3
"""Train binary seatbelt CNN: no_seatbelt classification on windshield crops.

Input : data/datasets/seatbelt_cls/labels.csv
Output: models/weights/seatbelt_cnn.pt + seatbelt_thresholds.json

Usage:
  python scripts/build_seatbelt_dataset.py
  python training/train_seatbelt_cnn.py --epochs 25 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training._common import WEIGHTS_DIR, ensure_weights_dir, rel


def _lazy_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except ImportError as exc:
        print("Install torch: pip install -r requirements-ml.txt", file=sys.stderr)
        raise SystemExit(2) from exc
    return torch, nn, DataLoader, Dataset, models, transforms


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc}


def main() -> int:
    torch, nn, DataLoader, Dataset, models, transforms = _lazy_torch()

    p = argparse.ArgumentParser(description="Train seatbelt binary CNN")
    p.add_argument("--data",     default="data/datasets/seatbelt_cls")
    p.add_argument("--epochs",   type=int,   default=25)
    p.add_argument("--batch",    type=int,   default=32)
    p.add_argument("--imgsz",    type=int,   default=224)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--device",   default="cpu")
    p.add_argument("--backbone", choices=("mobilenet_v3_small", "efficientnet_b0"),
                   default="mobilenet_v3_small")
    p.add_argument("--name",     default="seatbelt_cnn.pt")
    args = p.parse_args()

    data_root = REPO / args.data
    csv_path  = data_root / "labels.csv"
    if not csv_path.exists():
        print(
            f"Missing {csv_path}. Run scripts/build_seatbelt_dataset.py first.",
            file=sys.stderr,
        )
        return 1

    rows: list[dict] = []
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    train_rows = [r for r in rows if r["split"] != "test"]
    val_rows   = [r for r in rows if r["split"] == "test"]
    if not val_rows:
        val_rows = [r for r in rows if r["split"] == "valid"]
    if not val_rows:
        cut        = max(1, len(train_rows) // 10)
        val_rows   = train_rows[:cut]
        train_rows = train_rows[cut:]

    print(f"train={len(train_rows)}  val={len(val_rows)}")
    pos = sum(int(r["no_seatbelt"]) for r in train_rows)
    neg = len(train_rows) - pos
    print(f"train no_seatbelt={pos}  seatbelt={neg}")

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf  = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        normalize,
    ])

    class _DS(Dataset):
        def __init__(self, items, tf):
            self.items = items
            self.tf    = tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            row  = self.items[idx]
            path = data_root / row["path"]
            img  = cv2.imread(str(path))
            if img is None:
                img = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x   = self.tf(img)
            y   = torch.tensor(float(row["no_seatbelt"]), dtype=torch.float32)
            return x, y

    train_loader = DataLoader(_DS(train_rows, train_tf), batch_size=args.batch, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(_DS(val_rows,   val_tf),   batch_size=args.batch, shuffle=False, num_workers=2)

    # ── model ────────────────────────────────────────────────────────────────
    if args.backbone == "efficientnet_b0":
        base        = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
    else:
        base        = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = base.classifier[0].in_features
        base.classifier = nn.Identity()

    class SeatbeltNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = base
            self.head     = nn.Linear(in_features, 1)   # binary: logit for no_seatbelt

        def forward(self, x):
            return self.head(self.backbone(x)).squeeze(1)

    model = SeatbeltNet().to(args.device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # class-balanced pos_weight
    pos_w     = max(neg, 1) / max(pos, 1)
    pos_weight = torch.tensor([pos_w], dtype=torch.float32, device=args.device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1    = 0.0
    best_state = None
    threshold  = 0.5

    for epoch in range(args.epochs):
        # ── train ────────────────────────────────────────────────────────────
        model.train()
        loss_sum = 0.0
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            if batch_idx % 20 == 0:
                print(
                    f"  epoch {epoch+1}/{args.epochs} "
                    f"batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        # ── val ──────────────────────────────────────────────────────────────
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb   = xb.to(args.device)
                prob = torch.sigmoid(model(xb)).cpu().numpy()
                ps.append(prob)
                ys.append(yb.numpy())

        if not ys:
            continue

        y_true  = np.concatenate(ys)
        y_prob  = np.concatenate(ps)
        y_pred  = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred)

        print(
            f"epoch {epoch+1}/{args.epochs}  "
            f"loss={loss_sum/max(len(train_loader),1):.4f}  "
            f"F1={metrics['f1']:.3f}  "
            f"prec={metrics['precision']:.3f}  "
            f"rec={metrics['recall']:.3f}  "
            f"acc={metrics['accuracy']:.3f}"
        )

        if metrics["f1"] >= best_f1:
            best_f1    = metrics["f1"]
            best_state = {
                "model":     model.state_dict(),
                "backbone":  args.backbone,
                "imgsz":     args.imgsz,
                "threshold": threshold,
            }

    if best_state is None:
        best_state = {
            "model":     model.state_dict(),
            "backbone":  args.backbone,
            "imgsz":     args.imgsz,
            "threshold": threshold,
        }

    ensure_weights_dir()
    dest        = WEIGHTS_DIR / args.name
    thresh_path = WEIGHTS_DIR / "seatbelt_thresholds.json"
    torch.save(best_state, dest)
    thresh_path.write_text(json.dumps({"no_seatbelt": best_state["threshold"]}, indent=2))

    print(f"\nSaved {rel(dest)}  (best F1={best_f1:.3f})")
    print(f"Thresholds -> {rel(thresh_path)}")
    print("config: models.seatbelt_backend: cnn")
    print(f"        models.seatbelt_cnn_weights: {rel(dest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
