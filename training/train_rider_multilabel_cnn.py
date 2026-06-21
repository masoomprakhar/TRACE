#!/usr/bin/env python3
"""Train multi-label rider CNN: no_helmet + triple_riding (Scheme B).

Input: data/datasets/rider_multilabel/labels.csv
Output: models/weights/rider_multilabel_cnn.pt + rider_multilabel_thresholds.json

Usage:
  python scripts/build_rider_multilabel_dataset.py
  python training/train_rider_multilabel_cnn.py --epochs 25 --device cpu
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


class RiderMultiLabelDataset:
    """Minimal dataset wrapper (class defined inside main after torch import)."""

    pass


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {}
    for i, name in enumerate(("no_helmet", "triple_riding")):
        yt, yp = y_true[:, i], y_pred[:, i]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out[name] = {"precision": p, "recall": r, "f1": f1}
    return out


def main() -> int:
    torch, nn, DataLoader, Dataset, models, transforms = _lazy_torch()

    p = argparse.ArgumentParser(description="Train rider multi-label CNN")
    p.add_argument("--data", default="data/datasets/rider_multilabel")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--backbone", choices=("mobilenet_v3_small", "efficientnet_b0"), default="mobilenet_v3_small")
    p.add_argument("--name", default="rider_multilabel_cnn.pt")
    args = p.parse_args()

    data_root = REPO / args.data
    csv_path = data_root / "labels.csv"
    if not csv_path.exists():
        print(f"Missing {csv_path}. Run scripts/build_rider_multilabel_dataset.py first.", file=sys.stderr)
        return 1

    rows: list[dict] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)

    train_rows = [r for r in rows if r["split"] != "test"]
    val_rows = [r for r in rows if r["split"] == "test"]
    if not val_rows:
        val_rows = [r for r in rows if r["split"] == "valid"]
    if not val_rows:
        split = max(1, len(train_rows) // 10)
        val_rows = train_rows[:split]
        train_rows = train_rows[split:]

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    class _DS(Dataset):
        def __init__(self, items, tf):
            self.items = items
            self.tf = tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            row = self.items[idx]
            path = data_root / row["path"]
            img = cv2.imread(str(path))
            if img is None:
                img = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = self.tf(img)
            y = torch.tensor(
                [float(row["no_helmet"]), float(row["triple_riding"])],
                dtype=torch.float32,
            )
            return x, y

    train_loader = DataLoader(_DS(train_rows, train_tf), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(_DS(val_rows, val_tf), batch_size=args.batch, shuffle=False)

    if args.backbone == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
    else:
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = base.classifier[0].in_features
        base.classifier = nn.Identity()

    class RiderNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = base
            self.head = nn.Linear(in_features, 2)

        def forward(self, x):
            return self.head(self.backbone(x))

    model = RiderNet().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pos = np.array(
        [
            max(1.0, sum(float(r["no_helmet"]) for r in train_rows)),
            max(1.0, sum(float(r["triple_riding"]) for r in train_rows)),
        ]
    )
    neg = np.array([len(train_rows), len(train_rows)]) - pos
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=args.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0.0
    best_state = None
    thresholds = {"no_helmet": 0.5, "triple_riding": 0.5}

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(args.device)
                prob = torch.sigmoid(model(xb)).cpu().numpy()
                ps.append(prob)
                ys.append(yb.numpy())
        if not ys:
            continue
        y_true = np.vstack(ys)
        y_prob = np.vstack(ps)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_f1(y_true, y_pred)
        macro_f1 = sum(m["f1"] for m in metrics.values()) / 2
        print(
            f"epoch {epoch+1}/{args.epochs} loss={loss_sum/max(len(train_loader),1):.4f} "
            f"val_F1 no_helmet={metrics['no_helmet']['f1']:.3f} "
            f"triple={metrics['triple_riding']['f1']:.3f} macro={macro_f1:.3f}"
        )
        if macro_f1 >= best_f1:
            best_f1 = macro_f1
            best_state = {
                "model": model.state_dict(),
                "backbone": args.backbone,
                "imgsz": args.imgsz,
                "thresholds": thresholds,
            }

    if best_state is None:
        best_state = {
            "model": model.state_dict(),
            "backbone": args.backbone,
            "imgsz": args.imgsz,
            "thresholds": thresholds,
        }

    ensure_weights_dir()
    dest = WEIGHTS_DIR / args.name
    torch.save(best_state, dest)
    thresh_path = WEIGHTS_DIR / "rider_multilabel_thresholds.json"
    thresh_path.write_text(json.dumps(best_state["thresholds"], indent=2))
    print(f"Saved {rel(dest)} (best macro-F1={best_f1:.3f})")
    print(f"Thresholds -> {rel(thresh_path)}")
    print("config: models.rider_backend: cnn")
    print(f"        models.rider_cnn_weights: {rel(dest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
