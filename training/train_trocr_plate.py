#!/usr/bin/env python3
"""Fine-tune TrOCR on Indian license plate line crops.

Input: data/ocr/lines/manifest.json (from scripts/build_plate_line_dataset.py)
Output: models/weights/trocr_plate/  (HF format)

Usage:
  python scripts/build_plate_line_dataset.py --synthetic-only
  python training/train_trocr_plate.py --epochs 8 --device cpu
  python training/train_trocr_plate.py --export-base-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training._common import ensure_weights_dir, rel


def main() -> int:
    p = argparse.ArgumentParser(description="Fine-tune TrOCR for Indian plates")
    p.add_argument("--data", default="data/ocr/lines")
    p.add_argument("--manifest", default=None)
    p.add_argument("--base", default="microsoft/trocr-base-printed")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="models/weights/trocr_plate")
    p.add_argument("--max-samples", type=int, default=0, help="Limit total samples (0=all)")
    p.add_argument("--export-base-only", action="store_true")
    args = p.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as exc:
        print("Install: pip install -r requirements-ml.txt", file=sys.stderr)
        raise SystemExit(2) from exc

    data_root = REPO / args.data
    manifest_path = Path(args.manifest) if args.manifest else data_root / "manifest.json"
    if not manifest_path.exists():
        print(f"Missing {manifest_path}. Run scripts/build_plate_line_dataset.py first.", file=sys.stderr)
        return 1

    processor = TrOCRProcessor.from_pretrained(args.base)
    model = VisionEncoderDecoderModel.from_pretrained(args.base)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    if args.export_base_only:
        ensure_weights_dir()
        out_dir = REPO / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        processor.save_pretrained(out_dir)
        print(f"Exported base TrOCR -> {rel(out_dir)}")
        return 0

    samples = json.loads(manifest_path.read_text()).get("samples", [])
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    train = [s for s in samples if s.get("split") != "test"]
    val = [s for s in samples if s.get("split") == "test"]
    if not val:
        val = [s for s in samples if s.get("split") == "val"]
    if not val:
        split = max(1, len(train) // 10)
        val, train = train[:split], train[split:]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)

    class PlateDS(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            from PIL import Image

            row = self.items[idx]
            path = data_root / row["image"]
            img = cv2.imread(str(path))
            if img is None:
                img = np.zeros((64, 256, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            text = row["plate_text"]
            enc = processor(pil, text, return_tensors="pt", padding="max_length", truncation=True)
            return {
                "pixel_values": enc.pixel_values.squeeze(0),
                "labels": enc.labels.squeeze(0),
            }

    def collate(batch):
        pixels = torch.stack([b["pixel_values"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        return {"pixel_values": pixels, "labels": labels}

    train_loader = DataLoader(PlateDS(train), batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PlateDS(val), batch_size=args.batch, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    best_model = model
    best_processor = processor
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())

        model.eval()
        vloss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                vloss += float(out.loss.item())
                gen = model.generate(batch["pixel_values"], max_new_tokens=16)
                preds = processor.batch_decode(gen, skip_special_tokens=True)
                for i, pred in enumerate(preds):
                    gt = processor.decode(batch["labels"][i], skip_special_tokens=True)
                    ptxt = "".join(c for c in pred.upper() if c.isalnum())
                    gtxt = "".join(c for c in gt.upper() if c.isalnum())
                    if ptxt == gtxt:
                        correct += 1
                    total += 1

        mean_v = vloss / max(len(val_loader), 1)
        em = correct / max(total, 1)
        print(
            f"epoch {epoch+1}/{args.epochs} train_loss={loss_sum/max(len(train_loader),1):.4f} "
            f"val_loss={mean_v:.4f} exact_match={em:.3f}"
        )
        if mean_v < best_loss:
            best_loss = mean_v
            best_model = model
            best_processor = processor

    ensure_weights_dir()
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(out_dir)
    best_processor.save_pretrained(out_dir)
    print(f"Saved TrOCR -> {rel(out_dir)}")
    print("config: models.ocr_backend: trocr")
    print(f"        models.trocr_model_path: {rel(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
