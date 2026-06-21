#!/usr/bin/env python3
"""Live license-plate OCR demo (requires ML extras: easyocr).

Renders realistic Indian plates — exactly the cropped plate region the OCR
stage receives after plate detection — reads them with the real EasyOCR
wrapper, then shows the India-specific corrector fixing typical OCR errors.

Usage:  python scripts/demo_ocr.py
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from trace_cv.ocr.corrector import correct_plate, format_plate
from trace_cv.ocr.plate_ocr import PlateOCR

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_OUT = Path("data/output")


def render_plate(text: str, w: int = 660, h: int = 190) -> np.ndarray:
    img = Image.new("RGB", (w, h), (250, 250, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([5, 5, w - 5, h - 5], outline=(12, 12, 12), width=3)
    font = ImageFont.truetype(_FONT, 62)
    box = d.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    d.text(((w - tw) / 2 - box[0], (h - th) / 2 - box[1]), text,
           fill=(10, 10, 10), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main() -> int:
    ocr = PlateOCR(["en"])
    print("EasyOCR available:", ocr.available)
    if not ocr.available:
        print("Install ML extras first: pip install -r requirements-ml.txt")
        return 2

    _OUT.mkdir(parents=True, exist_ok=True)
    print("\n=== LIVE OCR (render -> EasyOCR -> corrector) ===")
    tiles = []
    for text in ["MH 12 DE 1433", "KA 03 MG 2255", "DL 8C AF 5078"]:
        img = render_plate(text)
        result = ocr.read(img)
        print(f"  {text:<16} OCR={result.raw_text!s:<12} "
              f"-> {result.text}  valid={result.valid_format} "
              f"conf={result.confidence}")
        band = np.full((46, img.shape[1], 3), (22, 36, 63), np.uint8)
        cv2.putText(band, f"OCR: {result.text or '-'} ({int(result.confidence*100)}%)",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 179, 1), 2)
        tiles.append(np.vstack([img, band]))
    cv2.imwrite(str(_OUT / "plate_ocr_demo.jpg"), np.vstack(tiles))
    print(f"\nsaved visual: {_OUT / 'plate_ocr_demo.jpg'}")

    print("\n=== INDIAN-PLATE CORRECTOR (OCR errors -> fixed) ===")
    for g in ["MH O1 A8 1234", "KA O3 MG 2Z55", "DLI2CA567B", "TN 21 8H OOO7"]:
        corrected, valid = correct_plate(g)
        print(f"  {g:<16} -> {format_plate(corrected):<14} valid={valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
