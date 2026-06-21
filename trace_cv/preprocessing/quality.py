"""Cheap per-frame quality heuristics that decide which corrections to apply.

Running an enhancement on every frame wastes compute and can degrade
already-good images. We measure four signals and only fix what's broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class QualityReport:
    blur_score: float          # Laplacian variance — lower = blurrier
    luminance: float           # mean brightness 0-255 — lower = darker
    haze: float                # dark-channel mean 0-1 — higher = hazier
    contrast: float            # normalized std — lower = flatter
    is_blurry: bool = False
    is_low_light: bool = False
    is_hazy: bool = False
    is_low_contrast: bool = False
    applied: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "blur_score": round(self.blur_score, 2),
            "luminance": round(self.luminance, 2),
            "haze": round(self.haze, 4),
            "contrast": round(self.contrast, 4),
            "is_blurry": self.is_blurry,
            "is_low_light": self.is_low_light,
            "is_hazy": self.is_hazy,
            "is_low_contrast": self.is_low_contrast,
            "applied": self.applied,
        }


def dark_channel(img: np.ndarray, size: int = 15) -> np.ndarray:
    """Min over channels, then a local min (erosion). Foundation of the
    dark-channel-prior haze estimate and dehazer."""
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_channel, kernel)


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def luminance(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def haze_score(img: np.ndarray) -> float:
    return float(dark_channel(img, 15).mean()) / 255.0


def contrast_score(gray: np.ndarray) -> float:
    return float(gray.std()) / 128.0


def assess_quality(
    img: np.ndarray,
    blur_thresh: float = 100.0,
    dark_thresh: float = 70.0,
    haze_thresh: float = 0.5,
    contrast_thresh: float = 0.22,
) -> QualityReport:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bs = blur_score(gray)
    lum = luminance(img)
    hz = haze_score(img)
    ct = contrast_score(gray)
    return QualityReport(
        blur_score=bs,
        luminance=lum,
        haze=hz,
        contrast=ct,
        is_blurry=bs < blur_thresh,
        is_low_light=lum < dark_thresh,
        is_hazy=hz > haze_thresh,
        is_low_contrast=ct < contrast_thresh,
    )
