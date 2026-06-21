"""Quality-adaptive preprocessing pipeline.

Each correction below is a real, dependency-light OpenCV operation so the
system runs anywhere. Where a heavier learned model would slot in (e.g.
Zero-DCE++ for low light, DeblurGAN-v2 for motion blur, MSBDN for dehazing)
is noted — the interface stays the same, so swapping in a model is a drop-in.
"""

from __future__ import annotations

import cv2
import numpy as np

from trace_cv.core.logging import get_logger
from trace_cv.preprocessing.quality import (
    QualityReport,
    assess_quality,
    dark_channel,
)

log = get_logger("preprocessing")


def enhance_low_light(img: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """CLAHE on the L channel + gamma brightening.
    (Drop-in slot for Zero-DCE++ when available.)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    # gamma < 1 brightens (out = in**gamma lifts the shadows).
    table = (((np.arange(256) / 255.0) ** max(gamma, 1e-3)) * 255).astype(np.uint8)
    return cv2.LUT(out, table)


def deblur(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Unsharp masking to counter motion blur.
    (Drop-in slot for DeblurGAN-v2.)"""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, amount, blurred, 1 - amount, 0)


def dehaze(img: np.ndarray, omega: float = 0.95, t0: float = 0.1,
           size: int = 15) -> np.ndarray:
    """Single-image dehazing via Dark Channel Prior (He et al.).
    (Drop-in slot for MSBDN.)"""
    I = img.astype(np.float64)
    dark = dark_channel(img, size)

    # Atmospheric light: brightest 0.1% of pixels in the dark channel.
    h, w = dark.shape
    n = max(int(h * w * 0.001), 1)
    idx = dark.ravel().argsort()[-n:]
    A = I.reshape(-1, 3)[idx].max(axis=0)
    A = np.maximum(A, 1.0)

    norm = (I / A * 255).astype(np.uint8)
    transmission = 1.0 - omega * (dark_channel(norm, size) / 255.0)
    transmission = np.clip(transmission, t0, 1.0)[:, :, None]

    J = (I - A) / transmission + A
    return np.clip(J, 0, 255).astype(np.uint8)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE on luminance to lift flat / shadowed scenes."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


class AdaptivePreprocessor:
    """Diagnose a frame, then apply only the corrections it needs."""

    def __init__(
        self,
        enabled: bool = True,
        blur_thresh: float = 100.0,
        dark_thresh: float = 70.0,
        haze_thresh: float = 0.5,
        contrast_thresh: float = 0.22,
    ):
        self.enabled = enabled
        self.blur_thresh = blur_thresh
        self.dark_thresh = dark_thresh
        self.haze_thresh = haze_thresh
        self.contrast_thresh = contrast_thresh

    def analyze(self, img: np.ndarray) -> QualityReport:
        return assess_quality(
            img,
            blur_thresh=self.blur_thresh,
            dark_thresh=self.dark_thresh,
            haze_thresh=self.haze_thresh,
            contrast_thresh=self.contrast_thresh,
        )

    def process(self, img: np.ndarray) -> tuple[np.ndarray, QualityReport]:
        """Return (enhanced_image, report). `report.applied` lists the
        corrections that were triggered."""
        report = self.analyze(img)
        if not self.enabled:
            return img, report

        out = img
        if report.is_hazy:
            out = dehaze(out)
            report.applied.append("dehaze")
        if report.is_low_light:
            out = enhance_low_light(out)
            report.applied.append("low_light")
        elif report.is_low_contrast:
            # low light already includes a CLAHE pass; only do standalone
            # contrast lift when the frame is bright but flat.
            out = enhance_contrast(out)
            report.applied.append("contrast")
        if report.is_blurry:
            out = deblur(out)
            report.applied.append("deblur")

        if report.applied:
            log.debug("preprocess applied: %s", ",".join(report.applied))
        return out, report
