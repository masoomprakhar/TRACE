"""
Shared feature extraction for the sklearn classifiers (helmet, seatbelt,
signal-state). HOG captures shape/edges (helmet contour, seatbelt diagonal
strap line), color histogram captures color (signal light state, skin vs.
helmet material, dark cabin vs. bright strap).

Keep this deterministic and dependency-light (skimage + opencv only) since
it has to run identically at train time and inference time — any drift
here silently tanks classifier accuracy in a way that's hard to debug.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog

# Resize target before feature extraction. Fixed size is required because
# HOG's output length depends on input dimensions.
DEFAULT_RESIZE = (128, 128)

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True,
)

COLOR_HIST_BINS = 16  # per channel, per color space


def _resize(crop: np.ndarray, size: tuple[int, int] = DEFAULT_RESIZE) -> np.ndarray:
    if crop.size == 0:
        raise ValueError("Empty crop passed to feature extractor.")
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """
    Contrast-limited adaptive histogram equalization on the L channel.
    Guide section 2.C / 5: CLAHE is preprocessing applied identically at
    train and inference time on dark/glare-prone crops (seatbelt cabin,
    plates). Call this BEFORE feature extraction for those classifiers.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def extract_hog(crop_bgr: np.ndarray, resize: tuple[int, int] = DEFAULT_RESIZE) -> np.ndarray:
    gray = cv2.cvtColor(_resize(crop_bgr, resize), cv2.COLOR_BGR2GRAY)
    features = hog(gray, **HOG_PARAMS)
    return features.astype(np.float32)


def extract_color_histogram(crop_bgr: np.ndarray, bins: int = COLOR_HIST_BINS) -> np.ndarray:
    """
    HSV histogram, normalized. HSV chosen over RGB because hue isolation
    is what makes the signal-light classifier and helmet-color separation
    robust to brightness changes (day vs. night, glare).
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist_channels = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        hist_channels.append(hist.flatten())
    return np.concatenate(hist_channels).astype(np.float32)


def extract_features(crop_bgr: np.ndarray, use_clahe: bool = False,
                      resize: tuple[int, int] = DEFAULT_RESIZE) -> np.ndarray:
    """
    Concatenated HOG + color histogram feature vector for one crop.
    This is the single function both training and inference call —
    do not duplicate this logic elsewhere.
    """
    if use_clahe:
        crop_bgr = apply_clahe(crop_bgr)
    resized = _resize(crop_bgr, resize)
    hog_feat = extract_hog(resized, resize)
    color_feat = extract_color_histogram(resized)
    return np.concatenate([hog_feat, color_feat])
