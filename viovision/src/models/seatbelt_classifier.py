"""
Seatbelt vs. no-seatbelt classifier on windshield/cabin ROI crops.

Guide section 2.C: "hardest of the easy three" — occlusion by arm, dark
cabins, windshield glare. Two things differ from helmet:
  1. CLAHE is mandatory preprocessing (guide explicitly calls this out:
     "published seatbelt-plate systems rely on CLAHE to make the cabin
     legible") — applied identically at train and inference via the
     shared extract_features(use_clahe=True) path.
  2. Higher review_threshold than helmet. Guide section 7: "Seatbelt
     unreliable? Keep it but lower its confidence so it routes to the
     VLM/review queue instead of auto-filing. Honest degradation beats
     false tickets." A higher threshold here means more crops get
     routed to VLM review by default, which is the intended behavior
     for this specific classifier, not a bug.
"""

from __future__ import annotations

from src.models.base_crop_classifier import BaseCropClassifier


class SeatbeltClassifier(BaseCropClassifier):
    class_names = ("no_seatbelt", "seatbelt")
    use_clahe = True
    review_threshold = 0.65   # deliberately higher than helmet's 0.55
    backend = "svm"
