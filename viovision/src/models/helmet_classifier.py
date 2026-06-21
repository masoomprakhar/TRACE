"""
Helmet vs. no-helmet classifier on rider head-region crops.

Guide section 2.B: "highest ROI, do this first." This is the two-stage
escalation path — only needed if folding helmet/no_helmet directly into
the YOLOv11 detector classes underperforms. Keep both options live;
config decides which path is active (see configs/pipeline.yaml).

Input crops: rider head region, ideally produced by cropping a fixed
ratio above a detected person/two_wheeler box, or a dedicated 'head'
detection if you add one to YOLO.
"""

from __future__ import annotations

from src.models.base_crop_classifier import BaseCropClassifier


class HelmetClassifier(BaseCropClassifier):
    class_names = ("no_helmet", "helmet")
    use_clahe = False          # head crops aren't typically glare/dark-cabin limited
    review_threshold = 0.55
    backend = "svm"
