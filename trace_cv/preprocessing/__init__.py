"""Adaptive image preprocessing: diagnose each frame, then apply only the
corrections it needs (low-light, deblur, dehaze, contrast)."""

from trace_cv.preprocessing.pipeline import AdaptivePreprocessor
from trace_cv.preprocessing.quality import QualityReport, assess_quality

__all__ = ["AdaptivePreprocessor", "QualityReport", "assess_quality"]
