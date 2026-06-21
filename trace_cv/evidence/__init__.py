"""Evidence generation: annotated images + persisted metadata records."""

from trace_cv.evidence.annotator import annotate, violation_bgr
from trace_cv.evidence.builder import EvidenceBuilder

__all__ = ["EvidenceBuilder", "annotate", "violation_bgr"]
