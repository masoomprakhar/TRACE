"""
VioVision adapter contract.

Every model in the pipeline (real or mocked) implements one of the
Protocols below. The orchestrator never imports a concrete model class
directly — it only ever talks to these shapes. That is what makes
USE_MOCKS=false a drop-in swap instead of a rewrite.

Design rules baked into these shapes (from the training guide):
- Everything is built on top of the primary detector's output. Detector
  runs once per frame; every downstream adapter consumes a crop + the
  detection that produced it, never the raw frame.
- Classifiers (helmet/seatbelt/signal) all return the SAME result shape
  (ClassifierResult) so the orchestrator can route them generically.
- Every prediction carries a confidence score because low-confidence
  routing to the VLM review queue is a first-class part of the design
  ("honest degradation beats false tickets" — guide section 7).
- cls strings are fixed string literals, not free text, so they match
  SharedState.cls exactly as the guide instructs in section 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Fixed class vocabularies — keep these as the single source of truth.
# Anything that writes a `cls` string must use one of these literals.
# ---------------------------------------------------------------------------

DetectorClass = Literal[
    "car",
    "two_wheeler",
    "person",
    "license_plate",
    "windshield",
    "signal_light",
]

HelmetClass = Literal["helmet", "no_helmet"]
SeatbeltClass = Literal["seatbelt", "no_seatbelt"]
SignalClass = Literal["red", "yellow", "green", "unknown"]

ViolationType = Literal[
    "triple_riding",
    "stop_line",
    "illegal_parking",
    "wrong_side_driving",
    "red_light",
    "no_helmet",
    "no_seatbelt",
]

# Confidence below this routes to VLM adjudication instead of auto-filing.
# Tune per-violation-type; seatbelt in particular should sit higher than
# this default given how often it's occluded (guide section 2.C, 7).
DEFAULT_REVIEW_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    """Pixel-space box, top-left origin, half-open interval semantics."""
    x1: int
    y1: int
    x2: int
    y2: int

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Slice this box out of a frame. Caller guarantees box is in-bounds."""
        return frame[self.y1:self.y2, self.x1:self.x2]


@dataclass(frozen=True)
class Detection:
    """One box from the primary YOLOv11 detector."""
    cls: DetectorClass
    bbox: BBox
    confidence: float
    track_id: int | None = None  # set when running with a tracker, else None
    frame_id: int = 0


@dataclass(frozen=True)
class ClassifierResult:
    """
    Generic result shape shared by helmet, seatbelt, and signal-state
    classifiers. The orchestrator routes on `needs_review` without caring
    which classifier produced the result.
    """
    cls: str                  # one of HelmetClass / SeatbeltClass / SignalClass
    confidence: float
    needs_review: bool        # True if confidence < threshold for this model
    source: Literal["sklearn", "yolo_class", "hsv_heuristic"] = "sklearn"

    @staticmethod
    def make(cls_: str, confidence: float, threshold: float,
              source: Literal["sklearn", "yolo_class", "hsv_heuristic"] = "sklearn"
              ) -> "ClassifierResult":
        return ClassifierResult(
            cls=cls_,
            confidence=confidence,
            needs_review=confidence < threshold,
            source=source,
        )


@dataclass(frozen=True)
class OCRResult:
    plate_text: str
    confidence: float
    needs_review: bool
    raw_candidates: list[str] = field(default_factory=list)  # multi-engine votes


@dataclass(frozen=True)
class VLMVerdict:
    violation_confirmed: bool
    justification: str
    violation_type: ViolationType


# ---------------------------------------------------------------------------
# Adapter protocols — implement these, real or mocked, nothing else matters
# ---------------------------------------------------------------------------

@runtime_checkable
class DetectorAdapter(Protocol):
    def predict(self, frame: np.ndarray) -> list[Detection]:
        """One frame in, all boxes out. No state, no side effects."""
        ...

    def track(self, frame: np.ndarray, persist: bool = True) -> list[Detection]:
        """
        Same as predict(), but Detection.track_id is populated using
        ByteTrack (real adapter) or a deterministic simulation (mock).
        Required input for the stateful violations in
        geometry_violations.py (ParkingTracker, check_wrong_side) — they
        cannot function on predict()'s output since track_id is always
        None there.
        """
        ...


@runtime_checkable
class CropClassifierAdapter(Protocol):
    """Shared shape for helmet / seatbelt / signal-state classifiers."""
    def predict(self, crop: np.ndarray) -> ClassifierResult:
        ...


@runtime_checkable
class OCRAdapter(Protocol):
    def predict(self, plate_crop: np.ndarray) -> OCRResult:
        ...


@runtime_checkable
class VLMAdjudicatorAdapter(Protocol):
    def adjudicate(self, crop: np.ndarray, candidate_violation: ViolationType,
                    context: str) -> VLMVerdict:
        ...
