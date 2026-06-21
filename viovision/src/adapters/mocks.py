"""
Mock adapters — satisfy every Protocol in schema.py with deterministic
fake outputs so the orchestrator/pipeline can be wired and tested before
any real model exists.

Guide section 6 ("Day 1: pipeline runs ... no custom training yet, prove
flow") depends on this existing first. Toggle real-vs-mock per component
via configs/pipeline.yaml's USE_MOCKS flag (see scripts/run_pipeline.py).

None of this file imports torch, sklearn, or anything heavy — it's pure
stdlib + numpy so it's instant to import and never breaks CI.
"""

from __future__ import annotations

import random

import numpy as np

from src.adapters.schema import (
    BBox,
    ClassifierResult,
    Detection,
    OCRResult,
    ViolationType,
    VLMVerdict,
)


class MockDetectorAdapter:
    """Returns a plausible-looking fixed set of detections regardless of
    input frame. Good enough to exercise downstream wiring.

    track() simulates a moving car and a stationary two-wheeler across
    successive calls so the stateful violation checks (illegal_parking,
    wrong_side_driving) have something realistic to chew on without
    needing real ByteTrack/torch installed."""

    def __init__(self) -> None:
        self._frame_counter = 0

    def predict(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        return [
            Detection(cls="car", bbox=BBox(10, 10, w // 3, h // 2), confidence=0.91),
            Detection(cls="two_wheeler", bbox=BBox(w // 2, h // 3, w - 10, h - 10),
                       confidence=0.87),
            Detection(cls="person", bbox=BBox(w // 2, 0, w // 2 + 80, h // 4),
                       confidence=0.78),
            Detection(cls="license_plate", bbox=BBox(20, h - 40, 140, h - 10),
                       confidence=0.69),
        ]

    def track(self, frame: np.ndarray, persist: bool = True) -> list[Detection]:
        h, w = frame.shape[:2]
        f = self._frame_counter
        self._frame_counter += 1

        # track_id=1: a car drifting rightward (would trip wrong_side if
        # allowed_direction points left).
        drift = min(f * 4, w // 4)
        moving_car = Detection(
            cls="car",
            bbox=BBox(10 + drift, 10, w // 3 + drift, h // 2),
            confidence=0.9, track_id=1, frame_id=f,
        )
        # track_id=2: a two_wheeler that never moves — candidate for
        # illegal_parking once it's been still long enough.
        stationary_bike = Detection(
            cls="two_wheeler",
            bbox=BBox(w // 2, h // 3, w // 2 + 100, h // 3 + 150),
            confidence=0.85, track_id=2, frame_id=f,
        )
        return [moving_car, stationary_bike]


class MockCropClassifierAdapter:
    """Generic mock for helmet/seatbelt/signal — pass the class options
    and it returns a random-but-deterministic-per-call pick with a
    plausible confidence, occasionally dipping below threshold to
    exercise the needs_review path."""

    def __init__(self, class_names: tuple[str, ...], review_threshold: float = 0.55,
                 seed: int = 42) -> None:
        self.class_names = class_names
        self.review_threshold = review_threshold
        self._rng = random.Random(seed)

    def predict(self, crop: np.ndarray) -> ClassifierResult:
        cls_ = self._rng.choice(self.class_names)
        confidence = self._rng.uniform(0.4, 0.99)
        return ClassifierResult.make(
            cls_=cls_, confidence=confidence, threshold=self.review_threshold,
            source="yolo_class",
        )


class MockOCRAdapter:
    def predict(self, plate_crop: np.ndarray) -> OCRResult:
        return OCRResult(
            plate_text="UP78AB1234",
            confidence=0.93,
            needs_review=False,
            raw_candidates=["UP78AB1234", "UP78AB1234"],
        )


class MockVLMAdjudicatorAdapter:
    def adjudicate(self, crop: np.ndarray, candidate_violation: ViolationType,
                    context: str) -> VLMVerdict:
        return VLMVerdict(
            violation_confirmed=True,
            justification=f"[MOCK] Crop appears consistent with {candidate_violation}.",
            violation_type=candidate_violation,
        )
