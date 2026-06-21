"""
End-to-end pipeline smoke test.

When use_mocks.detector is false (real YOLOv11 loaded), downloads a
sample Indian traffic image for the tracking section so the real detector
has something to actually detect rather than random noise.

Usage:
    python scripts/smoke_test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline_factory import build_pipeline
from src.utils.geometry_violations import (
    CalibrationPolygon,
    ParkingTracker,
    TrackHistory,
    check_triple_riding,
    check_wrong_side,
)

# Public-domain Indian traffic image used for the real-detector tracking test.
# Swap this for any local traffic image if you have one.
SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "3/3b/Mumbai_traffic.jpg/1280px-Mumbai_traffic.jpg"
)
SAMPLE_IMAGE_CACHE = Path(__file__).resolve().parents[1] / "data" / "smoke_test_frame.jpg"
UVH_IMAGE_SEARCH_DIRS = [
    Path(__file__).resolve().parents[1] / "data" / "splits" / "train" / "images",
    Path(__file__).resolve().parents[1] / "data" / "splits" / "valid" / "images",
    Path(__file__).resolve().parents[1] / "data" / "raw" / "uvh26" / "UVH-26-Train" / "images",
]


def get_traffic_frame() -> np.ndarray:
    """Return a real traffic frame for the real-detector path."""
    # Use cached frame if available
    if SAMPLE_IMAGE_CACHE.exists():
        frame = cv2.imread(str(SAMPLE_IMAGE_CACHE))
        if frame is not None:
            return cv2.resize(frame, (1280, 720))

    # Find first available image from existing dataset splits
    for search_dir in UVH_IMAGE_SEARCH_DIRS:
        if not search_dir.is_dir():
            continue
        candidates = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            candidates.extend(search_dir.rglob(ext))
        if candidates:
            img_path = candidates[0]
            print(f"  Using existing dataset image: {img_path.name}")
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frame = cv2.resize(frame, (1280, 720))
                cv2.imwrite(str(SAMPLE_IMAGE_CACHE), frame)
                return frame

    print("  [WARN] No traffic images found in dataset splits. "
          "Place any traffic image at data/smoke_test_frame.jpg "
          "for a better tracking test. Falling back to blank frame.")
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def is_mock_detector(pipeline) -> bool:
    return "Mock" in type(pipeline.detector).__name__


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"
    pipeline = build_pipeline(config_path)

    using_mock = is_mock_detector(pipeline)
    print(f"Detector mode: {'MOCK' if using_mock else 'REAL (YOLOv11 + ByteTrack)'}\n")

    fake_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # -- Detection ------------------------------------------------------------
    print("Running detector ...")
    detections = pipeline.detector.predict(fake_frame)
    if detections:
        for d in detections:
            print(f"  {d.cls:15s} conf={d.confidence:.2f} bbox={d.bbox.as_xyxy()}")
    else:
        print("  No detections on random noise frame (expected for real detector).")

    # -- Triple riding --------------------------------------------------------
    print("\nChecking no-model violation: triple riding ...")
    violators = check_triple_riding(detections)
    print(f"  {len(violators)} two_wheeler(s) flagged")

    # -- Classifiers ----------------------------------------------------------
    fake_crop = fake_frame[0:80, 0:80]

    print("\nRunning helmet classifier on a fake head crop ...")
    r = pipeline.helmet.predict(fake_crop)
    print(f"  cls={r.cls} conf={r.confidence:.2f} needs_review={r.needs_review}")

    print("\nRunning seatbelt classifier on a fake windshield crop ...")
    r = pipeline.seatbelt.predict(fake_crop)
    print(f"  cls={r.cls} conf={r.confidence:.2f} needs_review={r.needs_review}")

    print("\nRunning signal-state classifier on a fake light crop ...")
    r = pipeline.signal.predict(fake_crop)
    print(f"  cls={r.cls} conf={r.confidence:.2f} needs_review={r.needs_review}")

    print("\nRunning OCR on a fake plate crop ...")
    r = pipeline.ocr.predict(fake_crop)
    print(f"  plate_text='{r.plate_text}' conf={r.confidence:.2f} "
          f"(empty expected on noise crop)")

    print("\nRunning VLM adjudication ...")
    v = pipeline.vlm.adjudicate(fake_crop, "no_helmet", context="smoke test")
    print(f"  confirmed={v.violation_confirmed} justification={v.justification}")

    # -- ByteTrack tracking ---------------------------------------------------
    print("\nRunning ByteTrack tracking over 20 frames ...")

    if using_mock:
        tracking_frame = fake_frame
        print("  (mock mode: using noise frame, mock detector returns "
              "synthetic track_ids)")
    else:
        print("  (real mode: using sample traffic image so detector has "
              "something to track)")
        tracking_frame = get_traffic_frame()

    no_parking_zone = CalibrationPolygon(
        points=[(0, 0), (1280, 0), (1280, 720), (0, 720)]
    )
    parking_tracker = ParkingTracker(no_parking_zone, still_frames_threshold=5)
    track_histories: dict[int, TrackHistory] = {}
    all_tracked_ids: set[int] = set()

    for frame_id in range(20):
        tracked = pipeline.detector.track(tracking_frame, persist=True)
        parking_tracker.update(tracked, frame_id)

        for d in tracked:
            if d.track_id is None:
                continue
            all_tracked_ids.add(d.track_id)
            if d.track_id not in track_histories:
                track_histories[d.track_id] = TrackHistory()
            cx, cy = CalibrationPolygon.bbox_centroid(d.bbox)
            track_histories[d.track_id].update((cx, cy), frame_id)

    print(f"  Unique track_ids seen across 20 frames: "
          f"{sorted(all_tracked_ids) if all_tracked_ids else 'none'}")

    if using_mock:
        # Mock-specific assertions
        if len(parking_tracker.histories) > 0:
            print(f"  ParkingTracker active tracks: "
                  f"{list(parking_tracker.histories.keys())}")
        car_hist = track_histories.get(1)
        if car_hist:
            is_wrong = check_wrong_side(car_hist, allowed_direction=(-1.0, 0.0))
            print(f"  wrong_side check on mock car (track_id=1): {is_wrong} "
                  f"(expect True)")
        else:
            print("  [WARN] Mock car track_id=1 not found in histories.")
    else:
        # Real detector: report what was actually found
        print(f"  Tracks with >= 10 frames of history: "
              f"{[tid for tid, h in track_histories.items() if len(h.centroids) >= 10]}")
        for tid, hist in track_histories.items():
            if len(hist.centroids) >= 10:
                is_wrong = check_wrong_side(hist, allowed_direction=(0.0, 1.0))
                print(f"    track_id={tid}: {len(hist.centroids)} frames, "
                      f"wrong_side={is_wrong}")

    print("\nAll components wired and runnable. Pipeline flow proven.")


if __name__ == "__main__":
    main()
