"""Deterministic behaviour tests for the geometry/temporal violation rules.

These exercise the actual rule logic with synthetic detections + scene
calibration, so the five rule-based violation types (triple-riding, stop-line,
red-light, wrong-side, illegal-parking) and the seatbelt module are *measured*
without needing field imagery or trained weights. Each rule has a positive
(should fire) and a negative (should NOT fire) case to guard against a rule
that always-fires looking "correct".
"""

from __future__ import annotations

import numpy as np

from trace_cv.core.config import LaneConfig, SceneConfig, SignalROI, StopLine, Thresholds, Zone
from trace_cv.core.types import Detection
from trace_cv.violation.base import TrackState, ViolationContext
from trace_cv.violation.parking import ParkingDetector
from trace_cv.violation.red_light import RedLightDetector, classify_signal
from trace_cv.violation.seatbelt import SeatbeltDetector
from trace_cv.violation.stop_line import StopLineDetector
from trace_cv.violation.triple_riding import TripleRidingDetector
from trace_cv.violation.wrong_side import WrongSideDetector

H, W = 720, 1280


def _frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


def _ctx(detections, scene=None, thresholds=None, history=None, frame=None, fps=15.0):
    return ViolationContext(
        frame=_frame() if frame is None else frame,
        detections=detections,
        scene=scene or SceneConfig(),
        thresholds=thresholds or Thresholds(),
        track_history=history or {},
        frame_index=0,
        fps=fps,
    )


# --------------------------------------------------------------------------- #
# Triple riding
# --------------------------------------------------------------------------- #
def _person(x, y, tid=None):
    return Detection(cls="person", bbox=(x, y, x + 30, y + 60), confidence=0.9, track_id=tid)


def test_triple_riding_fires_with_three_riders():
    moto = Detection(cls="motorcycle", bbox=(500, 400, 600, 520), confidence=0.9)
    riders = [_person(510, 380), _person(540, 380), _person(570, 380)]
    out = TripleRidingDetector().check(_ctx([moto, *riders]))
    assert len(out) == 1
    assert out[0].detail["riders"] >= 3


def test_triple_riding_silent_with_two_riders():
    moto = Detection(cls="motorcycle", bbox=(500, 400, 600, 520), confidence=0.9)
    riders = [_person(510, 380), _person(540, 380)]
    assert TripleRidingDetector().check(_ctx([moto, *riders])) == []


# --------------------------------------------------------------------------- #
# Stop line
# --------------------------------------------------------------------------- #
def test_stop_line_fires_in_band():
    scene = SceneConfig(stop_line=StopLine(enabled=True, y=400))
    car = Detection(cls="car", bbox=(600, 360, 700, 410), confidence=0.9)  # bottom 410 in band
    out = StopLineDetector().check(_ctx([car], scene=scene))
    assert len(out) == 1


def test_stop_line_silent_above_line():
    scene = SceneConfig(stop_line=StopLine(enabled=True, y=400))
    car = Detection(cls="car", bbox=(600, 100, 700, 200), confidence=0.9)  # well above
    assert StopLineDetector().check(_ctx([car], scene=scene)) == []


def test_stop_line_disabled_by_default():
    car = Detection(cls="car", bbox=(600, 360, 700, 410), confidence=0.9)
    assert StopLineDetector().check(_ctx([car])) == []  # stop_line not enabled


# --------------------------------------------------------------------------- #
# Red light
# --------------------------------------------------------------------------- #
def test_classify_signal_detects_red_and_green():
    red = np.zeros((40, 20, 3), dtype=np.uint8)
    red[:, :] = (0, 0, 255)  # BGR red
    assert classify_signal(red)[0] == "red"
    green = np.zeros((40, 20, 3), dtype=np.uint8)
    green[:, :] = (0, 255, 0)
    assert classify_signal(green)[0] == "green"


def test_red_light_fires_when_red_and_past_line():
    frame = _frame()
    frame[20:80, 100:140] = (0, 0, 255)  # red signal lamp
    scene = SceneConfig(
        stop_line=StopLine(enabled=True, y=300),
        signal=SignalROI(enabled=True, bbox=[100, 20, 140, 80]),
    )
    car = Detection(cls="car", bbox=(600, 360, 720, 480), confidence=0.9)  # bottom 480 >> 300
    out = RedLightDetector().check(_ctx([car], scene=scene, frame=frame))
    assert len(out) == 1
    assert out[0].detail["signal"] == "red"


def test_red_light_silent_on_green():
    frame = _frame()
    frame[20:80, 100:140] = (0, 255, 0)  # green
    scene = SceneConfig(
        stop_line=StopLine(enabled=True, y=300),
        signal=SignalROI(enabled=True, bbox=[100, 20, 140, 80]),
    )
    car = Detection(cls="car", bbox=(600, 360, 720, 480), confidence=0.9)
    assert RedLightDetector().check(_ctx([car], scene=scene, frame=frame)) == []


# --------------------------------------------------------------------------- #
# Wrong-side driving
# --------------------------------------------------------------------------- #
def test_wrong_side_fires_against_flow():
    scene = SceneConfig(lane=LaneConfig(enabled=True, divider_x=640, correct_direction="down"))
    car = Detection(cls="car", bbox=(760, 80, 840, 160), confidence=0.9, track_id=1)  # right side
    # History moves UP (y decreasing) while the right side expects DOWN.
    hist = {1: [TrackState((760, 480, 840, 560), 0), TrackState((760, 80, 840, 160), 6)]}
    out = WrongSideDetector().check(_ctx([car], scene=scene, history=hist))
    assert len(out) == 1
    assert out[0].detail["moving"] == "up" and out[0].detail["expected"] == "down"


def test_wrong_side_silent_with_flow():
    scene = SceneConfig(lane=LaneConfig(enabled=True, divider_x=640, correct_direction="down"))
    car = Detection(cls="car", bbox=(760, 480, 840, 560), confidence=0.9, track_id=1)
    hist = {1: [TrackState((760, 80, 840, 160), 0), TrackState((760, 480, 840, 560), 6)]}  # down
    assert WrongSideDetector().check(_ctx([car], scene=scene, history=hist)) == []


# --------------------------------------------------------------------------- #
# Illegal parking
# --------------------------------------------------------------------------- #
def test_parking_fires_after_duration():
    zone = Zone(name="Gate", polygon=[[0, 0], [W, 0], [W, H], [0, H]])
    scene = SceneConfig(no_parking_zones=[zone])
    thr = Thresholds(parking_seconds=2.0, stationary_iou=0.90)
    bbox = (600, 360, 700, 460)
    car = Detection(cls="car", bbox=bbox, confidence=0.9, track_id=7)
    # need_frames = 2.0 * fps(10) = 20; give 25 stationary frames.
    hist = {7: [TrackState(bbox, i) for i in range(25)]}
    out = ParkingDetector().check(_ctx([car], scene=scene, thresholds=thr, history=hist, fps=10.0))
    assert len(out) == 1
    assert out[0].detail["zone"] == "Gate"


def test_parking_silent_before_duration():
    zone = Zone(name="Gate", polygon=[[0, 0], [W, 0], [W, H], [0, H]])
    scene = SceneConfig(no_parking_zones=[zone])
    thr = Thresholds(parking_seconds=2.0, stationary_iou=0.90)
    bbox = (600, 360, 700, 460)
    car = Detection(cls="car", bbox=bbox, confidence=0.9, track_id=7)
    hist = {7: [TrackState(bbox, i) for i in range(5)]}  # only 5 frames < 20
    assert ParkingDetector().check(_ctx([car], scene=scene, thresholds=thr, history=hist, fps=10.0)) == []


# --------------------------------------------------------------------------- #
# Seatbelt module logic (proves the rule is sound; the 0/0/4 eval failure was
# data/model availability, not this code path)
# --------------------------------------------------------------------------- #
class _FakeSeatbeltModel:
    def __init__(self, label, conf):
        self._label, self._conf = label, conf

    @property
    def available(self):
        return True

    def predict(self, region):
        return self._label, self._conf


def test_seatbelt_fires_on_no_belt_four_wheeler():
    car = Detection(cls="car", bbox=(100, 100, 400, 400), confidence=0.95)
    det = SeatbeltDetector(_FakeSeatbeltModel("no_belt", 0.9))
    out = det.check(_ctx([car]))
    assert len(out) == 1


def test_seatbelt_never_flags_occluded():
    car = Detection(cls="car", bbox=(100, 100, 400, 400), confidence=0.95)
    det = SeatbeltDetector(_FakeSeatbeltModel("occluded", 0.99))
    assert det.check(_ctx([car])) == []


def test_seatbelt_silent_when_belt_present():
    car = Detection(cls="car", bbox=(100, 100, 400, 400), confidence=0.95)
    det = SeatbeltDetector(_FakeSeatbeltModel("belt", 0.99))
    assert det.check(_ctx([car])) == []


def test_seatbelt_skips_two_wheeler():
    moto = Detection(cls="motorcycle", bbox=(100, 100, 400, 400), confidence=0.95)
    det = SeatbeltDetector(_FakeSeatbeltModel("no_belt", 0.99))
    assert det.check(_ctx([moto])) == []


def test_seatbelt_unavailable_model_is_silent():
    car = Detection(cls="car", bbox=(100, 100, 400, 400), confidence=0.95)
    assert SeatbeltDetector(None).check(_ctx([car])) == []
