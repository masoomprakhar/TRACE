import numpy as np

from trace_cv.core.config import SceneConfig, SignalROI, StopLine, Thresholds
from trace_cv.core.types import Detection
from trace_cv.violation.engine import ViolationEngine


def _frame_with_red_signal():
    f = np.zeros((480, 640, 3), np.uint8)
    f[10:80, 10:40] = (0, 0, 255)  # red (BGR) inside the signal ROI
    return f


def test_geometry_violations_fire():
    scene = SceneConfig(
        stop_line=StopLine(enabled=True, y=300),
        signal=SignalROI(enabled=True, bbox=[10, 10, 40, 80]),
    )
    engine = ViolationEngine(scene, Thresholds())
    frame = _frame_with_red_signal()
    detections = [
        # motorcycle with three riders -> triple riding
        Detection("motorcycle", (200, 200, 260, 300), 0.9),
        Detection("person", (205, 150, 225, 210), 0.8),
        Detection("person", (225, 150, 245, 210), 0.8),
        Detection("person", (215, 170, 235, 230), 0.8),
        # car well past the stop line while red -> red light
        Detection("car", (100, 250, 180, 360), 0.9),
        # car sitting on the stop line -> stop line
        Detection("car", (400, 250, 470, 310), 0.9),
    ]
    violations = engine.run(frame, detections)
    types = {v.type.value for v in violations}
    assert "triple_riding" in types
    assert "red_light" in types
    assert "stop_line" in types
    # No helmet/seatbelt model loaded -> those must NOT be fabricated.
    assert "no_helmet" not in types
    assert "no_seatbelt" not in types


def test_no_config_means_no_geometry_violations():
    engine = ViolationEngine(SceneConfig(), Thresholds())
    frame = np.zeros((480, 640, 3), np.uint8)
    detections = [Detection("car", (100, 250, 180, 360), 0.9)]
    # Stop line / signal disabled by default -> nothing fires for a lone car.
    assert engine.run(frame, detections) == []


def test_active_modules_excludes_unloaded_models():
    engine = ViolationEngine(SceneConfig(), Thresholds())
    active = engine.active_modules
    assert "triple_riding" in active
    assert "no_helmet" not in active  # no model configured
