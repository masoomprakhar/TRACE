from trace_cv.core.types import Detection
from trace_cv.detection.tracker import SimpleTracker


def test_persistent_id_across_frames():
    t = SimpleTracker(iou_threshold=0.3)
    f1 = [Detection("car", (0, 0, 10, 10), 0.9)]
    t.update(f1)
    first = f1[0].track_id
    assert first is not None

    f2 = [Detection("car", (1, 0, 11, 10), 0.9)]  # heavily overlapping
    t.update(f2)
    assert f2[0].track_id == first


def test_new_id_for_distinct_object():
    t = SimpleTracker()
    a = [Detection("car", (0, 0, 10, 10), 0.9)]
    t.update(a)
    b = [Detection("car", (200, 200, 210, 210), 0.9)]
    t.update(b)
    assert b[0].track_id != a[0].track_id


def test_class_does_not_cross_match():
    t = SimpleTracker(iou_threshold=0.1)
    a = [Detection("car", (0, 0, 10, 10), 0.9)]
    t.update(a)
    b = [Detection("person", (0, 0, 10, 10), 0.9)]  # same box, different class
    t.update(b)
    assert b[0].track_id != a[0].track_id
