from trace_cv.core.types import (
    bbox_center,
    bbox_contains_point,
    bbox_iou,
    bbox_overlap_ratio,
    point_in_polygon,
)


def test_iou_identical():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint():
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # 50 overlap / 150 union = 1/3
    assert abs(bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-6


def test_overlap_ratio_contained():
    assert bbox_overlap_ratio((2, 2, 4, 4), (0, 0, 10, 10)) == 1.0


def test_center():
    assert bbox_center((0, 0, 10, 20)) == (5, 10)


def test_contains_point():
    assert bbox_contains_point((0, 0, 10, 10), (5, 5))
    assert not bbox_contains_point((0, 0, 10, 10), (15, 5))


def test_point_in_polygon():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), square)
    assert not point_in_polygon((15, 5), square)
