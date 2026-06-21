from trace_cv.evaluation.metrics import (
    binary_prf,
    detection_map,
    multilabel_report,
    ocr_cer,
    ocr_exact_match,
)


def test_binary_prf():
    p = binary_prf(tp=8, fp=2, fn=2, tn=88)
    assert abs(p.precision - 0.8) < 1e-9
    assert abs(p.recall - 0.8) < 1e-9
    assert abs(p.f1 - 0.8) < 1e-9
    assert abs(p.accuracy - 0.96) < 1e-9


def test_detection_map_perfect():
    preds = [[("car", (0, 0, 10, 10), 0.9)]]
    gts = [[("car", (0, 0, 10, 10))]]
    m = detection_map(preds, gts, ["car"])
    assert m["map50"] == 1.0


def test_detection_map_miss():
    preds = [[("car", (100, 100, 110, 110), 0.9)]]
    gts = [[("car", (0, 0, 10, 10))]]
    m = detection_map(preds, gts, ["car"])
    assert m["map50"] == 0.0


def test_detection_map_excludes_absent_class():
    """A class with no GT and no predictions must be EXCLUDED from the mean,
    not awarded a free 1.0 (the old degenerate behaviour)."""
    preds = [[("car", (0, 0, 10, 10), 0.9)]]
    gts = [[("car", (0, 0, 10, 10))]]
    m = detection_map(preds, gts, ["car", "bus"])
    assert m["map50"] == 1.0  # driven only by car, not inflated by bus
    assert "bus" not in m["per_class_ap50"]
    assert m["classes_evaluated"] == ["car"]
    assert m["n_classes_evaluated"] == 1
    assert m["per_class_support"]["bus"] == {"n_gt": 0, "n_pred": 0}


def test_detection_map_hallucinated_class_is_false_positive():
    """Predictions for a class with no GT are false positives -> AP 0.0,
    and that class IS counted (it is not 'absent')."""
    preds = [[("car", (0, 0, 10, 10), 0.9), ("bus", (0, 0, 10, 10), 0.9)]]
    gts = [[("car", (0, 0, 10, 10))]]
    m = detection_map(preds, gts, ["car", "bus"])
    assert m["per_class_ap50"]["car"] == 1.0
    assert m["per_class_ap50"]["bus"] == 0.0  # hallucination penalised
    assert m["map50"] == 0.5
    assert m["n_classes_evaluated"] == 2


def test_detection_map_no_ground_truth_is_none_not_one():
    """The headline anti-cheat: no GT anywhere -> mAP is undefined (None),
    NEVER 1.0. This is the exact bug that inflated the old reports."""
    preds = [[]]
    gts = [[]]
    m = detection_map(preds, gts, ["car", "bus"])
    assert m["map50"] is None
    assert m["map5095"] is None
    assert m["n_classes_evaluated"] == 0


def test_detection_map_predictions_without_any_gt_is_zero():
    preds = [[("car", (0, 0, 10, 10), 0.9)]]
    gts = [[]]
    m = detection_map(preds, gts, ["car"])
    assert m["map50"] == 0.0  # pure false positive, not a free 1.0


def test_ocr_cer():
    assert ocr_cer("ABC", "ABC") == 0.0
    assert abs(ocr_cer("AXC", "ABC") - 1 / 3) < 1e-9


def test_ocr_cer_ignores_spacing_and_case():
    # A correctly-read but conventionally-spaced plate must score CER 0,
    # consistent with ocr_exact_match (this was the CER-inflation bug).
    assert ocr_cer("MH 01 AB 1234", "MH01AB1234") == 0.0
    assert ocr_cer("mh01ab1234", "MH01AB1234") == 0.0


def test_ocr_exact_match_ignores_spacing():
    assert ocr_exact_match(["MH 01 AB 1234"], ["MH01AB1234"]) == 1.0


def test_multilabel_report_shape():
    rep = multilabel_report(
        [{"a", "b"}, {"a"}], [{"a"}, {"a"}], ["a", "b"]
    )
    assert "per_label" in rep and "macro" in rep and "micro" in rep
    assert rep["per_label"]["a"]["f1"] == 1.0  # 'a' always predicted correctly
