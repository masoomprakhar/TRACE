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


def test_ocr_cer():
    assert ocr_cer("ABC", "ABC") == 0.0
    assert abs(ocr_cer("AXC", "ABC") - 1 / 3) < 1e-9


def test_ocr_exact_match_ignores_spacing():
    assert ocr_exact_match(["MH 01 AB 1234"], ["MH01AB1234"]) == 1.0


def test_multilabel_report_shape():
    rep = multilabel_report(
        [{"a", "b"}, {"a"}], [{"a"}, {"a"}], ["a", "b"]
    )
    assert "per_label" in rep and "macro" in rep and "micro" in rep
    assert rep["per_label"]["a"]["f1"] == 1.0  # 'a' always predicted correctly
