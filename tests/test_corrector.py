from trace_cv.ocr.corrector import (
    correct_plate,
    format_plate,
    is_valid_plate,
    normalize_plate,
    plate_similarity,
)


def test_normalize():
    assert normalize_plate("mh-01 ab") == "MH01AB"


def test_already_valid_passthrough():
    assert correct_plate("MH 01 AB 1234") == ("MH01AB1234", True)


def test_confusion_correction():
    # O->0, I->1 coerced into their digit positions.
    assert correct_plate("MH O1 AB I234") == ("MH01AB1234", True)


def test_invalid_format():
    assert not is_valid_plate("ABC")
    assert is_valid_plate("KA03MG2255")


def test_format_plate():
    assert format_plate("MH01AB1234") == "MH 01 AB 1234"


def test_similarity():
    assert plate_similarity("MH01AB1234", "MH01AB1234") == 1.0
    assert plate_similarity("MH01AB1234", "MH01AB1235") > 0.8
    assert plate_similarity("MH01AB1234", "DL09XY0000") < 0.5
