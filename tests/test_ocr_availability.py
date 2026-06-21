"""Regression tests for PlateOCR.available.

A TrOCR reader object is constructed even when its checkpoint is missing, so
`available` must gate on the reader's own `.available`, not merely on the
object existing — otherwise the API/health endpoint reports OCR as ready when
no backend can actually read a plate.
"""

import numpy as np

from trace_cv.ocr.plate_ocr import PlateOCR


class _FakeTrOCR:
    def __init__(self, available: bool):
        self.available = available


def _ocr(trocr, reader):
    o = PlateOCR(backend="trocr")
    o._tried = True  # skip real backend loading
    o._trocr = trocr
    o._reader = reader
    return o


def test_unavailable_when_no_backend_loads():
    # TrOCR object exists but its model never loaded, and EasyOCR absent.
    assert _ocr(_FakeTrOCR(False), None).available is False


def test_available_when_trocr_ready():
    assert _ocr(_FakeTrOCR(True), None).available is True


def test_available_when_easyocr_reader_present():
    assert _ocr(None, object()).available is True


def test_read_returns_empty_plate_when_unavailable():
    plate = _ocr(_FakeTrOCR(False), None).read(np.zeros((20, 60, 3), dtype=np.uint8))
    assert plate.text is None
