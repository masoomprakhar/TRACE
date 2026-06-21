import random

from trace_cv.evaluation.calibration import (
    TemperatureScaler,
    expected_calibration_error,
)


def test_ece_in_range():
    e = expected_calibration_error([0.9, 0.8, 0.7], [1, 0, 1], n_bins=5)
    assert 0.0 <= e <= 1.0


def test_perfectly_calibrated_has_low_ece():
    # 70%-confident predictions that are correct 70% of the time.
    confs = [0.7] * 100
    labels = [1] * 70 + [0] * 30
    assert expected_calibration_error(confs, labels) < 0.05


def test_temperature_softens_overconfidence():
    rng = random.Random(0)
    confs, labels = [], []
    for _ in range(300):
        confs.append(0.95)  # very confident
        labels.append(1 if rng.random() < 0.6 else 0)  # but only 60% right

    ts = TemperatureScaler().fit(confs, labels)
    assert ts.temperature > 1.0                     # softening
    assert ts.transform(0.95) < 0.95                # lowered confidence

    before = expected_calibration_error(confs, labels)
    after = expected_calibration_error(ts.transform_many(confs), labels)
    assert after <= before + 1e-9                   # calibration improved
