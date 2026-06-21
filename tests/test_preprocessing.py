import numpy as np

from trace_cv.preprocessing.pipeline import AdaptivePreprocessor


def test_process_preserves_shape():
    img = (np.random.rand(120, 160, 3) * 255).astype(np.uint8)
    out, report = AdaptivePreprocessor().process(img)
    assert out.shape == img.shape
    assert isinstance(report.to_dict(), dict)


def test_low_light_detected_and_brightened():
    dark = np.full((120, 160, 3), 10, np.uint8)
    out, report = AdaptivePreprocessor().process(dark)
    assert report.is_low_light
    assert "low_light" in report.applied
    assert out.mean() > dark.mean()  # brightened, not darkened


def test_disabled_is_passthrough():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    pre = AdaptivePreprocessor(enabled=False)
    out, _ = pre.process(img)
    assert np.array_equal(out, img)
