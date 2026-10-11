"""The three-window RGB trick, checked against MedGemma's own CT notebook."""
import numpy as np
import pytest

from medgemma_ct import WINDOWS, normalise, window_grey, window_rgb


def test_windows_match_the_published_values():
    """From google-health/medgemma notebooks/high_dimensional_ct_hugging_face."""
    assert WINDOWS == ((-1024, 1024), (-135, 215), (0, 80))


def test_reproduces_the_reference_implementation():
    """Byte-for-byte against the upstream `window()` function."""
    def reference(ct):                                   # upstream, verbatim shape
        def norm(vol, low, high):
            vol = np.clip(vol, low, high).astype(np.float32)
            vol -= low
            vol /= (high - low)
            return vol * 255.0
        clips = [(-1024, 1024), (-135, 215), (0, 80)]
        return np.round(
            np.stack([norm(ct, c[0], c[1]) for c in clips], axis=-1), 0
        ).astype(np.uint8)

    rng = np.random.default_rng(0)
    ct = rng.integers(-1200, 1400, (16, 24), dtype=np.int16)
    assert np.array_equal(window_rgb(ct), reference(ct))


def test_channels_differ():
    """The whole point: three channels must carry three different views."""
    rng = np.random.default_rng(0)
    ct = rng.normal(40, 120, (32, 32))            # soft-tissue-ish spread
    rgb = window_rgb(ct)
    assert not np.array_equal(rgb[..., 0], rgb[..., 1])
    assert not np.array_equal(rgb[..., 1], rgb[..., 2])


def test_narrow_window_gives_soft_tissue_more_levels():
    """A narrow window spends all 256 levels where the anatomy is."""
    rng = np.random.default_rng(0)
    soft = rng.normal(40, 25, 6000)
    ct = np.concatenate([soft, np.full(2000, -1000.0), np.full(2000, 900.0)])

    wide = np.unique(normalise(ct, -1024, 1024).round())
    narrow = np.unique(normalise(ct, -135, 215).round())
    assert len(narrow) > 2 * len(wide)


def test_shapes():
    assert window_rgb(np.zeros((8, 9))).shape == (8, 9, 3)
    assert window_rgb(np.zeros((4, 8, 9))).shape == (4, 8, 9, 3)
    assert window_grey(np.zeros((4, 8, 9))).shape == (4, 8, 9)


def test_dtype_and_range():
    rng = np.random.default_rng(1)
    ct = rng.integers(-1024, 1024, (10, 10), dtype=np.int16)
    rgb = window_rgb(ct)
    assert rgb.dtype == np.uint8
    assert rgb.min() >= 0 and rgb.max() <= 255


def test_clipping_saturates():
    ct = np.array([[-5000.0, 0.0, 5000.0]])
    rgb = window_rgb(ct)
    assert rgb[0, 0, 0] == 0        # far below every window
    assert rgb[0, 2, 0] == 255      # far above every window


def test_bad_window_rejected():
    with pytest.raises(ValueError):
        normalise(np.zeros((2, 2)), 100, 100)
    with pytest.raises(ValueError):
        window_rgb(np.zeros((2, 2)), windows=((0, 1), (0, 2)))
