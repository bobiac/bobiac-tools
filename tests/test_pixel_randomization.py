"""Tests for ``pixel_randomization`` (Costes pixel randomization test)."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bobiac_tools import pixel_randomization

plt.switch_backend("Agg")


def make_correlated_channels() -> tuple[np.ndarray, np.ndarray]:
    """Return two strongly correlated channels."""
    rng = np.random.default_rng(0)
    ch1 = rng.random((16, 16))
    ch2 = 2.0 * ch1 + rng.random((16, 16)) * 0.01
    return ch1, ch2


def test_return_structure():
    """The function returns (float, list[float] of length n, float in [0, 1])."""
    ch1, ch2 = make_correlated_channels()
    observed, randoms, p_value = pixel_randomization(ch1, ch2, n_iterations=50, seed=1)
    assert isinstance(observed, float)
    assert isinstance(randoms, list)
    assert len(randoms) == 50
    assert all(isinstance(r, float) for r in randoms)
    assert isinstance(p_value, float)
    assert 0.0 <= p_value <= 1.0


def test_observed_matches_corrcoef():
    """The observed correlation equals numpy's Pearson r of the inputs."""
    ch1, ch2 = make_correlated_channels()
    observed, _randoms, _p = pixel_randomization(ch1, ch2, n_iterations=10, seed=1)
    expected = np.corrcoef(ch1.ravel(), ch2.ravel())[0, 1]
    assert observed == pytest.approx(expected)


def test_correlated_channels_have_low_p_value():
    """Strongly correlated channels yield a small p-value."""
    ch1, ch2 = make_correlated_channels()
    _o, _r, p_value = pixel_randomization(ch1, ch2, n_iterations=200, seed=1)
    assert p_value < 0.05


def test_reproducible_with_same_seed():
    """The same seed reproduces the randomized correlations and p-value."""
    ch1, ch2 = make_correlated_channels()
    _o1, r1, p1 = pixel_randomization(ch1, ch2, n_iterations=50, seed=7)
    _o2, r2, p2 = pixel_randomization(ch1, ch2, n_iterations=50, seed=7)
    assert r1 == r2
    assert p1 == p2


def test_does_not_touch_global_numpy_rng():
    """The function uses a local Generator and leaves the global RNG untouched."""
    ch1, ch2 = make_correlated_channels()
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    pixel_randomization(ch1, ch2, n_iterations=10, seed=0)
    assert np.random.random() == expected


def test_images_to_display_calls_show():
    """Each requested display iteration triggers one ``plt.show`` call."""
    ch1, ch2 = make_correlated_channels()
    with patch.object(plt, "show") as mock_show:
        pixel_randomization(ch1, ch2, n_iterations=3, seed=0, images_to_display=[0, 2])
    assert mock_show.call_count == 2
    plt.close("all")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
