"""Tests for ``manders_image_translation_randomization``."""

from __future__ import annotations

import numpy as np
import pytest

from bobiac_tools import (
    manders_correlation_coefficient,
    manders_image_translation_randomization,
)


def make_corner_blob(shape: tuple[int, int] = (30, 30)) -> np.ndarray:
    """Return an image with a single bright blob near the top-left corner."""
    img = np.zeros(shape, dtype=float)
    img[3:7, 3:7] = 100.0
    return img


def test_return_structure():
    """The function returns observed M1/M2, two length-n lists and two p-values."""
    img = make_corner_blob()
    obs_m1, obs_m2, m1s, m2s, p1, p2 = manders_image_translation_randomization(
        img, img, n_iterations=100, seed=1
    )
    assert isinstance(obs_m1, float)
    assert isinstance(obs_m2, float)
    assert len(m1s) == 100
    assert len(m2s) == 100
    assert 0.0 <= p1 <= 1.0
    assert 0.0 <= p2 <= 1.0


def test_observed_matches_direct_manders():
    """Observed coefficients match a direct Manders calculation on the inputs."""
    img = make_corner_blob()
    obs_m1, obs_m2, *_ = manders_image_translation_randomization(
        img, img, n_iterations=10, seed=1
    )
    m1, m2 = manders_correlation_coefficient(img, img, 0.0, 0.0)
    assert obs_m1 == pytest.approx(m1)
    assert obs_m2 == pytest.approx(m2)


def test_colocalized_blob_low_p_value():
    """A perfectly overlapping blob yields small p-values under translation."""
    img = make_corner_blob()
    *_, p1, p2 = manders_image_translation_randomization(
        img, img, n_iterations=200, seed=1
    )
    assert p1 < 0.1
    assert p2 < 0.1


def test_reproducible_with_same_seed():
    """The same seed reproduces the randomized coefficients and p-values."""
    img = make_corner_blob()
    r1 = manders_image_translation_randomization(img, img, n_iterations=50, seed=3)
    r2 = manders_image_translation_randomization(img, img, n_iterations=50, seed=3)
    assert r1[2] == r2[2]  # M1 lists
    assert r1[4] == r2[4]  # M1 p-value


def test_does_not_touch_global_numpy_rng():
    """The function uses a local Generator and leaves the global RNG untouched."""
    img = make_corner_blob()
    np.random.seed(321)
    expected = np.random.random()
    np.random.seed(321)
    manders_image_translation_randomization(img, img, n_iterations=10, seed=0)
    assert np.random.random() == expected


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
