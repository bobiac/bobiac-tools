"""Tests for ``manders_image_rotation_test`` and its plotting helper."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bobiac_tools import (
    manders_correlation_coefficient,
    manders_image_rotation_test,
    manders_image_rotation_test_plot,
)

plt.switch_backend("Agg")


def make_corner_blob(shape: tuple[int, int] = (20, 20)) -> np.ndarray:
    """Return an image with one bright, asymmetric blob near the top-left corner.

    The blob is a 3x7 rectangle so that no rotation/flip in the test set maps it
    onto itself (a square, diagonally-symmetric blob would be preserved by the
    transpose transforms).
    """
    img = np.zeros(shape, dtype=float)
    img[2:5, 2:9] = 100.0
    return img


def test_return_structure():
    """The function returns observed M1/M2, two length-9 lists and two p-values."""
    img = make_corner_blob()
    obs_m1, obs_m2, m1s, m2s, p1, p2 = manders_image_rotation_test(img, img)
    assert isinstance(obs_m1, float)
    assert isinstance(obs_m2, float)
    assert len(m1s) == 9
    assert len(m2s) == 9
    assert 0.0 <= p1 <= 1.0
    assert 0.0 <= p2 <= 1.0


def test_observed_matches_direct_manders():
    """Observed coefficients match a direct Manders calculation on the inputs."""
    img = make_corner_blob()
    obs_m1, obs_m2, *_ = manders_image_rotation_test(img, img)
    m1, m2 = manders_correlation_coefficient(img, img, 0.0, 0.0)
    assert obs_m1 == pytest.approx(m1)
    assert obs_m2 == pytest.approx(m2)


def test_identical_offcenter_blob_gives_zero_p_value():
    """A perfectly overlapping off-center blob beats every rotation (p = 0)."""
    img = make_corner_blob()
    *_, p1, p2 = manders_image_rotation_test(img, img)
    assert p1 == 0.0
    assert p2 == 0.0


def test_non_square_images_are_padded():
    """Non-square images are padded to square and produce finite results."""
    img = np.zeros((20, 30), dtype=float)
    img[2:6, 2:6] = 100.0
    obs_m1, obs_m2, m1s, _m2s, _p1, _p2 = manders_image_rotation_test(img, img)
    assert len(m1s) == 9
    assert np.isfinite(obs_m1)
    assert np.isfinite(obs_m2)


def test_plot_returns_none_and_shows():
    """The plot helper returns None and shows exactly one figure."""
    with patch.object(plt, "show") as mock_show:
        result = manders_image_rotation_test_plot(
            1.0, 1.0, [0.0] * 9, [0.0] * 9, 0.0, 0.0
        )
    assert result is None
    assert mock_show.call_count == 1
    plt.close("all")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
