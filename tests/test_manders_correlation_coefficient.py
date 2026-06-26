"""Tests for ``manders_correlation_coefficient``."""

from __future__ import annotations

import numpy as np
import pytest

from bobiac_tools import manders_correlation_coefficient


def test_perfect_colocalization():
    """Identical channels give M1 = M2 = 1.0."""
    ch = np.array([[0, 5, 10], [0, 20, 0]], dtype=float)
    m1, m2 = manders_correlation_coefficient(ch, ch, 0.0, 0.0)
    assert m1 == 1.0
    assert m2 == 1.0


def test_no_overlap_gives_zero():
    """Disjoint signals give M1 = M2 = 0.0."""
    ch1 = np.array([[10, 10, 0, 0]], dtype=float)
    ch2 = np.array([[0, 0, 10, 10]], dtype=float)
    m1, m2 = manders_correlation_coefficient(ch1, ch2, 0.0, 0.0)
    assert m1 == 0.0
    assert m2 == 0.0


def test_partial_overlap_known_fractions():
    """Manders coefficients match a hand-computed partial overlap."""
    ch1 = np.array([10, 10, 10, 10], dtype=float)
    ch2 = np.array([10, 10, 0, 0], dtype=float)
    m1, m2 = manders_correlation_coefficient(ch1, ch2, 0.0, 0.0)
    assert m1 == 0.5  # overlap ch1 (20) / total ch1 above threshold (40)
    assert m2 == 1.0  # overlap ch2 (20) / total ch2 above threshold (20)


def test_thresholds_exclude_low_pixels():
    """Pixels at or below the threshold are excluded from the coefficients."""
    ch1 = np.array([5, 50, 50], dtype=float)
    ch2 = np.array([5, 50, 0], dtype=float)
    # threshold 10 -> mask_a = [F, T, T], mask_b = [F, T, F], overlap = [F, T, F]
    m1, m2 = manders_correlation_coefficient(ch1, ch2, 10.0, 10.0)
    assert m1 == 0.5  # 50 / (50 + 50)
    assert m2 == 1.0  # 50 / 50


def test_all_below_threshold_returns_zero():
    """When nothing is above threshold both coefficients are 0.0."""
    ch = np.full((4,), 5.0)
    m1, m2 = manders_correlation_coefficient(ch, ch, 10.0, 10.0)
    assert m1 == 0.0
    assert m2 == 0.0


def test_returns_python_floats_in_unit_range():
    """The coefficients are plain floats within [0, 1]."""
    rng = np.random.default_rng(0)
    ch1 = rng.random((8, 8))
    ch2 = rng.random((8, 8))
    m1, m2 = manders_correlation_coefficient(ch1, ch2, 0.2, 0.2)
    assert isinstance(m1, float)
    assert isinstance(m2, float)
    assert 0.0 <= m1 <= 1.0
    assert 0.0 <= m2 <= 1.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
