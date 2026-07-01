"""Tests for ``random_points_in_mask``."""

from __future__ import annotations

import numpy as np
import pytest

from bobiac_tools._random_points_in_mask import random_points_in_mask


def make_square_mask(size: int = 20, cell_id: int = 1) -> np.ndarray:
    mask = np.zeros((size, size), dtype=int)
    mask[2 : size - 2, 2 : size - 2] = cell_id
    return mask


def make_donut_mask(size: int = 30, cell_id: int = 1) -> np.ndarray:
    """Annular mask — tests that rejection sampling respects non-convex shapes."""
    mask = np.zeros((size, size), dtype=int)
    cy, cx = size // 2, size // 2
    ys, xs = np.ogrid[:size, :size]
    r2 = (ys - cy) ** 2 + (xs - cx) ** 2
    mask[(r2 >= 25) & (r2 <= 100)] = cell_id
    return mask


# --------------------------------------------------------------------------- #
# return shape
# --------------------------------------------------------------------------- #
def test_returns_correct_shape():
    mask = make_square_mask()
    pts = random_points_in_mask(mask, cell_label=1, n=10)
    assert pts.shape == (10, 2)


def test_returns_zero_points_when_n_is_zero():
    mask = make_square_mask()
    pts = random_points_in_mask(mask, cell_label=1, n=0)
    assert pts.shape == (0, 2)


# --------------------------------------------------------------------------- #
# correctness: all points inside the mask
# --------------------------------------------------------------------------- #
def test_all_points_inside_mask():
    mask = make_square_mask()
    pts = random_points_in_mask(mask, cell_label=1, n=200, rng=np.random.default_rng(0))
    rows = pts[:, 0].round().astype(int)
    cols = pts[:, 1].round().astype(int)
    assert np.all(mask[rows, cols] == 1)


def test_all_points_inside_donut_mask():
    """Points must lie inside the annular region, not the hole."""
    mask = make_donut_mask()
    pts = random_points_in_mask(mask, cell_label=1, n=100, rng=np.random.default_rng(1))
    rows = pts[:, 0].round().astype(int)
    cols = pts[:, 1].round().astype(int)
    assert np.all(mask[rows, cols] == 1)


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def test_same_rng_gives_same_points():
    mask = make_square_mask()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    pts1 = random_points_in_mask(mask, cell_label=1, n=20, rng=rng1)
    pts2 = random_points_in_mask(mask, cell_label=1, n=20, rng=rng2)
    assert np.array_equal(pts1, pts2)


def test_different_seeds_give_different_points():
    mask = make_square_mask()
    pts1 = random_points_in_mask(mask, cell_label=1, n=20, rng=np.random.default_rng(0))
    pts2 = random_points_in_mask(mask, cell_label=1, n=20, rng=np.random.default_rng(1))
    assert not np.array_equal(pts1, pts2)


# --------------------------------------------------------------------------- #
# multi-label mask
# --------------------------------------------------------------------------- #
def test_only_samples_requested_cell():
    mask = np.zeros((20, 20), dtype=int)
    mask[0:10, :] = 1
    mask[10:, :] = 2
    pts = random_points_in_mask(mask, cell_label=1, n=50, rng=np.random.default_rng(0))
    rows = pts[:, 0].round().astype(int)
    cols = pts[:, 1].round().astype(int)
    assert np.all(mask[rows, cols] == 1)


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #
def test_raises_on_missing_cell_label():
    mask = make_square_mask()
    with pytest.raises(ValueError, match="not found in mask"):
        random_points_in_mask(mask, cell_label=99, n=10)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
