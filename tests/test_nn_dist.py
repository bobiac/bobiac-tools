"""Tests for ``NNDist`` (nearest-neighbor distance clustering test)."""

from __future__ import annotations

import numpy as np
import pytest

from bobiac_tools._nn_dist import NNDist


def make_mask(size: int = 50, cell_id: int = 1) -> np.ndarray:
    mask = np.zeros((size, size), dtype=int)
    mask[2 : size - 2, 2 : size - 2] = cell_id
    return mask


def clustered_spots(n: int = 30, seed: int = 0) -> np.ndarray:
    """Return spots tightly clustered near the centre of a 50×50 mask."""
    rng = np.random.default_rng(seed)
    return rng.uniform(20, 30, size=(n, 2))


def dispersed_spots(n: int = 30, seed: int = 0) -> np.ndarray:
    """Return spots spread on a regular grid — more dispersed than CSR."""
    side = int(np.ceil(np.sqrt(n)))
    ys = np.linspace(5, 45, side)
    xs = np.linspace(5, 45, side)
    grid = np.array([[y, x] for y in ys for x in xs])
    return grid[:n]


# --------------------------------------------------------------------------- #
# return structure
# --------------------------------------------------------------------------- #
def test_returns_four_floats():
    mask = make_mask()
    result = NNDist(clustered_spots(), mask, cell_id=1, n_repeats=20, seed=0)
    assert len(result) == 4
    assert all(isinstance(v, float) for v in result)


def test_spot_dist_avg_is_positive():
    mask = make_mask()
    spot_dist_avg, *_ = NNDist(clustered_spots(), mask, cell_id=1, n_repeats=20, seed=0)
    assert spot_dist_avg > 0.0


def test_rnd_dist_avg_is_positive():
    mask = make_mask()
    _, rnd_dist_avg, *_ = NNDist(clustered_spots(), mask, cell_id=1, n_repeats=20, seed=0)
    assert rnd_dist_avg > 0.0


def test_p_value_in_unit_interval():
    mask = make_mask()
    *_, p_value = NNDist(clustered_spots(), mask, cell_id=1, n_repeats=20, seed=0)
    assert 0.0 <= p_value <= 1.0


def test_clark_evans_equals_ratio():
    mask = make_mask()
    spot_d, rnd_d, ce, _ = NNDist(clustered_spots(), mask, cell_id=1, n_repeats=20, seed=0)
    assert ce == pytest.approx(spot_d / rnd_d)


# --------------------------------------------------------------------------- #
# statistical behaviour
# --------------------------------------------------------------------------- #
def test_clustered_spots_have_low_p_value():
    """Tightly clustered spots have a smaller NND than random → low p-value."""
    mask = make_mask()
    *_, p_value = NNDist(clustered_spots(), mask, cell_id=1, n_repeats=100, seed=0)
    assert p_value < 0.05


def test_clustered_spots_have_ce_below_one():
    """Clark-Evans < 1 signals clustering."""
    mask = make_mask()
    _sd, _rd, ce, _ = NNDist(clustered_spots(), mask, cell_id=1, n_repeats=100, seed=0)
    assert ce < 1.0


def test_dispersed_spots_have_ce_above_one():
    """Clark-Evans > 1 signals dispersion."""
    mask = make_mask()
    _sd, _rd, ce, _ = NNDist(dispersed_spots(), mask, cell_id=1, n_repeats=100, seed=0)
    assert ce > 1.0


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def test_same_seed_gives_same_result():
    mask = make_mask()
    r1 = NNDist(clustered_spots(), mask, cell_id=1, n_repeats=30, seed=7)
    r2 = NNDist(clustered_spots(), mask, cell_id=1, n_repeats=30, seed=7)
    assert r1 == r2


def test_different_seeds_give_different_rnd_avg():
    mask = make_mask()
    _, rnd1, *_ = NNDist(clustered_spots(), mask, cell_id=1, n_repeats=30, seed=1)
    _, rnd2, *_ = NNDist(clustered_spots(), mask, cell_id=1, n_repeats=30, seed=2)
    assert rnd1 != pytest.approx(rnd2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
