"""Tests for ``RipleysL`` (Ripley's L function with Monte Carlo envelope)."""

from __future__ import annotations

import numpy as np
import pytest

from bobiac_tools._ripleys_l import RipleysL


def make_mask(size: int = 60, cell_id: int = 1) -> np.ndarray:
    mask = np.zeros((size, size), dtype=int)
    mask[4 : size - 4, 4 : size - 4] = cell_id
    return mask


def csr_spots(n: int = 30, seed: int = 0) -> np.ndarray:
    """Spots drawn uniformly at random — approximate CSR."""
    return np.random.default_rng(seed).uniform(5, 55, size=(n, 2))


def clustered_spots(n: int = 30, seed: int = 0) -> np.ndarray:
    """All spots packed tightly near the centre."""
    return np.random.default_rng(seed).uniform(27, 33, size=(n, 2))


# --------------------------------------------------------------------------- #
# return structure
# --------------------------------------------------------------------------- #
def test_returns_three_arrays():
    mask = make_mask()
    result = RipleysL(csr_spots(), mask, cell_id=1, n_repeats=5, seed=0)
    assert len(result) == 3


def test_r_values_shape():
    mask = make_mask()
    r_vals, L_obs, L_sims = RipleysL(csr_spots(), mask, cell_id=1, n_repeats=5, seed=0)
    assert r_vals.ndim == 1
    assert len(r_vals) == 30  # default 30 radii


def test_L_observed_shape_matches_r_values():
    mask = make_mask()
    r_vals, L_obs, _ = RipleysL(csr_spots(), mask, cell_id=1, n_repeats=5, seed=0)
    assert L_obs.shape == r_vals.shape


def test_L_sims_shape():
    mask = make_mask()
    r_vals, _, L_sims = RipleysL(csr_spots(), mask, cell_id=1, n_repeats=8, seed=0)
    assert L_sims.shape == (8, len(r_vals))


def test_custom_r_values_respected():
    mask = make_mask()
    custom_r = np.array([1.0, 2.0, 5.0, 10.0])
    r_vals, L_obs, L_sims = RipleysL(
        csr_spots(), mask, cell_id=1, n_repeats=5, r_values=custom_r, seed=0
    )
    assert np.array_equal(r_vals, custom_r)
    assert L_obs.shape == (4,)
    assert L_sims.shape == (5, 4)


def test_r_values_start_at_zero_by_default():
    mask = make_mask()
    r_vals, *_ = RipleysL(csr_spots(), mask, cell_id=1, n_repeats=5, seed=0)
    assert r_vals[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# statistical behaviour
# --------------------------------------------------------------------------- #
def test_clustered_spots_have_positive_L_at_cluster_scale():
    """L(r) - r > 0 at scales that capture the cluster."""
    mask = make_mask()
    custom_r = np.linspace(1, 10, 10)
    _, L_obs, _ = RipleysL(
        clustered_spots(), mask, cell_id=1, n_repeats=5, r_values=custom_r, seed=0
    )
    valid = ~np.isnan(L_obs)
    assert np.any(L_obs[valid] > 0)


def test_L_sims_envelope_brackets_csr():
    """For CSR-like spots, the observed L should often lie within the sim envelope."""
    mask = make_mask()
    custom_r = np.linspace(2, 15, 10)
    _, L_obs, L_sims = RipleysL(
        csr_spots(), mask, cell_id=1, n_repeats=50, r_values=custom_r, seed=0
    )
    env_min = np.nanmin(L_sims, axis=0)
    env_max = np.nanmax(L_sims, axis=0)
    valid = ~np.isnan(L_obs)
    inside = np.sum((L_obs[valid] >= env_min[valid]) & (L_obs[valid] <= env_max[valid]))
    assert inside >= valid.sum() * 0.5


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def test_same_seed_gives_same_result():
    mask = make_mask()
    r1, obs1, sims1 = RipleysL(csr_spots(), mask, cell_id=1, n_repeats=5, seed=9)
    r2, obs2, sims2 = RipleysL(csr_spots(), mask, cell_id=1, n_repeats=5, seed=9)
    assert np.array_equal(r1, r2)
    assert np.array_equal(obs1, obs2)
    assert np.array_equal(sims1, sims2)


def test_different_seeds_give_different_sims():
    mask = make_mask()
    _, _, sims1 = RipleysL(csr_spots(), mask, cell_id=1, n_repeats=5, seed=1)
    _, _, sims2 = RipleysL(csr_spots(), mask, cell_id=1, n_repeats=5, seed=2)
    assert not np.array_equal(sims1, sims2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
