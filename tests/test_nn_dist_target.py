"""Tests for ``nn_dist_target`` (nearest-neighbor distance to a fixed target set)."""

from __future__ import annotations

import numpy as np
import pytest

from bobiac_tools._nn_dist import nn_dist_target

EXPECTED_KEYS = {"nn_observed", "nn_random", "ce", "pval"}


def make_mask(size: int = 50, cell_id: int = 1) -> np.ndarray:
    mask = np.zeros((size, size), dtype=int)
    mask[2 : size - 2, 2 : size - 2] = cell_id
    return mask


def colocalized_spots(n: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`spots_from` placed right on top of `spots_to`."""
    rng = np.random.default_rng(seed)
    spots_to = rng.uniform(20, 30, size=(n, 2))
    spots_from = spots_to + rng.uniform(-0.5, 0.5, size=(n, 2))
    return spots_from, spots_to


def separated_spots(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """`spots_from` in one corner, `spots_to` in the opposite corner."""
    rng_from = np.random.default_rng(0)
    rng_to = np.random.default_rng(1)
    spots_from = rng_from.uniform(3, 10, size=(n, 2))
    spots_to = rng_to.uniform(38, 46, size=(n, 2))
    return spots_from, spots_to


# --------------------------------------------------------------------------- #
# return structure
# --------------------------------------------------------------------------- #
def test_returns_dict_with_correct_keys():
    mask = make_mask()
    spots_from, spots_to = colocalized_spots()
    result = nn_dist_target(
        spots_from, spots_to, mask, cell_id=1, n_repeats=20, seed=0
    )
    assert set(result.keys()) == EXPECTED_KEYS


def test_all_values_are_floats():
    mask = make_mask()
    spots_from, spots_to = colocalized_spots()
    result = nn_dist_target(
        spots_from, spots_to, mask, cell_id=1, n_repeats=20, seed=0
    )
    assert all(isinstance(v, float) for v in result.values())


def test_pval_in_unit_interval():
    mask = make_mask()
    spots_from, spots_to = colocalized_spots()
    result = nn_dist_target(
        spots_from, spots_to, mask, cell_id=1, n_repeats=20, seed=0
    )
    assert 0.0 <= result["pval"] <= 1.0


def test_ce_equals_observed_over_random():
    mask = make_mask()
    spots_from, spots_to = colocalized_spots()
    r = nn_dist_target(spots_from, spots_to, mask, cell_id=1, n_repeats=20, seed=0)
    assert r["ce"] == pytest.approx(r["nn_observed"] / r["nn_random"])


# --------------------------------------------------------------------------- #
# statistical behaviour
# --------------------------------------------------------------------------- #
def test_colocalized_spots_have_low_pval():
    """`spots_from` sitting on top of `spots_to` should have a low p-value."""
    mask = make_mask()
    spots_from, spots_to = colocalized_spots(seed=0)
    result = nn_dist_target(
        spots_from, spots_to, mask, cell_id=1, n_repeats=100, seed=0
    )
    assert result["pval"] < 0.05


def test_colocalized_spots_have_ce_below_one():
    """Co-located spots should be closer than expected by chance: ce < 1."""
    mask = make_mask()
    spots_from, spots_to = colocalized_spots(seed=0)
    result = nn_dist_target(
        spots_from, spots_to, mask, cell_id=1, n_repeats=100, seed=0
    )
    assert result["ce"] < 1.0


def test_separated_spots_have_ce_above_one():
    """`spots_from` far from `spots_to` should have ce > 1 (avoidance)."""
    mask = make_mask()
    spots_from, spots_to = separated_spots()
    result = nn_dist_target(
        spots_from, spots_to, mask, cell_id=1, n_repeats=100, seed=0
    )
    assert result["ce"] > 1.0


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def test_same_seed_gives_same_result():
    mask = make_mask()
    spots_from, spots_to = colocalized_spots()
    r1 = nn_dist_target(spots_from, spots_to, mask, cell_id=1, n_repeats=30, seed=3)
    r2 = nn_dist_target(spots_from, spots_to, mask, cell_id=1, n_repeats=30, seed=3)
    assert r1 == r2


def test_different_seeds_give_different_nn_random():
    mask = make_mask()
    spots_from, spots_to = colocalized_spots()
    r1 = nn_dist_target(spots_from, spots_to, mask, cell_id=1, n_repeats=30, seed=1)
    r2 = nn_dist_target(spots_from, spots_to, mask, cell_id=1, n_repeats=30, seed=2)
    assert r1["nn_random"] != pytest.approx(r2["nn_random"])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
