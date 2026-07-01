"""Tests for ``CrossNN`` (cross nearest-neighbor co-localization test)."""

from __future__ import annotations

import numpy as np
import pytest

from bobiac_tools._cross_nn import CrossNN

EXPECTED_KEYS = {
    "nn_AtoB", "nn_BtoA",
    "nn_random_AtoB", "nn_random_BtoA",
    "ce_AtoB", "ce_BtoA",
    "pval_AtoB", "pval_BtoA",
}


def make_mask(size: int = 50, cell_id: int = 1) -> np.ndarray:
    mask = np.zeros((size, size), dtype=int)
    mask[2 : size - 2, 2 : size - 2] = cell_id
    return mask


def colocalized_spots(n: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two sets placed in exactly the same region → strong co-localization."""
    rng = np.random.default_rng(seed)
    spots = rng.uniform(20, 30, size=(n, 2))
    return spots, spots + rng.uniform(-0.5, 0.5, size=(n, 2))


def separated_spots(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """A in top-left corner, B in bottom-right corner → no co-localization."""
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(1)
    a = rng_a.uniform(3, 10, size=(n, 2))
    b = rng_b.uniform(38, 46, size=(n, 2))
    return a, b


# --------------------------------------------------------------------------- #
# return structure
# --------------------------------------------------------------------------- #
def test_returns_dict_with_correct_keys():
    mask = make_mask()
    a, b = colocalized_spots()
    result = CrossNN(a, b, mask, cell_id=1, n_repeats=20, seed=0)
    assert set(result.keys()) == EXPECTED_KEYS


def test_all_values_are_floats():
    mask = make_mask()
    a, b = colocalized_spots()
    result = CrossNN(a, b, mask, cell_id=1, n_repeats=20, seed=0)
    assert all(isinstance(v, float) for v in result.values())


def test_p_values_in_unit_interval():
    mask = make_mask()
    a, b = colocalized_spots()
    result = CrossNN(a, b, mask, cell_id=1, n_repeats=20, seed=0)
    assert 0.0 <= result["pval_AtoB"] <= 1.0
    assert 0.0 <= result["pval_BtoA"] <= 1.0


def test_ce_values_are_positive():
    mask = make_mask()
    a, b = colocalized_spots()
    result = CrossNN(a, b, mask, cell_id=1, n_repeats=20, seed=0)
    assert result["ce_AtoB"] > 0.0
    assert result["ce_BtoA"] > 0.0


def test_ce_equals_observed_over_random():
    mask = make_mask()
    a, b = colocalized_spots()
    r = CrossNN(a, b, mask, cell_id=1, n_repeats=20, seed=0)
    assert r["ce_AtoB"] == pytest.approx(r["nn_AtoB"] / r["nn_random_AtoB"])
    assert r["ce_BtoA"] == pytest.approx(r["nn_BtoA"] / r["nn_random_BtoA"])


# --------------------------------------------------------------------------- #
# statistical behaviour
# --------------------------------------------------------------------------- #
def test_colocalized_spots_have_low_p_values():
    """Spots in the same region should have small p-values in both directions."""
    mask = make_mask()
    a, b = colocalized_spots(seed=0)
    result = CrossNN(a, b, mask, cell_id=1, n_repeats=100, seed=0)
    assert result["pval_AtoB"] < 0.05
    assert result["pval_BtoA"] < 0.05


def test_colocalized_spots_have_ce_below_one():
    """Co-localized spots have cross Clark-Evans < 1."""
    mask = make_mask()
    a, b = colocalized_spots(seed=0)
    result = CrossNN(a, b, mask, cell_id=1, n_repeats=100, seed=0)
    assert result["ce_AtoB"] < 1.0
    assert result["ce_BtoA"] < 1.0


def test_separated_spots_have_ce_above_one():
    """Spots far apart have cross Clark-Evans > 1 (avoidance)."""
    mask = make_mask()
    a, b = separated_spots()
    result = CrossNN(a, b, mask, cell_id=1, n_repeats=100, seed=0)
    assert result["ce_AtoB"] > 1.0
    assert result["ce_BtoA"] > 1.0


def test_equal_sized_sets_are_nearly_symmetric():
    """With equal-sized sets on the same points, AtoB and BtoA are equal."""
    mask = make_mask()
    rng = np.random.default_rng(5)
    pts = rng.uniform(20, 30, size=(20, 2))
    result = CrossNN(pts, pts, mask, cell_id=1, n_repeats=10, seed=0)
    assert result["nn_AtoB"] == pytest.approx(result["nn_BtoA"])


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def test_same_seed_gives_same_result():
    mask = make_mask()
    a, b = colocalized_spots()
    r1 = CrossNN(a, b, mask, cell_id=1, n_repeats=30, seed=3)
    r2 = CrossNN(a, b, mask, cell_id=1, n_repeats=30, seed=3)
    assert r1 == r2


def test_different_seeds_give_different_random_nns():
    mask = make_mask()
    a, b = colocalized_spots()
    r1 = CrossNN(a, b, mask, cell_id=1, n_repeats=30, seed=1)
    r2 = CrossNN(a, b, mask, cell_id=1, n_repeats=30, seed=2)
    assert r1["nn_random_AtoB"] != pytest.approx(r2["nn_random_AtoB"])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
