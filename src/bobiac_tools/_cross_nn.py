from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from bobiac_tools._random_points_in_mask import random_points_in_mask


def cross_nn(
    spots_A: np.ndarray,
    spots_B: np.ndarray,
    mask: np.ndarray,
    cell_id: int,
    n_repeats: int,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Test co-localization of two spot sets using cross nearest-neighbor distances.

    For each point in A the nearest neighbor in B is found (and vice versa).
    For the A→B direction, the null distribution holds A at its observed
    positions and randomizes only B (and vice versa for B→A) — this tests
    whether the *real* configuration of one set is closer to the other than
    a same-sized random configuration would be, controlling for the
    reference set's density without discarding the query set's real spatial
    structure.

    The two directions (A→B and B→A) can differ when the sets have different
    sizes: a small set is always close to a large one even by chance.

    Parameters
    ----------
    spots_A : np.ndarray
        Shape `(n, 2)` array of spot coordinates in (row, col) order.
    spots_B : np.ndarray
        Shape `(m, 2)` array of spot coordinates in (row, col) order.
    mask : np.ndarray
        2D label mask where each cell is identified by a unique integer ID.
    cell_id : int
        ID of the cell in `mask` that both spot sets belong to.
    n_repeats : int
        Number of CSR simulations to run.
    seed : int | None
        Random seed for reproducibility. If None, results will vary between runs.

    Returns
    -------
    dict with keys:

    nn_AtoB, nn_BtoA : float
        Observed mean cross-NN distance in each direction.
    nn_random_AtoB, nn_random_BtoA : float
        Mean of simulated cross-NN distances under CSR.
    ce_AtoB, ce_BtoA : float
        Cross Clark-Evans index (observed / random). Values < 1 indicate
        co-localization; values > 1 indicate avoidance.
    pval_AtoB, pval_BtoA : float
        Fraction of simulations with cross-NN distance <= observed distance.
        Small values indicate significant co-localization.
    """
    n_A, n_B = spots_A.shape[0], spots_B.shape[0]
    rng = np.random.default_rng(seed)

    nn_AtoB = float(KDTree(spots_B).query(spots_A, k=1)[0].mean())
    nn_BtoA = float(KDTree(spots_A).query(spots_B, k=1)[0].mean())

    sims_AtoB = np.empty(n_repeats)
    sims_BtoA = np.empty(n_repeats)
    for i in range(n_repeats):
        rnd_A = random_points_in_mask(mask, cell_label=cell_id, n=n_A, rng=rng)
        rnd_B = random_points_in_mask(mask, cell_label=cell_id, n=n_B, rng=rng)
        sims_AtoB[i] = KDTree(rnd_B).query(spots_A, k=1)[0].mean()
        sims_BtoA[i] = KDTree(rnd_A).query(spots_B, k=1)[0].mean()

    return {
        "nn_AtoB":        nn_AtoB,
        "nn_BtoA":        nn_BtoA,
        "nn_random_AtoB": float(sims_AtoB.mean()),
        "nn_random_BtoA": float(sims_BtoA.mean()),
        "ce_AtoB":        nn_AtoB / sims_AtoB.mean(),
        "ce_BtoA":        nn_BtoA / sims_BtoA.mean(),
        "pval_AtoB":      float((sims_AtoB <= nn_AtoB).mean()),
        "pval_BtoA":      float((sims_BtoA <= nn_BtoA).mean()),
    }
