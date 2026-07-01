from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from bobiac_tools._random_points_in_mask import random_points_in_mask


def NNDist(
    spots: np.ndarray,
    mask: np.ndarray,
    cell_id: int,
    n_repeats: int,
    seed: int | None = None,
) -> tuple[float, float, float, float]:
    """
    Test spot distribution using nearest-neighbor distances against a CSR null model.

    For each simulation run, ``n_repeats`` random point patterns are generated
    inside the cell mask and their mean nearest-neighbor distance is computed.
    The Clark-Evans index (observed / random NND) and a Monte Carlo p-value are
    derived from this null distribution.

    Parameters
    ----------
    spots : np.ndarray
        Shape ``(n, 2)`` array of spot coordinates in (row, col) order.
    mask : np.ndarray
        2D label mask where each cell is identified by a unique integer ID.
    cell_id : int
        ID of the cell in ``mask`` that spots belong to.
    n_repeats : int
        Number of CSR simulations to run.
    seed : int | None
        Random seed for reproducibility. If None, results will vary between runs.

    Returns
    -------
    spot_dist_avg : float
        Mean nearest-neighbor distance of the observed spots.
    rnd_dist_avg : float
        Mean of the per-simulation mean nearest-neighbor distances under CSR.
    clark_evans : float
        Clark-Evans index: ``spot_dist_avg / rnd_dist_avg``. Values < 1 indicate
        clustering; values > 1 indicate dispersion.
    p_value : float
        Fraction of CSR simulations whose mean NND is <= the observed mean NND.
        Small values (< 0.05) indicate significant clustering.
    """
    n_points = spots.shape[0]
    rng = np.random.default_rng(seed)

    # k=2: index 0 is the point itself, index 1 is its nearest neighbor
    dist_spots, _ = KDTree(spots).query(spots, k=2)
    spot_dist_avg = float(dist_spots[:, 1].mean())

    rnd_dists = np.empty(n_repeats)
    for i in range(n_repeats):
        rnd_points = random_points_in_mask(
            mask, cell_label=cell_id, n=n_points, rng=rng
        )
        dist_rnd, _ = KDTree(rnd_points).query(rnd_points, k=2)
        rnd_dists[i] = dist_rnd[:, 1].mean()

    rnd_dist_avg = float(rnd_dists.mean())
    clark_evans = spot_dist_avg / rnd_dist_avg
    p_value = float((rnd_dists <= spot_dist_avg).mean())

    return spot_dist_avg, rnd_dist_avg, clark_evans, p_value
