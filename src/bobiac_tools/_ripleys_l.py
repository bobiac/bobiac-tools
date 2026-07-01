from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree

from bobiac_tools._random_points_in_mask import random_points_in_mask


def ripleys_l(
    spots: np.ndarray,
    mask: np.ndarray,
    cell_id: int,
    n_repeats: int,
    r_values: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Ripley's L function for spots in a cell against a CSR null model.

    Returns the centered L function L(r) - r, which is 0 under CSR. Positive
    values indicate clustering at scale r; negative values indicate dispersion.
    Border correction is applied via mask erosion: only points at least r pixels
    from the cell boundary contribute as query points.

    Parameters
    ----------
    spots : np.ndarray
        Shape `(n, 2)` array of spot coordinates in (row, col) order.
    mask : np.ndarray
        2D label mask where each cell is identified by a unique integer ID.
    cell_id : int
        ID of the cell in `mask` that spots belong to.
    n_repeats : int
        Number of CSR simulations to run.
    r_values : np.ndarray | None
        Radii at which to evaluate L(r) - r. Defaults to 30 values from 0 to
        half the cell's effective radius (`sqrt(area / pi) * 0.5`).
    seed : int | None
        Random seed for reproducibility. If None, results will vary between runs.

    Returns
    -------
    r_values : np.ndarray
        Shape `(R,)` radii at which L(r) - r was evaluated.
    L_observed : np.ndarray
        Shape `(R,)` centered L function for the observed spots.
    L_sims : np.ndarray
        Shape `(n_repeats, R)` centered L function for each CSR simulation.
        Use `.min(0)` / `.max(0)` for the envelope, `.mean(0)` for the
        expected value under CSR.
    """
    cell_mask = mask == cell_id
    area = float(cell_mask.sum())
    n_points = spots.shape[0]
    rng = np.random.default_rng(seed)

    if r_values is None:
        r_max = np.sqrt(area / np.pi) * 0.5
        r_values = np.linspace(0, r_max, 30)
    else:
        r_values = np.asarray(r_values)

    # Precompute eroded masks once — reused for every simulation. Pad by one
    # pixel of background so the distance transform treats the array edge as
    # a cell boundary too, matching binary_erosion's border_value=0 behavior.
    padded = np.pad(cell_mask, 1, mode="constant", constant_values=False)
    dist_to_edge = np.asarray(distance_transform_edt(padded))[1:-1, 1:-1]
    eroded_masks = [(dist_to_edge > r) & cell_mask for r in r_values]

    def _compute_L(coords: np.ndarray) -> np.ndarray:
        tree = KDTree(coords)
        row_idx = coords[:, 0].astype(int)
        col_idx = coords[:, 1].astype(int)
        K: list[float] = []
        for r, eroded in zip(r_values, eroded_masks):
            valid = eroded[row_idx, col_idx]
            n_valid = int(valid.sum())
            if n_valid == 0:
                K.append(np.nan)
                continue
            valid_tree = KDTree(coords[valid])
            count = valid_tree.count_neighbors(tree, r) - n_valid
            K.append((area / (n_valid * n_points)) * count)
        return np.sqrt(np.array(K) / np.pi) - r_values

    L_observed = _compute_L(spots)

    L_sims = np.empty((n_repeats, len(r_values)))
    for i in range(n_repeats):
        rnd_points = random_points_in_mask(
            mask, cell_label=cell_id, n=n_points, rng=rng
        )
        L_sims[i] = _compute_L(rnd_points)

    return r_values, L_observed, L_sims
