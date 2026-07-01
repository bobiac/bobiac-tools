from __future__ import annotations

import numpy as np


def random_points_in_mask(
    mask: np.ndarray,
    cell_label: int,
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate n random points uniformly distributed within a labeled cell region.

    Points are placed continuously using rejection sampling: candidate points are
    drawn uniformly from the bounding box of the cell and accepted only if they
    fall within the cell mask. This correctly handles irregular cell shapes and
    serves as a null model for Complete Spatial Randomness (CSR).

    Parameters
    ----------
    mask : np.ndarray
        2D label mask where each cell is identified by a unique integer ID
        and background is 0.
    cell_label : int
        ID of the cell to sample within.
    n : int
        Number of random points to generate.
    rng : np.random.Generator | None
        Random number generator for reproducibility. If None, a new default
        generator is used. Pass a shared generator when calling from within a
        simulation loop to avoid repeated seeding overhead.

    Returns
    -------
    np.ndarray
        Shape `(n, 2)` array of random point coordinates in (row, col) order.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n == 0:
        return np.empty((0, 2))

    ys, xs = np.where(mask == cell_label)
    if ys.size == 0:
        raise ValueError(f"cell_label {cell_label!r} not found in mask")

    y_min, y_max = float(ys.min()), float(ys.max())
    x_min, x_max = float(xs.min()), float(xs.max())

    points: list[list[float]] = []
    while len(points) < n:
        # Oversample to reduce the number of loop iterations for irregular shapes
        batch = max(n - len(points), 64)
        cand_y = rng.uniform(y_min, y_max, size=batch)
        cand_x = rng.uniform(x_min, x_max, size=batch)
        for y, x in zip(cand_y, cand_x):
            if mask[round(y), round(x)] == cell_label:
                points.append([y, x])
                if len(points) == n:
                    break

    return np.array(points)
