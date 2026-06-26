from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def pixel_randomization(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    n_iterations: int = 500,
    seed: int = 3,
    images_to_display: list[int] | None = None,
) -> tuple[float, list[float], float]:
    """
    Perform Costes pixel randomization test for colocalization significance.

    Parameters
    ----------
    channel_1 : np.ndarray
        Reference channel (kept unchanged)
    channel_2 : np.ndarray
        Channel to be randomized
    n_iterations : int
        Number of randomization iterations. By default, 500.
    seed : int
        Random numpy seed for reproducibility. By default, 3.
    images_to_display : list[int] | None
        List of iteration indices to display images for debugging.
        If None, no images are displayed. By default, None.

    Returns
    -------
    Tuple containing:
        - float: Observed correlation coefficient
        - List[float]: Correlation coefficients from randomized iterations
        - float: P-value (fraction of random correlations >= observed)
    """
    # Dedicated random generator for reproducibility (no global RNG side effects)
    rng = np.random.default_rng(seed)

    # Flatten channel 1 once; it is reused for every correlation
    ch1_flat = channel_1.ravel()
    shape = channel_2.shape

    # Calculate observed correlation
    observed_correlation = float(np.corrcoef(ch1_flat, channel_2.ravel())[0, 1])

    # Initialize list to store randomized correlations
    random_correlations = []

    for i in range(n_iterations):
        # Flatten, shuffle, and reshape (permutation preserves the value set)
        randomized_channel_2 = rng.permutation(channel_2.ravel()).reshape(shape)

        if images_to_display is not None and i in images_to_display:
            # show original and randomized channel
            _fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(channel_2, cmap="gray")
            ax[0].set_title("Original Channel 2")
            ax[0].axis("off")
            ax[1].imshow(randomized_channel_2, cmap="gray")
            ax[1].set_title(f"Randomized Channel 2 (Iteration {i + 1})")
            ax[1].axis("off")
            plt.show()

        # Calculate correlation with randomized channel
        random_corr = float(np.corrcoef(ch1_flat, randomized_channel_2.ravel())[0, 1])
        random_correlations.append(random_corr)

    # Calculate p-value: fraction of random correlations >= observed correlation
    p_value = float(
        np.sum(np.array(random_correlations) >= observed_correlation) / n_iterations
    )

    return observed_correlation, random_correlations, p_value
