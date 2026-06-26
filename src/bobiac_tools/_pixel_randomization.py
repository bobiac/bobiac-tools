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
    # Set the numpy random seed for reproducibility
    np.random.seed(seed)

    # Calculate observed correlation
    observed_correlation = np.corrcoef(channel_1.ravel(), channel_2.ravel())[0, 1]

    # Initialize list to store randomized correlations
    random_correlations = []

    # Store original shape for reshaping
    shape = channel_2.shape

    # for _ in tqdm(range(n_iterations), desc="Costes pixel randomization"):
    for i in range(n_iterations):
        # Flatten, shuffle, and reshape
        randomized_channel_2 = np.random.permutation(channel_2.ravel()).reshape(shape)

        # assert sorted values in randomized_channel_2 are the same as in channel_2
        assert np.array_equal(
            np.sort(randomized_channel_2.ravel()), np.sort(channel_2.ravel())
        ), "Randomized channel values do not match original channel values"

        if images_to_display is not None and i in images_to_display:
            # show original and randomized channel
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(channel_2, cmap="gray")
            ax[0].set_title("Original Channel 2")
            ax[0].axis("off")
            ax[1].imshow(randomized_channel_2, cmap="gray")
            ax[1].set_title(f"Randomized Channel 2 (Iteration {i + 1})")
            ax[1].axis("off")
            plt.show()

        # Calculate correlation with randomized channel
        random_corr = float(
            np.corrcoef(channel_1.ravel(), randomized_channel_2.ravel())[0, 1]
        )
        random_correlations.append(random_corr)

    # Calculate p-value: fraction of random correlations >= observed correlation
    p_value = (
        np.sum(np.array(random_correlations) >= observed_correlation) / n_iterations
    )

    return observed_correlation, random_correlations, p_value
