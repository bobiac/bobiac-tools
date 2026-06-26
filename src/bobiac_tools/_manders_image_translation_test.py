from __future__ import annotations

import numpy as np

from ._manders_correlation_coefficient import manders_correlation_coefficient


def manders_image_translation_randomization(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    threshold_ch1: float = 0.0,
    threshold_ch2: float = 0.0,
    n_iterations: int = 1000,
    max_shift_fraction: float = 0.5,
    seed: int = 3,
) -> tuple[float, float, list[float], list[float], float, float]:
    """
    Perform image translation randomization test for Manders' coefficients validation.

    This method applies random translations (shifts) to one channel relative to the
    other, breaking spatial relationships while preserving intensity distributions
    and local patterns.

    Parameters
    ----------
    channel_1 : np.ndarray
        First fluorescent channel (kept fixed)
    channel_2 : np.ndarray
        Second fluorescent channel (will be translated)
    threshold_ch1 : float, optional
        Intensity threshold for channel A (if None, uses Otsu's method)
    threshold_ch2 : float, optional
        Intensity threshold for channel B (if None, uses Otsu's method)
    n_iterations : int
        Number of randomization iterations (default: 1000)
    max_shift_fraction : float
        Maximum shift as fraction of image dimensions (default: 0.5)
    seed : int
        Random numpy seed for reproducibility (default: 3)

    Returns
    -------
    Tuple containing:
        - float: Observed M1 coefficient
        - float: Observed M2 coefficient
        - List[float]: M1 coefficients from randomized iterations (A overlap with B)
        - List[float]: M2 coefficients from randomized iterations (B overlap with A)
        - float: P-value for M1 (fraction of random M1 >= observed M1)
        - float: P-value for M2 (fraction of random M2 >= observed M2)
    """
    # Set numpy random seed for reproducibility
    np.random.seed(seed)

    def _translate_image(image: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
        """Translate image by given shifts with wrap-around."""
        return np.roll(np.roll(image, shift_y, axis=0), shift_x, axis=1)

    # Calculate observed Manders' coefficients
    observed_m1, observed_m2 = manders_correlation_coefficient(
        channel_1, channel_2, threshold_ch1, threshold_ch2
    )

    # Calculate maximum shifts
    max_shift_y = int(channel_2.shape[0] * max_shift_fraction)
    max_shift_x = int(channel_2.shape[1] * max_shift_fraction)

    # Initialize lists to store randomized coefficients
    random_m1_values = []
    random_m2_values = []

    # for _ in tqdm(range(n_iterations), desc="Image translation randomization"):
    for _ in range(n_iterations):
        # Generate random shifts (excluding zero shift)
        shift_y = np.random.randint(-max_shift_y, max_shift_y + 1)
        shift_x = np.random.randint(-max_shift_x, max_shift_x + 1)

        # Ensure at least one shift is non-zero
        if shift_y == 0 and shift_x == 0:
            shift_y = np.random.choice([-1, 1])

        # Apply translation to channel B
        translated_ch2 = _translate_image(channel_2, shift_y, shift_x)

        # Calculate Manders' coefficients with translated channel B
        random_m1, random_m2 = manders_correlation_coefficient(
            channel_1, translated_ch2, threshold_ch1, threshold_ch2
        )
        random_m1_values.append(random_m1)
        random_m2_values.append(random_m2)

    # Calculate p-values
    p_value_m1 = np.sum(np.array(random_m1_values) >= observed_m1) / n_iterations
    p_value_m2 = np.sum(np.array(random_m2_values) >= observed_m2) / n_iterations

    return (
        float(observed_m1),
        float(observed_m2),
        random_m1_values,
        random_m2_values,
        float(p_value_m1),
        float(p_value_m2),
    )
