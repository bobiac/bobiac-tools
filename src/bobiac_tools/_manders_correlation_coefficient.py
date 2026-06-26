from __future__ import annotations

import numpy as np


def manders_correlation_coefficient(
    ch1: np.ndarray, ch2: np.ndarray, threshold_ch1: float, threshold_ch2: float
) -> tuple[float, float]:
    """Calculate Manders' correlation coefficients for colocalization analysis.

    This function computes the Manders' coefficients M1 and M2, which quantify the
    fractional overlap between two fluorescent channels.

    The algorithm works as follows:
    1. Create binary masks for pixels above the provided threshold in each channel
    2. Compute overlap mask where both channels have above-threshold pixels
    3. Calculate M1: sum of channel 1 intensities in overlap regions divided by
       total sum of channel 1 intensities above threshold
    4. Calculate M2: sum of channel 2 intensities in overlap regions divided by
       total sum of channel 2 intensities above threshold

    Both coefficients range from 0 (no colocalization) to 1 (perfect colocalization).

    Parameters
    ----------
    ch1 : np.ndarray
        First fluorescent channel (kept fixed)
    ch2 : np.ndarray
        Second fluorescent channel (will be rotated and flipped)
    threshold_ch1 : float
        Intensity threshold for channel 1
    threshold_ch2 : float
        Intensity threshold for channel 2

    Returns
    -------
    tuple[float, float]
        - M1 coefficient: fraction of A overlapping with B
        - M2 coefficient: fraction of B overlapping with A
    """
    # Apply thresholds and get overlap mask
    mask_a = ch1 > threshold_ch1
    mask_b = ch2 > threshold_ch2
    overlap_mask = mask_a & mask_b

    # Calculate M1: fraction of A overlapping with B
    m1_numerator = np.sum(ch1[overlap_mask])
    m1_denominator = np.sum(ch1[mask_a])
    m1 = m1_numerator / m1_denominator if m1_denominator > 0 else 0.0

    # Calculate M2: fraction of B overlapping with A
    m2_numerator = np.sum(ch2[overlap_mask])
    m2_denominator = np.sum(ch2[mask_b])
    m2 = m2_numerator / m2_denominator if m2_denominator > 0 else 0.0

    return m1, m2
