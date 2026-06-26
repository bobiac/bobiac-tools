from __future__ import annotations

import numpy as np

from bobiac_tools import manders_image_translation_randomization

# Generate random synthetic image data
np.random.seed(42)
ch1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
ch2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# Define thresholds
threshold_ch1 = float(np.percentile(ch1, 50))
threshold_ch2 = float(np.percentile(ch2, 50))

# Run translation randomization test
observed_m1, observed_m2, random_m1s, random_m2s, p_value_m1, p_value_m2 = (
    manders_image_translation_randomization(
        ch1, ch2, threshold_ch1, threshold_ch2, n_iterations=100, seed=42
    )
)

print("Manders Image Translation Randomization Results:")
print(f"Observed M1: {observed_m1:.4f}")
print(f"Observed M2: {observed_m2:.4f}")
print(f"P-value M1: {p_value_m1:.4f}")
print(f"P-value M2: {p_value_m2:.4f}")
print(f"Mean random M1: {np.mean(random_m1s):.4f}")
print(f"Mean random M2: {np.mean(random_m2s):.4f}")
