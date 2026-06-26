from __future__ import annotations

import numpy as np

from bobiac_tools import pixel_randomization

# Generate random synthetic image data
np.random.seed(42)
ch1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
ch2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# Run pixel randomization test
observed_pcc, random_pccs, p_value = pixel_randomization(
    ch1, ch2, n_iterations=100, seed=42
)

print("Costes Pixel Randomization Results:")
print(f"Observed Pearson correlation: {observed_pcc:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Mean random correlation: {np.mean(random_pccs):.4f}")
print(f"Std random correlation: {np.std(random_pccs):.4f}")
