from __future__ import annotations

import numpy as np

from bobiac_tools import manders_correlation_coefficient

# Generate random synthetic image data
np.random.seed(42)
ch1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
ch2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# Define thresholds (you might want to use auto-threshold methods to get these)
threshold_ch1 = float(np.percentile(ch1, 50))  # Use median as threshold
threshold_ch2 = float(np.percentile(ch2, 50))  # Use median as threshold

# Calculate Manders coefficients
m1, m2 = manders_correlation_coefficient(ch1, ch2, threshold_ch1, threshold_ch2)

print("Manders Correlation Coefficient Results:")
print(f"Channel 1 threshold: {threshold_ch1:.2f}")
print(f"Channel 2 threshold: {threshold_ch2:.2f}")
print(f"M1 coefficient: {m1:.4f}")
print(f"M2 coefficient: {m2:.4f}")
print(f"Average M coefficient: {(m1 + m2) / 2:.4f}")
