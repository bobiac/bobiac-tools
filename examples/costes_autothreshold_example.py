from __future__ import annotations

import numpy as np

from bobiac_tools import (
    fiji_bisection_auto_threshold,
    fiji_costes_auto_threshold,
    pca_auto_threshold,
)

# Generate random synthetic image data
np.random.seed(42)
ch1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
ch2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

ch1_thr_pca, ch2_thr_pca, slope_pca, intercept_pca = pca_auto_threshold(ch1, ch2)
ch1_thr_bic, ch2_thr_bic, slope_bic, intercept_bic = fiji_bisection_auto_threshold(
    ch1, ch2
)
ch1_thr_cos, ch2_thr_cos, slope_cos, intercept_cos = fiji_costes_auto_threshold(
    ch1, ch2
)

print("\nPCA Method Results:")
print(f"y = {slope_pca:.4f} * x + {intercept_pca:.4f}")
print(f"Ch1 Threshold: {ch1_thr_pca}, Ch2 Threshold: {ch2_thr_pca}")
print("\nBisection Method Results:")
print(f"y = {slope_bic:.4f} * x + {intercept_bic:.4f}")
print(f"Ch1 Threshold: {ch1_thr_bic}, Ch2 Threshold: {ch2_thr_bic}")
print("\nCostes Method Results:")
print(f"y = {slope_cos:.4f} * x + {intercept_cos:.4f}")
print(f"Ch1 Threshold: {ch1_thr_cos}, Ch2 Threshold: {ch2_thr_cos}")
