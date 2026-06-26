"""Test auto-threshold methods using synthetic test data."""

from __future__ import annotations

import numpy as np

from bobiac_tools import (
    AutoThresholdRegression,
    Implementation,
    fiji_bisection_auto_threshold,
    fiji_costes_auto_threshold,
    pca_auto_threshold,
)


def create_synthetic_test_data(size=(512, 512), noise_level=10):
    """Create synthetic test data for colocalization analysis."""
    np.random.seed(42)  # For reproducible tests

    # Create two channels with some correlation
    ch1 = np.random.randint(0, 100, size=size, dtype=np.uint16)
    ch2 = np.random.randint(0, 100, size=size, dtype=np.uint16)

    # Add some correlated regions
    center_x, center_y = size[0] // 2, size[1] // 2
    radius = min(size) // 4

    y, x = np.ogrid[: size[0], : size[1]]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

    # Add correlated signal in the center
    ch1[mask] += 150
    ch2[mask] += 120

    # Add some noise
    ch1 = ch1 + np.random.randint(0, noise_level, size=size)
    ch2 = ch2 + np.random.randint(0, noise_level, size=size)

    # Clip to valid range
    ch1 = np.clip(ch1, 0, 65535).astype(np.uint16)
    ch2 = np.clip(ch2, 0, 65535).astype(np.uint16)

    return ch1, ch2


def test_costes_auto_threshold():
    """Test COSTES auto-threshold implementation."""
    # Create synthetic test data
    ch1, ch2 = create_synthetic_test_data()

    # Test the convenience function
    ch1_thr, ch2_thr, slope, intercept = fiji_costes_auto_threshold(ch1, ch2)

    # Basic sanity checks
    assert isinstance(ch1_thr, (int, float)), "Ch1 threshold should be numeric"
    assert isinstance(ch2_thr, (int, float)), "Ch2 threshold should be numeric"
    assert isinstance(slope, (int, float)), "Slope should be numeric"
    assert isinstance(intercept, (int, float)), "Intercept should be numeric"

    # Thresholds should be non-negative and within reasonable range
    assert ch1_thr >= 0, "Ch1 threshold should be non-negative"
    assert ch2_thr >= 0, "Ch2 threshold should be non-negative"
    assert ch1_thr <= np.max(ch1), "Ch1 threshold should not exceed max pixel value"
    assert ch2_thr <= np.max(ch2), "Ch2 threshold should not exceed max pixel value"

    # Test with class interface
    auto_threshold = AutoThresholdRegression(Implementation.COSTES)
    ch1_thr2, ch2_thr2, slope2, intercept2 = auto_threshold.execute(ch1, ch2)

    # Results should be identical
    assert ch1_thr == ch1_thr2, (
        "Convenience function and class should give same ch1 threshold"
    )
    assert ch2_thr == ch2_thr2, (
        "Convenience function and class should give same ch2 threshold"
    )
    assert slope == slope2, "Convenience function and class should give same slope"
    assert intercept == intercept2, (
        "Convenience function and class should give same intercept"
    )

    print(
        f"COSTES: Ch1={ch1_thr:.1f}, Ch2={ch2_thr:.1f}, "
        f"slope={slope:.4f}, intercept={intercept:.1f}"
    )


def test_bisection_auto_threshold():
    """Test BISECTION auto-threshold implementation."""
    # Create synthetic test data
    ch1, ch2 = create_synthetic_test_data()

    # Test the convenience function
    ch1_thr, ch2_thr, slope, intercept = fiji_bisection_auto_threshold(ch1, ch2)

    # Basic sanity checks
    assert isinstance(ch1_thr, (int, float)), "Ch1 threshold should be numeric"
    assert isinstance(ch2_thr, (int, float)), "Ch2 threshold should be numeric"
    assert isinstance(slope, (int, float)), "Slope should be numeric"
    assert isinstance(intercept, (int, float)), "Intercept should be numeric"

    # Thresholds should be non-negative and within reasonable range
    assert ch1_thr >= 0, "Ch1 threshold should be non-negative"
    assert ch2_thr >= 0, "Ch2 threshold should be non-negative"
    assert ch1_thr <= np.max(ch1), "Ch1 threshold should not exceed max pixel value"
    assert ch2_thr <= np.max(ch2), "Ch2 threshold should not exceed max pixel value"

    # Test with class interface
    auto_threshold = AutoThresholdRegression(Implementation.BISECTION)
    ch1_thr2, ch2_thr2, slope2, intercept2 = auto_threshold.execute(ch1, ch2)

    # Results should be identical
    assert ch1_thr == ch1_thr2, (
        "Convenience function and class should give same ch1 threshold"
    )
    assert ch2_thr == ch2_thr2, (
        "Convenience function and class should give same ch2 threshold"
    )
    assert slope == slope2, "Convenience function and class should give same slope"
    assert intercept == intercept2, (
        "Convenience function and class should give same intercept"
    )

    print(
        f"BISECTION: Ch1={ch1_thr:.1f}, Ch2={ch2_thr:.1f}, "
        f"slope={slope:.4f}, intercept={intercept:.1f}"
    )


def test_pca_auto_threshold():
    """Test PCA auto-threshold implementation."""
    # Create synthetic test data
    ch1, ch2 = create_synthetic_test_data()

    # Test the convenience function
    ch1_thr, ch2_thr, slope, intercept = pca_auto_threshold(ch1, ch2)

    # Basic sanity checks
    assert isinstance(ch1_thr, (int, float)), "Ch1 threshold should be numeric"
    assert isinstance(ch2_thr, (int, float)), "Ch2 threshold should be numeric"
    assert isinstance(slope, (int, float)), "Slope should be numeric"
    assert isinstance(intercept, (int, float)), "Intercept should be numeric"

    # Thresholds should be non-negative and within reasonable range
    assert ch1_thr >= 0, "Ch1 threshold should be non-negative"
    assert ch2_thr >= 0, "Ch2 threshold should be non-negative"
    assert ch1_thr <= np.max(ch1), "Ch1 threshold should not exceed max pixel value"
    assert ch2_thr <= np.max(ch2), "Ch2 threshold should not exceed max pixel value"

    # Test with class interface
    auto_threshold = AutoThresholdRegression(Implementation.PCA)
    ch1_thr2, ch2_thr2, slope2, intercept2 = auto_threshold.execute(ch1, ch2)

    # Results should be identical
    assert ch1_thr == ch1_thr2, (
        "Convenience function and class should give same ch1 threshold"
    )
    assert ch2_thr == ch2_thr2, (
        "Convenience function and class should give same ch2 threshold"
    )
    assert slope == slope2, "Convenience function and class should give same slope"
    assert intercept == intercept2, (
        "Convenience function and class should give same intercept"
    )

    print(
        f"PCA: Ch1={ch1_thr:.1f}, Ch2={ch2_thr:.1f}, "
        f"slope={slope:.4f}, intercept={intercept:.1f}"
    )


def test_pca_symmetry():
    """Test that PCA method gives symmetric results when channels are swapped."""
    # Create synthetic test data
    ch1, ch2 = create_synthetic_test_data()

    # Test original order
    ch1_thr1, ch2_thr1, _slope1, _intercept1 = pca_auto_threshold(ch1, ch2)

    # Test swapped order
    ch2_thr2, ch1_thr2, _slope2, _intercept2 = pca_auto_threshold(ch2, ch1)

    # Thresholds should be swapped but otherwise identical
    assert np.isclose(ch1_thr1, ch1_thr2, rtol=1e-6), (
        "Ch1 threshold should be symmetric"
    )
    assert np.isclose(ch2_thr1, ch2_thr2, rtol=1e-6), (
        "Ch2 threshold should be symmetric"
    )

    print(
        f"PCA Symmetry test passed: Ch1_thr diff={abs(ch1_thr1 - ch1_thr2):.6f}, "
        f"Ch2_thr diff={abs(ch2_thr1 - ch2_thr2):.6f}"
    )


def test_auto_threshold_with_mask():
    """Test auto-threshold methods with a binary mask."""
    # Create synthetic test data
    ch1, ch2 = create_synthetic_test_data()

    # Create a simple mask (exclude border pixels)
    mask = np.ones_like(ch1, dtype=bool)
    mask[:10, :] = False  # Exclude top 10 rows
    mask[-10:, :] = False  # Exclude bottom 10 rows
    mask[:, :10] = False  # Exclude left 10 columns
    mask[:, -10:] = False  # Exclude right 10 columns

    # Test all three methods with mask
    ch1_thr_costes, ch2_thr_costes, _, _ = fiji_costes_auto_threshold(ch1, ch2, mask)
    ch1_thr_bisection, ch2_thr_bisection, _, _ = fiji_bisection_auto_threshold(
        ch1, ch2, mask
    )
    ch1_thr_pca, ch2_thr_pca, _, _ = pca_auto_threshold(ch1, ch2, mask)

    # All methods should return valid thresholds
    assert ch1_thr_costes >= 0 and ch2_thr_costes >= 0, (
        "COSTES with mask should give valid thresholds"
    )
    assert ch1_thr_bisection >= 0 and ch2_thr_bisection >= 0, (
        "BISECTION with mask should give valid thresholds"
    )
    assert ch1_thr_pca >= 0 and ch2_thr_pca >= 0, (
        "PCA with mask should give valid thresholds"
    )

    print(f"Masked COSTES: Ch1={ch1_thr_costes:.1f}, Ch2={ch2_thr_costes:.1f}")
    print(f"Masked BISECTION: Ch1={ch1_thr_bisection:.1f}, Ch2={ch2_thr_bisection:.1f}")
    print(f"Masked PCA: Ch1={ch1_thr_pca:.1f}, Ch2={ch2_thr_pca:.1f}")


def test_auto_threshold_comparison():
    """Compare auto-threshold methods and ensure they produce reasonable results."""
    # Create synthetic test data
    ch1, ch2 = create_synthetic_test_data()

    # Run all three methods
    ch1_cos, ch2_cos, _slope_cos, _intercept_cos = fiji_costes_auto_threshold(ch1, ch2)
    ch1_bic, ch2_bic, _slope_bic, _intercept_bic = fiji_bisection_auto_threshold(
        ch1, ch2
    )
    ch1_pca, ch2_pca, _slope_pca, _intercept_pca = pca_auto_threshold(ch1, ch2)

    # All methods should produce reasonable results (not too far apart)
    # This is a loose check - methods may differ but shouldn't be wildly different
    ch1_values = [ch1_cos, ch1_bic, ch1_pca]
    ch2_values = [ch2_cos, ch2_bic, ch2_pca]

    ch1_range = max(ch1_values) - min(ch1_values)
    ch2_range = max(ch2_values) - min(ch2_values)

    # Range shouldn't be more than max value (loose tolerance for synthetic data)
    assert ch1_range <= max(ch1_values), f"Ch1 threshold range too large: {ch1_range}"
    assert ch2_range <= max(ch2_values), f"Ch2 threshold range too large: {ch2_range}"

    print("Method comparison:")
    print(f"  COSTES:    Ch1={ch1_cos:.1f}, Ch2={ch2_cos:.1f}")
    print(f"  BISECTION: Ch1={ch1_bic:.1f}, Ch2={ch2_bic:.1f}")
    print(f"  PCA:       Ch1={ch1_pca:.1f}, Ch2={ch2_pca:.1f}")
    print(f"  Ch1 range: {ch1_range:.1f}, Ch2 range: {ch2_range:.1f}")


def test_auto_threshold_warnings():
    """Test that AutoThresholdRegression generates appropriate warnings."""
    # Create synthetic test data
    ch1, ch2 = create_synthetic_test_data()

    # Test warning generation for each method
    for implementation in [
        Implementation.COSTES,
        Implementation.BISECTION,
        Implementation.PCA,
    ]:
        auto_threshold = AutoThresholdRegression(implementation)
        auto_threshold.execute(ch1, ch2)

        # Check that warnings are accessible
        warnings = auto_threshold.get_warnings()
        assert isinstance(warnings, list), (
            f"{implementation.value} should return warnings as list"
        )

        # Print warnings if any
        if warnings:
            print(f"{implementation.value} warnings: {warnings}")
        else:
            print(f"{implementation.value}: No warnings generated")


if __name__ == "__main__":
    print("Testing auto-threshold methods with synthetic data...")

    test_costes_auto_threshold()
    test_bisection_auto_threshold()
    test_pca_auto_threshold()
    test_pca_symmetry()
    test_auto_threshold_with_mask()
    test_auto_threshold_comparison()
    test_auto_threshold_warnings()

    print("\nAll tests passed!")
