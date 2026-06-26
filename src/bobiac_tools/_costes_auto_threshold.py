from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class Implementation(Enum):
    """Implementation types for the auto threshold algorithm."""

    COSTES = "Costes"
    BISECTION = "Bisection"
    PCA = "PCA"


class Stepper(ABC):
    """Abstract base class for threshold stepping algorithms."""

    @abstractmethod
    def update(self, value: float) -> None:
        """Update the stepper with a correlation value."""
        pass

    @abstractmethod
    def get_value(self) -> float:
        """Get the current threshold value."""
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        """Check if the stepping algorithm has finished."""
        pass


class BisectionStepper(Stepper):
    """Try to converge a threshold based on an update value condition.

    If the update value is larger zero, the threshold is lowered by half the distance
    between the last thresholds. If the update value falls below zero or is not
    a number, the threshold is increased by such a half step.

    This matches Fiji's `BisectionStepper` implementation:
    https://github.com/fiji/Colocalisation_Analysis/blob/master/src/main/java/sc/fiji/coloc/algorithms/BisectionStepper.java
    """

    def __init__(self, threshold: float, last_threshold: float):
        self.threshold1 = threshold
        self.threshold2 = last_threshold
        self.the_diff = abs(self.threshold1 - self.threshold2)
        self.iterations = 0
        self.max_iterations = 100

    def update(self, value: float) -> None:
        """Update threshold by a bisection step."""
        # Update working thresholds for next iteration
        self.threshold2 = self.threshold1

        if math.isnan(value) or value < 0:
            # We went too far, increase by the absolute half
            self.threshold1 = self.threshold1 + self.the_diff * 0.5
        elif value > 0:
            # As long as r > 0 we go half the way down
            self.threshold1 = self.threshold1 - self.the_diff * 0.5

        # Update difference to last threshold
        self.the_diff = abs(self.threshold1 - self.threshold2)
        # Update iteration counter
        self.iterations += 1

    def get_value(self) -> float:
        """Get current threshold."""
        return self.threshold1

    def is_finished(self) -> bool:
        """Called to indicate if the stepper is finished.

        If the difference between both thresholds is < 1, we consider that as reasonable
        close to abort.
        """
        return self.iterations > self.max_iterations or self.the_diff < 1.0


class SimpleStepper(Stepper):
    """The simple stepper decrements a start threshold with every update() call.

    It is finished if the update value is not a number, below zero or larger than
    the last value. It also stops if the decremented threshold falls below one.

    This matches Fiji's `SimpleStepper` implementation:
    https://github.com/fiji/Colocalisation_Analysis/blob/master/src/main/java/sc/fiji/coloc/algorithms/SimpleStepper.java
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.current_value = 1.0
        self.last_value = float("inf")  # Use infinity instead of Double.MAX_VALUE
        self.finished = False

    def update(self, value: float) -> None:
        """Decrement the threshold if the stepper is not marked as finished.

        Rendering a stepper finished happens if value is not a number,
        below or equal zero or bigger than the last update value. The same
        thing happens if the internal threshold falls below one.
        """
        if not self.finished:
            # Remember current value and store new value
            self.last_value = self.current_value
            self.current_value = value
            # Decrement threshold
            self.threshold = self.threshold - 1.0

            # Stop if the threshold conditions are met
            self.finished = (
                math.isnan(value)
                or self.threshold < 1
                or value < 0.0001
                or value > self.last_value
            )

    def get_value(self) -> float:
        """Get the current threshold."""
        return self.threshold

    def is_finished(self) -> bool:
        """Indicates if the stepper is marked as finished."""
        return self.finished


class PCAStepper(Stepper):
    """PCA-based stepper for systematic threshold exploration along principal component.

    This implementation uses PCA to find the principal component direction and then
    systematically tests threshold points along this direction to find the global
    minimum correlation. It ensures symmetric results regardless of channel order.

    Unlike other steppers, this one requires special handling in the regression loop
    since it doesn't follow the traditional single-threshold stepping pattern.
    """

    def __init__(
        self, ch1_flat: np.ndarray, ch2_flat: np.ndarray, num_thresholds: int = 100
    ):
        """Initialize PCA stepper with data for PCA calculation.

        Parameters
        ----------
        ch1_flat : np.ndarray
            Flattened channel 1 data
        ch2_flat : np.ndarray
            Flattened channel 2 data
        num_thresholds : int
            Number of threshold values to test along the principal component
        """
        self.ch1_flat = ch1_flat
        self.ch2_flat = ch2_flat
        self.num_thresholds = num_thresholds

        # Filter out zero pixels if present
        if np.min(ch1_flat) == 0 or np.min(ch2_flat) == 0:
            mask = (ch1_flat > 0) & (ch2_flat > 0)
            self.ch1_masked = ch1_flat[mask]
            self.ch2_masked = ch2_flat[mask]
        else:
            self.ch1_masked = ch1_flat
            self.ch2_masked = ch2_flat

        if len(self.ch1_masked) == 0 or len(self.ch2_masked) == 0:
            self.finished = True
            self.best_ch1_threshold = 0
            self.best_ch2_threshold = 0
            return

        # Initialize iteration variables first
        self.current_index = 0
        self.best_correlation = 1.0
        self.best_ch1_threshold = 0
        self.best_ch2_threshold = 0
        self.current_ch1_threshold = 0
        self.current_ch2_threshold = 0
        self.finished = False

        # Perform PCA setup
        self._setup_pca()

    def _setup_pca(self) -> None:
        """Set up PCA and generate threshold candidates."""
        # Center the data
        self.ch1_mean = np.mean(self.ch1_masked)
        self.ch2_mean = np.mean(self.ch2_masked)
        ch1_centered = self.ch1_masked - self.ch1_mean
        ch2_centered = self.ch2_masked - self.ch2_mean

        # Perform PCA to find the orthogonal regression line
        data = np.vstack([ch1_centered, ch2_centered]).T
        cov_matrix = np.cov(data.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # The first principal component is the eigenvector with largest eigenvalue
        principal_component = eigenvectors[:, np.argmax(eigenvalues)]

        # Normalize the principal component
        self.pc_norm = principal_component / np.linalg.norm(principal_component)

        # Project all points onto the principal component line
        projections = ch1_centered * self.pc_norm[0] + ch2_centered * self.pc_norm[1]

        # Generate threshold parameters along the line
        proj_min, proj_max = np.min(projections), np.max(projections)
        self.t_values = np.linspace(proj_max, proj_min, self.num_thresholds)

        # Set initial threshold
        if len(self.t_values) > 0:
            self._advance_to_next_valid_threshold()

    def _advance_to_next_valid_threshold(self) -> None:
        """Advance to the next valid threshold in the sequence."""
        while self.current_index < len(self.t_values):
            t = self.t_values[self.current_index]
            ch1_thresh = self.ch1_mean + t * self.pc_norm[0]
            ch2_thresh = self.ch2_mean + t * self.pc_norm[1]

            # Check if threshold is within valid data range
            if (
                ch1_thresh >= np.min(self.ch1_masked)
                and ch1_thresh <= np.max(self.ch1_masked)
                and ch2_thresh >= np.min(self.ch2_masked)
                and ch2_thresh <= np.max(self.ch2_masked)
            ):
                self.current_ch1_threshold = ch1_thresh
                self.current_ch2_threshold = ch2_thresh
                return

            self.current_index += 1

        # No more valid thresholds
        self.finished = True

    def update(self, value: float) -> None:
        """Update with correlation value and track best threshold."""
        if not math.isnan(value) and abs(value) < abs(self.best_correlation):
            # Track best correlation (closest to zero)
            self.best_correlation = value
            self.best_ch1_threshold = self.current_ch1_threshold
            self.best_ch2_threshold = self.current_ch2_threshold

        # Move to next threshold
        self.current_index += 1
        if not self.is_finished():
            self._advance_to_next_valid_threshold()

    def get_value(self) -> float:
        """Get current working threshold (returns ch1 threshold for compatibility)."""
        return self.current_ch1_threshold

    def is_finished(self) -> bool:
        """Check if all thresholds have been tested."""
        return self.current_index >= len(self.t_values)

    def get_current_thresholds(self) -> tuple[float, float]:
        """Get current threshold pair (ch1, ch2)."""
        return self.current_ch1_threshold, self.current_ch2_threshold

    def get_best_thresholds(self) -> tuple[float, float]:
        """Get the best thresholds found so far."""
        return self.best_ch1_threshold, self.best_ch2_threshold


class ChannelMapper(ABC):
    """Interface for mapping working threshold to channel thresholds."""

    @abstractmethod
    def get_ch1_threshold(self, t: float) -> float:
        """Get the threshold for channel 1 based on the working threshold."""
        pass

    @abstractmethod
    def get_ch2_threshold(self, t: float) -> float:
        """Get the threshold for channel 2 based on the working threshold."""
        pass


class ChannelMapperCh1(ChannelMapper):
    """Channel mapper for when working threshold maps to channel 1."""

    def __init__(self, slope: float, intercept: float):
        self.slope = slope
        self.intercept = intercept

    def get_ch1_threshold(self, t: float) -> float:
        """Get the threshold for channel 1 based on the working threshold."""
        return t

    def get_ch2_threshold(self, t: float) -> float:
        """Get the threshold for channel 2 based on the working threshold."""
        return (t * self.slope) + self.intercept


class ChannelMapperCh2(ChannelMapper):
    """Channel mapper for when working threshold maps to channel 2."""

    def __init__(self, slope: float, intercept: float):
        self.slope = slope
        self.intercept = intercept

    def get_ch1_threshold(self, t: float) -> float:
        """Get the threshold for channel 1 based on the working threshold."""
        return (t - self.intercept) / self.slope

    def get_ch2_threshold(self, t: float) -> float:
        """Get the threshold for channel 2 based on the working threshold."""
        return t


class MissingPreconditionException(Exception):
    """Exception for missing preconditions."""

    pass


class AutoThresholdRegression:
    """A class implementing the automatic finding of a threshold based on regression.

    This is similar to Fiji's `AutoThresholdRegression` implementation:
    https://github.com/fiji/Colocalisation_Analysis/blob/master/src/main/java/sc/fiji/coloc/algorithms/BisectionStepper.java

    Three implementations are available:
    - BISECTION: Uses BisectionStepper for iterative threshold convergence (default).
    - COSTES: Uses SimpleStepper (matches original FIJI JACOP behavior)
    - PCA: Uses PCAStepper for systematic exploration along principal component
    """

    def __init__(self, implementation: Implementation = Implementation.BISECTION):
        self.implementation = implementation
        # The threshold for ratio of y-intercept : y-mean to raise a warning
        self.warn_y_intercept_to_y_mean_ratio_threshold = 0.01

        # Results
        self.auto_threshold_slope = 0.0
        self.auto_threshold_intercept = 0.0
        self.ch1_min_threshold = 0.0
        self.ch1_max_threshold = 0.0
        self.ch2_min_threshold = 0.0
        self.ch2_max_threshold = 0.0
        self.b_to_y_mean_ratio = 0.0

        # Warnings
        self.warnings: list[str] = []

    @staticmethod
    def clamp(val: float, min_val: float, max_val: float) -> float:
        """Clamp a value to a min or max value."""
        return max(min_val, min(val, max_val))

    def add_warning(self, title: str, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(f"{title}: {message}")

    def calculate_pearson_below_threshold(
        self,
        ch1: np.ndarray,
        ch2: np.ndarray,
        threshold_ch1: float,
        threshold_ch2: float,
    ) -> float:
        """Calculate Pearson correlation for pixels below thresholds.

        Uses NumPy's correlation function for efficient calculation.
        """
        # Create mask for pixels below thresholds
        mask = (ch1 < threshold_ch1) | (ch2 < threshold_ch2)

        if not np.any(mask):
            raise MissingPreconditionException("No pixels below thresholds")

        ch1_below = ch1[mask]
        ch2_below = ch2[mask]

        if len(ch1_below) < 2:
            raise MissingPreconditionException("Too few pixels below thresholds")

        # Use NumPy's correlation coefficient calculation
        correlation_matrix = np.corrcoef(ch1_below.ravel(), ch2_below.ravel())
        pearsons_r: float = correlation_matrix[0, 1]

        # Sanity check
        if np.isnan(pearsons_r) or np.isinf(pearsons_r):
            raise MissingPreconditionException("Numerical problem occurred")

        return pearsons_r

    def _prepare_data(
        self, ch1: np.ndarray, ch2: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flatten the arrays and apply mask if provided."""
        if mask is not None:
            ch1_flat = ch1[mask > 0]
            ch2_flat = ch2[mask > 0]
        else:
            ch1_flat = ch1.ravel()
            ch2_flat = ch2.ravel()
        return ch1_flat, ch2_flat

    def _calculate_regression_parameters(
        self, ch1_flat: np.ndarray, ch2_flat: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Calculate regression parameters using EXACT FIJI calculation."""
        # Calculate means
        ch1_mean = np.mean(ch1_flat)
        ch2_mean = np.mean(ch2_flat)
        combined_mean = ch1_mean + ch2_mean

        # Variables for summing up - EXACT FIJI style
        ch1_mean_diff_sum = 0.0
        ch2_mean_diff_sum = 0.0
        combined_mean_diff_sum = 0.0
        n = 0
        n_zero = 0

        # Iterate through all pixels - EXACT FIJI loop
        for i in range(len(ch1_flat)):
            ch1_val = float(ch1_flat[i])
            ch2_val = float(ch2_flat[i])
            combined_sum = ch1_val + ch2_val

            # Calculate the numerators for the variances
            ch1_mean_diff_sum += (ch1_val - ch1_mean) * (ch1_val - ch1_mean)
            ch2_mean_diff_sum += (ch2_val - ch2_mean) * (ch2_val - ch2_mean)
            combined_mean_diff_sum += (combined_sum - combined_mean) * (
                combined_sum - combined_mean
            )

            # Count only pixels that are above zero
            if (ch1_val + ch2_val) > 0.00001:
                n_zero += 1

            n += 1

        # Calculate variances - EXACT FIJI calculation
        ch1_variance = ch1_mean_diff_sum / (n - 1)
        ch2_variance = ch2_mean_diff_sum / (n - 1)
        combined_variance = combined_mean_diff_sum / (n - 1.0)

        # EXACT FIJI covariance calculation
        # http://mathworld.wolfram.com/Covariance.html
        # var(x+y) = var(x) + var(y) + 2*covar(x,y)
        # 2*covar(x,y) = var(x+y) - var(x) - var(y)
        ch1ch2_covariance = 0.5 * (combined_variance - (ch1_variance + ch2_variance))

        # Calculate regression parameters - EXACT FIJI calculation
        denom = 2 * ch1ch2_covariance
        num = (
            ch2_variance
            - ch1_variance
            + math.sqrt(
                (ch2_variance - ch1_variance) * (ch2_variance - ch1_variance)
                + (4 * ch1ch2_covariance * ch1ch2_covariance)
            )
        )

        slope = num / denom
        intercept = ch2_mean - slope * ch1_mean

        return float(slope), float(intercept), float(ch1_mean), float(ch2_mean)

    def _get_channel_bounds(self, channel_data: np.ndarray) -> tuple[float, float]:
        """Get min and max values for a channel."""
        return float(np.min(channel_data)), float(np.max(channel_data))

    def _create_mapper_and_stepper(
        self, slope: float, intercept: float, ch1_flat: np.ndarray, ch2_flat: np.ndarray
    ) -> tuple[ChannelMapper, Stepper]:
        """Create the appropriate channel mapper and stepper based on slope.

        Determines which channel to use for threshold walking based on slope value.
        """
        # Handle PCA implementation separately
        if self.implementation == Implementation.PCA:
            # For PCA, we create a dummy mapper since PCAStepper handles both thresholds
            mapper: ChannelMapper = ChannelMapperCh1(slope, intercept)
            stepper: Stepper = PCAStepper(ch1_flat, ch2_flat, num_thresholds=100)
            return mapper, stepper

        # Determine which channel to use for threshold walking - EXACT FIJI logic
        if slope > -1 and slope < 1.0:
            # Map working threshold to channel one
            mapper = ChannelMapperCh1(slope, intercept)
            min_ch1, max_ch1 = self._get_channel_bounds(ch1_flat)

            # Select stepper based on implementation
            if self.implementation == Implementation.BISECTION:
                stepper = BisectionStepper(abs(max_ch1 + min_ch1) * 0.5, max_ch1)
            else:  # COSTES
                stepper = SimpleStepper(max_ch1)
        else:
            # Map working threshold to channel two
            mapper = ChannelMapperCh2(slope, intercept)
            min_ch2, max_ch2 = self._get_channel_bounds(ch2_flat)

            # Select stepper based on implementation
            if self.implementation == Implementation.BISECTION:
                stepper = BisectionStepper(abs(max_ch2 + min_ch2) * 0.5, max_ch2)
            else:  # COSTES
                stepper = SimpleStepper(max_ch2)

        return mapper, stepper

    def _perform_threshold_regression(
        self,
        mapper: ChannelMapper,
        stepper: Stepper,
        ch1_flat: np.ndarray,
        ch2_flat: np.ndarray,
    ) -> tuple[float, float]:
        """Perform the threshold regression using the stepper and mapper."""
        # Handle PCA implementation separately
        if self.implementation == Implementation.PCA:
            return self._perform_pca_threshold_regression(stepper, ch1_flat, ch2_flat)

        # Get min and max values for clamping
        min_val = 0.0  # Assuming typical image data
        max_val = 65535.0  # Assuming 16-bit data, adjust as needed

        ch1_thresh_max = 0.0
        ch2_thresh_max = 0.0

        # Do regression - EXACT FIJI loop
        while not stepper.is_finished():
            # Round ch1 threshold and compute ch2 threshold
            ch1_thresh_max = round(mapper.get_ch1_threshold(stepper.get_value()))
            ch2_thresh_max = round(mapper.get_ch2_threshold(stepper.get_value()))

            # Clamp thresholds to valid range
            ch1_thresh_max = self.clamp(ch1_thresh_max, min_val, max_val)
            ch2_thresh_max = self.clamp(ch2_thresh_max, min_val, max_val)

            try:
                # Calculate Pearson's correlation below thresholds
                current_pearsons_r = self.calculate_pearson_below_threshold(
                    ch1_flat,
                    ch2_flat,
                    ch1_thresh_max,
                    ch2_thresh_max,
                )
                stepper.update(current_pearsons_r)
            except MissingPreconditionException:
                # Numerical problems within the Pearson's calculation
                stepper.update(float("nan"))

        return ch1_thresh_max, ch2_thresh_max

    def _perform_pca_threshold_regression(
        self,
        stepper: Stepper,
        ch1_flat: np.ndarray,
        ch2_flat: np.ndarray,
    ) -> tuple[float, float]:
        """Perform PCA-based threshold regression."""
        # Cast to PCAStepper for type safety
        pca_stepper = stepper
        if not isinstance(pca_stepper, PCAStepper):
            raise ValueError("Expected PCAStepper for PCA implementation")

        # PCA stepper handles the threshold iteration differently
        while not pca_stepper.is_finished():
            # Get current threshold pair from PCA stepper
            ch1_thresh, ch2_thresh = pca_stepper.get_current_thresholds()

            # Create mask for pixels below thresholds
            below_mask = (ch1_flat < ch1_thresh) & (ch2_flat < ch2_thresh)

            if np.sum(below_mask) < 10:  # Need minimum number of pixels
                pca_stepper.update(float("nan"))
                continue

            # Calculate correlation for pixels below threshold
            ch1_below = ch1_flat[below_mask]
            ch2_below = ch2_flat[below_mask]

            if len(ch1_below) > 1 and np.std(ch1_below) > 0 and np.std(ch2_below) > 0:
                try:
                    correlation = np.corrcoef(ch1_below.ravel(), ch2_below.ravel())[
                        0, 1
                    ]
                    pca_stepper.update(correlation)
                except Exception:
                    pca_stepper.update(float("nan"))
            else:
                pca_stepper.update(float("nan"))

        # Return the best thresholds found
        return pca_stepper.get_best_thresholds()

    def _store_results_and_warnings(
        self,
        ch1_thresh_max: float,
        ch2_thresh_max: float,
        slope: float,
        intercept: float,
        ch1_mean: float,
        ch2_mean: float,
        ch1_flat: np.ndarray,
        ch2_flat: np.ndarray,
    ) -> None:
        """Store final results and generate warnings."""
        min_val = 0.0
        max_val = 65535.0

        # Store regression parameters
        self.auto_threshold_slope = slope
        self.auto_threshold_intercept = intercept
        self.b_to_y_mean_ratio = intercept / ch2_mean

        # Store final thresholds
        self.ch1_min_threshold = min_val
        self.ch1_max_threshold = self.clamp(ch1_thresh_max, min_val, max_val)
        self.ch2_min_threshold = min_val
        self.ch2_max_threshold = self.clamp(ch2_thresh_max, min_val, max_val)

        # Add warnings if values are not in tolerance range
        if (
            abs(self.b_to_y_mean_ratio)
            > self.warn_y_intercept_to_y_mean_ratio_threshold
        ):
            self.add_warning(
                "y-intercept far from zero",
                "The ratio of the y-intercept of the auto threshold regression "
                + "line to the mean value of Channel 2 is high.",
            )

        # Add warning if threshold is above the image mean
        if ch1_thresh_max > ch1_mean:
            self.add_warning(
                "Threshold of ch. 1 too high",
                "Too few pixels are taken into account for above-threshold "
                "calculations.",
            )

        if ch2_thresh_max > ch2_mean:
            self.add_warning(
                "Threshold of ch. 2 too high",
                "Too few pixels are taken into account for above-threshold "
                "calculations.",
            )

        # Add warnings if values are below lowest pixel value of images
        if ch1_thresh_max < np.min(ch1_flat) or ch2_thresh_max < np.min(ch2_flat):
            msg = (
                "The auto threshold method could not find a positive threshold, "
                "so thresholded results are meaningless."
            )
            if self.implementation != Implementation.COSTES:
                msg += " Maybe you should try classic thresholding."
            self.add_warning("thresholds too low", msg)

    def execute(
        self, ch1: np.ndarray, ch2: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[float, float, float, float]:
        """
        Execute the auto threshold regression algorithm.

        Parameters
        ----------
        ch1 : np.ndarray
            First channel image data
        ch2 : np.ndarray
            Second channel image data
        mask : np.ndarray, optional
            Binary mask for the images

        Returns
        -------
        tuple[float, float, float, float]
            ch1_threshold, ch2_threshold, slope, intercept
        """
        # Clear previous warnings
        self.warnings = []

        # Prepare data
        ch1_flat, ch2_flat = self._prepare_data(ch1, ch2, mask)

        # Calculate regression parameters
        slope, intercept, ch1_mean, ch2_mean = self._calculate_regression_parameters(
            ch1_flat, ch2_flat
        )

        # Create mapper and stepper
        mapper, stepper = self._create_mapper_and_stepper(
            slope, intercept, ch1_flat, ch2_flat
        )

        # Perform threshold regression
        ch1_thresh_max, ch2_thresh_max = self._perform_threshold_regression(
            mapper, stepper, ch1_flat, ch2_flat
        )

        # Store results and generate warnings
        self._store_results_and_warnings(
            ch1_thresh_max,
            ch2_thresh_max,
            slope,
            intercept,
            ch1_mean,
            ch2_mean,
            ch1_flat,
            ch2_flat,
        )

        return (
            self.ch1_max_threshold,
            self.ch2_max_threshold,
            self.auto_threshold_slope,
            self.auto_threshold_intercept,
        )

    # Getter methods to match FIJI interface
    def get_auto_threshold_slope(self) -> float:
        """Get the calculated auto threshold slope."""
        return self.auto_threshold_slope

    def get_auto_threshold_intercept(self) -> float:
        """Get the calculated auto threshold intercept."""
        return self.auto_threshold_intercept

    def get_ch1_max_threshold(self) -> float:
        """Get the maximum threshold for channel 1."""
        return self.ch1_max_threshold

    def get_ch2_max_threshold(self) -> float:
        """Get the maximum threshold for channel 2."""
        return self.ch2_max_threshold

    def get_b_to_y_mean_ratio(self) -> float:
        """Get the blue to yellow mean ratio."""
        return self.b_to_y_mean_ratio

    def get_warnings(self) -> list[str]:
        """Get the list of warnings generated during threshold calculation."""
        return self.warnings


def fiji_costes_auto_threshold(
    ch1: np.ndarray, ch2: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, float, float, float]:
    """
    Convenience function that replicates FIJI JACOP's auto threshold behavior exactly.

    Uses COSTES implementation to match JACOP plugin behavior.

    Parameters
    ----------
    ch1 : np.ndarray
        First channel image data
    ch2 : np.ndarray
        Second channel image data
    mask : np.ndarray, optional
        Binary mask for the images

    Returns
    -------
    tuple[float, float, float, float]
        ch1_threshold, ch2_threshold, slope, intercept
    """
    auto_threshold = AutoThresholdRegression(Implementation.COSTES)
    return auto_threshold.execute(ch1, ch2, mask)


def fiji_bisection_auto_threshold(
    ch1: np.ndarray, ch2: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, float, float, float]:
    """
    Convenience function using BISECTION implementation for iterative convergence.

    Uses bisection algorithm to find optimal thresholds through iterative refinement.

    Parameters
    ----------
    ch1 : np.ndarray
        First channel image data
    ch2 : np.ndarray
        Second channel image data
    mask : np.ndarray, optional
        Binary mask for the images

    Returns
    -------
    tuple[float, float, float, float]
        ch1_threshold, ch2_threshold, slope, intercept
    """
    auto_threshold = AutoThresholdRegression(Implementation.BISECTION)
    return auto_threshold.execute(ch1, ch2, mask)


def pca_auto_threshold(
    ch1: np.ndarray, ch2: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, float, float, float]:
    """
    Convenience function using PCA implementation for systematic threshold exploration.

    Uses Principal Component Analysis to find the optimal regression line direction
    and systematically explores thresholds along this line to find the global minimum
    correlation. This approach ensures symmetric results regardless of channel order.

    Parameters
    ----------
    ch1 : np.ndarray
        First channel image data
    ch2 : np.ndarray
        Second channel image data
    mask : np.ndarray, optional
        Binary mask for the images

    Returns
    -------
    tuple[float, float, float, float]
        ch1_threshold, ch2_threshold, slope, intercept
    """
    auto_threshold = AutoThresholdRegression(Implementation.PCA)
    return auto_threshold.execute(ch1, ch2, mask)
