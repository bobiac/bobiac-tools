"""Tests for the auto-threshold steppers and ``AutoThresholdRegression`` internals."""

from __future__ import annotations

import numpy as np
import pytest

from bobiac_tools import (
    AutoThresholdRegression,
    BisectionStepper,
    Implementation,
    PCAStepper,
    SimpleStepper,
)
from bobiac_tools._costes_auto_threshold import MissingPreconditionException


# --------------------------------------------------------------------------- #
# Implementation enum
# --------------------------------------------------------------------------- #
def test_implementation_members():
    """The enum exposes the three supported implementations."""
    assert {i.value for i in Implementation} == {"Costes", "Bisection", "PCA"}


# --------------------------------------------------------------------------- #
# BisectionStepper
# --------------------------------------------------------------------------- #
def test_bisection_stepper_steps_down_then_up():
    """A positive value halves downward; a negative value steps back up."""
    s = BisectionStepper(threshold=100.0, last_threshold=200.0)
    assert s.get_value() == 100.0
    s.update(1.0)  # r > 0 -> step down by half the diff (100 -> 50)
    assert s.get_value() == 50.0
    s.update(-1.0)  # r < 0 -> step up by half the new diff (50 -> 75)
    assert s.get_value() == 75.0


def test_bisection_stepper_nan_steps_up():
    """A NaN correlation is treated like 'too far' and steps the threshold up."""
    s = BisectionStepper(threshold=100.0, last_threshold=200.0)
    s.update(float("nan"))
    assert s.get_value() == 150.0


def test_bisection_stepper_finishes_on_small_diff():
    """A sub-unit gap between thresholds marks the stepper finished."""
    s = BisectionStepper(threshold=10.0, last_threshold=10.4)
    assert s.is_finished()


def test_bisection_stepper_eventually_finishes():
    """Repeated steps converge so the stepper finishes."""
    s = BisectionStepper(threshold=0.0, last_threshold=1024.0)
    for _ in range(200):
        if s.is_finished():
            break
        s.update(1.0)
    assert s.is_finished()


# --------------------------------------------------------------------------- #
# SimpleStepper
# --------------------------------------------------------------------------- #
def test_simple_stepper_decrements_threshold():
    """Each update lowers the threshold by one while not finished."""
    s = SimpleStepper(threshold=10.0)
    assert s.get_value() == 10.0
    s.update(0.5)
    assert s.get_value() == 9.0
    assert not s.is_finished()


def test_simple_stepper_finishes_when_value_too_small():
    """A correlation below 0.0001 finishes the stepper."""
    s = SimpleStepper(threshold=10.0)
    s.update(0.5)
    s.update(0.00001)
    assert s.is_finished()


def test_simple_stepper_finishes_when_correlation_rises():
    """A correlation larger than the previous one finishes the stepper."""
    s = SimpleStepper(threshold=10.0)
    s.update(0.3)
    s.update(0.6)  # 0.6 > previous 0.3 -> finished
    assert s.is_finished()


def test_simple_stepper_finishes_on_nan():
    """A NaN correlation finishes the stepper."""
    s = SimpleStepper(threshold=10.0)
    s.update(float("nan"))
    assert s.is_finished()


# --------------------------------------------------------------------------- #
# PCAStepper
# --------------------------------------------------------------------------- #
def test_pca_stepper_finished_when_no_valid_pixels():
    """Disjoint nonzero supports leave no valid pixels, so the stepper finishes."""
    ch1 = np.array([1.0, 0.0, 2.0, 0.0])
    ch2 = np.array([0.0, 1.0, 0.0, 2.0])
    s = PCAStepper(ch1, ch2, num_thresholds=10)
    assert s.is_finished()
    assert s.get_best_thresholds() == (0, 0)


def test_pca_stepper_tracks_best_threshold():
    """The best threshold corresponds to the smallest absolute correlation seen."""
    rng = np.random.default_rng(0)
    ch1 = rng.random(200) + 0.5
    ch2 = ch1 * 1.5 + rng.random(200) * 0.05
    s = PCAStepper(ch1, ch2, num_thresholds=30)
    assert not s.is_finished()

    seen = []
    while not s.is_finished():
        seen.append(s.get_current_thresholds())
        s.update(0.5)  # constant value -> first visited pair stays best

    assert s.get_best_thresholds() == seen[0]


# --------------------------------------------------------------------------- #
# AutoThresholdRegression
# --------------------------------------------------------------------------- #
def test_clamp():
    """Clamp bounds a value between the given min and max."""
    assert AutoThresholdRegression.clamp(5, 0, 10) == 5
    assert AutoThresholdRegression.clamp(-1, 0, 10) == 0
    assert AutoThresholdRegression.clamp(11, 0, 10) == 10


def test_regression_parameters_recover_linear_slope():
    """For ch2 = 2*ch1 the total-least-squares slope is 2 and intercept 0."""
    ch1 = np.arange(100, dtype=float)
    ch2 = 2.0 * ch1
    atr = AutoThresholdRegression()
    slope, intercept, ch1_mean, ch2_mean = atr._calculate_regression_parameters(
        ch1, ch2
    )
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(0.0, abs=1e-9)
    assert ch1_mean == pytest.approx(ch1.mean())
    assert ch2_mean == pytest.approx(ch2.mean())


def test_pearson_below_threshold_raises_when_none_below():
    """No pixels below the thresholds raises MissingPreconditionException."""
    atr = AutoThresholdRegression()
    ch1 = np.array([5.0, 6.0, 7.0])
    ch2 = np.array([5.0, 6.0, 7.0])
    with pytest.raises(MissingPreconditionException):
        atr.calculate_pearson_below_threshold(ch1, ch2, 0.0, 0.0)


def test_pearson_below_threshold_returns_correlation():
    """With all pixels below threshold, identical channels correlate at 1.0."""
    atr = AutoThresholdRegression()
    ch1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ch2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = atr.calculate_pearson_below_threshold(ch1, ch2, 100.0, 100.0)
    assert r == pytest.approx(1.0)


def test_get_warnings_returns_list():
    """Running execute populates a list of warning strings."""
    atr = AutoThresholdRegression()
    ch1 = np.arange(50.0).reshape(5, 10)
    ch2 = ch1 * 2
    atr.execute(ch1, ch2)
    assert isinstance(atr.get_warnings(), list)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
