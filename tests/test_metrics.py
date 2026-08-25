import numpy as np
import pytest

from fraudx.metrics import (
    evaluate_probabilities,
    expected_calibration_error,
    optimize_threshold,
)


def test_threshold_optimization_returns_valid_report() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.05, 0.10, 0.40, 0.45, 0.90])
    threshold, report = optimize_threshold(labels, scores, false_negative_cost=10)
    assert 0 < threshold < 1
    assert report.fn == 0
    assert report.business_cost >= 0


def test_perfect_calibration_has_small_error() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(labels, scores) == 0.0


def test_exact_threshold_matches_brute_force_cost() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.10, 0.21, 0.60, 0.20, 0.55, 0.90])
    _, report = optimize_threshold(
        labels,
        scores,
        false_negative_cost=7,
        false_positive_cost=2,
    )
    candidates = np.append(np.nextafter(scores.max(), np.inf), np.unique(scores))
    costs = [
        evaluate_probabilities(
            labels,
            scores,
            threshold=float(candidate),
            false_negative_cost=7,
            false_positive_cost=2,
        ).business_cost
        for candidate in candidates
    ]
    assert report.business_cost == min(costs)


def test_threshold_respects_alert_capacity() -> None:
    labels = np.array([0] * 90 + [1] * 10)
    scores = np.linspace(0, 1, 100)
    _, report = optimize_threshold(
        labels,
        scores,
        false_negative_cost=100,
        maximum_alert_rate=0.05,
    )
    assert report.alert_rate <= 0.05
    assert report.business_cost_per_1000 >= 0
    with pytest.raises(ValueError, match="between 0 and 1"):
        optimize_threshold(
            labels,
            scores,
            false_negative_cost=10,
            maximum_alert_rate=1.1,
        )


def test_threshold_tie_breaking_is_deterministic() -> None:
    labels = np.array([0, 1])
    scores = np.array([0.9, 0.8])
    first = optimize_threshold(
        labels,
        scores,
        false_negative_cost=1,
        false_positive_cost=1,
    )
    second = optimize_threshold(
        labels,
        scores,
        false_negative_cost=1,
        false_positive_cost=1,
    )
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[0] == 0.8
    assert first[1].recall == 1.0
