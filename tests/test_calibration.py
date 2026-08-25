"""Tests for the three-layer calibration experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from fraudx.calibration import (
    CALIBRATION_METHODS,
    CALIBRATION_MODELS,
    IdentityCalibrator,
    _array_hash,
    _fit_calibrators,
)
from fraudx.calibration_metrics import (
    bayes_capacity_decisions,
    binary_decision_metrics,
    jaccard_similarity,
    probability_metrics,
    reliability_bins,
    threshold_raw_score_percentile,
    tie_statistics,
)
from fraudx.metrics import optimize_threshold
from fraudx.models import PlattCalibrator


def test_uncalibrated_branch_is_an_identity_without_fitted_state() -> None:
    scores = np.asarray([0.1, 0.2, 0.9])
    calibrator = IdentityCalibrator()
    transformed = calibrator.transform(scores)
    assert np.array_equal(transformed, scores)
    assert transformed is not scores
    assert vars(calibrator) == {}


def test_calibrators_fit_only_passed_calibration_subset() -> None:
    raw = np.asarray([0.01, 0.02, 0.2, 0.8, 0.9, 0.95])
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    branches = _fit_calibrators(raw, labels)
    assert set(branches) == set(CALIBRATION_METHODS)
    isotonic = branches["Isotonic"]
    thresholds = isotonic.X_thresholds_  # type: ignore[attr-defined]
    assert thresholds.min() == raw.min()
    assert thresholds.max() == raw.max()
    assert set(thresholds).issubset(set(raw))


def test_platt_is_monotone_and_preserves_rank() -> None:
    raw = np.linspace(0.001, 0.9, 50)
    platt = PlattCalibrator(slope=2.0, intercept=-1.0).transform(raw)
    assert (np.diff(platt) > 0).all()
    assert spearmanr(raw, platt).statistic == pytest.approx(1.0)


def test_empirical_threshold_decisions_are_invariant_under_strict_mapping() -> None:
    raw = np.linspace(0.001, 0.99, 200)
    labels = np.zeros(200, dtype=int)
    labels[[160, 170, 180, 190, 199]] = 1
    mapped = 1 / (1 + np.exp(-(3 * raw - 1)))
    raw_threshold, _ = optimize_threshold(
        labels, raw, false_negative_cost=100, maximum_alert_rate=0.1
    )
    mapped_threshold, _ = optimize_threshold(
        labels, mapped, false_negative_cost=100, maximum_alert_rate=0.1
    )
    assert np.array_equal(raw >= raw_threshold, mapped >= mapped_threshold)


def test_uniform_ece_has_empty_bins_and_positive_counts() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.01, 0.02, 0.03, 0.04])
    points, diagnostics = reliability_bins(labels, scores, method="uniform", n_bins=10)
    assert len(points) == 10
    assert diagnostics.nonempty_bin_count == 1
    assert diagnostics.empty_bin_count == 9
    assert points["positive_count"].sum() == 2
    assert points["sample_count"].sum() == 4


def test_quantile_ece_uses_all_samples_and_is_bounded() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.01, 0.02, 0.03, 0.4, 0.5, 0.6])
    points, diagnostics = reliability_bins(labels, scores, method="quantile", n_bins=3)
    assert points["sample_count"].sum() == len(labels)
    assert 0 <= diagnostics.ece <= 1


def test_probability_metrics_match_manual_brier_and_keep_log_loss_finite() -> None:
    labels = np.asarray([0, 1])
    scores = np.asarray([0.0, 1.0])
    metrics = probability_metrics(labels, scores)
    assert metrics["brier"] == 0.0
    assert np.isfinite(metrics["log_loss"])


def test_isotonic_compresses_outputs_and_tie_statistics_are_correct() -> None:
    raw = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    labels = np.asarray([0, 0, 1, 0, 1, 1])
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(raw, labels)
    outputs = isotonic.transform(raw)
    stats = tie_statistics(outputs)
    assert stats["unique_output_count"] < len(raw)
    assert stats["tie_fraction"] > 0
    assert stats["maximum_tie_group_size"] >= 2


def test_threshold_percentile_and_jaccard() -> None:
    raw = np.asarray([0.1, 0.2, 0.3, 0.4])
    selected = np.asarray([False, False, True, True])
    assert threshold_raw_score_percentile(raw, selected) == pytest.approx(0.75)
    assert jaccard_similarity(selected, selected) == 1.0


def test_bayes_threshold_is_one_over_101() -> None:
    fn_cost = 100.0
    fp_cost = 1.0
    assert fp_cost / (fp_cost + fn_cost) == pytest.approx(1 / 101)


def test_bayes_capacity_cap_and_tie_breaking_are_deterministic() -> None:
    scores = np.asarray([0.5, 0.5, 0.5, 0.4, 0.1])
    raw = np.asarray([0.2, 0.3, 0.3, 0.9, 0.1])
    indices = np.asarray([9, 8, 7, 6, 5])
    decisions, info = bayes_capacity_decisions(
        scores,
        raw,
        threshold=0.01,
        capacity=0.4,
        stable_indices=indices,
    )
    assert info["capacity_binding"] is True
    assert info["final_alert_count"] == 2
    assert np.array_equal(np.flatnonzero(decisions), np.asarray([1, 2]))


def test_binary_decision_cost_and_capacity_fields() -> None:
    labels = np.asarray([1, 1, 0, 0])
    decisions = np.asarray([1, 0, 1, 0])
    metrics = binary_decision_metrics(labels, decisions, fn_cost=100, fp_cost=1)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["business_cost"] == 101
    assert metrics["cost_saving"] == 99


def test_shared_raw_hash_and_theoretical_grid_size() -> None:
    raw = np.asarray([0.01, 0.2, 0.9])
    frame = pd.DataFrame(
        {
            "method": CALIBRATION_METHODS,
            "raw_hash": [_array_hash(raw)] * len(CALIBRATION_METHODS),
        }
    )
    assert frame["raw_hash"].nunique() == 1
    assert 3 * len(CALIBRATION_MODELS) * 5 * len(CALIBRATION_METHODS) == 90
