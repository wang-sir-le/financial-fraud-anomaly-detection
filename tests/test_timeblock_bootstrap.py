from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

import fraudx.timeblock_bootstrap as bootstrap_module
from fraudx.timeblock_bootstrap import (
    METRICS,
    _observed_effects,
    _overall_equal_weight_replicates,
    _safe_divide,
    _seed_bootstrap_effects,
    _summary_row,
    block_length_for_steps,
    bootstrap_average_precision,
    moving_block_positions,
    position_multiplicities,
)


def test_block_length_rule() -> None:
    assert block_length_for_steps(43) == 4
    assert block_length_for_steps(31) == 3
    assert block_length_for_steps(345) == 7
    with pytest.raises(ValueError):
        block_length_for_steps(1)


def test_moving_blocks_are_contiguous_overlapping_non_circular_and_truncated() -> None:
    rng = np.random.default_rng(7)
    positions = moving_block_positions(7, 3, 50, rng)
    assert positions.shape == (50, 7)
    assert positions.min() >= 0
    assert positions.max() <= 6
    for row in positions:
        assert np.all(np.diff(row[:3]) == 1)
        assert np.all(np.diff(row[3:6]) == 1)
    # T-L+1=5 legal starts; start 4 produces [4,5,6] and never wraps to zero.
    assert not np.any(
        np.all(positions[:, :3] == np.array([5, 6, 0], dtype=np.int32), axis=1)
    )


def test_moving_block_seed_is_reproducible_and_child_streams_differ() -> None:
    sequence = np.random.SeedSequence(20260820)
    first, second = sequence.spawn(2)
    one = moving_block_positions(20, 3, 10, np.random.default_rng(first))
    sequence_again = np.random.SeedSequence(20260820)
    first_again, _ = sequence_again.spawn(2)
    repeated = moving_block_positions(20, 3, 10, np.random.default_rng(first_again))
    other = moving_block_positions(20, 3, 10, np.random.default_rng(second))
    assert np.array_equal(one, repeated)
    assert not np.array_equal(one, other)


def test_position_multiplicity_repeats_all_step_transactions() -> None:
    sampled = np.array([[0, 1, 0, 2]], dtype=np.int32)
    multiplicities = position_multiplicities(sampled, 4)
    assert multiplicities.tolist() == [[2, 1, 1, 0]]
    transaction_counts = np.array([3, 2, 5, 7])
    assert int(multiplicities[0] @ transaction_counts) == 13


def test_bootstrap_average_precision_matches_weighted_sklearn() -> None:
    step_positions = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    labels = np.array([1, 0, 0, 1, 1, 0], dtype=np.int8)
    scores = np.array([0.9, 0.2, 0.6, 0.6, 0.7, 0.1])
    multiplicities = np.array([[1, 1, 1], [2, 0, 1], [0, 2, 1]], dtype=np.int16)
    actual = bootstrap_average_precision(
        multiplicities, step_positions, labels, scores, batch_size=2
    )
    expected = np.array(
        [
            average_precision_score(
                labels, scores, sample_weight=weights[step_positions]
            )
            for weights in multiplicities
        ]
    )
    assert np.allclose(actual, expected, atol=1e-14)


def test_effect_directions_cost_scaling_and_shared_sample() -> None:
    data = pd.DataFrame(
        {
            "y_true": [1, 0, 1, 0],
            "raw_binary_decision": [0, 0, 0, 0],
            "recipient_binary_decision": [1, 0, 1, 1],
            "raw_pr_auc_score": [0.4, 0.3, 0.2, 0.1],
            "recipient_pr_auc_score": [0.9, 0.2, 0.8, 0.1],
        }
    )
    positions = np.array([0, 0, 1, 1], dtype=np.int32)
    multiplicities = np.array([[1, 1], [2, 0]], dtype=np.int16)
    counts = multiplicities @ np.array([2, 2])
    fraud = multiplicities @ np.array([1, 1])
    effects = _seed_bootstrap_effects(
        data,
        step_positions=positions,
        multiplicities=multiplicities,
        bootstrap_transaction_count=counts,
        bootstrap_fraud_count=fraud,
    )
    assert np.all(effects["delta_recall"] > 0)
    assert np.all(effects["cost_reduction"] > 0)
    assert np.all(effects["cost_reduction_per_100k"] > 0)
    assert np.all(effects["delta_cost_saving_rate"] > 0)
    assert np.all(effects["delta_pr_auc"] >= 0)
    assert np.any(effects["delta_pr_auc"] > 0)
    assert effects["cost_reduction_per_100k"][0] == pytest.approx(
        effects["cost_reduction"][0] / counts[0] * 100000
    )


def test_observed_effect_uses_raw_minus_recipient_cost() -> None:
    data = pd.DataFrame(
        {
            "y_true": [1, 1, 0],
            "raw_binary_decision": [0, 0, 0],
            "recipient_binary_decision": [1, 0, 1],
            "raw_pr_auc_score": [0.4, 0.3, 0.2],
            "recipient_pr_auc_score": [0.9, 0.3, 0.8],
        }
    )
    observed = _observed_effects(data)
    assert observed["delta_recall"] == 0.5
    assert observed["cost_reduction"] == 99.0
    assert observed["delta_cost_saving_rate"] == pytest.approx(99 / 200)


def test_zero_fraud_produces_nan_without_redraw() -> None:
    numerator = np.array([0.0, 1.0])
    denominator = np.array([0.0, 2.0])
    result = _safe_divide(numerator, denominator)
    assert np.isnan(result[0])
    assert result[1] == 0.5


def test_percentile_ci_standard_error_and_valid_counts(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap_module, "N_BOOTSTRAP", 5)
    values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    row = _summary_row(scope=1, metric="delta_recall", observed_effect=2.0, values=values)
    finite = np.array([1.0, 2.0, 4.0, 5.0])
    assert row["valid_replicates"] == 4
    assert row["invalid_replicates"] == 1
    assert row["bootstrap_se"] == pytest.approx(np.std(finite, ddof=1))
    assert row["ci_lower_95"] == pytest.approx(np.percentile(finite, 2.5))
    assert row["ci_upper_95"] == pytest.approx(np.percentile(finite, 97.5))


def test_overall_effect_is_equal_fold_weight_not_transaction_weight() -> None:
    rows = []
    for fold, value in ((1, 1.0), (2, 2.0), (3, 9.0)):
        rows.append(
            {
                "bootstrap_id": 1,
                "fold": fold,
                **{metric: value for metric in METRICS},
            }
        )
    overall = _overall_equal_weight_replicates(pd.DataFrame(rows))
    assert overall.loc[0, "delta_recall"] == 4.0
    assert overall.loc[0, "weighting_method"] == "equal_fold_weight"


def test_bootstrap_module_cannot_refit_or_reselect() -> None:
    source = inspect.getsource(bootstrap_module)
    assert ".fit(" not in source
    assert "optimize_threshold" not in source
    assert "build_lightgbm_pipeline" not in source
