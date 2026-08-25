"""Tests for the fixed-fold repeated-seed experiment."""

from __future__ import annotations

import pandas as pd
import pytest

from fraudx.multiseed import (
    make_primary_pairwise,
    summarize_multiseed_results,
    summarize_win_rates,
    validate_multiseed_results,
)


def _results() -> pd.DataFrame:
    rows = []
    for fold in [1, 2]:
        for model_id, scheme, weight in [
            ("raw_safe", "Raw", 1.0),
            ("recipient_full_weight_2", "Recipient history (w=2)", 2.0),
        ]:
            for seed in [42, 52]:
                recipient = model_id != "raw_safe"
                tp = 4 if recipient else 2
                fn = 1 if recipient else 3
                fp = 6 if recipient else 3
                tn = 89 if recipient else 92
                business_cost = fn * 100 + fp
                rows.append(
                    {
                        "fold": fold,
                        "scheme": scheme,
                        "model_id": model_id,
                        "feature_set": "recipient_full" if recipient else "raw_safe",
                        "weight": weight,
                        "seed": seed,
                        "pr_auc": 0.1 if recipient else 0.2,
                        "precision": tp / (tp + fp),
                        "recall": tp / (tp + fn),
                        "f1": 0.5,
                        "threshold": 0.01 + seed / 100000,
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "alert_rate": (tp + fp) / 100,
                        "capacity_deviation": (tp + fp) / 100 - 0.03,
                        "false_negative_cost": fn * 100,
                        "false_positive_cost": fp,
                        "business_cost": business_cost,
                        "all_normal_cost": 500,
                        "cost_saving": 500 - business_cost,
                        "cost_saving_rate": (500 - business_cost) / 500,
                        "test_size": 100,
                        "test_fraud_count": 5,
                        "test_fraud_rate": 0.05,
                        "fn_cost": 100.0,
                        "train_min_step": 1,
                        "train_max_step": 10 * fold,
                        "validation_min_step": 11 * fold,
                        "validation_max_step": 12 * fold,
                        "test_min_step": 13 * fold,
                        "test_max_step": 14 * fold,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_has_threshold_cv_and_max_capacity_deviation() -> None:
    summary = summarize_multiseed_results(_results())
    assert len(summary) == 4
    assert "threshold_coefficient_of_variation" in summary
    assert "capacity_deviation_max_abs" in summary


def test_pairwise_uses_same_fold_and_seed_with_documented_directions() -> None:
    paired = make_primary_pairwise(_results())
    assert len(paired) == 4
    assert (paired["delta_recall"] > 0).all()
    assert (paired["delta_pr_auc"] < 0).all()
    assert (paired["business_cost_reduction"] > 0).all()


def test_win_rates_include_each_fold_and_overall() -> None:
    wins = summarize_win_rates(make_primary_pairwise(_results()))
    overall = wins.loc[wins["scope"] == "Overall"].set_index("metric")
    assert overall.loc["recall", "win_count"] == 4
    assert overall.loc["business_cost", "win_rate"] == 1.0
    assert overall.loc["pr_auc", "win_rate"] == 0.0


def test_validation_accepts_fixed_partitions_and_complete_seed_grid() -> None:
    checks = validate_multiseed_results(
        _results(), expected_rows=8, expected_seeds=[42, 52], expected_schemes=2
    )
    assert checks["fixed_partition_check"] is True


def test_validation_rejects_partition_change_between_seeds() -> None:
    results = _results()
    results.loc[1, "test_size"] = 101
    with pytest.raises(AssertionError, match="boundaries changed"):
        validate_multiseed_results(
            results, expected_rows=8, expected_seeds=[42, 52], expected_schemes=2
        )
