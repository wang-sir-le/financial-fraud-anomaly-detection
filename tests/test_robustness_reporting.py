from pathlib import Path

import numpy as np
import pandas as pd

from fraudx.config import load_config
from fraudx.metrics import MetricReport, evaluate_probabilities
from fraudx.robustness import run_rolling_validation
from fraudx.robustness_reporting import (
    cost_saving_fields,
    pairwise_comparisons,
    partition_profile,
    validate_full_results,
)


def test_cost_saving_decomposition_and_capacity_deviation() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.4, 0.9])
    report = evaluate_probabilities(
        labels,
        scores,
        threshold=0.5,
        false_negative_cost=10,
        false_positive_cost=1,
    )
    fields = cost_saving_fields(
        report,
        fn_cost=10,
        fp_cost=1,
        target_capacity=0.25,
        test_size=4,
        test_fraud_count=2,
    )
    assert fields["false_negative_cost"] == 10
    assert fields["false_positive_cost"] == 1
    assert fields["business_cost"] == 11
    assert fields["all_normal_cost"] == 20
    assert fields["cost_saving"] == 9
    assert fields["cost_saving_rate"] == 0.45
    assert fields["capacity_deviation"] == 0.25
    assert fields["capacity_deviation_abs"] == 0.25


def test_zero_fraud_all_normal_baseline_is_safe() -> None:
    report = MetricReport(
        accuracy=1.0,
        balanced_accuracy=0.5,
        precision=0.0,
        recall=0.0,
        specificity=1.0,
        false_positive_rate=0.0,
        alert_rate=0.0,
        prevalence=0.0,
        f1=0.0,
        pr_auc=0.0,
        roc_auc=0.0,
        mcc=0.0,
        brier=0.0,
        ece=0.0,
        tn=4,
        fp=0,
        fn=0,
        tp=0,
        threshold=0.5,
        business_cost=0.0,
        business_cost_per_1000=0.0,
    )
    fields = cost_saving_fields(
        report,
        fn_cost=100,
        fp_cost=1,
        target_capacity=0.03,
        test_size=4,
        test_fraud_count=0,
    )
    assert fields["all_normal_cost"] == 0
    assert np.isnan(fields["cost_saving_rate"])


def test_partition_profile_separates_overall_and_fraud_type_shares() -> None:
    frame = pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "type": ["PAYMENT", "TRANSFER", "TRANSFER", "CASH_OUT"],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "isFraud": [0, 1, 1, 0],
        }
    )
    profile = partition_profile(frame, target="isFraud", time_column="step")
    assert profile["sample_count"] == 4
    assert profile["fraud_count"] == 2
    assert profile["type_share_overall"]["TRANSFER"] == 0.5
    assert profile["type_share_among_fraud"]["TRANSFER"] == 1.0
    assert profile["amount_fraud"]["median"] == 25.0


def test_rolling_smoke_writes_complete_scenario_matrix(tmp_path: Path) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = load_config(config_path)
    raw = dict(config.raw)
    raw["output_dir"] = str(tmp_path)
    raw["robustness"] = {
        "folds": 2,
        "initial_train_fraction": 0.4,
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
        "false_negative_costs": [10.0, 100.0],
        "maximum_alert_rates": [0.01, 0.05],
        "primary_false_negative_cost": 100.0,
        "primary_maximum_alert_rate": 0.05,
        "models": [
            {"id": "raw_safe", "feature_group": "raw_safe", "weight": 1.0},
            {
                "id": "recipient_full_weight_2",
                "feature_group": "recipient_full",
                "weight": 2.0,
            },
        ],
    }
    object.__setattr__(config, "output_dir", tmp_path)
    object.__setattr__(config, "raw", raw)
    results = run_rolling_validation(config, smoke=True)
    assert len(results) == 16
    assert not results.duplicated(
        ["fold", "model_id", "weight", "fn_cost", "fp_cost", "target_capacity"]
    ).any()
    ranking_groups = results.groupby(["fold", "model_id", "weight"])["pr_auc"]
    assert (ranking_groups.nunique() == 1).all()
    validate_full_results(results, expected_rows=16)
    robustness = tmp_path / "robustness"
    expected = [
        "rolling_fold_results_full.csv",
        "rolling_fold_profiles_full.json",
        "rolling_summary.csv",
        "rolling_pairwise_comparison.csv",
        "rolling_experiment_report.md",
    ]
    assert all((robustness / name).exists() for name in expected)


def test_pairwise_business_cost_direction_is_unambiguous() -> None:
    base = {
        "fold": 1,
        "feature_set": "Current transaction safe features",
        "weight": 1.0,
        "calibration_method": "Platt",
        "fn_cost": 100.0,
        "fp_cost": 1.0,
        "target_capacity": 0.03,
        "pr_auc": 0.2,
        "precision": 0.2,
        "recall": 0.2,
        "f1": 0.2,
        "business_cost": 100.0,
        "cost_saving": 50.0,
        "cost_saving_rate": 0.25,
        "alert_rate": 0.01,
    }
    behavior = dict(base)
    behavior.update(
        {
            "scheme": "Recipient history (w=1)",
            "model_id": "recipient_full",
            "feature_set": "Multi-scale recipient history",
            "recall": 0.5,
            "business_cost": 70.0,
            "cost_saving": 80.0,
            "cost_saving_rate": 0.4,
        }
    )
    raw = dict(base)
    raw.update({"scheme": "Raw", "model_id": "raw_safe"})
    comparison = pairwise_comparisons(pd.DataFrame([raw, behavior])).iloc[0]
    assert comparison["delta_recall"] == 0.3
    assert comparison["business_cost_reduction"] == 30.0
    assert bool(comparison["business_cost_win"])
