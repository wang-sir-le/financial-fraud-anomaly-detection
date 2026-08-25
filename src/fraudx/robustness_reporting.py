"""Pure reporting helpers for cost-sensitive rolling validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudx.metrics import MetricReport

TYPE_ORDER = ("CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT")
PAIR_KEYS = ("fold", "fn_cost", "fp_cost", "target_capacity")
RESULT_KEY = ("fold", "model_id", "weight", "fn_cost", "fp_cost", "target_capacity")


def cost_saving_fields(
    report: MetricReport,
    *,
    fn_cost: float,
    fp_cost: float,
    target_capacity: float,
    test_size: int,
    test_fraud_count: int,
) -> dict[str, float | int]:
    """Calculate cost decomposition, all-normal baseline, and capacity drift."""
    false_negative_cost = float(report.fn * fn_cost)
    false_positive_cost = float(report.fp * fp_cost)
    business_cost = false_negative_cost + false_positive_cost
    all_normal_cost = float(test_fraud_count * fn_cost)
    cost_saving = all_normal_cost - business_cost
    cost_saving_rate = cost_saving / all_normal_cost if all_normal_cost else float("nan")
    capacity_deviation = float(report.alert_rate - target_capacity)
    if not np.isclose(report.business_cost, business_cost):
        raise AssertionError("Business cost does not equal FN and FP cost components")
    if report.tp + report.fp + report.tn + report.fn != test_size:
        raise AssertionError("Confusion matrix does not sum to test size")
    if report.tp + report.fn != test_fraud_count:
        raise AssertionError("Confusion matrix fraud count is inconsistent")
    expected_alert_rate = (report.tp + report.fp) / test_size if test_size else 0.0
    if not np.isclose(report.alert_rate, expected_alert_rate):
        raise AssertionError("Alert rate is inconsistent with the confusion matrix")
    return {
        "alert_count": int(report.tp + report.fp),
        "capacity_deviation": capacity_deviation,
        "capacity_deviation_abs": abs(capacity_deviation),
        "false_negative_cost": false_negative_cost,
        "false_positive_cost": false_positive_cost,
        "business_cost": business_cost,
        "all_normal_cost": all_normal_cost,
        "cost_saving": cost_saving,
        "cost_saving_rate": cost_saving_rate,
        "test_size": int(test_size),
        "test_fraud_count": int(test_fraud_count),
        "test_fraud_rate": float(test_fraud_count / test_size if test_size else 0.0),
    }


def partition_profile(
    frame: pd.DataFrame,
    *,
    target: str,
    time_column: str,
) -> dict[str, Any]:
    """Describe class, amount, type, and temporal composition of one partition."""
    fraud_mask = frame[target].astype(bool)
    fraud = frame.loc[fraud_mask]
    normal = frame.loc[~fraud_mask]
    return {
        "sample_count": int(len(frame)),
        "fraud_count": int(fraud_mask.sum()),
        "fraud_rate": float(fraud_mask.mean()) if len(frame) else 0.0,
        "min_step": int(frame[time_column].min()),
        "max_step": int(frame[time_column].max()),
        "amount_overall": _amount_profile(frame["amount"], include_std=True),
        "amount_normal": _amount_profile(normal["amount"]),
        "amount_fraud": _amount_profile(fraud["amount"]),
        "type_share_overall": _type_shares(frame["type"]),
        "type_share_among_fraud": _type_shares(fraud["type"], empty_as_none=True),
    }


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level results and append paired win rates against Raw."""
    group_columns = [
        "scheme",
        "model_id",
        "feature_set",
        "weight",
        "calibration_method",
        "fn_cost",
        "fp_cost",
        "target_capacity",
    ]
    metrics = [
        "pr_auc",
        "recall",
        "precision",
        "f1",
        "business_cost",
        "cost_saving",
        "cost_saving_rate",
        "alert_rate",
        "capacity_deviation",
    ]
    summary = results.groupby(group_columns, dropna=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.reset_index()
    comparisons = pairwise_comparisons(results)
    if comparisons.empty:
        return summary
    win_group = [
        "scheme",
        "model_id",
        "feature_set",
        "weight",
        "fn_cost",
        "fp_cost",
        "target_capacity",
    ]
    win_metrics = ["pr_auc", "recall", "business_cost", "cost_saving"]
    grouped = comparisons.groupby(win_group, dropna=False)
    wins = grouped[[f"{metric}_win" for metric in win_metrics]].sum().reset_index()
    fold_counts = grouped.size().rename("comparison_fold_count").reset_index()
    wins = wins.merge(fold_counts, on=win_group, validate="one_to_one")
    for metric in win_metrics:
        count_column = f"{metric}_win"
        wins = wins.rename(columns={count_column: f"{metric}_win_count"})
        wins[f"{metric}_win_rate"] = (
            wins[f"{metric}_win_count"] / wins["comparison_fold_count"]
        )
    return summary.merge(wins, on=win_group, how="left", validate="one_to_one")


def pairwise_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    """Compute fold-matched Behavior minus Raw differences for every scenario."""
    raw = results.loc[results["model_id"] == "raw_safe"].copy()
    behavior = results.loc[results["model_id"] != "raw_safe"].copy()
    if raw.empty or behavior.empty:
        return pd.DataFrame()
    raw_columns = list(PAIR_KEYS) + [
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "business_cost",
        "cost_saving",
        "cost_saving_rate",
        "alert_rate",
    ]
    raw = raw[raw_columns].rename(
        columns={column: f"raw_{column}" for column in raw_columns if column not in PAIR_KEYS}
    )
    merged = behavior.merge(raw, on=list(PAIR_KEYS), how="left", validate="many_to_one")
    metrics = [
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "business_cost",
        "cost_saving",
        "cost_saving_rate",
        "alert_rate",
    ]
    identity = [
        "fold",
        "scheme",
        "model_id",
        "feature_set",
        "weight",
        "calibration_method",
        "fn_cost",
        "fp_cost",
        "target_capacity",
    ]
    comparison = merged[identity].copy()
    for metric in metrics:
        comparison[f"delta_{metric}"] = merged[metric] - merged[f"raw_{metric}"]
    comparison["business_cost_reduction"] = -comparison["delta_business_cost"]
    comparison["pr_auc_win"] = comparison["delta_pr_auc"] > 0
    comparison["recall_win"] = comparison["delta_recall"] > 0
    comparison["business_cost_win"] = comparison["business_cost_reduction"] > 0
    comparison["cost_saving_win"] = comparison["delta_cost_saving"] > 0
    return comparison


def validate_full_results(
    results: pd.DataFrame,
    *,
    expected_rows: int,
) -> dict[str, int | bool]:
    """Fail fast on incomplete, duplicated, or internally inconsistent results."""
    if len(results) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} result rows, found {len(results)}")
    if results.duplicated(list(RESULT_KEY)).any():
        raise AssertionError("Duplicate Fold/Scheme/Cost/Capacity result rows detected")
    required_non_null = [column for column in results.columns if column != "cost_saving_rate"]
    if results[required_non_null].isna().any().any():
        bad = results[required_non_null].columns[
            results[required_non_null].isna().any()
        ].tolist()
        raise AssertionError(f"Unexpected NaN values in result columns: {bad}")
    for column in ["pr_auc", "roc_auc", "precision", "recall", "f1", "alert_rate"]:
        if not results[column].between(0, 1).all():
            raise AssertionError(f"{column} contains values outside [0, 1]")
    if (results[["business_cost", "all_normal_cost"]] < 0).any().any():
        raise AssertionError("Business cost fields cannot be negative")
    if (~np.isfinite(results["threshold"])).any() or (
        results["threshold"] < 0
    ).any():
        raise AssertionError("Threshold contains invalid values")
    if not np.allclose(
        results["business_cost"],
        results["false_negative_cost"] + results["false_positive_cost"],
    ):
        raise AssertionError("Business cost decomposition failed")
    if not (
        results["tp"] + results["fp"] + results["tn"] + results["fn"]
        == results["test_size"]
    ).all():
        raise AssertionError("Confusion matrix totals are inconsistent")
    if not (results["tp"] + results["fn"] == results["test_fraud_count"]).all():
        raise AssertionError("Fraud totals are inconsistent")
    if not np.allclose(
        results["alert_rate"],
        results["alert_count"] / results["test_size"],
    ):
        raise AssertionError("Alert-rate consistency check failed")
    if not np.allclose(
        results["all_normal_cost"],
        results["test_fraud_count"] * results["fn_cost"],
    ):
        raise AssertionError("All-normal baseline consistency check failed")
    ranking_group = ["fold", "model_id", "weight", "calibration_method"]
    if (results.groupby(ranking_group)["pr_auc"].nunique() != 1).any():
        raise AssertionError("PR-AUC changed across cost/capacity decision scenarios")
    return {
        "expected_rows": expected_rows,
        "actual_rows": int(len(results)),
        "duplicate_rows": 0,
        "unexpected_nan": False,
        "checks_passed": True,
    }


def _amount_profile(values: pd.Series, *, include_std: bool = False) -> dict[str, float | None]:
    clean = values.dropna().astype(float)
    keys = ["mean", "median", "p90", "p99"]
    if include_std:
        keys.append("std")
    if clean.empty:
        return dict.fromkeys(keys)
    profile: dict[str, float | None] = {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p90": float(clean.quantile(0.90)),
        "p99": float(clean.quantile(0.99)),
    }
    if include_std:
        profile["std"] = float(clean.std(ddof=0))
    return profile


def _type_shares(values: pd.Series, *, empty_as_none: bool = False) -> dict[str, float | None]:
    if values.empty:
        fill: float | None = None if empty_as_none else 0.0
        return {transaction_type: fill for transaction_type in TYPE_ORDER}
    shares = values.astype(str).value_counts(normalize=True)
    return {
        transaction_type: float(shares.get(transaction_type, 0.0))
        for transaction_type in TYPE_ORDER
    }
