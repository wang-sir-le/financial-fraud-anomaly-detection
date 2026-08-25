"""Fixed-fold, repeated-seed stability analysis for the primary business scenario."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudx.config import ExperimentConfig
from fraudx.data import rolling_temporal_splits, validation_period_masks
from fraudx.features import build_behavioral_features, feature_columns_for_group
from fraudx.metrics import evaluate_probabilities, optimize_threshold
from fraudx.models import PlattCalibrator, build_lightgbm_pipeline
from fraudx.pipeline import prepare_real_data
from fraudx.preprocess import load_prepared_frame
from fraudx.robustness import _fold_boundaries, _scheme_name
from fraudx.robustness_reporting import cost_saving_fields
from fraudx.synthetic import make_synthetic_paysim

PRIMARY_MODELS = (
    ("raw_safe", "raw_safe", 1.0),
    ("recipient_full", "recipient_full", 1.0),
    ("recipient_full_weight_2", "recipient_full", 2.0),
    ("recipient_full_weight_5", "recipient_full", 5.0),
)
SUMMARY_METRICS = {
    "pr_auc": ("mean", "std", "min", "max"),
    "precision": ("mean", "std", "min", "max"),
    "recall": ("mean", "std", "min", "max"),
    "f1": ("mean", "std"),
    "business_cost": ("mean", "std", "min", "max"),
    "cost_saving": ("mean", "std"),
    "cost_saving_rate": ("mean", "std", "min", "max"),
    "threshold": ("mean", "std", "min", "max"),
    "alert_rate": ("mean", "std"),
    "capacity_deviation": ("mean", "std"),
}


def run_multiseed_validation(
    config: ExperimentConfig,
    *,
    smoke: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run 3 folds x 4 schemes x 5 seeds without changing data partitions."""
    settings = config.raw.get("multiseed", {})
    _require_pair_audit_gate(config, smoke=smoke)
    seeds = [int(value) for value in settings.get("seeds", [42, 52, 62, 72, 82])]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("multiseed.seeds must be a non-empty unique list")
    folds = int(config.raw.get("robustness", {}).get("folds", 3))
    models = list(PRIMARY_MODELS)
    if smoke:
        folds = 2
        seeds = seeds[:2]
        models = models[:2]
        frame = build_behavioral_features(
            make_synthetic_paysim(rows=1200, seed=config.seed), config.windows
        )
    else:
        prepare_real_data(config)
        frame = load_prepared_frame(config.processed_path)
    robustness = config.raw.get("robustness", {})
    splits = rolling_temporal_splits(
        frame,
        folds=folds,
        initial_train_fraction=float(robustness.get("initial_train_fraction", 0.4)),
        validation_fraction=float(robustness.get("validation_fraction", 0.1)),
        test_fraction=float(robustness.get("test_fraction", 0.1)),
        time_column=config.time_column,
    )
    fn_cost = float(settings.get("false_negative_cost", 100.0))
    fp_cost = float(settings.get("false_positive_cost", 1.0))
    capacity = float(settings.get("maximum_alert_rate", 0.03))
    rows: list[dict[str, Any]] = []
    model_config = config.raw["model"]
    for fold, split in enumerate(splits, start=1):
        calibration_mask, threshold_mask = validation_period_masks(
            split.validation, config.time_column, config.target
        )
        calibration_labels = split.validation.loc[
            calibration_mask, config.target
        ].to_numpy()
        threshold_labels = split.validation.loc[threshold_mask, config.target].to_numpy()
        test_labels = split.test[config.target].to_numpy()
        boundaries = _fold_boundaries(split, config)
        for model_id, feature_group, weight in models:
            columns = feature_columns_for_group(split.train, feature_group)
            for seed in seeds:
                model, selected_columns = build_lightgbm_pipeline(
                    split.train,
                    seed=seed,
                    scale_pos_weight=weight,
                    n_estimators=int(model_config["n_estimators"]),
                    learning_rate=float(model_config["learning_rate"]),
                    num_leaves=int(model_config["num_leaves"]),
                    feature_columns=columns,
                    subsample=float(model_config.get("subsample", 1.0)),
                    colsample_bytree=float(model_config.get("colsample_bytree", 1.0)),
                )
                seed_parameters = {
                    "random_state": seed,
                    "bagging_seed": seed,
                    "feature_fraction_seed": seed,
                    "data_random_seed": seed,
                }
                model.named_steps["classifier"].set_params(**seed_parameters)
                model.fit(split.train[selected_columns], split.train[config.target])
                validation_raw = model.predict_proba(
                    split.validation[selected_columns]
                )[:, 1]
                calibrator = PlattCalibrator().fit(
                    validation_raw[calibration_mask], calibration_labels
                )
                threshold_scores = calibrator.transform(validation_raw[threshold_mask])
                test_raw = model.predict_proba(split.test[selected_columns])[:, 1]
                test_scores = calibrator.transform(test_raw)
                threshold, validation_report = optimize_threshold(
                    threshold_labels,
                    threshold_scores,
                    false_negative_cost=fn_cost,
                    false_positive_cost=fp_cost,
                    maximum_alert_rate=capacity,
                )
                report = evaluate_probabilities(
                    test_labels,
                    test_scores,
                    threshold=threshold,
                    false_negative_cost=fn_cost,
                    false_positive_cost=fp_cost,
                )
                cost_fields = cost_saving_fields(
                    report,
                    fn_cost=fn_cost,
                    fp_cost=fp_cost,
                    target_capacity=capacity,
                    test_size=len(test_labels),
                    test_fraud_count=int(test_labels.sum()),
                )
                rows.append(
                    {
                        "fold": fold,
                        "scheme": _scheme_name(model_id),
                        "model_id": model_id,
                        "feature_set": feature_group,
                        "weight": weight,
                        "seed": seed,
                        "fn_cost": fn_cost,
                        "fp_cost": fp_cost,
                        "target_capacity": capacity,
                        "calibration_method": "Platt",
                        "threshold": threshold,
                        "validation_alert_rate": validation_report.alert_rate,
                        "pr_auc": report.pr_auc,
                        "precision": report.precision,
                        "recall": report.recall,
                        "f1": report.f1,
                        "tp": report.tp,
                        "fp": report.fp,
                        "tn": report.tn,
                        "fn": report.fn,
                        "alert_rate": report.alert_rate,
                        **cost_fields,
                        **boundaries,
                        **seed_parameters,
                        "subsample": float(model_config.get("subsample", 1.0)),
                        "subsample_freq": int(
                            float(model_config.get("subsample", 1.0)) < 1.0
                        ),
                        "colsample_bytree": float(
                            model_config.get("colsample_bytree", 1.0)
                        ),
                        "raw_probability_hash": _array_hash(test_raw),
                        "calibrated_probability_hash": _array_hash(test_scores),
                    }
                )
    results = pd.DataFrame(rows)
    expected_rows = len(splits) * len(models) * len(seeds)
    checks = validate_multiseed_results(
        results,
        expected_rows=expected_rows,
        expected_seeds=seeds,
        expected_schemes=len(models),
    )
    summary = summarize_multiseed_results(results)
    pairwise = make_primary_pairwise(results)
    wins = summarize_win_rates(pairwise)
    output_dir = config.output_dir / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "rolling_multiseed_results.csv", index=False)
    summary.to_csv(output_dir / "rolling_multiseed_summary.csv", index=False)
    pairwise.to_csv(output_dir / "rolling_multiseed_pairwise.csv", index=False)
    wins.to_csv(output_dir / "rolling_multiseed_win_rates.csv", index=False)
    _write_multiseed_report(
        output_dir / "rolling_multiseed_report.md",
        results=results,
        summary=summary,
        pairwise=pairwise,
        wins=wins,
        checks=checks,
        pair_gate=json.loads(
            (config.output_dir / "pair_audit" / "pair_audit_status.json").read_text(
                encoding="utf-8"
            )
        )
        if not smoke
        else {"classification": "smoke-test bypass"},
    )
    return {"results": results, "summary": summary, "pairwise": pairwise, "wins": wins}


def summarize_multiseed_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize each fold/scheme over repeated model seeds."""
    keys = ["fold", "scheme", "model_id", "feature_set", "weight"]
    summary = results.groupby(keys, as_index=False).agg(SUMMARY_METRICS)
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    maximum_absolute = (
        results.assign(capacity_deviation_abs=results["capacity_deviation"].abs())
        .groupby(keys, as_index=False)["capacity_deviation_abs"]
        .max()
        .rename(columns={"capacity_deviation_abs": "capacity_deviation_max_abs"})
    )
    summary = summary.merge(maximum_absolute, on=keys, validate="one_to_one")
    summary["threshold_coefficient_of_variation"] = np.where(
        summary["threshold_mean"].abs() > np.finfo(float).eps,
        summary["threshold_std"] / summary["threshold_mean"].abs(),
        np.nan,
    )
    return summary


def make_primary_pairwise(results: pd.DataFrame) -> pd.DataFrame:
    """Pair Raw and Recipient w=2 by identical fold and seed."""
    keys = ["fold", "seed"]
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
    raw = results.loc[results["model_id"] == "raw_safe", keys + metrics]
    recipient = results.loc[
        results["model_id"] == "recipient_full_weight_2", keys + metrics
    ]
    raw = raw.rename(columns={metric: f"raw_{metric}" for metric in metrics})
    recipient = recipient.rename(
        columns={metric: f"recipient_{metric}" for metric in metrics}
    )
    paired = raw.merge(recipient, on=keys, validate="one_to_one")
    delta_metrics = [
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "cost_saving",
        "cost_saving_rate",
        "alert_rate",
    ]
    for metric in delta_metrics:
        paired[f"delta_{metric}"] = (
            paired[f"recipient_{metric}"] - paired[f"raw_{metric}"]
        )
    paired["business_cost_reduction"] = (
        paired["raw_business_cost"] - paired["recipient_business_cost"]
    )
    return paired


def summarize_win_rates(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Count paired wins by fold and over all 15 repeated-seed comparisons."""
    win_rules = {
        "recall": pairwise["delta_recall"] > 0,
        "business_cost": pairwise["business_cost_reduction"] > 0,
        "cost_saving": pairwise["delta_cost_saving"] > 0,
        "cost_saving_rate": pairwise["delta_cost_saving_rate"] > 0,
        "pr_auc": pairwise["delta_pr_auc"] > 0,
    }
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.Index]] = [
        (f"Fold {fold}", subset.index)
        for fold, subset in pairwise.groupby("fold", sort=True)
    ]
    groups.append(("Overall", pairwise.index))
    for scope, indices in groups:
        for metric, wins in win_rules.items():
            count = int(wins.loc[indices].sum())
            total = int(len(indices))
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "win_count": count,
                    "comparison_count": total,
                    "win_rate": count / total if total else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def validate_multiseed_results(
    results: pd.DataFrame,
    *,
    expected_rows: int,
    expected_seeds: list[int],
    expected_schemes: int,
) -> dict[str, int | bool]:
    """Fail fast on missing runs, moving partitions, or invalid metric identities."""
    key = ["fold", "model_id", "weight", "seed"]
    if len(results) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(results)}")
    if results.duplicated(key).any():
        raise AssertionError("Duplicate Fold/Scheme/Weight/Seed rows detected")
    seed_counts = results.groupby(["fold", "model_id", "weight"])["seed"].nunique()
    if not (seed_counts == len(expected_seeds)).all():
        raise AssertionError("Every Fold/Scheme must contain every configured seed")
    if results["model_id"].nunique() != expected_schemes:
        raise AssertionError("Unexpected number of schemes")
    if set(results["seed"].unique()) != set(expected_seeds):
        raise AssertionError("Result seeds differ from the locked seed list")
    fixed = [
        "test_size",
        "test_fraud_count",
        "test_fraud_rate",
        "all_normal_cost",
        "train_min_step",
        "train_max_step",
        "validation_min_step",
        "validation_max_step",
        "test_min_step",
        "test_max_step",
    ]
    if (results.groupby("fold")[fixed].nunique() != 1).any().any():
        raise AssertionError("Fold data or temporal boundaries changed between seeds")
    for metric in ["pr_auc", "precision", "recall", "f1", "alert_rate"]:
        if not results[metric].between(0, 1).all():
            raise AssertionError(f"{metric} contains values outside [0, 1]")
    costs = ["false_negative_cost", "false_positive_cost", "business_cost", "all_normal_cost"]
    if (results[costs] < 0).any().any():
        raise AssertionError("Cost fields cannot be negative")
    if not np.allclose(
        results["business_cost"],
        results["false_negative_cost"] + results["false_positive_cost"],
    ):
        raise AssertionError("Business-cost decomposition failed")
    if not np.allclose(
        results["all_normal_cost"], results["test_fraud_count"] * results["fn_cost"]
    ):
        raise AssertionError("All-normal baseline failed")
    return {
        "expected_rows": expected_rows,
        "actual_rows": int(len(results)),
        "duplicate_rows": 0,
        "fixed_partition_check": True,
        "metric_check": True,
        "cost_check": True,
    }


def _require_pair_audit_gate(config: ExperimentConfig, *, smoke: bool) -> None:
    if smoke:
        return
    status_path = config.output_dir / "pair_audit" / "pair_audit_status.json"
    if not status_path.exists():
        raise RuntimeError("Pair audit status is missing; multi-seed experiment is blocked")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not bool(status.get("safe_to_proceed")):
        raise RuntimeError(
            f"Pair audit blocked multi-seed experiment: {status.get('classification')}"
        )


def _array_hash(values: np.ndarray) -> str:
    normalized = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _write_multiseed_report(
    path: Path,
    *,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    wins: pd.DataFrame,
    checks: dict[str, int | bool],
    pair_gate: dict[str, Any],
) -> None:
    seeds = sorted(results["seed"].unique().tolist())
    random_exists = bool(
        (results.groupby(["fold", "model_id"])["raw_probability_hash"].nunique() > 1).any()
    )
    overall = wins.loc[wins["scope"] == "Overall"].set_index("metric")
    recall_wins = int(overall.loc["recall", "win_count"])
    cost_wins = int(overall.loc["business_cost", "win_count"])
    total_pairs = int(overall.loc["recall", "comparison_count"])
    if recall_wins >= 14 and cost_wins >= 14:
        verdict = "明显保持"
    elif recall_wins >= 11 and cost_wins >= 11:
        verdict = "基本保持但存在一定随机波动"
    elif recall_wins >= 8 and cost_wins >= 8:
        verdict = "部分保持"
    else:
        verdict = "不稳定"
    boundaries = results[
        [
            "fold",
            "train_min_step",
            "train_max_step",
            "validation_min_step",
            "validation_max_step",
            "test_min_step",
            "test_max_step",
        ]
    ].drop_duplicates()
    lines = [
        "# Five-seed rolling stability report",
        "",
        "## Purpose and locked protocol",
        "",
        "This experiment tests model-training randomness only. Time partitions, features, "
        "calibration protocol, threshold rule, FN cost 100, FP cost 1, and capacity 3% are fixed.",
        f"Seeds: {', '.join(map(str, seeds))}.",
        f"Pair gate: {pair_gate.get('classification')}; Pair is excluded because it is "
        "degenerate in PaySim.",
        "",
        "## Fold boundaries",
        "",
        "| Fold | Train | Validation | Test |",
        "|---:|---:|---:|---:|",
    ]
    for row in boundaries.sort_values("fold").itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.train_min_step}-{row.train_max_step} | "
            f"{row.validation_min_step}-{row.validation_max_step} | "
            f"{row.test_min_step}-{row.test_max_step} |"
        )
    first = results.iloc[0]
    lines.extend(
        [
            "",
            "## Model randomness and run integrity",
            "",
            "- Seed parameters: random_state, bagging_seed, feature_fraction_seed, "
            "data_random_seed = each run seed.",
            f"- Sampling: subsample={first.subsample:g}, "
            f"subsample_freq={int(first.subsample_freq)}, "
            f"colsample_bytree={first.colsample_bytree:g}.",
            f"- Actual probability variation across seeds detected: {random_exists}.",
            f"- Theoretical/actual runs: {checks['expected_rows']}/{checks['actual_rows']}.",
            f"- Fixed partitions and leakage-safe chronology: {checks['fixed_partition_check']}.",
            "",
            "## Fold-level mean ± standard deviation",
            "",
            "| Fold | Scheme | PR-AUC | Recall | Business cost | Cost-saving rate | Alert rate |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.sort_values(["fold", "weight", "model_id"]).itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.scheme} | {row.pr_auc_mean:.4f} ± {row.pr_auc_std:.4f} | "
            f"{row.recall_mean:.2%} ± {row.recall_std:.2%} | "
            f"{row.business_cost_mean:.2f} ± {row.business_cost_std:.2f} | "
            f"{row.cost_saving_rate_mean:.2%} ± {row.cost_saving_rate_std:.2%} | "
            f"{row.alert_rate_mean:.2%} ± {row.alert_rate_std:.2%} |"
        )
    deltas = pairwise.groupby("fold", as_index=False).agg(
        delta_pr_auc_mean=("delta_pr_auc", "mean"),
        delta_recall_mean=("delta_recall", "mean"),
        cost_reduction_mean=("business_cost_reduction", "mean"),
        delta_saving_rate_mean=("delta_cost_saving_rate", "mean"),
    )
    lines.extend(
        [
            "",
            "## Paired Raw vs Recipient history w=2",
            "",
            "| Fold | ΔPR-AUC | ΔRecall | Mean cost reduction | ΔCost-saving rate |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deltas.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.delta_pr_auc_mean:+.4f} | {row.delta_recall_mean:+.2%} | "
            f"{row.cost_reduction_mean:+.2f} | {row.delta_saving_rate_mean:+.2%} |"
        )
    lines.extend(
        ["", "## Win rates", "", "| Scope | Metric | Wins | Rate |", "|---|---|---:|---:|"]
    )
    for row in wins.itertuples(index=False):
        lines.append(
            f"| {row.scope} | {row.metric} | {row.win_count}/"
            f"{row.comparison_count} | {row.win_rate:.0%} |"
        )
    threshold = summary[
        [
            "fold",
            "scheme",
            "threshold_mean",
            "threshold_std",
            "threshold_min",
            "threshold_max",
            "threshold_coefficient_of_variation",
            "capacity_deviation_mean",
            "capacity_deviation_std",
            "capacity_deviation_max_abs",
        ]
    ]
    lines.extend(
        [
            "",
            "## Threshold and capacity-deviation stability",
            "",
            "| Fold | Scheme | Threshold mean ± std [min,max] | CV | "
            "Capacity deviation mean ± std | Max abs |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in threshold.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.scheme} | {row.threshold_mean:.6g} ± {row.threshold_std:.2g} "
            f"[{row.threshold_min:.6g}, {row.threshold_max:.6g}] | "
            f"{row.threshold_coefficient_of_variation:.2%} | "
            f"{row.capacity_deviation_mean:+.2%} ± {row.capacity_deviation_std:.2%} | "
            f"{row.capacity_deviation_max_abs:.2%} |"
        )
    fold3 = summary.loc[summary["fold"] == 3]
    fold3_rate = (
        results.loc[results["fold"] == 3, "test_fraud_rate"].iloc[0]
        if not fold3.empty
        else float("nan")
    )
    lines.extend(
        [
            "",
            "## Fold 3 temporal-drift case",
            "",
            f"Test fraud rate: {fold3_rate:.4%}. Fold 3 is retained as a "
            "pre-specified future window.",
        ]
    )
    for row in fold3.itertuples(index=False):
        lines.append(
            f"- {row.scheme}: PR-AUC {row.pr_auc_mean:.4f} ± {row.pr_auc_std:.4f}; "
            f"Recall {row.recall_mean:.2%} ± {row.recall_std:.2%}; cost "
            f"{row.business_cost_mean:.2f} ± {row.business_cost_std:.2f}; saving rate "
            f"{row.cost_saving_rate_mean:.2%} ± {row.cost_saving_rate_std:.2%}; alert rate "
            f"{row.alert_rate_mean:.2%} ± {row.alert_rate_std:.2%}; capacity deviation "
            f"{row.capacity_deviation_mean:+.2%} ± {row.capacity_deviation_std:.2%}."
        )
    lines.extend(
        [
            "",
            "## Conclusion and interpretation boundary",
            "",
            f"Core conclusion: **{verdict}** ({recall_wins}/{total_pairs} Recall wins; "
            f"{cost_wins}/{total_pairs} lower-business-cost wins).",
            "The 15 pairs are not 15 independent temporal test sets: they are five stochastic "
            "repetitions on each of three independent rolling positions. No test result "
            "selected a seed, "
            "threshold, or model. No future transaction was used in behavioral features.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
