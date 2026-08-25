"""Expanding-window robustness experiments for temporal fraud detection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraudx.config import ExperimentConfig
from fraudx.data import rolling_temporal_splits, validation_period_masks
from fraudx.features import build_behavioral_features, feature_columns_for_group
from fraudx.metrics import evaluate_probabilities, optimize_threshold
from fraudx.models import PlattCalibrator, build_lightgbm_pipeline
from fraudx.pipeline import prepare_real_data
from fraudx.preprocess import load_prepared_frame
from fraudx.robustness_reporting import (
    aggregate_results,
    cost_saving_fields,
    pairwise_comparisons,
    partition_profile,
    validate_full_results,
)
from fraudx.synthetic import make_synthetic_paysim


def run_rolling_validation(config: ExperimentConfig, smoke: bool = False) -> pd.DataFrame:
    """Run the locked three-fold, cost- and capacity-sensitive validation protocol."""
    settings = config.raw.get("robustness", {})
    folds = int(settings.get("folds", 3))
    if smoke:
        folds = 2
        behavior = build_behavioral_features(
            make_synthetic_paysim(seed=config.seed), config.windows
        )
    else:
        prepare_real_data(config)
        behavior = load_prepared_frame(config.processed_path)
    splits = rolling_temporal_splits(
        behavior,
        folds=folds,
        initial_train_fraction=float(settings.get("initial_train_fraction", 0.4)),
        validation_fraction=float(settings.get("validation_fraction", 0.1)),
        test_fraction=float(settings.get("test_fraction", 0.1)),
        time_column=config.time_column,
    )
    candidates = _candidate_models(settings)
    if smoke:
        candidates = candidates[:2]
    model_config = config.raw["model"]
    risk_config = config.raw["risk"]
    false_negative_costs = _scenario_values(
        settings,
        plural_key="false_negative_costs",
        singular_key="false_negative_cost",
        default=100.0,
    )
    false_positive_cost = float(risk_config.get("false_positive_cost", 1.0))
    maximum_alert_rates = _scenario_values(
        settings,
        plural_key="maximum_alert_rates",
        singular_key="maximum_alert_rate",
        default=0.03,
    )
    primary_false_negative_cost = float(
        settings.get("primary_false_negative_cost", settings.get("false_negative_cost", 100.0))
    )
    primary_maximum_alert_rate = float(
        settings.get("primary_maximum_alert_rate", settings.get("maximum_alert_rate", 0.03))
    )
    rows: list[dict[str, Any]] = []
    fold_profiles: list[dict[str, Any]] = []
    for fold_index, split in enumerate(splits, start=1):
        calibration_mask, threshold_mask = validation_period_masks(
            split.validation, config.time_column, config.target
        )
        fold_profiles.append(_fold_profile(fold_index, split, config))
        calibration_labels = split.validation.loc[calibration_mask, config.target].to_numpy()
        threshold_labels = split.validation.loc[threshold_mask, config.target].to_numpy()
        test_labels = split.test[config.target].to_numpy()
        for model_id, feature_group, weight in candidates:
            columns = feature_columns_for_group(split.train, feature_group)
            model, columns = build_lightgbm_pipeline(
                split.train,
                seed=config.seed,
                scale_pos_weight=weight,
                n_estimators=int(model_config["n_estimators"]),
                learning_rate=float(model_config["learning_rate"]),
                num_leaves=int(model_config["num_leaves"]),
                feature_columns=columns,
                subsample=float(model_config.get("subsample", 1.0)),
                colsample_bytree=float(model_config.get("colsample_bytree", 1.0)),
            )
            model.fit(split.train[columns], split.train[config.target])
            validation_raw = model.predict_proba(split.validation[columns])[:, 1]
            calibrator = PlattCalibrator().fit(
                validation_raw[calibration_mask], calibration_labels
            )
            threshold_scores = calibrator.transform(validation_raw[threshold_mask])
            test_scores = calibrator.transform(model.predict_proba(split.test[columns])[:, 1])
            for false_negative_cost in false_negative_costs:
                for maximum_alert_rate in maximum_alert_rates:
                    threshold, validation_report = optimize_threshold(
                        threshold_labels,
                        threshold_scores,
                        false_negative_cost=false_negative_cost,
                        false_positive_cost=false_positive_cost,
                        maximum_alert_rate=maximum_alert_rate,
                    )
                    test_report = evaluate_probabilities(
                        test_labels,
                        test_scores,
                        threshold=threshold,
                        false_negative_cost=false_negative_cost,
                        false_positive_cost=false_positive_cost,
                    )
                    test_size = int(test_labels.size)
                    test_fraud_count = int(test_labels.sum())
                    cost_fields = cost_saving_fields(
                        test_report,
                        fn_cost=false_negative_cost,
                        fp_cost=false_positive_cost,
                        target_capacity=maximum_alert_rate,
                        test_size=test_size,
                        test_fraud_count=test_fraud_count,
                    )
                    rows.append(
                        {
                            "fold": fold_index,
                            "scheme": _scheme_name(model_id),
                            "model_id": model_id,
                            "feature_set": _feature_set_name(feature_group),
                            "weight": weight,
                            "calibration_method": "Platt",
                            "fn_cost": false_negative_cost,
                            "fp_cost": false_positive_cost,
                            "target_capacity": maximum_alert_rate,
                            "threshold": threshold,
                            "validation_alert_rate": validation_report.alert_rate,
                            "validation_business_cost": validation_report.business_cost,
                            "pr_auc": test_report.pr_auc,
                            "roc_auc": test_report.roc_auc,
                            "precision": test_report.precision,
                            "recall": test_report.recall,
                            "f1": test_report.f1,
                            "brier": test_report.brier,
                            "ece": test_report.ece,
                            "mcc": test_report.mcc,
                            "tn": test_report.tn,
                            "fp": test_report.fp,
                            "fn": test_report.fn,
                            "tp": test_report.tp,
                            "alert_rate": test_report.alert_rate,
                            **cost_fields,
                            "seed": config.seed,
                            **_fold_boundaries(split, config),
                        }
                    )
    results = pd.DataFrame(rows)
    output_dir = config.output_dir / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_rows = (
        len(splits)
        * len(candidates)
        * len(false_negative_costs)
        * len(maximum_alert_rates)
    )
    checks = validate_full_results(results, expected_rows=expected_rows)
    results.to_csv(output_dir / "rolling_fold_results_full.csv", index=False)
    summary = aggregate_results(results)
    _preserve_legacy_summary(output_dir)
    summary.to_csv(output_dir / "rolling_summary.csv", index=False)
    comparisons = pairwise_comparisons(results)
    comparisons.to_csv(output_dir / "rolling_pairwise_comparison.csv", index=False)
    (output_dir / "rolling_fold_profiles_full.json").write_text(
        json.dumps(fold_profiles, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_experiment_report(
        output_dir / "rolling_experiment_report.md",
        results=results,
        checks=checks,
        folds=folds,
        candidates=candidates,
        false_negative_costs=false_negative_costs,
        maximum_alert_rates=maximum_alert_rates,
        primary_false_negative_cost=primary_false_negative_cost,
        primary_maximum_alert_rate=primary_maximum_alert_rate,
    )
    return results


def _candidate_models(settings: dict[str, Any]) -> list[tuple[str, str, float]]:
    raw = settings.get(
        "models",
        [
            {"id": "raw_safe", "feature_group": "raw_safe", "weight": 1.0},
            {"id": "recipient_full", "feature_group": "recipient_full", "weight": 1.0},
            {"id": "recipient_full_weight_2", "feature_group": "recipient_full", "weight": 2.0},
        ],
    )
    return [
        (str(item["id"]), str(item["feature_group"]), float(item.get("weight", 1.0)))
        for item in raw
    ]


def _plot_rolling_results(results: pd.DataFrame, path: str | Path) -> None:
    labels = {
        "raw_safe": "Raw safe",
        "recipient_full": "Recipient history",
        "recipient_full_weight_2": "Recipient history (weight 2)",
    }
    colors = ["#0072B2", "#E69F00", "#009E73"]
    markers = ["o", "s", "^"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    for index, (model_id, subset) in enumerate(results.groupby("model_id", sort=False)):
        ordered = subset.sort_values("fold")
        axes[0].plot(
            ordered["fold"],
            ordered["test_pr_auc"],
            color=colors[index],
            marker=markers[index],
            label=labels.get(str(model_id), str(model_id)),
        )
        axes[1].plot(
            ordered["fold"],
            ordered["test_recall"],
            color=colors[index],
            marker=markers[index],
            label=labels.get(str(model_id), str(model_id)),
        )
    axes[0].text(-0.12, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[0].set_ylabel("Test PR-AUC")
    axes[1].text(-0.12, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    axes[1].set_ylabel("Test recall")
    for axis in axes:
        axis.set_xlabel("Walk-forward fold")
        axis.set_xticks(sorted(results["fold"].unique()))
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    output_path = Path(path)
    figure.savefig(output_path, dpi=300)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def _fold_profile(fold: int, split: Any, config: ExperimentConfig) -> dict[str, Any]:
    return {
        "fold": fold,
        "train": partition_profile(
            split.train,
            target=config.target,
            time_column=config.time_column,
        ),
        "validation": partition_profile(
            split.validation,
            target=config.target,
            time_column=config.time_column,
        ),
        "test": partition_profile(
            split.test,
            target=config.target,
            time_column=config.time_column,
        ),
    }


def _scenario_values(
    settings: dict[str, Any],
    *,
    plural_key: str,
    singular_key: str,
    default: float,
) -> list[float]:
    raw = settings.get(plural_key)
    values = [float(value) for value in raw] if raw is not None else [
        float(settings.get(singular_key, default))
    ]
    unique = sorted(set(values))
    if not unique or any(value <= 0 for value in unique):
        raise ValueError(f"{plural_key} must contain positive values")
    if "capacity" in plural_key and any(value > 1 for value in unique):
        raise ValueError(f"{plural_key} must not exceed 1")
    return unique


def _scheme_name(model_id: str) -> str:
    labels = {
        "raw_safe": "Raw",
        "recipient_full": "Recipient history (w=1)",
        "recipient_full_weight_2": "Recipient history (w=2)",
        "recipient_full_weight_5": "Recipient history (w=5)",
        "recipient_full_with_pair": "Recipient history + Pair history (w=1)",
    }
    return labels.get(model_id, model_id)


def _feature_set_name(feature_group: str) -> str:
    labels = {
        "raw_safe": "Current transaction safe features",
        "recipient_full": "Multi-scale recipient history",
        "recipient_full_with_pair": "Recipient history + origin-recipient pair history",
    }
    return labels.get(feature_group, feature_group)


def _fold_boundaries(split: Any, config: ExperimentConfig) -> dict[str, int]:
    boundaries: dict[str, int] = {}
    for partition_name in ["train", "validation", "test"]:
        partition = getattr(split, partition_name)
        boundaries[f"{partition_name}_min_step"] = int(partition[config.time_column].min())
        boundaries[f"{partition_name}_max_step"] = int(partition[config.time_column].max())
    return boundaries


def _preserve_legacy_summary(output_dir: Path) -> None:
    current = output_dir / "rolling_summary.csv"
    legacy = output_dir / "rolling_summary_legacy.csv"
    if current.exists() and not legacy.exists():
        shutil.copy2(current, legacy)


def _write_experiment_report(
    path: Path,
    *,
    results: pd.DataFrame,
    checks: dict[str, int | bool],
    folds: int,
    candidates: list[tuple[str, str, float]],
    false_negative_costs: list[float],
    maximum_alert_rates: list[float],
    primary_false_negative_cost: float,
    primary_maximum_alert_rate: float,
) -> None:
    primary = results.loc[
        np.isclose(results["fn_cost"], primary_false_negative_cost)
        & np.isclose(results["target_capacity"], primary_maximum_alert_rate)
    ].sort_values(["fold", "weight", "model_id"])
    primary_comparison = pairwise_comparisons(primary)
    selected = (
        results.sort_values(
            [
                "fold",
                "fn_cost",
                "target_capacity",
                "validation_business_cost",
                "validation_alert_rate",
                "model_id",
            ]
        )
        .drop_duplicates(["fold", "fn_cost", "target_capacity"])
        .copy()
    )
    selected_summary = (
        selected.groupby(["fn_cost", "target_capacity"], as_index=False)
        .agg(
            selected_scheme=("scheme", lambda values: values.mode().iloc[0]),
            scheme_fold_count=("scheme", lambda values: int(values.value_counts().iloc[0])),
            recall_mean=("recall", "mean"),
            cost_saving_rate_mean=("cost_saving_rate", "mean"),
        )
        .sort_values(["fn_cost", "target_capacity"])
    )
    drift_summary = (
        results.groupby("target_capacity", as_index=False)
        .agg(
            capacity_deviation_mean=("capacity_deviation", "mean"),
            capacity_deviation_abs_mean=("capacity_deviation_abs", "mean"),
            capacity_deviation_abs_max=("capacity_deviation_abs", "max"),
        )
        .sort_values("target_capacity")
    )
    lines = [
        "# Cost-sensitive rolling validation report",
        "",
        "## Locked protocol",
        "",
        f"- Rolling folds: {folds}",
        f"- Schemes: {', '.join(model_id for model_id, _, _ in candidates)}",
        f"- FN costs: {', '.join(f'{value:g}' for value in false_negative_costs)}",
        f"- FP cost: {results['fp_cost'].iloc[0]:g}",
        f"- Capacities: {', '.join(f'{value:.0%}' for value in maximum_alert_rates)}",
        "- Calibration: fold-specific Platt scaling",
        f"- Primary scenario: FN cost {primary_false_negative_cost:g}, capacity "
        f"{primary_maximum_alert_rate:.0%}",
        "",
        "## Integrity checks",
        "",
        f"- Expected rows: {checks['expected_rows']}",
        f"- Actual rows: {checks['actual_rows']}",
        f"- Duplicate rows: {checks['duplicate_rows']}",
        f"- Unexpected NaN: {checks['unexpected_nan']}",
        f"- All checks passed: {checks['checks_passed']}",
        "",
        "## Primary scenario by fold",
        "",
        "| Fold | Scheme | PR-AUC | Precision | Recall | F1 | Alert rate | "
        "Business cost | Cost saving rate | Capacity deviation |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.scheme} | {row.pr_auc:.4f} | {row.precision:.2%} | "
            f"{row.recall:.2%} | {row.f1:.4f} | {row.alert_rate:.2%} | "
            f"{row.business_cost:.2f} | {row.cost_saving_rate:.2%} | "
            f"{row.capacity_deviation:+.2%} |"
        )
    lines.extend(
        [
            "",
            "## Primary paired comparison against Raw",
            "",
            "| Scheme | PR-AUC wins | Recall wins | Lower-cost wins | "
            "Mean recall delta | Mean business-cost reduction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _model_id, subset in primary_comparison.groupby("model_id", sort=False):
        lines.append(
            f"| {subset['scheme'].iloc[0]} | {int(subset['pr_auc_win'].sum())}/"
            f"{len(subset)} | {int(subset['recall_win'].sum())}/{len(subset)} | "
            f"{int(subset['business_cost_win'].sum())}/{len(subset)} | "
            f"{subset['delta_recall'].mean():+.2%} | "
            f"{subset['business_cost_reduction'].mean():+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Validation-selected sensitivity summary",
            "",
            "Selection uses validation business cost only; test metrics do not choose the scheme.",
            "",
            "| FN cost | Capacity | Most frequently selected scheme | Folds selected | "
            "Mean test recall | Mean test cost-saving rate |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in selected_summary.itertuples(index=False):
        lines.append(
            f"| {row.fn_cost:g} | {row.target_capacity:.0%} | {row.selected_scheme} | "
            f"{row.scheme_fold_count}/{folds} | {row.recall_mean:.2%} | "
            f"{row.cost_saving_rate_mean:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Capacity migration",
            "",
            "| Target capacity | Mean signed deviation | Mean absolute deviation | "
            "Maximum absolute deviation |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in drift_summary.itertuples(index=False):
        lines.append(
            f"| {row.target_capacity:.0%} | {row.capacity_deviation_mean:+.2%} | "
            f"{row.capacity_deviation_abs_mean:.2%} | "
            f"{row.capacity_deviation_abs_max:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Negative-result retention",
            "",
            f"- Rows with negative Cost Saving: {int((results['cost_saving'] < 0).sum())}",
            "- No row was removed based on test performance.",
            "",
            "## Interpretation boundary",
            "",
            "These three folds support descriptive claims about temporal trends only. "
            "They do not establish statistical significance. Negative cost saving values, "
            "if present, are retained.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
