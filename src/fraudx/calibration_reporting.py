"""Validation, aggregation, figures, and reporting for calibration ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ["Uncalibrated", "Platt", "Isotonic"]
METHOD_COLORS = {
    "Uncalibrated": "#0072B2",
    "Platt": "#E69F00",
    "Isotonic": "#009E73",
}
METHOD_MARKERS = {"Uncalibrated": "o", "Platt": "s", "Isotonic": "^"}


def summarize_calibration(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate five seeds within every Fold/Scheme/Calibration group."""
    keys = [
        "fold",
        "scheme",
        "model_id",
        "feature_set",
        "weight",
        "calibration_method",
    ]
    metrics = [
        "test_brier",
        "test_ece_uniform_10",
        "test_ece_quantile_10",
        "test_log_loss",
        "calibrated_pr_auc",
        "business_cost",
        "recall",
        "capacity_deviation",
        "capacity_deviation_abs",
        "threshold_raw_score_percentile",
        "validation_selected_fraction",
        "test_selected_fraction",
        "empirical_threshold",
    ]
    summary = results.groupby(keys, as_index=False)[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    summary["empirical_threshold_cv"] = np.where(
        summary["empirical_threshold_mean"].abs() > np.finfo(float).eps,
        summary["empirical_threshold_std"]
        / summary["empirical_threshold_mean"].abs(),
        np.nan,
    )
    return summary


def validate_calibration_outputs(
    results: pd.DataFrame,
    curves: pd.DataFrame,
    invariance: pd.DataFrame,
    bayes: pd.DataFrame,
    *,
    expected_model_trainings: int,
    expected_branch_rows: int,
) -> dict[str, int | bool | float]:
    """Fail fast on incomplete branches, mismatched raw scores, or invalid metrics."""
    branch_key = ["fold", "model_id", "weight", "seed", "calibration_method"]
    model_key = ["fold", "model_id", "weight", "seed"]
    if len(results) != expected_branch_rows:
        raise AssertionError(
            f"Expected {expected_branch_rows} calibration rows, found {len(results)}"
        )
    if results.duplicated(branch_key).any():
        raise AssertionError("Duplicate calibration branch rows detected")
    grouped = results.groupby(model_key)
    if grouped.ngroups != expected_model_trainings:
        raise AssertionError("Unexpected number of underlying LightGBM trainings")
    method_sets = grouped["calibration_method"].agg(lambda values: frozenset(values))
    if not (method_sets == frozenset(METHOD_ORDER)).all():
        raise AssertionError("Every model must contain all three calibration methods")
    if (grouped["raw_prediction_hash"].nunique() != 1).any():
        raise AssertionError("Calibration branches do not share the same raw predictions")
    if (grouped["raw_pr_auc"].nunique() != 1).any():
        raise AssertionError("Raw PR-AUC changed across calibration branches")
    fixed = [
        "test_size",
        "test_fraud_count",
        "test_fraud_rate",
        "train_min_step",
        "train_max_step",
        "validation_min_step",
        "validation_max_step",
        "test_min_step",
        "test_max_step",
    ]
    if (grouped[fixed].nunique() != 1).any().any():
        raise AssertionError("Data or Fold boundaries changed across calibration branches")
    for metric in [
        "test_brier",
        "val_brier",
        "test_ece_uniform_10",
        "test_ece_quantile_10",
        "test_alert_rate",
        "precision",
        "recall",
        "f1",
    ]:
        if not results[metric].between(0, 1).all():
            raise AssertionError(f"{metric} contains values outside [0, 1]")
    if (results["business_cost"] < 0).any() or (bayes["business_cost"] < 0).any():
        raise AssertionError("Business cost cannot be negative")
    if len(bayes) != expected_branch_rows or bayes.duplicated(branch_key).any():
        raise AssertionError("Bayes output is incomplete or duplicated")
    if (bayes["final_alert_rate"] > bayes["capacity"] + 1e-12).any():
        raise AssertionError("Bayes capacity truncation exceeded its fixed capacity")
    if len(invariance) != expected_model_trainings * 2:
        raise AssertionError("Decision-invariance comparison grid is incomplete")
    required_curve_groups = expected_branch_rows * 2 * 2
    actual_curve_groups = curves.groupby(
        branch_key + ["split", "binning_method"]
    ).ngroups
    if actual_curve_groups != required_curve_groups:
        raise AssertionError("Calibration curve groups are incomplete")
    numeric = results.select_dtypes(include=["number"])
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise AssertionError("Infinite values detected in calibration results")
    platt = invariance.loc[
        invariance["comparison"] == "Platt vs Uncalibrated"
    ]
    return {
        "expected_model_trainings": expected_model_trainings,
        "actual_model_trainings": int(grouped.ngroups),
        "expected_branch_rows": expected_branch_rows,
        "actual_branch_rows": int(len(results)),
        "shared_raw_hash_check": True,
        "fixed_data_check": True,
        "platt_spearman_min": float(platt["spearman_rank_correlation"].min()),
        "platt_validation_disagreement_total": int(
            platt["validation_disagreement_count"].sum()
        ),
        "platt_test_disagreement_total": int(platt["test_disagreement_count"].sum()),
    }


def make_calibration_figures(
    results: pd.DataFrame,
    curves: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Create full-scale and low-probability publication reliability diagrams."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "calibration_reliability_full.png",
        output_dir / "calibration_reliability_low_probability_zoom.png",
    ]
    _plot_reliability(results, curves, paths[0], zoom=False)
    _plot_reliability(results, curves, paths[1], zoom=True)
    return paths


def _plot_reliability(
    results: pd.DataFrame,
    curves: pd.DataFrame,
    output_path: Path,
    *,
    zoom: bool,
) -> None:
    data = curves.loc[
        (curves["split"] == "test")
        & (
            curves["binning_method"]
            == ("quantile" if zoom else "uniform")
        )
        & (curves["sample_count"] > 0)
    ].copy()
    schemes = ["Raw", "Recipient history (w=2)"]
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.4, 6.2),
        constrained_layout=True,
        sharex=not zoom,
        sharey=not zoom,
    )
    for row_index, scheme in enumerate(schemes):
        for column_index, fold in enumerate([1, 2, 3]):
            axis = axes[row_index, column_index]
            panel = data.loc[(data["scheme"] == scheme) & (data["fold"] == fold)]
            result_panel = results.loc[
                (results["scheme"] == scheme) & (results["fold"] == fold)
            ]
            if result_panel.empty:
                axis.set_axis_off()
                continue
            if zoom:
                zoom_upper = float(
                    np.clip(result_panel["test_probability_p99"].max() * 1.25, 0.005, 0.2)
                )
                visible = panel.loc[
                    panel["mean_predicted_probability"] <= zoom_upper
                ]
                observed_upper = (
                    float(visible["observed_fraud_rate"].max()) if not visible.empty else 0.0
                )
                y_upper = float(np.clip(max(zoom_upper, observed_upper * 1.15), 0.01, 0.3))
                axis.set_xlim(0, zoom_upper)
                axis.set_ylim(0, y_upper)
                reference_upper = min(zoom_upper, y_upper)
                axis.text(
                    0.98,
                    0.04,
                    f"99th-pct. zoom: x≤{zoom_upper:.3g}",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    color="#555555",
                )
            else:
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
                reference_upper = 1.0
            axis.plot(
                [0, reference_upper],
                [0, reference_upper],
                color="#555555",
                linestyle="--",
                linewidth=1.0,
                label="Perfect calibration",
                zorder=1,
            )
            for method in METHOD_ORDER:
                subset = panel.loc[panel["calibration_method"] == method]
                aggregated = (
                    subset.groupby("bin_id", as_index=False)
                    .agg(
                        mean_probability=("mean_predicted_probability", "mean"),
                        observed_mean=("observed_fraud_rate", "mean"),
                        observed_std=("observed_fraud_rate", "std"),
                    )
                    .dropna(subset=["mean_probability", "observed_mean"])
                    .sort_values("mean_probability")
                )
                if aggregated.empty:
                    continue
                x = aggregated["mean_probability"].to_numpy(dtype=float)
                y = aggregated["observed_mean"].to_numpy(dtype=float)
                spread = aggregated["observed_std"].fillna(0).to_numpy(dtype=float)
                color = METHOD_COLORS[method]
                axis.plot(
                    x,
                    y,
                    color=color,
                    marker=METHOD_MARKERS[method],
                    markersize=3.8,
                    linewidth=1.35,
                    label=method,
                    zorder=3,
                )
                axis.fill_between(
                    x,
                    np.clip(y - spread, 0, 1),
                    np.clip(y + spread, 0, 1),
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                    zorder=2,
                )
            axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.65)
            axis.tick_params(labelsize=8)
            if row_index == 0:
                axis.set_title(f"Fold {fold}", fontsize=10, fontweight="semibold")
            if column_index == 0:
                axis.set_ylabel(
                    f"{scheme}\nObserved fraud rate", fontsize=9, linespacing=1.5
                )
            if row_index == 1:
                axis.set_xlabel("Mean predicted probability", fontsize=9)
            axis.text(
                0.02,
                0.96,
                f"({chr(97 + row_index * 3 + column_index)})",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
                fontweight="semibold",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_calibration_report(
    path: Path,
    *,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    invariance: pd.DataFrame,
    bayes: pd.DataFrame,
    reliability_pairwise: pd.DataFrame,
    checks: dict[str, int | bool | float],
    figure_paths: list[Path],
) -> None:
    """Write the pre-specified three-layer calibration report."""
    reliability_overall = (
        results.groupby("calibration_method", as_index=False)
        .agg(
            test_brier_mean=("test_brier", "mean"),
            test_brier_std=("test_brier", "std"),
            ece_uniform_mean=("test_ece_uniform_10", "mean"),
            ece_uniform_std=("test_ece_uniform_10", "std"),
            ece_quantile_mean=("test_ece_quantile_10", "mean"),
            ece_quantile_std=("test_ece_quantile_10", "std"),
            log_loss_mean=("test_log_loss", "mean"),
            log_loss_std=("test_log_loss", "std"),
        )
        .set_index("calibration_method")
        .reindex(METHOD_ORDER)
        .reset_index()
    )
    platt = invariance.loc[invariance["comparison"] == "Platt vs Uncalibrated"]
    isotonic = invariance.loc[
        invariance["comparison"] == "Isotonic vs Uncalibrated"
    ]
    wins = _reliability_win_table(reliability_pairwise)
    bayes_summary = (
        bayes.groupby("calibration_method", as_index=False)
        .agg(
            alert_rate_mean=("final_alert_rate", "mean"),
            recall_mean=("recall", "mean"),
            business_cost_mean=("business_cost", "mean"),
            cost_saving_rate_mean=("cost_saving_rate", "mean"),
            capacity_binding_rate=("capacity_binding", "mean"),
        )
        .set_index("calibration_method")
        .reindex(METHOD_ORDER)
        .reset_index()
    )
    merged = results.pivot_table(
        index=["fold", "model_id", "seed"],
        columns="calibration_method",
        values=["val_brier", "test_brier", "val_log_loss", "test_log_loss"],
    )
    iso_overfit_brier = int(
        (
            (merged[("val_brier", "Isotonic")] < merged[("val_brier", "Uncalibrated")])
            & (merged[("test_brier", "Isotonic")] > merged[("test_brier", "Uncalibrated")])
        ).sum()
    )
    iso_overfit_log = int(
        (
            (merged[("val_log_loss", "Isotonic")] < merged[("val_log_loss", "Uncalibrated")])
            & (merged[("test_log_loss", "Isotonic")] > merged[("test_log_loss", "Uncalibrated")])
        ).sum()
    )
    lines = [
        "# Calibration ablation report",
        "",
        "## 1. Experiment Objective",
        "",
        "Separate probability reliability, empirical decision invariance, and the "
        "probability-semantic Bayes decision analysis.",
        "",
        "## 2. Frozen Protocol",
        "",
        "PaySim, three rolling folds, five seeds, feature definitions, LightGBM parameters, "
        "validation calibration/threshold split, FN cost 100, FP cost 1, and capacity 3% "
        "remain unchanged. Test labels select no method or threshold.",
        "",
        "## 3. Calibration Methods",
        "",
        "Uncalibrated identity scores, fold/seed-specific Platt Scaling, and fold/seed-specific "
        "Isotonic Regression share one underlying LightGBM score vector.",
        "",
        "## 4. Reliability Results",
        "",
        "| Method | Test Brier | ECE uniform-10 | ECE quantile-10 | Test log loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in reliability_overall.itertuples(index=False):
        lines.append(
            f"| {row.calibration_method} | {row.test_brier_mean:.6f} ± "
            f"{row.test_brier_std:.6f} | {row.ece_uniform_mean:.6f} ± "
            f"{row.ece_uniform_std:.6f} | {row.ece_quantile_mean:.6f} ± "
            f"{row.ece_quantile_std:.6f} | {row.log_loss_mean:.6f} ± "
            f"{row.log_loss_std:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 5. Reliability Diagrams",
            "",
            *[f"- `{figure_path.name}`" for figure_path in figure_paths],
            "The full-range figure uses uniform bins. The explicitly labelled low-probability "
            "zoom uses quantile bins to resolve the dense rare-event probability region; both "
            "retain per-seed machine-readable points.",
            "",
            "## 6. Platt Monotonicity Audit",
            "",
            f"Minimum Spearman correlation: {platt['spearman_rank_correlation'].min():.12f}. "
            f"Maximum absolute calibrated PR-AUC difference: "
            f"{platt['calibrated_pr_auc_difference'].abs().max():.8g}.",
            "",
            "## 7. Empirical Decision Invariance",
            "",
            f"Platt validation/test disagreement totals: "
            f"{int(platt['validation_disagreement_count'].sum())}/"
            f"{int(platt['test_disagreement_count'].sum())}. "
            f"Maximum absolute business-cost difference: "
            f"{platt['business_cost_difference'].abs().max():.2f}.",
            "",
            "## 8. Isotonic Tie Effects",
            "",
            f"Isotonic validation/test disagreement totals: "
            f"{int(isotonic['validation_disagreement_count'].sum())}/"
            f"{int(isotonic['test_disagreement_count'].sum())}. "
            f"Maximum absolute calibrated PR-AUC change: "
            f"{isotonic['calibrated_pr_auc_difference'].abs().max():.6f}.",
            "",
            "## 9. Threshold Percentile",
            "",
            "Raw numerical threshold CV is retained as a scale-specific diagnostic only. "
            "Cross-method interpretation uses raw-score threshold percentile, selected fraction, "
            "decision disagreement, and Jaccard similarity.",
            "",
            "## 10. Capacity Transfer",
            "",
            "Capacity deviation belongs to the future Test decision. Identical Raw/Platt "
            "decisions necessarily imply identical capacity deviation.",
            "",
            "## 11. Bayes Cost Threshold Auxiliary Analysis",
            "",
            "The auxiliary threshold is fixed at 1/101 and capped deterministically at 3%.",
            "",
            "| Method | Alert rate | Recall | Business cost | Cost-saving rate | Cap binding |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bayes_summary.itertuples(index=False):
        lines.append(
            f"| {row.calibration_method} | {row.alert_rate_mean:.2%} | "
            f"{row.recall_mean:.2%} | {row.business_cost_mean:.2f} | "
            f"{row.cost_saving_rate_mean:.2%} | {row.capacity_binding_rate:.0%} |"
        )
    fold3 = summary.loc[summary["fold"] == 3]
    lines.extend(
        [
            "",
            "## 12. Fold 3 Drift",
            "",
            "Fold 3 is retained as the high-base-rate future window. Reliability and decision "
            "metrics are reported without post-hoc method selection.",
        ]
    )
    for row in fold3.itertuples(index=False):
        lines.append(
            f"- {row.scheme}, {row.calibration_method}: Brier "
            f"{row.test_brier_mean:.6f} ± {row.test_brier_std:.6f}; ECE-u "
            f"{row.test_ece_uniform_10_mean:.6f}; cost "
            f"{row.business_cost_mean:.2f} ± {row.business_cost_std:.2f}."
        )
    lines.extend(
        [
            "",
            "## 13. Isotonic Generalization / Overfitting Risk",
            "",
            f"Validation-improved/Test-worsened cases: Brier {iso_overfit_brier}/30; "
            f"log loss {iso_overfit_log}/30. This is descriptive evidence, not a causal claim.",
            "",
            "## 14. Main Conclusions",
            "",
            f"Integrity checks: {checks['actual_model_trainings']}/"
            f"{checks['expected_model_trainings']} underlying trainings and "
            f"{checks['actual_branch_rows']}/{checks['expected_branch_rows']} branches.",
            "Reliability, empirical threshold decisions, and Bayes probability decisions must "
            "be interpreted separately. The 15 Fold-Seed comparisons per scheme are stochastic "
            "repetitions over three temporal positions, not 15 independent test sets.",
            "",
            "### Descriptive reliability wins",
            "",
            "| Comparison | Metric | Better | Equal | Worse |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in wins.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.metric} | {row.better} | {row.equal} | {row.worse} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reliability_win_table(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "brier_improvement",
        "ece_uniform_improvement",
        "ece_quantile_improvement",
        "log_loss_improvement",
    ]
    tolerance = 1e-12
    for comparison, subset in pairwise.groupby("comparison", sort=False):
        for metric in metrics:
            values = subset[metric]
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "better": int((values > tolerance).sum()),
                    "equal": int((values.abs() <= tolerance).sum()),
                    "worse": int((values < -tolerance).sum()),
                }
            )
    return pd.DataFrame(rows)
