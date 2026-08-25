"""Paired moving time-block Bootstrap over an accepted frozen prediction ledger.

This module deliberately has no dependency on model training, calibration fitting,
feature engineering, validation labels, or threshold optimization.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score

from fraudx.config import ExperimentConfig
from fraudx.prediction_reconstruction import require_reconstruction_passed

N_BOOTSTRAP = 5000
BOOTSTRAP_RNG_SEED = 20260820
LIGHTGBM_SEEDS = (42, 52, 62, 72, 82)
FN_COST = 100.0
FP_COST = 1.0
METRICS = (
    "delta_recall",
    "cost_reduction",
    "cost_reduction_per_100k",
    "delta_cost_saving_rate",
    "delta_pr_auc",
)
METRIC_LABELS = {
    "delta_recall": "Δ Recall",
    "cost_reduction": "Business cost reduction",
    "cost_reduction_per_100k": "Cost reduction per\n100k transactions",
    "delta_cost_saving_rate": "Δ Cost-saving rate",
    "delta_pr_auc": "Δ PR-AUC",
}


def block_length_for_steps(unique_step_count: int) -> int:
    """Return the pre-registered max(2, round(T^(1/3))) block length."""
    if unique_step_count < 2:
        raise ValueError("At least two unique Test steps are required")
    return max(2, int(round(unique_step_count ** (1 / 3))))


def moving_block_positions(
    unique_step_count: int,
    block_length: int,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw overlapping non-circular moving blocks and truncate to T positions."""
    if not 2 <= block_length <= unique_step_count:
        raise ValueError("Block length must be in [2, unique_step_count]")
    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive")
    legal_block_count = unique_step_count - block_length + 1
    blocks_needed = math.ceil(unique_step_count / block_length)
    starts = rng.integers(0, legal_block_count, size=(repetitions, blocks_needed))
    offsets = np.arange(block_length, dtype=np.int32)
    positions = (starts[..., None] + offsets).reshape(repetitions, -1)
    return positions[:, :unique_step_count].astype(np.int32, copy=False)


def position_multiplicities(
    sampled_positions: np.ndarray, unique_step_count: int
) -> np.ndarray:
    """Convert sampled step positions into per-replicate multiplicity weights."""
    positions = np.asarray(sampled_positions, dtype=np.int32)
    if positions.ndim != 2 or positions.shape[1] != unique_step_count:
        raise ValueError("Sampled positions must have shape (repetitions, T)")
    if positions.min() < 0 or positions.max() >= unique_step_count:
        raise ValueError("Sampled position is outside the legal non-circular range")
    result = np.zeros((positions.shape[0], unique_step_count), dtype=np.int16)
    for index, row in enumerate(positions):
        result[index] = np.bincount(row, minlength=unique_step_count)
    return result


def run_timeblock_bootstrap(config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """Run the locked 5000-replicate paired moving time-block Bootstrap."""
    output_dir = config.output_dir / "bootstrap"
    status_path = output_dir / "bootstrap_prediction_reconstruction_status.json"
    require_reconstruction_passed(status_path)
    ledger_path = output_dir / "rolling_multiseed_paired_predictions.parquet"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Accepted prediction ledger not found: {ledger_path}")

    connection = duckdb.connect(database=":memory:")
    seed_effect_frames: list[pd.DataFrame] = []
    fold_replicate_frames: list[pd.DataFrame] = []
    fold_observed: dict[int, dict[str, float]] = {}
    block_details: list[dict[str, int]] = []
    seed_sequence = np.random.SeedSequence(BOOTSTRAP_RNG_SEED)
    child_sequences = seed_sequence.spawn(3)
    try:
        for fold, child_sequence in zip((1, 2, 3), child_sequences, strict=True):
            common = _load_fold_seed(connection, ledger_path, fold, LIGHTGBM_SEEDS[0])
            unique_steps = np.sort(common["step"].unique())
            step_lookup = {int(step): index for index, step in enumerate(unique_steps)}
            step_positions = common["step"].map(step_lookup).to_numpy(dtype=np.int32)
            unique_step_count = len(unique_steps)
            block_length = block_length_for_steps(unique_step_count)
            legal_block_count = unique_step_count - block_length + 1
            rng = np.random.default_rng(child_sequence)
            sampled_positions = moving_block_positions(
                unique_step_count, block_length, N_BOOTSTRAP, rng
            )
            multiplicities = position_multiplicities(sampled_positions, unique_step_count)
            step_transaction_count = np.bincount(
                step_positions, minlength=unique_step_count
            ).astype(np.int64)
            step_fraud_count = np.bincount(
                step_positions,
                weights=common["y_true"].to_numpy(dtype=np.int8),
                minlength=unique_step_count,
            ).astype(np.int64)
            bootstrap_transaction_count = multiplicities @ step_transaction_count
            bootstrap_fraud_count = multiplicities @ step_fraud_count
            bootstrap_fraud_rate = np.divide(
                bootstrap_fraud_count,
                bootstrap_transaction_count,
                out=np.full(N_BOOTSTRAP, np.nan),
                where=bootstrap_transaction_count > 0,
            )
            block_details.append(
                {
                    "fold": fold,
                    "unique_test_steps": unique_step_count,
                    "block_length": block_length,
                    "legal_block_count": legal_block_count,
                }
            )
            print(
                f"Fold {fold}: T={unique_step_count}, L={block_length}, "
                f"legal blocks={legal_block_count}",
                flush=True,
            )

            fold_effect_sums = {metric: np.zeros(N_BOOTSTRAP) for metric in METRICS}
            observed_by_seed: list[dict[str, float]] = []
            for lightgbm_seed in LIGHTGBM_SEEDS:
                data = (
                    common
                    if lightgbm_seed == LIGHTGBM_SEEDS[0]
                    else _load_fold_seed(connection, ledger_path, fold, lightgbm_seed)
                )
                _assert_seed_alignment(common, data, fold, lightgbm_seed)
                effects = _seed_bootstrap_effects(
                    data,
                    step_positions=step_positions,
                    multiplicities=multiplicities,
                    bootstrap_transaction_count=bootstrap_transaction_count,
                    bootstrap_fraud_count=bootstrap_fraud_count,
                )
                observed = _observed_effects(data)
                observed_by_seed.append(observed)
                for metric in METRICS:
                    fold_effect_sums[metric] += effects[metric]
                seed_effect_frames.append(
                    pd.DataFrame(
                        {
                            "bootstrap_id": np.arange(1, N_BOOTSTRAP + 1),
                            "fold": fold,
                            "lightgbm_seed": lightgbm_seed,
                            **effects,
                        }
                    )
                )
                print(
                    f"Bootstrap PR-AUC/effects completed: Fold {fold}, seed {lightgbm_seed}",
                    flush=True,
                )

            fold_effects = {
                metric: fold_effect_sums[metric] / len(LIGHTGBM_SEEDS) for metric in METRICS
            }
            fold_observed[fold] = {
                metric: float(np.mean([row[metric] for row in observed_by_seed]))
                for metric in METRICS
            }
            fold_replicate_frames.append(
                pd.DataFrame(
                    {
                        "bootstrap_id": np.arange(1, N_BOOTSTRAP + 1),
                        "fold": fold,
                        "bootstrap_transaction_count": bootstrap_transaction_count,
                        "bootstrap_fraud_count": bootstrap_fraud_count,
                        "bootstrap_fraud_rate": bootstrap_fraud_rate,
                        "block_length": block_length,
                        "seed_effect_aggregation_method": "mean_across_5_fixed_seeds",
                        **fold_effects,
                    }
                )
            )
    finally:
        connection.close()

    seed_effects = pd.concat(seed_effect_frames, ignore_index=True)
    fold_replicates = pd.concat(fold_replicate_frames, ignore_index=True)
    overall_replicates = _overall_equal_weight_replicates(fold_replicates)
    fold_summary = _summarize_fold_effects(fold_replicates, fold_observed)
    overall_observed = {
        metric: float(np.mean([fold_observed[fold][metric] for fold in (1, 2, 3)]))
        for metric in METRICS
    }
    overall_summary = _summarize_overall_effects(overall_replicates, overall_observed)

    _write_parquet(seed_effects, output_dir / "timeblock_bootstrap_seed_effects.parquet")
    _write_parquet(fold_replicates, output_dir / "timeblock_bootstrap_replicates.parquet")
    _write_parquet(
        overall_replicates, output_dir / "timeblock_bootstrap_overall_replicates.parquet"
    )
    fold_summary.to_csv(output_dir / "timeblock_bootstrap_fold_summary.csv", index=False)
    overall_summary.to_csv(output_dir / "timeblock_bootstrap_overall_summary.csv", index=False)
    protocol = _protocol_metadata(block_details)
    (output_dir / "timeblock_bootstrap_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    _make_figures(output_dir / "figures", fold_summary, overall_summary, fold_replicates,
                  overall_replicates)
    _write_report(
        output_dir / "timeblock_bootstrap_report.md",
        fold_summary=fold_summary,
        overall_summary=overall_summary,
        block_details=block_details,
    )
    return {
        "seed_effects": seed_effects,
        "fold_replicates": fold_replicates,
        "overall_replicates": overall_replicates,
        "fold_summary": fold_summary,
        "overall_summary": overall_summary,
    }


def _seed_bootstrap_effects(
    data: pd.DataFrame,
    *,
    step_positions: np.ndarray,
    multiplicities: np.ndarray,
    bootstrap_transaction_count: np.ndarray,
    bootstrap_fraud_count: np.ndarray,
) -> dict[str, np.ndarray]:
    unique_step_count = multiplicities.shape[1]
    labels = data["y_true"].to_numpy(dtype=np.int8)
    raw_decision = data["raw_binary_decision"].to_numpy(dtype=np.int8)
    recipient_decision = data["recipient_binary_decision"].to_numpy(dtype=np.int8)
    raw_tp_step = np.bincount(
        step_positions, weights=labels * raw_decision, minlength=unique_step_count
    )
    recipient_tp_step = np.bincount(
        step_positions, weights=labels * recipient_decision, minlength=unique_step_count
    )
    raw_fp_step = np.bincount(
        step_positions, weights=(1 - labels) * raw_decision, minlength=unique_step_count
    )
    recipient_fp_step = np.bincount(
        step_positions,
        weights=(1 - labels) * recipient_decision,
        minlength=unique_step_count,
    )
    raw_tp = multiplicities @ raw_tp_step
    recipient_tp = multiplicities @ recipient_tp_step
    raw_fp = multiplicities @ raw_fp_step
    recipient_fp = multiplicities @ recipient_fp_step
    raw_fn = bootstrap_fraud_count - raw_tp
    recipient_fn = bootstrap_fraud_count - recipient_tp
    raw_cost = FN_COST * raw_fn + FP_COST * raw_fp
    recipient_cost = FN_COST * recipient_fn + FP_COST * recipient_fp
    cost_reduction = raw_cost - recipient_cost
    raw_recall = _safe_divide(raw_tp, bootstrap_fraud_count)
    recipient_recall = _safe_divide(recipient_tp, bootstrap_fraud_count)
    all_normal_cost = FN_COST * bootstrap_fraud_count
    raw_saving_rate = _safe_divide(all_normal_cost - raw_cost, all_normal_cost)
    recipient_saving_rate = _safe_divide(
        all_normal_cost - recipient_cost, all_normal_cost
    )
    raw_ap = bootstrap_average_precision(
        multiplicities,
        step_positions,
        labels,
        data["raw_pr_auc_score"].to_numpy(dtype=np.float64),
    )
    recipient_ap = bootstrap_average_precision(
        multiplicities,
        step_positions,
        labels,
        data["recipient_pr_auc_score"].to_numpy(dtype=np.float64),
    )
    return {
        "delta_recall": recipient_recall - raw_recall,
        "cost_reduction": cost_reduction,
        "cost_reduction_per_100k": _safe_divide(
            cost_reduction * 100000.0, bootstrap_transaction_count
        ),
        "delta_cost_saving_rate": recipient_saving_rate - raw_saving_rate,
        "delta_pr_auc": recipient_ap - raw_ap,
    }


def bootstrap_average_precision(
    multiplicities: np.ndarray,
    step_positions: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    batch_size: int = 50,
) -> np.ndarray:
    """Compute exact AP under step multiplicity weights without resorting samples."""
    weights = np.asarray(multiplicities)
    positions = np.asarray(step_positions, dtype=np.int32)
    targets = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(scores, dtype=np.float64)
    if not (len(positions) == len(targets) == len(probabilities)):
        raise ValueError("Step positions, labels, and scores must align")
    unique_scores, score_indices = np.unique(probabilities, return_inverse=True)
    shape = (weights.shape[1], len(unique_scores))
    total_matrix = sparse.coo_matrix(
        (np.ones(len(targets), dtype=np.float64), (positions, score_indices)), shape=shape
    ).tocsr()
    positive_mask = targets == 1
    positive_matrix = sparse.coo_matrix(
        (
            np.ones(int(positive_mask.sum()), dtype=np.float64),
            (positions[positive_mask], score_indices[positive_mask]),
        ),
        shape=shape,
    ).tocsr()
    result = np.full(weights.shape[0], np.nan, dtype=np.float64)
    for start in range(0, weights.shape[0], batch_size):
        stop = min(start + batch_size, weights.shape[0])
        batch = weights[start:stop].astype(np.float64, copy=False)
        positives = np.asarray(batch @ positive_matrix)[:, ::-1]
        totals = np.asarray(batch @ total_matrix)[:, ::-1]
        cumulative_positives = np.cumsum(positives, axis=1)
        cumulative_totals = np.cumsum(totals, axis=1)
        precision = np.divide(
            cumulative_positives,
            cumulative_totals,
            out=np.zeros_like(cumulative_positives),
            where=cumulative_totals > 0,
        )
        positive_totals = positives.sum(axis=1)
        result[start:stop] = np.divide(
            np.sum(positives * precision, axis=1),
            positive_totals,
            out=np.full(stop - start, np.nan),
            where=positive_totals > 0,
        )
    return result


def _observed_effects(data: pd.DataFrame) -> dict[str, float]:
    labels = data["y_true"].to_numpy(dtype=np.int8)
    raw_decision = data["raw_binary_decision"].to_numpy(dtype=np.int8)
    recipient_decision = data["recipient_binary_decision"].to_numpy(dtype=np.int8)
    fraud_count = int(labels.sum())
    transaction_count = len(labels)
    raw_tp = int(np.sum(labels * raw_decision))
    recipient_tp = int(np.sum(labels * recipient_decision))
    raw_fp = int(np.sum((1 - labels) * raw_decision))
    recipient_fp = int(np.sum((1 - labels) * recipient_decision))
    raw_cost = FN_COST * (fraud_count - raw_tp) + FP_COST * raw_fp
    recipient_cost = FN_COST * (fraud_count - recipient_tp) + FP_COST * recipient_fp
    all_normal_cost = FN_COST * fraud_count
    cost_reduction = raw_cost - recipient_cost
    return {
        "delta_recall": (recipient_tp - raw_tp) / fraud_count,
        "cost_reduction": cost_reduction,
        "cost_reduction_per_100k": cost_reduction / transaction_count * 100000.0,
        "delta_cost_saving_rate": cost_reduction / all_normal_cost,
        "delta_pr_auc": float(
            average_precision_score(labels, data["recipient_pr_auc_score"])
            - average_precision_score(labels, data["raw_pr_auc_score"])
        ),
    }


def _load_fold_seed(
    connection: duckdb.DuckDBPyConnection, ledger_path: Path, fold: int, seed: int
) -> pd.DataFrame:
    columns = [
        "transaction_id",
        "test_order_index",
        "step",
        "y_true",
        "raw_binary_decision",
        "recipient_binary_decision",
        "raw_pr_auc_score",
        "recipient_pr_auc_score",
        "test_split_hash",
    ]
    projection = ", ".join(columns)
    # Projection contains only the fixed internal column list declared above.
    return connection.execute(
        f"SELECT {projection} FROM read_parquet(?) WHERE fold=? AND seed=? "  # nosec B608
        "ORDER BY test_order_index",
        [str(ledger_path), fold, seed],
    ).df()


def _assert_seed_alignment(
    reference: pd.DataFrame, candidate: pd.DataFrame, fold: int, seed: int
) -> None:
    for column in ("transaction_id", "test_order_index", "step", "y_true"):
        if not np.array_equal(reference[column].to_numpy(), candidate[column].to_numpy()):
            raise AssertionError(f"Fold {fold}, seed {seed}: {column} alignment failed")
    if reference["test_split_hash"].iloc[0] != candidate["test_split_hash"].iloc[0]:
        raise AssertionError(f"Fold {fold}, seed {seed}: Test split hash changed")


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan),
        where=np.asarray(denominator) != 0,
    )
    return result


def _overall_equal_weight_replicates(fold_replicates: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_replicates.groupby("bootstrap_id", as_index=False)[list(METRICS)]
        .mean()
        .assign(weighting_method="equal_fold_weight")
    )


def _summary_row(
    *,
    scope: str | int,
    metric: str,
    observed_effect: float,
    values: np.ndarray,
) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    valid = len(finite)
    invalid = N_BOOTSTRAP - valid
    lower, upper = (
        np.percentile(finite, [2.5, 97.5]) if valid else (float("nan"), float("nan"))
    )
    crosses = bool(lower <= 0 <= upper) if valid else True
    return {
        "scope": scope,
        "metric": metric,
        "observed_effect": observed_effect,
        "bootstrap_mean": float(np.mean(finite)) if valid else np.nan,
        "bootstrap_se": float(np.std(finite, ddof=1)) if valid > 1 else np.nan,
        "ci_lower_95": float(lower),
        "ci_upper_95": float(upper),
        "total_replicates": N_BOOTSTRAP,
        "valid_replicates": valid,
        "invalid_replicates": invalid,
        "invalid_rate": invalid / N_BOOTSTRAP,
        "ci_crosses_zero": crosses,
        "invalid_rate_warning": invalid / N_BOOTSTRAP > 0.05,
    }


def _summarize_fold_effects(
    fold_replicates: pd.DataFrame, fold_observed: dict[int, dict[str, float]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in (1, 2, 3):
        subset = fold_replicates.loc[fold_replicates["fold"] == fold]
        for metric in METRICS:
            row = _summary_row(
                scope=fold,
                metric=metric,
                observed_effect=fold_observed[fold][metric],
                values=subset[metric].to_numpy(),
            )
            row["fold"] = row.pop("scope")
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_overall_effects(
    overall_replicates: pd.DataFrame, observed: dict[str, float]
) -> pd.DataFrame:
    rows = [
        _summary_row(
            scope="Overall",
            metric=metric,
            observed_effect=observed[metric],
            values=overall_replicates[metric].to_numpy(),
        )
        for metric in METRICS
    ]
    result = pd.DataFrame(rows).drop(columns="scope")
    result["weighting_method"] = "equal_fold_weight"
    return result


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("result_frame", frame)
        escaped = str(path.resolve()).replace("'", "''").replace("\\", "/")
        connection.execute(
            f"COPY result_frame TO '{escaped}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
    finally:
        connection.close()


def _protocol_metadata(block_details: list[dict[str, int]]) -> dict[str, Any]:
    return {
        "bootstrap_type": "moving_time_block",
        "circular": False,
        "repetitions": N_BOOTSTRAP,
        "rng_seed": BOOTSTRAP_RNG_SEED,
        "random_streams": "NumPy SeedSequence child stream per Fold",
        "confidence_interval": "percentile_95",
        "block_length_rule": "max(2, round(T^(1/3)))",
        "resampling_unit": "unique_test_step_position",
        "schemes": "Raw w1 vs Recipient history w2",
        "fold_weighting": "equal",
        "seed_aggregation": "mean_within_fold",
        "lightgbm_seeds": list(LIGHTGBM_SEEDS),
        "threshold_reoptimization": False,
        "recalibration": False,
        "retraining": False,
        "bootstrap_stage_fit_calls": {"lightgbm": 0, "platt": 0, "threshold_search": 0},
        "fold_block_details": block_details,
    }


def _make_figures(
    figures_dir: Path,
    fold_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    fold_replicates: pd.DataFrame,
    overall_replicates: pd.DataFrame,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ["#4477AA", "#66CCEE", "#228833", "#AA3377"]
    scopes = ["Fold 1", "Fold 2", "Fold 3", "Overall"]
    forest_metrics = (
        "delta_recall",
        "cost_reduction_per_100k",
        "delta_cost_saving_rate",
        "delta_pr_auc",
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.45), constrained_layout=True)
    for axis, metric in zip(axes, forest_metrics, strict=True):
        fold_rows = fold_summary.loc[fold_summary["metric"] == metric].sort_values("fold")
        overall_row = overall_summary.loc[overall_summary["metric"] == metric]
        points = np.r_[fold_rows["observed_effect"], overall_row["observed_effect"]]
        lower = np.r_[fold_rows["ci_lower_95"], overall_row["ci_lower_95"]]
        upper = np.r_[fold_rows["ci_upper_95"], overall_row["ci_upper_95"]]
        y = np.arange(4)
        for index in range(4):
            axis.errorbar(
                points[index],
                y[index],
                xerr=[[points[index] - lower[index]], [upper[index] - points[index]]],
                fmt="o",
                color=colors[index],
                ecolor=colors[index],
                capsize=2.2,
                markersize=4.2,
                linewidth=1.1,
            )
        axis.axvline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
        axis.set_yticks(y, scopes if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_xlabel(METRIC_LABELS[metric])
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    _save_figure(fig, figures_dir / "timeblock_bootstrap_forest")

    distribution_data = [
        fold_replicates.loc[fold_replicates["fold"] == fold,
                            "cost_reduction_per_100k"].to_numpy()
        for fold in (1, 2, 3)
    ] + [overall_replicates["cost_reduction_per_100k"].to_numpy()]
    summary_rows = [
        fold_summary.loc[
            (fold_summary["fold"] == fold)
            & (fold_summary["metric"] == "cost_reduction_per_100k")
        ].iloc[0]
        for fold in (1, 2, 3)
    ] + [overall_summary.loc[
        overall_summary["metric"] == "cost_reduction_per_100k"
    ].iloc[0]]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)
    for axis, values, summary, scope, color in zip(
        axes.ravel(), distribution_data, summary_rows, scopes, colors, strict=True
    ):
        axis.hist(values[np.isfinite(values)], bins=45, density=True, color=color,
                  alpha=0.78, edgecolor="white", linewidth=0.3)
        axis.axvline(0, color="#555555", linestyle="--", linewidth=0.9, label="No effect")
        axis.axvline(summary["observed_effect"], color="#111111", linewidth=1.2,
                     label="Observed")
        axis.axvspan(summary["ci_lower_95"], summary["ci_upper_95"], color="#BBBBBB",
                     alpha=0.25, label="95% CI")
        finite_values = values[np.isfinite(values)]
        display_upper = float(np.percentile(finite_values, 99.5))
        display_lower = min(0.0, float(np.percentile(finite_values, 0.5)))
        axis.set_xlim(display_lower, display_upper)
        axis.text(0.02, 0.96, scope, transform=axis.transAxes, va="top", fontweight="bold")
        axis.text(
            0.98,
            0.96,
            "Central 99% shown",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#555555",
        )
        axis.set_xlabel("Cost reduction per 100k transactions")
        axis.set_ylabel("Density")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    _save_figure(fig, figures_dir / "timeblock_bootstrap_cost_distribution")


def _save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    fold_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    block_details: list[dict[str, int]],
) -> None:
    lines = [
        "# Moving Time-block Bootstrap Report",
        "",
        "## 1. Objective",
        "",
        "Estimate uncertainty in the frozen paired Raw w=1 versus Recipient history w=2 "
        "effects while preserving local PaySim step dependence.",
        "",
        "## 2. Frozen Prediction Reconstruction",
        "",
        "The accepted transaction-level ledger contains the exact historical Test predictions "
        "for 3 Folds x 5 fixed model seeds x 2 paired schemes.",
        "",
        "## 3. Reconstruction Audit",
        "",
        "All Test indices, labels, probability hashes, locked and regenerated thresholds, "
        "confusion matrices, PR-AUC values, and business costs matched the historical "
        "multi-seed results.",
        "",
        "## 4. Bootstrap Protocol",
        "",
        f"Overlapping non-circular moving blocks; {N_BOOTSTRAP} repetitions; RNG seed "
        f"{BOOTSTRAP_RNG_SEED}; percentile 95% CI. No retraining, recalibration, or threshold "
        "selection occurred.",
        "",
        "## 5. Fold-specific Block Lengths",
        "",
        "| Fold | Unique Test steps | Block length | Legal blocks |",
        "|---:|---:|---:|---:|",
    ]
    for row in block_details:
        lines.append(
            f"| {row['fold']} | {row['unique_test_steps']} | {row['block_length']} | "
            f"{row['legal_block_count']} |"
        )
    lines.extend(["", "## 6. Fold-specific Effects", ""])
    for metric in METRICS:
        lines.extend([f"### {METRIC_LABELS[metric]}", ""])
        for row in fold_summary.loc[fold_summary["metric"] == metric].itertuples():
            lines.append(
                f"- Fold {row.fold}: observed {row.observed_effect:+.6g}; bootstrap mean "
                f"{row.bootstrap_mean:+.6g}; SE {row.bootstrap_se:.6g}; 95% CI "
                f"[{row.ci_lower_95:+.6g}, {row.ci_upper_95:+.6g}]; "
                f"crosses zero={row.ci_crosses_zero}."
            )
        lines.append("")
    lines.extend(["## 7. Overall Equal-weight Effects", ""])
    for row in overall_summary.itertuples():
        lines.append(
            f"- {METRIC_LABELS[row.metric]}: observed {row.observed_effect:+.6g}; "
            f"bootstrap mean {row.bootstrap_mean:+.6g}; SE {row.bootstrap_se:.6g}; "
            f"95% CI [{row.ci_lower_95:+.6g}, {row.ci_upper_95:+.6g}]; "
            f"crosses zero={row.ci_crosses_zero}."
        )
    invalid = pd.concat(
        [
            fold_summary.assign(scope=lambda frame: "Fold " + frame["fold"].astype(str)),
            overall_summary.assign(scope="Overall"),
        ],
        ignore_index=True,
    )
    lines.extend(
        [
            "",
            "## 8. Confidence Intervals",
            "",
            "All intervals are pre-specified 2.5th and 97.5th percentiles; observed effects "
            "remain the paper point estimates.",
            "",
            "## 9. Fold 3 Drift Analysis",
            "",
            _fold3_width_statement(fold_summary),
            "",
            "## 10. Invalid Bootstrap Replicates",
            "",
            f"Maximum invalid rate across all scope-metric summaries: "
            f"{invalid['invalid_rate'].max():.2%}. No replicate was redrawn.",
            "",
            "## 11. Seed Aggregation Explanation",
            "",
            "Each Fold-replicate used one common sampled step sequence for both schemes and all "
            "five fixed LightGBM seeds. Effects were calculated per seed and then averaged within "
            "Fold; the three Fold effects were subsequently averaged with equal weight.",
            "",
            "## 12. Statistical Interpretation Limits",
            "",
            "Time-block Bootstrap does not create new independent Test sets. It quantifies "
            "within-window uncertainty for three pre-specified future windows while retaining "
            "local time structure. The five seeds are training repetitions, not independent "
            "future environments.",
            "",
            "## 13. Main Conclusion",
            "",
            _main_conclusion(overall_summary),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fold3_width_statement(fold_summary: pd.DataFrame) -> str:
    widths = fold_summary.assign(width=fold_summary["ci_upper_95"] - fold_summary["ci_lower_95"])
    comparisons: list[str] = []
    for metric in METRICS:
        subset = widths.loc[widths["metric"] == metric].set_index("fold")["width"]
        wider = bool(subset.loc[3] > max(subset.loc[1], subset.loc[2]))
        comparisons.append(f"{METRIC_LABELS[metric]}={'wider' if wider else 'not wider'}")
    return "Fold 3 versus both earlier Folds: " + "; ".join(comparisons) + "."


def _main_conclusion(overall_summary: pd.DataFrame) -> str:
    direction = overall_summary.set_index("metric")
    business = all(
        not bool(direction.loc[metric, "ci_crosses_zero"])
        and float(direction.loc[metric, "ci_lower_95"]) > 0
        for metric in ("delta_recall", "cost_reduction", "delta_cost_saving_rate")
    )
    pr_lower = float(direction.loc["delta_pr_auc", "ci_lower_95"])
    pr_upper = float(direction.loc["delta_pr_auc", "ci_upper_95"])
    pr_text = (
        "the PR-AUC effect remained below zero"
        if pr_upper < 0
        else "the PR-AUC effect remained above zero"
        if pr_lower > 0
        else "the PR-AUC interval crossed zero"
    )
    business_text = (
        "Overall Recall and business-utility effects retained the Recipient-favouring direction"
        if business
        else "At least one overall business-effect interval crossed zero"
    )
    return f"{business_text} under the pre-specified block resampling; {pr_text}."
