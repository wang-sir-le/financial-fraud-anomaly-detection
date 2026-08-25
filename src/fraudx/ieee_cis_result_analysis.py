"""Zero-training audit and directional external validation of frozen IEEE-CIS results."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudx.ieee_cis_experiment import (
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    MODEL_ORDER,
    SEEDS,
    array_hash,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    file_hash,
    fixed_capacity_decision,
)

COMPARISONS = (
    ("C01", "M0", "M1", "history-only effect"),
    ("C12", "M1", "M2", "class-weight contribution"),
    ("C23", "M2", "M3", "entity-granularity sensitivity"),
    ("C02", "M0", "M2", "pre-specified framework effect"),
)
EFFECT_METRICS = (
    "delta_pr_auc",
    "delta_roc_auc",
    "delta_precision_at_3pct",
    "delta_recall_at_3pct",
    "delta_f1_at_3pct",
    "delta_tp",
    "cost_reduction",
    "delta_cost_saving_rate",
)
STABILITY_METRICS = (
    "pr_auc",
    "recall_at_3pct",
    "business_cost",
    "cost_saving_rate",
)
ANALYSIS_OUTPUTS = (
    "result_integrity_audit.csv",
    "metric_recomputation_audit.csv",
    "fixed_capacity_dependency_audit.csv",
    "paired_effects_seed_level.csv",
    "paired_effects_fold_summary.csv",
    "paired_effects_overall_summary.csv",
    "seed_stability_summary.csv",
    "seed_temporal_variability.csv",
    "temporal_heterogeneity_analysis.csv",
    "ranking_decision_alignment.csv",
    "external_validation_matrix.csv",
    "result_analysis_metadata.json",
    "checksums.sha256",
    "IEEE_CIS_external_validation_report.md",
)


@dataclass(frozen=True)
class AnalysisPaths:
    """Frozen inputs and deterministic analysis output directory."""

    model_results_dir: Path
    temporal_protocol_path: Path
    history_metadata_path: Path
    temporal_audit_dir: Path
    paysim_results_dir: Path
    paysim_pair_status_path: Path
    output_dir: Path


def run_result_analysis(paths: AnalysisPaths) -> dict[str, Any]:
    """Audit frozen results, build paired analyses, and write external validation outputs."""
    inputs = _load_and_verify_inputs(paths)
    config = inputs["config"]
    run_metadata = inputs["run_metadata"]
    seed_metrics = inputs["seed_metrics"]
    integrity = audit_frozen_result_structure(config, run_metadata, seed_metrics)
    recomputation, ledger_profiles = recompute_prediction_ledger(
        paths.model_results_dir / "prediction_scores.parquet",
        seed_metrics,
        config,
    )
    if not bool(recomputation["passed"].all()):
        raise RuntimeError("Prediction Ledger metric recomputation audit failed")
    paired = make_paired_effects(seed_metrics)
    fold_summary = make_fold_effect_summary(paired)
    overall = make_fold_equal_weight_summary(fold_summary)
    dependency = make_fixed_capacity_dependency_audit(paired, seed_metrics)
    if not bool(dependency["passed"].all()):
        raise RuntimeError("Fixed-capacity algebraic dependency audit failed")
    stability = make_seed_stability_summary(seed_metrics)
    variability = make_seed_temporal_variability(seed_metrics)
    heterogeneity = make_temporal_heterogeneity(
        fold_summary,
        paths.temporal_audit_dir,
    )
    alignment = make_ranking_decision_alignment(fold_summary)
    paysim = load_paysim_evidence(paths.paysim_results_dir, paths.paysim_pair_status_path)
    external = make_external_validation_matrix(
        fold_summary=fold_summary,
        paired_effects=paired,
        paysim=paysim,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "result_integrity_audit.csv": integrity,
        "metric_recomputation_audit.csv": recomputation,
        "fixed_capacity_dependency_audit.csv": dependency,
        "paired_effects_seed_level.csv": paired,
        "paired_effects_fold_summary.csv": fold_summary,
        "paired_effects_overall_summary.csv": overall,
        "seed_stability_summary.csv": stability,
        "seed_temporal_variability.csv": variability,
        "temporal_heterogeneity_analysis.csv": heterogeneity,
        "ranking_decision_alignment.csv": alignment,
        "external_validation_matrix.csv": external,
    }
    for name, frame in tables.items():
        atomic_write_csv(frame, paths.output_dir / name)
    report_path = paths.output_dir / "IEEE_CIS_external_validation_report.md"
    write_external_validation_report(
        report_path,
        seed_metrics=seed_metrics,
        fold_summary=fold_summary,
        overall=overall,
        stability=stability,
        variability=variability,
        heterogeneity=heterogeneity,
        alignment=alignment,
        external=external,
        ledger_profiles=ledger_profiles,
        paysim=paysim,
    )
    metadata_path = paths.output_dir / "result_analysis_metadata.json"
    non_metadata_names = [*tables, report_path.name]
    metadata = build_analysis_metadata(
        paths=paths,
        inputs=inputs,
        paysim=paysim,
        output_names=non_metadata_names,
    )
    atomic_write_json(metadata, metadata_path)
    write_checksum_manifest(paths.output_dir)
    return {
        "status": "PASS — IEEE-CIS cross-dataset directional external validation frozen",
        "model_fit_count": 0,
        "prediction_generation_count": 0,
        "paired_effect_rows": len(paired),
        "output_dir": str(paths.output_dir),
    }


def _load_and_verify_inputs(paths: AnalysisPaths) -> dict[str, Any]:
    required_model_files = (
        "experiment_config.json",
        "run_metadata.json",
        "seed_metrics.csv",
        "fold_metrics.csv",
        "business_cost_results.csv",
        "prediction_scores.parquet",
        "IEEE_CIS_lightgbm_report.md",
    )
    for name in required_model_files:
        _validate_file(paths.model_results_dir / name, name)
    for path, name in (
        (paths.temporal_protocol_path, "temporal protocol"),
        (paths.history_metadata_path, "history metadata"),
        (paths.paysim_pair_status_path, "PaySim pair status"),
    ):
        _validate_file(path, name)
    config_path = paths.model_results_dir / "experiment_config.json"
    run_metadata_path = paths.model_results_dir / "run_metadata.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    config_hash = file_hash(config_path)
    if config_hash != run_metadata["config_hash"]:
        raise RuntimeError("Frozen experiment config hash mismatch")
    for name, expected_hash in run_metadata["output_hashes"].items():
        path = paths.model_results_dir / name
        if file_hash(path) != expected_hash:
            raise RuntimeError(f"Frozen output hash mismatch: {name}")
    if file_hash(paths.temporal_protocol_path) != config["inputs"]["protocol_sha256"]:
        raise RuntimeError("Frozen temporal protocol hash mismatch")
    if file_hash(paths.history_metadata_path) != config["inputs"]["history_metadata_sha256"]:
        raise RuntimeError("Frozen history metadata hash mismatch")
    return {
        "config": config,
        "run_metadata": run_metadata,
        "seed_metrics": pd.read_csv(paths.model_results_dir / "seed_metrics.csv"),
        "config_hash": config_hash,
        "ledger_hash": file_hash(paths.model_results_dir / "prediction_scores.parquet"),
    }


def audit_frozen_result_structure(
    config: dict[str, Any],
    run_metadata: dict[str, Any],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Verify the frozen task grid, roles, feature hashes, and execution restrictions."""
    rows: list[dict[str, Any]] = []
    rows.append(
        _audit(
            "official_fit_count_is_60", run_metadata["actual_official_fit_count"] == 60, "actual=60"
        )
    )
    rows.append(_audit("seed_metric_row_count_is_60", len(metrics) == 60, f"rows={len(metrics)}"))
    key_unique = not metrics[["fold", "model_id", "seed"]].duplicated().any()
    rows.append(_audit("task_keys_unique", key_unique, "Fold/Model/Seed keys"))
    rows.append(
        _audit(
            "folds_frozen", set(metrics["fold"]) == {1, 2, 3}, str(sorted(metrics["fold"].unique()))
        )
    )
    rows.append(
        _audit(
            "models_frozen",
            set(metrics["model_id"]) == set(MODEL_ORDER),
            str(sorted(metrics["model_id"].unique())),
        )
    )
    rows.append(
        _audit(
            "seeds_frozen",
            set(metrics["seed"]) == set(SEEDS),
            str(sorted(metrics["seed"].unique())),
        )
    )
    fold_model_counts = metrics.groupby(["fold", "model_id"])["seed"].nunique()
    rows.append(
        _audit(
            "five_seeds_per_fold_model",
            bool((fold_model_counts == 5).all()),
            f"groups={len(fold_model_counts)}",
        )
    )
    fold_seed_counts = metrics.groupby(["fold", "seed"])["model_id"].nunique()
    rows.append(
        _audit(
            "four_models_per_fold_seed",
            bool((fold_seed_counts == 4).all()),
            f"groups={len(fold_seed_counts)}",
        )
    )
    schemes = {row["model_id"]: row for row in config["schemes"]}
    rows.append(
        _audit(
            "m2_role_primary",
            schemes["M2"]["role"] == "pre-specified primary framework",
            schemes["M2"]["role"],
        )
    )
    rows.append(
        _audit(
            "m3_role_sensitivity",
            schemes["M3"]["role"] == "entity-granularity sensitivity analysis",
            schemes["M3"]["role"],
        )
    )
    rows.append(
        _audit(
            "m1_m2_feature_hash_equal",
            metrics.loc[metrics["model_id"] == "M1", "feature_set_hash"].unique().tolist()
            == metrics.loc[metrics["model_id"] == "M2", "feature_set_hash"].unique().tolist(),
            "Only class weight changes",
        )
    )
    restrictions = run_metadata["restrictions"]
    rows.append(
        _audit(
            "restricted_execution_counts_zero",
            all(value in (0, False) for value in restrictions.values()),
            json.dumps(restrictions, sort_keys=True),
        )
    )
    if not all(row["passed"] for row in rows):
        failed = [row["check"] for row in rows if not row["passed"]]
        raise RuntimeError(f"Frozen result structure audit failed: {failed}")
    return pd.DataFrame(rows)


def recompute_prediction_ledger(
    ledger_path: Path,
    seed_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Independently recompute all 60 Test metrics and verify ledger alignment."""
    parquet = pq.ParquetFile(ledger_path)
    expected = seed_metrics.set_index(["fold", "model_id", "seed"])
    protocol_folds = {int(row["fold"]): row for row in config["fold_definitions"]}
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    alignment: dict[tuple[int, int], tuple[str, str, str]] = {}
    seen: set[tuple[int, str, int]] = set()
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(row_group).to_pandas()
        keys = frame[["fold", "model_id", "seed"]].drop_duplicates()
        if len(keys) != 1:
            raise RuntimeError(f"Prediction row group {row_group} mixes task keys")
        key_row = keys.iloc[0]
        key = (int(key_row["fold"]), str(key_row["model_id"]), int(key_row["seed"]))
        if key in seen:
            raise RuntimeError(f"Duplicate Prediction Ledger task: {key}")
        seen.add(key)
        expected_row = expected.loc[key]
        ids = frame["TransactionID"].to_numpy(dtype=np.int64)
        times = frame["TransactionDT"].to_numpy(dtype=np.int64)
        labels = frame["isFraud"].to_numpy(dtype=np.int8)
        probabilities = frame["raw_probability"].to_numpy(dtype=np.float64)
        if len(np.unique(ids)) != len(ids):
            raise RuntimeError(f"Duplicate Test TransactionID in {key}")
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"Missing or invalid raw_probability in {key}")
        test_definition = protocol_folds[key[0]]["test"]
        time_valid = bool(
            times.min() == test_definition["start_transaction_dt"]
            and times.max() == test_definition["end_transaction_dt"]
        )
        if not time_valid or len(frame) != int(test_definition["transaction_count"]):
            raise RuntimeError(f"Prediction Ledger contains non-Test rows for {key}")
        alignment_key = (key[0], key[2])
        signatures = (array_hash(ids), array_hash(times), array_hash(labels))
        if alignment_key in alignment and alignment[alignment_key] != signatures:
            raise RuntimeError(f"M0-M3 Test alignment failed for Fold/Seed {alignment_key}")
        alignment[alignment_key] = signatures
        decision = fixed_capacity_decision(ids, labels, probabilities)
        recomputed = {
            "pr_auc": float(average_precision_score(labels, probabilities)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "precision_at_3pct": decision.precision_at_3pct,
            "recall_at_3pct": decision.recall_at_3pct,
            "f1_at_3pct": decision.f1_at_3pct,
            "alert_rate": decision.alert_rate,
            "capacity_deviation": decision.capacity_deviation,
            "business_cost": decision.business_cost,
            "baseline_cost": decision.baseline_cost,
            "cost_saving_rate": decision.cost_saving_rate,
        }
        continuous_differences = {
            name: abs(float(expected_row[name]) - value) for name, value in recomputed.items()
        }
        integer_matches = all(
            int(expected_row[name]) == int(getattr(decision, name))
            for name in ("tp", "fp", "tn", "fn", "alert_count", "fraud_captured")
        )
        selection_match = np.array_equal(
            frame["selected_at_3pct"].to_numpy(dtype=bool), decision.selected
        )
        rank_match = np.array_equal(
            frame["score_rank"].to_numpy(dtype=np.int64), decision.score_rank
        )
        probability_hash_match = str(expected_row["raw_probability_hash"]) == array_hash(
            probabilities
        )
        passed = bool(
            max(continuous_differences.values()) <= 1e-12
            and integer_matches
            and selection_match
            and rank_match
            and probability_hash_match
        )
        rows.append(
            {
                "fold": key[0],
                "model_id": key[1],
                "seed": key[2],
                "test_row_count": len(frame),
                "max_continuous_absolute_difference": max(continuous_differences.values()),
                "integer_metrics_exact": integer_matches,
                "selection_exact": selection_match,
                "rank_exact": rank_match,
                "probability_hash_exact": probability_hash_match,
                "passed": passed,
            }
        )
        profiles.append(
            {
                "fold": key[0],
                "model_id": key[1],
                "seed": key[2],
                "test_rows": len(frame),
                "fraud_count": int(labels.sum()),
                "alert_count": decision.alert_count,
            }
        )
    if len(seen) != 60 or parquet.metadata.num_rows != int(seed_metrics["test_count"].sum()):
        raise RuntimeError("Prediction Ledger task or row count is incomplete")
    return (
        pd.DataFrame(rows).sort_values(["fold", "model_id", "seed"]).reset_index(drop=True),
        pd.DataFrame(profiles),
    )


def make_paired_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    """Construct the four exact same-Fold/same-Seed pre-registered comparisons."""
    indexed = metrics.set_index(["fold", "seed", "model_id"])
    if not indexed.index.is_unique:
        raise RuntimeError("Frozen metric Fold/Seed/Model keys are not unique")
    rows: list[dict[str, Any]] = []
    for comparison_id, baseline_model, comparison_model, question in COMPARISONS:
        for fold in (1, 2, 3):
            for seed in SEEDS:
                baseline = indexed.loc[(fold, seed, baseline_model)]
                candidate = indexed.loc[(fold, seed, comparison_model)]
                row: dict[str, Any] = {
                    "comparison_id": comparison_id,
                    "comparison_question": question,
                    "baseline_model": baseline_model,
                    "comparison_model": comparison_model,
                    "fold": fold,
                    "seed": seed,
                }
                for metric in (
                    "pr_auc",
                    "roc_auc",
                    "precision_at_3pct",
                    "recall_at_3pct",
                    "f1_at_3pct",
                    "tp",
                    "cost_saving_rate",
                ):
                    row[f"baseline_{metric}"] = baseline[metric]
                    row[f"comparison_{metric}"] = candidate[metric]
                    row[f"delta_{metric}"] = candidate[metric] - baseline[metric]
                row["baseline_business_cost"] = baseline["business_cost"]
                row["comparison_business_cost"] = candidate["business_cost"]
                row["cost_reduction"] = baseline["business_cost"] - candidate["business_cost"]
                row["alert_count_equal"] = int(baseline["alert_count"]) == int(
                    candidate["alert_count"]
                )
                rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 60 or not bool(result["alert_count_equal"].all()):
        raise RuntimeError("Paired effect grid or fixed-capacity alert alignment failed")
    return result


def make_fixed_capacity_dependency_audit(
    paired: pd.DataFrame,
    seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Verify Cost Reduction = 101*DeltaTP and the fixed-capacity assumptions."""
    fraud_counts = seed_metrics.groupby("fold")["test_count"].nunique()
    if not (fraud_counts == 1).all():
        raise RuntimeError("Test count changed within a Fold")
    rows: list[dict[str, Any]] = []
    for row in paired.itertuples(index=False):
        expected_reduction = (FALSE_NEGATIVE_COST + FALSE_POSITIVE_COST) * row.delta_tp
        difference = float(row.cost_reduction - expected_reduction)
        rows.append(
            {
                "comparison_id": row.comparison_id,
                "fold": row.fold,
                "seed": row.seed,
                "alert_count_equal": row.alert_count_equal,
                "delta_tp": row.delta_tp,
                "observed_cost_reduction": row.cost_reduction,
                "expected_cost_reduction_101_delta_tp": expected_reduction,
                "absolute_difference": abs(difference),
                "dependency_statement": ("Business Cost = AlertCount + 100*FraudCount - 101*TP"),
                "passed": bool(row.alert_count_equal and abs(difference) <= 1e-12),
            }
        )
    return pd.DataFrame(rows)


def make_fold_effect_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Aggregate five stochastic Seeds separately within each frozen Fold."""
    rows: list[dict[str, Any]] = []
    for (comparison_id, fold), group in paired.groupby(["comparison_id", "fold"], sort=True):
        row: dict[str, Any] = {"comparison_id": comparison_id, "fold": fold}
        for metric in EFFECT_METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_seed_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            row[f"{metric}_positive_seed_count"] = int((values > 0).sum())
            row[f"{metric}_exact_zero_seed_count"] = int((values == 0).sum())
            row[f"{metric}_negative_seed_count"] = int((values < 0).sum())
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 12:
        raise RuntimeError(f"Expected 12 comparison/Fold summaries, found {len(result)}")
    return result


def make_fold_equal_weight_summary(fold_summary: pd.DataFrame) -> pd.DataFrame:
    """Average the three five-Seed Fold means with equal Fold weight."""
    rows: list[dict[str, Any]] = []
    for comparison_id, group in fold_summary.groupby("comparison_id", sort=True):
        indexed = group.set_index("fold")
        for metric in EFFECT_METRICS:
            values = np.asarray(
                [indexed.loc[fold, f"{metric}_mean"] for fold in (1, 2, 3)],
                dtype=float,
            )
            signs = [_strict_sign(value) for value in values]
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "metric": metric,
                    "fold1_effect": values[0],
                    "fold2_effect": values[1],
                    "fold3_effect": values[2],
                    "equal_weight_overall_effect": float(values.mean()),
                    "strict_fold_signs": "/".join(signs),
                    "positive_fold_count": signs.count("positive"),
                    "zero_fold_count": signs.count("zero"),
                    "negative_fold_count": signs.count("negative"),
                }
            )
    return pd.DataFrame(rows)


def make_seed_stability_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize stochastic variability without selecting a best Seed."""
    rows: list[dict[str, Any]] = []
    for (fold, model_id), group in metrics.groupby(["fold", "model_id"], sort=True):
        row: dict[str, Any] = {"fold": fold, "model_id": model_id}
        for metric in STABILITY_METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            row[f"{metric}_range"] = float(values.max() - values.min())
        rows.append(row)
    return pd.DataFrame(rows)


def make_seed_temporal_variability(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare within-Fold Seed spread and between-Fold means per model/metric."""
    rows: list[dict[str, Any]] = []
    for model_id, model_rows in metrics.groupby("model_id", sort=True):
        for metric in STABILITY_METRICS:
            fold_groups = model_rows.groupby("fold")[metric]
            fold_means = fold_groups.mean().sort_index().to_numpy(dtype=float)
            fold_stds = fold_groups.std(ddof=1).sort_index().to_numpy(dtype=float)
            mean_seed_std = float(fold_stds.mean())
            between_range = float(fold_means.max() - fold_means.min())
            rows.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "fold1_mean": fold_means[0],
                    "fold2_mean": fold_means[1],
                    "fold3_mean": fold_means[2],
                    "fold1_seed_std": fold_stds[0],
                    "fold2_seed_std": fold_stds[1],
                    "fold3_seed_std": fold_stds[2],
                    "mean_within_fold_seed_std": mean_seed_std,
                    "between_fold_range": between_range,
                    "between_fold_std": float(fold_means.std(ddof=1)),
                    "between_range_to_mean_seed_std": (
                        between_range / mean_seed_std if mean_seed_std > 0 else np.nan
                    ),
                    "interpretation_boundary": (
                        "descriptive scale comparison; not a variance ratio test"
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_temporal_heterogeneity(
    fold_summary: pd.DataFrame,
    temporal_audit_dir: Path,
) -> pd.DataFrame:
    """Join Fold effects to frozen future-window composition descriptively."""
    distribution = pd.read_csv(temporal_audit_dir / "temporal_distribution_audit.csv")
    availability = pd.read_csv(temporal_audit_dir / "entity_history_availability_by_fold.csv")
    cold_start = pd.read_csv(temporal_audit_dir / "cold_start_audit.csv")
    test_distribution = distribution.loc[distribution["split"] == "test"].set_index("fold")
    history = availability.loc[availability["split"] == "test"].pivot(
        index="fold", columns="entity", values="has_history_ratio"
    )
    cold = cold_start.loc[cold_start["split"] == "test"].pivot(
        index="fold", columns="entity", values="first_seen_entity_transaction_ratio"
    )
    rows: list[dict[str, Any]] = []
    for effect in fold_summary.itertuples(index=False):
        dist = test_distribution.loc[effect.fold]
        rows.append(
            {
                "comparison_id": effect.comparison_id,
                "fold": effect.fold,
                "delta_pr_auc": effect.delta_pr_auc_mean,
                "delta_roc_auc": effect.delta_roc_auc_mean,
                "delta_tp": effect.delta_tp_mean,
                "delta_recall_at_3pct": effect.delta_recall_at_3pct_mean,
                "cost_reduction": effect.cost_reduction_mean,
                "delta_cost_saving_rate": effect.delta_cost_saving_rate_mean,
                "test_fraud_rate": dist["fraud_rate"],
                "entity_a_history_availability": history.loc[effect.fold, "Entity A"],
                "entity_c_history_availability": history.loc[effect.fold, "Entity C"],
                "entity_a_cold_start_rate": cold.loc[effect.fold, "Entity A"],
                "entity_c_cold_start_rate": cold.loc[effect.fold, "Entity C"],
                "transaction_amt_mean": dist["transaction_amt_mean"],
                "transaction_amt_median": dist["transaction_amt_median"],
                "transaction_amt_p90": dist["transaction_amt_p90"],
                "product_cd_composition": dist["product_cd_distribution"],
                "interpretation_boundary": (
                    "descriptive coexistence only; no correlation or causal attribution"
                ),
            }
        )
    return pd.DataFrame(rows)


def make_ranking_decision_alignment(fold_summary: pd.DataFrame) -> pd.DataFrame:
    """Preserve continuous effects and classify strict ranking/TP directions."""
    rows: list[dict[str, Any]] = []
    for row in fold_summary.itertuples(index=False):
        ranking_sign = _strict_sign(row.delta_pr_auc_mean)
        decision_sign = _strict_sign(row.delta_tp_mean)
        rows.append(
            {
                "comparison_id": row.comparison_id,
                "fold": row.fold,
                "delta_pr_auc": row.delta_pr_auc_mean,
                "delta_roc_auc": row.delta_roc_auc_mean,
                "delta_tp": row.delta_tp_mean,
                "delta_recall_at_3pct": row.delta_recall_at_3pct_mean,
                "cost_reduction": row.cost_reduction_mean,
                "strict_pr_auc_sign": ranking_sign,
                "strict_decision_sign": decision_sign,
                "alignment_type": _alignment_type(ranking_sign, decision_sign),
                "interpretation": _alignment_interpretation(ranking_sign, decision_sign),
            }
        )
    return pd.DataFrame(rows)


def load_paysim_evidence(
    paysim_results_dir: Path,
    pair_status_path: Path,
) -> dict[str, Any]:
    """Load and derive PaySim effects from actual frozen result files."""
    result_path = paysim_results_dir / "rolling_multiseed_results.csv"
    pairwise_path = paysim_results_dir / "rolling_multiseed_pairwise.csv"
    report_path = paysim_results_dir / "rolling_multiseed_report.md"
    for path in (result_path, pairwise_path, report_path):
        _validate_file(path, f"PaySim evidence {path.name}")
    results = pd.read_csv(result_path)
    pairwise = pd.read_csv(pairwise_path)
    pair_status = json.loads(pair_status_path.read_text(encoding="utf-8"))
    if len(results) != 60 or len(pairwise) != 15:
        raise RuntimeError("Frozen PaySim multiseed evidence grid is incomplete")
    schemes = {
        "M0_like": "raw_safe",
        "M1_like": "recipient_full",
        "M2_like": "recipient_full_weight_2",
    }
    comparisons = {
        "C01": (schemes["M0_like"], schemes["M1_like"]),
        "C12": (schemes["M1_like"], schemes["M2_like"]),
        "C02": (schemes["M0_like"], schemes["M2_like"]),
    }
    effects: dict[str, pd.DataFrame] = {}
    indexed = results.set_index(["fold", "seed", "model_id"])
    if not indexed.index.is_unique:
        raise RuntimeError("Frozen PaySim Fold/Seed/Model keys are not unique")
    for comparison_id, (baseline, candidate) in comparisons.items():
        rows = []
        for fold in (1, 2, 3):
            for seed in SEEDS:
                left = indexed.loc[(fold, seed, baseline)]
                right = indexed.loc[(fold, seed, candidate)]
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "delta_pr_auc": right["pr_auc"] - left["pr_auc"],
                        "delta_recall": right["recall"] - left["recall"],
                        "cost_reduction": left["business_cost"] - right["business_cost"],
                    }
                )
        effects[comparison_id] = pd.DataFrame(rows)
    return {
        "results": results,
        "pairwise": pairwise,
        "pair_status": pair_status,
        "effects": effects,
        "source_paths": [result_path, pairwise_path, report_path, pair_status_path],
        "source_hashes": {
            str(path.resolve()): file_hash(path)
            for path in (result_path, pairwise_path, report_path, pair_status_path)
        },
        "decision_protocol_difference": (
            "PaySim used validation-selected thresholds under a 3% maximum capacity; "
            "IEEE-CIS used exact Test top-3% fixed-capacity ranking."
        ),
    }


def make_external_validation_matrix(
    *,
    fold_summary: pd.DataFrame,
    paired_effects: pd.DataFrame,
    paysim: dict[str, Any],
) -> pd.DataFrame:
    """Classify eight neutral Findings from derived PaySim and IEEE-CIS evidence."""
    ieee = _ieee_effect_arrays(fold_summary)
    pay = {
        comparison: _paysim_effect_arrays(frame) for comparison, frame in paysim["effects"].items()
    }
    rows: list[dict[str, Any]] = []

    def add(
        finding_id: str,
        finding: str,
        pay_values: Sequence[float] | None,
        ieee_values: Sequence[float] | None,
        classification: str,
        interpretation: str,
    ) -> None:
        rows.append(
            {
                "finding_id": finding_id,
                "neutral_finding": finding,
                "paysim_fold_effects": _json_array(pay_values),
                "paysim_strict_signs": _sign_string(pay_values),
                "ieee_cis_fold_effects": _json_array(ieee_values),
                "ieee_cis_strict_signs": _sign_string(ieee_values),
                "replication_classification": classification,
                "interpretation": interpretation,
            }
        )

    pay_c01_pr = pay["C01"]["pr_auc"]
    ieee_c01_pr = ieee["C01"]["pr_auc"]
    add(
        "EV1",
        "Effect of historical behavioral features on global PR-AUC",
        pay_c01_pr,
        ieee_c01_pr,
        _directional_classification(pay_c01_pr, ieee_c01_pr),
        "Both datasets are evaluated by Fold direction; no absolute PR-AUC comparison.",
    )
    pay_c01_recall = pay["C01"]["recall"]
    ieee_c01_recall = ieee["C01"]["recall"]
    add(
        "EV2",
        "Effect of historical behavioral features on constrained fraud capture",
        pay_c01_recall,
        ieee_c01_recall,
        _directional_classification(pay_c01_recall, ieee_c01_recall),
        "PaySim threshold decisions and IEEE-CIS exact top-3% decisions are "
        "conceptually, not exactly, corresponding.",
    )
    pay_c12_recall = pay["C12"]["recall"]
    ieee_c12_recall = ieee["C12"]["recall"]
    add(
        "EV3",
        "Effect of class weighting on constrained operational capture",
        pay_c12_recall,
        ieee_c12_recall,
        _directional_classification(pay_c12_recall, ieee_c12_recall),
        "Class weighting produced time-dependent capture effects rather than a "
        "universal direction.",
    )
    pay_c02_cost = pay["C02"]["cost_reduction"]
    ieee_c02_cost = ieee["C02"]["cost_reduction"]
    add(
        "EV4",
        "Effect of the pre-specified complete framework on capacity-constrained business cost",
        pay_c02_cost,
        ieee_c02_cost,
        _directional_classification(pay_c02_cost, ieee_c02_cost),
        "Cost directions are compared; absolute costs are not comparable across datasets.",
    )
    pay_misaligned = _misalignment_count(pay["C02"]["pr_auc"], pay["C02"]["recall"])
    ieee_misaligned = _misalignment_count(ieee["C02"]["pr_auc"], ieee["C02"]["recall"])
    add(
        "EV5",
        "Relationship between global ranking quality and constrained decision utility",
        [float(pay_misaligned)],
        [float(ieee_misaligned)],
        "Replicated" if pay_misaligned > 0 and ieee_misaligned > 0 else "Not Replicated",
        "Both datasets contain Fold-level ranking/capture direction mismatch; "
        "PR-AUC improvement was not necessary for capture improvement.",
    )
    add(
        "EV6",
        "Temporal stability of estimated framework effects",
        pay_c02_cost,
        ieee_c02_cost,
        (
            "Replicated"
            if len(set(_strict_sign(value) for value in pay_c02_cost)) == 1
            and len(set(_strict_sign(value) for value in ieee_c02_cost)) == 1
            and _strict_sign(pay_c02_cost[0]) == _strict_sign(ieee_c02_cost[0])
            else "Dataset-Specific / Time-Dependent"
        ),
        "PaySim and IEEE-CIS temporal direction patterns are compared without "
        "treating Folds as independent datasets.",
    )
    add(
        "EV7",
        "Sensitivity of results to entity granularity",
        None,
        ieee["C23"]["cost_reduction"],
        "Not Assessable Cross-Dataset",
        "PaySim Pair features were frozen as degenerate, so no valid "
        "entity-granularity counterpart exists.",
    )
    pay_ratio = _paired_temporal_seed_ratio(paysim["effects"]["C02"], "cost_reduction")
    ieee_c02 = paired_effects.loc[paired_effects["comparison_id"] == "C02"]
    ieee_ratio = _paired_temporal_seed_ratio(ieee_c02, "cost_reduction")
    add(
        "EV8",
        "Relative magnitude of temporal variation and seed-level stochastic variability",
        [pay_ratio],
        [ieee_ratio],
        "Replicated" if (pay_ratio > 1) == (ieee_ratio > 1) else "Partially Replicated",
        "Ratios are descriptive same-metric scale comparisons, not variance-ratio tests.",
    )
    return pd.DataFrame(rows)


def build_analysis_metadata(
    *,
    paths: AnalysisPaths,
    inputs: dict[str, Any],
    paysim: dict[str, Any],
    output_names: Sequence[str],
) -> dict[str, Any]:
    """Build metadata that deliberately excludes its own and checksum-manifest hashes."""
    output_hashes = {name: file_hash(paths.output_dir / name) for name in output_names}
    return {
        "analysis": "IEEE-CIS Cross-Dataset Directional External Validation",
        "status": "FROZEN",
        "creation_time_utc": _reproducible_creation_time(
            paths.model_results_dir / "experiment_config.json",
            paths.model_results_dir / "prediction_scores.parquet",
            *paysim["source_paths"],
        ),
        "inputs": {
            "ieee_experiment_config_path": str(
                (paths.model_results_dir / "experiment_config.json").resolve()
            ),
            "ieee_experiment_config_sha256": inputs["config_hash"],
            "ieee_prediction_ledger_path": str(
                (paths.model_results_dir / "prediction_scores.parquet").resolve()
            ),
            "ieee_prediction_ledger_sha256": inputs["ledger_hash"],
            "temporal_protocol_path": str(paths.temporal_protocol_path.resolve()),
            "temporal_protocol_sha256": file_hash(paths.temporal_protocol_path),
            "history_metadata_path": str(paths.history_metadata_path.resolve()),
            "history_metadata_sha256": file_hash(paths.history_metadata_path),
            "paysim_source_hashes": paysim["source_hashes"],
        },
        "comparisons": [
            {
                "comparison_id": comparison,
                "baseline": baseline,
                "comparison": candidate,
                "question": question,
            }
            for comparison, baseline, candidate, question in COMPARISONS
        ],
        "fixed_capacity_dependency": (
            "Within an IEEE-CIS Fold: Business Cost = AlertCount + 100*FraudCount - 101*TP; "
            "therefore Cost Reduction = 101*DeltaTP when Alert Count is equal."
        ),
        "aggregation_rule": (
            "Mean paired effect across five Seeds within each Fold, then equal mean "
            "of three Fold means"
        ),
        "replication_classifications": [
            "Replicated",
            "Partially Replicated",
            "Not Replicated",
            "Dataset-Specific / Time-Dependent",
            "Not Assessable Cross-Dataset",
        ],
        "classification_precedence": (
            "When mixed temporal directions prevent a uniform portability statement, use "
            "Dataset-Specific / Time-Dependent; use Not Assessable only when no valid "
            "counterpart exists."
        ),
        "paysim_decision_protocol_difference": paysim["decision_protocol_difference"],
        "analysis_code_path": str(Path(__file__).resolve()),
        "analysis_code_sha256": file_hash(Path(__file__)),
        "output_hashes_excluding_metadata_and_checksum_manifest": output_hashes,
        "execution_counts": {
            "model_fit_count": 0,
            "prediction_generation_count": 0,
            "hyperparameter_tuning_count": 0,
            "feature_selection_count": 0,
            "shap_count": 0,
            "calibration_count": 0,
            "bootstrap_count": 0,
        },
        "interpretation_boundaries": {
            "statistical_significance_claimed": False,
            "post_hoc_neutral_threshold": False,
            "folds_treated_as_independent_datasets": False,
            "seeds_treated_as_independent_future_environments": False,
            "overall_external_validation_pass_fail": False,
        },
    }


def write_external_validation_report(
    path: Path,
    *,
    seed_metrics: pd.DataFrame,
    fold_summary: pd.DataFrame,
    overall: pd.DataFrame,
    stability: pd.DataFrame,
    variability: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    alignment: pd.DataFrame,
    external: pd.DataFrame,
    ledger_profiles: pd.DataFrame,
    paysim: dict[str, Any],
) -> None:
    """Write the frozen, descriptive, finding-level external validation report."""
    c02 = fold_summary.loc[
        fold_summary["comparison_id"] == "C02",
        [
            "fold",
            "delta_pr_auc_mean",
            "delta_roc_auc_mean",
            "delta_tp_mean",
            "delta_recall_at_3pct_mean",
            "cost_reduction_mean",
            "delta_cost_saving_rate_mean",
        ],
    ]
    c01 = fold_summary.loc[fold_summary["comparison_id"] == "C01"]
    c12 = fold_summary.loc[fold_summary["comparison_id"] == "C12"]
    c23 = fold_summary.loc[fold_summary["comparison_id"] == "C23"]
    primary = seed_metrics.loc[seed_metrics["model_id"].isin(["M0", "M2"])]
    baseline_view = primary.groupby(["fold", "model_id"], as_index=False).agg(
        pr_auc_mean=("pr_auc", "mean"),
        recall_at_3pct_mean=("recall_at_3pct", "mean"),
        business_cost_mean=("business_cost", "mean"),
        cost_saving_rate_mean=("cost_saving_rate", "mean"),
    )
    lines = [
        "# IEEE-CIS Cross-Dataset Directional External Validation Report",
        "",
        "## 1. Frozen experiment integrity",
        "",
        f"- Frozen official fits preserved: {len(seed_metrics)} task records.",
        f"- Test Prediction Ledger rows audited: {int(ledger_profiles['test_rows'].sum()):,}.",
        "- M2 remained the pre-specified primary framework; M3 remained the "
        "entity-granularity sensitivity scheme.",
        "- This analysis performed zero model fits and generated no new predictions.",
        "",
        "## 2. Metric recomputation audit",
        "",
        "All 60 PR-AUC, ROC-AUC, fixed-capacity confusion matrices, selection masks, ranking "
        "orders, probability hashes, Business Costs, and Cost Saving Rates were independently "
        "recomputed from the frozen Test ledger and matched within 1e-12 for continuous values.",
        "",
        "## 3. Fixed-capacity metric dependency",
        "",
        "Within the same IEEE-CIS Fold, FraudCount and AlertCount are fixed. Therefore:",
        "",
        "`Business Cost = AlertCount + 100 × FraudCount - 101 × TP`",
        "",
        "and `Cost Reduction = 101 × DeltaTP`. TP/Fraud Captured, Recall@3%, Business Cost, "
        "and Cost Saving Rate are related expressions of the same capacity-constrained capture "
        "effect, not four independent pieces of statistical evidence.",
        "",
        "## 4. M0 baseline",
        "",
        _markdown_table(baseline_view),
        "",
        "## 5. M0 vs M1 — history-only effect",
        "",
        _effect_table(c01),
        "",
        "The isolated Entity A history effect was not uniformly favorable across the three "
        "future windows.",
        "",
        "## 6. M1 vs M2 — class-weight contribution",
        "",
        _effect_table(c12),
        "",
        "Class-weighted training acted as a cost-sensitive training proxy; it was not itself a "
        "Business Cost optimization procedure, and its estimated effect varied by Fold.",
        "",
        "## 7. M2 vs M3 — entity granularity sensitivity",
        "",
        _effect_table(c23),
        "",
        "Entity C changed magnitudes but did not replace M2 as the pre-specified primary scheme. "
        "No valid PaySim entity-granularity counterpart exists because Pair features were frozen "
        "as degenerate.",
        "",
        "## 8. M0 vs M2 — pre-specified framework effect",
        "",
        _markdown_table(c02),
        "",
        "The equal-weight average cost difference was near zero, while Fold-level effects "
        "changed substantially in direction and magnitude. Opposing Fold effects therefore "
        "cancelled in the average rather than establishing a uniform benefit.",
        "",
        "## 9. Descriptive temporal heterogeneity",
        "",
        _markdown_table(
            heterogeneity.loc[heterogeneity["comparison_id"] == "C02"].drop(
                columns=["product_cd_composition", "interpretation_boundary"]
            )
        ),
        "",
        "The variation in estimated effects coincided with changes in prevalence, history "
        "availability, cold-start exposure, amount distribution, and ProductCD composition. "
        "With only three frozen windows, this is descriptive coexistence rather than correlation "
        "or causal attribution.",
        "",
        "## 10. Seed stability",
        "",
        _markdown_table(stability),
        "",
        "Five Seeds represent repeated stochastic fits on each same Test window, not independent "
        "future environments.",
        "",
        "## 11. Within-Fold Seed vs between-Fold temporal variation",
        "",
        _markdown_table(variability),
        "",
        "The ratios are descriptive, same-metric scale comparisons and are not variance-ratio "
        "tests, ICC estimates, or formal variance decompositions.",
        "",
        "## 12. Global ranking vs constrained capture",
        "",
        _markdown_table(alignment),
        "",
        "Global PR-AUC and fixed-capacity capture did not always move in the same direction. "
        "This is not contradictory because they evaluate different operational objectives.",
        "",
        "## 13. Cost interpretation",
        "",
        "The 100:1 cost maps fixed-capacity TP differences to a pre-specified business scale. "
        "It is not independent corroboration of Recall and does not represent institution-specific "
        "economic calibration.",
        "",
        "## 14. Cross-dataset directional external validation",
        "",
        paysim["decision_protocol_difference"],
        " PaySim and IEEE-CIS schemes are conceptually corresponding framework variants rather "
        "than identical interventions. Absolute PR-AUC and Business Cost values were not compared.",
        "",
        "## 15. Finding-level replication matrix",
        "",
        _markdown_table(external),
        "",
        "No overall external-validation PASS/FAIL label is assigned. Each neutral Finding is "
        "classified independently and preserves mixed or unfavorable evidence.",
        "",
        "## 16. Limitations",
        "",
        "IEEE-CIS entities and fields are anonymized; card1 is not asserted to be a real account. "
        "Feature spaces, history semantics, prevalence, transaction mechanisms, and temporal "
        "structures differ between synthetic PaySim and real-world-derived IEEE-CIS. The three "
        "rolling Folds are not independent datasets, and five Seeds are not independent future "
        "environments. This analysis is descriptive rather than inferential and assumes fixed "
        "3% IEEE-CIS capacity and a pre-specified 100:1 cost ratio.",
        "",
        "## 17. Next-stage recommendation",
        "",
        "IEEE-CIS Time-block Bootstrap is methodologically worthwhile because C02 changed "
        "direction across frozen windows and the equal-weight average masked Fold-specific harm "
        "and benefit. Its purpose would be to quantify within-window uncertainty, not to seek a "
        "preferred conclusion. Calibration is not currently necessary for the ranking/"
        "capacity question, and SHAP remains a separate future stage focused on pre-specified M2.",
        "",
        "**PASS — IEEE-CIS cross-dataset directional external validation frozen.**",
    ]
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def write_checksum_manifest(output_dir: Path) -> None:
    """Hash every final artifact except the self-referential manifest itself."""
    names = sorted(name for name in ANALYSIS_OUTPUTS if name != "checksums.sha256")
    lines = [f"{file_hash(output_dir / name)}  {name}" for name in names]
    atomic_write_bytes(output_dir / "checksums.sha256", ("\n".join(lines) + "\n").encode("utf-8"))


def _ieee_effect_arrays(fold_summary: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for comparison_id, group in fold_summary.groupby("comparison_id", sort=True):
        ordered = group.sort_values("fold")
        result[comparison_id] = {
            "pr_auc": ordered["delta_pr_auc_mean"].astype(float).tolist(),
            "recall": ordered["delta_recall_at_3pct_mean"].astype(float).tolist(),
            "cost_reduction": ordered["cost_reduction_mean"].astype(float).tolist(),
        }
    return result


def _paysim_effect_arrays(frame: pd.DataFrame) -> dict[str, list[float]]:
    grouped = frame.groupby("fold", sort=True).mean(numeric_only=True)
    return {
        "pr_auc": grouped["delta_pr_auc"].astype(float).tolist(),
        "recall": grouped["delta_recall"].astype(float).tolist(),
        "cost_reduction": grouped["cost_reduction"].astype(float).tolist(),
    }


def _directional_classification(
    paysim_values: Sequence[float], ieee_values: Sequence[float]
) -> str:
    pay_signs = [_strict_sign(value) for value in paysim_values]
    ieee_signs = [_strict_sign(value) for value in ieee_values]
    pay_nonzero = [sign for sign in pay_signs if sign != "zero"]
    ieee_nonzero = [sign for sign in ieee_signs if sign != "zero"]
    if not pay_nonzero or not ieee_nonzero:
        return "Dataset-Specific / Time-Dependent"
    pay_mixed = len(set(pay_nonzero)) > 1
    ieee_mixed = len(set(ieee_nonzero)) > 1
    pay_majority = _majority_sign(pay_nonzero)
    ieee_majority = _majority_sign(ieee_nonzero)
    if not pay_mixed and not ieee_mixed and pay_majority == ieee_majority:
        return "Replicated"
    if pay_mixed and ieee_mixed:
        return "Dataset-Specific / Time-Dependent"
    if pay_majority == ieee_majority:
        return "Partially Replicated"
    if not pay_mixed and not ieee_mixed:
        return "Not Replicated"
    return "Not Replicated" if not pay_mixed else "Dataset-Specific / Time-Dependent"


def _paired_temporal_seed_ratio(frame: pd.DataFrame, metric: str) -> float:
    grouped = frame.groupby("fold")[metric]
    fold_means = grouped.mean().to_numpy(dtype=float)
    mean_seed_std = float(grouped.std(ddof=1).mean())
    return float(np.ptp(fold_means) / mean_seed_std) if mean_seed_std > 0 else np.nan


def _misalignment_count(ranking_values: Sequence[float], decision_values: Sequence[float]) -> int:
    return sum(
        _strict_sign(ranking) != _strict_sign(decision)
        for ranking, decision in zip(ranking_values, decision_values, strict=True)
    )


def _alignment_type(ranking_sign: str, decision_sign: str) -> str:
    mapping = {
        ("positive", "positive"): "Type A",
        ("negative", "positive"): "Type B",
        ("positive", "negative"): "Type C",
        ("negative", "negative"): "Type D",
    }
    return (
        "Type E"
        if "zero" in (ranking_sign, decision_sign)
        else mapping.get((ranking_sign, decision_sign), "Type F")
    )


def _alignment_interpretation(ranking_sign: str, decision_sign: str) -> str:
    if ranking_sign == decision_sign:
        return "global ranking and fixed-capacity capture moved in the same strict direction"
    if "zero" in (ranking_sign, decision_sign):
        return "at least one layer had an exact-zero Fold mean effect"
    return "global ranking and fixed-capacity capture moved in different strict directions"


def _majority_sign(signs: Sequence[str]) -> str:
    positive = signs.count("positive")
    negative = signs.count("negative")
    if positive == negative:
        return "mixed"
    return "positive" if positive > negative else "negative"


def _strict_sign(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _sign_string(values: Sequence[float] | None) -> str:
    if values is None:
        return "not_available"
    return "/".join(_strict_sign(float(value)) for value in values)


def _json_array(values: Sequence[float] | None) -> str:
    if values is None:
        return "null"
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def _effect_table(frame: pd.DataFrame) -> str:
    columns = [
        "fold",
        "delta_pr_auc_mean",
        "delta_roc_auc_mean",
        "delta_tp_mean",
        "delta_recall_at_3pct_mean",
        "cost_reduction_mean",
        "delta_cost_saving_rate_mean",
    ]
    return _markdown_table(frame[columns])


def _audit(check: str, passed: bool, details: str) -> dict[str, Any]:
    return {"check": check, "passed": bool(passed), "details": details}


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(_format_value(value) for value in values) + " |")
    return "\n".join(rows)


def _format_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NaN"
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|")


def _reproducible_creation_time(*paths: Path) -> str:
    timestamp = max(path.stat().st_mtime for path in paths)
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


def _validate_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{name} is empty: {path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the zero-training result-analysis command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_results_dir", type=Path, required=True)
    parser.add_argument("--temporal_protocol_path", type=Path, required=True)
    parser.add_argument("--history_metadata_path", type=Path, required=True)
    parser.add_argument("--temporal_audit_dir", type=Path, required=True)
    parser.add_argument("--paysim_results_dir", type=Path, required=True)
    parser.add_argument("--paysim_pair_status_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def main() -> None:
    """Run deterministic analysis and print its zero-training completion summary."""
    args = build_parser().parse_args()
    result = run_result_analysis(
        AnalysisPaths(
            model_results_dir=args.model_results_dir.resolve(),
            temporal_protocol_path=args.temporal_protocol_path.resolve(),
            history_metadata_path=args.history_metadata_path.resolve(),
            temporal_audit_dir=args.temporal_audit_dir.resolve(),
            paysim_results_dir=args.paysim_results_dir.resolve(),
            paysim_pair_status_path=args.paysim_pair_status_path.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
