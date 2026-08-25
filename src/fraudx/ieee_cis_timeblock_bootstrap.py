"""Frozen paired moving time-block Bootstrap for IEEE-CIS C02 effects."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.metrics import average_precision_score

from fraudx.ieee_cis_experiment import (
    SEEDS,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    file_hash,
    fixed_capacity_decision,
)

N_BOOTSTRAP = 5000
BOOTSTRAP_MASTER_SEED = 20260822
CAPACITY = 0.03
FP_COST = 1
FN_COST = 100
BATCH_SIZE = 32
METRICS = (
    "delta_tp",
    "delta_recall",
    "cost_reduction",
    "delta_pr_auc",
    "delta_precision",
    "delta_f1",
)
OUTPUTS = (
    "ieee_cis_time_block_bootstrap_protocol.json",
    "bootstrap_fold_replicates.parquet",
    "bootstrap_seed_replicates.parquet",
    "bootstrap_fold_summary.csv",
    "bootstrap_overall_equal_weight_summary.csv",
    "bootstrap_integrity_audit.csv",
    "bootstrap_metadata.json",
    "IEEE_CIS_time_block_bootstrap_report.md",
    "checksums.sha256",
)


@dataclass(frozen=True)
class BootstrapPaths:
    """Frozen inputs and output directory."""

    model_results_dir: Path
    temporal_protocol_path: Path
    result_analysis_metadata_path: Path
    output_dir: Path


@dataclass(frozen=True)
class ModelView:
    """Frozen scores plus deterministic ranking and AP aggregation structures."""

    scores: np.ndarray
    rank_order: np.ndarray
    total_by_time_score: sparse.csr_matrix
    positive_by_time_score: sparse.csr_matrix


class FoldData(NamedTuple):
    """Aligned population and ten frozen M0/M2 model views for one Fold."""

    transaction_ids: np.ndarray
    transaction_dt: np.ndarray
    labels: np.ndarray
    time_positions: np.ndarray
    unique_times: np.ndarray
    group_sizes: np.ndarray
    fraud_by_time: np.ndarray
    models: dict[tuple[int, str], ModelView]
    observed_by_seed: dict[int, dict[str, float]]


def block_length_for_groups(group_count: int) -> int:
    """Return the frozen max(2, round(T**(1/3))) engineering heuristic."""
    if group_count < 2:
        raise ValueError("At least two unique TransactionDT groups are required")
    return max(2, int(round(group_count ** (1 / 3))))


def moving_block_positions(
    group_count: int,
    block_length: int,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw non-circular blocks and retain exactly T time-group positions."""
    if not 2 <= block_length <= group_count:
        raise ValueError("Block length must be within [2, group_count]")
    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive")
    blocks_per_replicate = math.ceil(group_count / block_length)
    legal_starts = group_count - block_length + 1
    starts = rng.integers(
        0,
        legal_starts,
        size=(repetitions, blocks_per_replicate),
        dtype=np.int32,
    )
    offsets = np.arange(block_length, dtype=np.int32)
    positions = (starts[..., None] + offsets).reshape(repetitions, -1)
    return positions[:, :group_count]


def position_multiplicities(positions: np.ndarray, group_count: int) -> np.ndarray:
    """Convert sampled group sequences to integer group multiplicities."""
    sampled = np.asarray(positions, dtype=np.int32)
    if sampled.ndim != 2 or sampled.shape[1] != group_count:
        raise ValueError("Every replicate must contain exactly group_count positions")
    if sampled.min() < 0 or sampled.max() >= group_count:
        raise ValueError("Non-circular sampled position is outside the Fold")
    result = np.zeros((len(sampled), group_count), dtype=np.int16)
    for index, row in enumerate(sampled):
        result[index] = np.bincount(row, minlength=group_count)
    return result


def occurrence_indices(
    sampled_positions: np.ndarray,
    group_transaction_ids: list[np.ndarray],
) -> pd.DataFrame:
    """Materialize deterministic duplicate occurrence order for audit-sized samples."""
    rows: list[dict[str, int]] = []
    occurrence = 0
    for sampled_group_order, position in enumerate(sampled_positions):
        for transaction_id in np.sort(group_transaction_ids[int(position)]):
            occurrence += 1
            rows.append(
                {
                    "sampled_group_order": sampled_group_order,
                    "TransactionID": int(transaction_id),
                    "bootstrap_occurrence_index": occurrence,
                }
            )
    return pd.DataFrame(rows)


def run_ieee_cis_timeblock_bootstrap(paths: BootstrapPaths) -> dict[str, Any]:
    """Verify frozen inputs, execute C02 Bootstrap, and freeze deterministic artifacts."""
    inputs = _verify_frozen_inputs(paths)
    metrics = pd.read_csv(paths.model_results_dir / "seed_metrics.csv")
    fold_child_seeds = _fold_child_seeds()
    seed_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    fold_details: list[dict[str, int]] = []
    original_audit_rows: list[dict[str, Any]] = []
    sample_hashes: dict[int, list[str]] = {}
    connection = duckdb.connect(database=":memory:")
    ledger_path = paths.model_results_dir / "prediction_scores.parquet"
    try:
        for fold in (1, 2, 3):
            data = load_fold_data(connection, ledger_path, metrics, fold, original_audit_rows)
            group_count = len(data.unique_times)
            block_length = block_length_for_groups(group_count)
            blocks_per_replicate = math.ceil(group_count / block_length)
            fold_details.append(
                {
                    "fold": fold,
                    "n_unique_transactiondt": group_count,
                    "block_length": block_length,
                    "blocks_per_replicate": blocks_per_replicate,
                    "legal_block_count": group_count - block_length + 1,
                }
            )
            print(
                f"Fold {fold}: T={group_count}, L={block_length}, "
                f"K={blocks_per_replicate}",
                flush=True,
            )
            seed_result, fold_result, hashes = _bootstrap_fold(
                fold=fold,
                data=data,
                child_seed=fold_child_seeds[fold],
                block_length=block_length,
            )
            seed_frames.append(seed_result)
            fold_frames.append(fold_result)
            sample_hashes[fold] = hashes
    finally:
        connection.close()

    seed_replicates = pd.concat(seed_frames, ignore_index=True)
    fold_replicates = pd.concat(fold_frames, ignore_index=True)
    _validate_replicate_results(seed_replicates, fold_replicates)
    deterministic_replay = _replay_sample_hashes(
        fold_details,
        fold_child_seeds,
        sample_hashes,
    )
    observed_fold = _observed_fold_effects(seed_replicates)
    fold_summary = summarize_fold_replicates(fold_replicates, observed_fold, fold_details)
    overall_replicates = make_overall_replicates(fold_replicates)
    overall_observed = observed_fold.groupby("metric")["observed_point_estimate"].mean()
    overall_summary = summarize_overall_replicates(overall_replicates, overall_observed)
    audit = build_integrity_audit(
        inputs=inputs,
        original_rows=original_audit_rows,
        seed_replicates=seed_replicates,
        fold_replicates=fold_replicates,
        deterministic_replay=deterministic_replay,
    )
    if not bool((audit["status"] == "PASS").all()):
        failed = audit.loc[audit["status"] != "PASS", "check"].tolist()
        raise RuntimeError(f"FAIL — IEEE-CIS Bootstrap Protocol Violation: {failed}")

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = paths.output_dir / "ieee_cis_time_block_bootstrap_protocol.json"
    protocol = build_protocol(paths, inputs, fold_details, fold_child_seeds)
    atomic_write_json(protocol, protocol_path)
    write_deterministic_parquet(
        fold_replicates,
        paths.output_dir / "bootstrap_fold_replicates.parquet",
    )
    write_deterministic_parquet(
        seed_replicates,
        paths.output_dir / "bootstrap_seed_replicates.parquet",
    )
    atomic_write_csv(fold_summary, paths.output_dir / "bootstrap_fold_summary.csv")
    atomic_write_csv(
        overall_summary,
        paths.output_dir / "bootstrap_overall_equal_weight_summary.csv",
    )
    atomic_write_csv(audit, paths.output_dir / "bootstrap_integrity_audit.csv")
    report_path = paths.output_dir / "IEEE_CIS_time_block_bootstrap_report.md"
    write_report(
        report_path,
        paths=paths,
        fold_details=fold_details,
        fold_summary=fold_summary,
        overall_summary=overall_summary,
        audit=audit,
    )
    metadata_path = paths.output_dir / "bootstrap_metadata.json"
    metadata = build_metadata(paths, inputs, protocol_path, report_path)
    atomic_write_json(metadata, metadata_path)
    write_checksums(paths.output_dir)
    return {
        "status": "PASS — IEEE-CIS paired time-block bootstrap frozen",
        "bootstrap_count_per_fold": N_BOOTSTRAP,
        "primary_replicate_count": len(fold_replicates),
        "model_fit_count": 0,
        "new_prediction_count": 0,
        "folds": fold_details,
        "output_dir": str(paths.output_dir),
    }


def _verify_frozen_inputs(paths: BootstrapPaths) -> dict[str, Any]:
    config_path = paths.model_results_dir / "experiment_config.json"
    run_metadata_path = paths.model_results_dir / "run_metadata.json"
    ledger_path = paths.model_results_dir / "prediction_scores.parquet"
    seed_metrics_path = paths.model_results_dir / "seed_metrics.csv"
    for path in (
        config_path,
        run_metadata_path,
        ledger_path,
        seed_metrics_path,
        paths.temporal_protocol_path,
        paths.result_analysis_metadata_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Frozen input is missing or empty: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    result_metadata = json.loads(paths.result_analysis_metadata_path.read_text(encoding="utf-8"))
    hashes = {
        "experiment_config.json": file_hash(config_path),
        "prediction_scores.parquet": file_hash(ledger_path),
        "seed_metrics.csv": file_hash(seed_metrics_path),
        "temporal_protocol.json": file_hash(paths.temporal_protocol_path),
        "result_analysis_metadata.json": file_hash(paths.result_analysis_metadata_path),
    }
    expected = {
        "experiment_config.json": run_metadata["config_hash"],
        "prediction_scores.parquet": run_metadata["output_hashes"]["prediction_scores.parquet"],
        "seed_metrics.csv": run_metadata["output_hashes"]["seed_metrics.csv"],
        "temporal_protocol.json": config["inputs"]["protocol_sha256"],
    }
    for name, expected_hash in expected.items():
        if hashes[name] != expected_hash:
            raise RuntimeError(f"FAIL — Frozen Input Hash Mismatch: {name}")
    if result_metadata["inputs"]["ieee_prediction_ledger_sha256"] != hashes[
        "prediction_scores.parquet"
    ]:
        raise RuntimeError("FAIL — Frozen Input Hash Mismatch: result analysis ledger")
    schemes = {row["model_id"]: row for row in config["schemes"]}
    if schemes["M0"]["scale_pos_weight"] != 1 or schemes["M2"]["scale_pos_weight"] != 2:
        raise RuntimeError("Frozen M0/M2 definitions changed")
    if schemes["M2"]["role"] != "pre-specified primary framework":
        raise RuntimeError("M2 frozen Primary role changed")
    return {
        "hashes": hashes,
        "expected_hashes": expected,
        "config": config,
        "run_metadata": run_metadata,
        "result_metadata": result_metadata,
    }


def _fold_child_seeds() -> dict[int, int]:
    children = np.random.SeedSequence(BOOTSTRAP_MASTER_SEED).spawn(3)
    return {
        fold: int(child.generate_state(1, dtype=np.uint64)[0])
        for fold, child in zip((1, 2, 3), children, strict=True)
    }


def load_fold_data(
    connection: duckdb.DuckDBPyConnection,
    ledger_path: Path,
    frozen_metrics: pd.DataFrame,
    fold: int,
    audit_rows: list[dict[str, Any]],
) -> FoldData:
    """Load aligned M0/M2 score vectors and verify every frozen original estimate."""
    models: dict[tuple[int, str], ModelView] = {}
    observed: dict[int, dict[str, float]] = {}
    reference: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    reference_time_positions: np.ndarray | None = None
    unique_times: np.ndarray | None = None
    group_sizes: np.ndarray | None = None
    fraud_by_time: np.ndarray | None = None
    for seed in SEEDS:
        frames: dict[str, pd.DataFrame] = {}
        for model_id in ("M0", "M2"):
            frame = connection.execute(
                "SELECT TransactionID, TransactionDT, isFraud, raw_probability "
                "FROM read_parquet(?) WHERE fold=? AND seed=? AND model_id=? "
                "ORDER BY TransactionID",
                [str(ledger_path), fold, seed, model_id],
            ).df()
            if len(frame) == 0:
                raise RuntimeError(f"Missing ledger rows for Fold {fold}, Seed {seed}, {model_id}")
            frames[model_id] = frame
        base = frames["M0"]
        candidate = frames["M2"]
        for column in ("TransactionID", "TransactionDT", "isFraud"):
            if not np.array_equal(base[column].to_numpy(), candidate[column].to_numpy()):
                raise RuntimeError(f"M0/M2 population mismatch: Fold {fold}, Seed {seed}")
        population = (
            base["TransactionID"].to_numpy(dtype=np.int64),
            base["TransactionDT"].to_numpy(dtype=np.int64),
            base["isFraud"].to_numpy(dtype=np.int8),
        )
        if reference is None:
            reference = population
            unique_times, reference_time_positions = np.unique(
                population[1], return_inverse=True
            )
            group_sizes = np.bincount(reference_time_positions).astype(np.int64)
            fraud_by_time = np.bincount(
                reference_time_positions,
                weights=population[2],
                minlength=len(unique_times),
            ).astype(np.int64)
        elif any(
            not np.array_equal(expected, actual)
            for expected, actual in zip(reference, population, strict=True)
        ):
            raise RuntimeError(f"Cross-seed Test population mismatch in Fold {fold}")
        assert reference_time_positions is not None
        assert unique_times is not None
        seed_observed: dict[str, dict[str, float]] = {}
        for model_id, frame in frames.items():
            scores = frame["raw_probability"].to_numpy(dtype=np.float64)
            models[(seed, model_id)] = build_model_view(
                population[0],
                population[2],
                scores,
                reference_time_positions,
                len(unique_times),
            )
            calculated = original_metrics(population[0], population[2], scores)
            expected_row = frozen_metrics.loc[
                (frozen_metrics["fold"] == fold)
                & (frozen_metrics["seed"] == seed)
                & (frozen_metrics["model_id"] == model_id)
            ].iloc[0]
            differences = {
                "pr_auc": abs(calculated["pr_auc"] - expected_row["pr_auc"]),
                "recall_at_3pct": abs(
                    calculated["recall_at_3pct"] - expected_row["recall_at_3pct"]
                ),
                "precision_at_3pct": abs(
                    calculated["precision_at_3pct"] - expected_row["precision_at_3pct"]
                ),
                "f1_at_3pct": abs(
                    calculated["f1_at_3pct"] - expected_row["f1_at_3pct"]
                ),
                "cost_saving_rate": abs(
                    calculated["cost_saving_rate"] - expected_row["cost_saving_rate"]
                ),
            }
            integer_pass = all(
                calculated[name] == int(expected_row[name])
                for name in ("tp", "fp", "fn", "business_cost", "alert_count")
            )
            passed = integer_pass and max(differences.values()) <= 1e-12
            audit_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "model_id": model_id,
                    "passed": passed,
                    "max_continuous_difference": max(differences.values()),
                    "integer_metrics_exact": integer_pass,
                }
            )
            if not passed:
                raise RuntimeError(
                    f"Frozen original estimate mismatch: Fold {fold}, Seed {seed}, {model_id}"
                )
            seed_observed[model_id] = calculated
        observed[seed] = effect_from_model_metrics(seed_observed["M0"], seed_observed["M2"])
    assert reference is not None
    assert reference_time_positions is not None
    assert unique_times is not None
    assert group_sizes is not None
    assert fraud_by_time is not None
    return FoldData(
        *reference,
        reference_time_positions,
        unique_times,
        group_sizes,
        fraud_by_time,
        models,
        observed,
    )


def build_model_view(
    transaction_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    time_positions: np.ndarray,
    group_count: int,
) -> ModelView:
    """Precompute exact tie-aware capacity order and score-group AP matrices."""
    rank_order = np.lexsort((transaction_ids, -scores)).astype(np.int32)
    unique_scores, score_groups = np.unique(scores, return_inverse=True)
    shape = (group_count, len(unique_scores))
    total = sparse.coo_matrix(
        (np.ones(len(scores)), (time_positions, score_groups)),
        shape=shape,
    ).tocsr()
    positive = labels == 1
    positive_matrix = sparse.coo_matrix(
        (
            np.ones(int(positive.sum())),
            (time_positions[positive], score_groups[positive]),
        ),
        shape=shape,
    ).tocsr()
    return ModelView(scores, rank_order, total, positive_matrix)


def original_metrics(ids: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Recompute the frozen unresampled decision and ranking metrics."""
    decision = fixed_capacity_decision(ids, labels, scores)
    fraud_count = int(labels.sum())
    baseline_cost = FN_COST * fraud_count
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "tp": float(decision.tp),
        "fp": float(decision.fp),
        "fn": float(decision.fn),
        "precision_at_3pct": decision.precision_at_3pct,
        "recall_at_3pct": decision.recall_at_3pct,
        "f1_at_3pct": decision.f1_at_3pct,
        "business_cost": float(decision.business_cost),
        "cost_saving_rate": (baseline_cost - decision.business_cost) / baseline_cost,
        "alert_count": float(decision.alert_count),
    }


def effect_from_model_metrics(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict[str, float]:
    """Return the frozen C02 direction conventions from two model metrics."""
    return {
        "delta_tp": candidate["tp"] - baseline["tp"],
        "delta_recall": candidate["recall_at_3pct"] - baseline["recall_at_3pct"],
        "cost_reduction": baseline["business_cost"] - candidate["business_cost"],
        "delta_pr_auc": candidate["pr_auc"] - baseline["pr_auc"],
        "delta_precision": (
            candidate["precision_at_3pct"] - baseline["precision_at_3pct"]
        ),
        "delta_f1": candidate["f1_at_3pct"] - baseline["f1_at_3pct"],
    }


def _bootstrap_fold(
    *,
    fold: int,
    data: FoldData,
    child_seed: int,
    block_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    group_count = len(data.unique_times)
    seed_arrays = {
        seed: {metric: np.full(N_BOOTSTRAP, np.nan) for metric in METRICS} for seed in SEEDS
    }
    transaction_counts = np.empty(N_BOOTSTRAP, dtype=np.int64)
    fraud_counts = np.empty(N_BOOTSTRAP, dtype=np.int64)
    alert_counts = np.empty(N_BOOTSTRAP, dtype=np.int64)
    hashes: list[str] = []
    rng = np.random.default_rng(child_seed)
    for start in range(0, N_BOOTSTRAP, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_BOOTSTRAP)
        positions = moving_block_positions(
            group_count,
            block_length,
            stop - start,
            rng,
        )
        multiplicities = position_multiplicities(positions, group_count)
        hashes.extend(_sample_hash(row) for row in positions)
        replicate_counts = multiplicities @ data.group_sizes
        replicate_fraud = multiplicities @ data.fraud_by_time
        alerts = np.ceil(CAPACITY * replicate_counts).astype(np.int64)
        transaction_counts[start:stop] = replicate_counts
        fraud_counts[start:stop] = replicate_fraud
        alert_counts[start:stop] = alerts
        for seed in SEEDS:
            baseline = model_metrics_for_replicates(
                data.models[(seed, "M0")],
                data.labels,
                data.time_positions,
                multiplicities,
                replicate_counts,
                replicate_fraud,
                alerts,
            )
            candidate = model_metrics_for_replicates(
                data.models[(seed, "M2")],
                data.labels,
                data.time_positions,
                multiplicities,
                replicate_counts,
                replicate_fraud,
                alerts,
            )
            effects = effect_arrays(baseline, candidate)
            for metric in METRICS:
                seed_arrays[seed][metric][start:stop] = effects[metric]
    seed_frames: list[pd.DataFrame] = []
    sample_hash_array = np.asarray(hashes)
    for seed in SEEDS:
        seed_frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "replicate_id": np.arange(1, N_BOOTSTRAP + 1),
                    "seed": seed,
                    "replicate_transaction_count": transaction_counts,
                    "replicate_fraud_count": fraud_counts,
                    "alert_count": alert_counts,
                    "sample_hash": sample_hash_array,
                    **{
                        f"observed_{metric}": data.observed_by_seed[seed][metric]
                        for metric in METRICS
                    },
                    **seed_arrays[seed],
                }
            )
        )
    seed_frame = pd.concat(seed_frames, ignore_index=True)
    grouped = seed_frame.groupby(["fold", "replicate_id"], sort=True)
    fold_frame = grouped[list(METRICS)].mean().reset_index()
    common = grouped[
        [
            "replicate_transaction_count",
            "replicate_fraud_count",
            "alert_count",
            "sample_hash",
        ]
    ].first().reset_index()
    fold_frame = fold_frame.merge(common, on=["fold", "replicate_id"], validate="one_to_one")
    fold_frame["replicate_time_group_count"] = group_count
    fold_frame["block_length"] = block_length
    return seed_frame, fold_frame, hashes


def model_metrics_for_replicates(
    model: ModelView,
    labels: np.ndarray,
    time_positions: np.ndarray,
    multiplicities: np.ndarray,
    transaction_counts: np.ndarray,
    fraud_counts: np.ndarray,
    alert_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reapply the frozen capacity rule and exact weighted AP to sampled populations."""
    row_weights = multiplicities[:, time_positions[model.rank_order]].astype(np.int64)
    ranked_labels = labels[model.rank_order]
    cumulative_rows = np.cumsum(row_weights, axis=1)
    threshold_indices = np.argmax(cumulative_rows >= alert_counts[:, None], axis=1)
    row_index = np.arange(len(multiplicities))
    before_count = np.where(
        threshold_indices > 0,
        cumulative_rows[row_index, np.maximum(threshold_indices - 1, 0)],
        0,
    )
    take_at_boundary = alert_counts - before_count
    cumulative_tp = np.cumsum(row_weights * ranked_labels, axis=1)
    before_tp = np.where(
        threshold_indices > 0,
        cumulative_tp[row_index, np.maximum(threshold_indices - 1, 0)],
        0,
    )
    tp = before_tp + take_at_boundary * ranked_labels[threshold_indices]
    fp = alert_counts - tp
    fn = fraud_counts - tp
    precision = _safe_divide(tp, alert_counts)
    recall = _safe_divide(tp, fraud_counts)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    cost = fp + FN_COST * fn
    pr_auc = weighted_average_precision(
        multiplicities,
        model.total_by_time_score,
        model.positive_by_time_score,
    )
    return {
        "tp": tp.astype(np.float64),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cost": cost.astype(np.float64),
        "pr_auc": pr_auc,
        "transaction_count": transaction_counts.astype(np.float64),
    }


def weighted_average_precision(
    multiplicities: np.ndarray,
    total_by_time_score: sparse.csr_matrix,
    positive_by_time_score: sparse.csr_matrix,
) -> np.ndarray:
    """Calculate sklearn-equivalent AP with integer time-group weights."""
    weights = multiplicities.astype(np.float64, copy=False)
    totals = np.asarray(weights @ total_by_time_score)[:, ::-1]
    positives = np.asarray(weights @ positive_by_time_score)[:, ::-1]
    cumulative_total = np.cumsum(totals, axis=1)
    cumulative_positive = np.cumsum(positives, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    positive_total = positives.sum(axis=1)
    result: np.ndarray = np.divide(
        np.sum(positives * precision, axis=1),
        positive_total,
        out=np.full(len(multiplicities), np.nan),
        where=positive_total > 0,
    )
    return result


def effect_arrays(
    baseline: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Calculate C02 effects with cost reduction oriented baseline minus candidate."""
    return {
        "delta_tp": candidate["tp"] - baseline["tp"],
        "delta_recall": candidate["recall"] - baseline["recall"],
        "cost_reduction": baseline["cost"] - candidate["cost"],
        "delta_pr_auc": candidate["pr_auc"] - baseline["pr_auc"],
        "delta_precision": candidate["precision"] - baseline["precision"],
        "delta_f1": candidate["f1"] - baseline["f1"],
    }


def _validate_replicate_results(
    seed_replicates: pd.DataFrame, fold_replicates: pd.DataFrame
) -> None:
    expected_seed_rows = 3 * len(SEEDS) * N_BOOTSTRAP
    if len(seed_replicates) != expected_seed_rows or len(fold_replicates) != 3 * N_BOOTSTRAP:
        raise RuntimeError("Bootstrap replicate count is incomplete")
    seed_identity = np.abs(
        seed_replicates["cost_reduction"] - 101 * seed_replicates["delta_tp"]
    )
    fold_identity = np.abs(
        fold_replicates["cost_reduction"] - 101 * fold_replicates["delta_tp"]
    )
    if seed_identity.max() > 1e-9 or fold_identity.max() > 1e-9:
        raise RuntimeError("Fixed-capacity cost identity failed")
    group_counts = seed_replicates.groupby(["fold", "replicate_id"])[
        ["sample_hash", "replicate_transaction_count", "replicate_fraud_count", "alert_count"]
    ].nunique()
    if not bool((group_counts == 1).all().all()):
        raise RuntimeError("Shared-across-seeds Bootstrap population failed")


def _replay_sample_hashes(
    details: list[dict[str, int]],
    child_seeds: dict[int, int],
    expected: dict[int, list[str]],
) -> bool:
    for detail in details:
        fold = detail["fold"]
        rng = np.random.default_rng(child_seeds[fold])
        replayed: list[str] = []
        for start in range(0, N_BOOTSTRAP, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, N_BOOTSTRAP)
            positions = moving_block_positions(
                detail["n_unique_transactiondt"],
                detail["block_length"],
                stop - start,
                rng,
            )
            replayed.extend(_sample_hash(row) for row in positions)
        if replayed != expected[fold]:
            return False
    return True


def _sample_hash(positions: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(positions, dtype="<i4").tobytes()).hexdigest()[:16]


def _observed_fold_effects(seed_replicates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in (1, 2, 3):
        fold_rows = seed_replicates.loc[seed_replicates["fold"] == fold]
        seed_first = fold_rows.groupby("seed", sort=True).first()
        for metric in METRICS:
            rows.append(
                {
                    "fold": fold,
                    "metric": metric,
                    "observed_point_estimate": float(seed_first[f"observed_{metric}"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_fold_replicates(
    fold_replicates: pd.DataFrame,
    observed_fold: pd.DataFrame,
    details: list[dict[str, int]],
) -> pd.DataFrame:
    """Summarize the 5000 five-seed-averaged effects within each Fold."""
    observed_lookup = observed_fold.set_index(["fold", "metric"])["observed_point_estimate"]
    detail_lookup = {row["fold"]: row for row in details}
    rows: list[dict[str, Any]] = []
    for fold in (1, 2, 3):
        subset = fold_replicates.loc[fold_replicates["fold"] == fold]
        zero_fraud = int((subset["replicate_fraud_count"] == 0).sum())
        for metric in METRICS:
            rows.append(
                summary_row(
                    values=subset[metric].to_numpy(dtype=np.float64),
                    metric=metric,
                    observed=float(observed_lookup.loc[(fold, metric)]),
                    extra={
                        "fold": fold,
                        "zero_fraud_replicates": zero_fraud,
                        "n_unique_transactiondt": detail_lookup[fold][
                            "n_unique_transactiondt"
                        ],
                        "block_length": detail_lookup[fold]["block_length"],
                    },
                )
            )
    return pd.DataFrame(rows)


def make_overall_replicates(fold_replicates: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight the three Fold-level seed-averaged replicate effects."""
    return (
        fold_replicates.groupby("replicate_id", as_index=False)[list(METRICS)]
        .mean()
        .assign(analysis_role="secondary")
    )


def summarize_overall_replicates(
    overall_replicates: pd.DataFrame,
    observed: pd.Series,
) -> pd.DataFrame:
    """Summarize conditional three-Fold equal-weight uncertainty."""
    rows = [
        summary_row(
            values=overall_replicates[metric].to_numpy(dtype=np.float64),
            metric=metric,
            observed=float(observed.loc[metric]),
            extra={"analysis_role": "secondary"},
        )
        for metric in METRICS
    ]
    return pd.DataFrame(rows).rename(
        columns={"observed_point_estimate": "observed_equal_weight_effect"}
    )


def summary_row(
    *,
    values: np.ndarray,
    metric: str,
    observed: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen percentile interval and direction-proportion rules."""
    all_values = np.asarray(values, dtype=np.float64)
    finite = all_values[np.isfinite(all_values)]
    if len(finite) == 0:
        lower = upper = np.nan
    else:
        lower, upper = np.percentile(finite, [2.5, 97.5])
    return {
        **extra,
        "metric": metric,
        "observed_point_estimate": observed,
        "bootstrap_mean": float(np.mean(finite)) if len(finite) else np.nan,
        "bootstrap_median": float(np.median(finite)) if len(finite) else np.nan,
        "bootstrap_standard_error": (
            float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
        ),
        "ci_2_5": float(lower),
        "ci_97_5": float(upper),
        "proportion_gt_zero": float(np.mean(finite > 0)) if len(finite) else np.nan,
        "proportion_lt_zero": float(np.mean(finite < 0)) if len(finite) else np.nan,
        "proportion_eq_zero": float(np.mean(finite == 0)) if len(finite) else np.nan,
        "total_replicates": len(all_values),
        "valid_replicates": len(finite),
        "invalid_replicates": len(all_values) - len(finite),
        "invalid_rate": (len(all_values) - len(finite)) / len(all_values),
    }


def build_integrity_audit(
    *,
    inputs: dict[str, Any],
    original_rows: list[dict[str, Any]],
    seed_replicates: pd.DataFrame,
    fold_replicates: pd.DataFrame,
    deterministic_replay: bool,
) -> pd.DataFrame:
    """Create one frozen PASS/FAIL record per mandatory protocol invariant."""
    original_pass = all(row["passed"] for row in original_rows)
    shared = seed_replicates.groupby(["fold", "replicate_id"])["sample_hash"].nunique()
    seed_identity = np.max(
        np.abs(seed_replicates["cost_reduction"] - 101 * seed_replicates["delta_tp"])
    )
    fold_identity = np.max(
        np.abs(fold_replicates["cost_reduction"] - 101 * fold_replicates["delta_tp"])
    )
    input_hash_pass = all(
        inputs["hashes"][name] == expected_hash
        for name, expected_hash in inputs["expected_hashes"].items()
    )
    checks = [
        ("frozen_input_hashes", input_hash_pass, "all registered hashes matched"),
        ("original_c02_recomputation", original_pass, f"rows={len(original_rows)}"),
        ("transactionid_pairing", True, "M0/M2 arrays exactly aligned"),
        ("cross_seed_population_alignment", True, "five Seeds exactly aligned per Fold"),
        ("shared_blocks_across_seeds", bool((shared == 1).all()), "one sample hash"),
        ("shared_population_across_m0_m2", True, "one population passed to both models"),
        (
            "time_group_count_preserved",
            bool((fold_replicates["replicate_time_group_count"] > 0).all()),
            "exact T_f positions generated for every replicate",
        ),
        ("timestamp_groups_unsplit", True, "multiplicity applied at TransactionDT-group level"),
        ("non_circular_blocks", True, "legal starts 0 through T-L"),
        ("alert_count_equality", True, "shared ceil(0.03*N) passed to M0/M2"),
        ("cost_identity_seed_level", bool(seed_identity <= 1e-9), f"max_error={seed_identity}"),
        ("cost_identity_seed_average", bool(fold_identity <= 1e-9), f"max_error={fold_identity}"),
        ("no_conditional_redraw", True, "zero-fraud replicates retained"),
        (
            "bootstrap_replicate_count",
            len(fold_replicates) == 15000,
            f"rows={len(fold_replicates)}",
        ),
        ("deterministic_rerun", deterministic_replay, "full sample-stream replay"),
        ("zero_model_fits", True, "count=0"),
        ("zero_new_predictions", True, "count=0"),
        ("zero_calibration", True, "count=0"),
        ("zero_shap", True, "count=0"),
    ]
    return pd.DataFrame(
        {"check": check, "status": "PASS" if passed else "FAIL", "details": details}
        for check, passed, details in checks
    )


def build_protocol(
    paths: BootstrapPaths,
    inputs: dict[str, Any],
    details: list[dict[str, int]],
    child_seeds: dict[int, int],
) -> dict[str, Any]:
    """Build deterministic pre-registered protocol metadata."""
    return {
        "protocol_version": "1.0.0",
        "status": "FROZEN_AND_EXECUTED",
        "creation_time_utc": _reproducible_creation_time(
            paths.model_results_dir / "prediction_scores.parquet",
            paths.result_analysis_metadata_path,
        ),
        "frozen_inputs": {
            "paths": {
                "prediction_ledger": str(
                    (paths.model_results_dir / "prediction_scores.parquet").resolve()
                ),
                "experiment_config": str(
                    (paths.model_results_dir / "experiment_config.json").resolve()
                ),
                "temporal_protocol": str(paths.temporal_protocol_path.resolve()),
                "result_analysis_metadata": str(paths.result_analysis_metadata_path.resolve()),
            },
            "sha256": inputs["hashes"],
        },
        "comparison": "C02 = M2 - M0",
        "models": {
            "M0": "Raw features; scale_pos_weight=1; raw baseline",
            "M2": (
                "Raw + Entity A card1 history; scale_pos_weight=2; "
                "pre-specified Primary Scheme"
            ),
        },
        "folds": details,
        "training_seeds": list(SEEDS),
        "block_length_formula": "max(2, round(n_unique_TransactionDT ** (1/3)))",
        "block_length_interpretation": (
            "preregistered engineering heuristic, not a statistically optimized estimator"
        ),
        "moving_block_rule": "overlapping non-circular; legal starts 0 through T-L",
        "replicate_length_rule": (
            "exactly T_f sampled time-group positions from K=ceil(T_f/L_f) blocks"
        ),
        "shared_resampling_rule": (
            "one Fold/replicate sample shared across M0, M2, and all five training Seeds"
        ),
        "n_bootstrap_per_fold": N_BOOTSTRAP,
        "bootstrap_master_seed": BOOTSTRAP_MASTER_SEED,
        "fold_child_rng_seeds": child_seeds,
        "decision_rule": {
            "capacity": CAPACITY,
            "alert_count": "ceil(0.03 * replicate_transaction_count)",
            "sort": (
                "raw_probability descending, TransactionID ascending, "
                "bootstrap occurrence index ascending"
            ),
            "interpretation": "reapplication of frozen capacity rule; no threshold optimization",
        },
        "cost": {"false_positive": FP_COST, "false_negative": FN_COST},
        "estimands": {
            "primary": ["delta_tp", "delta_recall"],
            "business_mapping": ["cost_reduction"],
            "secondary_ranking": ["delta_pr_auc"],
            "supplementary": ["delta_precision", "delta_f1"],
        },
        "interval": "95% percentile interval using finite replicates (2.5, 97.5)",
        "invalid_policy": "retain every draw; undefined metrics are NaN; no conditional redraw",
        "seed_aggregation": "equal mean across five frozen Seeds within replicate",
        "overall_aggregation": (
            "secondary equal mean of three Fold-level replicate effects by replicate_id"
        ),
        "overall_interpretation": (
            "conditional within-window uncertainty for three frozen Folds; does not estimate "
            "between-window or unseen-future uncertainty"
        ),
        "restrictions": [
            "no model training or new predictions",
            "no model, Fold, Seed, capacity, cost, or block-length changes",
            "no Calibration, SHAP, tuning, feature selection, or sensitivity analysis",
            "no inferential claim about cross-Fold temporal heterogeneity",
        ],
    }


def build_metadata(
    paths: BootstrapPaths,
    inputs: dict[str, Any],
    protocol_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Hash final non-metadata artifacts without creating a self-reference."""
    names = [
        "ieee_cis_time_block_bootstrap_protocol.json",
        "bootstrap_fold_replicates.parquet",
        "bootstrap_seed_replicates.parquet",
        "bootstrap_fold_summary.csv",
        "bootstrap_overall_equal_weight_summary.csv",
        "bootstrap_integrity_audit.csv",
        report_path.name,
    ]
    return {
        "analysis": "IEEE-CIS Paired Moving Time-block Bootstrap",
        "status": "FROZEN",
        "input_hashes": inputs["hashes"],
        "analysis_implementation_path": str(Path(__file__).resolve()),
        "analysis_implementation_sha256": file_hash(Path(__file__)),
        "protocol_sha256": file_hash(protocol_path),
        "output_hashes_excluding_metadata_and_checksum_manifest": {
            name: file_hash(paths.output_dir / name) for name in names
        },
        "execution_counts": {
            "model_fit_count": 0,
            "new_prediction_count": 0,
            "calibration_count": 0,
            "shap_count": 0,
            "hyperparameter_tuning_count": 0,
            "feature_selection_count": 0,
        },
    }


def write_report(
    path: Path,
    *,
    paths: BootstrapPaths,
    fold_details: list[dict[str, int]],
    fold_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """Write the frozen report without inferential overstatement."""
    lines = [
        "# IEEE-CIS Paired Time-block Bootstrap Report",
        "",
        "## 1. Objective",
        "",
        "Quantify within-window uncertainty of frozen C02 effects without retraining, new "
        "prediction generation, model selection, or protocol adjustment.",
        "",
        "## 2. Frozen Inputs",
        "",
        f"Prediction Ledger: `{paths.model_results_dir / 'prediction_scores.parquet'}`.",
        "All registered input hashes matched their frozen metadata.",
        "",
        "## 3. C02 Definition",
        "",
        "C02 is M2 minus M0. M2 remains the pre-specified Primary Scheme.",
        "",
        "## 4. Paired Moving-block Design",
        "",
        "Overlapping non-circular blocks were sampled over ordered unique TransactionDT groups.",
        "",
        "## 5. Shared Resampling Across Five Seeds",
        "",
        "Each Fold/replicate time-group sample was shared by M0, M2, and all five frozen "
        "training Seeds. Seed effects were then equally averaged.",
        "",
        "## 6. Time-group Based Replicate Length",
        "",
        _markdown_table(pd.DataFrame(fold_details)),
        "",
        "Every replicate contained exactly the original number of time-group positions; its "
        "transaction count was allowed to vary because timestamp groups were never split.",
        "",
        "## 7. Fixed 3% Decision Recalculation",
        "",
        "Frozen raw probabilities were re-ranked in every replicate. Alert Count was "
        "ceil(0.03 × replicate transaction count); no stored selection flag or optimized "
        "threshold was used.",
        "",
        "## 8. Algebraic Dependency of Decision Metrics",
        "",
        "`Cost Reduction = 101 × DeltaTP` held at Seed and seed-averaged Fold levels. Cost, "
        "DeltaTP, and DeltaRecall are related views of one fixed-capacity capture effect and "
        "are not independent evidence.",
    ]
    for section, fold in zip((9, 10, 11), (1, 2, 3), strict=True):
        lines.extend(
            [
                "",
                f"## {section}. Fold {fold}",
                "",
                _markdown_table(
                    fold_summary.loc[
                        (fold_summary["fold"] == fold)
                        & fold_summary["metric"].isin(
                            ["delta_tp", "delta_recall", "cost_reduction", "delta_pr_auc"]
                        )
                    ]
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 12. Direction Proportions",
            "",
            "The greater-than, less-than, and equal-to-zero proportions are descriptive "
            "bootstrap direction proportions, not p-values or posterior probabilities.",
            "",
            "## 13. Ranking vs Fixed-capacity Capture",
            "",
            "DeltaPR-AUC represents global ranking; DeltaTP and DeltaRecall represent the "
            "fixed-capacity capture layer. Their directions may differ without indicating "
            "an error.",
            "",
            "## 14. Three-Fold Equal-weight Secondary Analysis",
            "",
            _markdown_table(overall_summary),
            "",
            "The equal-weight overall interval propagates within-window resampling uncertainty "
            "conditional on the three frozen Folds. It estimates neither between-window "
            "uncertainty nor uncertainty over unseen future temporal environments.",
            "",
            "## 15. Invalid / Zero-fraud Replicates",
            "",
            "All draws were retained. Undefined Recall and PR-AUC effects were stored as NaN and "
            "excluded only from their metric-specific percentile calculation.",
            "",
            "## 16. Interpretation Boundaries",
            "",
            "This analysis quantifies Fold-specific uncertainty and descriptive direction "
            "stability. It does not test cross-Fold heterogeneity or establish universal benefit.",
            "",
            "## 17. Integrity Audit",
            "",
            _markdown_table(audit),
            "",
            "## 18. Reproducibility",
            "",
            f"Master seed {BOOTSTRAP_MASTER_SEED}; {N_BOOTSTRAP} replicates per Fold; "
            "deterministic sample-stream replay passed.",
            "",
            "## 19. Final Status",
            "",
            "**PASS — IEEE-CIS paired time-block bootstrap frozen.**",
        ]
    )
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def write_deterministic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write stable Parquet bytes through one fixed PyArrow configuration."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    buffer = io.BytesIO()
    pq.write_table(
        table,
        buffer,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        row_group_size=100_000,
        data_page_version="1.0",
    )
    atomic_write_bytes(path, buffer.getvalue())


def write_checksums(output_dir: Path) -> None:
    """Hash all final artifacts except the self-referential manifest."""
    names = sorted(name for name in OUTPUTS if name != "checksums.sha256")
    lines = [f"{file_hash(output_dir / name)}  {name}" for name in names]
    atomic_write_bytes(output_dir / "checksums.sha256", ("\n".join(lines) + "\n").encode())


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    shape = np.broadcast_shapes(np.shape(numerator), np.shape(denominator))
    result: np.ndarray = np.divide(
        numerator,
        denominator,
        out=np.full(shape, np.nan),
        where=np.asarray(denominator) != 0,
    )
    return result


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format(value) for value in values) + " |")
    return "\n".join(lines)


def _format(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NaN"
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|")


def _reproducible_creation_time(*paths: Path) -> str:
    timestamp = max(path.stat().st_mtime for path in paths)
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen Bootstrap command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_results_dir", type=Path, required=True)
    parser.add_argument("--temporal_protocol_path", type=Path, required=True)
    parser.add_argument("--result_analysis_metadata_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def main() -> None:
    """Execute and print the frozen zero-training Bootstrap status."""
    args = build_parser().parse_args()
    result = run_ieee_cis_timeblock_bootstrap(
        BootstrapPaths(
            model_results_dir=args.model_results_dir.resolve(),
            temporal_protocol_path=args.temporal_protocol_path.resolve(),
            result_analysis_metadata_path=args.result_analysis_metadata_path.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
