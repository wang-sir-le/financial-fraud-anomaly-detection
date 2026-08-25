"""Leakage-first entity and temporal audit for IEEE-CIS Fraud Detection.

The module performs descriptive data auditing only. It contains no model,
prediction, calibration, feature-importance, or threshold-selection code.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("fraudx.ieee_cis_audit")

TRANSACTION_REQUIRED_COLUMNS = (
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "card6",
    "addr1",
)
IDENTITY_REQUIRED_COLUMNS = ("TransactionID",)
ENTITY_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "Entity A": ("card1",),
    "Entity B": ("card1", "addr1"),
    "Entity C": ("card1", "card2", "card3", "card5", "card6"),
    "Entity D": ("card1", "card2", "card3", "card5", "card6", "addr1"),
}
HISTORY_FEATURE_REQUIREMENTS = {
    "entity_previous_transaction_count": 1,
    "entity_historical_amount_mean": 1,
    "entity_historical_amount_std": 2,
    "time_since_previous_transaction": 1,
}


@dataclass(frozen=True)
class AuditOutputs:
    """Materialized audit tables and report paths."""

    data_integrity: pd.DataFrame
    label_audit: pd.DataFrame
    join_audit: pd.DataFrame
    temporal_distribution: pd.DataFrame
    temporal_split: pd.DataFrame
    entity_repeatability: pd.DataFrame
    entity_temporal_coverage: pd.DataFrame
    historical_feasibility: pd.DataFrame
    report_path: Path
    log_path: Path


def run_ieee_cis_audit(
    transaction_path: Path,
    identity_path: Path,
    output_dir: Path,
    *,
    memory_chunk_rows: int = 50_000,
) -> AuditOutputs:
    """Run the complete non-model IEEE-CIS entity and temporal audit."""
    transaction_path = transaction_path.resolve()
    identity_path = identity_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "IEEE_CIS_entity_temporal_audit.log"
    _configure_logging(log_path)
    LOGGER.info("Starting IEEE-CIS Entity and Temporal Audit")
    LOGGER.info("Transaction input: %s", transaction_path)
    LOGGER.info("Identity input: %s", identity_path)
    _validate_input_file(transaction_path, "transaction")
    _validate_input_file(identity_path, "identity")

    transaction_header = pd.read_csv(transaction_path, nrows=0)
    identity_header = pd.read_csv(identity_path, nrows=0)
    _require_columns(transaction_header.columns, TRANSACTION_REQUIRED_COLUMNS, "transaction")
    _require_columns(identity_header.columns, IDENTITY_REQUIRED_COLUMNS, "identity")

    LOGGER.info("Loading audit working columns")
    transaction = pd.read_csv(
        transaction_path,
        usecols=list(TRANSACTION_REQUIRED_COLUMNS),
        low_memory=False,
    )
    identity = pd.read_csv(
        identity_path,
        usecols=list(IDENTITY_REQUIRED_COLUMNS),
        low_memory=False,
    )
    transaction_rows = len(transaction)
    identity_rows = len(identity)
    LOGGER.info("Loaded %d transaction rows and %d identity rows", transaction_rows, identity_rows)

    label_audit = audit_labels(transaction)
    _fail_on_integrity_error(transaction, identity, label_audit)
    LOGGER.info("Estimating full CSV DataFrame memory in %d-row chunks", memory_chunk_rows)
    transaction_memory = estimate_csv_dataframe_memory(transaction_path, memory_chunk_rows)
    identity_memory = estimate_csv_dataframe_memory(identity_path, memory_chunk_rows)
    data_integrity = audit_file_integrity(
        transaction_path,
        identity_path,
        transaction,
        identity,
        transaction_columns=len(transaction_header.columns),
        identity_columns=len(identity_header.columns),
        transaction_memory_bytes=transaction_memory,
        identity_memory_bytes=identity_memory,
    )

    identity_ids = pd.Index(identity["TransactionID"].dropna().unique())
    transaction = transaction.copy()
    transaction["has_identity"] = transaction["TransactionID"].isin(identity_ids)
    join_audit = audit_identity_join(transaction, identity)
    LOGGER.info(
        "Identity coverage: %.4f%%",
        float(join_audit.loc[join_audit["metric"] == "identity_coverage_rate", "value"].iloc[0])
        * 100,
    )

    LOGGER.info("Auditing ten equal-width TransactionDT bins")
    temporal_distribution = audit_temporal_distribution(transaction, bins=10)
    split_labels, temporal_split = assign_temporal_audit_split(transaction)
    transaction["audit_split"] = split_labels
    LOGGER.info("Temporary timestamp-preserving 70/15/15 audit split created")

    repeatability_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    feasibility_rows: list[dict[str, Any]] = []
    for entity_name, columns in ENTITY_DEFINITIONS.items():
        LOGGER.info("Auditing %s: %s", entity_name, " + ".join(columns))
        repeatability_rows.append(audit_entity_repeatability(transaction, entity_name, columns))
        coverage_rows.extend(audit_entity_temporal_coverage(transaction, entity_name, columns))
        feasibility_rows.extend(audit_history_feasibility(transaction, entity_name, columns))
    entity_repeatability = pd.DataFrame(repeatability_rows)
    entity_temporal_coverage = pd.DataFrame(coverage_rows)
    historical_feasibility = pd.DataFrame(feasibility_rows)

    tables = {
        "data_integrity_audit.csv": data_integrity,
        "label_audit.csv": label_audit,
        "join_audit.csv": join_audit,
        "temporal_distribution.csv": temporal_distribution,
        "temporal_split_audit.csv": temporal_split,
        "entity_repeatability_audit.csv": entity_repeatability,
        "entity_temporal_coverage.csv": entity_temporal_coverage,
        "historical_feature_feasibility.csv": historical_feasibility,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
        LOGGER.info("Saved %s (%d rows)", filename, len(table))

    metadata = {
        "audit_type": "IEEE-CIS Entity and Temporal Audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_training_performed": False,
        "prediction_generation_performed": False,
        "random_split_performed": False,
        "label_used_for_history": False,
        "history_rule": "TransactionDT strictly less than current TransactionDT",
        "same_timestamp_transactions_share_no_history_with_each_other": True,
        "temporary_split": "timestamp-preserving approximately 70/15/15 by row count",
        "entity_selection_frozen": False,
        "input_paths": {
            "transaction": str(transaction_path),
            "identity": str(identity_path),
        },
    }
    (output_dir / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_path = output_dir / "IEEE_CIS_entity_temporal_audit_report.md"
    write_audit_report(
        report_path,
        data_integrity=data_integrity,
        label_audit=label_audit,
        join_audit=join_audit,
        temporal_distribution=temporal_distribution,
        temporal_split=temporal_split,
        entity_repeatability=entity_repeatability,
        entity_temporal_coverage=entity_temporal_coverage,
        historical_feasibility=historical_feasibility,
    )
    LOGGER.info("Audit complete. Report: %s", report_path)
    return AuditOutputs(
        data_integrity=data_integrity,
        label_audit=label_audit,
        join_audit=join_audit,
        temporal_distribution=temporal_distribution,
        temporal_split=temporal_split,
        entity_repeatability=entity_repeatability,
        entity_temporal_coverage=entity_temporal_coverage,
        historical_feasibility=historical_feasibility,
        report_path=report_path,
        log_path=log_path,
    )


def estimate_csv_dataframe_memory(path: Path, chunk_rows: int) -> int:
    """Estimate full pandas deep memory by summing independently parsed chunks."""
    if chunk_rows <= 0:
        raise ValueError("memory_chunk_rows must be positive")
    total = 0
    for chunk in pd.read_csv(path, chunksize=chunk_rows, low_memory=False):
        total += int(chunk.memory_usage(index=True, deep=True).sum())
    return total


def audit_file_integrity(
    transaction_path: Path,
    identity_path: Path,
    transaction: pd.DataFrame,
    identity: pd.DataFrame,
    *,
    transaction_columns: int,
    identity_columns: int,
    transaction_memory_bytes: int,
    identity_memory_bytes: int,
) -> pd.DataFrame:
    """Return file, shape, memory, and TransactionID integrity metrics."""
    rows: list[dict[str, int | float | bool | str]] = []
    for dataset, path, frame, column_count, memory_bytes in (
        (
            "train_transaction",
            transaction_path,
            transaction,
            transaction_columns,
            transaction_memory_bytes,
        ),
        ("train_identity", identity_path, identity, identity_columns, identity_memory_bytes),
    ):
        values: dict[str, int | float | bool | str] = {
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "row_count": len(frame),
            "column_count": column_count,
            "estimated_pandas_deep_memory_bytes": memory_bytes,
            "transaction_id_unique_count": frame["TransactionID"].nunique(dropna=True),
            "transaction_id_missing_count": int(frame["TransactionID"].isna().sum()),
            "transaction_id_is_unique": bool(frame["TransactionID"].is_unique),
        }
        rows.extend(
            {"dataset": dataset, "metric": metric, "value": value}
            for metric, value in values.items()
        )
    return pd.DataFrame(rows)


def audit_labels(transaction: pd.DataFrame) -> pd.DataFrame:
    """Audit fraud labels without using them in any historical statistic."""
    labels = transaction["isFraud"]
    missing_count = int(labels.isna().sum())
    valid_mask = labels.isin([0, 1])
    invalid_count = int((~labels.isna() & ~valid_mask).sum())
    fraud_count = int((labels == 1).sum())
    normal_count = int((labels == 0).sum())
    valid_count = fraud_count + normal_count
    values = {
        "fraud_count": fraud_count,
        "normal_count": normal_count,
        "fraud_rate": fraud_count / valid_count if valid_count else np.nan,
        "missing_label_count": missing_count,
        "invalid_label_count": invalid_count,
        "valid_label_count": valid_count,
    }
    return pd.DataFrame([{"metric": metric, "value": value} for metric, value in values.items()])


def audit_identity_join(transaction: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Audit a left-join coverage indicator without dropping unlinked transactions."""
    covered = int(transaction["has_identity"].sum())
    total = len(transaction)
    identity_unique = int(identity["TransactionID"].nunique(dropna=True))
    values = {
        "transaction_count_before_left_join": total,
        "transaction_count_after_left_join": total,
        "identity_row_count": len(identity),
        "identity_unique_transaction_id_count": identity_unique,
        "identity_covered_transaction_count": covered,
        "identity_missing_transaction_count": total - covered,
        "identity_coverage_rate": covered / total if total else np.nan,
        "identity_missing_rate": (total - covered) / total if total else np.nan,
        "unmatched_transactions_retained": total - covered,
    }
    return pd.DataFrame([{"metric": metric, "value": value} for metric, value in values.items()])


def audit_temporal_distribution(transaction: pd.DataFrame, *, bins: int) -> pd.DataFrame:
    """Summarize ten equal-width TransactionDT intervals and identity coverage."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    time_values = transaction["TransactionDT"]
    if time_values.isna().any():
        raise ValueError("TransactionDT contains missing values")
    minimum = float(time_values.min())
    maximum = float(time_values.max())
    if minimum >= maximum:
        raise ValueError("TransactionDT must span more than one value")
    edges = np.linspace(minimum, maximum, bins + 1)
    bucket = pd.cut(
        time_values,
        bins=edges,
        labels=np.arange(1, bins + 1),
        include_lowest=True,
        right=True,
    )
    working = transaction.assign(time_bucket=bucket)
    grouped = working.groupby("time_bucket", observed=False, sort=True)
    result = grouped.agg(
        bucket_min_transaction_dt=("TransactionDT", "min"),
        bucket_max_transaction_dt=("TransactionDT", "max"),
        transaction_count=("TransactionID", "size"),
        fraud_count=("isFraud", "sum"),
        transaction_amount_median=("TransactionAmt", "median"),
        transaction_amount_mean=("TransactionAmt", "mean"),
        identity_covered_count=("has_identity", "sum"),
    ).reset_index()
    result["time_bucket"] = result["time_bucket"].astype(int)
    result["fraud_count"] = result["fraud_count"].astype(int)
    result["fraud_rate"] = result["fraud_count"] / result["transaction_count"]
    result["identity_coverage_rate"] = (
        result["identity_covered_count"] / result["transaction_count"]
    )
    result["global_time_min"] = minimum
    result["global_time_max"] = maximum
    result["global_time_duration"] = maximum - minimum
    return result


def assign_temporal_audit_split(transaction: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Create timestamp-preserving approximate 70/15/15 audit partitions."""
    counts = transaction.groupby("TransactionDT", sort=True).size()
    times = counts.index.to_numpy()
    cumulative = counts.cumsum().to_numpy()
    train_end = _nearest_time_boundary(cumulative, len(transaction) * 0.70, 1, len(times) - 2)
    validation_end = _nearest_time_boundary(
        cumulative, len(transaction) * 0.85, train_end + 1, len(times) - 1
    )
    train_max = times[train_end - 1]
    validation_max = times[validation_end - 1]
    labels = pd.Series(
        np.where(
            transaction["TransactionDT"] <= train_max,
            "Train",
            np.where(transaction["TransactionDT"] <= validation_max, "Validation", "Test"),
        ),
        index=transaction.index,
        name="audit_split",
    )
    rows: list[dict[str, Any]] = []
    for split_name in ("Train", "Validation", "Test"):
        subset = transaction.loc[labels == split_name]
        rows.append(
            {
                "split": split_name,
                "transaction_count": len(subset),
                "transaction_ratio": len(subset) / len(transaction),
                "min_transaction_dt": subset["TransactionDT"].min(),
                "max_transaction_dt": subset["TransactionDT"].max(),
                "fraud_count": int(subset["isFraud"].sum()),
                "fraud_rate": float(subset["isFraud"].mean()),
                "same_timestamp_cross_boundary": False,
            }
        )
    return labels, pd.DataFrame(rows)


def audit_entity_repeatability(
    transaction: pd.DataFrame, entity_name: str, columns: Sequence[str]
) -> dict[str, Any]:
    """Summarize coverage, repeatability, cold start, and temporal span."""
    valid_mask = transaction[list(columns)].notna().all(axis=1)
    valid = transaction.loc[valid_mask, [*columns, "TransactionDT"]]
    counts = valid.groupby(list(columns), dropna=False, sort=False).size()
    time_bounds = valid.groupby(list(columns), dropna=False, sort=False)[
        "TransactionDT"
    ].agg(["min", "max"])
    spans = time_bounds["max"] - time_bounds["min"]
    unique_entities = len(counts)
    return {
        "entity": entity_name,
        "entity_columns": " + ".join(columns),
        "total_transactions": len(transaction),
        "valid_entity_transactions": int(valid_mask.sum()),
        "missing_entity_transactions": int((~valid_mask).sum()),
        "missing_entity_ratio": float((~valid_mask).mean()),
        "unique_entity_count": unique_entities,
        "transactions_per_entity_mean": float(counts.mean()),
        "transactions_per_entity_median": float(counts.median()),
        "transactions_per_entity_p90": float(counts.quantile(0.90)),
        "transactions_per_entity_p95": float(counts.quantile(0.95)),
        "transactions_per_entity_max": int(counts.max()),
        "single_transaction_entity_ratio": float((counts == 1).mean()),
        "ge_2_transactions_entity_ratio": float((counts >= 2).mean()),
        "ge_5_transactions_entity_ratio": float((counts >= 5).mean()),
        "ge_10_transactions_entity_ratio": float((counts >= 10).mean()),
        "entity_time_span_mean": float(spans.mean()),
        "entity_time_span_median": float(spans.median()),
        "entity_time_span_p90": float(spans.quantile(0.90)),
    }


def audit_entity_temporal_coverage(
    transaction: pd.DataFrame, entity_name: str, columns: Sequence[str]
) -> list[dict[str, Any]]:
    """Measure future transaction-level and unique-entity coverage from Train."""
    train = transaction.loc[transaction["audit_split"] == "Train"]
    train_valid = train[list(columns)].notna().all(axis=1)
    train_entities = _entity_index(train.loc[train_valid], columns).unique()
    rows: list[dict[str, Any]] = []
    for split_name in ("Validation", "Test"):
        future = transaction.loc[transaction["audit_split"] == split_name]
        valid_mask = future[list(columns)].notna().all(axis=1)
        valid_future = future.loc[valid_mask]
        future_index = _entity_index(valid_future, columns)
        seen_transaction_mask = future_index.isin(train_entities)
        unique_future = future_index.unique()
        seen_unique_mask = unique_future.isin(train_entities)
        valid_count = len(valid_future)
        unique_count = len(unique_future)
        rows.append(
            {
                "entity": entity_name,
                "entity_columns": " + ".join(columns),
                "future_split": split_name,
                "future_transaction_count": len(future),
                "valid_entity_transaction_count": valid_count,
                "missing_entity_transaction_count": int((~valid_mask).sum()),
                "transaction_seen_in_train_count": int(seen_transaction_mask.sum()),
                "transaction_seen_in_train_ratio": (
                    float(seen_transaction_mask.mean()) if valid_count else np.nan
                ),
                "transaction_new_entity_count": int((~seen_transaction_mask).sum()),
                "transaction_new_entity_ratio": (
                    float((~seen_transaction_mask).mean()) if valid_count else np.nan
                ),
                "unique_future_entity_count": unique_count,
                "unique_entity_seen_in_train_count": int(seen_unique_mask.sum()),
                "unique_entity_seen_in_train_ratio": (
                    float(seen_unique_mask.mean()) if unique_count else np.nan
                ),
                "unique_new_entity_count": int((~seen_unique_mask).sum()),
                "unique_new_entity_ratio": (
                    float((~seen_unique_mask).mean()) if unique_count else np.nan
                ),
            }
        )
    return rows


def audit_history_feasibility(
    transaction: pd.DataFrame, entity_name: str, columns: Sequence[str]
) -> list[dict[str, Any]]:
    """Audit strictly earlier-time history availability without generating features."""
    valid_mask = transaction[list(columns)].notna().all(axis=1)
    valid = transaction.loc[valid_mask, [*columns, "TransactionDT"]]
    group_columns = [*columns, "TransactionDT"]
    time_group_counts = valid.groupby(group_columns, dropna=False, sort=True).size().rename(
        "same_timestamp_transaction_count"
    )
    grouped = time_group_counts.reset_index()
    grouped["prior_transaction_count"] = (
        grouped.groupby(list(columns), dropna=False, sort=False)[
            "same_timestamp_transaction_count"
        ].cumsum()
        - grouped["same_timestamp_transaction_count"]
    )
    valid_transaction_count = int(grouped["same_timestamp_transaction_count"].sum())
    rows: list[dict[str, Any]] = []
    for feature, minimum_prior_count in HISTORY_FEATURE_REQUIREMENTS.items():
        sufficient_mask = grouped["prior_transaction_count"] >= minimum_prior_count
        sufficient_count = int(
            grouped.loc[sufficient_mask, "same_timestamp_transaction_count"].sum()
        )
        insufficient_count = valid_transaction_count - sufficient_count
        rows.append(
            {
                "entity": entity_name,
                "entity_columns": " + ".join(columns),
                "candidate_historical_feature": feature,
                "minimum_strictly_prior_transactions_required": minimum_prior_count,
                "valid_entity_transaction_count": valid_transaction_count,
                "missing_entity_transaction_count": int((~valid_mask).sum()),
                "sufficient_history_transaction_count": sufficient_count,
                "sufficient_history_ratio_among_valid": (
                    sufficient_count / valid_transaction_count
                    if valid_transaction_count
                    else np.nan
                ),
                "insufficient_history_transaction_count": insufficient_count,
                "insufficient_history_ratio_among_valid": (
                    insufficient_count / valid_transaction_count
                    if valid_transaction_count
                    else np.nan
                ),
                "sufficient_history_ratio_all_transactions": sufficient_count / len(transaction),
                "history_time_rule": "TransactionDT < current TransactionDT",
                "same_timestamp_history_allowed": False,
                "label_used": False,
            }
        )
    return rows


def write_audit_report(
    path: Path,
    *,
    data_integrity: pd.DataFrame,
    label_audit: pd.DataFrame,
    join_audit: pd.DataFrame,
    temporal_distribution: pd.DataFrame,
    temporal_split: pd.DataFrame,
    entity_repeatability: pd.DataFrame,
    entity_temporal_coverage: pd.DataFrame,
    historical_feasibility: pd.DataFrame,
) -> None:
    """Write a reproducible Markdown report without freezing an entity choice."""
    integrity = _metric_lookup(data_integrity, key_columns=("dataset", "metric"))
    labels = _metric_lookup(label_audit, key_columns=("metric",))
    join = _metric_lookup(join_audit, key_columns=("metric",))
    temporal_fraud_range = (
        float(temporal_distribution["fraud_rate"].min()),
        float(temporal_distribution["fraud_rate"].max()),
    )
    identity_range = (
        float(temporal_distribution["identity_coverage_rate"].min()),
        float(temporal_distribution["identity_coverage_rate"].max()),
    )
    provisional = _provisional_entity_recommendation(
        entity_repeatability, entity_temporal_coverage, historical_feasibility
    )
    lines = [
        "# IEEE-CIS Entity and Temporal Audit Report",
        "",
        "## 1. Data integrity conclusion",
        "",
        f"- train_transaction: {int(integrity[('train_transaction', 'row_count')]):,} rows, "
        f"{int(integrity[('train_transaction', 'column_count')]):,} columns.",
        f"- train_identity: {int(integrity[('train_identity', 'row_count')]):,} rows, "
        f"{int(integrity[('train_identity', 'column_count')]):,} columns.",
        "- TransactionID is unique and non-missing in both input tables.",
        f"- Fraud/normal transactions: {int(labels[('fraud_count',)]):,}/"
        f"{int(labels[('normal_count',)]):,}; fraud rate "
        f"{float(labels[('fraud_rate',)]):.4%}.",
        f"- Missing/invalid labels: {int(labels[('missing_label_count',)])}/"
        f"{int(labels[('invalid_label_count',)])}.",
        "- No model was trained and no prediction was generated.",
        "",
        "## 2. Temporal drift",
        "",
        f"TransactionDT ranges from {temporal_distribution['global_time_min'].iloc[0]:g} to "
        f"{temporal_distribution['global_time_max'].iloc[0]:g}, with duration "
        f"{temporal_distribution['global_time_duration'].iloc[0]:g}.",
        f"Across ten equal-width time bins, fraud rate ranges from "
        f"{temporal_fraud_range[0]:.4%} to {temporal_fraud_range[1]:.4%}. "
        "This supports time-ordered validation and warrants explicit drift reporting.",
        "",
        "### Temporary timestamp-preserving audit split",
        "",
        _markdown_table(
            temporal_split,
            [
                "split",
                "transaction_count",
                "transaction_ratio",
                "min_transaction_dt",
                "max_transaction_dt",
                "fraud_rate",
            ],
        ),
        "",
        "This 70/15/15 split is used only for entity coverage auditing and does not freeze the "
        "later experimental protocol. Identical TransactionDT values never cross a boundary.",
        "",
        "## 3. Identity coverage",
        "",
        f"Identity rows cover {int(join[('identity_covered_transaction_count',)]):,} "
        f"transactions ({float(join[('identity_coverage_rate',)]):.4%}); missing identity "
        f"accounts for {float(join[('identity_missing_rate',)]):.4%}. All unmatched "
        "transactions were retained under LEFT JOIN semantics.",
        f"Across time bins, identity coverage ranges from {identity_range[0]:.4%} to "
        f"{identity_range[1]:.4%}.",
        "",
        "## 4. Candidate entity comparison",
        "",
        _markdown_table(
            entity_repeatability,
            [
                "entity",
                "entity_columns",
                "missing_entity_ratio",
                "unique_entity_count",
                "transactions_per_entity_median",
                "transactions_per_entity_p95",
                "single_transaction_entity_ratio",
                "entity_time_span_median",
            ],
        ),
        "",
        "### Future coverage",
        "",
        _markdown_table(
            entity_temporal_coverage,
            [
                "entity",
                "future_split",
                "transaction_seen_in_train_ratio",
                "transaction_new_entity_ratio",
                "unique_entity_seen_in_train_ratio",
                "unique_new_entity_ratio",
            ],
        ),
        "",
        "### Strictly prior history feasibility",
        "",
        _markdown_table(
            historical_feasibility,
            [
                "entity",
                "candidate_historical_feature",
                "minimum_strictly_prior_transactions_required",
                "sufficient_history_ratio_among_valid",
                "sufficient_history_ratio_all_transactions",
            ],
        ),
        "",
        "History is defined only by TransactionDT strictly less than the current transaction. "
        "Transactions sharing a timestamp do not enter one another's history, and isFraud was "
        "not used.",
        "",
        "## 5. Provisional entity recommendation",
        "",
        provisional,
        "",
        "The audit measures availability, repeatability, future coverage, and history depth; it "
        "cannot establish that an anonymized card combination is a semantically unique customer. "
        "The collision-versus-sparsity trade-off must therefore be reviewed before freezing.",
        "",
        "**Entity selection requires researcher decision.**",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _provisional_entity_recommendation(
    repeatability: pd.DataFrame,
    coverage: pd.DataFrame,
    feasibility: pd.DataFrame,
) -> str:
    test_coverage = coverage.loc[coverage["future_split"] == "Test"].set_index("entity")
    count_history = feasibility.loc[
        feasibility["candidate_historical_feature"] == "entity_previous_transaction_count"
    ].set_index("entity")
    repeat = repeatability.set_index("entity")
    score = (
        (1 - repeat["missing_entity_ratio"])
        * test_coverage["transaction_seen_in_train_ratio"]
        * count_history["sufficient_history_ratio_among_valid"]
    )
    best = str(score.idxmax())
    columns = str(repeat.loc[best, "entity_columns"])
    return (
        f"Availability-based provisional candidate: **{best} ({columns})**. It has the highest "
        "combined observed coverage, Test transaction history coverage, and strictly-prior "
        "history availability under this descriptive audit. This is a recommendation only, "
        "not a frozen entity definition."
    )


def _metric_lookup(
    frame: pd.DataFrame, *, key_columns: tuple[str, ...]
) -> dict[tuple[str, ...], Any]:
    result: dict[tuple[str, ...], Any] = {}
    for row in frame.to_dict("records"):
        key = tuple(str(row[column]) for column in key_columns)
        result[key] = row["value"]
    return result


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    headers = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = [headers, divider]
    for values in selected.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(_format_markdown_value(value) for value in values) + " |")
    return "\n".join(rows)


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NaN"
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|")


def _entity_index(frame: pd.DataFrame, columns: Sequence[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[list(columns)], names=list(columns))


def _nearest_time_boundary(
    cumulative_rows: np.ndarray, target: float, minimum: int, maximum: int
) -> int:
    candidates = np.arange(minimum, maximum + 1, dtype=int)
    distances = np.abs(cumulative_rows[candidates - 1] - target)
    return int(candidates[np.argmin(distances)])


def _fail_on_integrity_error(
    transaction: pd.DataFrame, identity: pd.DataFrame, label_audit: pd.DataFrame
) -> None:
    if transaction["TransactionID"].isna().any() or not transaction["TransactionID"].is_unique:
        raise ValueError("train_transaction TransactionID must be unique and non-missing")
    if identity["TransactionID"].isna().any() or not identity["TransactionID"].is_unique:
        raise ValueError("train_identity TransactionID must be unique and non-missing")
    lookup = label_audit.set_index("metric")["value"]
    if int(lookup["missing_label_count"]) or int(lookup["invalid_label_count"]):
        raise ValueError("isFraud contains missing or invalid labels")
    if transaction["TransactionDT"].isna().any():
        raise ValueError("TransactionDT contains missing values")
    if transaction["TransactionAmt"].isna().any():
        raise ValueError("TransactionAmt contains missing values")


def _validate_input_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"IEEE-CIS {name} file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"IEEE-CIS {name} file is empty: {path}")


def _require_columns(actual: Sequence[str], required: Sequence[str], dataset: str) -> None:
    missing = sorted(set(required).difference(actual))
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")


def _configure_logging(log_path: Path) -> None:
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    LOGGER.propagate = False


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone IEEE-CIS audit CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run a leakage-first IEEE-CIS entity and temporal audit without training"
    )
    parser.add_argument("--transaction_path", type=Path, required=True)
    parser.add_argument("--identity_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--memory_chunk_rows", type=int, default=50_000)
    return parser


def main() -> None:
    """Execute the command-line audit."""
    args = build_parser().parse_args()
    run_ieee_cis_audit(
        args.transaction_path,
        args.identity_path,
        args.output_dir,
        memory_chunk_rows=args.memory_chunk_rows,
    )


if __name__ == "__main__":
    main()
