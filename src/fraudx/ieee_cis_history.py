"""Strictly causal IEEE-CIS entity-history feature construction and auditing.

Only transaction attributes with ``historical TransactionDT < current TransactionDT``
are used. Labels are preserved in the output but excluded from every computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow

LOGGER = logging.getLogger("fraudx.ieee_cis_history")
EPSILON = 1e-6
ENTITY_A_COLUMNS = ("card1",)
ENTITY_C_COLUMNS = ("card1", "card2", "card3", "card5", "card6")
INPUT_COLUMNS = (
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "isFraud",
    *ENTITY_C_COLUMNS,
)
KEY_OUTPUT_COLUMNS = ("TransactionID", "TransactionDT", "TransactionAmt", "isFraud")
CORE_SUFFIXES = (
    "prev_count",
    "prev_amt_mean",
    "prev_amt_std",
    "amt_to_prev_mean",
    "time_since_last",
    "prev_unique_product_count",
)
STATUS_SUFFIXES = ("has_history", "has_std_history")


@dataclass(frozen=True)
class BuildArtifacts:
    """Paths and tables produced by a successful feature build."""

    feature_path: Path
    availability_path: Path
    leakage_path: Path
    metadata_path: Path
    report_path: Path
    log_path: Path
    row_count: int
    column_count: int


def build_ieee_history_features(
    transaction: pd.DataFrame,
    *,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """Build Entity A and Entity C features without using the fraud label."""
    if epsilon != EPSILON:
        raise ValueError(f"epsilon is frozen at {EPSILON:g}")
    _validate_transaction_frame(transaction)
    canonical = transaction.loc[:, list(INPUT_COLUMNS)].copy()
    canonical = canonical.sort_values(
        ["TransactionDT", "TransactionID"], kind="stable"
    ).reset_index(drop=True)
    # Keep the target only as an untouched output column. It is not passed to either
    # entity-history computation.
    history_source = canonical.drop(columns="isFraud")
    card_features = _build_entity_features(
        history_source,
        entity_columns=ENTITY_A_COLUMNS,
        prefix="card",
        epsilon=epsilon,
    )
    combo_features = _build_entity_features(
        history_source,
        entity_columns=ENTITY_C_COLUMNS,
        prefix="combo",
        epsilon=epsilon,
    )
    result = pd.concat(
        [canonical.loc[:, list(KEY_OUTPUT_COLUMNS)], card_features, combo_features],
        axis=1,
    )
    if len(result) != len(transaction):
        raise AssertionError("Feature builder changed the transaction count")
    if result["TransactionID"].duplicated().any():
        raise AssertionError("Feature output contains duplicate TransactionID values")
    return result


def run_history_feature_builder(
    transaction_path: Path,
    identity_path: Path,
    output_dir: Path,
) -> BuildArtifacts:
    """Build, audit, hash, and report the frozen IEEE-CIS history features."""
    transaction_path = transaction_path.resolve()
    identity_path = identity_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "IEEE_CIS_history_feature_builder.log"
    _configure_logging(log_path)
    LOGGER.info("Starting IEEE-CIS Entity History Feature Builder")
    _validate_input_file(transaction_path, "train_transaction")
    _validate_input_file(identity_path, "train_identity")
    transaction_header = pd.read_csv(transaction_path, nrows=0)
    _require_columns(transaction_header.columns, INPUT_COLUMNS, "train_transaction")
    identity_header = pd.read_csv(identity_path, nrows=0)
    _require_columns(identity_header.columns, ("TransactionID",), "train_identity")
    identity_ids = pd.read_csv(identity_path, usecols=["TransactionID"], low_memory=False)
    if identity_ids["TransactionID"].isna().any() or not identity_ids[
        "TransactionID"
    ].is_unique:
        raise ValueError("train_identity TransactionID must be unique and non-missing")

    LOGGER.info("Reading %d frozen transaction columns", len(INPUT_COLUMNS))
    transaction = pd.read_csv(
        transaction_path,
        usecols=list(INPUT_COLUMNS),
        low_memory=False,
    )
    LOGGER.info("Loaded %d transactions", len(transaction))
    output = build_ieee_history_features(transaction)
    LOGGER.info("Built %d history columns", len(output.columns) - len(KEY_OUTPUT_COLUMNS))

    availability = make_feature_availability_audit(output)
    LOGGER.info("Running leakage and invariance audits")
    leakage = run_history_leakage_audit(transaction, output)
    availability_path = output_dir / "history_feature_availability_audit.csv"
    leakage_path = output_dir / "history_feature_leakage_audit.csv"
    availability.to_csv(availability_path, index=False)
    leakage.to_csv(leakage_path, index=False)
    if not bool(leakage["passed"].all()):
        failed = leakage.loc[~leakage["passed"], "check"].tolist()
        raise RuntimeError(f"History feature leakage audit failed: {failed}")

    feature_path = output_dir / "ieee_cis_entity_history_features.parquet"
    output.to_parquet(
        feature_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    LOGGER.info("Saved feature Parquet: %s", feature_path)
    input_hashes = {
        "train_transaction_csv_sha256": _file_hash(transaction_path),
        "train_identity_csv_sha256": _file_hash(identity_path),
    }
    output_hash = _file_hash(feature_path)
    metadata = _feature_metadata(
        transaction_path=transaction_path,
        identity_path=identity_path,
        feature_path=feature_path,
        input_hashes=input_hashes,
        output_hash=output_hash,
        output=output,
    )
    metadata_path = output_dir / "history_feature_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_path = output_dir / "IEEE_CIS_history_feature_builder_report.md"
    write_history_feature_report(
        report_path,
        metadata=metadata,
        availability=availability,
        leakage=leakage,
    )
    LOGGER.info("All frozen acceptance checks passed")
    LOGGER.info("Report: %s", report_path)
    return BuildArtifacts(
        feature_path=feature_path,
        availability_path=availability_path,
        leakage_path=leakage_path,
        metadata_path=metadata_path,
        report_path=report_path,
        log_path=log_path,
        row_count=len(output),
        column_count=len(output.columns),
    )


def make_feature_availability_audit(output: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness and distributions for every output audit feature."""
    rows: list[dict[str, Any]] = []
    for entity, prefix in (("Entity A", "card"), ("Entity C", "combo")):
        features = [
            (f"{prefix}_entity_available", "availability"),
            *[(f"{prefix}_{suffix}", "core") for suffix in CORE_SUFFIXES],
            *[(f"{prefix}_{suffix}", "status") for suffix in STATUS_SUFFIXES],
        ]
        for feature, role in features:
            values = pd.to_numeric(output[feature], errors="coerce")
            non_missing = values.dropna()
            rows.append(
                {
                    "entity": entity,
                    "feature": feature,
                    "feature_role": role,
                    "row_count": len(values),
                    "non_missing_count": len(non_missing),
                    "missing_count": int(values.isna().sum()),
                    "missing_rate": float(values.isna().mean()),
                    "mean": float(non_missing.mean()) if len(non_missing) else np.nan,
                    "median": float(non_missing.median()) if len(non_missing) else np.nan,
                    "p90": float(non_missing.quantile(0.90)) if len(non_missing) else np.nan,
                    "p95": float(non_missing.quantile(0.95)) if len(non_missing) else np.nan,
                    "max": float(non_missing.max()) if len(non_missing) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_history_leakage_audit(
    transaction: pd.DataFrame,
    output: pd.DataFrame,
    *,
    sample_target_rows: int | None = None,
) -> pd.DataFrame:
    """Run structural checks and recomputation-based invariance audits.

    The production default recomputes all rows. ``sample_target_rows`` exists only
    to keep synthetic unit tests small and is never set by the production builder.
    """
    rows: list[dict[str, Any]] = []
    rows.append(
        _audit_row(
            "entity_definitions_frozen",
            ENTITY_A_COLUMNS == ("card1",)
            and ENTITY_C_COLUMNS == ("card1", "card2", "card3", "card5", "card6"),
            "Entity A=card1; Entity C=card1+card2+card3+card5+card6",
        )
    )
    rows.extend(_full_output_leakage_checks(transaction, output))
    audit_frame = (
        transaction
        if sample_target_rows is None
        else _representative_temporal_sample(transaction, sample_target_rows)
    )
    audit_output = (
        output
        if sample_target_rows is None
        else build_ieee_history_features(audit_frame)
    )
    feature_columns = _history_output_columns()

    mutated_labels = audit_frame.copy()
    mutated_labels["isFraud"] = 1 - mutated_labels["isFraud"].astype(int)
    mutated_output = build_ieee_history_features(mutated_labels)
    label_passed = _feature_frames_equal(audit_output, mutated_output, feature_columns)
    rows.append(
        _audit_row(
            "label_independence",
            label_passed,
            f"Flipped isFraud for {len(audit_frame):,} rows; feature columns unchanged",
        )
    )

    shuffled = audit_frame.sample(frac=1.0, random_state=20260821).reset_index(drop=True)
    shuffled_output = build_ieee_history_features(shuffled)
    order_passed = _feature_frames_equal(audit_output, shuffled_output, feature_columns)
    rows.append(
        _audit_row(
            "input_order_invariance",
            order_passed,
            f"Deterministically shuffled {len(audit_frame):,} rows and restored canonical order",
        )
    )

    extended = _append_strictly_future_transactions(audit_frame)
    extended_output = build_ieee_history_features(extended)
    original_after_append = extended_output.loc[
        extended_output["TransactionID"].isin(audit_frame["TransactionID"])
    ]
    prefix_passed = _feature_frames_equal(audit_output, original_after_append, feature_columns)
    rows.append(
        _audit_row(
            "prefix_invariance",
            prefix_passed,
            f"Prefix={len(audit_frame):,}; extended={len(extended):,}; original features unchanged",
        )
    )
    rows.append(
        _audit_row(
            "future_append_does_not_change_past",
            prefix_passed,
            "Appended rows have TransactionDT strictly greater than prefix maximum",
        )
    )
    rows.append(
        _audit_row(
            "isFraud_not_used",
            label_passed,
            "Target is preserved only as an output key field and excluded from history source",
        )
    )
    rows.append(
        _audit_row(
            "model_training_count_zero",
            True,
            "Feature builder contains no model, fit, prediction, tuning, or importance stage",
        )
    )
    return pd.DataFrame(rows)


def write_history_feature_report(
    path: Path,
    *,
    metadata: dict[str, Any],
    availability: pd.DataFrame,
    leakage: pd.DataFrame,
) -> None:
    """Write the frozen feature-builder report."""
    passed = bool(leakage["passed"].all())
    availability_view = availability.loc[
        availability["feature_role"] == "core",
        ["entity", "feature", "missing_rate", "mean", "median", "p90", "p95", "max"],
    ]
    lines = [
        "# IEEE-CIS Entity History Feature Builder Report",
        "",
        "## 1. Data input",
        "",
        f"- train_transaction.csv: `{metadata['input_files']['transaction_path']}`",
        f"- train_identity.csv: `{metadata['input_files']['identity_path']}`",
        f"- Output rows/columns: {metadata['row_count']:,}/{metadata['column_count']:,}",
        "- Identity input was integrity-checked and hashed but was not required by the frozen "
        "Entity A/C definitions.",
        "",
        "## 2. Entity definitions",
        "",
        "- Entity A: `card1` — anonymized card identifier.",
        "- Entity C: `card1 + card2 + card3 + card5 + card6` — finer-grained "
        "anonymized card-attribute combination.",
        "- Entity C is unavailable if any component is missing; missing values are never filled "
        "to create a synthetic entity.",
        "",
        "## 3. Historical rule",
        "",
        "Every feature uses only transactions satisfying `historical TransactionDT < current "
        "TransactionDT`. Transactions at the same timestamp never contribute to one another. "
        "Unlabeled transaction attributes from strictly earlier times remain available.",
        "",
        "## 4. Feature formulas",
        "",
        "- `*_prev_count`: number of strictly earlier entity transactions.",
        "- `*_prev_amt_mean`: mean amount over strictly earlier transactions.",
        "- `*_prev_amt_std`: sample standard deviation (`ddof=1`), requiring at least two "
        "strictly earlier transactions.",
        f"- `*_amt_to_prev_mean`: `TransactionAmt / (previous mean + {EPSILON:g})`.",
        "- `*_time_since_last`: current time minus the latest strictly earlier timestamp.",
        "- `*_prev_unique_product_count`: distinct non-missing ProductCD values observed at "
        "strictly earlier timestamps.",
        "",
        "## 5. Missing handling",
        "",
        "For an available entity with no history, previous count and previous unique ProductCD "
        "count are zero; mean, standard deviation, amount ratio, and time since last are NaN. "
        "For an unavailable entity, all six core values are NaN, entity_available is zero, and "
        "history status fields are zero. No global imputation is performed.",
        "",
        "## 6. Feature availability",
        "",
        _markdown_table(availability_view),
        "",
        "## 7. Leakage and invariance tests",
        "",
        _markdown_table(leakage[["check", "passed", "sample_examples"]]),
        "",
        "## 8. Frozen acceptance result",
        "",
        f"- Entity definitions frozen: {passed}",
        f"- Strict timestamp rule passed: {passed}",
        f"- Same-timestamp isolation passed: {passed}",
        f"- Prefix, future append, label, and input-order invariance passed: {passed}",
        f"- Entity C missing handling passed: {passed}",
        f"- Input/output hashes recorded: {bool(metadata['output_file']['sha256'])}",
        f"- Model training count: {metadata['model_training_count']}",
        "",
        f"**Feature builder frozen acceptance: {'PASS' if passed else 'FAIL'}.**",
        "",
        "Next stage is intentionally not executed: IEEE-CIS Rolling Temporal Validation + "
        "LightGBM Experiment.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_entity_features(
    frame: pd.DataFrame,
    *,
    entity_columns: Sequence[str],
    prefix: str,
    epsilon: float,
) -> pd.DataFrame:
    valid_mask = frame[list(entity_columns)].notna().all(axis=1)
    valid = frame.loc[
        valid_mask,
        [*entity_columns, "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"],
    ].copy()
    valid["_amount_squared"] = np.square(valid["TransactionAmt"].astype(float))
    group_columns = [*entity_columns, "TransactionDT"]
    summary = (
        valid.groupby(group_columns, dropna=False, sort=True)
        .agg(
            current_time_count=("TransactionID", "size"),
            current_time_amount_sum=("TransactionAmt", "sum"),
            current_time_amount_squared_sum=("_amount_squared", "sum"),
        )
        .reset_index()
    )
    entity_group = summary.groupby(list(entity_columns), dropna=False, sort=False)
    summary["_prior_count"] = (
        entity_group["current_time_count"].cumsum() - summary["current_time_count"]
    )
    summary["_prior_amount_sum"] = (
        entity_group["current_time_amount_sum"].cumsum()
        - summary["current_time_amount_sum"]
    )
    summary["_prior_amount_squared_sum"] = (
        entity_group["current_time_amount_squared_sum"].cumsum()
        - summary["current_time_amount_squared_sum"]
    )
    summary["_previous_time"] = entity_group["TransactionDT"].shift(1)
    count = summary["_prior_count"].astype(float)
    mean = np.divide(
        summary["_prior_amount_sum"],
        count,
        out=np.full(len(summary), np.nan),
        where=count > 0,
    )
    variance_numerator = (
        summary["_prior_amount_squared_sum"]
        - np.divide(
            np.square(summary["_prior_amount_sum"]),
            count,
            out=np.zeros(len(summary)),
            where=count > 0,
        )
    )
    sample_variance = np.divide(
        np.maximum(variance_numerator, 0.0),
        count - 1,
        out=np.full(len(summary), np.nan),
        where=count >= 2,
    )
    summary[f"{prefix}_prev_count"] = count
    summary[f"{prefix}_prev_amt_mean"] = mean
    summary[f"{prefix}_prev_amt_std"] = np.sqrt(sample_variance)
    summary[f"{prefix}_time_since_last"] = (
        summary["TransactionDT"] - summary["_previous_time"]
    )
    summary[f"{prefix}_prev_unique_product_count"] = _previous_unique_product_count(
        valid, summary, entity_columns
    )

    feature_columns = [
        f"{prefix}_{suffix}"
        for suffix in CORE_SUFFIXES
        if suffix != "amt_to_prev_mean"
    ]
    mapped = frame.loc[:, [*entity_columns, "TransactionDT", "TransactionAmt"]].merge(
        summary[[*group_columns, *feature_columns]],
        on=group_columns,
        how="left",
        sort=False,
        validate="many_to_one",
    )
    mapped[f"{prefix}_amt_to_prev_mean"] = np.divide(
        mapped["TransactionAmt"],
        mapped[f"{prefix}_prev_amt_mean"] + epsilon,
        out=np.full(len(mapped), np.nan),
        where=mapped[f"{prefix}_prev_amt_mean"].notna(),
    )
    mapped[f"{prefix}_entity_available"] = valid_mask.to_numpy(dtype=np.int8)
    mapped[f"{prefix}_has_history"] = (
        mapped[f"{prefix}_prev_count"].fillna(0) >= 1
    ).to_numpy(dtype=np.int8)
    mapped[f"{prefix}_has_std_history"] = (
        mapped[f"{prefix}_prev_count"].fillna(0) >= 2
    ).to_numpy(dtype=np.int8)
    ordered = [
        f"{prefix}_entity_available",
        *[f"{prefix}_{suffix}" for suffix in CORE_SUFFIXES],
        *[f"{prefix}_{suffix}" for suffix in STATUS_SUFFIXES],
    ]
    return mapped.loc[:, ordered].reset_index(drop=True)


def _previous_unique_product_count(
    valid: pd.DataFrame,
    summary: pd.DataFrame,
    entity_columns: Sequence[str],
) -> np.ndarray:
    product_rows = valid.loc[valid["ProductCD"].notna()]
    if product_rows.empty:
        return np.zeros(len(summary), dtype=float)
    first_appearance = (
        product_rows.groupby([*entity_columns, "ProductCD"], dropna=False, sort=True)[
            "TransactionDT"
        ]
        .min()
        .reset_index()
    )
    new_products = (
        first_appearance.groupby(
            [*entity_columns, "TransactionDT"], dropna=False, sort=True
        )
        .size()
        .rename("_new_product_count")
        .reset_index()
    )
    group_columns = [*entity_columns, "TransactionDT"]
    product_summary = summary[group_columns].merge(
        new_products,
        on=group_columns,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    product_summary["_new_product_count"] = product_summary["_new_product_count"].fillna(0)
    grouped = product_summary.groupby(list(entity_columns), dropna=False, sort=False)
    previous = (
        grouped["_new_product_count"].cumsum() - product_summary["_new_product_count"]
    )
    return np.asarray(previous, dtype=float)


def _full_output_leakage_checks(
    transaction: pd.DataFrame, output: pd.DataFrame
) -> list[dict[str, Any]]:
    aligned_input = transaction.sort_values(
        ["TransactionDT", "TransactionID"], kind="stable"
    ).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    first_passes: list[bool] = []
    same_time_passes: list[bool] = []
    strict_time_passes: list[bool] = []
    for prefix, columns in (("card", ENTITY_A_COLUMNS), ("combo", ENTITY_C_COLUMNS)):
        valid = aligned_input[list(columns)].notna().all(axis=1)
        working = pd.concat([aligned_input.loc[:, list(columns)], output], axis=1)
        valid_working = working.loc[valid]
        earliest = valid_working.groupby(list(columns), dropna=False)["TransactionDT"].transform(
            "min"
        )
        first_mask = valid_working["TransactionDT"] == earliest
        first_passes.append(
            bool((valid_working.loc[first_mask, f"{prefix}_prev_count"] == 0).all())
        )
        invariant_columns = [
            f"{prefix}_prev_count",
            f"{prefix}_prev_amt_mean",
            f"{prefix}_prev_amt_std",
            f"{prefix}_time_since_last",
            f"{prefix}_prev_unique_product_count",
            f"{prefix}_has_history",
            f"{prefix}_has_std_history",
        ]
        group_nunique = valid_working.groupby(
            [*columns, "TransactionDT"], dropna=False
        )[invariant_columns].nunique(dropna=False)
        same_time_passes.append(bool((group_nunique <= 1).all().all()))
        history_mask = valid_working[f"{prefix}_has_history"] == 1
        strict_time_passes.append(
            bool((valid_working.loc[history_mask, f"{prefix}_time_since_last"] > 0).all())
        )
    rows.append(
        _audit_row(
            "first_entity_transaction_count_zero",
            all(first_passes),
            "All earliest-timestamp rows per available Entity A/C have previous count zero",
        )
    )
    rows.append(
        _audit_row(
            "same_timestamp_transactions_isolated",
            all(same_time_passes),
            "Within each entity/timestamp group, all prior-history states are identical",
        )
    )
    rows.append(
        _audit_row(
            "strictly_earlier_timestamp_rule",
            all(strict_time_passes),
            "Every row with history has time_since_last > 0",
        )
    )
    combo_valid = aligned_input[list(ENTITY_C_COLUMNS)].notna().all(axis=1)
    combo_core = [f"combo_{suffix}" for suffix in CORE_SUFFIXES]
    missing_combo_passed = bool(
        output.loc[~combo_valid, combo_core].isna().all().all()
        and (output.loc[~combo_valid, "combo_entity_available"] == 0).all()
        and (output.loc[~combo_valid, list(map(lambda x: f"combo_{x}", STATUS_SUFFIXES))] == 0)
        .all()
        .all()
    )
    missing_ids = output.loc[~combo_valid, "TransactionID"].head(5).tolist()
    rows.append(
        _audit_row(
            "entity_c_missing_components_unavailable",
            missing_combo_passed,
            f"Example unavailable TransactionID values: {missing_ids}",
        )
    )
    input_ids = aligned_input["TransactionID"].to_numpy()
    output_integrity = bool(
        len(output) == len(transaction)
        and output["TransactionID"].is_unique
        and np.array_equal(output["TransactionID"].to_numpy(), input_ids)
    )
    rows.append(
        _audit_row(
            "output_integrity_and_stable_order",
            output_integrity,
            f"Rows={len(output):,}; unique TransactionID={output['TransactionID'].nunique():,}",
        )
    )
    return rows


def _representative_temporal_sample(
    transaction: pd.DataFrame, target_rows: int
) -> pd.DataFrame:
    ordered = transaction.sort_values(
        ["TransactionDT", "TransactionID"], kind="stable"
    ).reset_index(drop=True)
    if len(ordered) <= target_rows:
        return ordered
    candidate_time = ordered.loc[target_rows - 1, "TransactionDT"]
    return ordered.loc[ordered["TransactionDT"] <= candidate_time].copy()


def _append_strictly_future_transactions(transaction: pd.DataFrame) -> pd.DataFrame:
    if transaction.empty:
        raise ValueError("Cannot append future rows to an empty transaction frame")
    future = transaction.sort_values(
        ["TransactionDT", "TransactionID"], kind="stable"
    ).tail(min(5, len(transaction))).copy()
    maximum_id = int(transaction["TransactionID"].max())
    maximum_time = int(transaction["TransactionDT"].max())
    future["TransactionID"] = np.arange(maximum_id + 1, maximum_id + 1 + len(future))
    future["TransactionDT"] = maximum_time + np.arange(
        1, len(future) + 1, dtype=np.int64
    )
    return pd.concat([transaction, future], ignore_index=True)


def _feature_frames_equal(
    left: pd.DataFrame, right: pd.DataFrame, columns: Sequence[str]
) -> bool:
    left_sorted = left.sort_values("TransactionID").reset_index(drop=True)
    right_sorted = right.sort_values("TransactionID").reset_index(drop=True)
    if not np.array_equal(left_sorted["TransactionID"], right_sorted["TransactionID"]):
        return False
    try:
        pd.testing.assert_frame_equal(
            left_sorted[list(columns)],
            right_sorted[list(columns)],
            check_exact=True,
            check_dtype=True,
        )
    except AssertionError:
        return False
    return True


def _feature_metadata(
    *,
    transaction_path: Path,
    identity_path: Path,
    feature_path: Path,
    input_hashes: dict[str, str],
    output_hash: str,
    output: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "builder": "IEEE-CIS Entity History Feature Builder",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_files": {
            "transaction_path": str(transaction_path),
            "identity_path": str(identity_path),
            **input_hashes,
        },
        "output_file": {
            "path": str(feature_path),
            "sha256": output_hash,
            "format": "Parquet",
            "compression": "zstd",
        },
        "entity_definitions": {
            "Entity A": {
                "columns": list(ENTITY_A_COLUMNS),
                "paper_term": "anonymized card identifier",
            },
            "Entity C": {
                "columns": list(ENTITY_C_COLUMNS),
                "paper_term": "finer-grained anonymized card-attribute combination",
            },
        },
        "feature_formulas": {
            "previous_count": "count of entity transactions with TransactionDT < current",
            "previous_amount_mean": "mean amount where TransactionDT < current",
            "previous_amount_std": "sample std (ddof=1), at least 2 strictly prior rows",
            "amount_to_previous_mean": "TransactionAmt / (previous mean + epsilon)",
            "time_since_last": "current TransactionDT - max(strictly prior TransactionDT)",
            "previous_unique_product_count": (
                "distinct non-missing ProductCD where TransactionDT < current"
            ),
        },
        "epsilon": EPSILON,
        "timestamp_rule": "historical TransactionDT < current TransactionDT",
        "same_timestamp_transactions_contribute_history": False,
        "history_update_mode": "chronological unlabeled transaction-attribute updates",
        "missing_handling": {
            "available_entity_without_history": {
                "previous_count": 0,
                "previous_unique_product_count": 0,
                "mean_std_ratio_time_since_last": "NaN",
            },
            "unavailable_entity": (
                "all six core features NaN; entity_available=0; history statuses=0"
            ),
            "entity_component_imputation": "none",
            "global_imputation": "none",
        },
        "row_count": len(output),
        "column_count": len(output.columns),
        "columns": list(output.columns),
        "isFraud_used_for_feature_construction": False,
        "isFraud_preserved_as_output_only": True,
        "model_training_count": 0,
        "prediction_generation_count": 0,
        "feature_selection_count": 0,
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }


def _history_output_columns() -> list[str]:
    return [
        *[
            f"{prefix}_entity_available"
            for prefix in ("card", "combo")
        ],
        *[
            f"{prefix}_{suffix}"
            for prefix in ("card", "combo")
            for suffix in (*CORE_SUFFIXES, *STATUS_SUFFIXES)
        ],
    ]


def _audit_row(check: str, passed: bool, sample_examples: str) -> dict[str, Any]:
    return {"check": check, "passed": bool(passed), "sample_examples": sample_examples}


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


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_transaction_frame(transaction: pd.DataFrame) -> None:
    _require_columns(transaction.columns, INPUT_COLUMNS, "transaction frame")
    if transaction.empty:
        raise ValueError("Transaction frame is empty")
    if transaction["TransactionID"].isna().any() or not transaction["TransactionID"].is_unique:
        raise ValueError("TransactionID must be unique and non-missing")
    if transaction["TransactionDT"].isna().any():
        raise ValueError("TransactionDT must be non-missing")
    if transaction["TransactionAmt"].isna().any():
        raise ValueError("TransactionAmt must be non-missing")
    if not transaction["isFraud"].dropna().isin([0, 1]).all():
        raise ValueError("isFraud must contain only 0/1 when present")


def _validate_input_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{name} is empty: {path}")


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
    """Build the standalone frozen feature-builder CLI."""
    parser = argparse.ArgumentParser(
        description="Build strictly causal IEEE-CIS Entity A/C history features"
    )
    parser.add_argument("--transaction_path", type=Path, required=True)
    parser.add_argument("--identity_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def main() -> None:
    """Execute the feature builder without starting any model experiment."""
    args = build_parser().parse_args()
    run_history_feature_builder(args.transaction_path, args.identity_path, args.output_dir)


if __name__ == "__main__":
    main()
