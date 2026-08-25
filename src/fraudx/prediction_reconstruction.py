"""Reconstruct and audit the frozen multi-seed transaction-level predictions."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn

from fraudx.config import ExperimentConfig
from fraudx.data import rolling_temporal_splits, validation_period_masks
from fraudx.features import feature_columns_for_group
from fraudx.metrics import evaluate_probabilities, optimize_threshold
from fraudx.models import PlattCalibrator, build_lightgbm_pipeline
from fraudx.multiseed import _array_hash
from fraudx.pipeline import prepare_real_data
from fraudx.preprocess import load_prepared_frame

RECONSTRUCTION_MODELS = (
    ("raw_safe", "raw_safe", 1.0, "Raw"),
    ("recipient_full_weight_2", "recipient_full", 2.0, "Recipient history (w=2)"),
)
ABSOLUTE_TOLERANCE = 1e-10
EXACT_METRICS = {
    "test_size",
    "test_fraud_count",
    "tp",
    "fp",
    "tn",
    "fn",
    "alert_count",
    "business_cost",
    "all_normal_cost",
}
AUDIT_METRICS = (
    "test_size",
    "test_fraud_count",
    "test_fraud_rate",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "tn",
    "fn",
    "alert_rate",
    "alert_count",
    "business_cost",
    "all_normal_cost",
    "cost_saving",
    "cost_saving_rate",
    "threshold",
    "regenerated_validation_threshold",
)


def run_prediction_reconstruction(config: ExperimentConfig) -> dict[str, Any]:
    """Deterministically rebuild the 3 x 2 x 5 frozen paired prediction ledger."""
    output_dir = config.output_dir / "bootstrap"
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "bootstrap_prediction_reconstruction_status.json"
    historical_path = config.output_dir / "robustness" / "rolling_multiseed_results.csv"
    if not historical_path.is_file():
        raise FileNotFoundError(f"Historical multi-seed results not found: {historical_path}")

    # Thresholds can sit exactly on a repeated Test score. Round-trip parsing is
    # therefore required: a one-ULP upward change can alter the locked decision set.
    historical = pd.read_csv(historical_path, float_precision="round_trip")
    expected_model_ids = {item[0] for item in RECONSTRUCTION_MODELS}
    historical = historical.loc[historical["model_id"].isin(expected_model_ids)].copy()
    seeds = [int(value) for value in config.raw["multiseed"]["seeds"]]
    folds = int(config.raw["robustness"]["folds"])
    expected_count = folds * len(RECONSTRUCTION_MODELS) * len(seeds)
    _validate_historical_baseline(historical, expected_count, seeds)

    prepare_real_data(config)
    frame = load_prepared_frame(config.processed_path)
    robustness = config.raw["robustness"]
    splits = rolling_temporal_splits(
        frame,
        folds=folds,
        initial_train_fraction=float(robustness["initial_train_fraction"]),
        validation_fraction=float(robustness["validation_fraction"]),
        test_fraction=float(robustness["test_fraction"]),
        time_column=config.time_column,
    )
    fn_cost = float(config.raw["multiseed"]["false_negative_cost"])
    fp_cost = float(config.raw["multiseed"]["false_positive_cost"])
    capacity = float(config.raw["multiseed"]["maximum_alert_rate"])
    model_config = config.raw["model"]

    parts_dir = output_dir / ".prediction_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)
    temporary_ledger = output_dir / "rolling_multiseed_paired_predictions.tmp.parquet"
    final_ledger = output_dir / "rolling_multiseed_paired_predictions.parquet"
    if temporary_ledger.exists():
        temporary_ledger.unlink()

    audit_rows: list[dict[str, Any]] = []
    split_hashes: dict[str, str] = {}
    threshold_metadata: list[dict[str, Any]] = []
    all_indices_match = True
    all_labels_match = True
    actual_count = 0

    for fold, split in enumerate(splits, start=1):
        calibration_mask, threshold_mask = validation_period_masks(
            split.validation, config.time_column, config.target
        )
        calibration_labels = split.validation.loc[
            calibration_mask, config.target
        ].to_numpy(dtype=np.int8)
        threshold_labels = split.validation.loc[
            threshold_mask, config.target
        ].to_numpy(dtype=np.int8)
        test_labels = split.test[config.target].to_numpy(dtype=np.int8)
        transaction_ids = split.test.index.to_numpy(dtype=np.int64)
        test_steps = split.test[config.time_column].to_numpy(dtype=np.int32)
        split_hash = _test_split_hash(transaction_ids, test_steps, test_labels)
        split_hashes[str(fold)] = split_hash

        for seed in seeds:
            scheme_predictions: dict[str, dict[str, Any]] = {}
            for model_id, feature_group, weight, scheme_name in RECONSTRUCTION_MODELS:
                historical_row = _historical_row(historical, fold, model_id, seed)
                columns = feature_columns_for_group(split.train, feature_group)
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
                model.named_steps["classifier"].set_params(
                    random_state=seed,
                    bagging_seed=seed,
                    feature_fraction_seed=seed,
                    data_random_seed=seed,
                )
                model.fit(split.train[selected_columns], split.train[config.target])
                validation_raw = model.predict_proba(
                    split.validation[selected_columns]
                )[:, 1]
                calibrator = PlattCalibrator().fit(
                    validation_raw[calibration_mask], calibration_labels
                )
                threshold_scores = calibrator.transform(validation_raw[threshold_mask])
                regenerated_threshold, _ = optimize_threshold(
                    threshold_labels,
                    threshold_scores,
                    false_negative_cost=fn_cost,
                    false_positive_cost=fp_cost,
                    maximum_alert_rate=capacity,
                )
                test_raw = model.predict_proba(split.test[selected_columns])[:, 1]
                test_platt = calibrator.transform(test_raw)
                locked_threshold = float(historical_row["threshold"])
                decisions = (test_platt >= locked_threshold).astype(np.int8)
                reconstructed = _aggregate_metrics(
                    test_labels,
                    test_platt,
                    locked_threshold,
                    fn_cost=fn_cost,
                    fp_cost=fp_cost,
                )
                reconstructed["regenerated_validation_threshold"] = regenerated_threshold
                _append_audit_rows(
                    audit_rows,
                    historical_row=historical_row,
                    reconstructed=reconstructed,
                    fold=fold,
                    model_id=model_id,
                    scheme=scheme_name,
                    seed=seed,
                )
                raw_hash = _array_hash(test_raw)
                calibrated_hash = _array_hash(test_platt)
                _append_hash_audit(
                    audit_rows,
                    fold=fold,
                    model_id=model_id,
                    scheme=scheme_name,
                    seed=seed,
                    metric="raw_probability_hash",
                    historical_value=str(historical_row["raw_probability_hash"]),
                    reconstructed_value=raw_hash,
                )
                _append_hash_audit(
                    audit_rows,
                    fold=fold,
                    model_id=model_id,
                    scheme=scheme_name,
                    seed=seed,
                    metric="calibrated_probability_hash",
                    historical_value=str(historical_row["calibrated_probability_hash"]),
                    reconstructed_value=calibrated_hash,
                )
                scheme_predictions[model_id] = {
                    "raw": test_raw,
                    "platt": test_platt,
                    "threshold": locked_threshold,
                    "decision": decisions,
                    "transaction_ids": transaction_ids.copy(),
                    "steps": test_steps.copy(),
                    "labels": test_labels.copy(),
                }
                threshold_metadata.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "model_id": model_id,
                        "historical_locked_threshold": locked_threshold,
                        "regenerated_validation_threshold": regenerated_threshold,
                    }
                )
                actual_count += 1
                print(f"Reconstructed Fold {fold}, {model_id}, seed {seed}", flush=True)

            raw = scheme_predictions["raw_safe"]
            recipient = scheme_predictions["recipient_full_weight_2"]
            index_match = np.array_equal(raw["transaction_ids"], recipient["transaction_ids"])
            step_match = np.array_equal(raw["steps"], recipient["steps"])
            label_match = np.array_equal(raw["labels"], recipient["labels"])
            all_indices_match &= bool(index_match and step_match)
            all_labels_match &= bool(label_match)
            if not index_match or not step_match or not label_match:
                raise AssertionError(f"Paired Test alignment failed for Fold {fold}, seed {seed}")
            part = _paired_partition(
                fold=fold,
                seed=seed,
                split_hash=split_hash,
                transaction_ids=transaction_ids,
                steps=test_steps,
                labels=test_labels,
                raw=raw,
                recipient=recipient,
            )
            _write_parquet(part, parts_dir / f"fold_{fold}_seed_{seed}.parquet")

    audit = pd.DataFrame(audit_rows)
    audit_path = output_dir / "bootstrap_prediction_reconstruction_audit.csv"
    audit.to_csv(audit_path, index=False)
    status = _reconstruction_status(
        audit,
        expected_count=expected_count,
        actual_count=actual_count,
        all_indices_match=all_indices_match,
        all_labels_match=all_labels_match,
    )
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if not status["reconstruction_passed"]:
        _write_reconstruction_report(
            output_dir / "bootstrap_prediction_reconstruction_report.md",
            status=status,
            audit=audit,
            ledger_path=None,
        )
        raise RuntimeError(
            "Frozen prediction reconstruction failed strict audit; Bootstrap is blocked"
        )

    _combine_parquet_parts(parts_dir, temporary_ledger)
    if final_ledger.exists():
        final_ledger.unlink()
    temporary_ledger.replace(final_ledger)
    shutil.rmtree(parts_dir)
    prediction_hash = _file_hash(final_ledger)
    metadata = {
        "source_dataset": "PaySim",
        "processed_dataset_path": str(config.processed_path),
        "processed_dataset_size_bytes": config.processed_path.stat().st_size,
        "fold_boundaries": _fold_boundaries_from_history(historical),
        "schemes": ["Raw w=1", "Recipient history w=2"],
        "lightgbm_seeds": seeds,
        "calibration": "Platt",
        "pr_auc_score": "Platt-calibrated continuous probability",
        "thresholds": threshold_metadata,
        "false_negative_cost": fn_cost,
        "false_positive_cost": fp_cost,
        "capacity": capacity,
        "feature_sets": {"Raw w=1": "raw_safe", "Recipient history w=2": "recipient_full"},
        "library_versions": _library_versions(),
        "reconstruction_timestamp_utc": datetime.now(UTC).isoformat(),
        "test_split_hashes": split_hashes,
        "prediction_file_sha256": prediction_hash,
        "prediction_file_rows": _parquet_row_count(final_ledger),
        "reconstruction_mode": "deterministic frozen reconstruction",
    }
    (output_dir / "bootstrap_prediction_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_reconstruction_report(
        output_dir / "bootstrap_prediction_reconstruction_report.md",
        status=status,
        audit=audit,
        ledger_path=final_ledger,
    )
    return {"status": status, "audit": audit, "metadata": metadata, "ledger": final_ledger}


def require_reconstruction_passed(status_path: Path) -> dict[str, Any]:
    """Hard gate that prevents Bootstrap from using an unaudited prediction ledger."""
    if not status_path.is_file():
        raise RuntimeError("Prediction reconstruction status is missing; Bootstrap is blocked")
    status = cast(dict[str, Any], json.loads(status_path.read_text(encoding="utf-8")))
    if not bool(status.get("reconstruction_passed")):
        raise RuntimeError("Prediction reconstruction did not pass; Bootstrap is blocked")
    return status


def _validate_historical_baseline(
    historical: pd.DataFrame, expected_count: int, seeds: list[int]
) -> None:
    if len(historical) != expected_count:
        raise AssertionError(
            f"Expected {expected_count} historical Fold/Scheme/Seed rows, found {len(historical)}"
        )
    key = ["fold", "model_id", "seed"]
    if historical.duplicated(key).any():
        raise AssertionError("Historical multi-seed baseline contains duplicate keys")
    if set(historical["seed"].astype(int).unique()) != set(seeds):
        raise AssertionError("Historical seed set differs from the frozen protocol")


def _historical_row(
    historical: pd.DataFrame, fold: int, model_id: str, seed: int
) -> pd.Series:
    match = historical.loc[
        (historical["fold"] == fold)
        & (historical["model_id"] == model_id)
        & (historical["seed"] == seed)
    ]
    if len(match) != 1:
        raise AssertionError(f"Missing historical row for Fold {fold}, {model_id}, seed {seed}")
    return match.iloc[0]


def _aggregate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    fn_cost: float,
    fp_cost: float,
) -> dict[str, float | int]:
    report = evaluate_probabilities(
        labels,
        scores,
        threshold=threshold,
        false_negative_cost=fn_cost,
        false_positive_cost=fp_cost,
    )
    test_size = int(labels.size)
    fraud_count = int(labels.sum())
    all_normal_cost = float(fn_cost * fraud_count)
    cost_saving = float(all_normal_cost - report.business_cost)
    return {
        "test_size": test_size,
        "test_fraud_count": fraud_count,
        "test_fraud_rate": fraud_count / test_size,
        "pr_auc": report.pr_auc,
        "precision": report.precision,
        "recall": report.recall,
        "f1": report.f1,
        "tp": report.tp,
        "fp": report.fp,
        "tn": report.tn,
        "fn": report.fn,
        "alert_rate": report.alert_rate,
        "alert_count": int(report.tp + report.fp),
        "business_cost": report.business_cost,
        "all_normal_cost": all_normal_cost,
        "cost_saving": cost_saving,
        "cost_saving_rate": cost_saving / all_normal_cost if all_normal_cost else np.nan,
        "threshold": threshold,
    }


def _append_audit_rows(
    rows: list[dict[str, Any]],
    *,
    historical_row: pd.Series,
    reconstructed: dict[str, float | int],
    fold: int,
    model_id: str,
    scheme: str,
    seed: int,
) -> None:
    for metric in AUDIT_METRICS:
        historical_metric = "threshold" if metric == "regenerated_validation_threshold" else metric
        historical_value = float(historical_row[historical_metric])
        reconstructed_value = float(reconstructed[metric])
        absolute_difference = abs(historical_value - reconstructed_value)
        relative_difference = (
            absolute_difference / abs(historical_value)
            if historical_value != 0
            else (0.0 if reconstructed_value == 0 else np.inf)
        )
        passed = (
            reconstructed_value == historical_value
            if metric in EXACT_METRICS
            else absolute_difference <= ABSOLUTE_TOLERANCE
        )
        rows.append(
            {
                "fold": fold,
                "scheme": scheme,
                "model_id": model_id,
                "seed": seed,
                "metric": metric,
                "historical_value": historical_value,
                "reconstructed_value": reconstructed_value,
                "absolute_difference": absolute_difference,
                "relative_difference": relative_difference,
                "tolerance": 0.0 if metric in EXACT_METRICS else ABSOLUTE_TOLERANCE,
                "passed": bool(passed),
            }
        )


def _append_hash_audit(
    rows: list[dict[str, Any]],
    *,
    fold: int,
    model_id: str,
    scheme: str,
    seed: int,
    metric: str,
    historical_value: str,
    reconstructed_value: str,
) -> None:
    rows.append(
        {
            "fold": fold,
            "scheme": scheme,
            "model_id": model_id,
            "seed": seed,
            "metric": metric,
            "historical_value": historical_value,
            "reconstructed_value": reconstructed_value,
            "absolute_difference": 0.0 if historical_value == reconstructed_value else np.nan,
            "relative_difference": 0.0 if historical_value == reconstructed_value else np.nan,
            "tolerance": 0.0,
            "passed": historical_value == reconstructed_value,
        }
    )


def _paired_partition(
    *,
    fold: int,
    seed: int,
    split_hash: str,
    transaction_ids: np.ndarray,
    steps: np.ndarray,
    labels: np.ndarray,
    raw: dict[str, Any],
    recipient: dict[str, Any],
) -> pd.DataFrame:
    size = len(labels)
    return pd.DataFrame(
        {
            "fold": np.full(size, fold, dtype=np.int8),
            "seed": np.full(size, seed, dtype=np.int16),
            "transaction_id": transaction_ids,
            "original_row_index": transaction_ids,
            "step": steps,
            "y_true": labels,
            "raw_model_probability_raw": raw["raw"],
            "raw_model_probability_platt": raw["platt"],
            "raw_locked_threshold": np.full(size, raw["threshold"], dtype=np.float64),
            "raw_binary_decision": raw["decision"],
            "recipient_model_probability_raw": recipient["raw"],
            "recipient_model_probability_platt": recipient["platt"],
            "recipient_locked_threshold": np.full(
                size, recipient["threshold"], dtype=np.float64
            ),
            "recipient_binary_decision": recipient["decision"],
            "test_split_hash": np.full(size, split_hash, dtype=object),
            "test_order_index": np.arange(size, dtype=np.int32),
            "raw_pr_auc_score": raw["platt"],
            "recipient_pr_auc_score": recipient["platt"],
        }
    )


def _reconstruction_status(
    audit: pd.DataFrame,
    *,
    expected_count: int,
    actual_count: int,
    all_indices_match: bool,
    all_labels_match: bool,
) -> dict[str, Any]:
    passed_by_metric = audit.groupby("metric")["passed"].all()
    confusion_metrics = ["tp", "fp", "tn", "fn", "alert_count"]
    status = {
        "expected_fold_seed_scheme_count": expected_count,
        "actual_count": actual_count,
        "all_test_indices_match": all_indices_match,
        "all_labels_match": all_labels_match,
        "all_confusion_matrices_match": bool(passed_by_metric[confusion_metrics].all()),
        "all_business_costs_match": bool(passed_by_metric["business_cost"]),
        "all_thresholds_match": bool(
            passed_by_metric[["threshold", "regenerated_validation_threshold"]].all()
        ),
        "all_pr_auc_match_within_tolerance": bool(passed_by_metric["pr_auc"]),
        "all_probability_hashes_match": bool(
            passed_by_metric[["raw_probability_hash", "calibrated_probability_hash"]].all()
        ),
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
    }
    status["reconstruction_passed"] = bool(
        actual_count == expected_count
        and all_indices_match
        and all_labels_match
        and audit["passed"].all()
    )
    return status


def _test_split_hash(indices: np.ndarray, steps: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values, dtype in ((indices, np.int64), (steps, np.int32), (labels, np.int8)):
        digest.update(np.ascontiguousarray(values, dtype=dtype).tobytes())
    return digest.hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("prediction_chunk", frame)
        connection.execute(
            f"COPY prediction_chunk TO '{_escape_path(path)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
        )
    finally:
        connection.close()


def _combine_parquet_parts(parts_dir: Path, output_path: Path) -> None:
    glob = _escape_path(parts_dir / "*.parquet")
    connection = duckdb.connect(database=":memory:")
    try:
        # Paths are escaped local paths; DuckDB COPY does not support a path bind here.
        connection.execute(
            f"COPY (SELECT * FROM read_parquet('{glob}', union_by_name=true) "  # nosec B608
            "ORDER BY fold, seed, test_order_index) "
            f"TO '{_escape_path(output_path)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
        )
    finally:
        connection.close()


def _parquet_row_count(path: Path) -> int:
    connection = duckdb.connect(database=":memory:")
    try:
        # The ledger path is a trusted local path and is escaped above.
        row = connection.execute(
            f"SELECT COUNT(*) FROM read_parquet('{_escape_path(path)}')"  # nosec B608
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("Unable to count prediction ledger")
    return int(row[0])


def _escape_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_boundaries_from_history(historical: pd.DataFrame) -> list[dict[str, int]]:
    columns = [
        "fold",
        "train_min_step",
        "train_max_step",
        "validation_min_step",
        "validation_max_step",
        "test_min_step",
        "test_max_step",
    ]
    records = historical[columns].drop_duplicates().sort_values("fold").to_dict("records")
    return cast(list[dict[str, int]], records)


def _library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "lightgbm": lightgbm.__version__,
        "duckdb": duckdb.__version__,
    }


def _write_reconstruction_report(
    path: Path,
    *,
    status: dict[str, Any],
    audit: pd.DataFrame,
    ledger_path: Path | None,
) -> None:
    failed = audit.loc[~audit["passed"]]
    lines = [
        "# Frozen Prediction Reconstruction Report",
        "",
        "## Protocol",
        "",
        "The frozen 3 Fold x 2 Scheme x 5 Seed protocol was rebuilt deterministically. "
        "Historical locked validation thresholds were applied to the reconstructed Test "
        "probabilities; regenerated validation thresholds were used only for audit.",
        "",
        "## Audit status",
        "",
        f"- Reconstruction passed: **{status['reconstruction_passed']}**",
        f"- Expected/actual model runs: {status['expected_fold_seed_scheme_count']}/"
        f"{status['actual_count']}",
        f"- Stable Test indices: {status['all_test_indices_match']}",
        f"- Labels aligned: {status['all_labels_match']}",
        f"- Confusion matrices exact: {status['all_confusion_matrices_match']}",
        f"- Business costs exact: {status['all_business_costs_match']}",
        f"- Thresholds matched: {status['all_thresholds_match']}",
        f"- PR-AUC within {ABSOLUTE_TOLERANCE:g}: "
        f"{status['all_pr_auc_match_within_tolerance']}",
        f"- Probability hashes exact: {status['all_probability_hashes_match']}",
        "",
        "## Ledger",
        "",
        str(ledger_path) if ledger_path is not None else "Not finalized because audit failed.",
        "",
        "## Failed audit rows",
        "",
    ]
    if failed.empty:
        lines.append("None.")
    else:
        lines.extend(["```text", failed.to_string(index=False), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
