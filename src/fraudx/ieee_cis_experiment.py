"""Frozen IEEE-CIS M0-M3 LightGBM experiment with auditable checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from lightgbm import LGBMClassifier
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import average_precision_score, roc_auc_score

DATASET_ROWS = 590_540
SEEDS = (42, 52, 62, 72, 82)
CAPACITY = 0.03
FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 100
RAW_EXCLUDED = ("TransactionID", "isFraud")
KEY_COLUMNS = ("TransactionID", "TransactionDT", "isFraud")
HISTORY_KEY_COLUMNS = ("TransactionID", "TransactionDT", "TransactionAmt", "isFraud")
CATEGORICAL_RAW_COLUMNS = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
)
MODEL_ORDER = ("M0", "M1", "M2", "M3")
FIXED_MODEL_PARAMETERS: dict[str, Any] = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}
REQUIRED_OUTPUTS = (
    "experiment_config.json",
    "seed_metrics.csv",
    "fold_metrics.csv",
    "business_cost_results.csv",
    "prediction_scores.parquet",
    "IEEE_CIS_lightgbm_report.md",
)


@dataclass(frozen=True)
class Scheme:
    """One pre-registered experimental scheme."""

    model_id: str
    feature_set: str
    history_prefix: str | None
    scale_pos_weight: float
    role: str


SCHEMES = (
    Scheme("M0", "Raw", None, 1.0, "raw transaction baseline"),
    Scheme("M1", "Raw + Entity A history", "card_", 1.0, "history contribution"),
    Scheme(
        "M2",
        "Raw + Entity A history",
        "card_",
        2.0,
        "pre-specified primary framework",
    ),
    Scheme(
        "M3",
        "Raw + Entity C history",
        "combo_",
        2.0,
        "entity-granularity sensitivity analysis",
    ),
)


@dataclass(frozen=True)
class ExperimentPaths:
    """Frozen input and output paths."""

    transaction_path: Path
    history_path: Path
    history_metadata_path: Path
    protocol_path: Path
    output_dir: Path


@dataclass(frozen=True)
class FoldIndices:
    """Row indices for one protocol-defined temporal fold."""

    fold: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class CapacityResult:
    """Deterministic fixed-capacity decision and metrics."""

    selected: np.ndarray
    score_rank: np.ndarray
    ranking_boundary_probability: float
    alert_count: int
    alert_rate: float
    capacity_deviation: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision_at_3pct: float
    recall_at_3pct: float
    f1_at_3pct: float
    fraud_captured: int
    baseline_cost: float
    business_cost: float
    cost_saving_rate: float


class TrainOnlyCategoryEncoder:
    """Stable missing/unknown-aware integer encoding learned from Train only."""

    missing_code = 0
    unknown_code = 1

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = list(columns)
        self.vocabularies: dict[str, tuple[str, ...]] = {}

    def fit(self, train: pd.DataFrame) -> TrainOnlyCategoryEncoder:
        """Learn sorted category vocabularies without reading future frames."""
        _require_columns(train.columns, self.columns, "categorical Train frame")
        self.vocabularies = {
            column: tuple(sorted(train[column].dropna().astype(str).unique().tolist()))
            for column in self.columns
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Encode known, missing, and future-unknown values deterministically."""
        if set(self.vocabularies) != set(self.columns):
            raise RuntimeError("Category encoder must be fitted before transform")
        encoded: dict[str, pd.Series] = {}
        for column in self.columns:
            source = frame[column]
            mapping = {
                category: code
                for code, category in enumerate(self.vocabularies[column], start=2)
            }
            values = source.astype("string").map(mapping).fillna(self.unknown_code)
            values = values.mask(source.isna(), self.missing_code)
            encoded[column] = values.astype(np.int32)
        return pd.DataFrame(encoded, index=frame.index)

    def vocabulary_hash(self) -> str:
        """Hash the complete ordered Train vocabulary."""
        return canonical_hash(self.vocabularies)

    def audit_rows(
        self,
        fold: int,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Return missing and future-unknown counts by categorical feature."""
        rows: list[dict[str, Any]] = []
        for column in self.columns:
            vocabulary = set(self.vocabularies[column])
            row: dict[str, Any] = {
                "fold": fold,
                "feature": column,
                "train_vocabulary_size": len(vocabulary),
                "train_vocabulary_hash": canonical_hash(sorted(vocabulary)),
            }
            for split_name, frame in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            ):
                strings = frame[column].astype("string")
                missing = frame[column].isna()
                unknown = ~missing & ~strings.isin(vocabulary)
                row[f"{split_name}_missing_count"] = int(missing.sum())
                row[f"{split_name}_unknown_count"] = int(unknown.sum())
            rows.append(row)
        return rows


def load_frozen_inputs(
    paths: ExperimentPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Verify frozen hashes, then load compact raw and history frames."""
    for path, name in (
        (paths.transaction_path, "transaction CSV"),
        (paths.history_path, "history Parquet"),
        (paths.history_metadata_path, "history metadata"),
        (paths.protocol_path, "temporal protocol"),
    ):
        _validate_file(path, name)
    metadata = json.loads(paths.history_metadata_path.read_text(encoding="utf-8"))
    protocol = json.loads(paths.protocol_path.read_text(encoding="utf-8"))
    transaction_hash = file_hash(paths.transaction_path)
    history_hash = file_hash(paths.history_path)
    expected_transaction_hashes = {
        metadata["input_files"]["train_transaction_csv_sha256"],
        protocol["input_files"]["train_transaction_sha256"],
    }
    expected_history_hashes = {
        metadata["output_file"]["sha256"],
        protocol["input_files"]["history_feature_parquet_sha256"],
    }
    if expected_transaction_hashes != {transaction_hash}:
        raise RuntimeError("Frozen transaction SHA-256 does not match the current CSV")
    if expected_history_hashes != {history_hash}:
        raise RuntimeError("Frozen history SHA-256 does not match the current Parquet")

    header = pd.read_csv(paths.transaction_path, nrows=0)
    _require_columns(header.columns, RAW_EXCLUDED, "transaction CSV")
    _require_columns(header.columns, CATEGORICAL_RAW_COLUMNS, "transaction CSV")
    dtype_map: dict[str, Any] = {
        column: ("string" if column in CATEGORICAL_RAW_COLUMNS else np.float32)
        for column in header.columns
    }
    dtype_map.update(
        {"TransactionID": np.int64, "TransactionDT": np.int64, "isFraud": np.int8}
    )
    transaction = pd.read_csv(
        paths.transaction_path,
        dtype=dtype_map,
        low_memory=False,
    )
    history = pd.read_parquet(paths.history_path, engine="pyarrow")
    validate_join_inputs(transaction, history, expected_rows=DATASET_ROWS)
    return transaction, history, metadata, protocol


def validate_join_inputs(
    transaction: pd.DataFrame,
    history: pd.DataFrame,
    *,
    expected_rows: int | None = None,
) -> None:
    """Validate one-to-one raw/history alignment without changing either input."""
    _require_columns(transaction.columns, KEY_COLUMNS, "transaction frame")
    _require_columns(history.columns, HISTORY_KEY_COLUMNS, "history frame")
    if expected_rows is not None and len(transaction) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} transactions, found {len(transaction):,}")
    if len(transaction) != len(history):
        raise ValueError("Raw and history row counts differ")
    for frame, name in ((transaction, "raw"), (history, "history")):
        if frame["TransactionID"].isna().any() or not frame["TransactionID"].is_unique:
            raise ValueError(f"{name} TransactionID must be unique and non-missing")
    raw_keys = transaction[[*KEY_COLUMNS]].sort_values("TransactionID").reset_index(drop=True)
    history_keys = history[[*KEY_COLUMNS]].sort_values("TransactionID").reset_index(drop=True)
    for column in KEY_COLUMNS:
        if not np.array_equal(raw_keys[column].to_numpy(), history_keys[column].to_numpy()):
            raise ValueError(f"Raw/history {column} values differ")


def indices_from_protocol(transaction: pd.DataFrame, protocol: dict[str, Any]) -> list[FoldIndices]:
    """Apply exact stored boundaries; never estimate or modify them."""
    if int(protocol["split_rule"]["fold_count"]) != 3:
        raise ValueError("Frozen protocol must contain exactly three folds")
    folds: list[FoldIndices] = []
    times = transaction["TransactionDT"].to_numpy()
    for definition in protocol["fold_definitions"]:
        parts: dict[str, np.ndarray] = {}
        seen_times: list[set[int]] = []
        for split_name in ("train", "validation", "test"):
            split = definition[split_name]
            mask = (times >= split["start_transaction_dt"]) & (
                times <= split["end_transaction_dt"]
            )
            indices = np.flatnonzero(mask)
            if len(indices) != int(split["transaction_count"]):
                raise RuntimeError(
                    f"Fold {definition['fold']} {split_name} row count differs from protocol"
                )
            parts[split_name] = indices
            seen_times.append(set(times[indices].tolist()))
        if any(
            seen_times[left] & seen_times[right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise RuntimeError("A TransactionDT group crosses frozen split boundaries")
        if not (
            times[parts["train"]].max() < times[parts["validation"]].min()
            and times[parts["validation"]].max() < times[parts["test"]].min()
        ):
            raise RuntimeError("Frozen split temporal ordering failed")
        folds.append(
            FoldIndices(
                fold=int(definition["fold"]),
                train=parts["train"],
                validation=parts["validation"],
                test=parts["test"],
            )
        )
    return folds


def history_feature_lists(metadata: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Extract explicit Entity A/C lists from frozen output schema."""
    columns = list(metadata["columns"])
    card = [column for column in columns if column.startswith("card_")]
    combo = [column for column in columns if column.startswith("combo_")]
    if len(card) != 9 or len(combo) != 9:
        raise RuntimeError(
            f"Expected 9 Entity A and 9 Entity C features, found {len(card)}/{len(combo)}"
        )
    return card, combo


def feature_specifications(
    transaction: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Create canonical ordered feature specifications and model column lists."""
    raw = [column for column in transaction.columns if column not in RAW_EXCLUDED]
    categorical = [column for column in raw if not is_numeric_dtype(transaction[column])]
    card, combo = history_feature_lists(metadata)
    feature_lists = {
        "M0": raw,
        "M1": [*raw, *card],
        "M2": [*raw, *card],
        "M3": [*raw, *combo],
    }
    specifications: dict[str, Any] = {}
    for scheme in SCHEMES:
        history = (
            []
            if scheme.history_prefix is None
            else (card if scheme.history_prefix == "card_" else combo)
        )
        column_spec = [
            {
                "name": column,
                "source": "train_transaction.csv" if column in raw else "history_parquet",
                "role": "categorical" if column in categorical else "numeric",
                "history_feature": column in history,
            }
            for column in feature_lists[scheme.model_id]
        ]
        feature_only = {
            "ordered_columns": column_spec,
            "preprocessing": {
                "numeric": "float32; infinities to NaN; native LightGBM missing handling",
                "categorical": "Train-only sorted vocabulary; missing=0; unknown=1",
                "target_encoding": False,
            },
            "entity_type": (
                "none"
                if scheme.history_prefix is None
                else ("Entity A" if scheme.history_prefix == "card_" else "Entity C")
            ),
        }
        specifications[scheme.model_id] = {
            **feature_only,
            "feature_columns_hash": canonical_hash(feature_only),
            "scale_pos_weight": scheme.scale_pos_weight,
            "scheme_spec_hash": canonical_hash(
                {**feature_only, "scale_pos_weight": scheme.scale_pos_weight}
            ),
            "role": scheme.role,
        }
    if specifications["M1"]["feature_columns_hash"] != specifications["M2"]["feature_columns_hash"]:
        raise AssertionError("M1 and M2 feature hashes must match")
    return specifications, feature_lists


def fixed_capacity_decision(
    transaction_ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    capacity: float = CAPACITY,
) -> CapacityResult:
    """Select exactly ceil(capacity*n) scores with deterministic TransactionID ties."""
    ids = np.asarray(transaction_ids, dtype=np.int64)
    y_true = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if not (ids.shape == y_true.shape == scores.shape) or ids.size == 0:
        raise ValueError("IDs, labels, and probabilities must be equal non-empty vectors")
    if not np.isfinite(scores).all():
        raise ValueError("Prediction probabilities must be finite")
    if not 0 < capacity <= 1:
        raise ValueError("Capacity must be in (0, 1]")
    alert_count = math.ceil(capacity * len(scores))
    order = np.lexsort((ids, -scores))
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:alert_count]] = True
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    tp = int(np.sum(selected & (y_true == 1)))
    fp = int(np.sum(selected & (y_true == 0)))
    fn = int(np.sum(~selected & (y_true == 1)))
    tn = int(np.sum(~selected & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    baseline_cost = float(FALSE_NEGATIVE_COST * (tp + fn))
    business_cost = float(FALSE_POSITIVE_COST * fp + FALSE_NEGATIVE_COST * fn)
    saving = (baseline_cost - business_cost) / baseline_cost if baseline_cost else 0.0
    alert_rate = alert_count / len(scores)
    return CapacityResult(
        selected=selected,
        score_rank=ranks,
        ranking_boundary_probability=float(scores[order[alert_count - 1]]),
        alert_count=alert_count,
        alert_rate=float(alert_rate),
        capacity_deviation=float(alert_rate - capacity),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision_at_3pct=float(precision),
        recall_at_3pct=float(recall),
        f1_at_3pct=float(f1),
        fraud_captured=tp,
        baseline_cost=baseline_cost,
        business_cost=business_cost,
        cost_saving_rate=float(saving),
    )


def build_experiment_config(
    paths: ExperimentPaths,
    transaction: pd.DataFrame,
    metadata: dict[str, Any],
    protocol: dict[str, Any],
    specifications: dict[str, Any],
) -> dict[str, Any]:
    """Build the immutable experiment specification used by every checkpoint."""
    return {
        "experiment": "IEEE-CIS LightGBM M0-M3 Frozen Experimental Phase",
        "status": "FROZEN_FOR_EXECUTION",
        "creation_time_utc": _reproducible_creation_time(
            paths.transaction_path,
            paths.history_path,
            paths.history_metadata_path,
            paths.protocol_path,
        ),
        "inputs": {
            "transaction_path": str(paths.transaction_path),
            "transaction_sha256": file_hash(paths.transaction_path),
            "history_path": str(paths.history_path),
            "history_sha256": file_hash(paths.history_path),
            "history_metadata_path": str(paths.history_metadata_path),
            "history_metadata_sha256": file_hash(paths.history_metadata_path),
            "protocol_path": str(paths.protocol_path),
            "protocol_sha256": file_hash(paths.protocol_path),
        },
        "dataset": {
            "rows": len(transaction),
            "raw_feature_count": len(transaction.columns) - len(RAW_EXCLUDED),
            "target": "isFraud",
            "time_column": "TransactionDT",
        },
        "fold_definitions": protocol["fold_definitions"],
        "schemes": [asdict(scheme) for scheme in SCHEMES],
        "feature_specifications": specifications,
        "training": {
            "classifier": "lightgbm.LGBMClassifier",
            "fixed_parameters": FIXED_MODEL_PARAMETERS,
            "seeds": list(SEEDS),
            "expected_official_fit_count": 60,
            "actual_official_fit_count_source": "run_metadata.json",
            "train_only": True,
            "validation_usage": "descriptive distribution only",
            "early_stopping": False,
        },
        "decision": {
            "type": "fixed-capacity ranking evaluation",
            "capacity": CAPACITY,
            "alert_count_rule": "ceil(capacity * test_count)",
            "sort_rule": "raw_probability descending, TransactionID ascending",
            "false_positive_cost": FALSE_POSITIVE_COST,
            "false_negative_cost": FALSE_NEGATIVE_COST,
        },
        "restrictions": {
            "hyperparameter_tuning_count": 0,
            "feature_selection_count": 0,
            "shap_count": 0,
            "calibration_count": 0,
            "bootstrap_count": 0,
            "test_threshold_optimization_count": 0,
            "historical_labels_used": False,
        },
        "history_metadata_frozen_sha256": file_hash(paths.history_metadata_path),
        "protocol_fold_definition_sha256": protocol["fold_definition_sha256"],
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lightgbm.__version__,
            "scikit_learn": sklearn.__version__,
            "pyarrow": pa.__version__,
        },
        "history_columns": metadata["columns"],
    }


def run_experiment(paths: ExperimentPaths) -> dict[str, Any]:
    """Execute or resume the exact 60-task experiment and freeze six artifacts."""
    transaction, history, metadata, protocol = load_frozen_inputs(paths)
    folds = indices_from_protocol(transaction, protocol)
    specifications, feature_lists = feature_specifications(transaction, metadata)
    config = build_experiment_config(
        paths, transaction, metadata, protocol, specifications
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = paths.output_dir / "experiment_config.json"
    config_bytes = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if config_path.exists() and config_path.read_bytes() != config_bytes:
        raise RuntimeError("Existing experiment_config.json differs from frozen configuration")
    if not config_path.exists():
        atomic_write_bytes(config_path, config_bytes)
    config_hash = file_hash(config_path)
    protocol_hash = config["inputs"]["protocol_sha256"]

    history_by_id = history.set_index("TransactionID")
    card_columns, combo_columns = history_feature_lists(metadata)
    raw_columns = feature_lists["M0"]
    categorical_columns = [
        column for column in raw_columns if not is_numeric_dtype(transaction[column])
    ]
    numerical_columns = [column for column in raw_columns if column not in categorical_columns]
    preprocessing_rows: list[dict[str, Any]] = []
    official_fits_this_run = 0
    completed_tasks: list[dict[str, Any]] = []

    for fold in folds:
        split_frames = {
            "train": transaction.iloc[fold.train],
            "validation": transaction.iloc[fold.validation],
            "test": transaction.iloc[fold.test],
        }
        encoder = TrainOnlyCategoryEncoder(categorical_columns).fit(split_frames["train"])
        preprocessing_rows.extend(
            encoder.audit_rows(fold.fold, **split_frames)
        )
        encoded_train = encoder.transform(split_frames["train"])
        encoded_test = encoder.transform(split_frames["test"])
        raw_train = _assemble_raw_matrix(
            split_frames["train"], numerical_columns, encoded_train, raw_columns
        )
        raw_test = _assemble_raw_matrix(
            split_frames["test"], numerical_columns, encoded_test, raw_columns
        )
        train_ids = split_frames["train"]["TransactionID"].to_numpy(dtype=np.int64)
        test_ids = split_frames["test"]["TransactionID"].to_numpy(dtype=np.int64)
        y_train = split_frames["train"]["isFraud"].to_numpy(dtype=np.int8)
        y_test = split_frames["test"]["isFraud"].to_numpy(dtype=np.int8)
        test_times = split_frames["test"]["TransactionDT"].to_numpy(dtype=np.int64)
        test_id_hash = array_hash(test_ids)
        history_train = history_by_id.loc[train_ids]
        history_test = history_by_id.loc[test_ids]

        for scheme in SCHEMES:
            scheme_history_columns = (
                []
                if scheme.history_prefix is None
                else (card_columns if scheme.history_prefix == "card_" else combo_columns)
            )
            x_train = _append_history(raw_train, history_train, scheme_history_columns)
            x_test = _append_history(raw_test, history_test, scheme_history_columns)
            expected_columns = feature_lists[scheme.model_id]
            if (
                list(x_train.columns) != expected_columns
                or list(x_test.columns) != expected_columns
            ):
                raise AssertionError(f"{scheme.model_id} assembled feature order changed")
            categorical_for_model = [
                column for column in categorical_columns if column in x_train.columns
            ]
            feature_hash = specifications[scheme.model_id]["feature_columns_hash"]
            for seed in SEEDS:
                task_key = f"fold_{fold.fold}_{scheme.model_id}_seed_{seed}"
                prediction_path, metric_path = _checkpoint_paths(paths.output_dir, task_key)
                resumed = _load_valid_checkpoint(
                    prediction_path=prediction_path,
                    metric_path=metric_path,
                    config_hash=config_hash,
                    protocol_hash=protocol_hash,
                    feature_hash=feature_hash,
                    expected_test_ids=test_ids,
                    expected_test_id_hash=test_id_hash,
                )
                if resumed is not None:
                    completed_tasks.append(resumed)
                    print(f"RESUME {task_key}", flush=True)
                    continue

                model = LGBMClassifier(
                    **FIXED_MODEL_PARAMETERS,
                    scale_pos_weight=scheme.scale_pos_weight,
                    random_state=seed,
                    bagging_seed=seed,
                    feature_fraction_seed=seed,
                    data_random_seed=seed,
                )
                training_start = time.perf_counter()
                model.fit(
                    x_train,
                    y_train,
                    categorical_feature=categorical_for_model,
                )
                training_seconds = time.perf_counter() - training_start
                prediction_start = time.perf_counter()
                prediction_matrix = np.asarray(model.predict_proba(x_test), dtype=np.float64)
                probabilities = prediction_matrix[:, 1]
                prediction_seconds = time.perf_counter() - prediction_start
                official_fits_this_run += 1
                probability_hash = array_hash(probabilities)
                decision = fixed_capacity_decision(test_ids, y_test, probabilities)
                ledger = pd.DataFrame(
                    {
                        "TransactionID": test_ids,
                        "TransactionDT": test_times,
                        "isFraud": y_test,
                        "fold": np.full(len(test_ids), fold.fold, dtype=np.int8),
                        "model_id": scheme.model_id,
                        "seed": np.full(len(test_ids), seed, dtype=np.int16),
                        "raw_probability": probabilities,
                        "score_rank": decision.score_rank,
                        "selected_at_3pct": decision.selected,
                        "rank_percentile": decision.score_rank / len(test_ids),
                        "feature_set_hash": feature_hash,
                        "protocol_hash": protocol_hash,
                    }
                )
                atomic_write_parquet(ledger, prediction_path)
                metric = {
                    "fold": fold.fold,
                    "model_id": scheme.model_id,
                    "model_role": scheme.role,
                    "seed": seed,
                    "feature_set": scheme.feature_set,
                    "feature_set_hash": feature_hash,
                    "scheme_spec_hash": specifications[scheme.model_id]["scheme_spec_hash"],
                    "scale_pos_weight": scheme.scale_pos_weight,
                    "train_count": len(fold.train),
                    "validation_count": len(fold.validation),
                    "test_count": len(fold.test),
                    "train_fraud_rate": float(y_train.mean()),
                    "validation_fraud_rate": float(
                        split_frames["validation"]["isFraud"].mean()
                    ),
                    "test_fraud_rate": float(y_test.mean()),
                    "pr_auc": float(average_precision_score(y_test, probabilities)),
                    "roc_auc": float(roc_auc_score(y_test, probabilities)),
                    **{
                        key: value
                        for key, value in vars(decision).items()
                        if key not in {"selected", "score_rank"}
                    },
                    "capacity": CAPACITY,
                    "false_positive_cost": FALSE_POSITIVE_COST,
                    "false_negative_cost": FALSE_NEGATIVE_COST,
                    "training_seconds": float(training_seconds),
                    "prediction_seconds": float(prediction_seconds),
                    "protocol_hash": protocol_hash,
                    "config_hash": config_hash,
                    "test_transaction_id_hash": test_id_hash,
                    "raw_probability_hash": probability_hash,
                    "train_vocabulary_hash": encoder.vocabulary_hash(),
                }
                atomic_write_json(metric, metric_path)
                completed_tasks.append(metric)
                del model, probabilities, ledger
                print(
                    f"DONE {task_key} pr_auc={metric['pr_auc']:.6f} "
                    f"recall@3%={metric['recall_at_3pct']:.6f}",
                    flush=True,
                )
            del x_train, x_test
        del raw_train, raw_test, encoded_train, encoded_test, history_train, history_test

    metrics = pd.DataFrame(completed_tasks).sort_values(
        ["fold", "model_id", "seed"], kind="stable"
    ).reset_index(drop=True)
    validate_completed_metrics(metrics)
    _validate_checkpoint_alignment(paths.output_dir, folds)
    _write_final_outputs(
        paths=paths,
        metrics=metrics,
        preprocessing_rows=preprocessing_rows,
        specifications=specifications,
        config=config,
        config_hash=config_hash,
        official_fits_this_run=official_fits_this_run,
    )
    return {
        "total_completed_tasks": len(metrics),
        "official_fits_this_run": official_fits_this_run,
        "resumed_tasks": len(metrics) - official_fits_this_run,
        "config_hash": config_hash,
        "output_dir": str(paths.output_dir),
    }


def validate_completed_metrics(metrics: pd.DataFrame) -> None:
    """Require the exact 3x4x5 official result grid and fixed alert counts."""
    required = {"fold", "model_id", "seed", "alert_count"}
    _require_columns(metrics.columns, required, "completed metrics")
    if len(metrics) != 60:
        raise RuntimeError(f"Expected exactly 60 completed tasks, found {len(metrics)}")
    if metrics[["fold", "model_id", "seed"]].duplicated().any():
        raise RuntimeError("Duplicate Fold/Model/Seed task records detected")
    if set(metrics["fold"]) != {1, 2, 3} or set(metrics["model_id"]) != set(MODEL_ORDER):
        raise RuntimeError("Fold or model grid differs from frozen experiment")
    counts = metrics.groupby(["fold", "model_id"])["seed"].agg(["nunique", lambda x: set(x)])
    complete_seeds = all(value == set(SEEDS) for value in counts["<lambda_0>"])
    if not (counts["nunique"] == 5).all() or not complete_seeds:
        raise RuntimeError("Every Fold/Model must contain exactly the five frozen seeds")
    alerts = metrics.groupby("fold")["alert_count"].nunique()
    if not (alerts == 1).all():
        raise RuntimeError("Fixed-capacity alert count differs within a fold")


def fold_metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all five seeds without selecting a best run."""
    metric_columns = [
        "pr_auc",
        "roc_auc",
        "precision_at_3pct",
        "recall_at_3pct",
        "f1_at_3pct",
        "business_cost",
        "cost_saving_rate",
    ]
    aggregated = metrics.groupby(["fold", "model_id", "model_role"], sort=True)[
        metric_columns
    ].agg(["mean", "std", "min", "max"])
    aggregated.columns = [f"{metric}_{stat}" for metric, stat in aggregated.columns]
    return aggregated.reset_index()


def _write_final_outputs(
    *,
    paths: ExperimentPaths,
    metrics: pd.DataFrame,
    preprocessing_rows: list[dict[str, Any]],
    specifications: dict[str, Any],
    config: dict[str, Any],
    config_hash: str,
    official_fits_this_run: int,
) -> None:
    seed_metrics_path = paths.output_dir / "seed_metrics.csv"
    fold_metrics_path = paths.output_dir / "fold_metrics.csv"
    business_path = paths.output_dir / "business_cost_results.csv"
    prediction_path = paths.output_dir / "prediction_scores.parquet"
    atomic_write_csv(metrics, seed_metrics_path)
    atomic_write_csv(fold_metric_summary(metrics), fold_metrics_path)
    business_columns = [
        "fold",
        "model_id",
        "seed",
        "capacity",
        "false_positive_cost",
        "false_negative_cost",
        "alert_count",
        "ranking_boundary_probability",
        "alert_rate",
        "capacity_deviation",
        "tp",
        "fp",
        "tn",
        "fn",
        "business_cost",
        "baseline_cost",
        "cost_saving_rate",
    ]
    atomic_write_csv(metrics[business_columns], business_path)
    _combine_prediction_checkpoints(paths.output_dir, prediction_path)
    atomic_write_json(specifications, paths.output_dir / "feature_specifications.json")
    atomic_write_csv(
        pd.DataFrame(preprocessing_rows), paths.output_dir / "preprocessing_audit.csv"
    )
    run_metadata = {
        "status": "PASS — IEEE-CIS M0–M3 experimental phase frozen",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "config_hash": config_hash,
        "expected_official_fit_count": 60,
        "actual_official_fit_count": 60,
        "official_fits_this_process": official_fits_this_run,
        "resumed_checkpoint_count": 60 - official_fits_this_run,
        "prediction_row_count": int(sum(metrics["test_count"])),
        "restrictions": config["restrictions"],
        "output_hashes": {},
    }
    report_path = paths.output_dir / "IEEE_CIS_lightgbm_report.md"
    _write_experiment_report(report_path, metrics, fold_metric_summary(metrics), config)
    primary_outputs = [paths.output_dir / name for name in REQUIRED_OUTPUTS]
    run_metadata["output_hashes"] = {
        path.name: file_hash(path) for path in primary_outputs
    }
    atomic_write_json(run_metadata, paths.output_dir / "run_metadata.json")


def _write_experiment_report(
    path: Path,
    metrics: pd.DataFrame,
    fold_summary: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    overall = (
        metrics.groupby(["model_id", "model_role"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            recall_at_3pct_mean=("recall_at_3pct", "mean"),
            recall_at_3pct_std=("recall_at_3pct", "std"),
            business_cost_mean=("business_cost", "mean"),
            cost_saving_rate_mean=("cost_saving_rate", "mean"),
        )
        .sort_values("model_id")
    )
    comparisons = _descriptive_comparisons(metrics)
    lines = [
        "# IEEE-CIS LightGBM M0–M3 Frozen Experimental Report",
        "",
        "## 1. Frozen protocol",
        "",
        f"- Protocol SHA-256: `{config['inputs']['protocol_sha256']}`",
        "- Three stored expanding-window folds were read directly; no boundary was recomputed.",
        "- Validation was retained for temporal structure and distribution description only.",
        "",
        "## 2. Dataset and input hashes",
        "",
        f"- Transaction rows: {config['dataset']['rows']:,}",
        f"- Transaction SHA-256: `{config['inputs']['transaction_sha256']}`",
        f"- History Parquet SHA-256: `{config['inputs']['history_sha256']}`",
        f"- History metadata SHA-256: `{config['inputs']['history_metadata_sha256']}`",
        "",
        "## 3. M0–M3 definitions",
        "",
        *[
            f"- {scheme.model_id}: {scheme.feature_set}; scale_pos_weight="
            f"{scheme.scale_pos_weight:g}; role={scheme.role}."
            for scheme in SCHEMES
        ],
        "",
        "M2 remained the pre-specified primary framework. M3 remained an entity-granularity "
        "sensitivity analysis regardless of observed performance.",
        "",
        "## 4. Feature definitions and Train-only preprocessing",
        "",
        f"- Raw features: {config['dataset']['raw_feature_count']}.",
        "- Numeric values used float32 with infinities mapped to NaN; no global imputation.",
        "- Categorical vocabularies were learned separately from each Fold's Train only.",
        "- Missing and unseen categories used fixed distinct codes; no target encoding.",
        "",
        "## 5. Fixed LightGBM parameters and Seed design",
        "",
        "```json",
        json.dumps(config["training"]["fixed_parameters"], indent=2),
        "```",
        "",
        f"Seeds: {config['training']['seeds']}. Exactly 60 official fits were completed.",
        "",
        "## 6. Fold-level five-seed results",
        "",
        _markdown_table(fold_summary),
        "",
        "## 7. Overall descriptive results",
        "",
        _markdown_table(overall),
        "",
        "## 8. Pre-registered descriptive comparisons",
        "",
        _markdown_table(comparisons),
        "",
        "These are descriptive paired differences over the same Fold and Seed. No formal "
        "external-validation conclusion or bootstrap significance claim is made here.",
        "",
        "## 9. Capacity-constrained decision and business cost",
        "",
        "Every Test window selected exactly ceil(0.03 × test rows) transactions by descending "
        "raw score with TransactionID as the deterministic tie-break. Labels were used only "
        "after selection. Business Cost = FP + 100 × FN; the no-review baseline is 100 × fraud.",
        "",
        "## 10. Temporal heterogeneity observations",
        "",
        "Fold-specific results are retained without deleting unfavorable windows. Differences "
        "across the three future windows are descriptive and are not claimed to establish a "
        "causal concept-drift mechanism.",
        "",
        "## 11. Limitations",
        "",
        "IEEE-CIS fields are anonymized; card1 is not asserted to be a real account. The three "
        "rolling folds overlap through expanding training windows, and five seeds are repeated "
        "stochastic fits rather than independent future environments. The 3% capacity and 100:1 "
        "cost ratio are pre-specified operational assumptions.",
        "",
        "## 12. Methodological compliance statement",
        "",
        "No hyperparameter search, feature selection, early stopping, calibration, SHAP, "
        "bootstrap, validation-driven selection, or test-driven modification was performed.",
        "",
        "**PASS — IEEE-CIS M0–M3 experimental phase frozen.**",
    ]
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def _descriptive_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    definitions = (
        ("M0_vs_M1", "M0", "M1"),
        ("M1_vs_M2", "M1", "M2"),
        ("M2_vs_M3", "M2", "M3"),
        ("M0_vs_M2", "M0", "M2"),
    )
    rows: list[dict[str, Any]] = []
    indexed = metrics.set_index(["fold", "seed", "model_id"])
    for comparison, baseline, candidate in definitions:
        for fold in (1, 2, 3):
            effects = []
            for seed in SEEDS:
                left = indexed.loc[(fold, seed, baseline)]
                right = indexed.loc[(fold, seed, candidate)]
                effects.append(
                    {
                        "pr_auc": right["pr_auc"] - left["pr_auc"],
                        "recall_at_3pct": right["recall_at_3pct"] - left["recall_at_3pct"],
                        "cost_reduction": left["business_cost"] - right["business_cost"],
                    }
                )
            effect_frame = pd.DataFrame(effects)
            rows.append(
                {
                    "comparison": comparison,
                    "fold": fold,
                    "delta_pr_auc_mean": effect_frame["pr_auc"].mean(),
                    "delta_recall_at_3pct_mean": effect_frame["recall_at_3pct"].mean(),
                    "cost_reduction_mean": effect_frame["cost_reduction"].mean(),
                }
            )
    return pd.DataFrame(rows)


def _assemble_raw_matrix(
    frame: pd.DataFrame,
    numerical_columns: Sequence[str],
    encoded_categories: pd.DataFrame,
    ordered_columns: Sequence[str],
) -> pd.DataFrame:
    numeric = frame[list(numerical_columns)].astype(np.float32).replace(
        [np.inf, -np.inf], np.nan
    )
    matrix = pd.concat([numeric, encoded_categories], axis=1)
    return matrix.loc[:, list(ordered_columns)]


def _append_history(
    raw: pd.DataFrame,
    history: pd.DataFrame,
    history_columns: Sequence[str],
) -> pd.DataFrame:
    if not history_columns:
        return raw
    additions = history[list(history_columns)].astype(np.float32).reset_index(drop=True)
    raw_aligned = raw.reset_index(drop=True)
    return pd.concat([raw_aligned, additions], axis=1)


def _checkpoint_paths(output_dir: Path, task_key: str) -> tuple[Path, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    return (
        checkpoint_dir / "predictions" / f"{task_key}.parquet",
        checkpoint_dir / "metrics" / f"{task_key}.json",
    )


def _load_valid_checkpoint(
    *,
    prediction_path: Path,
    metric_path: Path,
    config_hash: str,
    protocol_hash: str,
    feature_hash: str,
    expected_test_ids: np.ndarray,
    expected_test_id_hash: str,
) -> dict[str, Any] | None:
    if not prediction_path.exists() and not metric_path.exists():
        return None
    if not prediction_path.exists() or not metric_path.exists():
        raise RuntimeError(f"Incomplete checkpoint pair: {metric_path.stem}")
    loaded_metric = json.loads(metric_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_metric, dict):
        raise RuntimeError(f"Checkpoint {metric_path.stem} metric payload is not an object")
    metric: dict[str, Any] = loaded_metric
    expected = {
        "config_hash": config_hash,
        "protocol_hash": protocol_hash,
        "feature_set_hash": feature_hash,
        "test_transaction_id_hash": expected_test_id_hash,
    }
    for key, value in expected.items():
        if metric.get(key) != value:
            raise RuntimeError(f"Checkpoint {metric_path.stem} has mismatched {key}")
    ledger = pd.read_parquet(
        prediction_path,
        columns=["TransactionID", "raw_probability"],
        engine="pyarrow",
    )
    if not np.array_equal(ledger["TransactionID"].to_numpy(), expected_test_ids):
        raise RuntimeError(f"Checkpoint {metric_path.stem} Test IDs changed")
    probability_hash = array_hash(ledger["raw_probability"].to_numpy(dtype=np.float64))
    if probability_hash != metric["raw_probability_hash"]:
        raise RuntimeError(f"Checkpoint {metric_path.stem} probability hash changed")
    return metric


def _validate_checkpoint_alignment(output_dir: Path, folds: Sequence[FoldIndices]) -> None:
    for fold in folds:
        expected: np.ndarray | None = None
        for seed in SEEDS:
            for model_id in MODEL_ORDER:
                prediction_path, _ = _checkpoint_paths(
                    output_dir, f"fold_{fold.fold}_{model_id}_seed_{seed}"
                )
                ids = pd.read_parquet(
                    prediction_path, columns=["TransactionID"], engine="pyarrow"
                )["TransactionID"].to_numpy()
                if expected is None:
                    expected = ids
                elif not np.array_equal(expected, ids):
                    raise RuntimeError(f"Fold {fold.fold} prediction checkpoint alignment failed")


def _combine_prediction_checkpoints(output_dir: Path, final_path: Path) -> None:
    temporary = final_path.with_suffix(".tmp.parquet")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        temporary.unlink()
    writer: pq.ParquetWriter | None = None
    try:
        for fold in (1, 2, 3):
            for model_id in MODEL_ORDER:
                for seed in SEEDS:
                    path, _ = _checkpoint_paths(
                        output_dir, f"fold_{fold}_{model_id}_seed_{seed}"
                    )
                    table = pq.read_table(path)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary, table.schema, compression="zstd"
                        )
                    writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if not temporary.exists():
        raise RuntimeError("No prediction checkpoints were combined")
    os.replace(temporary, final_path)


def atomic_write_json(value: Any, path: Path) -> None:
    atomic_write_bytes(
        path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
    os.replace(temporary, path)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reproducible_creation_time(*paths: Path) -> str:
    timestamp = max(path.stat().st_mtime for path in paths)
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


def _require_columns(
    actual: Iterable[str], required: Iterable[str], dataset: str
) -> None:
    missing = sorted(set(required).difference(actual))
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")


def _validate_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{name} is empty: {path}")


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


def build_parser() -> argparse.ArgumentParser:
    """Build the frozen experiment command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction_path", type=Path, required=True)
    parser.add_argument("--history_path", type=Path, required=True)
    parser.add_argument("--history_metadata_path", type=Path, required=True)
    parser.add_argument("--protocol_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def main() -> None:
    """Run or resume the frozen official experiment."""
    args = build_parser().parse_args()
    result = run_experiment(
        ExperimentPaths(
            transaction_path=args.transaction_path.resolve(),
            history_path=args.history_path.resolve(),
            history_metadata_path=args.history_metadata_path.resolve(),
            protocol_path=args.protocol_path.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
