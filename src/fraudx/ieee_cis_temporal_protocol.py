"""Deterministic IEEE-CIS rolling temporal protocol construction and auditing."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudx.data import TemporalSplit, rolling_temporal_splits

DATASET_NAME = "IEEE-CIS Fraud Detection"
TIME_COLUMN = "TransactionDT"
TARGET_COLUMN = "isFraud"
FOLD_COUNT = 3
INITIAL_TRAIN_FRACTION = 0.40
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
BOUNDARY_DECLARATION = (
    "Temporal boundaries were determined before model training and without observing "
    "model performance."
)
RAW_COLUMNS = (
    "TransactionID",
    TIME_COLUMN,
    "TransactionAmt",
    "ProductCD",
    TARGET_COLUMN,
)
FEATURE_COLUMNS = (
    "TransactionID",
    TIME_COLUMN,
    "card_entity_available",
    "card_has_history",
    "combo_entity_available",
    "combo_has_history",
)
OUTPUT_FILENAMES = (
    "ieee_cis_temporal_protocol.json",
    "temporal_fold_summary.csv",
    "temporal_distribution_audit.csv",
    "entity_history_availability_by_fold.csv",
    "cold_start_audit.csv",
    "IEEE_CIS_temporal_protocol_report.md",
)
PROHIBITED_IMPORT_ROOTS = {"lightgbm", "xgboost", "sklearn"}
PROHIBITED_CALL_NAMES = {"fit", "fit_predict", "predict", "predict_proba"}


@dataclass(frozen=True)
class TemporalProtocolArtifacts:
    """Paths returned after a successful protocol freeze."""

    protocol_path: Path
    fold_summary_path: Path
    distribution_audit_path: Path
    history_availability_path: Path
    cold_start_path: Path
    report_path: Path


def build_temporal_folds(frame: pd.DataFrame) -> list[TemporalSplit]:
    """Create three expanding folds using only complete TransactionDT groups."""
    _require_columns(frame.columns, ("TransactionID", TIME_COLUMN), "transaction frame")
    if frame.empty:
        raise ValueError("Transaction frame is empty")
    if frame["TransactionID"].isna().any() or not frame["TransactionID"].is_unique:
        raise ValueError("TransactionID must be unique and non-missing")
    if frame[TIME_COLUMN].isna().any():
        raise ValueError("TransactionDT must be non-missing")
    boundary_frame = frame.loc[:, ["TransactionID", TIME_COLUMN]]
    return rolling_temporal_splits(
        boundary_frame,
        folds=FOLD_COUNT,
        initial_train_fraction=INITIAL_TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
        time_column=TIME_COLUMN,
    )


def fold_boundaries(folds: Sequence[TemporalSplit]) -> list[dict[str, Any]]:
    """Return serializable temporal boundaries and row counts."""
    rows: list[dict[str, Any]] = []
    for fold_number, fold in enumerate(folds, start=1):
        definition: dict[str, Any] = {"fold": fold_number}
        for split_name, split_ids in _split_parts(fold):
            definition[split_name] = {
                "start_transaction_dt": _native(split_ids[TIME_COLUMN].min()),
                "end_transaction_dt": _native(split_ids[TIME_COLUMN].max()),
                "transaction_count": len(split_ids),
            }
        rows.append(definition)
    return rows


def make_fold_summary(
    folds: Sequence[TemporalSplit], transaction: pd.DataFrame
) -> pd.DataFrame:
    """Create the requested Fold/Split temporal and label summary."""
    rows: list[dict[str, Any]] = []
    indexed = transaction.set_index("TransactionID")
    for fold_number, fold in enumerate(folds, start=1):
        for split_name, split_ids in _split_parts(fold):
            split = indexed.loc[split_ids["TransactionID"]]
            rows.append(
                {
                    "fold": fold_number,
                    "split": split_name,
                    "start_transaction_dt": _native(split[TIME_COLUMN].min()),
                    "end_transaction_dt": _native(split[TIME_COLUMN].max()),
                    "transaction_count": len(split),
                    "fraud_count": int(split[TARGET_COLUMN].sum()),
                    "fraud_rate": float(split[TARGET_COLUMN].mean()),
                }
            )
    return pd.DataFrame(rows)


def make_distribution_audit(
    folds: Sequence[TemporalSplit], transaction: pd.DataFrame
) -> pd.DataFrame:
    """Audit amounts and ProductCD composition for every Fold/Split."""
    rows: list[dict[str, Any]] = []
    indexed = transaction.set_index("TransactionID")
    for fold_number, fold in enumerate(folds, start=1):
        for split_name, split_ids in _split_parts(fold):
            split = indexed.loc[split_ids["TransactionID"]]
            amount = split["TransactionAmt"].astype(float)
            product = split["ProductCD"].fillna("<MISSING>").astype(str)
            counts = product.value_counts(sort=False).sort_index()
            distribution = {
                key: {
                    "count": int(value),
                    "ratio": float(value / len(split)),
                }
                for key, value in counts.items()
            }
            rows.append(
                {
                    "fold": fold_number,
                    "split": split_name,
                    "transaction_count": len(split),
                    "fraud_rate": float(split[TARGET_COLUMN].mean()),
                    "transaction_amt_mean": float(amount.mean()),
                    "transaction_amt_median": float(amount.median()),
                    "transaction_amt_p90": float(amount.quantile(0.90)),
                    "product_cd_distribution": json.dumps(
                        distribution, sort_keys=True, separators=(",", ":")
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_history_availability_audit(
    folds: Sequence[TemporalSplit], features: pd.DataFrame
) -> pd.DataFrame:
    """Report Entity A/C history and no-history ratios by Fold/Split."""
    rows: list[dict[str, Any]] = []
    indexed = features.set_index("TransactionID")
    for fold_number, fold in enumerate(folds, start=1):
        for split_name, split_ids in _split_parts(fold):
            split = indexed.loc[split_ids["TransactionID"]]
            for entity, prefix in (("Entity A", "card"), ("Entity C", "combo")):
                available = split[f"{prefix}_entity_available"].eq(1)
                has_history = split[f"{prefix}_has_history"].eq(1)
                eligible_count = int(available.sum())
                history_count = int((available & has_history).sum())
                no_history_count = int((available & ~has_history).sum())
                rows.append(
                    {
                        "fold": fold_number,
                        "split": split_name,
                        "entity": entity,
                        "transaction_count": len(split),
                        "entity_available_count": eligible_count,
                        "entity_availability_rate": float(available.mean()),
                        "has_history_count": history_count,
                        "has_history_ratio": _safe_ratio(history_count, eligible_count),
                        "no_history_count": no_history_count,
                        "no_history_ratio": _safe_ratio(no_history_count, eligible_count),
                    }
                )
    return pd.DataFrame(rows)


def make_cold_start_audit(
    folds: Sequence[TemporalSplit], features: pd.DataFrame
) -> pd.DataFrame:
    """Report first-seen entity rates in future Validation and Test segments."""
    history = make_history_availability_audit(folds, features)
    future = history.loc[history["split"].isin(["validation", "test"])].copy()
    future = future.rename(
        columns={
            "no_history_count": "first_seen_entity_transaction_count",
            "no_history_ratio": "first_seen_entity_transaction_ratio",
        }
    )
    return future[
        [
            "fold",
            "split",
            "entity",
            "transaction_count",
            "entity_available_count",
            "first_seen_entity_transaction_count",
            "first_seen_entity_transaction_ratio",
        ]
    ].reset_index(drop=True)


def validate_protocol(
    folds: Sequence[TemporalSplit], transaction: pd.DataFrame
) -> pd.DataFrame:
    """Run structural acceptance checks without using labels to select boundaries."""
    rows: list[dict[str, Any]] = []
    previous_train_end: int | float | None = None
    previous_validation_end: int | float | None = None
    previous_test_end: int | float | None = None
    for fold_number, fold in enumerate(folds, start=1):
        train_max = fold.train[TIME_COLUMN].max()
        validation_min = fold.validation[TIME_COLUMN].min()
        validation_max = fold.validation[TIME_COLUMN].max()
        test_min = fold.test[TIME_COLUMN].min()
        test_max = fold.test[TIME_COLUMN].max()
        rows.append(
            _check(
                f"fold_{fold_number}_strict_time_order",
                train_max < validation_min and validation_max < test_min,
                f"{train_max} < {validation_min}; {validation_max} < {test_min}",
            )
        )
        time_sets = [set(part[TIME_COLUMN]) for _, part in _split_parts(fold)]
        isolated = not any(
            time_sets[left] & time_sets[right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
        rows.append(
            _check(
                f"fold_{fold_number}_timestamp_isolation",
                isolated,
                "Each TransactionDT occurs in one split only",
            )
        )
        if previous_train_end is not None:
            future_direction = bool(
                train_max > previous_train_end
                and validation_max > previous_validation_end
                and test_max > previous_test_end
            )
            rows.append(
                _check(
                    f"fold_{fold_number}_future_direction",
                    future_direction,
                    "Train, Validation, and Test endpoints move forward",
                )
            )
        previous_train_end = train_max
        previous_validation_end = validation_max
        previous_test_end = test_max
    final_ids = set(
        pd.concat([folds[-1].train, folds[-1].validation, folds[-1].test])[
            "TransactionID"
        ]
    )
    rows.append(
        _check(
            "final_fold_covers_dataset",
            final_ids == set(transaction["TransactionID"]),
            f"covered={len(final_ids)}; input={len(transaction)}",
        )
    )
    return pd.DataFrame(rows)


def audit_source_model_independence(source_path: Path) -> list[str]:
    """Return prohibited imports or estimator-like calls found in the builder source."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in PROHIBITED_IMPORT_ROOTS:
                    violations.append(f"import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in PROHIBITED_IMPORT_ROOTS:
                violations.append(f"import:{node.module}")
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in PROHIBITED_CALL_NAMES:
                violations.append(f"call:{call_name}@{node.lineno}")
    return violations


def run_temporal_protocol_builder(
    transaction_path: Path,
    feature_path: Path,
    output_dir: Path,
) -> TemporalProtocolArtifacts:
    """Build, validate, and persist the complete frozen protocol package."""
    transaction_path = transaction_path.resolve()
    feature_path = feature_path.resolve()
    output_dir = output_dir.resolve()
    _validate_input_file(transaction_path, "train_transaction.csv")
    _validate_input_file(feature_path, "history feature Parquet")

    transaction = pd.read_csv(transaction_path, usecols=list(RAW_COLUMNS), low_memory=False)
    features = pd.read_parquet(feature_path, columns=list(FEATURE_COLUMNS), engine="pyarrow")
    _validate_inputs(transaction, features)
    folds = build_temporal_folds(transaction)
    checks = validate_protocol(folds, transaction)
    if not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"Temporal protocol checks failed: {failed}")
    source_violations = audit_source_model_independence(Path(__file__))
    if source_violations:
        raise RuntimeError(f"Model-independence audit failed: {source_violations}")

    fold_summary = make_fold_summary(folds, transaction)
    distribution_audit = make_distribution_audit(folds, transaction)
    history_availability = make_history_availability_audit(folds, features)
    cold_start = make_cold_start_audit(folds, features)
    input_hashes = {
        "train_transaction_sha256": _file_hash(transaction_path),
        "history_feature_parquet_sha256": _file_hash(feature_path),
    }
    boundaries = fold_boundaries(folds)
    deterministic_core = {
        "dataset_name": DATASET_NAME,
        "input_hashes": input_hashes,
        "fold_definitions": boundaries,
    }
    fold_definition_hash = _json_hash(deterministic_core)
    protocol = _make_protocol(
        transaction_path=transaction_path,
        feature_path=feature_path,
        input_hashes=input_hashes,
        boundaries=boundaries,
        fold_definition_hash=fold_definition_hash,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = TemporalProtocolArtifacts(
        protocol_path=output_dir / OUTPUT_FILENAMES[0],
        fold_summary_path=output_dir / OUTPUT_FILENAMES[1],
        distribution_audit_path=output_dir / OUTPUT_FILENAMES[2],
        history_availability_path=output_dir / OUTPUT_FILENAMES[3],
        cold_start_path=output_dir / OUTPUT_FILENAMES[4],
        report_path=output_dir / OUTPUT_FILENAMES[5],
    )
    artifacts.protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fold_summary.to_csv(artifacts.fold_summary_path, index=False)
    distribution_audit.to_csv(artifacts.distribution_audit_path, index=False)
    history_availability.to_csv(artifacts.history_availability_path, index=False)
    cold_start.to_csv(artifacts.cold_start_path, index=False)
    _write_report(
        artifacts.report_path,
        protocol=protocol,
        fold_summary=fold_summary,
        distribution_audit=distribution_audit,
        history_availability=history_availability,
        cold_start=cold_start,
        checks=checks,
    )
    return artifacts


def _make_protocol(
    *,
    transaction_path: Path,
    feature_path: Path,
    input_hashes: dict[str, str],
    boundaries: list[dict[str, Any]],
    fold_definition_hash: str,
) -> dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "creation_time_utc": _reproducible_creation_time(transaction_path, feature_path),
        "creation_time_basis": (
            "Latest frozen-input modification time, used instead of wall-clock build time so "
            "identical inputs produce byte-identical protocol artifacts."
        ),
        "input_files": {
            "transaction_path": str(transaction_path),
            "feature_path": str(feature_path),
            **input_hashes,
        },
        "fold_definition_sha256": fold_definition_hash,
        "split_rule": {
            "method": "expanding-window rolling validation on complete TransactionDT groups",
            "fold_count": FOLD_COUNT,
            "initial_train_fraction": INITIAL_TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": TEST_FRACTION,
            "later_fold_shift_fraction": 0.20,
            "boundary_selection_inputs": ["TransactionDT", "time-group transaction counts"],
            "row_number_split": False,
            "label_used_for_boundary_selection": False,
        },
        "fold_definitions": boundaries,
        "timestamp_isolation_rule": (
            "Transactions sharing the same TransactionDT always belong to the same split and "
            "cannot contribute history to one another."
        ),
        "history_continuation_rule": {
            "train": "Only strictly earlier Train transactions are available.",
            "validation": (
                "Earlier Train transactions and strictly earlier Validation transactions are "
                "available; future Validation and all Test transactions are unavailable."
            ),
            "test": (
                "All earlier Train and Validation transactions plus strictly earlier Test "
                "transactions are available; future Test transactions are unavailable."
            ),
            "same_timestamp_history_contribution": False,
            "historical_labels_used": False,
        },
        "leakage_prevention_statement": (
            "All boundaries use TransactionDT and complete time-group counts only. isFraud is "
            "used after freezing solely for descriptive summaries. No future timestamp or "
            "same-timestamp transaction contributes historical state."
        ),
        "boundary_independence_statement": BOUNDARY_DECLARATION,
        "model_training_performed": False,
        "next_stage": "IEEE-CIS LightGBM Experimental Phase (not started)",
    }


def _write_report(
    path: Path,
    *,
    protocol: dict[str, Any],
    fold_summary: pd.DataFrame,
    distribution_audit: pd.DataFrame,
    history_availability: pd.DataFrame,
    cold_start: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    boundaries = fold_summary[
        [
            "fold",
            "split",
            "start_transaction_dt",
            "end_transaction_dt",
            "transaction_count",
            "fraud_count",
            "fraud_rate",
        ]
    ]
    distributions = distribution_audit.drop(columns="product_cd_distribution")
    lines = [
        "# IEEE-CIS Temporal Protocol Report",
        "",
        "## 1. 数据说明",
        "",
        f"- Dataset: {protocol['dataset_name']}",
        f"- Protocol snapshot time (UTC): {protocol['creation_time_utc']}",
        f"- Snapshot time basis: {protocol['creation_time_basis']}",
        f"- train_transaction.csv SHA-256: `{protocol['input_files']['train_transaction_sha256']}`",
        "- Entity history Parquet SHA-256: "
        f"`{protocol['input_files']['history_feature_parquet_sha256']}`",
        "- `TransactionDT` is relative time and is not interpreted as a calendar date.",
        "",
        "## 2. 时间切分方法",
        "",
        "Three expanding-window folds were selected from the ordered distribution of complete "
        "`TransactionDT` groups. Target row proportions are 40% initial Train, 10% Validation, "
        "and 10% Test, followed by 20-percentage-point forward shifts. Boundaries are placed at "
        "the nearest complete time-group boundary, never at an arbitrary row number.",
        "",
        protocol["boundary_independence_statement"],
        "",
        "## 3. Fold边界表",
        "",
        _markdown_table(boundaries),
        "",
        "## 4. 数据分布",
        "",
        _markdown_table(distributions),
        "",
        "Complete ProductCD count/ratio distributions are stored as deterministic JSON objects "
        "in `temporal_distribution_audit.csv`.",
        "",
        "## 5. Entity历史可用性",
        "",
        _markdown_table(history_availability),
        "",
        "The has-history and no-history ratios use available entities as the denominator. "
        "Unavailable Entity C rows are reported separately through entity availability.",
        "",
        "## 6. 冷启动分析",
        "",
        _markdown_table(cold_start),
        "",
        "A cold-start transaction has an available entity but no transaction with strictly "
        "earlier `TransactionDT`. Only future Validation and Test segments are included.",
        "",
        "## 7. 泄漏防护规则",
        "",
        "- Train uses only strictly earlier Train history.",
        "- Validation may use earlier Train and strictly earlier Validation history.",
        "- Test may use earlier Train/Validation and strictly earlier Test history.",
        "- Equal-timestamp transactions never contribute history to one another.",
        "- Future information and historical labels are excluded from history features.",
        "- `isFraud` is read only after boundaries are frozen, for descriptive audit columns.",
        "",
        "## 8. 边界未参考模型结果声明",
        "",
        protocol["boundary_independence_statement"],
        "",
        "No estimator training, score generation, feature selection, explanation analysis, or "
        "hyperparameter search was performed in this stage.",
        "",
        "## 9. 自动验收",
        "",
        _markdown_table(checks),
        "",
        f"- Deterministic fold-definition SHA-256: `{protocol['fold_definition_sha256']}`",
        "- Status: **PASS — protocol frozen.**",
        "- Next stage is recorded but not started.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_inputs(transaction: pd.DataFrame, features: pd.DataFrame) -> None:
    _require_columns(transaction.columns, RAW_COLUMNS, "train_transaction.csv")
    _require_columns(features.columns, FEATURE_COLUMNS, "history feature Parquet")
    for frame, name in ((transaction, "transaction"), (features, "feature")):
        if frame["TransactionID"].isna().any() or not frame["TransactionID"].is_unique:
            raise ValueError(f"{name} TransactionID must be unique and non-missing")
    if set(transaction["TransactionID"]) != set(features["TransactionID"]):
        raise ValueError("Transaction and feature TransactionID sets differ")
    aligned = transaction[["TransactionID", TIME_COLUMN]].merge(
        features[["TransactionID", TIME_COLUMN]],
        on="TransactionID",
        suffixes=("_transaction", "_feature"),
        validate="one_to_one",
    )
    if not np.array_equal(
        aligned[f"{TIME_COLUMN}_transaction"], aligned[f"{TIME_COLUMN}_feature"]
    ):
        raise ValueError("TransactionDT differs between transaction and feature inputs")


def _split_parts(fold: TemporalSplit) -> tuple[tuple[str, pd.DataFrame], ...]:
    return (("train", fold.train), ("validation", fold.validation), ("test", fold.test))


def _call_name(function: ast.expr) -> str | None:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _check(name: str, passed: bool, details: str) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "details": details}


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reproducible_creation_time(*paths: Path) -> str:
    timestamp = max(path.stat().st_mtime for path in paths)
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(actual: Sequence[str], required: Sequence[str], dataset: str) -> None:
    missing = sorted(set(required).difference(actual))
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")


def _validate_input_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{name} is empty: {path}")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in values) + " |")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NaN"
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|")


def build_parser() -> argparse.ArgumentParser:
    """Create the standalone protocol-builder CLI parser."""
    parser = argparse.ArgumentParser(
        description="Build the frozen IEEE-CIS rolling temporal validation protocol"
    )
    parser.add_argument("--transaction_path", type=Path, required=True)
    parser.add_argument("--feature_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def main() -> None:
    """Run protocol construction only and print the six output paths."""
    args = build_parser().parse_args()
    artifacts = run_temporal_protocol_builder(
        args.transaction_path, args.feature_path, args.output_dir
    )
    for path in artifacts.__dict__.values():
        print(path)


if __name__ == "__main__":
    main()
