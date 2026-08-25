from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from fraudx.ieee_cis_temporal_protocol import (
    OUTPUT_FILENAMES,
    audit_source_model_independence,
    build_temporal_folds,
    fold_boundaries,
    run_temporal_protocol_builder,
)


def _transaction() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    transaction_id = 1
    for timestamp in range(1, 101):
        for within_timestamp in range(2):
            rows.append(
                {
                    "TransactionID": transaction_id,
                    "TransactionDT": timestamp,
                    "TransactionAmt": float(timestamp + within_timestamp),
                    "ProductCD": ["C", "H", "R", "S", "W"][timestamp % 5],
                    "isFraud": int((timestamp + within_timestamp) % 11 == 0),
                }
            )
            transaction_id += 1
    return pd.DataFrame(rows)


def _features(transaction: pd.DataFrame) -> pd.DataFrame:
    result = transaction[["TransactionID", "TransactionDT"]].copy()
    result["card_entity_available"] = 1
    result["card_has_history"] = (result["TransactionDT"] > 7).astype(int)
    result["combo_entity_available"] = 1
    result["combo_has_history"] = (result["TransactionDT"] > 14).astype(int)
    return result


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    transaction = _transaction()
    transaction_path = tmp_path / "train_transaction.csv"
    feature_path = tmp_path / "features.parquet"
    transaction.to_csv(transaction_path, index=False)
    _features(transaction).to_parquet(feature_path, index=False)
    return transaction_path, feature_path


def _artifact_hashes(output_dir: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in OUTPUT_FILENAMES
    }


def test_1_time_ranges_do_not_overlap() -> None:
    folds = build_temporal_folds(_transaction())
    for fold in folds:
        assert fold.train["TransactionDT"].max() < fold.validation["TransactionDT"].min()
        assert fold.validation["TransactionDT"].max() < fold.test["TransactionDT"].min()


def test_2_same_transactiondt_never_crosses_splits() -> None:
    folds = build_temporal_folds(_transaction())
    for fold in folds:
        time_sets = [
            set(fold.train["TransactionDT"]),
            set(fold.validation["TransactionDT"]),
            set(fold.test["TransactionDT"]),
        ]
        assert time_sets[0].isdisjoint(time_sets[1])
        assert time_sets[0].isdisjoint(time_sets[2])
        assert time_sets[1].isdisjoint(time_sets[2])


def test_3_fold_endpoints_move_toward_the_future() -> None:
    definitions = fold_boundaries(build_temporal_folds(_transaction()))
    for previous, current in zip(definitions[:-1], definitions[1:], strict=True):
        for split in ("train", "validation", "test"):
            assert current[split]["end_transaction_dt"] > previous[split]["end_transaction_dt"]


def test_4_same_inputs_produce_identical_artifact_hashes(tmp_path: Path) -> None:
    transaction_path, feature_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "output"
    run_temporal_protocol_builder(transaction_path, feature_path, output_dir)
    first = _artifact_hashes(output_dir)
    run_temporal_protocol_builder(transaction_path, feature_path, output_dir)
    second = _artifact_hashes(output_dir)
    assert first == second


def test_5_builder_source_is_independent_of_estimators() -> None:
    import fraudx.ieee_cis_temporal_protocol as builder

    assert audit_source_model_independence(Path(builder.__file__)) == []


def test_6_changing_labels_does_not_change_boundaries() -> None:
    original = _transaction()
    changed = original.copy()
    changed["isFraud"] = 1 - changed["isFraud"]
    assert fold_boundaries(build_temporal_folds(original)) == fold_boundaries(
        build_temporal_folds(changed)
    )


def test_required_outputs_and_statistics_are_complete(tmp_path: Path) -> None:
    transaction_path, feature_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "output"
    run_temporal_protocol_builder(transaction_path, feature_path, output_dir)
    assert {path.name for path in output_dir.iterdir()} == set(OUTPUT_FILENAMES)
    fold_summary = pd.read_csv(output_dir / "temporal_fold_summary.csv")
    distribution = pd.read_csv(output_dir / "temporal_distribution_audit.csv")
    history = pd.read_csv(output_dir / "entity_history_availability_by_fold.csv")
    cold_start = pd.read_csv(output_dir / "cold_start_audit.csv")
    assert len(fold_summary) == 9
    assert len(distribution) == 9
    assert len(history) == 18
    assert len(cold_start) == 12
    assert set(history["entity"]) == {"Entity A", "Entity C"}
    report = (output_dir / "IEEE_CIS_temporal_protocol_report.md").read_text(
        encoding="utf-8"
    )
    assert (
        "Temporal boundaries were determined before model training and without observing "
        "model performance."
    ) in report
