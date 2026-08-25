from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudx.ieee_cis_history import (
    CORE_SUFFIXES,
    EPSILON,
    build_ieee_history_features,
    run_history_feature_builder,
    run_history_leakage_audit,
)


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6, 7, 8],
            "TransactionDT": [100, 100, 200, 300, 150, 250, 400, 500],
            "TransactionAmt": [10.0, 20.0, 30.0, 50.0, 7.0, 9.0, 60.0, 11.0],
            "ProductCD": ["W", "C", "W", "H", "R", "R", "S", "C"],
            "isFraud": [0, 1, 0, 0, 1, 0, 1, 0],
            "card1": [1, 1, 1, 1, 2, 2, 1, 3],
            "card2": [10.0, 10.0, 10.0, 10.0, np.nan, np.nan, 10.0, 30.0],
            "card3": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 40.0],
            "card5": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 50.0],
            "card6": ["credit", "credit", "credit", "credit", "debit", "debit", "credit", "debit"],
        }
    )


def _feature_columns() -> list[str]:
    return [
        column
        for column in build_ieee_history_features(_transactions()).columns
        if column not in {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"}
    ]


def test_first_entity_transactions_have_zero_count() -> None:
    result = build_ieee_history_features(_transactions()).set_index("TransactionID")
    assert result.loc[1, "card_prev_count"] == 0
    assert result.loc[2, "card_prev_count"] == 0
    assert result.loc[5, "card_prev_count"] == 0
    assert result.loc[8, "card_prev_count"] == 0


def test_same_timestamp_transactions_do_not_contribute_to_each_other() -> None:
    result = build_ieee_history_features(_transactions()).set_index("TransactionID")
    for transaction_id in (1, 2):
        assert result.loc[transaction_id, "card_prev_count"] == 0
        assert np.isnan(result.loc[transaction_id, "card_prev_amt_mean"])
        assert result.loc[transaction_id, "card_prev_unique_product_count"] == 0
        assert np.isnan(result.loc[transaction_id, "card_time_since_last"])
    # Both t=100 products become visible only to the t=200 transaction.
    assert result.loc[3, "card_prev_count"] == 2
    assert result.loc[3, "card_prev_unique_product_count"] == 2
    assert result.loc[3, "card_time_since_last"] == 100


def test_amount_statistics_and_ratio_follow_frozen_formulas() -> None:
    result = build_ieee_history_features(_transactions()).set_index("TransactionID")
    assert result.loc[3, "card_prev_amt_mean"] == pytest.approx(15.0)
    assert result.loc[3, "card_prev_amt_std"] == pytest.approx(np.std([10.0, 20.0], ddof=1))
    assert result.loc[3, "card_amt_to_prev_mean"] == pytest.approx(30.0 / (15.0 + EPSILON))
    assert result.loc[4, "card_prev_amt_mean"] == pytest.approx(20.0)
    assert result.loc[4, "card_prev_amt_std"] == pytest.approx(10.0)


def test_prefix_and_future_append_invariance() -> None:
    frame = _transactions()
    prefix = frame.loc[frame["TransactionDT"] <= 300].copy()
    prefix_result = build_ieee_history_features(prefix).sort_values("TransactionID")
    full_result = build_ieee_history_features(frame)
    original = full_result.loc[
        full_result["TransactionID"].isin(prefix["TransactionID"])
    ].sort_values("TransactionID")
    pd.testing.assert_frame_equal(
        prefix_result.reset_index(drop=True),
        original.reset_index(drop=True),
        check_exact=True,
    )


def test_label_independence() -> None:
    frame = _transactions()
    original = build_ieee_history_features(frame)
    mutated = frame.copy()
    mutated["isFraud"] = 1 - mutated["isFraud"]
    changed = build_ieee_history_features(mutated)
    pd.testing.assert_frame_equal(
        original[_feature_columns()],
        changed[_feature_columns()],
        check_exact=True,
    )


def test_input_order_invariance_and_stable_output_order() -> None:
    frame = _transactions()
    original = build_ieee_history_features(frame)
    shuffled = build_ieee_history_features(frame.sample(frac=1, random_state=99))
    pd.testing.assert_frame_equal(original, shuffled, check_exact=True)
    assert original[["TransactionDT", "TransactionID"]].to_records(index=False).tolist() == sorted(
        original[["TransactionDT", "TransactionID"]].to_records(index=False).tolist()
    )


def test_entity_c_missing_components_never_create_synthetic_entity() -> None:
    result = build_ieee_history_features(_transactions()).set_index("TransactionID")
    for transaction_id in (5, 6):
        assert result.loc[transaction_id, "combo_entity_available"] == 0
        assert result.loc[transaction_id, "combo_has_history"] == 0
        assert result.loc[transaction_id, "combo_has_std_history"] == 0
        for suffix in CORE_SUFFIXES:
            assert np.isnan(result.loc[transaction_id, f"combo_{suffix}"])


def test_output_integrity_and_no_transaction_loss() -> None:
    frame = _transactions()
    result = build_ieee_history_features(frame)
    assert len(result) == len(frame)
    assert result["TransactionID"].is_unique
    assert set(result["TransactionID"]) == set(frame["TransactionID"])
    assert result["isFraud"].sum() == frame["isFraud"].sum()


def test_complete_leakage_audit_passes() -> None:
    frame = _transactions()
    result = build_ieee_history_features(frame)
    audit = run_history_leakage_audit(frame, result, sample_target_rows=6)
    assert audit["passed"].all(), audit.loc[~audit["passed"]].to_dict("records")


def test_end_to_end_builder_writes_hashes_parquet_report_and_audits(tmp_path: Path) -> None:
    transaction_path = tmp_path / "train_transaction.csv"
    identity_path = tmp_path / "train_identity.csv"
    output_dir = tmp_path / "ieee_cis_history_features"
    _transactions().to_csv(transaction_path, index=False)
    pd.DataFrame({"TransactionID": [1, 3], "DeviceType": ["desktop", "mobile"]}).to_csv(
        identity_path, index=False
    )
    artifacts = run_history_feature_builder(transaction_path, identity_path, output_dir)
    expected_names = {
        "ieee_cis_entity_history_features.parquet",
        "history_feature_availability_audit.csv",
        "history_feature_leakage_audit.csv",
        "history_feature_metadata.json",
        "IEEE_CIS_history_feature_builder_report.md",
        "IEEE_CIS_history_feature_builder.log",
    }
    assert expected_names.issubset({path.name for path in output_dir.iterdir()})
    saved = pd.read_parquet(artifacts.feature_path, engine="pyarrow")
    assert len(saved) == len(_transactions())
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_training_count"] == 0
    assert metadata["isFraud_used_for_feature_construction"] is False
    digest = hashlib.sha256(artifacts.feature_path.read_bytes()).hexdigest()
    assert metadata["output_file"]["sha256"] == digest
    leakage = pd.read_csv(artifacts.leakage_path)
    assert leakage["passed"].all()
    report = artifacts.report_path.read_text(encoding="utf-8")
    assert "Feature builder frozen acceptance: PASS" in report
