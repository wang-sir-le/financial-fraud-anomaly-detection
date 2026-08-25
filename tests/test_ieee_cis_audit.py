from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudx.ieee_cis_audit import (
    ENTITY_DEFINITIONS,
    assign_temporal_audit_split,
    audit_entity_repeatability,
    audit_entity_temporal_coverage,
    audit_history_feasibility,
    audit_temporal_distribution,
    run_ieee_cis_audit,
)


def _audit_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    times = [10, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    cards = [1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 1, 6]
    for index, (time_value, card) in enumerate(zip(times, cards, strict=True), start=1):
        rows.append(
            {
                "TransactionID": index,
                "isFraud": int(index in {3, 10}),
                "TransactionDT": time_value,
                "TransactionAmt": float(index * 10),
                "card1": card,
                "card2": 20.0,
                "card3": 30.0,
                "card5": 50.0,
                "card6": "credit",
                "addr1": 100.0 if index != 12 else np.nan,
                "has_identity": index % 2 == 0,
            }
        )
    return pd.DataFrame(rows)


def test_temporal_distribution_has_ten_equal_width_buckets() -> None:
    frame = _audit_frame()
    result = audit_temporal_distribution(frame, bins=10)
    assert result["time_bucket"].tolist() == list(range(1, 11))
    assert result["transaction_count"].sum() == len(frame)
    assert result["fraud_count"].sum() == int(frame["isFraud"].sum())
    assert result["global_time_duration"].iloc[0] == 100


def test_timestamp_preserving_split_never_crosses_equal_time() -> None:
    frame = _audit_frame()
    labels, split = assign_temporal_audit_split(frame)
    for _, group in frame.assign(split=labels).groupby("TransactionDT"):
        assert group["split"].nunique() == 1
    boundaries = split.set_index("split")
    assert boundaries.loc["Train", "max_transaction_dt"] < boundaries.loc[
        "Validation", "min_transaction_dt"
    ]
    assert boundaries.loc["Validation", "max_transaction_dt"] < boundaries.loc[
        "Test", "min_transaction_dt"
    ]


def test_repeatability_uses_valid_entities_as_cold_start_denominator() -> None:
    frame = _audit_frame()
    result = audit_entity_repeatability(frame, "Entity B", ("card1", "addr1"))
    assert result["missing_entity_transactions"] == 1
    assert result["unique_entity_count"] == 5
    assert result["single_transaction_entity_ratio"] == pytest.approx(2 / 5)
    assert result["ge_2_transactions_entity_ratio"] == pytest.approx(3 / 5)


def test_future_coverage_reports_transaction_and_unique_entity_views() -> None:
    frame = _audit_frame()
    labels = pd.Series(
        ["Train"] * 7 + ["Validation"] * 2 + ["Test"] * 3,
        index=frame.index,
    )
    frame["audit_split"] = labels
    result = pd.DataFrame(audit_entity_temporal_coverage(frame, "Entity A", ("card1",)))
    validation = result.loc[result["future_split"] == "Validation"].iloc[0]
    test = result.loc[result["future_split"] == "Test"].iloc[0]
    assert validation["transaction_seen_in_train_ratio"] == 0.5
    assert validation["unique_entity_seen_in_train_ratio"] == 0.5
    assert test["transaction_seen_in_train_ratio"] == pytest.approx(1 / 3)
    assert test["unique_entity_seen_in_train_ratio"] == pytest.approx(1 / 3)


def test_history_feasibility_excludes_same_timestamp_transactions() -> None:
    frame = _audit_frame().iloc[:5].copy()
    result = pd.DataFrame(audit_history_feasibility(frame, "Entity A", ("card1",)))
    previous = result.loc[
        result["candidate_historical_feature"] == "entity_previous_transaction_count"
    ].iloc[0]
    amount_std = result.loc[
        result["candidate_historical_feature"] == "entity_historical_amount_std"
    ].iloc[0]
    # card1=1 has two transactions at time 10. Both have zero strict history.
    # Its time-20 and time-30 transactions have two and three prior rows.
    assert previous["sufficient_history_transaction_count"] == 2
    assert amount_std["sufficient_history_transaction_count"] == 2
    assert not bool(previous["same_timestamp_history_allowed"])
    assert not bool(previous["label_used"])


def test_end_to_end_audit_writes_all_required_artifacts(tmp_path: Path) -> None:
    transaction = _audit_frame().drop(columns="has_identity")
    identity = pd.DataFrame(
        {
            "TransactionID": [2, 4, 6, 8, 10, 12],
            "DeviceType": ["desktop"] * 6,
        }
    )
    transaction_path = tmp_path / "train_transaction.csv"
    identity_path = tmp_path / "train_identity.csv"
    output_dir = tmp_path / "audit_results"
    transaction.to_csv(transaction_path, index=False)
    identity.to_csv(identity_path, index=False)
    outputs = run_ieee_cis_audit(
        transaction_path,
        identity_path,
        output_dir,
        memory_chunk_rows=5,
    )
    expected = {
        "data_integrity_audit.csv",
        "label_audit.csv",
        "join_audit.csv",
        "temporal_distribution.csv",
        "temporal_split_audit.csv",
        "entity_repeatability_audit.csv",
        "entity_temporal_coverage.csv",
        "historical_feature_feasibility.csv",
        "IEEE_CIS_entity_temporal_audit_report.md",
        "IEEE_CIS_entity_temporal_audit.log",
        "audit_metadata.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    assert len(outputs.temporal_distribution) == 10
    assert len(outputs.entity_repeatability) == len(ENTITY_DEFINITIONS)
    metadata = json.loads((output_dir / "audit_metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_training_performed"] is False
    assert metadata["label_used_for_history"] is False
    report = outputs.report_path.read_text(encoding="utf-8")
    assert "Entity selection requires researcher decision." in report


def test_missing_or_invalid_label_blocks_audit(tmp_path: Path) -> None:
    transaction = _audit_frame().drop(columns="has_identity")
    transaction.loc[0, "isFraud"] = 2
    identity = pd.DataFrame({"TransactionID": [1]})
    transaction_path = tmp_path / "train_transaction.csv"
    identity_path = tmp_path / "train_identity.csv"
    transaction.to_csv(transaction_path, index=False)
    identity.to_csv(identity_path, index=False)
    with pytest.raises(ValueError, match="missing or invalid labels"):
        run_ieee_cis_audit(transaction_path, identity_path, tmp_path / "out")
