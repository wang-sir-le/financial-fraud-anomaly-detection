from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fraudx.prediction_reconstruction import (
    _aggregate_metrics,
    _append_audit_rows,
    _paired_partition,
    _reconstruction_status,
    _test_split_hash,
    require_reconstruction_passed,
)


def test_paired_partition_preserves_stable_alignment() -> None:
    ids = np.array([10, 11, 12], dtype=np.int64)
    steps = np.array([5, 5, 6], dtype=np.int32)
    labels = np.array([0, 1, 0], dtype=np.int8)
    common = {
        "raw": np.array([0.1, 0.8, 0.2]),
        "platt": np.array([0.05, 0.9, 0.1]),
        "threshold": 0.5,
        "decision": np.array([0, 1, 0], dtype=np.int8),
    }
    frame = _paired_partition(
        fold=1,
        seed=42,
        split_hash="abc",
        transaction_ids=ids,
        steps=steps,
        labels=labels,
        raw=common,
        recipient=common,
    )
    assert frame["transaction_id"].tolist() == [10, 11, 12]
    assert frame["original_row_index"].tolist() == [10, 11, 12]
    assert frame["test_order_index"].tolist() == [0, 1, 2]
    assert frame["step"].tolist() == [5, 5, 6]
    assert frame["y_true"].tolist() == [0, 1, 0]
    assert np.array_equal(frame["raw_pr_auc_score"], frame["raw_model_probability_platt"])


def test_test_split_hash_is_stable_and_order_sensitive() -> None:
    ids = np.array([1, 2], dtype=np.int64)
    steps = np.array([3, 4], dtype=np.int32)
    labels = np.array([0, 1], dtype=np.int8)
    first = _test_split_hash(ids, steps, labels)
    assert first == _test_split_hash(ids.copy(), steps.copy(), labels.copy())
    assert first != _test_split_hash(ids[::-1], steps[::-1], labels[::-1])


def test_aggregate_metrics_recomputes_confusion_cost_and_saving() -> None:
    labels = np.array([1, 1, 0, 0], dtype=np.int8)
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    metrics = _aggregate_metrics(labels, scores, 0.5, fn_cost=100.0, fp_cost=1.0)
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (1, 1, 1, 1)
    assert metrics["alert_count"] == 2
    assert metrics["business_cost"] == 101.0
    assert metrics["all_normal_cost"] == 200.0
    assert metrics["cost_saving"] == 99.0
    assert metrics["cost_saving_rate"] == pytest.approx(0.495)


def test_audit_mismatch_blocks_reconstruction() -> None:
    historical = pd.Series(
        {
            "test_size": 4,
            "test_fraud_count": 2,
            "test_fraud_rate": 0.5,
            "pr_auc": 0.8,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "tp": 1,
            "fp": 1,
            "tn": 1,
            "fn": 1,
            "alert_rate": 0.5,
            "alert_count": 2,
            "business_cost": 101.0,
            "all_normal_cost": 200.0,
            "cost_saving": 99.0,
            "cost_saving_rate": 0.495,
            "threshold": 0.5,
        }
    )
    reconstructed = historical.to_dict()
    reconstructed["tp"] = 0
    reconstructed["regenerated_validation_threshold"] = 0.5
    rows: list[dict[str, object]] = []
    _append_audit_rows(
        rows,
        historical_row=historical,
        reconstructed=reconstructed,
        fold=1,
        model_id="raw_safe",
        scheme="Raw",
        seed=42,
    )
    audit = pd.DataFrame(rows)
    for metric in ("raw_probability_hash", "calibrated_probability_hash"):
        audit.loc[len(audit)] = {
            "fold": 1,
            "scheme": "Raw",
            "model_id": "raw_safe",
            "seed": 42,
            "metric": metric,
            "historical_value": "same",
            "reconstructed_value": "same",
            "absolute_difference": 0.0,
            "relative_difference": 0.0,
            "tolerance": 0.0,
            "passed": True,
        }
    status = _reconstruction_status(
        audit,
        expected_count=1,
        actual_count=1,
        all_indices_match=True,
        all_labels_match=True,
    )
    assert not status["all_confusion_matrices_match"]
    assert not status["reconstruction_passed"]


def test_reconstruction_gate_rejects_missing_or_false_status(tmp_path) -> None:
    status_path = tmp_path / "status.json"
    with pytest.raises(RuntimeError, match="missing"):
        require_reconstruction_passed(status_path)
    status_path.write_text(json.dumps({"reconstruction_passed": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not pass"):
        require_reconstruction_passed(status_path)


def test_reconstruction_gate_accepts_true_status(tmp_path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"reconstruction_passed": True}), encoding="utf-8")
    assert require_reconstruction_passed(status_path)["reconstruction_passed"] is True
