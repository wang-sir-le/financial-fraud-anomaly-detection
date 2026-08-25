"""Protocol and numerical tests for IEEE-CIS paired time-block Bootstrap."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraudx.ieee_cis_experiment import SEEDS, file_hash
from fraudx.ieee_cis_timeblock_bootstrap import (
    METRICS,
    BootstrapPaths,
    _verify_frozen_inputs,
    block_length_for_groups,
    build_model_view,
    effect_arrays,
    load_fold_data,
    make_overall_replicates,
    model_metrics_for_replicates,
    moving_block_positions,
    occurrence_indices,
    position_multiplicities,
    summary_row,
    weighted_average_precision,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "ieee_cis" / "model_results"
TEMPORAL_PROTOCOL = (
    ROOT
    / "outputs"
    / "ieee_cis"
    / "temporal_protocol"
    / "ieee_cis_temporal_protocol.json"
)
RESULT_METADATA = (
    ROOT / "outputs" / "ieee_cis" / "result_analysis" / "result_analysis_metadata.json"
)
SOURCE = ROOT / "src" / "fraudx" / "ieee_cis_timeblock_bootstrap.py"
FROZEN_RESULTS_AVAILABLE = all(
    path.is_file()
    for path in (
        MODEL_DIR / "experiment_config.json",
        MODEL_DIR / "run_metadata.json",
        MODEL_DIR / "prediction_scores.parquet",
        MODEL_DIR / "seed_metrics.csv",
        TEMPORAL_PROTOCOL,
        RESULT_METADATA,
    )
)
requires_frozen_results = pytest.mark.skipif(
    not FROZEN_RESULTS_AVAILABLE,
    reason="requires locally regenerated frozen result assets that are not redistributed",
)


def _paths(tmp_path: Path) -> BootstrapPaths:
    return BootstrapPaths(MODEL_DIR, TEMPORAL_PROTOCOL, RESULT_METADATA, tmp_path)


@requires_frozen_results
def test_frozen_input_hashes_are_registered_and_valid(tmp_path: Path) -> None:
    inputs = _verify_frozen_inputs(_paths(tmp_path))
    assert all(
        inputs["hashes"][name] == expected
        for name, expected in inputs["expected_hashes"].items()
    )
    run_metadata = json.loads((MODEL_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    assert file_hash(MODEL_DIR / "prediction_scores.parquet") == run_metadata["output_hashes"][
        "prediction_scores.parquet"
    ]


def test_block_length_and_non_circular_exact_group_length_are_deterministic() -> None:
    assert block_length_for_groups(57_052) == 38
    first = moving_block_positions(11, 3, 20, np.random.default_rng(20260822))
    second = moving_block_positions(11, 3, 20, np.random.default_rng(20260822))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, 11)
    assert first.min() >= 0
    assert first.max() < 11
    for row in first:
        for start in range(0, 9, 3):
            np.testing.assert_array_equal(np.diff(row[start : start + 3]), [1, 1])


def test_group_multiplicity_preserves_groups_but_transaction_count_can_vary() -> None:
    positions = np.array([[0, 1, 2], [1, 1, 1]], dtype=np.int32)
    multiplicities = position_multiplicities(positions, 3)
    group_sizes = np.array([1, 2, 4])
    assert multiplicities.sum(axis=1).tolist() == [3, 3]
    assert (multiplicities @ group_sizes).tolist() == [7, 6]


def test_duplicate_occurrence_order_is_deterministic() -> None:
    groups = [np.array([20, 10]), np.array([30])]
    first = occurrence_indices(np.array([0, 1, 0]), groups)
    second = occurrence_indices(np.array([0, 1, 0]), groups)
    pd.testing.assert_frame_equal(first, second)
    assert first["TransactionID"].tolist() == [10, 20, 30, 10, 20]
    assert first["bootstrap_occurrence_index"].tolist() == [1, 2, 3, 4, 5]


def test_weighted_average_precision_matches_explicit_duplicated_population() -> None:
    ids = np.array([1, 2, 3, 4])
    labels = np.array([1, 0, 1, 0], dtype=np.int8)
    scores = np.array([0.8, 0.8, 0.3, 0.1])
    time_positions = np.array([0, 0, 1, 2], dtype=np.int32)
    model = build_model_view(ids, labels, scores, time_positions, 3)
    multiplicities = np.array([[2, 0, 1], [0, 3, 1]], dtype=np.int16)
    actual = weighted_average_precision(
        multiplicities,
        model.total_by_time_score,
        model.positive_by_time_score,
    )
    expected = []
    for weights in multiplicities:
        row_weights = weights[time_positions]
        expected.append(average_precision_score(labels, scores, sample_weight=row_weights))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_top3_is_recalculated_with_ceil_and_deterministic_ties() -> None:
    size = 101
    ids = np.arange(1, size + 1)
    labels = (ids % 7 == 0).astype(np.int8)
    scores = np.full(size, 0.5)
    time_positions = np.arange(size, dtype=np.int32)
    model = build_model_view(ids, labels, scores, time_positions, size)
    multiplicities = np.ones((1, size), dtype=np.int16)
    alert_count = np.array([math.ceil(0.03 * size)])
    metrics = model_metrics_for_replicates(
        model,
        labels,
        time_positions,
        multiplicities,
        np.array([size]),
        np.array([labels.sum()]),
        alert_count,
    )
    assert alert_count[0] == 4
    assert metrics["tp"][0] == labels[:4].sum()


def test_shared_capacity_cost_identity_and_seed_average() -> None:
    baseline = {
        "tp": np.array([5.0, 6.0]),
        "recall": np.array([0.5, 0.6]),
        "cost": np.array([600.0, 500.0]),
        "pr_auc": np.array([0.2, 0.3]),
        "precision": np.array([0.4, 0.5]),
        "f1": np.array([0.44, 0.54]),
    }
    candidate = {
        "tp": np.array([7.0, 5.0]),
        "recall": np.array([0.7, 0.5]),
        "cost": np.array([398.0, 601.0]),
        "pr_auc": np.array([0.25, 0.28]),
        "precision": np.array([0.6, 0.4]),
        "f1": np.array([0.64, 0.44]),
    }
    effects = effect_arrays(baseline, candidate)
    np.testing.assert_array_equal(effects["cost_reduction"], 101 * effects["delta_tp"])
    assert effects["cost_reduction"].mean() == 101 * effects["delta_tp"].mean()


def test_zero_fraud_is_not_redrawn_and_undefined_metrics_are_nan() -> None:
    ids = np.arange(10)
    labels = np.zeros(10, dtype=np.int8)
    scores = np.linspace(0.1, 0.9, 10)
    positions = np.arange(10, dtype=np.int32)
    model = build_model_view(ids, labels, scores, positions, 10)
    metrics = model_metrics_for_replicates(
        model,
        labels,
        positions,
        np.ones((1, 10), dtype=np.int16),
        np.array([10]),
        np.array([0]),
        np.array([1]),
    )
    assert metrics["tp"][0] == 0
    assert np.isnan(metrics["recall"][0])
    assert np.isnan(metrics["pr_auc"][0])


def test_percentile_ci_directions_and_invalid_counts_are_exact() -> None:
    values = np.array([-2.0, 0.0, 1.0, 3.0, np.nan])
    row = summary_row(values=values, metric="delta_tp", observed=0.5, extra={})
    finite = values[np.isfinite(values)]
    expected_low, expected_high = np.percentile(finite, [2.5, 97.5])
    assert row["ci_2_5"] == pytest.approx(expected_low)
    assert row["ci_97_5"] == pytest.approx(expected_high)
    assert row["proportion_gt_zero"] == 0.5
    assert row["proportion_lt_zero"] == 0.25
    assert row["proportion_eq_zero"] == 0.25
    assert row["valid_replicates"] == 4
    assert row["invalid_replicates"] == 1


def test_three_fold_secondary_aggregation_is_exactly_equal_weighted() -> None:
    rows = []
    for fold, effect in ((1, -2.0), (2, 0.5), (3, 4.0)):
        for replicate_id in (1, 2):
            row: dict[str, float | int] = {"fold": fold, "replicate_id": replicate_id}
            row.update({metric: effect + replicate_id for metric in METRICS})
            rows.append(row)
    overall = make_overall_replicates(pd.DataFrame(rows))
    observed = overall.loc[overall["replicate_id"] == 1, "delta_tp"].iloc[0]
    assert observed == pytest.approx(1.8333333333333333)
    assert set(overall["analysis_role"]) == {"secondary"}


@requires_frozen_results
def test_actual_m0_m2_and_five_seed_populations_align_and_observed_reproduce() -> None:
    metrics = pd.read_csv(MODEL_DIR / "seed_metrics.csv")
    audit_rows: list[dict[str, object]] = []
    connection = duckdb.connect(database=":memory:")
    try:
        fold = load_fold_data(
            connection,
            MODEL_DIR / "prediction_scores.parquet",
            metrics,
            1,
            audit_rows,
        )
    finally:
        connection.close()
    assert len(fold.models) == 2 * len(SEEDS)
    assert len(audit_rows) == 2 * len(SEEDS)
    assert all(bool(row["passed"]) for row in audit_rows)
    mean_cost = np.mean([fold.observed_by_seed[seed]["cost_reduction"] for seed in SEEDS])
    assert mean_cost == pytest.approx(-1999.8)
    assert len(fold.unique_times) == 57_052


def test_source_performs_no_model_operation_or_stored_selection_reuse() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    invoked = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {"fit", "predict", "predict_proba"} & invoked
    assert "selected_at_3pct" not in source
    assert "calibration_count\": 0" in source
    assert "shap_count\": 0" in source
