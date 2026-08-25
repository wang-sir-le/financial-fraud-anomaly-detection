from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

import fraudx.ieee_cis_experiment as experiment_module
from fraudx.ieee_cis_experiment import (
    CAPACITY,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    FIXED_MODEL_PARAMETERS,
    MODEL_ORDER,
    SCHEMES,
    SEEDS,
    ExperimentPaths,
    TrainOnlyCategoryEncoder,
    _load_valid_checkpoint,
    array_hash,
    atomic_write_json,
    atomic_write_parquet,
    build_experiment_config,
    feature_specifications,
    fixed_capacity_decision,
    fold_metric_summary,
    history_feature_lists,
    indices_from_protocol,
    validate_completed_metrics,
    validate_join_inputs,
)


def _transaction(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, rows + 1, dtype=np.int64),
            "TransactionDT": np.arange(1, rows + 1, dtype=np.int64),
            "TransactionAmt": np.linspace(1, 100, rows, dtype=np.float32),
            "ProductCD": pd.Series(
                np.resize(np.asarray(["W", "C", "R"], dtype=object), rows), dtype="string"
            ),
            "isFraud": (np.arange(rows) % 7 == 0).astype(np.int8),
        }
    )


def _metadata() -> dict[str, object]:
    card = [
        "card_entity_available",
        "card_prev_count",
        "card_prev_amt_mean",
        "card_prev_amt_std",
        "card_amt_to_prev_mean",
        "card_time_since_last",
        "card_prev_unique_product_count",
        "card_has_history",
        "card_has_std_history",
    ]
    combo = [column.replace("card_", "combo_") for column in card]
    return {
        "columns": [
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "isFraud",
            *card,
            *combo,
        ]
    }


def _history(transaction: pd.DataFrame) -> pd.DataFrame:
    card, combo = history_feature_lists(_metadata())
    history = transaction[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]].copy()
    for column in [*card, *combo]:
        history[column] = np.arange(len(history), dtype=np.float32)
    return history


def _protocol() -> dict[str, object]:
    definitions = []
    boundary_rows = (
        (1, 20, 21, 25, 26, 30),
        (1, 35, 36, 40, 41, 45),
        (1, 50, 51, 55, 56, 60),
    )
    for fold, bounds in enumerate(boundary_rows, start=1):
        definitions.append(
            {
                "fold": fold,
                "train": {
                    "start_transaction_dt": bounds[0],
                    "end_transaction_dt": bounds[1],
                    "transaction_count": bounds[1] - bounds[0] + 1,
                },
                "validation": {
                    "start_transaction_dt": bounds[2],
                    "end_transaction_dt": bounds[3],
                    "transaction_count": bounds[3] - bounds[2] + 1,
                },
                "test": {
                    "start_transaction_dt": bounds[4],
                    "end_transaction_dt": bounds[5],
                    "transaction_count": bounds[5] - bounds[4] + 1,
                },
            }
        )
    return {
        "split_rule": {"fold_count": 3},
        "fold_definitions": definitions,
        "fold_definition_sha256": "fold-hash",
    }


def _complete_metrics() -> pd.DataFrame:
    rows = []
    for fold in (1, 2, 3):
        for model_id in MODEL_ORDER:
            for seed in SEEDS:
                rows.append(
                    {
                        "fold": fold,
                        "model_id": model_id,
                        "model_role": next(
                            scheme.role for scheme in SCHEMES if scheme.model_id == model_id
                        ),
                        "seed": seed,
                        "alert_count": 10,
                        "pr_auc": 0.1 + fold / 100,
                        "roc_auc": 0.7,
                        "precision_at_3pct": 0.2,
                        "recall_at_3pct": 0.3,
                        "f1_at_3pct": 0.24,
                        "business_cost": 1000.0,
                        "cost_saving_rate": 0.2,
                    }
                )
    return pd.DataFrame(rows)


def test_raw_and_history_join_is_one_to_one_without_row_loss() -> None:
    transaction = _transaction()
    history = _history(transaction)
    validate_join_inputs(transaction, history, expected_rows=len(transaction))
    with pytest.raises(ValueError, match="row counts differ"):
        validate_join_inputs(transaction, history.iloc[:-1])
    duplicated = pd.concat([history, history.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_join_inputs(transaction, duplicated)


def test_frozen_fold_boundaries_are_read_not_recomputed_and_remain_ordered() -> None:
    transaction = _transaction()
    folds = indices_from_protocol(transaction, _protocol())
    assert len(folds) == 3
    assert [len(fold.train) for fold in folds] == [20, 35, 50]
    for fold in folds:
        assert transaction.iloc[fold.train]["TransactionDT"].max() < transaction.iloc[
            fold.validation
        ]["TransactionDT"].min()
        assert transaction.iloc[fold.validation]["TransactionDT"].max() < transaction.iloc[
            fold.test
        ]["TransactionDT"].min()


def test_timestamp_groups_cannot_cross_frozen_splits() -> None:
    transaction = _transaction()
    transaction.loc[25, "TransactionDT"] = 25
    with pytest.raises(RuntimeError, match="row count differs"):
        indices_from_protocol(transaction, _protocol())


def test_history_lists_are_explicit_and_exact() -> None:
    card, combo = history_feature_lists(_metadata())
    assert len(card) == len(combo) == 9
    assert all(column.startswith("card_") for column in card)
    assert all(column.startswith("combo_") for column in combo)


def test_m0_to_m3_feature_differences_and_roles_are_frozen() -> None:
    specifications, lists = feature_specifications(_transaction(), _metadata())
    card, combo = history_feature_lists(_metadata())
    assert lists["M0"] == ["TransactionDT", "TransactionAmt", "ProductCD"]
    assert lists["M1"] == [*lists["M0"], *card]
    assert lists["M2"] == lists["M1"]
    assert lists["M3"] == [*lists["M0"], *combo]
    assert specifications["M1"]["feature_columns_hash"] == specifications["M2"][
        "feature_columns_hash"
    ]
    assert specifications["M1"]["scheme_spec_hash"] != specifications["M2"][
        "scheme_spec_hash"
    ]
    assert SCHEMES[2].role == "pre-specified primary framework"
    assert SCHEMES[3].role == "entity-granularity sensitivity analysis"


def test_train_only_encoder_handles_missing_and_unknown_without_future_fit() -> None:
    train = pd.DataFrame({"category": pd.Series(["b", "a", None], dtype="string")})
    future = pd.DataFrame({"category": pd.Series(["a", "future", None], dtype="string")})
    encoder = TrainOnlyCategoryEncoder(["category"]).fit(train)
    assert encoder.vocabularies["category"] == ("a", "b")
    assert encoder.transform(future)["category"].tolist() == [2, 1, 0]
    assert "future" not in encoder.vocabularies["category"]


def test_encoder_is_input_order_invariant_and_uses_no_target_encoding() -> None:
    frame = pd.DataFrame(
        {
            "category": pd.Series(["z", "a", "m", None], dtype="string"),
            "isFraud": [1, 0, 1, 0],
        }
    )
    first = TrainOnlyCategoryEncoder(["category"]).fit(frame)
    second = TrainOnlyCategoryEncoder(["category"]).fit(frame.iloc[::-1])
    assert first.vocabularies == second.vocabularies
    assert "isFraud" not in first.columns


def test_top_three_percent_count_ties_and_label_independence() -> None:
    ids = np.asarray([4, 1, 3, 2, 5, 6, 7, 8, 9, 10])
    scores = np.ones(10)
    labels = np.asarray([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    first = fixed_capacity_decision(ids, labels, scores)
    changed = fixed_capacity_decision(ids, 1 - labels, scores)
    assert first.alert_count == math.ceil(CAPACITY * len(ids))
    assert ids[first.selected].tolist() == [1]
    assert np.array_equal(first.selected, changed.selected)


def test_business_cost_and_saving_formulas_are_exact() -> None:
    ids = np.arange(1, 101)
    labels = np.zeros(100, dtype=np.int8)
    labels[:10] = 1
    scores = np.linspace(1, 0, 100)
    result = fixed_capacity_decision(ids, labels, scores)
    assert result.alert_count == 3
    assert result.tp == 3 and result.fp == 0 and result.fn == 7 and result.tn == 90
    assert result.baseline_cost == FALSE_NEGATIVE_COST * 10
    assert result.business_cost == FALSE_POSITIVE_COST * 0 + FALSE_NEGATIVE_COST * 7
    assert result.cost_saving_rate == pytest.approx(0.3)


def test_probability_hash_uses_binary_values_and_is_reproducible() -> None:
    values = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    assert array_hash(values) == array_hash(values.copy())
    assert array_hash(values) != array_hash(values.astype(np.float32))


def test_synthetic_lightgbm_smoke_prediction_and_parquet(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "amount": np.arange(80, dtype=np.float32),
            "category": np.resize(np.asarray([2, 3], dtype=np.int32), 80),
        }
    )
    labels = (train["amount"].to_numpy() % 9 == 0).astype(np.int8)
    model = LGBMClassifier(
        **FIXED_MODEL_PARAMETERS,
        random_state=42,
        bagging_seed=42,
        feature_fraction_seed=42,
        data_random_seed=42,
        scale_pos_weight=1.0,
    )
    model.fit(train, labels, categorical_feature=["category"])
    prediction_matrix = np.asarray(model.predict_proba(train), dtype=np.float64)
    probabilities = prediction_matrix[:, 1]
    assert probabilities.shape == (80,)
    assert np.isfinite(probabilities).all()
    ledger = pd.DataFrame({"TransactionID": np.arange(80), "raw_probability": probabilities})
    path = tmp_path / "smoke.parquet"
    atomic_write_parquet(ledger, path)
    pd.testing.assert_frame_equal(ledger, pd.read_parquet(path))


def test_checkpoint_resume_validates_hashes_ids_and_probabilities(tmp_path: Path) -> None:
    ids = np.asarray([10, 11, 12], dtype=np.int64)
    probabilities = np.asarray([0.3, 0.2, 0.1], dtype=np.float64)
    prediction_path = tmp_path / "prediction.parquet"
    metric_path = tmp_path / "metric.json"
    atomic_write_parquet(
        pd.DataFrame({"TransactionID": ids, "raw_probability": probabilities}),
        prediction_path,
    )
    metric = {
        "config_hash": "config",
        "protocol_hash": "protocol",
        "feature_set_hash": "feature",
        "test_transaction_id_hash": array_hash(ids),
        "raw_probability_hash": array_hash(probabilities),
    }
    atomic_write_json(metric, metric_path)
    resumed = _load_valid_checkpoint(
        prediction_path=prediction_path,
        metric_path=metric_path,
        config_hash="config",
        protocol_hash="protocol",
        feature_hash="feature",
        expected_test_ids=ids,
        expected_test_id_hash=array_hash(ids),
    )
    assert resumed == metric
    with pytest.raises(RuntimeError, match="mismatched config_hash"):
        _load_valid_checkpoint(
            prediction_path=prediction_path,
            metric_path=metric_path,
            config_hash="changed",
            protocol_hash="protocol",
            feature_hash="feature",
            expected_test_ids=ids,
            expected_test_id_hash=array_hash(ids),
        )


def test_completed_grid_requires_exact_sixty_unique_tasks_and_alert_counts() -> None:
    metrics = _complete_metrics()
    validate_completed_metrics(metrics)
    with pytest.raises(RuntimeError, match="exactly 60"):
        validate_completed_metrics(metrics.iloc[:-1])
    duplicated = pd.concat([metrics.iloc[:-1], metrics.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="Duplicate"):
        validate_completed_metrics(duplicated)


def test_fold_summary_has_twelve_rows_and_all_seed_statistics() -> None:
    summary = fold_metric_summary(_complete_metrics())
    assert len(summary) == 12
    for suffix in ("mean", "std", "min", "max"):
        assert f"pr_auc_{suffix}" in summary.columns


def test_config_freezes_parameters_seeds_roles_and_zero_restricted_counts(tmp_path: Path) -> None:
    transaction = _transaction()
    metadata = _metadata()
    specifications, _ = feature_specifications(transaction, metadata)
    files = [
        tmp_path / name
        for name in ("tx.csv", "history.parquet", "metadata.json", "protocol.json")
    ]
    for path in files:
        path.write_text(path.name, encoding="utf-8")
    paths = ExperimentPaths(
        transaction_path=files[0],
        history_path=files[1],
        history_metadata_path=files[2],
        protocol_path=files[3],
        output_dir=tmp_path / "output",
    )
    config = build_experiment_config(paths, transaction, metadata, _protocol(), specifications)
    assert config["training"]["expected_official_fit_count"] == 60
    assert config["training"]["seeds"] == list(SEEDS)
    assert config["training"]["early_stopping"] is False
    assert config["restrictions"] == {
        "hyperparameter_tuning_count": 0,
        "feature_selection_count": 0,
        "shap_count": 0,
        "calibration_count": 0,
        "bootstrap_count": 0,
        "test_threshold_optimization_count": 0,
        "historical_labels_used": False,
    }


def test_source_has_no_disallowed_experiment_imports_or_search_calls() -> None:
    source_path = Path(experiment_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    prohibited_roots = {"shap", "optuna", "xgboost", "catboost"}
    prohibited_calls = {"GridSearchCV", "RandomizedSearchCV", "calibrate", "bootstrap"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in prohibited_roots for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in prohibited_roots
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in prohibited_calls
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in prohibited_calls


def test_prediction_ledger_schema_contains_test_only_identifiers() -> None:
    required = {
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "fold",
        "model_id",
        "seed",
        "raw_probability",
        "score_rank",
        "selected_at_3pct",
    }
    source = Path(experiment_module.__file__).read_text(encoding="utf-8")
    assert all(f'"{column}"' in source for column in required)
    assert "validation_probability" not in source
