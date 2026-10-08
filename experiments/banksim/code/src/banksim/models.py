"""Fixed-budget model adapters. No fitting occurs during import or catalog creation."""

from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .common import require, write_json


def candidates(family: str) -> list[dict[str, Any]]:
    require(family in ("LightGBM", "XGBoost"), "Unknown model family")
    keys = ("num_leaves", "min_child_samples", "n_estimators") if family == "LightGBM" else ("max_depth", "min_child_weight", "n_estimators")
    values = ((15, 31), (20, 100), (200, 500)) if family == "LightGBM" else ((3, 6), (1, 10), (200, 500))
    return [{"id": f"C{i:02d}", **dict(zip(keys, v, strict=True))} for i, v in enumerate(itertools.product(*values), start=1)]


def parameters(family: str, candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    grid = {k: v for k, v in candidate.items() if k != "id"}
    common = {"learning_rate": .05, "subsample": .8, "colsample_bytree": .8, "reg_lambda": 1., "reg_alpha": 0., "max_bin": 255, "n_jobs": 4, "random_state": seed, **grid}
    if family == "LightGBM":
        return {**common, "objective": "binary", "boosting_type": "gbdt", "max_depth": -1, "subsample_freq": 1, "subsample_for_bin": 200000, "min_split_gain": 0., "min_child_weight": .001, "class_weight": None, "importance_type": "split", "deterministic": True, "force_col_wise": True, "force_row_wise": False, "device_type": "cpu", "data_random_seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed, "extra_seed": seed, "drop_seed": seed, "boost_from_average": False, "is_unbalance": False, "scale_pos_weight": 1., "zero_as_missing": False, "use_missing": True, "feature_pre_filter": False, "enable_bundle": True, "min_data_in_bin": 3, "verbosity": 1}
    require(family == "XGBoost", "Unknown model family")
    return {**common, "objective": "binary:logistic", "booster": "gbtree", "tree_method": "hist", "sampling_method": "uniform", "device": "cpu", "grow_policy": "depthwise", "max_leaves": 0, "max_delta_step": 0, "gamma": 0., "colsample_bylevel": 1., "colsample_bynode": 1., "scale_pos_weight": 1., "base_score": .5, "eval_metric": "logloss", "enable_categorical": False, "validate_parameters": True, "verbosity": 1, "num_parallel_tree": 1, "early_stopping_rounds": None, "callbacks": None, "missing": np.nan}


def make_estimator(family: str, candidate: dict[str, Any], seed: int) -> Any:
    if family == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**parameters(family, candidate, seed))
    from xgboost import XGBClassifier
    return XGBClassifier(**parameters(family, candidate, seed))


def catalog() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in ("LightGBM", "XGBoost"):
        result[family] = []
        for config in candidates(family):
            model = make_estimator(family, config, 42)
            # Nulls in wrapper options mean the pinned library's documented defaults.
            params = model.get_params(deep=False)
            if family == "XGBoost":
                params["missing"] = "NaN (IEEE missing-value sentinel, not an unspecified value)"
            result[family].append({"candidate": config, "constructor_parameters_seed42": params})
    return result


def choose_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(records) == 8 and len({r["candidate"]["id"] for r in records}) == 8, "All eight tuning candidates required")
    require(all(r["status"] == "success" for r in records), "Incomplete tuning grid: no selection from a survivor subset")
    require(all(np.isfinite(r["validation"]["ap"]) for r in records), "Undefined tuning AP")
    return min(records, key=lambda r: (-r["validation"]["tp@0.03"], -r["validation"]["ap"], r["candidate"]["n_estimators"], r["candidate"]["id"]))


def fit_save(family: str, candidate: dict[str, Any], seed: int, x: np.ndarray, y: np.ndarray, path: Path) -> Any:
    """Called only through guarded future training stage; not exercised on real data now."""
    model = make_estimator(family, candidate, seed)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)  # No validation/test eval_set, no early stopping, all unit weights.
    elapsed = time.perf_counter() - started
    if family == "LightGBM":
        model.booster_.save_model(str(path))
        backend = model.booster_.params
        rounds = model.booster_.current_iteration()
    else:
        model.save_model(path)
        backend = json.loads(model.get_booster().save_config())
        rounds = model.get_booster().num_boosted_rounds()
    write_json(path.with_suffix(path.suffix + ".fit.json"), {"fit_seconds": elapsed, "actual_rounds": rounds, "planned_rounds": candidate["n_estimators"], "warnings": [str(w.message) for w in caught], "backend_parameters_after_fit": backend, "initial_margin": 0., "early_stopping": False})
    return model


def predict_margin(model: Any, family: str, x: np.ndarray) -> np.ndarray:
    result = model.predict(x, raw_score=True) if family == "LightGBM" else model.predict(x, output_margin=True)
    result = np.asarray(result, dtype=np.float64)
    require(result.shape == (len(x),) and bool(np.isfinite(result).all()), "Prediction shape/nonfinite failure")
    return result


def load_predict(path: Path, family: str, x: np.ndarray) -> np.ndarray:
    if family == "LightGBM":
        from lightgbm import Booster as LGBBooster
        result = LGBBooster(model_file=str(path)).predict(x, raw_score=True)
    else:
        from xgboost import Booster as XGBBooster
        from xgboost import DMatrix
        booster = XGBBooster()
        booster.load_model(path)
        result = booster.predict(DMatrix(x, nthread=4), output_margin=True)
    result = np.asarray(result, dtype=np.float64)
    require(result.shape == (len(x),) and bool(np.isfinite(result).all()), "Saved model prediction failure")
    return result
