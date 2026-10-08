"""Only the six reserved model calls and their counted native prediction checks."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import Booster, LGBMClassifier
from native_common import (
    ROOT,
    events,
    read_json,
    require,
    sha,
    utc,
    verify_lock,
    write_json,
)


def run_model_worker(action: str, payload_path: Path) -> None:
    verify_lock()
    payload = read_json(payload_path)
    cell = payload["cell"]
    config = read_json(ROOT / "config/EXECUTION_CONFIG.json")
    require(cell in config["cells"], "cell outside six-item schedule")
    if action == "train":
        token = read_json(ROOT / "audit/FIT_ATTEMPTS.json")
        require(
            token["current_cell"] == cell and token["reserved_attempts"] <= 6,
            "fit not reserved by authorized controller",
        )
        folder = ROOT / "models" / cell
        require(not folder.exists(), "model attempt already exists")
        folder.mkdir()
        record: dict[str, Any] = {
            "cell": cell,
            "status": "STARTED",
            "start_utc": utc(),
            "fit_calls_entered": 0,
            "model_saves": 0,
            "model_loads": 0,
            "prediction_calls": 0,
            "training_check_rows_per_call": 0,
            "optimizer_seconds": None,
            "fit_calls_returned": 0,
            "model_loads_returned": 0,
            "prediction_calls_returned": 0,
        }
        write_json(folder / "WORKER.json", record)
        x = np.load(ROOT / "features" / f"{cell}_train.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(ROOT / "labels/train_only.npy", allow_pickle=False)
        ids = np.load(ROOT / "features/train_identity.npy", allow_pickle=False)
        require(
            x.shape[0] == len(labels) == len(ids), "training source/label matrix alignment failed"
        )
        require(
            bool(np.isin(labels, [0, 1]).all()) and len(np.unique(labels)) == 2,
            "invalid training classes",
        )
        selection = np.argsort(ids[:, 0], kind="stable")[: config["sample_check"]["n_max"]]
        sample = np.asarray(x[selection], dtype=np.float32)
        write_json(
            folder / "TRAIN_CHECK_IDENTITIES.json",
            {
                "selection": config["sample_check"],
                "source_row_ids": ids[selection, 0].astype(int).tolist(),
                "steps": ids[selection, 1].astype(int).tolist(),
            },
            exclusive=True,
        )
        params = read_json(ROOT / "config/C07_PARAMETERS.json")
        model = LGBMClassifier(**params)
        require(
            model.get_params(deep=False) == params, "constructor changed since native acceptance"
        )
        record["fit_calls_entered"] = 1
        write_json(folder / "WORKER.json", record)
        events("model_fit_entered", cell=cell, count=1)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x, labels)
        record["optimizer_seconds"] = time.perf_counter() - started
        record["fit_calls_returned"] = 1
        record["warnings"] = [str(w.message) for w in caught]
        record["actual_rounds"] = model.booster_.current_iteration()
        write_json(folder / "WORKER.json", record)
        require(record["actual_rounds"] == 200, "planned rounds not completed")
        record["prediction_calls"] += 1
        write_json(folder / "WORKER.json", record)
        prediction_started = time.perf_counter()
        predicted_before = np.asarray(
            model.predict(sample, raw_score=True, num_threads=4), dtype=np.float64
        )
        record["check_before_prediction_seconds"] = time.perf_counter() - prediction_started
        record["prediction_calls_returned"] += 1
        record["training_check_rows_per_call"] = len(selection)
        write_json(folder / "WORKER.json", record)
        saved = folder / "model.txt"
        save_started = time.perf_counter()
        model.booster_.save_model(str(saved))
        record["model_saves"] += 1
        record["save_seconds"] = time.perf_counter() - save_started
        record["model_loads"] += 1
        write_json(folder / "WORKER.json", record)
        load_started = time.perf_counter()
        reloaded = Booster(model_file=str(saved))
        record["reload_seconds"] = time.perf_counter() - load_started
        record["model_loads_returned"] += 1
        record["prediction_calls"] += 1
        write_json(folder / "WORKER.json", record)
        prediction_started = time.perf_counter()
        predicted_after = np.asarray(
            reloaded.predict(sample, raw_score=True, num_threads=4), dtype=np.float64
        )
        record["check_after_prediction_seconds"] = time.perf_counter() - prediction_started
        record["prediction_calls_returned"] += 1
        write_json(folder / "WORKER.json", record)
        require(
            bool(np.isfinite(predicted_before).all())
            and np.array_equal(predicted_before, predicted_after),
            "exact train-sample save/reload margin check failed",
        )
        np.save(folder / "train_check_margin.npy", predicted_after, allow_pickle=False)
        write_json(folder / "BACKEND_PARAMETERS.json", model.booster_.params, exclusive=True)
        record.update(
            {
                "status": "SUCCESS",
                "end_utc": utc(),
                "model_sha256": sha(saved),
                "train_matrix_sha256": sha(ROOT / "features" / f"{cell}_train.npy"),
                "preprocessor_sha256": sha(ROOT / "preprocessing" / f"{cell}.json"),
                "training_check_exact": True,
            }
        )
        write_json(folder / "WORKER.json", record)
        return
    require(action == "predict", "unknown worker action")
    locked = read_json(ROOT / "config/MODELS_LOCK.json")
    expected = locked["models"][cell]
    saved = ROOT / "models" / cell / "model.txt"
    require(sha(saved) == expected["sha256"], "model changed after six-model lock")
    folder = ROOT / "results" / cell
    require(not folder.exists(), "review prediction task already exists")
    folder.mkdir()
    record = {
        "cell": cell,
        "status": "STARTED",
        "model_loads": 0,
        "prediction_calls": 0,
        "fit_calls_entered": 0,
        "start_utc": utc(),
        "model_loads_returned": 0,
        "prediction_calls_returned": 0,
    }
    write_json(folder / "PREDICTION.json", record)
    x = np.load(ROOT / "features" / f"{cell}_review.npy", mmap_mode="r", allow_pickle=False)
    record["model_loads"] = 1
    write_json(folder / "PREDICTION.json", record)
    load_started = time.perf_counter()
    review_model = Booster(model_file=str(saved))
    record["model_load_seconds"] = time.perf_counter() - load_started
    record["model_loads_returned"] = 1
    record["prediction_calls"] = 1
    write_json(folder / "PREDICTION.json", record)
    started = time.perf_counter()
    margins = np.asarray(review_model.predict(x, raw_score=True, num_threads=4), dtype=np.float64)
    record["native_prediction_seconds"] = time.perf_counter() - started
    record["prediction_calls_returned"] = 1
    record["rows"] = len(x)
    write_json(folder / "PREDICTION.json", record)
    require(margins.shape == (len(x),) and bool(np.isfinite(margins).all()), "invalid raw margins")
    np.save(folder / "review_margin.npy", margins, allow_pickle=False)
    record.update(
        {
            "status": "SUCCESS",
            "rows": len(x),
            "end_utc": utc(),
            "margin_sha256": sha(folder / "review_margin.npy"),
            "review_matrix_sha256": sha(ROOT / "features" / f"{cell}_review.npy"),
        }
    )
    write_json(folder / "PREDICTION.json", record)
