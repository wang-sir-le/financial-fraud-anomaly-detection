"""Unlabelled preparation and future, explicitly guarded experimental stages."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from . import models
from .common import (
    CELLS,
    FAMILIES,
    SEEDS,
    SPLITS,
    append_event,
    read_json,
    require,
    sha256,
    utc_now,
    write_json,
)
from .data import acquire, assign_splits, build_history, load_unlabelled
from .guards import environment, labels_for_ids, test_permit, training_permit, verify_freeze
from .metrics import ScoreOrder, bootstrap_orders, paired_summary, priority
from .preprocess import Preprocessor, check_common_rules, fit_cells


def file_records(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    return [{"path": str(p.relative_to(root)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(set(files))]


def verify_records(root: Path, records: list[dict[str, Any]]) -> None:
    for item in records:
        p = root / item["path"]
        require(p.is_file() and sha256(p) == item["sha256"], f"Artifact changed: {item['path']}")


def source_records(root: Path) -> list[dict[str, Any]]:
    files = list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py"))
    files += [root / "run.py", root / "verify.py", root / "pyproject.toml"]
    return file_records(root, files)


def prepare(root: Path) -> dict[str, Any]:
    require(not (root / "protocol/FREEZE.json").exists(), "Preparation cannot modify a frozen run")
    acquire(root)
    append_event(root, "unlabelled_preparation_started", labels_read=False)
    started = time.perf_counter()
    frame = assign_splits(build_history(load_unlabelled(root / "inputs/banksim_v1.zip")))
    require("fraud" not in frame.columns, "Label column entered feature cache")
    target = root / "features/unlabelled.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False, compression="zstd")
    files = [target]
    for split in SPLITS:
        part = frame.loc[frame["split"] == split, ["source_row_id", "step"]]
        p = root / f"features/{split}_identity.npy"
        np.save(p, part.to_numpy(dtype=np.int64), allow_pickle=False)
        files.append(p)
    train = frame.loc[frame["split"] == "train"]
    processors = fit_cells(train)
    check_common_rules(processors)
    column_hashes: dict[str, dict[str, str]] = {}
    shape_log: dict[str, Any] = {}
    for cell, proc in processors.items():
        path = root / f"preprocessing/{cell}.json"
        write_json(path, proc.to_dict())
        files.append(path)
        for split in ("train", "validation"):
            matrix = proc.transform(frame.loc[frame["split"] == split])
            p = root / f"preprocessing/{cell}_{split}.npy"
            np.save(p, matrix, allow_pickle=False)
            files.append(p)
            shape_log[f"{cell}/{split}"] = list(matrix.shape)
            for j, name in enumerate(proc.feature_names()):
                digest = hashlib.sha256(matrix[:, j].tobytes()).hexdigest()
                by_name = column_hashes.setdefault(f"{split}/{name}", {})
                require(not by_name or all(value == digest for value in by_name.values()), "Common encoded column differs between cells")
                by_name[cell] = digest
            del matrix
    write_json(root / "preprocessing/common_column_hashes.json", column_hashes)
    files.append(root / "preprocessing/common_column_hashes.json")
    record = {"status": "PREPARED_WITHOUT_LABELS", "rows": len(frame), "split_rows": frame.groupby("split").size().to_dict(), "encoded_train_validation_shapes": shape_log, "test_encoded_matrix_created": False, "fraud_column_loaded": False, "model_fit_calls": 0, "seconds": time.perf_counter() - started, "artifacts": file_records(root, files)}
    write_json(root / "audit/PREPARED.json", record)
    append_event(root, "unlabelled_preparation_complete", rows=len(frame), model_fit_calls=0, labels_read=False)
    return record


def configure(root: Path) -> dict[str, Any]:
    require(not (root / "protocol/FREEZE.json").exists(), "Cannot reconfigure a frozen run")
    draft = read_json(root / "protocol/draft_source.json")
    draft["version"] = "banksim-validation-v1-20260925"
    draft["status"] = "IMPLEMENTED_SPECIFICATION_FREEZE_STATUS_IN_FREEZE_JSON"
    draft["datasets"][0]["independence"] = "No prior modeling/tuning/performance evaluation found; recommendations and unlabelled feasibility screening acknowledged."
    draft["implementation"] = {
        "initial_margin": "0 for both libraries: LightGBM boost_from_average=false; XGBoost base_score=0.5",
        "precision": "Python arbitrary-precision integer moments; exact Decimal integer ratio to cents; no decimal context rounding",
        "constant_columns": "Remove constant encoded float32 output columns using training rows only",
        "missing_category": "__BANKSIM_MISSING_V1__; source collision is an error",
        "catalog": "protocol/MODEL_CATALOG.json; includes all sklearn constructor options for pinned libraries; runtime backend config exported after authorized fitting",
        "parameter_defaults": "Null wrapper options are the defaults of the pinned package; explicit task-affecting tree/sampling/weight/initialization settings recorded in models.py and catalog; no early stopping/eval_set",
        "candidate_order": "C01–C08 enumerate declared grid key order, last key n_estimators varying fastest",
        "fit_budget": "54 actual fits with mandatory exact reuse of two chosen B1 seed42 fits; never more than original cap 56; no automatic retries",
        "failed_tuning": "Record all 16 attempted candidates; any failed candidate blocks selection, never select among successful subset",
        "failed_formal": "Record all scheduled cells; any failure blocks test release until a separately audited resolution; no silent seed deletion",
        "no_positive_test": "After authorized one-time test opening, preserve no-positive status and metrics definitions but stop inferential claim",
        "interval_scope": "95% pointwise for whole-window seed-mean absolute AP/TP/Recall/Precision and H0/H1/D at 1,3,5%; direct cross-library H0/H1; Bonferroni only four primary H1/D TP@3% within each fixed block length; three segments descriptive",
        "raw_test_performance_before_release": "forbidden; no test prediction at training stage",
        "resource_measurement": "sample process RSS every 50ms in future authorized fits; record elapsed wall time; no current fit resource claim",
        "recovery": "Completed fit records verified and reused; interrupted/failed fits do not auto-retry; test release written before access and prohibits rescore; bootstrap replay uses fixed cached predictions"
    }
    write_json(root / "protocol/effective.json", draft)
    write_json(root / "protocol/MODEL_CATALOG.json", models.catalog())
    write_json(root / "reproduction/environment.json", environment())
    schedule = [{"family": f, "cell": c, "seed": s, "status": "NOT_STARTED"} for f in FAMILIES for c in CELLS for s in SEEDS]
    write_json(root / "protocol/FORMAL_SCHEDULE.json", schedule)
    append_event(root, "configured_without_fitting", model_fit_calls=0)
    return {"status": "CONFIGURED_NOT_FROZEN", "formal_conditions": len(schedule)}


def freeze(root: Path) -> dict[str, Any]:
    require(not (root / "protocol/FREEZE.json").exists(), "Freeze is append-only")
    validation = read_json(root / "audit/VALIDATION.json")
    require(validation["status"] == "PASS", "Validation incomplete or failed")
    require(validation["source_records"] == source_records(root), "Code changed after validation")
    verify_records(root, read_json(root / "audit/PREPARED.json")["artifacts"])
    require(read_json(root / "protocol/effective.json")["seeds"] == list(SEEDS), "Seed list differs")
    require(read_json(root / "reproduction/environment.json") == environment(), "Environment changed since configure")
    events = [read for read in (root / "audit/events.jsonl").read_text(encoding="utf-8").splitlines() if read]
    require(not any('"labels_accessed"' in row or '"fit_started"' in row or '"test_release"' in row for row in events), "Real execution occurred before freeze")
    files = [root / r["path"] for r in source_records(root)]
    files += [root / r["path"] for r in read_json(root / "audit/PREPARED.json")["artifacts"]]
    files += [root / "inputs/banksim_v1.zip", root / "inputs/expected_structure.json", root / "audit/PREPARED.json", root / "audit/VALIDATION.json"]
    for check in validation["checks"]:
        files.append(root / check["log"])
    last_checks = (root / validation["checks"][0]["log"]).parent
    files += [last_checks / "pytest.xml", last_checks / "summary.json"]
    files += list((root / "protocol").glob("*.json"))
    files += [root / "reproduction/environment.json", root / "reproduction/requirements-lock.txt", root / "reproduction/install_report.json"]
    record = {"status": "FROZEN_BEFORE_TRAINING", "time_utc": utc_now(), "files": file_records(root, files), "real_model_fit_calls": 0, "real_test_scores_computed": 0, "row_label_analysis_performed": False, "future_gates": ["train/validation each have two label classes", "observed memory and elapsed fitting time", "successful complete grid and selected-configuration lock", "all forty cells accounted for and required models valid before test release"], "not_a_claim": "Not external real-business validation; not evidence of scientific benefit or publication acceptance."}
    write_json(root / "protocol/FREEZE.json", record, exclusive=True)
    append_event(root, "protocol_frozen", freeze_sha256=sha256(root / "protocol/FREEZE.json"))
    return {"status": record["status"], "protected_files": len(set(files))}


class MemorySampler:
    def __init__(self) -> None:
        self.peak = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        process = psutil.Process()
        while not self.done.is_set():
            self.peak = max(self.peak, process.memory_info().rss)
            self.done.wait(.05)

    def __enter__(self) -> MemorySampler:
        self.peak = psutil.Process().memory_info().rss
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self.done.set()
        self.thread.join()


def attempt(root: Path, name: str, family: str, cell: str, config: dict[str, Any], seed: int, x: np.ndarray, y: np.ndarray, validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None, *, adapter: Any = models) -> dict[str, Any]:
    folder = root / "models" / name
    record_path = folder / "record.json"
    if record_path.exists():
        previous = read_json(record_path)
        require(previous["status"] == "success", "Failed/interrupted fit cannot auto-retry")
        require(previous["family"] == family and previous["cell"] == cell and previous["seed"] == seed and previous["candidate"] == config, "Cached fit identity/configuration mismatch")
        require(previous["freeze_sha256"] == sha256(root / "protocol/FREEZE.json"), "Cached fit freeze mismatch")
        require(previous["preprocessor_sha256"] == sha256(root / f"preprocessing/{cell}.json") and previous["training_identity_sha256"] == sha256(root / "features/train_identity.npy"), "Cached fit input/transform mismatch")
        verify_records(root, previous["artifacts"])
        return previous
    attempts = list((root / "models").rglob("record.json"))
    require(len(attempts) < 56, "Original upper fitting budget exhausted")
    folder.mkdir(parents=True, exist_ok=True)
    record = {"family": family, "cell": cell, "candidate": config, "seed": seed, "status": "started", "time_utc": utc_now(), "freeze_sha256": sha256(root / "protocol/FREEZE.json"), "training_identity_sha256": sha256(root / "features/train_identity.npy"), "preprocessor_sha256": sha256(root / f"preprocessing/{cell}.json")}
    write_json(record_path, record, exclusive=True)
    append_event(root, "fit_started", run=name, family=family, cell=cell, seed=seed)
    model_path = folder / ("model.txt" if family == "LightGBM" else "model.ubj")
    started = time.perf_counter()
    try:
        with MemorySampler() as memory, (folder / "fit.log").open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            model = adapter.fit_save(family, config, seed, x, y, model_path)
            if validation_data is not None:
                vx, vy, ids = validation_data
                scores = adapter.predict_margin(model, family, vx)
                np.save(folder / "validation_margin.npy", scores, allow_pickle=False)
                record["validation"] = ScoreOrder(vy, scores, ids).evaluate()
            del model
        record.update({"status": "success", "fit_and_validation_seconds": time.perf_counter() - started, "peak_process_rss_bytes_50ms_sampled": memory.peak, "artifacts": file_records(root, [p for p in folder.iterdir() if p != record_path])})
    except Exception as exc:
        record.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "seconds": time.perf_counter() - started})
    write_json(record_path, record)
    append_event(root, "fit_finished", run=name, status=record["status"])
    return record


def train_select(root: Path, execute: bool) -> dict[str, Any]:
    permit = training_permit(root, execute)  # Must precede every label/model operation.
    require(not (root / "protocol/TRAINING_COMPLETE.json").exists(), "Training already completed")
    ids = {part: np.load(root / f"features/{part}_identity.npy", allow_pickle=False)[:, 0] for part in ("train", "validation")}
    labels = {part: labels_for_ids(root, rows, permit) for part, rows in ids.items()}
    feasibility = {part: {"rows": len(y), "positive": int(y.sum()), "negative": int((y == 0).sum()), "two_classes": len(np.unique(y)) == 2} for part, y in labels.items()}
    write_json(root / "audit/TRAIN_VALIDATION_LABEL_GATE.json", feasibility)
    require(all(len(np.unique(y)) == 2 for y in labels.values()), "Training/validation label feasibility failed; do not move boundaries")
    tuning: dict[str, list[dict[str, Any]]] = {}
    tx = np.load(root / "preprocessing/B1_train.npy", mmap_mode="r", allow_pickle=False)
    vx = np.load(root / "preprocessing/B1_validation.npy", mmap_mode="r", allow_pickle=False)
    for family in FAMILIES:
        tuning[family] = [attempt(root, f"tuning/{family}_{config['id']}", family, "B1", config, 42, tx, labels["train"], (vx, labels["validation"], ids["validation"])) for config in models.candidates(family)]
    choices = {family: models.choose_candidate(records) for family, records in tuning.items()}
    chosen_configs = {f: r["candidate"] for f, r in choices.items()}
    selection = {"time_utc": utc_now(), "freeze_sha256": permit.freeze_sha256, "chosen": chosen_configs, "rule": "validation TP@3%, AP, fewer trees, C01-C08", "tuning_records": file_records(root, list((root / "models/tuning").rglob("record.json")))}
    selection_path = root / "protocol/SELECTION.json"
    if selection_path.exists():
        old = read_json(selection_path)
        require(old["chosen"] == chosen_configs and old["freeze_sha256"] == permit.freeze_sha256, "Selection differs on resume")
        verify_records(root, old["tuning_records"])
    else:
        write_json(selection_path, selection, exclusive=True)
    del tx, vx
    outcomes = []
    for family in FAMILIES:
        for cell in CELLS:
            x = np.load(root / f"preprocessing/{cell}_train.npy", mmap_mode="r", allow_pickle=False)
            for seed in SEEDS:
                if cell == "B1" and seed == 42:
                    record = choices[family]
                    path = f"models/tuning/{family}_{record['candidate']['id']}"
                    reused = True
                else:
                    path = f"models/formal/{family}_{cell}_{seed}"
                    record = attempt(root, path.removeprefix("models/"), family, cell, chosen_configs[family], seed, x, labels["train"], None)
                    reused = False
                outcomes.append({"family": family, "cell": cell, "seed": seed, "status": record["status"], "folder": path, "reused_tuning_fit": reused})
            del x
            gc.collect()
    summary = {"status": "success" if all(r["status"] == "success" for r in outcomes) else "completed_with_failures", "freeze_sha256": permit.freeze_sha256, "selection_sha256": sha256(selection_path), "formal_conditions": outcomes, "artifacts": file_records(root, list((root / "models").rglob("*.*"))), "real_test_scoring_performed": False}
    write_json(root / "protocol/TRAINING_COMPLETE.json", summary, exclusive=True)
    append_event(root, "training_complete", status=summary["status"], formal_conditions=40)
    return {"status": summary["status"], "formal_conditions": 40}


def evaluate(root: Path, open_test: bool) -> dict[str, Any]:
    require(open_test, "BLOCKED: test access requires --open-test-once")
    training = read_json(root / "protocol/TRAINING_COMPLETE.json")
    require(training["status"] == "success", "Required formal models incomplete; test remains closed")
    permit = test_permit(root, open_test)
    frame = pd.read_parquet(root / "features/unlabelled.parquet", filters=[("split", "==", "test")])
    identity = np.load(root / "features/test_identity.npy", allow_pickle=False)
    require(np.array_equal(frame[["source_row_id", "step"]].to_numpy(), identity), "Test row alignment changed")
    y = labels_for_ids(root, identity[:, 0], permit)
    np.save(root / "predictions/test_labels.npy", y, allow_pickle=False)
    outcomes: dict[str, dict[str, Any]] = {}
    files = [root / "predictions/test_labels.npy"]
    ties = priority(identity[:, 0])
    score_files: dict[str, str] = {}
    segments: dict[str, Any] = {}
    for cell in CELLS:
        proc = Preprocessor.from_dict(read_json(root / f"preprocessing/{cell}.json"))
        x = proc.transform(frame)
        for record in training["formal_conditions"]:
            if record["cell"] != cell:
                continue
            family, seed = record["family"], record["seed"]
            key = f"{family}|{cell}|{seed}"
            model_path = root / record["folder"] / ("model.txt" if family == "LightGBM" else "model.ubj")
            scores = models.load_predict(model_path, family, x)
            target = root / f"predictions/{family}_{cell}_{seed}.npy"
            np.save(target, scores, allow_pickle=False)
            files.append(target)
            score_files[key] = str(target.relative_to(root)).replace("\\", "/")
            outcomes[key] = ScoreOrder(y, scores, identity[:, 0], ties).evaluate()
            for first, last in ((126, 143), (144, 161), (162, 179)):
                mask = (identity[:, 1] >= first) & (identity[:, 1] <= last)
                segments.setdefault(f"{first}-{last}", {})[key] = ScoreOrder(y[mask], scores[mask], identity[mask, 0], ties[mask]).evaluate()
        del x
        gc.collect()
    report = {"scope": "one completed synthetic test window", "status": "evaluated" if int(y.sum()) else "no_positive_test_not_inferential", "per_seed": outcomes, "seed_mean_and_paired_effects": paired_summary(outcomes), "descriptive_segments": segments}
    write_json(root / "analysis/test_metrics.json", report)
    files.append(root / "analysis/test_metrics.json")
    record = {"freeze_sha256": permit.freeze_sha256, "test_release_sha256": sha256(root / "protocol/TEST_RELEASE.json"), "score_files": score_files, "artifacts": file_records(root, files), "status": report["status"]}
    write_json(root / "protocol/EVALUATION_COMPLETE.json", record, exclusive=True)
    append_event(root, "test_evaluation_complete", status=record["status"])
    return {"status": record["status"], "models_scored": len(outcomes)}


def analyse(root: Path, replay_name: str | None = None) -> dict[str, Any]:
    verify_freeze(root)
    record = read_json(root / "protocol/EVALUATION_COMPLETE.json")
    require(record["freeze_sha256"] == sha256(root / "protocol/FREEZE.json"), "Evaluation freeze mismatch")
    require(record["test_release_sha256"] == sha256(root / "protocol/TEST_RELEASE.json"), "Test release changed")
    verify_records(root, record["artifacts"])
    require(record["status"] == "evaluated", "No inferential analysis for unevaluable test")
    require(replay_name is None or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", replay_name) is not None, "Replay name must be a short lowercase identifier")
    out = root / "analysis" / ("bootstrap" if replay_name is None else f"replay_{replay_name}")
    require(not out.exists(), "Bootstrap output already exists; replay must use a separate directory without altering this run")
    identity = np.load(root / "features/test_identity.npy", allow_pickle=False)
    y = np.load(root / "predictions/test_labels.npy", allow_pickle=False)
    ties = priority(identity[:, 0])
    orders = {key: ScoreOrder(y, np.load(root / path, allow_pickle=False), identity[:, 0], ties) for key, path in record["score_files"].items()}
    result = bootstrap_orders(orders, identity[:, 1], out, grid=np.arange(126, 180))
    append_event(root, "fixed_prediction_bootstrap_complete", draws_each=5000, block_lengths=[7, 3, 14])
    receipt = {"status": result["status"], "artifacts": file_records(root, list(out.iterdir())), "estimand_scope": "conditional_fixed_models_and_window", "new_independent_evidence": False, "evaluation_sha256": sha256(root / "protocol/EVALUATION_COMPLETE.json")}
    write_json(out / "analysis_receipt.json", receipt)
    return {"status": result["status"], "output": str(out), "new_independent_evidence": False}
