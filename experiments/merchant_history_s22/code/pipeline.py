"""Single ordered controller; phase-one failure irrevocably blocks real execution."""

from __future__ import annotations

import csv
import gc
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, cast

from native_common import (
    CELL_COLUMNS,
    CELLS,
    ROOT,
    events,
    read_json,
    require,
    sha,
    utc,
    verify_lock,
    write_json,
)


def install_scope_audit(config: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {"phase": "artificial", "authorized_source_opens": 0}
    run = os.path.normcase(str(ROOT.resolve()))
    declared_workspaces = config.get("workspace_roots")
    require(
        isinstance(declared_workspaces, list) and bool(declared_workspaces),
        "Fresh reconstruction must explicitly declare its workspace roots",
    )
    declared_workspaces = cast(list[str], declared_workspaces)
    workspaces = [os.path.normcase(os.path.abspath(p)) for p in declared_workspaces]
    environment = os.path.normcase(str(Path(config["python"]).parents[1].resolve()))
    source = os.path.normcase(str(Path(config["source"]["archive"]).resolve()))
    references = {os.path.normcase(str(Path(p).resolve())) for p in config["reference"].values()}

    def inside(path: str, directory: str) -> bool:
        return path == directory or path.startswith(directory + os.sep)

    def audit(event: str, arguments: Any) -> None:
        if event != "open" or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        path = os.path.normcase(os.path.abspath(os.fsdecode(arguments[0])))
        if inside(path, run) or inside(path, environment) or path in references:
            return
        if path == source and state["phase"] == "project":
            state["authorized_source_opens"] += 1
            return
        if any(inside(path, workspace) for workspace in workspaces):
            events("scope_open_denied", phase=state["phase"], path=path)
            raise RuntimeError("Python audit denied a project path outside the input list")

    sys.addaudithook(audit)
    write_json(
        ROOT / "audit/PYTHON_SCOPE_AUDIT.json",
        {
            "installed_utc": utc(),
            "covered_process": os.getpid(),
            "mechanism": (
                "Python open event with declared workspace allowlist; not full OS isolation"
            ),
            "child_processes": (
                "fixed, code-hashed workers; OS job supervises resources and termination"
            ),
            "limitations": (
                "native DLL IO, external processes and OS security boundaries are not sandboxed"
            ),
        },
        exclusive=True,
    )
    return state


def elapsed_limit(started: float, seconds: float, name: str) -> float:
    elapsed = time.perf_counter() - started
    require(elapsed <= seconds, f"{name} stage wall budget exceeded: {elapsed:.6f} > {seconds}")
    return elapsed


def write_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_row(cell: str, result: dict[str, Any], **extra: Any) -> dict[str, Any]:
    tie = result["boundary_tie"]
    return {
        "cell": cell,
        "model_id": {
            "A0": "directM_off_Hnone",
            "B0": "directM_on_Hnone",
            "A1": "directM_off_H7",
            "B1": "directM_on_H7",
            "A2": "directM_off_Horig",
            "B2": "directM_on_Horig",
        }[cell],
        **{key: result[key] for key in ("N", "P", "K", "TP", "Recall", "Precision", "AP")},
        "boundary_margin": tie.get("margin"),
        "boundary_tied_rows": tie.get("tied_rows"),
        "boundary_tied_positives": tie.get("tied_positives"),
        "boundary_slots": tie.get("slots_from_tie"),
        "boundary_TP_min": tie.get("tp_min"),
        "boundary_TP_max": tie.get("tp_max"),
        **extra,
    }


def run_pipeline() -> None:
    started = time.perf_counter()
    config = read_json(ROOT / "config/EXECUTION_CONFIG.json")
    status: dict[str, Any] = {
        "status": "STARTED",
        "stage": "phase1",
        "start_utc": utc(),
        "project_model_fits_entered": 0,
        "project_preprocessor_fits_entered": 0,
        "completed_cells": [],
        "review_labels_used": False,
        "retries": 0,
    }
    costs: dict[str, Any] = {"shared": {}, "preprocessing": {}, "training": {}, "review": {}}
    state = install_scope_audit(config)
    write_json(ROOT / "audit/PIPELINE_STATUS.json", status)
    write_json(ROOT / "audit/COSTS.json", costs)
    try:
        from phase1 import run_phase1

        native = run_phase1(config)
        costs["shared"]["native_artificial_acceptance_seconds"] = time.perf_counter() - started
        require(native["status"] == "PASS", "phase-one acceptance not fully passed")
        verify_lock()
        status.update(stage="source_preparation", native_acceptance="PASS")
        write_json(ROOT / "audit/PIPELINE_STATUS.json", status)
        state["phase"] = "project"
        events("real_source_access_enabled_after_native_pass")
        import numpy as np
        from native_batch import COUNTERS, NativePreprocessor, build_history, contrasts, metrics
        from source_io import read_labels, read_public
        from supervision import run_supervised

        source_started = time.perf_counter()
        public = read_public(config)
        costs["shared"]["raw_source_hash_projection_normalization_seconds"] = (
            time.perf_counter() - source_started
        )
        costs["shared"]["phase1_and_input_seconds"] = elapsed_limit(
            started, config["caps"]["stage1_and_input_seconds"], "phase1 plus input"
        )
        identities: dict[str, Any] = {}
        for region in ("train", "review"):
            mask = public["split"].eq(region)
            identities[region] = public.loc[mask, ["source_row_id", "step"]].to_numpy(
                dtype=np.int64
            )
            np.save(
                ROOT / "features" / f"{region}_identity.npy", identities[region], allow_pickle=False
            )
        require(
            len(identities["train"]) == config["splits"]["train"][2]
            and len(identities["review"]) == config["splits"]["review"][2],
            "common matrix identities disagree with contract",
        )
        status.update(stage="history", real_source_public_rows=len(public))
        write_json(ROOT / "audit/PIPELINE_STATUS.json", status)
        history_started = time.perf_counter()
        frames: dict[str, Any] = {"H0": public}
        history_audit: dict[str, Any] = {}
        for mode in ("HS", "HL"):
            frame, record = build_history(public, mode)
            frames[mode] = frame
            history_audit[mode] = record
            require(
                np.array_equal(
                    frame[["source_row_id", "step"]].to_numpy(),
                    public[["source_row_id", "step"]].to_numpy(),
                ),
                "history mode changed common source identity",
            )
        costs["shared"]["history_total_seconds"] = elapsed_limit(
            history_started, config["caps"]["history_seconds"], "history"
        )
        costs["shared"]["history_modes_once"] = history_audit
        costs["shared"]["raw_history_scan_saving_claim"] = False
        write_json(ROOT / "audit/HISTORY_CONSTRUCTION.json", history_audit, exclusive=True)
        write_json(ROOT / "audit/COSTS.json", costs)
        status.update(stage="preprocessing")
        write_json(ROOT / "audit/PIPELINE_STATUS.json", status)
        pre_start = time.perf_counter()
        snapshots: dict[str, Any] = {}
        ledger: dict[str, Any] = {"fit_calls_entered": 0, "attempts": []}
        write_json(ROOT / "audit/PREPROCESSING_ATTEMPTS.json", ledger, exclusive=True)
        matrices: dict[str, Any] = {}
        for cell in config["cells"]:
            frame = frames[CELLS[cell][1]]
            train = frame.loc[frame["split"].eq("train")].copy()
            review = frame.loc[frame["split"].eq("review")].copy()
            train.attrs["purpose"] = review.attrs["purpose"] = "project"
            require(
                np.array_equal(train[["source_row_id", "step"]].to_numpy(), identities["train"])
                and np.array_equal(
                    review[["source_row_id", "step"]].to_numpy(), identities["review"]
                ),
                f"{cell} train/review source identity mismatch",
            )
            processor = NativePreprocessor(CELL_COLUMNS[cell])
            entry = {"cell": cell, "status": "FIT_ENTERED", "time_utc": utc()}
            ledger["attempts"].append(entry)
            ledger["fit_calls_entered"] += 1
            write_json(ROOT / "audit/PREPROCESSING_ATTEMPTS.json", ledger)
            status["project_preprocessor_fits_entered"] = ledger["fit_calls_entered"]
            write_json(ROOT / "audit/PIPELINE_STATUS.json", status)
            fit_started = time.perf_counter()
            processor.fit(train)
            fit_seconds = time.perf_counter() - fit_started
            snapshot = processor.snapshot()
            snapshots[cell] = snapshot
            write_json(ROOT / "preprocessing" / f"{cell}.json", snapshot, exclusive=True)
            row: dict[str, Any] = {
                "fit_seconds": fit_seconds,
                "encoded_columns": len(snapshot["names"]),
                "predictor_columns": snapshot["columns"],
                "transforms": {},
            }
            for region, local in (("train", train), ("review", review)):
                transform_start = time.perf_counter()
                matrix = processor.transform(local)
                transform_seconds = time.perf_counter() - transform_start
                save_start = time.perf_counter()
                path = ROOT / "features" / f"{cell}_{region}.npy"
                np.save(path, matrix, allow_pickle=False)
                save_seconds = time.perf_counter() - save_start
                row["transforms"][region] = {
                    "seconds": transform_seconds,
                    "save_seconds": save_seconds,
                    "shape": list(matrix.shape),
                    "bytes": path.stat().st_size,
                    "sha256": sha(path),
                    "identity_sha256": sha(ROOT / "features" / f"{region}_identity.npy"),
                }
                del matrix
            restored = NativePreprocessor.restore(
                read_json(ROOT / "preprocessing" / f"{cell}.json")
            )
            require(
                restored.snapshot() == snapshot, "project preprocessing persistence changed rules"
            )
            entry.update(status="SUCCESS", fit_seconds=fit_seconds, end_utc=utc())
            write_json(ROOT / "audit/PREPROCESSING_ATTEMPTS.json", ledger)
            costs["preprocessing"][cell] = row
            write_json(ROOT / "audit/COSTS.json", costs)
            del train, review, processor, restored
        for off, on in (("A0", "B0"), ("A1", "B1"), ("A2", "B2")):
            other = [r for r in snapshots[on]["rules"] if r["column"] != "merchant"]
            names_on = snapshots[on]["names"]
            selection = [
                j for j, name in enumerate(names_on) if not name.startswith("cat:merchant=")
            ]
            require(
                snapshots[off]["rules"] == other
                and [names_on[j] for j in selection] == snapshots[off]["names"],
                f"{off}/{on} common preprocessing rule mismatch",
            )
            for region in ("train", "review"):
                left = np.load(
                    ROOT / "features" / f"{off}_{region}.npy", mmap_mode="r", allow_pickle=False
                )
                right = np.load(
                    ROOT / "features" / f"{on}_{region}.npy", mmap_mode="r", allow_pickle=False
                )
                for i in range(0, len(left), 8192):
                    require(
                        np.array_equal(left[i : i + 8192], right[i : i + 8192][:, selection]),
                        f"{off}/{on} common encoded values differ",
                    )
                del left, right
        require(
            COUNTERS["project_preprocessor_fits"] == 6, "project preprocessing fit count mismatch"
        )
        costs["shared"]["preprocessing_total_seconds"] = elapsed_limit(
            pre_start, config["caps"]["preprocessing_seconds"], "preprocessing"
        )
        write_json(
            ROOT / "audit/COMMON_COLUMN_ALIGNMENT.json",
            {
                "status": "PASS",
                "pairs": [["A0", "B0"], ["A1", "B1"], ["A2", "B2"]],
                "train_and_review_exact_float32": True,
                "all_six_common_identities": True,
                "no_history_equality_required_across_H_conditions": True,
                "train_identity_sha256": sha(ROOT / "features/train_identity.npy"),
                "review_identity_sha256": sha(ROOT / "features/review_identity.npy"),
            },
            exclusive=True,
        )
        del frames, public, matrices
        gc.collect()
        label_started = time.perf_counter()
        train_labels = read_labels(config, identities["train"], "train")
        del train_labels
        costs["shared"]["train_label_projection_seconds"] = time.perf_counter() - label_started
        status.update(stage="six_fits")
        write_json(ROOT / "audit/PIPELINE_STATUS.json", status)
        fits: dict[str, Any] = {"reserved_attempts": 0, "current_cell": None, "attempts": []}
        write_json(ROOT / "audit/FIT_ATTEMPTS.json", fits, exclusive=True)
        total_fit_wall = 0.0
        for cell in config["cells"]:
            fits["reserved_attempts"] += 1
            fits["current_cell"] = cell
            attempt = {
                "cell": cell,
                "attempt": fits["reserved_attempts"],
                "status": "LAUNCH_RESERVED",
                "reserved_utc": utc(),
            }
            fits["attempts"].append(attempt)
            write_json(ROOT / "audit/FIT_ATTEMPTS.json", fits)
            payload = ROOT / "config" / f"train_{cell}.json"
            write_json(payload, {"cell": cell}, exclusive=True)
            resource = run_supervised(
                "train",
                payload,
                f"train_{cell}",
                seconds=config["caps"]["fit_worker_seconds"],
                rss_limit=config["caps"]["rss_bytes"],
                disk_limit=config["caps"]["disk_bytes"],
            )
            total_fit_wall += resource["seconds"]
            attempt.update(stop_reason=resource["stop_reason"], seconds=resource["seconds"])
            worker_path = ROOT / "models" / cell / "WORKER.json"
            worker = read_json(worker_path) if worker_path.exists() else {}
            status["project_model_fits_entered"] += worker.get("fit_calls_entered", 0)
            write_json(ROOT / "audit/PIPELINE_STATUS.json", status)
            costs["training"][cell] = {"resource": resource, "worker": worker}
            write_json(ROOT / "audit/COSTS.json", costs)
            attempt["status"] = worker.get("status", "NO_WORKER_RECEIPT")
            write_json(ROOT / "audit/FIT_ATTEMPTS.json", fits)
            require(
                resource["stop_reason"] == "NORMAL_EXIT" and worker.get("status") == "SUCCESS",
                f"{cell} worker failed or stopped: {resource['stop_reason']}",
            )
            require(
                worker["training_check_exact"] and worker["fit_calls_entered"] == 1,
                "project fit/save/reload receipt incomplete",
            )
            require(
                total_fit_wall <= config["caps"]["fit_worker_cumulative_seconds"],
                "cumulative training worker budget exceeded",
            )
            status["completed_cells"].append(cell)
            write_json(ROOT / "audit/PIPELINE_STATUS.json", status)
        require(status["project_model_fits_entered"] == 6, "model fit count not exactly six")
        lock: dict[str, Any] = {
            "time_utc": utc(),
            "all_six_models_and_preprocessors_locked": True,
            "models": {},
            "preprocessors": {},
            "matrices": {},
            "cells": config["cells"],
            "label_evaluation_not_started": True,
            "review_identity_sha256": sha(ROOT / "features/review_identity.npy"),
        }
        for cell in config["cells"]:
            lock["models"][cell] = {"sha256": sha(ROOT / "models" / cell / "model.txt")}
            lock["preprocessors"][cell] = sha(ROOT / "preprocessing" / f"{cell}.json")
            lock["matrices"][cell] = {
                region: sha(ROOT / "features" / f"{cell}_{region}.npy")
                for region in ("train", "review")
            }
        write_json(ROOT / "config/MODELS_LOCK.json", lock, exclusive=True)
        verify_lock()
        events("all_six_models_and_preprocessors_locked_before_review")
        status.update(stage="review_prediction")
        write_json(ROOT / "audit/PIPELINE_STATUS.json", status)
        review_started = time.perf_counter()
        for cell in config["cells"]:
            payload = ROOT / "config" / f"predict_{cell}.json"
            write_json(payload, {"cell": cell}, exclusive=True)
            resource = run_supervised(
                "predict",
                payload,
                f"predict_{cell}",
                seconds=config["caps"]["per_evaluation_worker_seconds"],
                rss_limit=config["caps"]["rss_bytes"],
                disk_limit=config["caps"]["disk_bytes"],
            )
            pred_path = ROOT / "results" / cell / "PREDICTION.json"
            prediction = read_json(pred_path) if pred_path.exists() else {}
            costs["review"][cell] = {"resource": resource, "prediction": prediction}
            write_json(ROOT / "audit/COSTS.json", costs)
            require(
                resource["stop_reason"] == "NORMAL_EXIT" and prediction.get("status") == "SUCCESS",
                f"{cell} review prediction failed or stopped: {resource['stop_reason']}",
            )
        # No review label use or performance comparison until every model and score is fixed.
        write_json(
            ROOT / "config/SCORES_LOCK.json",
            {
                "time_utc": utc(),
                "scores": {
                    c: sha(ROOT / "results" / c / "review_margin.npy") for c in config["cells"]
                },
                "all_six_scores_fixed_before_review_label_use": True,
            },
            exclusive=True,
        )
        label_started = time.perf_counter()
        y = read_labels(config, identities["review"], "review")
        costs["shared"]["review_label_projection_seconds"] = time.perf_counter() - label_started
        status.update(stage="descriptive_evaluation", review_labels_used=True)
        write_json(ROOT / "audit/PIPELINE_STATUS.json", status)
        ids, steps = identities["review"][:, 0], identities["review"][:, 1]
        scores = {
            c: np.load(ROOT / "results" / c / "review_margin.npy", allow_pickle=False)
            for c in config["cells"]
        }
        whole: dict[str, Any] = {}
        segments: dict[str, Any] = {}
        calculation_started = time.perf_counter()
        for cell in config["cells"]:
            require(len(scores[cell]) == len(y) == len(ids), "review score/source/label alignment")
            whole[cell] = metrics(y, scores[cell], ids)
        effects = contrasts(whole)
        recall_delta0 = whole["A2"]["Recall"] - whole["A1"]["Recall"]
        recall_delta1 = whole["B2"]["Recall"] - whole["B1"]["Recall"]
        recall_interaction = recall_delta1 - recall_delta0
        for observed, integer_value in (
            (recall_delta0, effects["delta0"]),
            (recall_delta1, effects["delta1"]),
            (recall_interaction, effects["I"]),
        ):
            require(
                math.isclose(observed, integer_value, rel_tol=1e-12, abs_tol=1e-12),
                "Recall arithmetic disagrees with integer capture contrast",
            )
        effects["floating_Recall_arithmetic_check"] = {
            "delta0": recall_delta0,
            "delta1": recall_delta1,
            "I": recall_interaction,
            "maximum_absolute_rounding_difference": max(
                abs(recall_delta0 - effects["delta0"]),
                abs(recall_delta1 - effects["delta1"]),
                abs(recall_interaction - effects["I"]),
            ),
        }
        require(
            effects["I"] == effects["J"] / effects["common_P"],
            "integer interaction conversion mismatch",
        )
        for lower, upper in config["segments"]:
            key = f"{lower}-{upper}"
            mask = (steps >= lower) & (steps <= upper)
            local = {c: metrics(y[mask], scores[c][mask], ids[mask]) for c in config["cells"]}
            segments[key] = {
                "rows": local,
                "effects": contrasts(local),
                "independent_segment_capacity": True,
                "segment_capture_differences_need_not_sum_to_main": True,
            }
        costs["shared"]["ranking_AP_contrasts_whole_and_segments_seconds"] = (
            time.perf_counter() - calculation_started
        )
        results = {
            "status": "COMPLETE_DESCRIPTIVE_RETROSPECTIVE",
            "whole": whole,
            "effects": effects,
            "segments": segments,
            "capacity": config["capacity"],
            "tie": config["tie"],
            "independent_final_validation": False,
            "bootstrap": 0,
            "significance_inference": False,
        }
        write_json(ROOT / "results/DESCRIPTIVE_RESULTS.json", results, exclusive=True)
        columns = list(metric_row("A0", whole["A0"]))
        write_table(
            ROOT / "results/六组结果.csv",
            [metric_row(c, whole[c]) for c in config["cells"]],
            columns,
        )
        write_table(
            ROOT / "results/固定三分段.csv",
            [
                metric_row(c, block["rows"][c], segment=segment)
                for segment, block in segments.items()
                for c in config["cells"]
            ],
            ["segment", *columns],
        )
        write_table(
            ROOT / "results/预定差值.csv",
            [
                {"window": window, **block}
                for window, block in [
                    ("126-179", effects),
                    *[(s, value["effects"]) for s, value in segments.items()],
                ]
            ],
            ["window", "common_P", "delta0_TP", "delta1_TP", "J", "delta0", "delta1", "I"],
        )
        costs["shared"]["review_total_seconds"] = elapsed_limit(
            review_started, config["caps"]["evaluation_seconds"], "evaluation"
        )
        costs["shared"]["training_workers_total_external_seconds"] = total_fit_wall
        costs["shared"]["pipeline_elapsed_seconds"] = time.perf_counter() - started
        write_json(ROOT / "audit/COSTS.json", costs)
        status.update(
            status="COMPLETED",
            stage="normal_close",
            end_utc=utc(),
            evaluation_status=results["status"],
            project_model_fits_entered=6,
            project_preprocessor_fits_entered=6,
        )
        write_json(ROOT / "audit/PIPELINE_STATUS.json", status)
        events("authorized_six_cell_task_completed", fits=6, preprocessors=6)
    except BaseException as exc:
        status.update(
            status="STOPPED_NO_RETRY",
            end_utc=utc(),
            error=f"{type(exc).__name__}:{exc}",
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(ROOT / "audit/PIPELINE_STATUS.json", status)
        write_json(ROOT / "audit/COSTS.json", costs)
        (ROOT / "audit/FAILURE_TRACEBACK.txt").write_text(traceback.format_exc(), encoding="utf-8")
        events("authorized_task_stopped", stage=status["stage"], error=status["error"])
        raise
    finally:
        write_json(ROOT / "audit/SCOPE_ACCESS_SUMMARY.json", state)
