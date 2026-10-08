"""Complete LR effects and LR-minus-LightGBM effects on identical frozen draws."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fraudx.cross_model.statistics import build_view, calculate, components, transferred
from fraudx.paysim_deterministic.training import decision_metrics
from fraudx.q2_extension.bootstrap import interval_row
from fraudx.q2_extension.common import (
    SEEDS,
    array_hash,
    assert_aligned,
    file_hash,
    read_json,
    require,
    write_json,
)
from fraudx.q2_extension.kernels import capacity_profile, stage_metrics


def score_values(frame: pd.DataFrame, stage: str, family: str) -> np.ndarray:
    if stage == "clipped":
        source = "raw_probability" if family == "LR" else "raw"
        return np.clip(frame[source].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    column = "raw_probability" if stage == "raw" and family == "LR" else stage
    return frame[column].to_numpy(dtype=float)


def load_lgbm(
    spec: dict[str, Any], dataset: str, fold: int, model: str, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if dataset == "paysim":
        path = Path(spec["predictions"]) / f"paysim_{fold}_{model}_{seed}_test.parquet"
        frame = pd.read_parquet(path)
        meta = read_json(path.with_name(path.name.replace("_test.parquet", ".json")))
    else:
        frame = pd.read_parquet(
            spec["predictions"],
            filters=[("fold", "=", fold), ("model_id", "=", model), ("seed", "=", seed)],
        )
        meta = {}
    return frame.sort_values(spec["id"], kind="stable").reset_index(drop=True), meta


def observed_rows(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    key: dict[str, Any],
    config: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[list, list]:
    rows, scenarios = [], []
    ids, labels = frame[spec["id"]].to_numpy(), frame.isFraud.to_numpy(np.int8)
    for stage in spec["stages"]:
        scores = score_values(frame, stage, key["family"])
        metrics = stage_metrics(scores, labels)
        for q in config["capacities"]:
            profile, _ = capacity_profile(ids, labels, scores, q)
            rows.append({**key, "score_stage": stage, **metrics, **profile})
    if key["dataset"] == "paysim":
        entries = meta["scenarios"] if key["family"] == "LR" else meta["rows"]
        for item in entries:
            decisions = frame.platt.to_numpy() >= item["threshold"]
            scenarios.append(
                {
                    **key,
                    "capacity": item["capacity"],
                    "fn_cost": item["fn_cost"],
                    "threshold": item["threshold"],
                    "validation_alert_rate": item["validation_alert_rate"],
                    **decision_metrics(labels, decisions, item["fn_cost"]),
                }
            )
    return rows, scenarios


def threshold_replicates(
    frame: pd.DataFrame, positions: np.ndarray, weights: np.ndarray, threshold: float
) -> pd.DataFrame:
    labels = frame.isFraud.to_numpy(np.int64)
    selected = (frame.platt.to_numpy() >= threshold).astype(np.int64)
    group_f = np.bincount(positions, weights=labels, minlength=weights.shape[1]).astype(np.int64)
    group_tp = np.bincount(positions, weights=labels * selected, minlength=weights.shape[1]).astype(
        np.int64
    )
    group_fp = np.bincount(
        positions, weights=(1 - labels) * selected, minlength=weights.shape[1]
    ).astype(np.int64)
    tp, fp, frauds = weights @ group_tp, weights @ group_fp, weights @ group_f
    return pd.DataFrame(
        {"threshold_tp": tp, "threshold_fp": fp, "threshold_cost": fp + 100 * (frauds - tp)}
    )


def calculate_dataset(config_path: Path, dataset: str) -> None:
    config = read_json(config_path)
    spec = config["datasets"][dataset]
    root = Path(config["output_root"])
    folder = root / "analysis" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    signature = {
        "protocol_sha256": file_hash(config_path),
        "analysis": file_hash(Path(__file__)),
        "statistics": file_hash(Path(build_view.__code__.co_filename)),
    }
    points, scenarios, alignments = [], [], []
    for fold in (1, 2, 3):
        weights_path = Path(spec["draws"]) / f"{dataset}_{fold}_multiplicities.npy"
        weights = np.load(weights_path, mmap_mode="r")
        require(weights.shape[0] == 5000, "Changed draw count")
        for model in spec["models"]:
            trained = root / "training" / dataset / f"{dataset}_{fold}_{model}"
            meta = read_json(trained / "COMPLETED.json")
            if not meta["converged"]:
                write_json(
                    folder / f"{fold}_{model}_FAILED.json",
                    {
                        "reason": "Frozen retry did not converge; no formal inference",
                        "model": model,
                    },
                )
                continue
            destination = folder / f"{fold}_{model}"
            done = destination / "COMPLETED.json"
            if done.exists():
                record = read_json(done)
                require(record["signature"] == signature, "Analysis resume code changed")
                for item in record["files"]:
                    require(
                        file_hash(destination / item["name"]) == item["sha256"],
                        "Analysis artifact changed",
                    )
                points.extend(read_json(destination / "points.json"))
                scenarios.extend(read_json(destination / "scenarios.json"))
                alignments.extend(read_json(destination / "alignments.json"))
                continue
            require(not destination.exists(), "Preserve incomplete analysis")
            destination.mkdir()
            started = time.monotonic()
            frame = (
                pd.read_parquet(trained / "test.parquet")
                .sort_values(spec["id"], kind="stable")
                .reset_index(drop=True)
            )
            ids, labels = frame[spec["id"]].to_numpy(), frame.isFraud.to_numpy(np.int8)
            times, positions = np.unique(frame[spec["time"]].to_numpy(), return_inverse=True)
            require(len(times) == weights.shape[1], "Changed time-block population")
            keys = [spec["id"], spec["time"], "isFraud"]
            key = {"dataset": dataset, "fold": fold, "model": model}
            model_points, model_scenarios = observed_rows(
                frame, spec, {**key, "family": "LR", "seed": 0}, config, meta
            )
            model_alignments = []
            for seed in SEEDS:
                reference, old_meta = load_lgbm(spec, dataset, fold, model, seed)
                assert_aligned(frame, reference, keys)
                model_alignments.append(
                    {
                        **key,
                        "lgbm_seed": seed,
                        "id_time_label_hash": array_hash(frame[keys].to_numpy()),
                        "rows": len(frame),
                        "aligned": True,
                    }
                )
                point, scenario = observed_rows(
                    reference, spec, {**key, "family": "LightGBM", "seed": seed}, config, old_meta
                )
                model_points.extend(point)
                model_scenarios.extend(scenario)
                del reference
            last_partition, last_result = None, None
            for stage in spec["stages"]:
                scores = score_values(frame, stage, "LR")
                with threadpool_limits(limits=1):
                    view = build_view(ids, labels, scores, positions, len(times))
                    if last_partition is not None and np.array_equal(
                        view.partition, last_partition
                    ):
                        if last_result is None:
                            raise ValueError("Missing equivalent-stage result")
                        result = last_result.copy()
                    else:
                        result = calculate(view, positions, labels, weights)
                    last_partition, last_result = view.partition, result.copy()
                    # One original-population draw must match sklearn and exact top-k counting.
                    original = calculate(
                        view, positions, labels, np.ones((1, len(times)), dtype=np.int16)
                    ).iloc[0]
                    kernel_point = next(
                        x
                        for x in model_points
                        if x["family"] == "LR" and x["q"] == 0.03 and x["score_stage"] == stage
                    )
                    require(
                        bool(
                            np.isclose(original.ap, kernel_point["ap"], rtol=1e-10, atol=1e-12)
                            and original.tp == kernel_point["tp"]
                        ),
                        "Original metric kernel mismatch",
                    )
                    if dataset == "paysim" and stage == "platt":
                        threshold = next(
                            x["threshold"]
                            for x in meta["scenarios"]
                            if x["capacity"] == 0.03 and x["fn_cost"] == 100
                        )
                        result = pd.concat(
                            [result, threshold_replicates(frame, positions, weights, threshold)],
                            axis=1,
                        )
                reference_cache = pd.read_parquet(
                    Path(spec["lgbm_replicates"]) / f"{dataset}_{fold}_{model}_42_{stage}.parquet"
                )
                require(
                    np.array_equal(
                        result[["replicate_id", "n", "frauds", "alerts"]],
                        reference_cache[["replicate_id", "n", "frauds", "alerts"]],
                    ),
                    "LR/LightGBM draw pairing mismatch",
                )
                result.to_parquet(destination / f"{stage}_replicates.parquet", index=False)
                print(f"Bootstrap {dataset}/{fold}/{model}/{stage}", flush=True)
                del view
                gc.collect()
            for name, value in [
                ("points", model_points),
                ("scenarios", model_scenarios),
                ("alignments", model_alignments),
            ]:
                write_json(destination / f"{name}.json", value)
            write_json(
                done,
                {
                    "signature": signature,
                    "seconds": time.monotonic() - started,
                    "files": [
                        {"name": p.name, "sha256": file_hash(p)}
                        for p in destination.iterdir()
                        if p.is_file()
                    ],
                },
            )
            points.extend(model_points)
            scenarios.extend(model_scenarios)
            alignments.extend(model_alignments)
    point_frame = (
        pd.DataFrame(points)
        if points
        else pd.DataFrame(
            columns=[
                "dataset",
                "fold",
                "model",
                "family",
                "seed",
                "score_stage",
                "q",
                "ap",
                "tp",
                "recall",
            ]
        )
    )
    point_frame.to_csv(folder / "model_points.csv", index=False)
    scenario_frame = (
        pd.DataFrame(scenarios)
        if scenarios
        else pd.DataFrame(
            columns=[
                "dataset",
                "fold",
                "model",
                "family",
                "seed",
                "capacity",
                "fn_cost",
                "recall",
                "cost",
                "alert_rate",
            ]
        )
    )
    scenario_frame.to_csv(folder / "threshold_scenarios.csv", index=False)
    alignment_frame = (
        pd.DataFrame(alignments)
        if alignments
        else pd.DataFrame(
            columns=[
                "dataset",
                "fold",
                "model",
                "lgbm_seed",
                "id_time_label_hash",
                "rows",
                "aligned",
            ]
        )
    )
    alignment_frame.to_csv(folder / "paired_input_checks.csv", index=False)
    summarize(config_path, dataset)


def point_effect(base: pd.Series, candidate: pd.Series, transfer: bool = False) -> dict[str, float]:
    if transfer:
        return {
            "delta_recall": candidate["recall"] - base["recall"],
            "cost_reduction": base["cost"] - candidate["cost"],
            "delta_alert_rate": candidate["alert_rate"] - base["alert_rate"],
        }
    return {
        "delta_ap": candidate["ap"] - base["ap"],
        "delta_tp": candidate["tp"] - base["tp"],
        "delta_recall": candidate["recall"] - base["recall"],
    }


def summarize(config_path: Path, dataset: str) -> None:
    config = read_json(config_path)
    spec = config["datasets"][dataset]
    folder = Path(config["output_root"]) / "analysis" / dataset
    points = pd.read_csv(folder / "model_points.csv", float_precision="round_trip")
    points = points[points.q == 0.03]
    scenarios = (
        pd.read_csv(folder / "threshold_scenarios.csv", float_precision="round_trip")
        if dataset == "paysim"
        else pd.DataFrame()
    )
    interval_rows, ledgers, observed_rows_out, coverage = [], [], [], []
    contrasts = spec["core_contrasts"] + spec["auxiliary_contrasts"]
    core_ids = {x["id"] for x in spec["core_contrasts"]}
    for fold in (1, 2, 3):
        for contrast in contrasts:
            for stage in spec["stages"]:
                pair = [contrast["baseline"], contrast["candidate"]]
                paths = [folder / f"{fold}_{m}" / f"{stage}_replicates.parquet" for m in pair]
                missing = [m for m, p in zip(pair, paths, strict=True) if not p.exists()]
                for unavailable in missing:
                    meta = read_json(
                        Path(config["output_root"])
                        / "training"
                        / dataset
                        / f"{dataset}_{fold}_{unavailable}/COMPLETED.json"
                    )
                    require(not meta["converged"], "A converged condition lacks bootstrap output")
                coverage.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "contrast": contrast["id"],
                        "score_stage": stage,
                        "estimable": not missing,
                        "unavailable_models": ";".join(missing),
                        "reason": "" if not missing else "nonconverged component",
                    }
                )
                if not all(p.exists() for p in paths):
                    continue
                frames = [pd.read_parquet(p) for p in paths]
                sets = {"fixed_budget": components(*frames)}
                is_transfer = (
                    dataset == "paysim" and stage == "platt" and contrast["id"] == "P_PRIMARY"
                )
                if is_transfer:
                    sets["threshold_transfer"] = transferred(*frames)
                for rule, effect in sets.items():
                    metrics = (
                        ["delta_ap", "delta_tp", "delta_recall"]
                        if rule == "fixed_budget"
                        else ["delta_ap", "delta_recall", "cost_reduction", "delta_alert_rate"]
                    )
                    psubset = points[(points.fold == fold) & (points.score_stage == stage)]
                    observed = {}
                    old_effects = []
                    for family, seed in [("LR", 0), *[("LightGBM", s) for s in SEEDS]]:
                        subgroup = psubset[(psubset.family == family) & (psubset.seed == seed)]
                        values = point_effect(
                            *[subgroup[subgroup.model == m].iloc[0] for m in pair]
                        )
                        if rule == "threshold_transfer":
                            sub = scenarios[
                                (scenarios.fold == fold)
                                & (scenarios.family == family)
                                & (scenarios.seed == seed)
                                & (scenarios.capacity == 0.03)
                                & (scenarios.fn_cost == 100)
                            ]
                            values.update(
                                point_effect(
                                    sub[sub.model == pair[0]].iloc[0],
                                    sub[sub.model == pair[1]].iloc[0],
                                    transfer=True,
                                )
                            )
                        observed[(family, seed)] = values
                        if family == "LightGBM":
                            cached = [
                                pd.read_parquet(
                                    Path(spec["lgbm_replicates"])
                                    / f"{dataset}_{fold}_{m}_{seed}_{stage}.parquet"
                                )
                                for m in pair
                            ]
                            old_effects.append(
                                components(*cached)
                                if rule == "fixed_budget"
                                else transferred(*cached)
                            )
                    key = {
                        "dataset": dataset,
                        "fold": fold,
                        "contrast": contrast["id"],
                        "role": "core" if contrast["id"] in core_ids else "auxiliary",
                        "score_stage": stage,
                        "decision_rule": rule,
                    }
                    for metric in metrics:
                        lr = effect[metric].to_numpy()
                        lgb = np.mean(np.stack([x[metric].to_numpy() for x in old_effects]), axis=0)
                        vector_values = {"LR": lr, "LightGBM": lgb, "LR_minus_LightGBM": lr - lgb}
                        estimates = {
                            "LR": observed[("LR", 0)][metric],
                            "LightGBM": float(
                                np.mean([observed[("LightGBM", s)][metric] for s in SEEDS])
                            ),
                        }
                        estimates["LR_minus_LightGBM"] = estimates["LR"] - estimates["LightGBM"]
                        for family, vector in vector_values.items():
                            row_key = {**key, "family": family, "metric": metric}
                            interval_rows.append(interval_row(vector, estimates[family], row_key))
                            observed_rows_out.append({**row_key, "estimate": estimates[family]})
                            ledgers.append(
                                pd.DataFrame(
                                    {
                                        **row_key,
                                        "replicate_id": effect.replicate_id,
                                        "value": vector,
                                    }
                                )
                            )
    keys = [
        "dataset",
        "fold",
        "contrast",
        "role",
        "score_stage",
        "decision_rule",
        "family",
        "metric",
    ]
    columns = [*keys, "estimate", "ci_lower", "ci_upper", "draws", "valid", "invalid", "ci_type"]
    pd.DataFrame(interval_rows, columns=columns).to_csv(
        folder / "effect_intervals.csv", index=False
    )
    pd.DataFrame(observed_rows_out, columns=[*keys, "estimate"]).to_csv(
        folder / "observed_effects.csv", index=False
    )
    replicas = (
        pd.concat(ledgers, ignore_index=True)
        if ledgers
        else pd.DataFrame(columns=[*keys, "replicate_id", "value"])
    )
    replicas.to_parquet(folder / "effect_replicates.parquet", index=False)
    pd.DataFrame(coverage).to_csv(folder / "contrast_coverage.csv", index=False)
    write_json(
        folder / "COMPLETED.json",
        {
            "protocol_sha256": file_hash(config_path),
            "intervals": len(interval_rows),
            "replicate_rows": len(interval_rows) * 5000,
            "third_party_reproduction": False,
            "conditional_on": "Fixed models and observed windows; no retraining uncertainty",
            "family_difference": "LR effect minus five-seed mean LightGBM effect",
            "estimable_contrast_stage_cells": sum(x["estimable"] for x in coverage),
            "unavailable_contrast_stage_cells": sum(not x["estimable"] for x in coverage),
            "no_formal_inference_for_failed_fits": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", choices=["paysim", "ieee_cis"], required=True)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args.protocol, args.dataset)
    else:
        calculate_dataset(args.protocol, args.dataset)


if __name__ == "__main__":
    main()
