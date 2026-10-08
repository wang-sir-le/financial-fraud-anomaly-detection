"""Complete score-stage and capacity grids, with shared label-free tie priorities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudx.models import PlattCalibrator
from fraudx.q2_extension.common import (
    FOLDS,
    SEEDS,
    Context,
    array_hash,
    load_frame,
    prediction_path,
    read_json,
    require,
    scores_for,
)
from fraudx.q2_extension.kernels import (
    capacity_profile,
    priority_ranks,
    priority_tp,
    stage_metrics,
)


def contrast_table(ctx: Context, data: pd.DataFrame, capacity: bool = False,
                   trial: bool = False) -> pd.DataFrame:
    keys = ["dataset", "fold", "seed", "score_stage"]
    if capacity:
        keys += ["q"]
    if trial:
        keys += ["trial"]
    results = []
    for dataset in ctx.config["datasets"]:
        for spec in ctx.contrasts(dataset):
            part = data[data.dataset == dataset]
            base = part[part.model == spec["baseline"]]
            candidate = part[part.model == spec["candidate"]]
            paired = base.merge(candidate, on=keys, suffixes=("_base", "_candidate"),
                                how="outer", validate="one_to_one", indicator=True)
            require(bool(paired._merge.eq("both").all()), "Incomplete contrast grid")
            out = paired[keys].copy()
            out["contrast"] = spec["id"]
            out["baseline"], out["candidate"] = spec["baseline"], spec["candidate"]
            if capacity:
                require(bool((paired.frauds_base == paired.frauds_candidate).all()),
                        "Different fraud denominators")
                out["delta_tp"] = paired.tp_candidate - paired.tp_base
                out["delta_recall"] = out.delta_tp / paired.frauds_base
                out["cost_reduction"] = 101 * out.delta_tp
                if not trial:
                    require(bool((paired.alerts_base == paired.alerts_candidate).all()),
                            "Unequal review budgets")
                    out["envelope_lower"] = paired.tp_min_candidate - paired.tp_max_base
                    out["envelope_upper"] = paired.tp_max_candidate - paired.tp_min_base
                    out["expected_delta_tp"] = (
                        paired.tp_expectation_candidate - paired.tp_expectation_base)
                    out["envelope_is_sharp"] = False
            else:
                out["delta_ap"] = paired.ap_candidate - paired.ap_base
            results.append(out)
    return pd.concat(results, ignore_index=True)


def run_points(ctx: Context) -> None:
    require(read_json(ctx.output / "audit/inputs/STATUS.json")["status"] == "COMPLETED",
            "Input pairing must complete first")
    folder = ctx.start("b1")
    caps = ctx.start("b3/profiles")
    metrics, capacities, tie_checks = [], [], []
    for dataset in ctx.config["datasets"]:
        spec = ctx.spec(dataset)
        for fold in FOLDS:
            for model in spec["models"]:
                for seed in SEEDS:
                    frame = load_frame(ctx, dataset, fold, model, seed)
                    ids = frame[spec["id"]].to_numpy(np.int64)
                    labels = frame[spec["target"]].to_numpy(np.int8)
                    previous = None
                    for stage in spec["stages"]:
                        score = scores_for(frame, stage)
                        key = {"dataset": dataset, "fold": fold, "model": model,
                               "seed": seed, "score_stage": stage}
                        result = stage_metrics(score, labels)
                        metrics.append({**key, **result, "n": len(ids),
                                        "frauds": int(labels.sum()),
                                        "score_hash": array_hash(score)})
                        if previous is not None:
                            order = np.argsort(previous, kind="stable")
                            monotone = bool((np.diff(score[order]) >= 0).all())
                            require(monotone, "Score stage unexpectedly reverses order")
                            tie_checks.append({**key, "input_unique": len(np.unique(previous)),
                                               "output_unique": result["unique_scores"],
                                               "weakly_monotone": monotone,
                                               "merged_score_groups":
                                               len(np.unique(previous)) - result["unique_scores"]})
                        previous = score
                        for q in ctx.config["experiment_design"]["capacities"]:
                            profile, _ = capacity_profile(ids, labels, score, q)
                            capacities.append({**key, **profile})
                    if dataset == "paysim":
                        path = prediction_path(ctx, dataset, fold, model, seed)
                        meta_path = path.with_name(path.name.replace("_test.parquet", ".json"))
                        meta = read_json(meta_path)
                        cal = PlattCalibrator(**meta["calibrator"])
                        require(bool(np.allclose(cal.transform(frame.raw.to_numpy()),
                                                frame.platt.to_numpy(), rtol=1e-10, atol=1e-12)),
                                "Stored calibrated probabilities do not match parameters")
                print(f"Point grid {dataset} fold={fold} model={model}", flush=True)
    models = pd.DataFrame(metrics)
    capacity_models = pd.DataFrame(capacities)
    contrasts = contrast_table(ctx, models)
    capacity_contrasts = contrast_table(ctx, capacity_models, capacity=True)
    expected = ctx.config["expected_B_outputs"]
    require(len(models) == expected["stage_model_metric_rows"], "Wrong model-stage count")
    require(len(contrasts) == expected["stage_contrast_seed_rows"], "Wrong contrast count")
    require(len(capacity_models) == expected["capacity_model_rows"], "Wrong capacity count")
    models.to_parquet(folder / "stage_model_metrics.parquet", index=False)
    contrasts.to_parquet(folder / "stage_contrasts_seed.parquet", index=False)
    contrasts.groupby(["dataset", "fold", "contrast", "score_stage"], as_index=False)[
        "delta_ap"].mean().to_csv(folder / "stage_contrasts_fold.csv", index=False)
    pd.DataFrame(tie_checks).to_csv(folder / "score_tie_audit.csv", index=False)
    stage_pairs = contrasts[contrasts.dataset == "paysim"].pivot(
        index=["dataset", "fold", "seed", "contrast"], columns="score_stage", values="delta_ap")
    stage_pairs["clipped_minus_raw"] = stage_pairs.clipped - stage_pairs.raw
    stage_pairs["platt_minus_clipped"] = stage_pairs.platt - stage_pairs.clipped
    stage_pairs["platt_minus_raw"] = stage_pairs.platt - stage_pairs.raw
    require(bool(np.allclose(stage_pairs.platt_minus_raw,
                            stage_pairs.clipped_minus_raw + stage_pairs.platt_minus_clipped,
                            rtol=1e-10, atol=1e-12)), "Stage decomposition failed")
    stage_pairs.reset_index().to_csv(folder / "processing_effects_seed.csv", index=False)
    capacity_models.to_csv(caps / "capacity_tie_model.csv", index=False)
    capacity_contrasts.to_parquet(caps / "capacity_contrasts_seed.parquet", index=False)
    capacity_contrasts.groupby(["dataset", "fold", "contrast", "score_stage", "q"],
                              as_index=False)[["delta_tp", "delta_recall", "cost_reduction",
                                               "envelope_lower", "envelope_upper",
                                               "expected_delta_tp"]].mean().to_csv(
        caps / "capacity_contrasts_fold.csv", index=False)
    ctx.finish(folder, {"model_rows": len(models), "contrast_rows": len(contrasts),
                        "new_fits": 0})
    ctx.finish(caps, {"model_capacity_rows": len(capacity_models), "new_fits": 0})


def run_priorities(ctx: Context) -> None:
    folder = ctx.start("b3/priorities")
    profiles = pd.read_csv(ctx.output / "b3/profiles/capacity_tie_model.csv",
                           float_precision="round_trip")
    outputs = []
    for dataset in ctx.config["datasets"]:
        spec = ctx.spec(dataset)
        for fold in FOLDS:
            part = profiles[(profiles.dataset == dataset) & (profiles.fold == fold)]
            boundaries: dict[tuple[str, int, str, float], tuple[dict[str, Any], np.ndarray]] = {}
            first = load_frame(ctx, dataset, fold, spec["models"][0], SEEDS[0])
            ids = first[spec["id"]].to_numpy(np.int64)
            labels = first[spec["target"]].to_numpy(np.int8)
            for model in spec["models"]:
                for seed in SEEDS:
                    frame = load_frame(ctx, dataset, fold, model, seed)
                    for stage in spec["stages"]:
                        scores = scores_for(frame, stage)
                        rows = part[(part.model == model) & (part.seed == seed)
                                    & (part.score_stage == stage)]
                        for row in rows.to_dict("records"):
                            members = np.flatnonzero(scores == row["boundary_score"])
                            require(len(members) == row["tie_n"], "Boundary round-trip mismatch")
                            boundaries[(model, seed, stage, row["q"])] = (row, members)
            split_members = [members for row, members in boundaries.values()
                             if row["remaining"] < row["tie_n"]]
            union = np.unique(np.concatenate(split_members)) if split_members else np.array([], int)
            lookup: np.ndarray = np.full(len(ids), -1, dtype=np.int64)
            lookup[union] = np.arange(len(union))
            output_rows = []
            for trial in range(100):
                ranks = priority_ranks(ids[union], dataset, fold, trial)
                for (model, seed, stage, q), (row, members) in boundaries.items():
                    if row["remaining"] == row["tie_n"]:
                        tp = int(row["above_tp"] + row["tie_frauds"])
                    else:
                        tp = priority_tp(row, lookup[members], labels[union], ranks)
                    require(row["tp_min"] <= tp <= row["tp_max"], "Priority outside bounds")
                    output_rows.append({"dataset": dataset, "fold": fold, "model": model,
                                        "seed": seed, "score_stage": stage, "q": q,
                                        "trial": trial, "tp": tp, "frauds": int(labels.sum())})
                if trial % 20 == 0:
                    print(f"Priorities {dataset} fold={fold} trial={trial} union={len(union)}",
                          flush=True)
            result = pd.DataFrame(output_rows)
            result.to_parquet(folder / f"{dataset}_{fold}_model_trials.parquet", index=False)
            outputs.append(result)
    models = pd.concat(outputs, ignore_index=True)
    contrasts = contrast_table(ctx, models, capacity=True, trial=True)
    contrasts.to_parquet(folder / "shared_priority_trials.parquet", index=False)
    folds = contrasts.groupby(["dataset", "fold", "contrast", "score_stage", "q", "trial"],
                              as_index=False)[["delta_tp", "delta_recall"]].mean()
    summary = folds.groupby(["dataset", "fold", "contrast", "score_stage", "q"])[
        "delta_tp"].agg(["mean", "min", "max", "count"]).reset_index()
    summary["interpretation"] = "100_shared_label_free_priorities_not_CI_or_sharp_range"
    summary.to_csv(folder / "shared_priority_summary.csv", index=False)
    require(len(models) == 114000, "Incomplete priority trial grid")
    ctx.finish(folder, {"model_trial_rows": len(models), "contrast_trial_rows": len(contrasts),
                        "trials": 100, "is_confidence_interval": False})
