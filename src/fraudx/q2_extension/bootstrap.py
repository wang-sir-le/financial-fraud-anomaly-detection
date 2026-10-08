"""Paired component intervals conditional on fixed models and observed windows."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fraudx.q2_extension.common import (
    FOLDS,
    SEEDS,
    Context,
    array_hash,
    file_hash,
    load_frame,
    prediction_path,
    read_json,
    require,
    scores_for,
    write_json,
)
from fraudx.q2_extension.kernels import ModelView, model_view, weighted_ap, weighted_tp


def calculate_model(view: ModelView, positions: np.ndarray, labels: np.ndarray,
                    weights: np.ndarray) -> pd.DataFrame:
    group_n = np.bincount(positions, minlength=weights.shape[1]).astype(np.int64)
    group_f = np.bincount(positions, weights=labels,
                          minlength=weights.shape[1]).astype(np.int64)
    n = weights.astype(np.int64) @ group_n if weights.size < 2000000 else weights @ group_n
    f = weights @ group_f
    alerts = np.ceil(.03 * n).astype(np.int64)
    tp: np.ndarray = np.empty(len(weights), dtype=np.int64)
    ap: np.ndarray = np.empty(len(weights), dtype=np.float64)
    for start in range(0, len(weights), 4):
        stop = min(start + 4, len(weights))
        batch = weights[start:stop]
        ap[start:stop] = weighted_ap(view, batch)
        tp[start:stop] = weighted_tp(view, labels, positions, batch, alerts[start:stop])
    return pd.DataFrame({"replicate_id": np.arange(1, len(weights) + 1), "n": n,
                         "frauds": f, "alerts": alerts, "tp": tp, "ap": ap})


def threshold_metrics(ctx: Context, fold: int, model: str, seed: int,
                      frame: pd.DataFrame, positions: np.ndarray,
                      weights: np.ndarray) -> pd.DataFrame:
    path = prediction_path(ctx, "paysim", fold, model, seed)
    meta = read_json(path.with_name(path.name.replace("_test.parquet", ".json")))
    rows = [x for x in meta["rows"] if x["capacity"] == .03 and x["fn_cost"] == 100]
    require(len(rows) == 1, "Frozen validation threshold missing")
    labels = frame.isFraud.to_numpy(np.int64)
    selected = (frame.platt.to_numpy() >= rows[0]["threshold"]).astype(np.int64)
    group_f = np.bincount(positions, weights=labels, minlength=weights.shape[1]).astype(np.int64)
    group_tp = np.bincount(positions, weights=labels * selected,
                           minlength=weights.shape[1]).astype(np.int64)
    group_fp = np.bincount(positions, weights=(1 - labels) * selected,
                           minlength=weights.shape[1]).astype(np.int64)
    tp, fp, frauds = weights @ group_tp, weights @ group_fp, weights @ group_f
    return pd.DataFrame({"threshold_tp": tp, "threshold_fp": fp,
                         "threshold_cost": fp + 100 * (frauds - tp)})


def assert_numeric_equal(left: np.ndarray, right: np.ndarray, message: str,
                         integer: bool = False) -> float:
    require(left.shape == right.shape, f"Shape mismatch: {message}")
    same = np.array_equal(left, right) if integer else np.allclose(
        left, right, rtol=1e-10, atol=1e-12, equal_nan=True)
    require(bool(same), f"Numeric mismatch: {message}")
    finite = np.isfinite(left) & np.isfinite(right)
    return float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0


def legacy_regression(ctx: Context, dataset: str, fold: int, seed: int,
                      cached: dict[tuple[str, str], pd.DataFrame]) -> list[dict[str, Any]]:
    if dataset == "paysim":
        base, candidate = cached[("P0", "platt")], cached[("PTHW", "platt")]
        ref = pd.read_parquet(ctx.resource / "evidence/paysim_overall/"
                              "timeblock_bootstrap_seed_effects.parquet")
        original = ref[(ref.fold == fold) & (ref.lightgbm_seed == seed)].sort_values("bootstrap_id")
        frauds = base.frauds.to_numpy()
        cost = (base.threshold_cost - candidate.threshold_cost).to_numpy()
        recall = np.divide(candidate.threshold_tp - base.threshold_tp, frauds,
                           out=np.full(len(base), np.nan), where=frauds > 0)
        values = {"delta_recall": recall, "cost_reduction": cost,
                  "cost_reduction_per_100k": cost * 100000 / base.n.to_numpy(),
                  "delta_cost_saving_rate": cost / (100 * frauds),
                  "delta_pr_auc": (candidate.ap - base.ap).to_numpy()}
    else:
        base, candidate = cached[("M0", "raw_probability")], cached[("M2", "raw_probability")]
        ref = pd.read_parquet(ctx.resource / "evidence/ieee_primary_replay/"
                              "bootstrap_seed_replicates.parquet")
        original = ref[(ref.fold == fold) & (ref.seed == seed)].sort_values("replicate_id")
        tp = (candidate.tp - base.tp).to_numpy()
        frauds = base.frauds.to_numpy()
        values = {"delta_tp": tp, "delta_recall": tp / frauds,
                  "cost_reduction": 101 * tp, "delta_pr_auc": (candidate.ap - base.ap).to_numpy()}
    output = []
    for metric, value in values.items():
        expected = original[metric].to_numpy()[:len(value)]
        maximum = assert_numeric_equal(np.asarray(value), expected,
                                       f"{dataset}/{fold}/{seed}/{metric}",
                                       metric in ("delta_tp", "cost_reduction"))
        output.append({"dataset": dataset, "fold": fold, "seed": seed, "metric": metric,
                       "draws": len(value), "max_abs_error": maximum, "status": "PASS"})
    return output


def benchmark(ctx: Context) -> None:
    folder = ctx.start("b2/benchmark")
    regression, durations = [], []
    for dataset in ctx.config["datasets"]:
        models = ["P0", "PTHW"] if dataset == "paysim" else ["M0", "M2"]
        stage = "platt" if dataset == "paysim" else "raw_probability"
        spec = ctx.spec(dataset)
        weights = np.load(ctx.output / f"b2/draws/{dataset}_1_multiplicities.npy", mmap_mode="r")
        cache = {}
        for model in models:
            frame = load_frame(ctx, dataset, 1, model, 42)
            ids, labels = frame[spec["id"]].to_numpy(), frame[spec["target"]].to_numpy(np.int8)
            _, positions = np.unique(frame[spec["time"]].to_numpy(), return_inverse=True)
            started = time.perf_counter()
            view = model_view(ids, labels, scores_for(frame, stage), positions, weights.shape[1])
            result = calculate_model(view, positions, labels, weights[:100])
            if dataset == "paysim":
                result = pd.concat([result, threshold_metrics(ctx, 1, model, 42, frame, positions,
                                                              weights[:100])], axis=1)
            cache[(model, stage)] = result
            durations.append({"dataset": dataset, "fold": 1, "model": model, "seed": 42,
                              "replicates": 100, "seconds": time.perf_counter() - started,
                              "n": len(ids), "score_groups": view.total.shape[1],
                              "paper_result": False})
        regression.extend(legacy_regression(ctx, dataset, 1, 42, cache))
    pd.DataFrame(regression).to_csv(folder / "legacy_regression.csv", index=False)
    pd.DataFrame(durations).to_csv(folder / "benchmark.csv", index=False)
    ctx.finish(folder, {"first_100_draws_match": True, "not_paper_results": True})


def calculate_cached_model(ctx: Context, dataset: str, fold: int, seed: int, model: str,
                           weights: np.ndarray, model_folder: Path) -> tuple[
                               dict[tuple[str, str], pd.DataFrame], list[dict[str, Any]]]:
    """One independent model; equivalent score partitions reuse exact metrics."""
    spec = ctx.spec(dataset)
    frame = load_frame(ctx, dataset, fold, model, seed)
    ids = frame[spec["id"]].to_numpy(np.int64)
    labels = frame[spec["target"]].to_numpy(np.int8)
    _, positions = np.unique(frame[spec["time"]].to_numpy(), return_inverse=True)
    last_view, last_result = None, None
    cache, timing = {}, []
    for stage in spec["stages"]:
        started = time.perf_counter()
        score = scores_for(frame, stage)
        view = model_view(ids, labels, score, positions, weights.shape[1])
        reused = last_view is not None and np.array_equal(last_view.partition, view.partition)
        if reused:
            if last_result is None:
                raise ValueError("Missing equivalent-stage result")
            result = last_result.copy()
        else:
            result = calculate_model(view, positions, labels, weights)
        last_view, last_result = view, result
        if dataset == "paysim" and stage == "platt":
            result = pd.concat([result, threshold_metrics(
                ctx, fold, model, seed, frame, positions, weights)], axis=1)
        cache[(model, stage)] = result
        name = f"{dataset}_{fold}_{model}_{seed}_{stage}.parquet"
        result.to_parquet(model_folder / name, index=False)
        timing.append({"dataset": dataset, "fold": fold, "seed": seed,
                       "model": model, "stage": stage, "same_partition": reused,
                       "score_hash": array_hash(score),
                       "seconds": time.perf_counter() - started})
    print(f"Bootstrap {dataset} fold={fold} seed={seed} model={model}", flush=True)
    return cache, timing


def run_bootstrap(ctx: Context) -> None:
    folder = ctx.start("b2/calculation")
    model_folder = folder / "models"
    model_folder.mkdir()
    seed_effects, regressions, timing = [], [], []
    for dataset in ctx.config["datasets"]:
        spec = ctx.spec(dataset)
        for fold in FOLDS:
            weights = np.load(ctx.output / f"b2/draws/{dataset}_{fold}_multiplicities.npy",
                              mmap_mode="r")
            for seed in SEEDS:
                cache: dict[tuple[str, str], pd.DataFrame] = {}
                with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(calculate_cached_model, ctx, dataset, fold, seed,
                                               model, weights, model_folder)
                               for model in spec["models"]]
                    for future in futures:
                        model_cache, model_timing = future.result()
                        cache.update(model_cache)
                        timing.extend(model_timing)
                regressions.extend(legacy_regression(ctx, dataset, fold, seed, cache))
                for contrast in ctx.contrasts(dataset, core=True):
                    for stage in spec["stages"]:
                        base, candidate = cache[(contrast["baseline"], stage)], cache[
                            (contrast["candidate"], stage)]
                        count_keys = ["replicate_id", "n", "frauds", "alerts"]
                        require(np.array_equal(base[count_keys], candidate[count_keys]),
                                "Unpaired bootstrap populations")
                        out = base[["replicate_id", "n", "frauds", "alerts"]].copy()
                        out["dataset"], out["fold"], out["seed"] = dataset, fold, seed
                        out["contrast"], out["score_stage"] = contrast["id"], stage
                        out["delta_tp"] = candidate.tp - base.tp
                        out["delta_ap"] = candidate.ap - base.ap
                        out["delta_recall"] = out.delta_tp / out.frauds.replace(0, np.nan)
                        out["cost_reduction"] = 101 * out.delta_tp
                        seed_effects.append(out)
                pd.DataFrame(regressions).to_csv(folder / "legacy_regression.csv", index=False)
                pd.DataFrame(timing).to_csv(folder / "timing.csv", index=False)
    effects = pd.concat(seed_effects, ignore_index=True)
    require(len(effects) == 33 * 5000 * 5, "Incomplete component grid")
    effects.to_parquet(folder / "component_seed_replicates.parquet", index=False)
    summary_from_ledgers(ctx, effects, folder)
    pd.DataFrame(timing).to_csv(folder / "timing.csv", index=False)
    write_json(folder / "kernel_provenance.json",
               {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")})
    ctx.finish(folder, {"seed_effect_rows": len(effects), "regression_rows": len(regressions),
                        "new_model_fits": 0, "intervals": "pointwise_conditional"})


def interval_row(values: np.ndarray, observed: float, key: dict[str, Any]) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    lo, hi = np.quantile(finite, [.025, .975], method="linear") if len(finite) else (np.nan, np.nan)
    return {**key, "estimate": observed, "ci_lower": lo, "ci_upper": hi,
            "draws": len(values), "valid": len(finite), "invalid": len(values) - len(finite),
            "ci_type": "pointwise_95_percentile_conditional_fixed_models_and_windows"}


def summary_from_ledgers(ctx: Context, effects: pd.DataFrame, folder: Path) -> None:
    keys = ["dataset", "fold", "contrast", "score_stage", "replicate_id"]
    require(not effects.duplicated(keys + ["seed"]).any(), "Duplicate seed effect keys")
    group = effects.groupby(keys, sort=True)
    require(bool(group["seed"].nunique().eq(5).all()), "Incomplete fixed-seed average")
    for col in ("n", "frauds", "alerts"):
        require(bool(group[col].nunique(dropna=False).eq(1).all()), "Unpaired count fields")
    folded = group[["delta_tp", "delta_ap"]].sum(min_count=5) / 5
    folded["frauds"] = group.frauds.first()
    folded["delta_recall"] = folded.delta_tp / folded.frauds.replace(0, np.nan)
    folded["cost_reduction"] = 101 * folded.delta_tp
    folded = folded.reset_index()
    folded.to_parquet(folder / "component_fold_replicates.parquet", index=False)
    points = pd.read_parquet(ctx.output / "b1/stage_contrasts_seed.parquet")
    capacities = pd.read_parquet(ctx.output / "b3/profiles/capacity_contrasts_seed.parquet")
    summary = []
    for key, part in folded.groupby(keys[:-1]):
        dataset, fold, contrast, stage = key
        p = points[(points.dataset == dataset) & (points.fold == fold)
                   & (points.contrast == contrast) & (points.score_stage == stage)]
        cap = capacities[(capacities.dataset == dataset) & (capacities.fold == fold)
                         & (capacities.contrast == contrast) & (capacities.score_stage == stage)
                         & (capacities.q == .03)]
        observed = {"delta_ap": p.delta_ap.mean(), "delta_tp": cap.delta_tp.mean(),
                    "delta_recall": cap.delta_recall.mean(),
                    "cost_reduction": cap.cost_reduction.mean()}
        for metric, value in observed.items():
            record_key = dict(zip(keys[:-1], key, strict=True)) | {"metric": metric}
            summary.append(interval_row(part[metric].to_numpy(), float(value), record_key))
    pd.DataFrame(summary).to_csv(folder / "component_intervals.csv", index=False)
    processing = []
    pay = folded[folded.dataset == "paysim"]
    for (fold, contrast), part in pay.groupby(["fold", "contrast"]):
        for metric in ("delta_ap", "delta_recall"):
            source_metric = "delta_tp" if metric == "delta_recall" else metric
            pivot = part.pivot(index="replicate_id", columns="score_stage", values=source_metric)
            for before, after in (("raw", "clipped"), ("clipped", "platt"), ("raw", "platt")):
                effects_array = (pivot[after] - pivot[before]).to_numpy()
                if metric == "delta_recall":
                    frauds = part.groupby("replicate_id").frauds.first().reindex(pivot.index)
                    effects_array = np.divide(effects_array, frauds.to_numpy(),
                        out=np.full(len(effects_array), np.nan), where=frauds.to_numpy() > 0)
                observed = {r["score_stage"]: r["estimate"] for r in summary
                            if r["dataset"] == "paysim" and r["fold"] == fold
                            and r["contrast"] == contrast and r["metric"] == metric}
                processing.append(interval_row(effects_array, observed[after] - observed[before],
                    {"dataset": "paysim", "fold": fold, "contrast": contrast, "metric": metric,
                     "processing": f"{after}_minus_{before}"}))
    pd.DataFrame(processing).to_csv(folder / "processing_effect_intervals.csv", index=False)
    invalid = folded[folded[["delta_ap", "delta_recall"]].isna().any(axis=1)]
    invalid.to_csv(folder / "invalid_replicates.csv", index=False)
