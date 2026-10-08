"""Frozen-threshold primary intervals and the original block-length sensitivities."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fraudx import timeblock_bootstrap as legacy
from fraudx.paysim_deterministic.analysis import context
from fraudx.q2_extension.bootstrap import threshold_metrics
from fraudx.q2_extension.common import FOLDS, SEEDS, load_frame, prediction_path, read_json, require
from fraudx.q2_extension.kernels import model_view, weighted_ap


def effects(base: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    require(np.array_equal(base[["n", "frauds"]], candidate[["n", "frauds"]]), "Population mismatch")
    cost = (base.threshold_cost - candidate.threshold_cost).to_numpy()
    frauds, n = base.frauds.to_numpy(), base.n.to_numpy()
    return pd.DataFrame({"delta_recall": legacy._safe_divide((candidate.threshold_tp - base.threshold_tp).to_numpy(), frauds),
        "cost_reduction": cost, "cost_reduction_per_100k": cost * 100000 / n,
        "delta_cost_saving_rate": legacy._safe_divide(cost, 100 * frauds), "delta_pr_auc": (candidate.ap - base.ap).to_numpy()})


def run(config: Path) -> None:
    ctx = context(config)
    folder = ctx.start("primary")
    seed_rows, fold_rows, summary_rows, diagnostics = [], [], [], []
    observed: dict[int, dict[str, float]] = {}
    for fold in FOLDS:
        primary_length = ctx.spec("paysim")["bootstrap"]["block_lengths"][fold - 1]
        points, captures = [], []
        for seed in SEEDS:
            pair = {}
            captured_groups = {}
            for model in ["P0", "PTHW"]:
                path = prediction_path(ctx, "paysim", fold, model, seed)
                meta = read_json(path.with_name(path.name.replace("_test.parquet", ".json")))
                pair[model] = next(r for r in meta["rows"] if r["capacity"] == .03 and r["fn_cost"] == 100)
                scored = load_frame(ctx, "paysim", fold, model, seed)
                unique_times, group_positions = np.unique(scored.step, return_inverse=True)
                captured_groups[model] = np.bincount(group_positions,
                    weights=scored.isFraud.to_numpy() * (scored.platt.to_numpy() >= pair[model]["threshold"]))
            a, b = pair["P0"], pair["PTHW"]
            cost = a["cost"] - b["cost"]
            points.append({"delta_recall": b["recall"] - a["recall"], "cost_reduction": cost,
                "cost_reduction_per_100k": cost * 100000 / a["n"],
                "delta_cost_saving_rate": cost / (100 * a["frauds"]), "delta_pr_auc": b["ap"] - a["ap"]})
            captures.append(captured_groups["PTHW"] - captured_groups["P0"])
        sizes = np.bincount(group_positions)
        group_frauds = np.bincount(group_positions, weights=scored.isFraud.to_numpy())
        for series, values in {"fraud_rate": group_frauds / sizes,
                               "mean_delta_captured_fraud": np.mean(captures, axis=0)}.items():
            for lag in sorted({1, math.ceil(primary_length / 2), primary_length, 2 * primary_length}):
                left, right = values[:-lag], values[lag:]
                correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 0 and np.std(right) > 0 else np.nan
                diagnostics.append({"dataset": "PaySim", "fold": fold, "series": series,
                    "lag_groups": lag, "correlation": correlation, "paired_groups": len(left),
                    "time_groups": len(unique_times), "time_gap_median": float(np.median(np.diff(unique_times))),
                    "time_gap_max": float(np.max(np.diff(unique_times))), "group_size_median": float(np.median(sizes)),
                    "group_size_max": int(sizes.max())})
        observed[fold] = {metric: float(np.mean([p[metric] for p in points])) for metric in legacy.METRICS}
        for length in [math.ceil(primary_length / 2), primary_length, 2 * primary_length]:
            calculated = []
            frame = load_frame(ctx, "paysim", fold, "P0", 42)
            times, positions = np.unique(frame.step.to_numpy(), return_inverse=True)
            child = np.random.SeedSequence(20260820).spawn(3)[fold - 1]
            sampled = legacy.moving_block_positions(len(times), length, 5000, np.random.default_rng(child))
            weights = legacy.position_multiplicities(sampled, len(times))
            size = np.bincount(positions, minlength=len(times))
            frauds = np.bincount(positions, weights=frame.isFraud, minlength=len(times)).astype(np.int64)
            n, f = weights @ size, weights @ frauds
            for seed in SEEDS:
                cached = {}
                for model in ["P0", "PTHW"]:
                    if length == primary_length:
                        cached[model] = pd.read_parquet(ctx.output / "b2/calculation/models" /
                                                       f"paysim_{fold}_{model}_{seed}_platt.parquet")
                    else:
                        data = load_frame(ctx, "paysim", fold, model, seed)
                        _, position = np.unique(data.step.to_numpy(), return_inverse=True)
                        labels = data.isFraud.to_numpy(np.int8)
                        view = model_view(data.row_id.to_numpy(), labels, data.platt.to_numpy(), position, len(times))
                        ap = np.empty(5000)
                        with threadpool_limits(limits=1):
                            for start in range(0, 5000, 4):
                                ap[start:start + 4] = weighted_ap(view, weights[start:start + 4])
                        metrics = threshold_metrics(ctx, fold, model, seed, data, position, weights)
                        metrics["n"], metrics["frauds"], metrics["ap"] = n, f, ap
                        cached[model] = metrics
                    require(np.array_equal(cached[model].n, n) and np.array_equal(cached[model].frauds, f),
                            "Draw population changed")
                value = effects(cached["P0"], cached["PTHW"])
                calculated.append(value)
                seed_rows.append(value.assign(fold=fold, lightgbm_seed=seed, block_length=length,
                                              bootstrap_id=np.arange(1, 5001)))
                print(f"Primary intervals fold={fold} L={length} seed={seed}", flush=True)
            fold_value = pd.DataFrame({m: np.mean([v[m].to_numpy() for v in calculated], axis=0)
                                      for m in legacy.METRICS})
            fold_value["bootstrap_id"], fold_value["fold"] = np.arange(1, 5001), fold
            fold_value["block_length"] = length
            fold_value["bootstrap_transaction_count"], fold_value["bootstrap_fraud_count"] = n, f
            fold_value.to_parquet(folder / f"paysim_fold{fold}_L{length}_replicates.parquet", index=False)
            for metric in legacy.METRICS:
                summary_rows.append(legacy._summary_row(scope=fold, metric=metric,
                    observed_effect=observed[fold][metric], values=fold_value[metric].to_numpy()) |
                    {"fold": fold, "block_length": length, "primary_length": length == primary_length})
            if length == primary_length:
                fold_rows.append(fold_value)
    seed_frame = pd.concat(seed_rows, ignore_index=True)
    primary_folds = pd.concat(fold_rows, ignore_index=True)
    seed_frame.to_parquet(folder / "all_block_seed_replicates.parquet", index=False)
    primary_folds.to_parquet(folder / "timeblock_bootstrap_replicates.parquet", index=False)
    pd.DataFrame(summary_rows).to_csv(folder / "block_sensitivity_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(folder / "temporal_dependence_diagnostics.csv", index=False)
    legacy._summarize_fold_effects(primary_folds, observed).to_csv(folder / "timeblock_bootstrap_fold_summary.csv", index=False)
    overall = legacy._overall_equal_weight_replicates(primary_folds)
    overall.to_parquet(folder / "timeblock_bootstrap_overall_replicates.parquet", index=False)
    overall_point = {m: float(np.mean([observed[f][m] for f in FOLDS])) for m in legacy.METRICS}
    legacy._summarize_overall_effects(overall, overall_point).to_csv(folder / "timeblock_bootstrap_overall_summary.csv", index=False)
    ctx.finish(folder, {"block_cells": 9, "seed_effect_rows": len(seed_frame),
                        "primary_draws": 15000, "overall": "existing secondary equal-fold estimand only"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
