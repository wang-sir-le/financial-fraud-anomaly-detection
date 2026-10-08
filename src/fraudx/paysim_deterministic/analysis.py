"""Version-local PaySim reanalysis using the already validated B numerical kernels."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fraudx.q2_extension.bootstrap import calculate_cached_model, summary_from_ledgers
from fraudx.q2_extension.common import (
    FOLDS,
    SEEDS,
    Context,
    assert_aligned,
    file_hash,
    load_frame,
    read_json,
    require,
    scores_for,
    write_json,
)
from fraudx.q2_extension.kernels import priority_ranks, priority_tp
from fraudx.q2_extension.point_analysis import contrast_table, run_points
from fraudx.q2_extension.resampling import draw_batches, sample_hash


def context(config_path: Path) -> Context:
    """Explicit version-local context; preserve the old extension's path contract."""
    config = read_json(config_path)
    root, output = Path(config["workspace_root"]).resolve(), Path(config["output_root"]).resolve()
    scores = (root / config["datasets"]["paysim"]["predictions"]).resolve()
    require(not scores.is_relative_to(output), "Analysis cannot contain its source scores")
    require(list(config["datasets"]) == ["paysim"], "This adapter handles only changed PaySim")
    return Context(config_path, config, root, output, root)


def audit(ctx: Context) -> None:
    folder = ctx.start("audit/inputs")
    rows = []
    for fold in FOLDS:
        for split in ["validation", "test"]:
            reference = None
            for model in ctx.spec("paysim")["models"]:
                for seed in SEEDS:
                    frame = load_frame(ctx, "paysim", fold, model, seed, split)
                    keys = ["source_row_id", "row_id", "step", "isFraud"]
                    require(np.array_equal(frame.source_row_id, frame.row_id), "Wrong tie identity")
                    if reference is None:
                        reference = frame[keys].copy()
                    else:
                        assert_aligned(reference, frame, keys)
                    rows.append({"fold": fold, "split": split, "model": model,
                                 "seed": seed, "n": len(frame), "frauds": int(frame.isFraud.sum())})
    pd.DataFrame(rows).to_csv(folder / "alignment.csv", index=False)
    ctx.finish(folder, {"vectors": len(rows), "same_source_identity": True})


def draws(ctx: Context) -> None:
    folder = ctx.start("b2/draws")
    rows = []
    for fold in FOLDS:
        frame = load_frame(ctx, "paysim", fold, "P0", 42)
        times = np.unique(frame.step.to_numpy())
        require(len(times) == ctx.spec("paysim")["bootstrap"]["groups"][fold - 1], "Group drift")
        positions = next(draw_batches("paysim", fold, len(times)))
        weights = np.stack([np.bincount(p, minlength=len(times)) for p in positions]).astype(np.int16)
        np.save(folder / f"paysim_{fold}_multiplicities.npy", weights)
        for index, sample in enumerate(positions):
            rows.append({"dataset": "paysim", "fold": fold, "replicate_id": index + 1,
                         "sample_hash": sample_hash(sample), "groups": len(times)})
        pd.DataFrame({"step": times, "n": frame.groupby("step").size().to_numpy(),
                      "frauds": frame.groupby("step").isFraud.sum().to_numpy()}).to_csv(
            folder / f"paysim_{fold}_population.csv", index=False)
    pd.DataFrame(rows).to_parquet(folder / "draw_index_hashes.parquet", index=False)
    ctx.finish(folder, {"fold_draws": len(rows), "rng": "unchanged_original_paysim_stream"})


def bootstrap(ctx: Context) -> None:
    folder = ctx.start("b2/calculation")
    model_folder = folder / "models"
    model_folder.mkdir()
    effects, timings = [], []
    for fold in FOLDS:
        weights = np.load(ctx.output / f"b2/draws/paysim_{fold}_multiplicities.npy", mmap_mode="r")
        for seed in SEEDS:
            cache = {}
            with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(calculate_cached_model, ctx, "paysim", fold,
                    seed, model, weights, model_folder) for model in ctx.spec("paysim")["models"]]
                for future in futures:
                    calculated, timing = future.result()
                    cache.update(calculated)
                    timings.extend(timing)
            for contrast in ctx.contrasts("paysim", core=True):
                for stage in ctx.spec("paysim")["stages"]:
                    base = cache[(contrast["baseline"], stage)]
                    candidate = cache[(contrast["candidate"], stage)]
                    keys = ["replicate_id", "n", "frauds", "alerts"]
                    require(np.array_equal(base[keys], candidate[keys]), "Unpaired populations")
                    out = base[keys].copy()
                    out["dataset"], out["fold"], out["seed"] = "paysim", fold, seed
                    out["contrast"], out["score_stage"] = contrast["id"], stage
                    out["delta_tp"], out["delta_ap"] = candidate.tp - base.tp, candidate.ap - base.ap
                    out["delta_recall"] = out.delta_tp / out.frauds.replace(0, np.nan)
                    out["cost_reduction"] = 101 * out.delta_tp
                    effects.append(out)
            pd.DataFrame(timings).to_csv(folder / "timing.csv", index=False)
    result = pd.concat(effects, ignore_index=True)
    require(len(result) == 27 * 5000 * 5, "Incomplete component effects")
    result.to_parquet(folder / "component_seed_replicates.parquet", index=False)
    summary_from_ledgers(ctx, result, folder)
    ctx.finish(folder, {"seed_effect_rows": len(result), "model_fits": 0,
                        "intervals": "pointwise_conditional", "IEEE": "unchanged; separate baseline"})


def priorities(ctx: Context) -> None:
    """Same shared hash priorities as B; count contract covers only changed PaySim."""
    folder = ctx.start("b3/priorities")
    profiles = pd.read_csv(ctx.output / "b3/profiles/capacity_tie_model.csv", float_precision="round_trip")
    outputs = []
    for fold in FOLDS:
        part = profiles[profiles.fold == fold]
        boundaries: dict[tuple[str, int, str, float], tuple[dict[str, Any], np.ndarray]] = {}
        first = load_frame(ctx, "paysim", fold, "P0", 42)
        ids, labels = first.row_id.to_numpy(np.int64), first.isFraud.to_numpy(np.int8)
        for model in ctx.spec("paysim")["models"]:
            for seed in SEEDS:
                frame = load_frame(ctx, "paysim", fold, model, seed)
                for stage in ctx.spec("paysim")["stages"]:
                    scores = scores_for(frame, stage)
                    selected = part[(part.model == model) & (part.seed == seed) & (part.score_stage == stage)]
                    for row in selected.to_dict("records"):
                        members = np.flatnonzero(scores == row["boundary_score"])
                        require(len(members) == row["tie_n"], "Boundary round-trip mismatch")
                        boundaries[(model, seed, stage, row["q"])] = row, members
        split = [members for row, members in boundaries.values() if row["remaining"] < row["tie_n"]]
        union = np.unique(np.concatenate(split)) if split else np.array([], dtype=np.int64)
        lookup = np.full(len(ids), -1, dtype=np.int64)
        lookup[union] = np.arange(len(union))
        rows = []
        for trial in range(100):
            ranks = priority_ranks(ids[union], "paysim", fold, trial)
            for (model, seed, stage, q), (row, members) in boundaries.items():
                tp = int(row["above_tp"] + row["tie_frauds"]) if row["remaining"] == row["tie_n"] else (
                    priority_tp(row, lookup[members], labels[union], ranks))
                require(row["tp_min"] <= tp <= row["tp_max"], "Priority outside bounds")
                rows.append({"dataset": "paysim", "fold": fold, "model": model, "seed": seed,
                    "score_stage": stage, "q": q, "trial": trial, "tp": tp, "frauds": int(labels.sum())})
            if trial % 20 == 0:
                print(f"Priority fold={fold} trial={trial}", flush=True)
        output = pd.DataFrame(rows)
        output.to_parquet(folder / f"paysim_{fold}_model_trials.parquet", index=False)
        outputs.append(output)
    models = pd.concat(outputs, ignore_index=True)
    contrasts = contrast_table(ctx, models, capacity=True, trial=True)
    contrasts.to_parquet(folder / "shared_priority_trials.parquet", index=False)
    folds = contrasts.groupby(["dataset", "fold", "contrast", "score_stage", "q", "trial"],
                              as_index=False)[["delta_tp", "delta_recall"]].mean()
    summary = folds.groupby(["dataset", "fold", "contrast", "score_stage", "q"])["delta_tp"].agg(
        ["mean", "min", "max", "count"]).reset_index()
    summary["interpretation"] = "100_shared_label_free_priorities_not_CI_or_sharp_range"
    summary.to_csv(folder / "shared_priority_summary.csv", index=False)
    require(len(models) == 90000 and len(contrasts) == 90000, "Incomplete priority grid")
    ctx.finish(folder, {"model_trial_rows": len(models), "contrast_trial_rows": len(contrasts),
                        "trials": 100, "is_confidence_interval": False})


def run(config: Path, stage: str) -> None:
    ctx = context(config)
    operations = {"audit": audit, "draws": draws, "points": run_points,
                  "bootstrap": bootstrap, "priorities": priorities}
    provenance = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    write_json(ctx.output / f"{stage}_source_provenance.json", provenance)
    operations[stage](ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("stage", choices=["audit", "draws", "points", "bootstrap", "priorities"])
    args = parser.parse_args()
    run(args.config, args.stage)
