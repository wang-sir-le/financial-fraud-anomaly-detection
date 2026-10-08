"""Complete v1/v2 differences, retaining adverse values and unmatched inventories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudx.q2_extension.bootstrap import interval_row
from fraudx.q2_extension.common import file_hash, require, write_json


def read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, float_precision="round_trip")


def table_difference(old: pd.DataFrame, new: pd.DataFrame, keys: list[str],
                     name: str) -> pd.DataFrame:
    """Long-form every comparable numeric field; never omit unchanged/negative cells."""
    require(not old.duplicated(keys).any() and not new.duplicated(keys).any(), f"Duplicate keys: {name}")
    paired = old.merge(new, on=keys, how="outer", suffixes=("_v1", "_v2"), indicator=True, validate="one_to_one")
    require(bool(paired._merge.eq("both").all()), f"Unmatched evidence grid: {name}")
    columns = [c for c in old.select_dtypes(include=["number"]).columns
               if c not in keys and c in new.select_dtypes(include=["number"]).columns]
    rows = []
    for metric in columns:
        out = paired[keys].copy()
        out["analysis"], out["field"] = name, metric
        out["v1"], out["v2"] = paired[metric + "_v1"], paired[metric + "_v2"]
        out["v2_minus_v1"] = out.v2 - out.v1
        out["changed_exact"] = ~(out.v1.eq(out.v2) | (out.v1.isna() & out.v2.isna()))
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def run(root: Path, run_dir: Path, old_b: Path) -> None:
    output = run_dir / "assessment/version_comparison"
    require(not output.exists(), "Preserve existing comparison")
    output.mkdir(parents=True)
    sources, coverage = [], []

    def compare(old_path: Path, new_path: Path, keys: list[str], name: str,
                renames: dict[str, str] | None = None, paysim_only: bool = False) -> None:
        a, b = read(old_path), read(new_path)
        if paysim_only:
            a = a[a.dataset == "paysim"]
        if renames:
            a = a.rename(columns=renames)
        difference = table_difference(a, b, keys, name)
        difference.to_csv(output / f"{name}.csv", index=False)
        coverage.append({"analysis": name, "key_rows": len(b), "numeric_cells": len(difference),
                         "changed_cells": int(difference.changed_exact.sum())})
        sources.extend([{"path": str(p.resolve()), "sha256": file_hash(p)} for p in [old_path, new_path]])

    old = read(root / "experiment/outputs/kais_revision_20260919/paysim_all_seed_results.csv")
    w5 = read(root / "experiment/outputs/optimized_first_round/robustness/rolling_multiseed_results.csv")
    w5 = w5[w5.model_id == "recipient_full_weight_5"].rename(columns={"target_capacity": "capacity",
        "pr_auc": "ap", "business_cost": "cost", "test_size": "n", "test_fraud_count": "frauds",
        "alert_count": "alerts"}).assign(model="PTHW5")
    w5["cost_per_10000"] = w5.cost * 10000 / w5.n
    old = pd.concat([old, w5[old.columns]], ignore_index=True)
    new = read(run_dir / "models/reference/all_seed_results.csv")
    old.to_csv(output / "v1_all_660_scenarios.csv", index=False)
    difference = table_difference(old, new, ["fold", "model", "seed", "capacity", "fn_cost"], "all_scenarios")
    difference.to_csv(output / "all_scenarios.csv", index=False)
    coverage.append({"analysis": "all_scenarios", "key_rows": len(new), "numeric_cells": len(difference),
                     "changed_cells": int(difference.changed_exact.sum())})
    analysis = run_dir / "analysis"
    draw_checks = []
    for fold in [1, 2, 3]:
        old_draw = np.load(old_b / f"b2/draws/paysim_{fold}_multiplicities.npy")
        new_draw = np.load(analysis / f"b2/draws/paysim_{fold}_multiplicities.npy")
        require(np.array_equal(old_draw, new_draw), "Version time-block weights differ")
        old_scores = read(root / f"experiment/outputs/kais_revision_20260919/checkpoints/paysim_{fold}_P0_42_test.parquet")
        new_scores = read(run_dir / f"models/reference/paysim_{fold}_P0_42_test.parquet")
        old_groups = old_scores.groupby("step").isFraud.agg(["size", "sum"])
        new_groups = new_scores.groupby("step").isFraud.agg(["size", "sum"])
        require(np.array_equal(old_groups.index, new_groups.index)
                and np.array_equal(old_groups.to_numpy(), new_groups.to_numpy()),
                "Version time-group membership counts differ")
        draw_checks.append({"fold": fold, "same_weights": True, "same_group_counts_and_frauds": True})
    write_json(output / "VERSION_PAIRING_AUDIT.json", draw_checks)
    specs = [
        ("b1/stage_model_metrics.parquet", ["dataset", "fold", "model", "seed", "score_stage"], "score_stage_models"),
        ("b1/stage_contrasts_seed.parquet", ["dataset", "fold", "contrast", "seed", "score_stage"], "score_stage_contrasts"),
        ("b3/profiles/capacity_tie_model.csv", ["dataset", "fold", "model", "seed", "score_stage", "q"], "capacity_models"),
        ("b3/profiles/capacity_contrasts_fold.csv", ["dataset", "fold", "contrast", "score_stage", "q"], "capacity_contrasts"),
        ("b3/priorities/shared_priority_summary.csv", ["dataset", "fold", "contrast", "score_stage", "q"], "priority_summaries"),
        ("b2/calculation/component_intervals.csv", ["dataset", "fold", "contrast", "score_stage", "metric"], "component_intervals"),
        ("b2/calculation/processing_effect_intervals.csv", ["dataset", "fold", "contrast", "processing", "metric"], "processing_intervals"),
    ]
    for relative, keys, name in specs:
        # Existing metric-key columns remain join keys; differences are their numeric endpoints.
        compare(old_b / relative, analysis / relative, keys, name, paysim_only=True)
    primary = root / "experiment/outputs/optimized_first_round/bootstrap"
    compare(primary / "timeblock_bootstrap_fold_summary.csv", analysis / "primary/timeblock_bootstrap_fold_summary.csv",
            ["fold", "metric"], "primary_intervals")
    compare(primary / "timeblock_bootstrap_overall_summary.csv", analysis / "primary/timeblock_bootstrap_overall_summary.csv",
            ["metric"], "secondary_overall_intervals")
    for name, keys in [
        ("calibration_ablation_results", ["fold", "model_id", "seed", "calibration_method"]),
        ("calibration_bayes_decision_results", ["fold", "model_id", "seed", "calibration_method"]),
        ("calibration_decision_invariance", ["fold", "model_id", "seed", "comparison"]),
        ("calibration_pairwise_reliability", ["fold", "model_id", "seed", "comparison"]),
    ]:
        compare(root / f"experiment/outputs/optimized_first_round/calibration/{name}.csv",
                analysis / f"calibration/{name}.csv", keys, name)
    compare(root / "experiment/outputs/kais_targeted_revision_20260920/audit/paysim_block_sensitivity_summary.csv",
            analysis / "primary/block_sensitivity_summary.csv", ["fold", "block_length", "metric"], "block_sensitivity")
    # A paired comparison of implementation versions uses common time-group draws, not invented row matches.
    old_effects = read(old_b / "b2/calculation/component_seed_replicates.parquet")
    old_effects = old_effects[old_effects.dataset == "paysim"]
    new_effects = read(analysis / "b2/calculation/component_seed_replicates.parquet")
    keys = ["dataset", "fold", "contrast", "score_stage", "replicate_id", "seed"]
    old_effects, new_effects = old_effects.sort_values(keys), new_effects.sort_values(keys)
    require(np.array_equal(old_effects[keys], new_effects[keys]), "Version replicate keys differ")
    require(np.array_equal(old_effects[["n", "frauds", "alerts"]], new_effects[["n", "frauds", "alerts"]]),
            "Version draws have different populations")
    result = new_effects[keys].reset_index(drop=True)
    metrics = ["delta_ap", "delta_tp", "delta_recall", "cost_reduction"]
    for metric in metrics:
        result[metric] = new_effects[metric].to_numpy() - old_effects[metric].to_numpy()
    result = result.groupby(keys[:-1], as_index=False)[metrics].mean()
    result.to_parquet(output / "paired_version_component_fold_replicates.parquet", index=False)
    ci_old, ci_new = read(old_b / "b2/calculation/component_intervals.csv"), read(analysis / "b2/calculation/component_intervals.csv")
    ci_old = ci_old[ci_old.dataset == "paysim"]
    point_keys = ["dataset", "fold", "contrast", "score_stage", "metric"]
    point = ci_old.merge(ci_new, on=point_keys, suffixes=("_old", "_new"), validate="one_to_one")
    intervals: list[dict[str, Any]] = []
    for key, part in result.groupby(keys[:-2]):
        details = dict(zip(keys[:-2], key, strict=True))
        for metric in metrics:
            selected = point[(point.fold == details["fold"]) & (point.contrast == details["contrast"])
                             & (point.score_stage == details["score_stage"]) & (point.metric == metric)].iloc[0]
            intervals.append(interval_row(part[metric].to_numpy(), selected.estimate_new - selected.estimate_old,
                                          details | {"metric": metric}))
    pd.DataFrame(intervals).to_csv(output / "paired_version_component_intervals.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    write_json(output / "SOURCE_MANIFEST.json", sources)
    write_json(output / "REPORT.json", {"status": "ALL_SPECIFIED_COMPARISONS_RECORDED", "analyses": coverage,
        "version_pairing": "same time groups and draws; no old/new transaction identity matching",
        "interpretation": "joint implementation sensitivity; neither factor isolation nor new external evidence"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--old-b", type=Path, required=True)
    args = parser.parse_args()
    run(args.workspace, args.run, args.old_b)
