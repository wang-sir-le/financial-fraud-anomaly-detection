"""Replay v2 interval summaries from distributed ledgers without raw transactions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraudx import timeblock_bootstrap as legacy
from fraudx.q2_extension.bootstrap import summary_from_ledgers
from fraudx.q2_extension.common import Context, require, write_json


def check(actual: pd.DataFrame, expected_path: Path, keys: list[str], fields: list[str]) -> dict:
    expected = pd.read_csv(expected_path, float_precision="round_trip")
    actual, expected = actual.sort_values(keys), expected.sort_values(keys)
    require(np.array_equal(actual[keys].astype(str), expected[keys].astype(str)), "Replay keys differ")
    a, b = actual[fields].to_numpy(float), expected[fields].to_numpy(float)
    require(bool(np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)), "Replay values differ")
    return {"file": expected_path.name, "rows": len(actual), "fields": fields,
            "max_absolute_error": float(np.nanmax(np.abs(a-b)))}


def run(evidence: Path, output: Path) -> None:
    require(not output.exists(), "Replay destination must be new")
    output.mkdir(parents=True)
    component = pd.read_parquet(evidence / "b2/calculation/component_seed_replicates.parquet")
    ctx = Context(evidence, {}, evidence, evidence, evidence)
    summary_from_ledgers(ctx, component, output)
    reports = []
    for filename, keys in [
        ("component_intervals.csv", ["dataset", "fold", "contrast", "score_stage", "metric"]),
        ("processing_effect_intervals.csv", ["dataset", "fold", "contrast", "processing", "metric"]),
    ]:
        reports.append(check(pd.read_csv(output / filename), evidence / "b2/calculation" / filename,
                             keys, ["estimate", "ci_lower", "ci_upper", "draws", "valid", "invalid"]))
    seeds = pd.read_parquet(evidence / "primary/all_block_seed_replicates.parquet")
    keys = ["fold", "block_length", "bootstrap_id"]
    grouped = seeds.groupby(keys)
    require(bool(grouped.lightgbm_seed.nunique().eq(5).all()), "Incomplete primary fixed-seed average")
    folds = (grouped[list(legacy.METRICS)].sum(min_count=5)/5).reset_index()
    expected = pd.read_csv(evidence / "primary/block_sensitivity_summary.csv", float_precision="round_trip")
    rows = []
    for (fold, length), group in folds.groupby(["fold", "block_length"]):
        for metric in legacy.METRICS:
            ref = expected[(expected.fold == fold) & (expected.block_length == length) & (expected.metric == metric)].iloc[0]
            rows.append(legacy._summary_row(scope=fold, metric=metric, observed_effect=ref.observed_effect,
                                            values=group[metric].to_numpy()) | {"fold": fold, "block_length": length})
    table = pd.DataFrame(rows)
    table.to_csv(output / "block_sensitivity_summary.csv", index=False)
    reports.append(check(table, evidence / "primary/block_sensitivity_summary.csv", ["fold", "block_length", "metric"],
                         ["ci_lower_95", "ci_upper_95", "valid_replicates", "invalid_replicates"]))
    primary = pd.concat([folds[(folds.fold == f) & (folds.block_length == length)]
                         for f, length in [(1, 4), (2, 3), (3, 7)]], ignore_index=True)
    observed = {f: {metric: float(expected[(expected.fold == f) & expected.primary_length & (expected.metric == metric)].iloc[0].observed_effect)
                    for metric in legacy.METRICS} for f in [1, 2, 3]}
    fold_summary = legacy._summarize_fold_effects(primary, observed)
    overall = legacy._overall_equal_weight_replicates(primary)
    overall_point = {m: float(np.mean([observed[f][m] for f in [1, 2, 3]])) for m in legacy.METRICS}
    overall_summary = legacy._summarize_overall_effects(overall, overall_point)
    for name, table, key in [("fold", fold_summary, ["fold", "metric"]), ("overall", overall_summary, ["metric"])]:
        filename = f"timeblock_bootstrap_{name}_summary.csv"
        table.to_csv(output / filename, index=False)
        reports.append(check(table, evidence / "primary" / filename, key, ["ci_lower_95", "ci_upper_95"]))
    write_json(output / "REPLAY_REPORT.json", {"status": "PASS", "checks": reports,
        "scope": "aggregation and interval endpoints from ledgers; no raw-data or resample-generation claim"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.evidence, args.output)
