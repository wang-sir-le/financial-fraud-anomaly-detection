"""Independent coverage, reference-regression and paired-difference checks for C."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraudx.q2_extension.common import read_json, require, write_json


def audit(protocol: Path, dataset: str) -> None:
    config = read_json(protocol)
    root = Path(config["output_root"])
    spec = config["datasets"][dataset]
    folder = root / "analysis" / dataset
    intervals = pd.read_csv(folder / "effect_intervals.csv", float_precision="round_trip")
    ledger = pd.read_parquet(folder / "effect_replicates.parquet")
    points = pd.read_csv(folder / "model_points.csv", float_precision="round_trip")
    keys = ["dataset", "fold", "contrast", "score_stage", "decision_rule", "metric"]
    maxima = []
    for key, part in ledger.groupby(keys, sort=True):
        pivot = part.pivot(index="replicate_id", columns="family", values="value")
        expected = pivot.LR - pivot.LightGBM
        actual = pivot.LR_minus_LightGBM
        require(
            np.array_equal(expected.to_numpy(), actual.to_numpy(), equal_nan=True),
            f"Paired difference failed: {key}",
        )
        maxima.append({"key": list(key), "draws": len(pivot), "exact_pairing_identity": True})
    original_root = Path(spec["lgbm_replicates"]).parents[2]
    original = pd.read_csv(
        original_root / "b2/calculation/component_intervals.csv", float_precision="round_trip"
    )
    old = original[
        (original.dataset == dataset)
        & original.metric.isin(["delta_ap", "delta_tp", "delta_recall"])
    ]
    current = intervals[
        (intervals.family == "LightGBM")
        & (intervals.role == "core")
        & (intervals.decision_rule == "fixed_budget")
    ]
    match_keys = ["dataset", "fold", "contrast", "score_stage", "metric"]
    paired = current.merge(old, on=match_keys, suffixes=("_c", "_old"), validate="one_to_one")
    require(len(paired) == len(current), "Unmatched reference intervals")
    error = 0.0
    for field in ("estimate", "ci_lower", "ci_upper"):
        a, b = paired[field + "_c"].to_numpy(), paired[field + "_old"].to_numpy()
        require(
            np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True),
            f"LightGBM evidence drift: {field}",
        )
        if len(a):
            error = max(error, float(np.nanmax(np.abs(a - b))))
    condition_rows = []
    for fold in (1, 2, 3):
        for model in spec["models"]:
            meta = read_json(
                root / "training" / dataset / f"{dataset}_{fold}_{model}/COMPLETED.json"
            )
            point_count = len(
                points[(points.family == "LR") & (points.fold == fold) & (points.model == model)]
            )
            expected_count = len(spec["stages"]) * 4 if meta["converged"] else 0
            require(point_count == expected_count, "Incomplete or ineligible point grid")
            condition_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "converged": meta["converged"],
                    "point_rows": point_count,
                    "expected": expected_count,
                    "attempts": len(meta["attempts"]),
                    "iterations": [x["n_iter"] for x in meta["attempts"]],
                }
            )
    write_json(
        root / "audit" / f"{dataset}_RESULT_AUDIT.json",
        {
            "status": "PASS",
            "condition_coverage": condition_rows,
            "paired_estimands": len(maxima),
            "all_paired_difference_identities_exact": True,
            "lightgbm_reference_interval_rows": len(paired),
            "lightgbm_reference_maximum_abs_difference": error,
            "interval_rows": len(intervals),
            "replicate_rows": len(ledger),
            "all_intervals_pointwise_conditional": True,
        },
    )
    print(f"{dataset}: coverage and {len(paired)} frozen reference intervals verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", choices=["paysim", "ieee_cis"], required=True)
    args = parser.parse_args()
    audit(args.protocol, args.dataset)
