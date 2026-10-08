"""Replay every released C confidence interval without private source transactions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudx.q2_extension.bootstrap import interval_row
from fraudx.q2_extension.common import file_hash, require, write_json

KEYS = ["dataset", "fold", "contrast", "role", "score_stage", "decision_rule", "family", "metric"]


def replay(
    evidence: Path, output: Path, datasets: tuple[str, ...] = ("paysim", "ieee_cis")
) -> None:
    require(not output.exists(), "Use a new replay directory")
    output.mkdir(parents=True)
    report: list[dict[str, Any]] = []
    for dataset in datasets:
        folder = evidence / dataset
        ledger = pd.read_parquet(folder / "effect_replicates.parquet")
        points = pd.read_csv(folder / "observed_effects.csv", float_precision="round_trip")
        expected = pd.read_csv(folder / "effect_intervals.csv", float_precision="round_trip")
        if not len(ledger):
            require(not len(points) and not len(expected), "Empty ledger has claimed estimates")
            coverage = pd.read_csv(folder / "contrast_coverage.csv")
            require(not coverage.estimable.any(), "Estimable contrasts have no ledger")
            report.append(
                {
                    "dataset": dataset,
                    "rows": 0,
                    "maximum_absolute_error": 0.0,
                    "status": "No eligible contrasts; absence retained, no interval claim",
                }
            )
            expected.to_csv(output / f"{dataset}_replayed_intervals.csv", index=False)
            continue
        require(not ledger.duplicated(KEYS + ["replicate_id"]).any(), "Duplicate draw identity")
        require(not points.duplicated(KEYS).any(), "Duplicate observed effect")
        rows = []
        indexed = points.set_index(KEYS)
        for key, group in ledger.groupby(KEYS, sort=True, dropna=False):
            group = group.sort_values("replicate_id")
            require(np.array_equal(group.replicate_id, np.arange(1, 5001)), "Draw gap")
            rows.append(
                interval_row(
                    group.value.to_numpy(),
                    float(indexed.loc[key, "estimate"]),
                    dict(zip(KEYS, key, strict=True)),
                )
            )
        actual = pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)
        expected = expected.sort_values(KEYS).reset_index(drop=True)
        require(actual[KEYS].equals(expected[KEYS]), "Interval key mismatch")
        for c in ["draws", "valid", "invalid"]:
            require(np.array_equal(actual[c], expected[c]), f"Count mismatch: {c}")
        columns = ["estimate", "ci_lower", "ci_upper"]
        a, b = actual[columns].to_numpy(), expected[columns].to_numpy()
        require(np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True), "Endpoint mismatch")
        actual.to_csv(output / f"{dataset}_replayed_intervals.csv", index=False)
        report.append(
            {
                "dataset": dataset,
                "rows": len(actual),
                "maximum_absolute_error": float(np.nanmax(np.abs(a - b))),
                "ledger_sha256": file_hash(folder / "effect_replicates.parquet"),
            }
        )
    write_json(
        output / "REPLAY_REPORT.json",
        {
            "status": "PASS",
            "datasets": report,
            "total_interval_rows": sum(x["rows"] for x in report),
            "rtol": 1e-10,
            "atol": 1e-12,
            "raw_transactions_required": False,
            "model_refits": 0,
            "not_independent_external_reproduction": True,
        },
    )
    print(f"Replayed {sum(x['rows'] for x in report)} C interval rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=["paysim", "ieee_cis"])
    args = parser.parse_args()
    replay(args.evidence, args.output, (args.dataset,) if args.dataset else ("paysim", "ieee_cis"))


if __name__ == "__main__":
    main()
