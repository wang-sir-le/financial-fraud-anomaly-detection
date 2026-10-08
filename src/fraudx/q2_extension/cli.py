"""Explicit extension steps; no implicit training, downloads or publication."""

from __future__ import annotations

import argparse
from pathlib import Path

from fraudx.q2_extension.bootstrap import benchmark, run_bootstrap, summary_from_ledgers
from fraudx.q2_extension.common import Context, inventory, read_json, require, write_json
from fraudx.q2_extension.point_analysis import run_points, run_priorities
from fraudx.q2_extension.resampling import run_draws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["inventory", "points", "priorities", "draws",
                                         "benchmark", "bootstrap", "replay"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    ctx = Context.load(args.config)
    functions = {"inventory": inventory, "points": run_points, "priorities": run_priorities,
                 "draws": run_draws, "benchmark": benchmark, "bootstrap": run_bootstrap}
    if args.step == "replay":
        import numpy as np
        import pandas as pd

        folder = ctx.start("b2/replay")
        original = ctx.output / "b2/calculation"
        require(read_json(original / "STATUS.json")["status"] == "COMPLETED",
                "Full calculation required")
        effects = pd.read_parquet(original / "component_seed_replicates.parquet")
        summary_from_ledgers(ctx, effects, folder)
        checks = []
        for name in ["component_intervals.csv", "processing_effect_intervals.csv"]:
            a = pd.read_csv(original / name, float_precision="round_trip")
            b = pd.read_csv(folder / name, float_precision="round_trip")
            require(a.shape == b.shape, "Interval replay shape mismatch")
            for col in a.columns:
                if pd.api.types.is_numeric_dtype(a[col]):
                    require(bool(np.allclose(a[col], b[col], rtol=1e-10, atol=1e-12,
                                             equal_nan=True)), f"Interval replay mismatch: {col}")
                else:
                    require(a[col].equals(b[col]), f"Interval keys differ: {col}")
            checks.append({"file": name, "rows": len(a), "endpoints": 2 * len(a)})
        write_json(folder / "REPLAY_REPORT.json", {"status": "PASS", "checks": checks,
                                                   "scope": "new interval ledger replay"})
        ctx.finish(folder, {"status": "PASS", "checks": checks})
    else:
        functions[args.step](ctx)


if __name__ == "__main__":
    main()
