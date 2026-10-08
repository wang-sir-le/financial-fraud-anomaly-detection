"""Compare every scientific output of the two independently trained v2 grids."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraudx.paysim_deterministic.preprocess import compare_parquets, file_hash
from fraudx.q2_extension.common import FOLDS, SEEDS, read_json, require, write_json


def verify(reference: Path, reconstruction: Path, destination: Path) -> None:
    require(not destination.exists(), "Verification destination must be new")
    destination.mkdir(parents=True)
    records = []
    for fold in FOLDS:
        for model in ["P0", "PT", "PH", "PTH", "PTHW", "PTHW5"]:
            for seed in SEEDS:
                prefix = f"paysim_{fold}_{model}_{seed}"
                first, second = read_json(reference / (prefix + ".json")), read_json(reconstruction / (prefix + ".json"))
                for field in ["columns", "calibrator", "raw_array_sha256", "platt_array_sha256", "rows"]:
                    require(first[field] == second[field], f"Scientific field mismatch: {prefix}/{field}")
                for split in ["validation", "test"]:
                    name = prefix + f"_{split}.parquet"
                    # Ledger column types and every bit are compared, including calibration membership.
                    report = compare_parquets(reference / name, reconstruction / name)
                    records.append({"fold": fold, "model": model, "seed": seed, "split": split, **report})
                txt = prefix + "_booster.txt"
                require(file_hash(reference / txt) == file_hash(reconstruction / txt), f"Booster changed: {prefix}")
                if model in ["P0", "PTHW"]:
                    name = prefix + "_isotonic.json"
                    require(read_json(reference / name) == read_json(reconstruction / name), "Isotonic knots changed")
    a = pd.read_csv(reference / "all_seed_results.csv", float_precision="round_trip")
    b = pd.read_csv(reconstruction / "all_seed_results.csv", float_precision="round_trip")
    require(np.array_equal(a.to_numpy(), b.to_numpy()), "Scenario grid differs")
    pd.DataFrame(records).drop(columns="columns").to_csv(destination / "vector_comparison.csv", index=False)
    write_json(destination / "REPORT.json", {"status": "PASS_ALL_90_FITS_EXACT", "fits": 90,
        "platt_fits": 90, "isotonic_fits": 30, "vectors": len(records),
        "vector_rows": sum(r["rows"] for r in records), "scenario_rows": len(a),
        "numeric_tolerance": "bit-exact; no widened tolerance", "booster_files": "90 byte-exact",
        "scope": "local independent ordinary-wheel environment; not third-party reproduction"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.reference, args.reconstruction, args.output)
