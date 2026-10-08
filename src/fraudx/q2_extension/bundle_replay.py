"""Replay extension intervals from a portable, hash-verified seed ledger bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraudx.q2_extension.bootstrap import summary_from_ledgers
from fraudx.q2_extension.common import Context, file_hash, read_json, require, write_json


def replay(bundle: Path, output: Path) -> None:
    bundle, output = bundle.resolve(), output.resolve()
    require(not output.exists(), "Choose a new output directory; previous evidence is preserved")
    require(not output.is_relative_to(bundle), "Write replay results outside the input bundle")
    manifest = read_json(bundle / "MANIFEST.json")
    for entry in manifest["files"]:
        path = (bundle / entry["path"]).resolve()
        require(path.is_relative_to(bundle), "Manifest path escapes bundle")
        require(file_hash(path) == entry["sha256"], f"Input hash mismatch: {entry['path']}")
    config = read_json(bundle / "protocol_effective.json")
    ctx = Context(bundle / "protocol_effective.json", config, bundle,
                  bundle / "evidence", bundle)
    original = bundle / "evidence/b2/calculation"
    effects = pd.read_parquet(original / "component_seed_replicates.parquet")
    require(len(effects) == 825000, "Incomplete seed ledger")
    output.mkdir(parents=True)
    summary_from_ledgers(ctx, effects, output)
    checks = []
    for name, expected_rows in [("component_intervals.csv", 132),
                                ("processing_effect_intervals.csv", 54)]:
        reference = pd.read_csv(original / name, float_precision="round_trip")
        actual = pd.read_csv(output / name, float_precision="round_trip")
        require(len(actual) == expected_rows and reference.shape == actual.shape,
                f"Interval coverage differs: {name}")
        for column in reference.columns:
            if pd.api.types.is_numeric_dtype(reference[column]):
                require(bool(np.allclose(reference[column], actual[column], rtol=1e-10,
                                         atol=1e-12, equal_nan=True)),
                        f"Numeric replay mismatch: {name}/{column}")
            else:
                require(reference[column].equals(actual[column]),
                        f"Key replay mismatch: {name}/{column}")
        checks.append({"file": name, "rows": len(actual), "endpoints": 2 * len(actual)})
    write_json(output / "REPLAY_REPORT.json", {
        "status": "PASS", "checks": checks, "ledger_rows": len(effects),
        "verified_bundle_files": len(manifest["files"]), "module_path": __file__,
        "scope": "L1 local isolated ledger replay; no raw-data or external-reader validation",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.bundle, args.output)


if __name__ == "__main__":
    main()
