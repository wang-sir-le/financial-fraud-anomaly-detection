"""Configure full C reconstruction using reader-supplied, verified baseline inputs."""

from __future__ import annotations

import argparse
import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from fraudx.cross_model.analysis import load_lgbm
from fraudx.q2_extension.common import (
    SEEDS,
    array_hash,
    file_hash,
    read_json,
    require,
    write_json,
)
from fraudx.q2_extension.resampling import draw_batches


def semantic_reference_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset, spec in config["datasets"].items():
        columns = [spec["id"], spec["time"], "isFraud"] + (
            ["raw", "platt"] if dataset == "paysim" else ["raw_probability"]
        )
        for fold in (1, 2, 3):
            for model in spec["models"]:
                for seed in SEEDS:
                    frame, meta = load_lgbm(spec, dataset, fold, model, seed)
                    rows.append(
                        {
                            "dataset": dataset,
                            "fold": fold,
                            "model": model,
                            "seed": seed,
                            "rows": len(frame),
                            "columns": {c: array_hash(frame[c].to_numpy()) for c in columns},
                            "threshold_rows": meta.get("rows", []),
                        }
                    )
    return rows


def configure(resource: Path, output: Path, supplied: dict[str, Path]) -> None:
    """Only paths/provenance adapt; feature lists, parameters, draws and estimands stay fixed."""
    require(not output.exists(), "C rebuild output must be a new directory")
    original = read_json(resource / "protocol/protocol_effective.json")
    config = copy.deepcopy(original)
    config["output_root"] = str(output.resolve())
    config["created_utc"] = datetime.now(UTC).isoformat()
    config["reconstruction_of_protocol_sha256"] = file_hash(
        resource / "protocol/protocol_effective.json"
    )
    pay = config["datasets"]["paysim"]
    ieee = config["datasets"]["ieee_cis"]
    pay["features"] = str(supplied["paysim_features"].resolve())
    pay["predictions"] = str(supplied["paysim_lgbm_predictions"].resolve())
    ieee["predictions"] = str(supplied["ieee_lgbm_predictions"].resolve())
    for key in ("transaction_path", "history_path"):
        ieee["input_paths"][key] = str(supplied[f"ieee_{key}"].resolve())
    for key, name in [
        ("history_metadata_path", "ieee_history_metadata.json"),
        ("protocol_path", "ieee_temporal_protocol.json"),
    ]:
        ieee["input_paths"][key] = str((resource / "protocol" / name).resolve())
    ieee["input_paths"]["output_dir"] = str(output / "unused_ieee_legacy_output")
    # Metadata file serialization can differ after a reconstruction; compare all consumed
    # scientific score arrays and threshold rows, not runtime durations or container bytes.
    actual = semantic_reference_manifest(config)
    expected = read_json(resource / "audit/REFERENCE_VECTOR_MANIFEST.json")
    require(actual == expected, "Reconstructed reference scientific values differ")
    manifest = []
    for dataset, spec in config["datasets"].items():
        spec["lgbm_replicates"] = str((resource / "reference_replicates").resolve())
        spec["draws"] = str((output / "draws").resolve())
        for fold in (1, 2, 3):
            count = spec["bootstrap"]["groups"][fold - 1]
            path = output / "draws" / f"{dataset}_{fold}_multiplicities.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            weights = open_memmap(path, mode="w+", dtype=np.int16, shape=(5000, count))
            offset = 0
            for batch in draw_batches(dataset, fold, count):
                for row in batch:
                    values = np.bincount(row, minlength=count)
                    require(values.max() <= np.iinfo(np.int16).max, "Draw weight overflow")
                    weights[offset] = values
                    offset += 1
            weights.flush()
            del weights
            require(offset == 5000, "Missing regenerated draws")
            recorded = next(
                x for x in original["input_manifest"] if Path(x["path"]).name == path.name
            )
            require(file_hash(path) == recorded["sha256"], "Draw stream differs from protocol")
        raw = (
            [Path(spec["features"])]
            if dataset == "paysim"
            else [Path(v) for k, v in spec["input_paths"].items() if k != "output_dir"]
        )
        for path in raw:
            role = "features.parquet" if dataset == "paysim" else path.name
            matching = [
                x
                for x in original["input_manifest"]
                if dataset in x["datasets"]
                and (
                    Path(x["path"]).name == role
                    or (
                        role == "ieee_history_metadata.json"
                        and Path(x["path"]).name == "history_feature_metadata.json"
                    )
                    or (
                        role == "ieee_temporal_protocol.json"
                        and Path(x["path"]).name == "ieee_cis_temporal_protocol.json"
                    )
                )
            ]
            require(
                len(matching) == 1 and file_hash(path) == matching[0]["sha256"],
                f"Rebuild raw/feature input mismatch: {path}",
            )
        dependencies = (
            raw
            + list(Path(spec["draws"]).glob(f"{dataset}_*.npy"))
            + list(Path(spec["lgbm_replicates"]).glob(f"{dataset}_*.parquet"))
        )
        if dataset == "paysim":
            for fold in (1, 2, 3):
                for model in spec["models"]:
                    for seed in SEEDS:
                        base = Path(spec["predictions"]) / f"paysim_{fold}_{model}_{seed}"
                        dependencies += [
                            base.with_suffix(".json"),
                            base.with_name(base.name + "_test.parquet"),
                        ]
        else:
            dependencies.append(Path(spec["predictions"]))
        for path in dependencies:
            manifest.append(
                {
                    "path": str(path.resolve()),
                    "sha256": file_hash(path),
                    "bytes": path.stat().st_size,
                    "datasets": [dataset],
                }
            )
    config["input_manifest"] = manifest
    write_json(output / "protocol/protocol_effective.json", config)
    write_json(
        output / "protocol/PORTABLE_CONFIGURATION_CHECK.json",
        {
            "status": "PASS",
            "reference_vectors_exact": len(actual),
            "draw_files_rebuilt_and_hash_matched": 6,
            "raw_and_features_hash_matched": True,
            "protocol_sha256": file_hash(output / "protocol/protocol_effective.json"),
            "scientific_settings_changed": False,
            "scientific_fits": 0,
        },
    )
    print(f"Portable C protocol ready: {output / 'protocol/protocol_effective.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for field in [
        "paysim-features",
        "paysim-lgbm-predictions",
        "ieee-transaction-path",
        "ieee-history-path",
        "ieee-lgbm-predictions",
    ]:
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    supplied = {k: v for k, v in vars(args).items() if k not in {"resource", "output"}}
    configure(args.resource, args.output, supplied)


if __name__ == "__main__":
    main()
