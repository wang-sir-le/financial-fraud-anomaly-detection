"""Reproduce the two distinct frozen random streams, retaining draw fingerprints."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from fraudx import ieee_cis_timeblock_bootstrap as ieee
from fraudx import timeblock_bootstrap as pay
from fraudx.q2_extension.common import FOLDS, Context, load_frame, require


def draw_batches(dataset: str, fold: int, groups: int,
                 repetitions: int = 5000) -> Iterator[np.ndarray]:
    """Keep the original seed conversion, dtype and draw-call boundaries."""
    if dataset == "paysim":
        child = np.random.SeedSequence(20260820).spawn(3)[fold - 1]
        positions = pay.moving_block_positions(groups, pay.block_length_for_steps(groups),
                                               repetitions, np.random.default_rng(child))
        yield positions
        return
    children = np.random.SeedSequence(20260822).spawn(3)
    seed = int(children[fold - 1].generate_state(1, dtype=np.uint64)[0])
    rng = np.random.default_rng(seed)
    for start in range(0, repetitions, 32):
        yield ieee.moving_block_positions(groups, ieee.block_length_for_groups(groups),
                                          min(32, repetitions - start), rng)


def sample_hash(positions: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(positions, dtype="<i4").tobytes()).hexdigest()[:16]


def run_draws(ctx: Context) -> None:
    folder = ctx.start("b2/draws")
    records = []
    legacy = pd.read_parquet(ctx.resource / "evidence/ieee_primary_replay/"
                             "bootstrap_seed_replicates.parquet")
    for dataset in ctx.config["datasets"]:
        spec = ctx.spec(dataset)
        for fold in FOLDS:
            frame = load_frame(ctx, dataset, fold, spec["models"][0], 42)
            times = np.unique(frame[spec["time"]].to_numpy())
            require(len(times) == spec["bootstrap"]["groups"][fold - 1], "Changed time groups")
            weights = open_memmap(folder / f"{dataset}_{fold}_multiplicities.npy", mode="w+",
                                  dtype=np.int16, shape=(5000, len(times)))
            offset = 0
            for batch in draw_batches(dataset, fold, len(times)):
                for row in batch:
                    counts = np.bincount(row, minlength=len(times))
                    require(int(counts.max()) <= np.iinfo(np.int16).max, "Weight overflow")
                    weights[offset] = counts
                    records.append({"dataset": dataset, "fold": fold,
                                    "replicate_id": offset + 1, "sample_hash": sample_hash(row),
                                    "groups": len(times), "max_group_multiplicity": counts.max()})
                    offset += 1
            weights.flush()
            require(offset == 5000, "Incomplete draws")
            if dataset == "ieee_cis":
                original = legacy[(legacy.fold == fold) & (legacy.seed == 42)].sort_values(
                    "replicate_id")
                require(np.array_equal(original.sample_hash.to_numpy(),
                                       [x["sample_hash"] for x in records
                                        if x["dataset"] == dataset and x["fold"] == fold]),
                        "IEEE draw fingerprints differ from frozen ledger")
            print(f"Draws {dataset} fold={fold} T={len(times)}", flush=True)
    pd.DataFrame(records).to_parquet(folder / "draw_index_hashes.parquet", index=False)
    ctx.finish(folder, {"fold_draws": len(records), "IEEE_hashes_match": True,
                        "PaySim_effect_regression_required": True})
