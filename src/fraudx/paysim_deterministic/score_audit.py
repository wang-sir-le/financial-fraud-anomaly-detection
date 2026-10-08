"""All saved-score processing checks, without fitting or changing decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraudx.models import PlattCalibrator
from fraudx.paysim_deterministic.analysis import context
from fraudx.q2_extension.common import FOLDS, SEEDS, load_frame, prediction_path, read_json, require


def run(config: Path) -> None:
    ctx = context(config)
    folder = ctx.start("score_audit")
    rows = []
    for fold in FOLDS:
        for model in ctx.spec("paysim")["models"]:
            for seed in SEEDS:
                path = prediction_path(ctx, "paysim", fold, model, seed)
                meta = read_json(path.with_name(path.name.replace("_test.parquet", ".json")))
                calibrator = PlattCalibrator(**meta["calibrator"])
                for split in ["validation", "test"]:
                    frame = load_frame(ctx, "paysim", fold, model, seed, split)
                    raw, platt = frame.raw.to_numpy(), frame.platt.to_numpy()
                    clipped = np.clip(raw, 1e-6, 1-1e-6)
                    rebuilt = calibrator.transform(raw)
                    require(np.array_equal(platt, rebuilt), "Saved calibrator does not reconstruct exact scores")
                    unique_clip, clip_partition = np.unique(clipped, return_inverse=True)
                    unique_platt, platt_partition = np.unique(platt, return_inverse=True)
                    rows.append({"fold": fold, "model": model, "seed": seed, "split": split,
                        "raw_ap": average_precision_score(frame.isFraud, raw),
                        "clipped_ap": average_precision_score(frame.isFraud, clipped),
                        "platt_ap": average_precision_score(frame.isFraud, platt),
                        "raw_unique": len(np.unique(raw)), "clipped_unique": len(unique_clip),
                        "platt_unique": len(unique_platt), "lower_clip_count": int((raw<=1e-6).sum()),
                        "upper_clip_count": int((raw>=1-1e-6).sum()),
                        "clipped_platt_same_ordered_partition": bool(np.array_equal(clip_partition, platt_partition)),
                        "coefficient_replay_exact": True})
            print(f"Score-stage audit fold={fold} model={model}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(folder / "all_score_processing.csv", index=False)
    ctx.finish(folder, {"vectors": len(table), "test_vectors": int((table.split == "test").sum()),
                        "all_coefficients_exact": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
