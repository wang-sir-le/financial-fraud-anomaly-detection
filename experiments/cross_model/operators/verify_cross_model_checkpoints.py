"""Recompute saved C scores in an installed environment without repeating detector fits."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import fraudx.cross_model.training as training
from fraudx.models import PlattCalibrator
from fraudx.q2_extension.common import array_hash, file_hash, read_json, require, write_json

# The frozen `python -m` trainer serialized its dataclass under __main__.
# Re-export the identical installed class for those local checkpoint files.
FrozenPreprocessor = training.FrozenPreprocessor


def verify(protocol: Path, dataset: str, output: Path) -> None:
    require(not output.exists(), "Preserve earlier verification")
    config = read_json(protocol)
    spec = config["datasets"][dataset]
    frame = training.load_dataset(config, dataset)
    root = Path(config["output_root"]) / "training" / dataset
    rows, skipped = [], []
    for boundary in spec["folds"]:
        fold = boundary["fold"]
        for model in spec["models"]:
            folder = root / f"{dataset}_{fold}_{model}"
            metadata = read_json(folder / "COMPLETED.json")
            if not metadata["converged"]:
                require(
                    not (folder / "test.parquet").exists(),
                    "Failed fit incorrectly has formal predictions",
                )
                skipped.append({"fold": fold, "model": model, "reason": "nonconverged"})
                continue
            # Load only our local training artifact; no untrusted remote pickle is accepted.
            processor = joblib.load(folder / "preprocessing.joblib")
            require(
                processor.audit() == read_json(folder / "preprocessing.json"),
                "Preprocessor serialization mismatch",
            )
            require(processor.columns == spec["models"][model]["columns"], "Feature order changed")
            with np.load(folder / "coefficients.npz", allow_pickle=False) as fitted:
                classifier = LogisticRegression(**config["cross_model"]["parameters"])
                classifier.coef_ = fitted["coefficient"]
                classifier.intercept_ = fitted["intercept"]
                classifier.classes_ = fitted["classes"]
                classifier.n_features_in_ = classifier.coef_.shape[1]
            for split in ("validation", "test"):
                sample = frame[frame[spec["time"]].between(*boundary[split])]
                reference = pd.read_parquet(folder / f"{split}.parquet")
                columns = [spec["id"], spec["time"], "isFraud"]
                require(np.array_equal(sample[columns], reference[columns]), "Score input mismatch")
                with threadpool_limits(limits=1):
                    matrix = processor.transform(sample)
                    rebuilt = {
                        "margin": classifier.decision_function(matrix),
                        "raw_probability": classifier.predict_proba(matrix)[:, 1],
                    }
                    del matrix
                    if dataset == "paysim":
                        rebuilt["clipped"] = np.clip(rebuilt["raw_probability"], 1e-6, 1 - 1e-6)
                        calibrator = PlattCalibrator(**metadata["calibrator"])
                        rebuilt["platt"] = calibrator.transform(rebuilt["raw_probability"])
                for stage, values in rebuilt.items():
                    original = reference[stage].to_numpy()
                    require(np.array_equal(values, original), "Checkpoint-to-score mismatch")
                    rows.append(
                        {
                            "dataset": dataset,
                            "fold": fold,
                            "model": model,
                            "split": split,
                            "stage": stage,
                            "rows": len(sample),
                            "array_sha256": array_hash(values),
                            "bit_exact": True,
                        }
                    )
                del sample, reference, rebuilt
                gc.collect()
            print(f"Verified installed checkpoint predictions {dataset}/{fold}/{model}", flush=True)
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output / "score_checks.csv", index=False)
    write_json(
        output / "REPORT.json",
        {
            "status": "PASS",
            "score_columns": len(rows),
            "value_rows": sum(x["rows"] for x in rows),
            "all_bit_exact": True,
            "skipped_nonconverged": skipped,
            "detector_refits": 0,
            "installed_training_module": training.__file__,
            "training_module_sha256": file_hash(Path(training.__file__)),
            "protocol_sha256": file_hash(protocol),
            "third_party_reproduction": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", choices=["paysim", "ieee_cis"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.protocol, args.dataset, args.output)
