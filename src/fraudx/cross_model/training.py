"""Train the complete fixed logistic grid without Test-guided model selection."""

from __future__ import annotations

import argparse
import gc
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from fraudx.data import validation_period_masks
from fraudx.ieee_cis_experiment import ExperimentPaths, load_frozen_inputs
from fraudx.models import PlattCalibrator
from fraudx.paysim_deterministic.training import threshold_grid
from fraudx.q2_extension.common import array_hash, file_hash, read_json, require, write_json


@dataclass
class FrozenPreprocessor:
    """Train-only median/indicators/scaling and missing-aware one-hot categories."""

    columns: list[str]
    categorical: list[str]

    def numeric_values(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.numeric].to_numpy(dtype=np.float64, copy=True)
        values[~np.isfinite(values)] = np.nan
        return values

    def category_values(self, frame: pd.DataFrame) -> np.ndarray:
        # Prefix real values so that a source string cannot collide with the missing token.
        return np.column_stack(
            [
                ("value:" + frame[c].astype("string")).fillna("missing:").to_numpy(dtype=str)
                for c in self.categorical
            ]
        )

    def fit(self, frame: pd.DataFrame) -> FrozenPreprocessor:
        require(
            not {"isFraud", "source_row_id", "row_id", "TransactionID"}.intersection(self.columns),
            "Identity or label in features",
        )
        self.numeric = [c for c in self.columns if c not in self.categorical]
        self.imputer = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )
        self.scaler = StandardScaler(copy=False)
        values = self.imputer.fit_transform(self.numeric_values(frame))
        self.scaler.fit(values)
        if self.categorical:
            categories = self.category_values(frame)
            vocabulary = [
                np.unique(np.append(categories[:, i], "missing:"))
                for i in range(len(self.categorical))
            ]
            self.encoder = OneHotEncoder(
                categories=vocabulary, handle_unknown="ignore", sparse_output=True, dtype=np.float64
            )
            self.encoder.fit(categories)
        return self

    def transform(self, frame: pd.DataFrame) -> Any:
        # Chunking limits temporary dense memory and does not alter fitted statistics.
        chunks = []
        for start in range(0, len(frame), 50000):
            part = frame.iloc[start : start + 50000]
            numeric = sparse.csr_matrix(
                self.scaler.transform(self.imputer.transform(self.numeric_values(part)))
            )
            if self.categorical:
                numeric = sparse.hstack(
                    [numeric, self.encoder.transform(self.category_values(part))],
                    format="csr",
                    dtype=np.float64,
                )
            chunks.append(numeric)
        matrix = sparse.vstack(chunks, format="csr", dtype=np.float64)
        require(bool(np.isfinite(matrix.data).all()), "Nonfinite transformed features")
        return matrix

    def audit(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "numeric": self.numeric,
            "categorical": self.categorical,
            "medians": self.imputer.statistics_.tolist(),
            "missing_indicator_indices": self.imputer.indicator_.features_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "categories": [x.tolist() for x in self.encoder.categories_]
            if self.categorical
            else [],
            "fit_scope": "Train only",
        }


def load_dataset(config: dict[str, Any], dataset: str) -> pd.DataFrame:
    spec = config["datasets"][dataset]
    if dataset == "paysim":
        frame = pd.read_parquet(spec["features"])
        require(
            frame[["step", "source_row_id"]].equals(
                frame[["step", "source_row_id"]].sort_values(
                    ["step", "source_row_id"], kind="stable"
                )
            ),
            "PaySim order drift",
        )
        return frame
    paths = ExperimentPaths(**{k: Path(v) for k, v in spec["input_paths"].items()})
    transaction, history, _, _ = load_frozen_inputs(paths)
    extra = [c for c in history.columns if c.startswith(("card_", "combo_"))]
    history = history.set_index("TransactionID").loc[transaction.TransactionID, extra]
    for column in extra:
        transaction[column] = history[column].to_numpy()
    return transaction


def code_signature() -> dict[str, str]:
    import fraudx.data
    import fraudx.ieee_cis_experiment
    import fraudx.models
    import fraudx.paysim_deterministic.training

    return {
        Path(str(m.__file__)).name: file_hash(Path(str(m.__file__)))
        for m in [fraudx.data, fraudx.models, fraudx.ieee_cis_experiment]
    }


def train(config_path: Path, dataset: str) -> None:
    config = read_json(config_path)
    spec = config["datasets"][dataset]
    output = Path(config["output_root"]) / "training" / dataset
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "protocol_sha256": file_hash(config_path),
        "training_code_sha256": file_hash(Path(__file__)),
        "threshold_code_sha256": file_hash(Path(threshold_grid.__code__.co_filename)),
        "dependencies": code_signature(),
    }
    signature_path = output / "INPUT_SIGNATURE.json"
    if signature_path.exists():
        require(read_json(signature_path) == signature, "Resume signature changed")
    else:
        require(not list(output.iterdir()), "Nonempty unsigned training directory")
        write_json(signature_path, signature)
    for item in config["input_manifest"]:
        if dataset in item["datasets"]:
            require(
                file_hash(Path(item["path"])) == item["sha256"],
                f"Frozen input changed: {item['path']}",
            )
    frame = load_dataset(config, dataset)
    require(not frame[spec["id"]].duplicated().any(), "Duplicate source IDs")
    completed = []
    for boundary in spec["folds"]:
        fold = boundary["fold"]
        split = {
            name: frame[frame[spec["time"]].between(*boundary[name])]
            for name in ("train", "validation", "test")
        }
        for model_id, definition in spec["models"].items():
            prefix = f"{dataset}_{fold}_{model_id}"
            destination = output / prefix
            final = destination / "COMPLETED.json"
            if final.exists():
                meta = read_json(final)
                require(meta["signature"] == signature, "Checkpoint provenance mismatch")
                for record in meta["files"]:
                    require(
                        file_hash(destination / record["name"]) == record["sha256"],
                        "Completed model artifact changed",
                    )
                completed.append(meta)
                print(f"Verified {prefix}", flush=True)
                continue
            require(not destination.exists(), f"Preserve incomplete attempt: {destination}")
            destination.mkdir()
            write_json(
                destination / "STARTED.json",
                {
                    "signature": signature,
                    "started_utc": datetime.now(UTC).isoformat(),
                    "dataset": dataset,
                    "fold": fold,
                    "model": model_id,
                },
            )
            started = time.monotonic()
            print(f"Starting {prefix}: Train={len(split['train'])}", flush=True)
            columns = definition["columns"]
            preprocessing = FrozenPreprocessor(
                columns, [c for c in spec["categorical"] if c in columns]
            )
            with threadpool_limits(limits=1):
                preprocessing.fit(split["train"])
                x_train = preprocessing.transform(split["train"])
                y_train = split["train"].isFraud.to_numpy(dtype=np.int8)
                attempts = []
                for limit in (
                    config["cross_model"]["parameters"]["max_iter"],
                    config["cross_model"]["convergence_retry"]["max_iter"],
                ):
                    params = dict(config["cross_model"]["parameters"], max_iter=limit)
                    classifier = LogisticRegression(
                        **params, class_weight={0: 1, 1: definition["weight"]}
                    )
                    attempt_start = time.monotonic()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        classifier.fit(x_train, y_train)
                    convergence_warning = any(
                        issubclass(w.category, ConvergenceWarning) for w in caught
                    )
                    converged = not convergence_warning and int(classifier.n_iter_[0]) < limit
                    record = {
                        "max_iter": limit,
                        "n_iter": int(classifier.n_iter_[0]),
                        "converged": converged,
                        "seconds": time.monotonic() - attempt_start,
                        "warnings": [str(w.message) for w in caught],
                    }
                    attempts.append(record)
                    joblib.dump(classifier, destination / f"attempt_{limit}.joblib")
                    write_json(destination / "ATTEMPTS.json", attempts)
                    print(f"Fit {prefix}: {record}", flush=True)
                    if converged:
                        break
                write_json(destination / "preprocessing.json", preprocessing.audit())
                joblib.dump(preprocessing, destination / "preprocessing.joblib")
                np.savez(
                    destination / "coefficients.npz",
                    coefficient=classifier.coef_,
                    intercept=classifier.intercept_,
                    classes=classifier.classes_,
                )
                del x_train
                gc.collect()
                calibrator, scenarios = None, []
                if converged:
                    predicted = {}
                    for name in ("validation", "test"):
                        part = split[name]
                        matrix = preprocessing.transform(part)
                        vector = part[[spec["id"], spec["time"], "isFraud"]].reset_index(drop=True)
                        vector["margin"] = classifier.decision_function(matrix)
                        vector["raw_probability"] = classifier.predict_proba(matrix)[:, 1]
                        predicted[name] = vector
                        del matrix
                    if dataset == "paysim":
                        val = predicted["validation"]
                        calmask, thmask = validation_period_masks(val, "step", "isFraud")
                        require(
                            int(val.loc[calmask, "step"].max())
                            == spec["calibration_max_steps"][fold - 1],
                            "Calibration drift",
                        )
                        cal = PlattCalibrator().fit(
                            val.raw_probability.to_numpy()[calmask], val.isFraud.to_numpy()[calmask]
                        )
                        calibrator = {"slope": cal.slope, "intercept": cal.intercept}
                        val["calibration_part"] = calmask
                        for vector in predicted.values():
                            vector["clipped"] = np.clip(vector.raw_probability, 1e-6, 1 - 1e-6)
                            vector["platt"] = cal.transform(vector.raw_probability.to_numpy())
                        # All schemes retain the same descriptive capacity/cost grid.
                        selected = threshold_grid(
                            val.isFraud.to_numpy()[thmask],
                            val.platt.to_numpy()[thmask],
                            config["capacities"],
                            config["costs"],
                        )
                        scenarios = [
                            {"capacity": q, "fn_cost": c, **item}
                            for (q, c), item in selected.items()
                        ]
                    for name, vector in predicted.items():
                        vector.to_parquet(destination / f"{name}.parquet", index=False)
                    write_json(
                        destination / "score_hashes.json",
                        {
                            name: {c: array_hash(vector[c].to_numpy()) for c in vector.columns}
                            for name, vector in predicted.items()
                        },
                    )
                    del predicted
            meta = {
                "signature": signature,
                "dataset": dataset,
                "fold": fold,
                "model": model_id,
                "converged": converged,
                "attempts": attempts,
                "calibrator": calibrator,
                "scenarios": scenarios,
                "seconds": time.monotonic() - started,
                "split_counts": {k: len(v) for k, v in split.items()},
                "completed_utc": datetime.now(UTC).isoformat(),
                "files": [
                    {"name": p.name, "sha256": file_hash(p)}
                    for p in sorted(destination.iterdir())
                    if p.is_file()
                ],
            }
            write_json(final, meta)
            completed.append(meta)
            del classifier, preprocessing
            gc.collect()
    require(len(completed) == len(spec["models"]) * 3, "Missing formal conditions")
    write_json(
        output / "GRID_COMPLETED.json",
        {
            "formal_conditions": len(completed),
            "fit_attempts": sum(len(x["attempts"]) for x in completed),
            "converged": sum(x["converged"] for x in completed),
            "platt_fits": sum(x["calibrator"] is not None for x in completed),
            "signature": signature,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", choices=["paysim", "ieee_cis"], required=True)
    args = parser.parse_args()
    train(args.protocol, args.dataset)


if __name__ == "__main__":
    main()
