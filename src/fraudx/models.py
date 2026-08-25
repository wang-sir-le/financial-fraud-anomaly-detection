"""Static LightGBM models and probability calibration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pandas.api.types import is_numeric_dtype
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LEAKAGE_COLUMNS = {
    "isFraud",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
}


@dataclass
class PlattCalibrator:
    """Monotone one-dimensional logistic calibration for validation scores."""

    slope: float | None = None
    intercept: float | None = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped))
        targets = np.asarray(labels, dtype=float)

        def objective(parameters: np.ndarray) -> float:
            calibrated = np.clip(
                expit(parameters[0] * logits + parameters[1]),
                1e-12,
                1 - 1e-12,
            )
            return float(
                -np.mean(
                    targets * np.log(calibrated)
                    + (1 - targets) * np.log(1 - calibrated)
                )
            )

        result = minimize(
            objective,
            x0=np.asarray([1.0, 0.0]),
            method="L-BFGS-B",
            bounds=[(1e-6, None), (None, None)],
        )
        if not result.success:
            raise RuntimeError(f"Probability calibration failed: {result.message}")
        self.slope = float(result.x[0])
        self.intercept = float(result.x[1])
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.slope is None or self.intercept is None:
            raise RuntimeError("Calibrator must be fitted before transform")
        clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped))
        return np.asarray(expit(self.slope * logits + self.intercept), dtype=float)


def build_lightgbm_pipeline(
    frame: pd.DataFrame,
    seed: int,
    scale_pos_weight: float = 1.0,
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    feature_columns: Sequence[str] | None = None,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
) -> tuple[Pipeline, list[str]]:
    """Create preprocessing and model without account identifiers or label leakage."""
    if feature_columns is None:
        selected_columns = [
            column for column in frame.columns if column not in LEAKAGE_COLUMNS
        ]
    else:
        selected_columns = list(feature_columns)
    missing = [column for column in selected_columns if column not in frame.columns]
    unsafe = sorted(set(selected_columns).intersection(LEAKAGE_COLUMNS))
    if missing:
        raise ValueError(f"Requested model features are missing: {missing}")
    if unsafe:
        raise ValueError(f"Requested model features contain leakage columns: {unsafe}")
    if len(selected_columns) != len(set(selected_columns)):
        raise ValueError("Requested model features must be unique")
    if not selected_columns:
        raise ValueError("At least one model feature is required")
    categorical = [
        column for column in selected_columns if not is_numeric_dtype(frame[column])
    ]
    numerical = [column for column in selected_columns if column not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numerical", "passthrough", numerical),
        ],
        remainder="drop",
    )
    classifier = LGBMClassifier(
        objective="binary",
        random_state=seed,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        scale_pos_weight=scale_pos_weight,
        subsample=subsample,
        subsample_freq=1 if subsample < 1.0 else 0,
        colsample_bytree=colsample_bytree,
        deterministic=True,
        force_col_wise=True,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)]), selected_columns
