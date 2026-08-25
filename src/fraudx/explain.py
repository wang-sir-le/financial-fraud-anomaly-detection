"""TreeSHAP feature attribution and failure-case exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraudx.config import ExperimentConfig
from fraudx.data import temporal_split
from fraudx.features import build_behavioral_features, feature_columns_for_group
from fraudx.models import build_lightgbm_pipeline
from fraudx.pipeline import prepare_real_data
from fraudx.preprocess import load_prepared_frame
from fraudx.synthetic import make_synthetic_paysim


def run_shap_analysis(config: ExperimentConfig, smoke: bool = False) -> pd.DataFrame:
    """Fit a declared model and export exact LightGBM TreeSHAP contributions."""
    settings = config.raw.get("explainability", {})
    if smoke:
        behavior = build_behavioral_features(
            make_synthetic_paysim(seed=config.seed), config.windows
        )
    else:
        prepare_real_data(config)
        behavior = load_prepared_frame(config.processed_path)
    split = temporal_split(
        behavior,
        config.train_fraction,
        config.validation_fraction,
        config.time_column,
        split_mode=str(config.raw["data"].get("split_mode", "time_fraction")),
    )
    feature_group = str(settings.get("feature_group", "recipient_full"))
    weight = float(settings.get("scale_pos_weight", 2.0))
    columns = feature_columns_for_group(split.train, feature_group)
    model_config = config.raw["model"]
    model, columns = build_lightgbm_pipeline(
        split.train,
        seed=config.seed,
        scale_pos_weight=weight,
        n_estimators=int(model_config["n_estimators"]),
        learning_rate=float(model_config["learning_rate"]),
        num_leaves=int(model_config["num_leaves"]),
        feature_columns=columns,
        subsample=float(model_config.get("subsample", 1.0)),
        colsample_bytree=float(model_config.get("colsample_bytree", 1.0)),
    )
    model.fit(split.train[columns], split.train[config.target])
    sample_size = min(int(settings.get("sample_rows", 5000)), len(split.test))
    sample = _balanced_sample(split.test, config.target, sample_size, config.seed)
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(sample[columns])
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    contributions = np.asarray(classifier.booster_.predict(transformed, pred_contrib=True))
    shap_values = contributions[:, :-1]
    if shap_values.shape[1] != len(feature_names):
        raise RuntimeError("TreeSHAP output does not match transformed feature names")
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    output_dir = config.output_dir / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    _plot_importance(importance, output_dir / "shap_feature_importance.png")
    probabilities = model.predict_proba(sample[columns])[:, 1]
    failures = sample[[config.time_column, "type", "amount", config.target]].copy()
    failures["probability"] = probabilities
    failures["absolute_error"] = np.abs(failures[config.target] - probabilities)
    failures.sort_values("absolute_error", ascending=False).head(200).to_csv(
        output_dir / "failure_cases.csv", index=False
    )
    metadata: dict[str, Any] = {
        "method": "LightGBM exact TreeSHAP contributions",
        "feature_group": feature_group,
        "scale_pos_weight": weight,
        "sample_rows": len(sample),
        "sample_fraud_rows": int(sample[config.target].sum()),
        "seed": config.seed,
    }
    (output_dir / "shap_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return importance


def _balanced_sample(frame: pd.DataFrame, target: str, rows: int, seed: int) -> pd.DataFrame:
    positives = frame.loc[frame[target] == 1]
    negatives = frame.loc[frame[target] == 0]
    positive_rows = min(len(positives), max(1, rows // 2))
    negative_rows = min(len(negatives), rows - positive_rows)
    sampled = pd.concat(
        [
            positives.sample(n=positive_rows, random_state=seed),
            negatives.sample(n=negative_rows, random_state=seed),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def _plot_importance(importance: pd.DataFrame, path: str | Path) -> None:
    top = importance.head(15).copy()
    top["display_name"] = top["feature"].map(_display_feature_name)
    top = top.sort_values("mean_abs_shap")
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    axis.barh(top["display_name"], top["mean_abs_shap"], color="#0072B2")
    axis.set_xscale("log")
    axis.set_xlabel("Mean absolute TreeSHAP value")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.25)
    output_path = Path(path)
    figure.savefig(output_path, dpi=300)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def _display_feature_name(name: str) -> str:
    prefixes = ("numerical__", "categorical__")
    result = name
    for prefix in prefixes:
        if result.startswith(prefix):
            result = result.removeprefix(prefix)
    return result.replace("_", " ").replace("dest", "recipient")
