"""End-to-end reproducible smoke and full-data experiment runner."""

from __future__ import annotations

import gc
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from fraudx.config import ExperimentConfig
from fraudx.data import (
    TemporalSplit,
    temporal_split,
    validate_paysim,
    validation_period_masks,
)
from fraudx.features import build_behavioral_features, feature_columns_for_group
from fraudx.metrics import MetricReport, evaluate_probabilities, optimize_threshold
from fraudx.models import PlattCalibrator, build_lightgbm_pipeline
from fraudx.online import prequential_evaluate
from fraudx.preprocess import load_prepared_frame, prepare_paysim_csv, write_profile
from fraudx.synthetic import make_synthetic_paysim


def prepare_real_data(config: ExperimentConfig, force: bool = False) -> None:
    """Build and audit the real PaySim feature cache."""
    profile = prepare_paysim_csv(
        config.data_path,
        config.processed_path,
        config.windows,
        force=force,
    )
    write_profile(profile, config.output_dir / "metadata" / "preprocessing_profile.json")


def run_experiment(config: ExperimentConfig, smoke: bool = False) -> pd.DataFrame:
    """Run fair feature ablations, weight sensitivity, and constrained decisions."""
    output_dir = config.output_dir
    tables_dir = output_dir / "tables"
    metadata_dir = output_dir / "metadata"
    predictions_dir = output_dir / "predictions"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        raw = make_synthetic_paysim(seed=config.seed)
        validate_paysim(raw)
        behavior = build_behavioral_features(raw, config.windows)
    else:
        if not config.data_path.exists():
            raise FileNotFoundError(
                f"PaySim not found at {config.data_path}. Use --smoke or place the CSV there."
            )
        prepare_real_data(config)
        behavior = load_prepared_frame(config.processed_path)

    split_mode = str(config.raw["data"].get("split_mode", "time_fraction"))
    split = temporal_split(
        behavior,
        config.train_fraction,
        config.validation_fraction,
        config.time_column,
        split_mode=split_mode,
    )
    del behavior
    gc.collect()
    calibration_mask, threshold_mask = validation_period_masks(
        split.validation,
        config.time_column,
        config.target,
    )
    split_profile = _split_profile(
        split,
        calibration_mask,
        threshold_mask,
        config.target,
        config.time_column,
        split_mode,
    )
    _write_json(metadata_dir / "split_profile.json", split_profile)

    model_config = config.raw["model"]
    risk_config = config.raw["risk"]
    experiment_specs = _experiment_specs(config, smoke)
    risk_costs = [float(value) for value in risk_config["false_negative_costs"]]
    selection_cost = float(risk_config["selection_false_negative_cost"])
    if selection_cost not in risk_costs:
        raise ValueError("selection_false_negative_cost must be in false_negative_costs")
    false_positive_cost = float(risk_config.get("false_positive_cost", 1.0))
    alert_caps = _alert_caps(risk_config)
    maximum_fpr = _optional_float(risk_config.get("maximum_false_positive_rate"))
    minimum_precision = _optional_float(risk_config.get("minimum_precision"))

    validation_labels = split.validation.loc[threshold_mask, config.target].to_numpy()
    test_labels = split.test[config.target].to_numpy()
    records: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    model_metadata: list[dict[str, Any]] = []
    best_validation_pr_auc = -np.inf
    fixed_model_id = ""
    fixed_feature_group = ""
    fixed_columns: list[str] = []
    save_probabilities = bool(
        config.raw.get("experiment", {}).get("save_selected_probabilities", False)
    )

    for model_id, feature_group, weight in experiment_specs:
        columns = feature_columns_for_group(split.train, feature_group)
        started = perf_counter()
        pipeline, columns = build_lightgbm_pipeline(
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
        pipeline.fit(split.train[columns], split.train[config.target])
        training_seconds = perf_counter() - started
        validation_raw = pipeline.predict_proba(split.validation[columns])[:, 1]
        calibrator = PlattCalibrator().fit(
            validation_raw[calibration_mask],
            split.validation.loc[calibration_mask, config.target].to_numpy(),
        )
        validation_probability = calibrator.transform(validation_raw[threshold_mask])
        test_probability = calibrator.transform(
            pipeline.predict_proba(split.test[columns])[:, 1]
        )

        validation_default = evaluate_probabilities(
            validation_labels,
            validation_probability,
            false_negative_cost=selection_cost,
            false_positive_cost=false_positive_cost,
        )
        test_default = evaluate_probabilities(
            test_labels,
            test_probability,
            false_negative_cost=selection_cost,
            false_positive_cost=false_positive_cost,
        )
        ranking = _ranking_record(
            model_id,
            feature_group,
            weight,
            config.seed,
            split_mode,
            len(columns),
            training_seconds,
            validation_default,
            test_default,
        )
        rankings.append(ranking)
        model_metadata.append(
            {
                "model_id": model_id,
                "feature_group": feature_group,
                "scale_pos_weight": weight,
                "feature_columns": columns,
                "calibrator": {
                    "method": "Platt scaling",
                    "slope": calibrator.slope,
                    "intercept": calibrator.intercept,
                },
                "training_seconds": training_seconds,
            }
        )

        if validation_default.pr_auc > best_validation_pr_auc:
            best_validation_pr_auc = validation_default.pr_auc
            fixed_model_id = model_id
            fixed_feature_group = feature_group
            fixed_columns = columns
            if save_probabilities:
                predictions_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    predictions_dir / "validation_selected_model_scores.npz",
                    model_id=np.asarray([model_id]),
                    validation_labels=validation_labels,
                    validation_probabilities=validation_probability,
                    test_labels=test_labels,
                    test_probabilities=test_probability,
                    test_steps=split.test[config.time_column].to_numpy(),
                    test_amounts=split.test["amount"].to_numpy(),
                )

        for false_negative_cost in risk_costs:
            for maximum_alert_rate in alert_caps:
                threshold, validation_report = optimize_threshold(
                    validation_labels,
                    validation_probability,
                    false_negative_cost=false_negative_cost,
                    false_positive_cost=false_positive_cost,
                    candidates=int(risk_config.get("thresholds", 99)),
                    maximum_false_positive_rate=maximum_fpr,
                    maximum_alert_rate=maximum_alert_rate,
                    minimum_precision=minimum_precision,
                )
                test_report = evaluate_probabilities(
                    test_labels,
                    test_probability,
                    threshold=threshold,
                    false_negative_cost=false_negative_cost,
                    false_positive_cost=false_positive_cost,
                )
                record = {
                    "split_protocol": split_mode,
                    "experiment": _experiment_name(
                        model_id,
                        false_negative_cost,
                        maximum_alert_rate,
                    ),
                    "model_id": model_id,
                    "feature_group": feature_group,
                    "feature_count": len(columns),
                    "scale_pos_weight": weight,
                    "seed": config.seed,
                    "false_negative_cost": false_negative_cost,
                    "false_positive_cost": false_positive_cost,
                    "maximum_alert_rate": maximum_alert_rate,
                    **{
                        f"validation_{key}": value
                        for key, value in validation_report.to_dict().items()
                    },
                    **{
                        f"test_{key}": value
                        for key, value in test_report.to_dict().items()
                    },
                    **_deployment_metrics(
                        "validation",
                        split.validation.loc[threshold_mask, "amount"].to_numpy(),
                        validation_labels,
                        validation_probability,
                        threshold,
                        validation_report,
                        false_negative_cost,
                    ),
                    **_deployment_metrics(
                        "test",
                        split.test["amount"].to_numpy(),
                        test_labels,
                        test_probability,
                        threshold,
                        test_report,
                        false_negative_cost,
                    ),
                }
                records.append(record)

        del pipeline, validation_raw, validation_probability, test_probability
        gc.collect()

    results = pd.DataFrame(records)
    ranking_results = pd.DataFrame(rankings).sort_values(
        ["validation_pr_auc", "validation_brier"],
        ascending=[False, True],
    )
    results.to_csv(tables_dir / "ablation_results.csv", index=False)
    ranking_results.to_csv(tables_dir / "model_ranking_results.csv", index=False)
    test_columns = [column for column in ranking_results if column.startswith("test_")]
    ranking_results.drop(columns=test_columns).to_csv(
        tables_dir / "candidate_validation_results.csv",
        index=False,
    )

    selection_groups = ["false_negative_cost", "maximum_alert_rate"]
    selected_indices = results.groupby(selection_groups, dropna=False)[
        "validation_business_cost"
    ].idxmin()
    envelope = results.loc[selected_indices].sort_values(selection_groups)
    envelope.to_csv(
        tables_dir / "validation_selected_test_results.csv",
        index=False,
    )
    fixed_results = results.loc[results["model_id"] == fixed_model_id].sort_values(
        selection_groups
    )
    fixed_results.to_csv(
        tables_dir / "fixed_model_cost_sensitivity.csv",
        index=False,
    )
    ranking_results.loc[ranking_results["scale_pos_weight"] == 1.0].to_csv(
        tables_dir / "feature_ablation_ranking.csv",
        index=False,
    )
    weighted_group = str(
        config.raw.get("experiment", {}).get("weighted_feature_group", "recipient_full")
    )
    ranking_results.loc[ranking_results["feature_group"] == weighted_group].to_csv(
        tables_dir / "class_weight_ranking.csv",
        index=False,
    )

    run_summary: dict[str, Any] = {
        "status": "complete",
        "split_protocol": split_mode,
        "model_selection_rule": "highest threshold-selection validation PR-AUC",
        "fixed_model_for_cost_sensitivity": fixed_model_id,
        "fixed_feature_group": fixed_feature_group,
        "seed": config.seed,
        "selection_false_negative_cost": selection_cost,
        "false_positive_cost": false_positive_cost,
        "maximum_alert_rates": alert_caps,
        "maximum_false_positive_rate": maximum_fpr,
        "minimum_precision": minimum_precision,
        "software_versions": _software_versions(),
    }
    online_config = config.raw.get("online", {})
    if bool(online_config.get("enabled", True)):
        run_summary["online_baseline"] = _run_online_baseline(
            split,
            fixed_columns,
            config.target,
            selection_cost,
            false_positive_cost,
            online_config,
        )
    else:
        run_summary["online_baseline"] = {"enabled": False}
    _write_json(metadata_dir / "run_summary.json", run_summary)
    _write_json(metadata_dir / "model_metadata.json", model_metadata)
    return results


def _experiment_specs(
    config: ExperimentConfig,
    smoke: bool,
) -> list[tuple[str, str, float]]:
    experiment_config = config.raw.get("experiment", {})
    feature_groups = list(
        experiment_config.get("feature_groups", ["raw_safe", "recipient_full"])
    )
    if smoke:
        feature_groups = feature_groups[:3]
    weighted_group = str(experiment_config.get("weighted_feature_group", "recipient_full"))
    weights = [float(value) for value in config.raw["model"]["class_weights"]]
    if smoke:
        weights = weights[:2]
    specs = [(group, group, 1.0) for group in feature_groups]
    for weight in weights:
        if weight == 1.0:
            continue
        specs.append((f"{weighted_group}_weight_{weight:g}", weighted_group, weight))
    identifiers = [model_id for model_id, _, _ in specs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Experiment model identifiers must be unique")
    return specs


def _alert_caps(risk_config: dict[str, Any]) -> list[float | None]:
    raw_caps = risk_config.get("maximum_alert_rates")
    if raw_caps is None:
        single_cap = risk_config.get("maximum_alert_rate")
        return [None if single_cap is None else float(single_cap)]
    caps = [float(value) for value in raw_caps]
    if not caps:
        raise ValueError("maximum_alert_rates cannot be empty")
    if any(cap < 0 or cap > 1 for cap in caps):
        raise ValueError("maximum_alert_rates must be between 0 and 1")
    result: list[float | None] = []
    result.extend(sorted(set(caps)))
    return result


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _ranking_record(
    model_id: str,
    feature_group: str,
    weight: float,
    seed: int,
    split_mode: str,
    feature_count: int,
    training_seconds: float,
    validation: MetricReport,
    test: MetricReport,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "split_protocol": split_mode,
        "model_id": model_id,
        "feature_group": feature_group,
        "feature_count": feature_count,
        "scale_pos_weight": weight,
        "seed": seed,
        "training_seconds": training_seconds,
    }
    for prefix, report in (("validation", validation), ("test", test)):
        record.update(
            {
                f"{prefix}_rows": report.tn + report.fp + report.fn + report.tp,
                f"{prefix}_prevalence": report.prevalence,
                f"{prefix}_majority_accuracy": 1 - report.prevalence,
                f"{prefix}_pr_auc": report.pr_auc,
                f"{prefix}_pr_auc_lift": report.pr_auc / report.prevalence,
                f"{prefix}_roc_auc": report.roc_auc,
                f"{prefix}_brier": report.brier,
                f"{prefix}_ece": report.ece,
            }
        )
    return record


def _deployment_metrics(
    prefix: str,
    amounts: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    report: MetricReport,
    false_negative_cost: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    fraud_amount = float(amounts[labels == 1].sum())
    captured_amount = float(amounts[(labels == 1) & predictions].sum())
    rows = labels.size
    all_normal_cost = float((labels == 1).sum() * false_negative_cost)
    return {
        f"{prefix}_majority_accuracy": 1 - report.prevalence,
        f"{prefix}_accuracy_delta_vs_majority": report.accuracy - (1 - report.prevalence),
        f"{prefix}_pr_auc_lift": report.pr_auc / report.prevalence,
        f"{prefix}_fp_per_10000": report.fp * 10000 / rows,
        f"{prefix}_fn_per_10000": report.fn * 10000 / rows,
        f"{prefix}_fraud_amount_recall": (
            captured_amount / fraud_amount if fraud_amount else 0.0
        ),
        f"{prefix}_missed_fraud_amount": fraud_amount - captured_amount,
        f"{prefix}_business_cost_per_10000": report.business_cost * 10000 / rows,
        f"{prefix}_all_normal_business_cost": all_normal_cost,
        f"{prefix}_cost_saving_vs_all_normal": all_normal_cost - report.business_cost,
        f"{prefix}_cost_saving_rate_vs_all_normal": (
            (all_normal_cost - report.business_cost) / all_normal_cost
            if all_normal_cost
            else 0.0
        ),
    }


def _experiment_name(
    model_id: str,
    false_negative_cost: float,
    maximum_alert_rate: float | None,
) -> str:
    cap = "unconstrained" if maximum_alert_rate is None else f"alert_{maximum_alert_rate:g}"
    return f"{model_id}_cost_{false_negative_cost:g}_{cap}"


def _run_online_baseline(
    split: TemporalSplit,
    columns: list[str],
    target: str,
    false_negative_cost: float,
    false_positive_cost: float,
    online_config: dict[str, Any],
) -> dict[str, Any]:
    numeric_columns = [
        column
        for column in columns
        if column != "type" and pd.api.types.is_numeric_dtype(split.test[column])
    ]
    online_frame = split.test.head(int(online_config["max_rows"]))
    rows = [
        (
            {column: float(row[column]) for column in numeric_columns},
            int(row[target]),
        )
        for _, row in online_frame.iterrows()
    ]
    online = prequential_evaluate(
        rows,
        label_delay=int(online_config["label_delay"]),
        false_negative_cost=false_negative_cost,
    )
    report = evaluate_probabilities(
        np.asarray(online["labels"]),
        np.asarray(online["probabilities"]),
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    return {
        "enabled": True,
        "model": "independent River online logistic-regression baseline",
        "metrics": report.to_dict(),
        "drift_points": online["drift_points"],
    }


def _split_profile(
    split: TemporalSplit,
    calibration_mask: np.ndarray,
    threshold_mask: np.ndarray,
    target: str,
    time_column: str,
    split_mode: str,
) -> dict[str, Any]:
    total_rows = len(split.train) + len(split.validation) + len(split.test)

    def profile(frame: pd.DataFrame) -> dict[str, int | float]:
        positives = int(frame[target].sum())
        return {
            "rows": len(frame),
            "row_fraction": len(frame) / total_rows,
            "fraud_rows": positives,
            "fraud_rate": positives / len(frame),
            "min_step": int(frame[time_column].min()),
            "max_step": int(frame[time_column].max()),
        }

    return {
        "split_protocol": split_mode,
        "train": profile(split.train),
        "validation_full": profile(split.validation),
        "calibration": profile(split.validation.loc[calibration_mask]),
        "threshold_selection": profile(split.validation.loc[threshold_mask]),
        "test": profile(split.test),
    }


def _software_versions() -> dict[str, str]:
    packages = ["fraudx", "numpy", "pandas", "scikit-learn", "lightgbm", "duckdb"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "editable/local"
    return versions


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
