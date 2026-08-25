"""Three-layer probability calibration experiment on frozen rolling folds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from fraudx.calibration_metrics import (
    bayes_capacity_decisions,
    binary_decision_metrics,
    jaccard_similarity,
    probability_metrics,
    reliability_bins,
    threshold_raw_score_percentile,
    tie_statistics,
)
from fraudx.calibration_reporting import (
    make_calibration_figures,
    summarize_calibration,
    validate_calibration_outputs,
    write_calibration_report,
)
from fraudx.config import ExperimentConfig
from fraudx.data import rolling_temporal_splits, validation_period_masks
from fraudx.features import build_behavioral_features, feature_columns_for_group
from fraudx.metrics import evaluate_probabilities, optimize_threshold
from fraudx.models import PlattCalibrator, build_lightgbm_pipeline
from fraudx.pipeline import prepare_real_data
from fraudx.preprocess import load_prepared_frame
from fraudx.robustness import _fold_boundaries, _scheme_name
from fraudx.robustness_reporting import cost_saving_fields
from fraudx.synthetic import make_synthetic_paysim

CALIBRATION_METHODS = ("Uncalibrated", "Platt", "Isotonic")
CALIBRATION_MODELS = (
    ("raw_safe", "raw_safe", 1.0),
    ("recipient_full_weight_2", "recipient_full", 2.0),
)


class ProbabilityTransform(Protocol):
    """Minimal shared interface for calibration branches."""

    def transform(self, probabilities: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class IdentityCalibrator:
    """Explicit no-calibration branch with no fitted state."""

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(probabilities, dtype=float).copy()


@dataclass
class BranchArtifacts:
    """Scores and decisions needed for within-model comparisons."""

    validation_scores: np.ndarray
    test_scores: np.ndarray
    validation_decisions: np.ndarray
    test_decisions: np.ndarray
    result: dict[str, Any]


def run_calibration_ablation(
    config: ExperimentConfig,
    *,
    smoke: bool = False,
) -> dict[str, pd.DataFrame]:
    """Train 30 locked models and evaluate 90 shared-score calibration branches."""
    _require_prerequisites(config, smoke=smoke)
    settings = config.raw.get("calibration", {})
    seeds = [int(value) for value in settings.get("seeds", [42, 52, 62, 72, 82])]
    folds = int(config.raw.get("robustness", {}).get("folds", 3))
    models = list(CALIBRATION_MODELS)
    if smoke:
        seeds = seeds[:2]
        folds = 2
        frame = build_behavioral_features(
            make_synthetic_paysim(rows=1500, seed=config.seed), config.windows
        )
    else:
        prepare_real_data(config)
        frame = load_prepared_frame(config.processed_path)
    robustness = config.raw.get("robustness", {})
    splits = rolling_temporal_splits(
        frame,
        folds=folds,
        initial_train_fraction=float(robustness.get("initial_train_fraction", 0.4)),
        validation_fraction=float(robustness.get("validation_fraction", 0.1)),
        test_fraction=float(robustness.get("test_fraction", 0.1)),
        time_column=config.time_column,
    )
    fn_cost = float(settings.get("false_negative_cost", 100.0))
    fp_cost = float(settings.get("false_positive_cost", 1.0))
    capacity = float(settings.get("maximum_alert_rate", 0.03))
    bins = int(settings.get("ece_bins", 10))
    bayes_threshold = fp_cost / (fp_cost + fn_cost)
    result_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    invariance_rows: list[dict[str, Any]] = []
    bayes_rows: list[dict[str, Any]] = []
    reliability_pair_rows: list[dict[str, Any]] = []
    model_config = config.raw["model"]
    for fold, split in enumerate(splits, start=1):
        calibration_mask, threshold_mask = validation_period_masks(
            split.validation, config.time_column, config.target
        )
        calibration_labels = split.validation.loc[
            calibration_mask, config.target
        ].to_numpy()
        validation_labels = split.validation.loc[threshold_mask, config.target].to_numpy()
        test_labels = split.test[config.target].to_numpy()
        boundaries = _fold_boundaries(split, config)
        for model_id, feature_group, weight in models:
            columns = feature_columns_for_group(split.train, feature_group)
            for seed in seeds:
                model, selected_columns = build_lightgbm_pipeline(
                    split.train,
                    seed=seed,
                    scale_pos_weight=weight,
                    n_estimators=int(model_config["n_estimators"]),
                    learning_rate=float(model_config["learning_rate"]),
                    num_leaves=int(model_config["num_leaves"]),
                    feature_columns=columns,
                    subsample=float(model_config.get("subsample", 1.0)),
                    colsample_bytree=float(model_config.get("colsample_bytree", 1.0)),
                )
                model.named_steps["classifier"].set_params(
                    random_state=seed,
                    bagging_seed=seed,
                    feature_fraction_seed=seed,
                    data_random_seed=seed,
                )
                model.fit(split.train[selected_columns], split.train[config.target])
                validation_raw_all = model.predict_proba(
                    split.validation[selected_columns]
                )[:, 1]
                calibration_raw = validation_raw_all[calibration_mask]
                validation_raw = validation_raw_all[threshold_mask]
                test_raw = model.predict_proba(split.test[selected_columns])[:, 1]
                raw_hash = _array_hash(test_raw)
                calibrators = _fit_calibrators(calibration_raw, calibration_labels)
                branches: dict[str, BranchArtifacts] = {}
                for method, calibrator in calibrators.items():
                    calibration_scores = calibrator.transform(calibration_raw)
                    validation_scores = calibrator.transform(validation_raw)
                    test_scores = calibrator.transform(test_raw)
                    branch, curves = _evaluate_branch(
                        fold=fold,
                        scheme=_scheme_name(model_id),
                        model_id=model_id,
                        feature_group=feature_group,
                        weight=weight,
                        seed=seed,
                        method=method,
                        calibration_labels=calibration_labels,
                        calibration_raw=calibration_raw,
                        calibration_scores=calibration_scores,
                        validation_labels=validation_labels,
                        validation_raw=validation_raw,
                        validation_scores=validation_scores,
                        test_labels=test_labels,
                        test_raw=test_raw,
                        test_scores=test_scores,
                        raw_hash=raw_hash,
                        calibrator=calibrator,
                        fn_cost=fn_cost,
                        fp_cost=fp_cost,
                        capacity=capacity,
                        bins=bins,
                        boundaries=boundaries,
                    )
                    branches[method] = branch
                    result_rows.append(branch.result)
                    curve_frames.extend(curves)
                    bayes_rows.append(
                        _evaluate_bayes_branch(
                            fold=fold,
                            scheme=_scheme_name(model_id),
                            model_id=model_id,
                            feature_group=feature_group,
                            weight=weight,
                            seed=seed,
                            method=method,
                            labels=test_labels,
                            scores=test_scores,
                            raw_scores=test_raw,
                            bayes_threshold=bayes_threshold,
                            capacity=capacity,
                            fn_cost=fn_cost,
                            fp_cost=fp_cost,
                            stable_indices=split.test.index.to_numpy(dtype=np.int64),
                        )
                    )
                invariance_rows.extend(
                    _compare_decisions(fold, model_id, seed, branches)
                )
                reliability_pair_rows.extend(
                    _compare_reliability(fold, model_id, seed, branches)
                )
    results = pd.DataFrame(result_rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    invariance = pd.DataFrame(invariance_rows)
    bayes = pd.DataFrame(bayes_rows)
    reliability_pairwise = pd.DataFrame(reliability_pair_rows)
    summary = summarize_calibration(results)
    expected_models = len(splits) * len(models) * len(seeds)
    checks = validate_calibration_outputs(
        results,
        curves,
        invariance,
        bayes,
        expected_model_trainings=expected_models,
        expected_branch_rows=expected_models * len(CALIBRATION_METHODS),
    )
    output_dir = config.output_dir / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "calibration_ablation_results.csv", index=False)
    summary.to_csv(output_dir / "calibration_ablation_summary.csv", index=False)
    curves.to_csv(output_dir / "calibration_curve_points.csv", index=False)
    invariance.to_csv(output_dir / "calibration_decision_invariance.csv", index=False)
    bayes.to_csv(output_dir / "calibration_bayes_decision_results.csv", index=False)
    reliability_pairwise.to_csv(
        output_dir / "calibration_pairwise_reliability.csv", index=False
    )
    figure_paths = make_calibration_figures(results, curves, output_dir / "figures")
    write_calibration_report(
        output_dir / "calibration_ablation_report.md",
        results=results,
        summary=summary,
        invariance=invariance,
        bayes=bayes,
        reliability_pairwise=reliability_pairwise,
        checks=checks,
        figure_paths=figure_paths,
    )
    return {
        "results": results,
        "summary": summary,
        "curves": curves,
        "invariance": invariance,
        "bayes": bayes,
        "reliability_pairwise": reliability_pairwise,
    }


def _fit_calibrators(
    raw_scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, ProbabilityTransform]:
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw_scores, labels)
    return {
        "Uncalibrated": IdentityCalibrator(),
        "Platt": PlattCalibrator().fit(raw_scores, labels),
        "Isotonic": isotonic,
    }


def _evaluate_branch(
    *,
    fold: int,
    scheme: str,
    model_id: str,
    feature_group: str,
    weight: float,
    seed: int,
    method: str,
    calibration_labels: np.ndarray,
    calibration_raw: np.ndarray,
    calibration_scores: np.ndarray,
    validation_labels: np.ndarray,
    validation_raw: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_raw: np.ndarray,
    test_scores: np.ndarray,
    raw_hash: str,
    calibrator: ProbabilityTransform,
    fn_cost: float,
    fp_cost: float,
    capacity: float,
    bins: int,
    boundaries: dict[str, int],
) -> tuple[BranchArtifacts, list[pd.DataFrame]]:
    val_metrics = probability_metrics(validation_labels, validation_scores)
    test_metrics = probability_metrics(test_labels, test_scores)
    raw_metrics = probability_metrics(test_labels, test_raw)
    curves: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}
    for split_name, labels, scores in [
        ("validation_threshold", validation_labels, validation_scores),
        ("test", test_labels, test_scores),
    ]:
        for binning in ["uniform", "quantile"]:
            points, diagnostic = reliability_bins(
                labels, scores, method=binning, n_bins=bins
            )
            points.insert(0, "binning_method", binning)
            points.insert(0, "split", split_name)
            points.insert(0, "calibration_method", method)
            points.insert(0, "seed", seed)
            points.insert(0, "weight", weight)
            points.insert(0, "scheme", scheme)
            points.insert(0, "model_id", model_id)
            points.insert(0, "fold", fold)
            curves.append(points)
            prefix = "val" if split_name == "validation_threshold" else "test"
            key = f"{prefix}_{binning}"
            diagnostics[f"{key}_ece_10"] = diagnostic.ece
            diagnostics[f"{key}_nonempty_bin_count"] = diagnostic.nonempty_bin_count
            diagnostics[f"{key}_empty_bin_count"] = diagnostic.empty_bin_count
            diagnostics[f"{key}_maximum_bin_sample_share"] = (
                diagnostic.maximum_bin_sample_share
            )
            diagnostics[f"{key}_minimum_nonempty_bin_sample_count"] = (
                diagnostic.minimum_nonempty_bin_sample_count
            )
            diagnostics[f"{key}_maximum_bin_sample_count"] = (
                diagnostic.maximum_bin_sample_count
            )
    threshold, validation_report = optimize_threshold(
        validation_labels,
        validation_scores,
        false_negative_cost=fn_cost,
        false_positive_cost=fp_cost,
        maximum_alert_rate=capacity,
    )
    validation_decisions = validation_scores >= threshold
    test_decisions = test_scores >= threshold
    test_report = evaluate_probabilities(
        test_labels,
        test_scores,
        threshold=threshold,
        false_negative_cost=fn_cost,
        false_positive_cost=fp_cost,
    )
    cost_fields = cost_saving_fields(
        test_report,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        target_capacity=capacity,
        test_size=len(test_labels),
        test_fraud_count=int(test_labels.sum()),
    )
    isotonic_fields: dict[str, float | int] = {
        "isotonic_step_count": np.nan,
        "isotonic_unique_output_count": np.nan,
        "largest_step_sample_share": np.nan,
        "fraction_output_zero": np.nan,
        "fraction_output_one": np.nan,
        "tie_fraction": np.nan,
        "number_tied_after_isotonic": np.nan,
        "maximum_tie_group_size": np.nan,
    }
    if method == "Isotonic":
        ties = tie_statistics(test_scores)
        isotonic_fields = {
            "isotonic_step_count": int(
                len(getattr(calibrator, "X_thresholds_", np.asarray([])))
            ),
            "isotonic_unique_output_count": ties["unique_output_count"],
            "largest_step_sample_share": ties["largest_step_sample_share"],
            "fraction_output_zero": ties["fraction_output_zero"],
            "fraction_output_one": ties["fraction_output_one"],
            "tie_fraction": ties["tie_fraction"],
            "number_tied_after_isotonic": ties["number_samples_in_ties"],
            "maximum_tie_group_size": ties["maximum_tie_group_size"],
        }
    result: dict[str, Any] = {
        "fold": fold,
        "scheme": scheme,
        "model_id": model_id,
        "feature_set": feature_group,
        "weight": weight,
        "seed": seed,
        "calibration_method": method,
        "raw_prediction_hash": raw_hash,
        "calibration_subset_size": len(calibration_labels),
        "calibration_subset_fraud_count": int(calibration_labels.sum()),
        "calibration_subset_fraud_rate": float(calibration_labels.mean()),
        "raw_unique_probability_count": int(np.unique(test_raw).size),
        "calibrated_unique_probability_count": int(np.unique(test_scores).size),
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "target_capacity": capacity,
        "decision_strategy": "empirical_cost_optimal",
        "empirical_threshold": threshold,
        "empirical_threshold_candidate_count": int(
            np.unique(validation_scores).size + 1
        ),
        "threshold_raw_score_percentile": threshold_raw_score_percentile(
            validation_raw, validation_decisions
        ),
        "raw_pr_auc": raw_metrics["pr_auc"],
        "calibrated_pr_auc": test_metrics["pr_auc"],
        "val_brier": val_metrics["brier"],
        "test_brier": test_metrics["brier"],
        "val_ece_uniform_10": diagnostics["val_uniform_ece_10"],
        "test_ece_uniform_10": diagnostics["test_uniform_ece_10"],
        "val_ece_quantile_10": diagnostics["val_quantile_ece_10"],
        "test_ece_quantile_10": diagnostics["test_quantile_ece_10"],
        "val_log_loss": val_metrics["log_loss"],
        "test_log_loss": test_metrics["log_loss"],
        "validation_alert_rate": validation_report.alert_rate,
        "validation_business_cost": validation_report.business_cost,
        "validation_selected_fraction": float(validation_decisions.mean()),
        "test_alert_rate": test_report.alert_rate,
        "test_selected_fraction": float(test_decisions.mean()),
        "capacity_deviation": test_report.alert_rate - capacity,
        "capacity_deviation_abs": abs(test_report.alert_rate - capacity),
        "precision": test_report.precision,
        "recall": test_report.recall,
        "f1": test_report.f1,
        "tp": test_report.tp,
        "fp": test_report.fp,
        "tn": test_report.tn,
        "fn": test_report.fn,
        **cost_fields,
        **diagnostics,
        **isotonic_fields,
        **boundaries,
        "test_probability_p99": float(np.quantile(test_scores, 0.99)),
        "test_probability_p999": float(np.quantile(test_scores, 0.999)),
    }
    return (
        BranchArtifacts(
            validation_scores=validation_scores,
            test_scores=test_scores,
            validation_decisions=validation_decisions,
            test_decisions=test_decisions,
            result=result,
        ),
        curves,
    )


def _evaluate_bayes_branch(
    *,
    fold: int,
    scheme: str,
    model_id: str,
    feature_group: str,
    weight: float,
    seed: int,
    method: str,
    labels: np.ndarray,
    scores: np.ndarray,
    raw_scores: np.ndarray,
    bayes_threshold: float,
    capacity: float,
    fn_cost: float,
    fp_cost: float,
    stable_indices: np.ndarray,
) -> dict[str, Any]:
    decisions, selection = bayes_capacity_decisions(
        scores,
        raw_scores,
        threshold=bayes_threshold,
        capacity=capacity,
        stable_indices=stable_indices,
    )
    metrics = binary_decision_metrics(
        labels, decisions, fn_cost=fn_cost, fp_cost=fp_cost
    )
    return {
        "fold": fold,
        "scheme": scheme,
        "model_id": model_id,
        "feature_set": feature_group,
        "weight": weight,
        "seed": seed,
        "calibration_method": method,
        "decision_strategy": "bayes_cost_threshold",
        "bayes_threshold": bayes_threshold,
        "capacity": capacity,
        **selection,
        **metrics,
    }


def _compare_decisions(
    fold: int,
    model_id: str,
    seed: int,
    branches: dict[str, BranchArtifacts],
) -> list[dict[str, Any]]:
    baseline = branches["Uncalibrated"]
    rows: list[dict[str, Any]] = []
    for method in ["Platt", "Isotonic"]:
        candidate = branches[method]
        validation_disagreement = (
            baseline.validation_decisions != candidate.validation_decisions
        )
        test_disagreement = baseline.test_decisions != candidate.test_decisions
        rho = spearmanr(
            baseline.validation_scores, candidate.validation_scores
        ).statistic
        rows.append(
            {
                "fold": fold,
                "scheme": baseline.result["scheme"],
                "model_id": model_id,
                "weight": baseline.result["weight"],
                "seed": seed,
                "comparison": f"{method} vs Uncalibrated",
                "spearman_rank_correlation": float(rho),
                "validation_disagreement_count": int(validation_disagreement.sum()),
                "validation_disagreement_rate": float(validation_disagreement.mean()),
                "validation_selected_count_raw": int(
                    baseline.validation_decisions.sum()
                ),
                "validation_selected_count_method": int(
                    candidate.validation_decisions.sum()
                ),
                "validation_jaccard": jaccard_similarity(
                    baseline.validation_decisions, candidate.validation_decisions
                ),
                "validation_business_cost_raw": baseline.result[
                    "validation_business_cost"
                ]
                if "validation_business_cost" in baseline.result
                else np.nan,
                "validation_business_cost_method": candidate.result[
                    "validation_business_cost"
                ]
                if "validation_business_cost" in candidate.result
                else np.nan,
                "test_disagreement_count": int(test_disagreement.sum()),
                "test_disagreement_rate": float(test_disagreement.mean()),
                "test_jaccard": jaccard_similarity(
                    baseline.test_decisions, candidate.test_decisions
                ),
                "recall_difference": candidate.result["recall"]
                - baseline.result["recall"],
                "fp_difference": candidate.result["fp"] - baseline.result["fp"],
                "fn_difference": candidate.result["fn"] - baseline.result["fn"],
                "business_cost_difference": candidate.result["business_cost"]
                - baseline.result["business_cost"],
                "alert_rate_difference": candidate.result["test_alert_rate"]
                - baseline.result["test_alert_rate"],
                "raw_pr_auc_difference": candidate.result["raw_pr_auc"]
                - baseline.result["raw_pr_auc"],
                "calibrated_pr_auc_difference": candidate.result["calibrated_pr_auc"]
                - baseline.result["calibrated_pr_auc"],
            }
        )
    return rows


def _compare_reliability(
    fold: int,
    model_id: str,
    seed: int,
    branches: dict[str, BranchArtifacts],
) -> list[dict[str, Any]]:
    comparisons = [
        ("Platt", "Uncalibrated"),
        ("Isotonic", "Uncalibrated"),
        ("Isotonic", "Platt"),
    ]
    rows: list[dict[str, Any]] = []
    for method, baseline_method in comparisons:
        current = branches[method].result
        baseline = branches[baseline_method].result
        rows.append(
            {
                "fold": fold,
                "scheme": current["scheme"],
                "model_id": model_id,
                "weight": current["weight"],
                "seed": seed,
                "comparison": f"{method} vs {baseline_method}",
                "brier_improvement": baseline["test_brier"]
                - current["test_brier"],
                "ece_uniform_improvement": baseline["test_ece_uniform_10"]
                - current["test_ece_uniform_10"],
                "ece_quantile_improvement": baseline["test_ece_quantile_10"]
                - current["test_ece_quantile_10"],
                "log_loss_improvement": baseline["test_log_loss"]
                - current["test_log_loss"],
                "business_cost_reduction": baseline["business_cost"]
                - current["business_cost"],
                "capacity_stability_improvement": baseline[
                    "capacity_deviation_abs"
                ]
                - current["capacity_deviation_abs"],
            }
        )
    return rows


def _array_hash(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _require_prerequisites(config: ExperimentConfig, *, smoke: bool) -> None:
    if smoke:
        return
    pair_status = config.output_dir / "pair_audit" / "pair_audit_status.json"
    multiseed = config.output_dir / "robustness" / "rolling_multiseed_results.csv"
    if not pair_status.exists() or not multiseed.exists():
        raise RuntimeError("Frozen Pair audit and multi-seed outputs are required")
    status = json.loads(pair_status.read_text(encoding="utf-8"))
    if not bool(status.get("safe_to_proceed")):
        raise RuntimeError("Pair audit does not permit the calibration experiment")
