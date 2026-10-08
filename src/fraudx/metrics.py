"""Imbalance-aware metrics, calibration diagnostics, and risk thresholds."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class MetricReport:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    false_positive_rate: float
    alert_rate: float
    prevalence: float
    f1: float
    pr_auc: float
    roc_auc: float
    mcc: float
    brier: float
    ece: float
    tn: int
    fp: int
    fn: int
    tp: int
    threshold: float
    business_cost: float
    business_cost_per_1000: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    false_negative_cost: float = 10.0,
    false_positive_cost: float = 1.0,
) -> MetricReport:
    """Evaluate predictions without treating accuracy as a meaningful target."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    negative_total = tn + fp
    total = tn + fp + fn + tp
    specificity = tn / negative_total if negative_total else 0.0
    recall = float(recall_score(labels, predictions, zero_division=0))
    return MetricReport(
        accuracy=float((tn + tp) / total if total else 0.0),
        balanced_accuracy=float((recall + specificity) / 2),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=recall,
        specificity=float(specificity),
        false_positive_rate=float(1 - specificity),
        alert_rate=float((tp + fp) / total if total else 0.0),
        prevalence=float((tp + fn) / total if total else 0.0),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        pr_auc=float(average_precision_score(labels, scores)),
        roc_auc=float(roc_auc_score(labels, scores)),
        mcc=float(matthews_corrcoef(labels, predictions)),
        brier=float(brier_score_loss(labels, scores)),
        ece=expected_calibration_error(labels, scores),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        threshold=float(threshold),
        business_cost=float(false_negative_cost * fn + false_positive_cost * fp),
        business_cost_per_1000=float(
            (false_negative_cost * fn + false_positive_cost * fp) * 1000 / total
            if total
            else 0.0
        ),
    )


def optimize_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    false_negative_cost: float,
    false_positive_cost: float = 1.0,
    candidates: int = 99,
    maximum_fnr: float | None = None,
    maximum_false_positive_rate: float | None = None,
    maximum_alert_rate: float | None = None,
    minimum_precision: float | None = None,
) -> tuple[float, MetricReport]:
    """Select the exact validation threshold minimizing asymmetric business cost.

    ``candidates`` remains part of the public interface for backward compatibility,
    but threshold search now evaluates every distinct score through the ROC operating
    points instead of a coarse quantile grid.
    Ties require exact equality of the computed cost, followed by maximum integer
    true positives, minimum integer alerts, and the highest threshold. Approximate
    equality must not admit a more expensive operating point.
    """
    if candidates < 3:
        raise ValueError("At least three threshold candidates are required")
    labels = np.asarray(y_true, dtype=int)
    scores = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("Labels and probabilities must be non-empty arrays of equal shape")
    if np.unique(labels).size < 2:
        raise ValueError("Threshold optimization requires both classes")
    _validate_rate_constraint("maximum_fnr", maximum_fnr)
    _validate_rate_constraint(
        "maximum_false_positive_rate",
        maximum_false_positive_rate,
    )
    _validate_rate_constraint("maximum_alert_rate", maximum_alert_rate)
    _validate_rate_constraint("minimum_precision", minimum_precision)

    false_positive_rates, true_positive_rates, thresholds = roc_curve(
        labels,
        scores,
        drop_intermediate=False,
    )
    positive_total = int(labels.sum())
    negative_total = int(labels.size - positive_total)
    false_positives = np.rint(false_positive_rates * negative_total).astype(int)
    true_positives = np.rint(true_positive_rates * positive_total).astype(int)
    false_negatives = positive_total - true_positives
    predicted_positives = true_positives + false_positives
    precisions = np.divide(
        true_positives,
        predicted_positives,
        out=np.zeros_like(true_positive_rates, dtype=float),
        where=predicted_positives > 0,
    )
    costs = false_negative_cost * false_negatives + false_positive_cost * false_positives
    eligible = np.ones_like(costs, dtype=bool)
    if maximum_fnr is not None:
        eligible &= false_negatives / positive_total <= maximum_fnr
    if maximum_false_positive_rate is not None:
        eligible &= false_positive_rates <= maximum_false_positive_rate
    if maximum_alert_rate is not None:
        eligible &= predicted_positives / labels.size <= maximum_alert_rate
    if minimum_precision is not None:
        eligible &= precisions >= minimum_precision
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        raise ValueError("No threshold satisfies the requested operating constraints")

    best_cost = float(costs[eligible_indices].min())
    tied = eligible_indices[costs[eligible_indices] == best_cost]
    tied = tied[true_positives[tied] == true_positives[tied].max()]
    tied = tied[predicted_positives[tied] == predicted_positives[tied].min()]
    finite_tied = tied[np.isfinite(thresholds[tied])]
    selected_index = int(
        finite_tied[np.argmax(thresholds[finite_tied])]
        if finite_tied.size
        else tied[0]
    )
    threshold = float(thresholds[selected_index])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(scores.max(), np.inf))
    report = evaluate_probabilities(
        labels,
        scores,
        threshold=threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    return threshold, report


def _validate_rate_constraint(name: str, value: float | None) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error."""
    labels = np.asarray(y_true, dtype=float)
    scores = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (scores >= lower) & (scores < upper if upper < 1 else scores <= upper)
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(float(labels[mask].mean() - scores[mask].mean()))
    return error if total else 0.0
