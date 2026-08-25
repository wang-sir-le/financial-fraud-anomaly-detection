"""Pure helpers for probability-calibration and decision-policy audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class ReliabilityDiagnostics:
    """ECE value plus information needed to judge bin quality."""

    ece: float
    nonempty_bin_count: int
    empty_bin_count: int
    maximum_bin_sample_share: float
    minimum_nonempty_bin_sample_count: int
    maximum_bin_sample_count: int


def reliability_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    method: str,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, ReliabilityDiagnostics]:
    """Return machine-readable calibration bins and weighted ECE diagnostics."""
    labels, scores = _validated_arrays(y_true, probabilities)
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    if method == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        points = _points_from_edges(labels, scores, edges, include_empty=True)
    elif method == "quantile":
        edges = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size == 1:
            edges = np.asarray([edges[0], np.nextafter(edges[0], np.inf)])
        points = _points_from_edges(labels, scores, edges, include_empty=False)
    else:
        raise ValueError("method must be 'uniform' or 'quantile'")
    nonempty = points.loc[points["sample_count"] > 0]
    counts = nonempty["sample_count"].to_numpy(dtype=int)
    weights = counts / len(labels)
    errors = np.abs(
        nonempty["observed_fraud_rate"].to_numpy(dtype=float)
        - nonempty["mean_predicted_probability"].to_numpy(dtype=float)
    )
    diagnostics = ReliabilityDiagnostics(
        ece=float(np.sum(weights * errors)),
        nonempty_bin_count=int(len(nonempty)),
        empty_bin_count=int(n_bins - len(nonempty)),
        maximum_bin_sample_share=float(counts.max() / len(labels)),
        minimum_nonempty_bin_sample_count=int(counts.min()),
        maximum_bin_sample_count=int(counts.max()),
    )
    return points, diagnostics


def probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate calibration-focused metrics under one shared clipping rule."""
    labels, scores = _validated_arrays(y_true, probabilities)
    clipped = np.clip(scores, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {
        "brier": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def tie_statistics(probabilities: np.ndarray) -> dict[str, float | int]:
    """Describe score compression and ties, particularly after Isotonic mapping."""
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    _values, counts = np.unique(scores, return_counts=True)
    tied = counts[counts > 1]
    return {
        "unique_output_count": int(len(counts)),
        "number_tied_groups": int(len(tied)),
        "number_samples_in_ties": int(tied.sum()) if tied.size else 0,
        "tie_fraction": float(tied.sum() / len(scores)) if tied.size else 0.0,
        "maximum_tie_group_size": int(tied.max()) if tied.size else 1,
        "largest_step_sample_share": float(counts.max() / len(scores)),
        "fraction_output_zero": float(np.mean(scores == 0.0)),
        "fraction_output_one": float(np.mean(scores == 1.0)),
    }


def threshold_raw_score_percentile(
    raw_scores: np.ndarray,
    selected: np.ndarray,
) -> float:
    """Locate the selected-set lower boundary in the raw-score distribution."""
    scores = np.asarray(raw_scores, dtype=float)
    mask = np.asarray(selected, dtype=bool)
    if scores.shape != mask.shape or scores.size == 0:
        raise ValueError("raw_scores and selected must be equal non-empty vectors")
    if not mask.any():
        return 1.0
    boundary = float(scores[mask].min())
    return float(np.mean(scores <= boundary))


def jaccard_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Jaccard similarity for two binary decision vectors."""
    first = np.asarray(left, dtype=bool)
    second = np.asarray(right, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Decision vectors must have equal shape")
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def bayes_capacity_decisions(
    probabilities: np.ndarray,
    raw_scores: np.ndarray,
    *,
    threshold: float,
    capacity: float,
    stable_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | bool | int]]:
    """Apply a fixed Bayes threshold and deterministic top-capacity truncation."""
    scores = np.asarray(probabilities, dtype=float)
    raw = np.asarray(raw_scores, dtype=float)
    if scores.shape != raw.shape or scores.ndim != 1 or scores.size == 0:
        raise ValueError("probabilities and raw_scores must be equal non-empty vectors")
    if not 0 < threshold < 1 or not 0 < capacity <= 1:
        raise ValueError("threshold and capacity must lie in (0, 1]")
    indices = (
        np.arange(scores.size, dtype=np.int64)
        if stable_indices is None
        else np.asarray(stable_indices, dtype=np.int64)
    )
    if indices.shape != scores.shape or np.unique(indices).size != indices.size:
        raise ValueError("stable_indices must be unique and aligned with scores")
    candidates = scores >= threshold
    candidate_count = int(candidates.sum())
    limit = int(np.floor(capacity * len(scores)))
    decisions = candidates.copy()
    binding = candidate_count > limit
    if binding:
        candidate_positions = np.flatnonzero(candidates)
        ordering = np.lexsort(
            (
                indices[candidate_positions],
                -raw[candidate_positions],
                -scores[candidate_positions],
            )
        )
        decisions[:] = False
        decisions[candidate_positions[ordering[:limit]]] = True
    return decisions, {
        "candidate_count_before_cap": candidate_count,
        "candidate_alert_rate_before_cap": candidate_count / len(scores),
        "final_alert_count": int(decisions.sum()),
        "final_alert_rate": float(decisions.mean()),
        "capacity_binding": bool(binding),
    }


def binary_decision_metrics(
    y_true: np.ndarray,
    decisions: np.ndarray,
    *,
    fn_cost: float,
    fp_cost: float,
) -> dict[str, float | int]:
    """Evaluate a pre-computed decision vector, including asymmetric cost fields."""
    labels = np.asarray(y_true, dtype=int)
    predicted = np.asarray(decisions, dtype=int)
    if labels.shape != predicted.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels and decisions must be equal non-empty vectors")
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    false_negative_cost = fn * fn_cost
    false_positive_cost = fp * fp_cost
    business_cost = false_negative_cost + false_positive_cost
    all_normal_cost = int(labels.sum()) * fn_cost
    cost_saving = all_normal_cost - business_cost
    return {
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "alert_rate": float(predicted.mean()),
        "false_negative_cost": float(false_negative_cost),
        "false_positive_cost": float(false_positive_cost),
        "business_cost": float(business_cost),
        "all_normal_cost": float(all_normal_cost),
        "cost_saving": float(cost_saving),
        "cost_saving_rate": float(cost_saving / all_normal_cost)
        if all_normal_cost
        else float("nan"),
    }


def _validated_arrays(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if labels.shape != scores.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels and probabilities must be equal non-empty vectors")
    return labels, scores


def _points_from_edges(
    labels: np.ndarray,
    scores: np.ndarray,
    edges: np.ndarray,
    *,
    include_empty: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bin_id, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        last = bin_id == len(edges) - 2
        mask = (scores >= lower) & (scores <= upper if last else scores < upper)
        count = int(mask.sum())
        if not count and not include_empty:
            continue
        rows.append(
            {
                "bin_id": bin_id,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "sample_count": count,
                "positive_count": int(labels[mask].sum()) if count else 0,
                "mean_predicted_probability": float(scores[mask].mean())
                if count
                else float("nan"),
                "observed_fraud_rate": float(labels[mask].mean())
                if count
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)
