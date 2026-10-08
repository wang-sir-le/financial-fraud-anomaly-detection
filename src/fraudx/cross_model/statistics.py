"""Exact grouped bootstrap kernels and explicitly paired component estimands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraudx.q2_extension.common import require
from fraudx.q2_extension.kernels import ModelView, model_view, weighted_ap, weighted_tp


@dataclass
class PositiveLevelView:
    """AP needs cumulative counts only at score levels containing positives."""

    order: np.ndarray
    partition: np.ndarray
    total_at_positive: np.ndarray
    positive: np.ndarray


def positive_level_view(
    ids: np.ndarray, labels: np.ndarray, scores: np.ndarray, positions: np.ndarray, groups: int
) -> PositiveLevelView:
    _, partition = np.unique(scores, return_inverse=True)
    mask = labels == 1
    positive_levels = np.unique(partition[mask])
    totals = np.zeros((groups, len(positive_levels)), dtype=np.float64)
    positives = np.zeros_like(totals)
    np.add.at(positives, (positions[mask], np.searchsorted(positive_levels, partition[mask])), 1)
    by_time = np.argsort(positions, kind="stable")
    ends = np.cumsum(np.bincount(positions, minlength=groups))
    start = 0
    for group, end in enumerate(ends):
        levels = np.sort(partition[by_time[start:end]])
        totals[group] = len(levels) - np.searchsorted(levels, positive_levels, side="left")
        start = int(end)
    return PositiveLevelView(np.lexsort((ids, -scores)), partition, totals, positives)


def positive_level_ap(view: PositiveLevelView, weights: np.ndarray) -> np.ndarray:
    multipliers = weights.astype(np.float64)
    positive = (multipliers @ view.positive)[:, ::-1]
    totals = (multipliers @ view.total_at_positive)[:, ::-1]
    cumulative_f = np.cumsum(positive, axis=1)
    precision = np.divide(cumulative_f, totals, out=np.zeros_like(totals), where=totals > 0)
    frauds = positive.sum(axis=1)
    return np.divide(
        (positive * precision).sum(axis=1),
        frauds,
        out=np.full(len(weights), np.nan),
        where=frauds > 0,
    )


def build_view(
    ids: np.ndarray, labels: np.ndarray, scores: np.ndarray, positions: np.ndarray, groups: int
) -> Any:
    # Same estimand and exact tie treatment; choose arithmetic representation by time dimension.
    if groups < 1000:
        return positive_level_view(ids, labels, scores, positions, groups)
    return model_view(ids, labels, scores, positions, groups)


def calculate(
    view: Any, positions: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> pd.DataFrame:
    group_n = np.bincount(positions, minlength=weights.shape[1]).astype(np.int64)
    group_f = np.bincount(positions, weights=labels, minlength=weights.shape[1]).astype(np.int64)
    n, frauds = weights @ group_n, weights @ group_f
    alerts = np.ceil(0.03 * n).astype(np.int64)
    ap = np.empty(len(weights), dtype=np.float64)
    tp = np.empty(len(weights), dtype=np.int64)
    rank_view = ModelView(view.order, None, None, view.partition)
    for start in range(0, len(weights), 4):
        stop = min(start + 4, len(weights))
        batch = weights[start:stop]
        ap[start:stop] = (
            positive_level_ap(view, batch)
            if isinstance(view, PositiveLevelView)
            else weighted_ap(view, batch)
        )
        tp[start:stop] = weighted_tp(rank_view, labels, positions, batch, alerts[start:stop])
    return pd.DataFrame(
        {
            "replicate_id": np.arange(1, len(weights) + 1),
            "n": n,
            "frauds": frauds,
            "alerts": alerts,
            "tp": tp,
            "ap": ap,
        }
    )


def components(base: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    keys = ["replicate_id", "n", "frauds", "alerts"]
    require(np.array_equal(base[keys], candidate[keys]), "Unpaired bootstrap populations")
    out = base[keys].copy()
    out["delta_ap"] = candidate.ap - base.ap
    out["delta_tp"] = candidate.tp - base.tp
    out["delta_recall"] = out.delta_tp / out.frauds.replace(0, np.nan)
    return out


def transferred(base: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    require(
        np.array_equal(
            base[["replicate_id", "n", "frauds"]], candidate[["replicate_id", "n", "frauds"]]
        ),
        "Unpaired transfer",
    )
    out = base[["replicate_id", "n", "frauds"]].copy()
    out["delta_recall"] = (candidate.threshold_tp - base.threshold_tp) / base.frauds.replace(
        0, np.nan
    )
    out["cost_reduction"] = base.threshold_cost - candidate.threshold_cost
    out["delta_alert_rate"] = (
        candidate.threshold_tp + candidate.threshold_fp - base.threshold_tp - base.threshold_fp
    ) / base.n
    out["delta_ap"] = candidate.ap - base.ap
    return out


def paired_difference(lr: np.ndarray, lgbm_seed_effects: list[np.ndarray]) -> np.ndarray:
    require(len(lgbm_seed_effects) == 5, "Expected exactly five frozen LightGBM effects")
    require(all(x.shape == lr.shape for x in lgbm_seed_effects), "Unpaired effect shapes")
    return lr - np.mean(np.stack(lgbm_seed_effects), axis=0)
