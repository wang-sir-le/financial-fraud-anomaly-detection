"""Small exact numerical kernels; tie ranges are not confidence intervals."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score

from fraudx.q2_extension.common import require


def capacity_profile(ids: np.ndarray, labels: np.ndarray, scores: np.ndarray,
                     q: float) -> tuple[dict[str, Any], np.ndarray]:
    require(0 < q <= 1 and len(ids) == len(labels) == len(scores), "Invalid capacity inputs")
    require(len(np.unique(ids)) == len(ids), "Duplicate IDs in original population")
    order = np.lexsort((ids, -scores))
    k = math.ceil(q * len(ids))
    boundary = scores[order[k - 1]]
    above = scores > boundary
    members = np.flatnonzero(scores == boundary)
    a = int(labels[above].sum())
    m, f = len(members), int(labels[members].sum())
    r = k - int(above.sum())
    tp = int(labels[order[:k]].sum())
    lo, hi = a + max(0, r - (m - f)), a + min(r, f)
    require(lo <= tp <= hi and 1 <= r <= m, "Invalid boundary counts")
    frauds = int(labels.sum())
    result = {"n": len(ids), "frauds": frauds, "q": q, "alerts": k,
              "tp": tp, "recall": tp / frauds if frauds else np.nan,
              "precision": tp / k, "cost": k - tp + 100 * (frauds - tp),
              "boundary_score": float(boundary), "above_tp": a, "tie_n": m,
              "tie_frauds": f, "remaining": r, "tp_min": lo, "tp_max": hi,
              "tp_expectation": a + r * f / m, "split_tie": r < m,
              "range_type": "offline_attainable_single_model_not_CI"}
    return result, members


def priority_ranks(ids: np.ndarray, dataset: str, fold: int, trial: int,
                   master_seed: int = 2026092301) -> np.ndarray:
    require(bool(np.all(ids[1:] > ids[:-1])), "Priority IDs must be sorted and unique")
    prefix = f"{master_seed}|{trial}|{dataset}|{fold}|".encode()
    digests = b"".join(hashlib.sha256(prefix + str(int(i)).encode()).digest() for i in ids)
    keys = np.frombuffer(digests, dtype="V32")
    order = np.argsort(keys, kind="stable")  # ID is secondary because ids are ascending.
    ranks: np.ndarray = np.empty(len(ids), dtype=np.int64)
    ranks[order] = np.arange(len(ids))
    return ranks


def priority_tp(profile: dict[str, Any], members: np.ndarray, labels: np.ndarray,
                ranks: np.ndarray) -> int:
    r = int(profile["remaining"])
    if r == len(members):
        return int(profile["above_tp"] + profile["tie_frauds"])
    selected = np.argpartition(ranks[members], r - 1)[:r]
    return int(profile["above_tp"] + labels[members[selected]].sum())


@dataclass
class ModelView:
    order: np.ndarray
    total: Any
    positive: Any
    partition: np.ndarray


def model_view(ids: np.ndarray, labels: np.ndarray, scores: np.ndarray,
               positions: np.ndarray, groups: int) -> ModelView:
    unique, index = np.unique(scores, return_inverse=True)
    shape = (groups, len(unique))
    total = sparse.coo_matrix((np.ones(len(ids)), (positions, index)), shape=shape).tocsr()
    mask = labels == 1
    positive = sparse.coo_matrix((np.ones(int(mask.sum())),
                                 (positions[mask], index[mask])), shape=shape).tocsr()
    return ModelView(np.lexsort((ids, -scores)), total, positive, index)


def weighted_ap(view: ModelView, multiplicities: np.ndarray) -> np.ndarray:
    weights: np.ndarray = multiplicities.astype(np.float64)
    totals = np.asarray(weights @ view.total)[:, ::-1]
    positives = np.asarray(weights @ view.positive)[:, ::-1]
    cumulative_n = np.cumsum(totals, axis=1)
    cumulative_f = np.cumsum(positives, axis=1)
    precision = np.divide(cumulative_f, cumulative_n, out=np.zeros_like(cumulative_f),
                          where=cumulative_n > 0)
    f = positives.sum(axis=1)
    return cast(np.ndarray, np.divide((positives * precision).sum(axis=1), f,
                                     out=np.full(len(f), np.nan), where=f > 0))


def weighted_tp(view: ModelView, labels: np.ndarray, positions: np.ndarray,
                multiplicities: np.ndarray, alerts: np.ndarray) -> np.ndarray:
    """Exact expanded-row selection, extending the prefix until every draw fits."""
    limit = min(len(labels), max(32, 4 * int(alerts.max())))
    while True:
        order = view.order[:limit]
        weights: np.ndarray = multiplicities[:, positions[order]].astype(np.int64)
        cumulative = np.cumsum(weights, axis=1)
        if bool((cumulative[:, -1] >= alerts).all()):
            break
        require(limit < len(labels), "Capacity exceeds sampled population")
        limit = min(len(labels), limit * 2)
    boundary = np.argmax(cumulative >= alerts[:, None], axis=1)
    row = np.arange(len(multiplicities))
    before_index = np.maximum(boundary - 1, 0)
    before_n = np.where(boundary > 0, cumulative[row, before_index], 0)
    cumulative_f = np.cumsum(weights * labels[order], axis=1)
    before_f = np.where(boundary > 0, cumulative_f[row, before_index], 0)
    return cast(np.ndarray, np.asarray(before_f + (alerts - before_n) * labels[order[boundary]],
                                      dtype=np.int64))


def stage_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    unique, counts = np.unique(scores, return_counts=True)
    return {"ap": float(average_precision_score(labels, scores)), "unique_scores": len(unique),
            "tied_rows": int(counts[counts > 1].sum()),
            "low_saturated": int((scores <= 1e-6).sum()),
            "high_saturated": int((scores >= 1 - 1e-6).sum())}
