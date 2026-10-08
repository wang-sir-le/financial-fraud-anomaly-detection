"""Production regressions retained from the September exact-threshold audit."""
from __future__ import annotations

import numpy as np
import pytest

from fraudx.metrics import optimize_threshold


def test_exact_cost_selection_keeps_no_alert_optimum() -> None:
    labels = np.zeros(12000, dtype=int)
    labels[0] = 1
    labels[1100:1199] = 1
    scores = np.full(12000, 0.1)
    scores[:1002] = 0.9
    threshold, report = optimize_threshold(labels, scores, 1000, maximum_alert_rate=0.1)
    assert threshold > scores.max()
    assert report.business_cost == 100000


@pytest.mark.parametrize("seed", [1, 15, 82])
def test_matches_independent_exhaustive_integer_objective(seed: int) -> None:
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, 0.15, 250)
    scores = np.round(rng.random(250), 2)
    q, cost = 0.4, 1000
    feasible = []
    for threshold in np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)]:
        decisions = scores >= threshold
        alerts, tp = int(decisions.sum()), int(labels[decisions].sum())
        if alerts / len(labels) <= q:
            objective = cost * (int(labels.sum()) - tp) + alerts - tp
            feasible.append((objective, -tp, alerts, -threshold, threshold))
    expected = min(feasible)[-1]
    actual, _ = optimize_threshold(labels, scores, cost, maximum_alert_rate=q)
    assert actual == expected
