"""Independent small-data checks for pairing, tied decisions and weighted metrics."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraudx.q2_extension.common import assert_aligned
from fraudx.q2_extension.kernels import (
    capacity_profile,
    model_view,
    priority_ranks,
    priority_tp,
    weighted_ap,
    weighted_tp,
)


def test_tie_bounds_against_all_subsets() -> None:
    ids = np.arange(7)
    y = np.array([1, 0, 1, 0, 0, 1, 0])
    scores = np.array([.9, .6, .6, .6, .6, .2, .1])
    profile, members = capacity_profile(ids, y, scores, 3 / 7)
    possibilities = [1 + int(y[list(c)].sum()) for c in combinations(members, 2)]
    assert profile["tp_min"] == min(possibilities)
    assert profile["tp_max"] == max(possibilities)
    assert profile["tp_expectation"] == np.mean(possibilities)
    ranks = priority_ranks(ids, "test", 1, 0)
    chosen = np.lexsort((ranks, -scores))[:3]
    assert priority_tp(profile, members, y, ranks) == int(y[chosen].sum())


@pytest.mark.parametrize("q", [.03, .4, 1.0])
def test_bootstrap_metrics_equal_expanded_rows(q: float) -> None:
    ids = np.arange(9)
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0])
    scores = np.array([.8, .8, .2, .4, .4, .3, .2, .4, .1])
    positions = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3])
    mult = np.array([[1, 1, 1, 1], [0, 2, 1, 1], [0, 0, 4, 0]], dtype=np.int16)
    view = model_view(ids, y, scores, positions, 4)
    n = mult @ np.bincount(positions, minlength=4)
    alerts = np.ceil(q * n).astype(np.int64)
    ap, tp = weighted_ap(view, mult), weighted_tp(view, y, positions, mult, alerts)
    for i, weights in enumerate(mult):
        row_weights = weights[positions]
        expanded = np.repeat(ids, row_weights)
        order = np.lexsort((np.arange(len(expanded)), ids[expanded], -scores[expanded]))
        assert tp[i] == int(y[expanded[order[:alerts[i]]]].sum())
        if y[expanded].sum():
            assert ap[i] == pytest.approx(average_precision_score(y[expanded], scores[expanded]))
            assert ap[i] == pytest.approx(average_precision_score(y, scores,
                                                                  sample_weight=row_weights))
        else:
            assert np.isnan(ap[i])


def test_alignment_rejects_changed_labels_and_dropped_rows() -> None:
    a = pd.DataFrame({"id": [1, 2], "time": [3, 3], "y": [0, 1]})
    with pytest.raises(ValueError, match="sample sizes"):
        assert_aligned(a, a.iloc[:1], ["id", "time", "y"])
    with pytest.raises(ValueError, match="y"):
        assert_aligned(a, a.assign(y=[1, 0]), ["id", "time", "y"])


def test_priority_has_shared_reproducible_order_and_full_tie_degenerates() -> None:
    ids, y = np.arange(8), np.array([0, 0, 0, 1, 0, 0, 1, 0])
    ranks = priority_ranks(ids, "paysim", 1, 3)
    np.testing.assert_array_equal(ranks, priority_ranks(ids, "paysim", 1, 3))
    assert len(np.unique(ranks)) == len(ids)
    result, members = capacity_profile(ids, y, np.ones(8), 1.0)
    assert result["tp_min"] == result["tp_max"] == result["tp_expectation"] == 2
    assert priority_tp(result, members, y, ranks) == 2
