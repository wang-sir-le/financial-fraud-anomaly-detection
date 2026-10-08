from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from banksim.common import CELLS, FAMILIES, ProtocolError
from banksim.metrics import (
    ScoreOrder,
    bootstrap_orders,
    capacity,
    interval,
    moving_blocks,
    multiplicities,
    paired_summary,
    priority,
)


def test_capacity_integer_arithmetic() -> None:
    assert capacity(198311, .03) == 5950
    assert capacity(63740, .03) == 1913
    assert capacity(100, .03) == 3
    assert capacity(1, .03) == 1
    assert capacity(100, .01) == 1
    assert capacity(100, 1) == 100


def test_hand_calculated_ap_and_boundary_ties() -> None:
    y = np.array([1, 0, 1, 0])
    scores = np.array([3., 2., 2., 1.])
    ids = np.arange(4)
    got = ScoreOrder(y, scores, ids).evaluate(fractions=(.5,))
    assert got["ap"] == pytest.approx(5 / 6)
    assert got["ap"] == pytest.approx(average_precision_score(y, scores))
    assert got["tie_n@0.50"] == 2
    assert got["tie_tp_min@0.50"] == 1
    assert got["tie_tp_max@0.50"] == 2
    manual_order = np.lexsort((ids, priority(ids), -scores))
    assert got["tp@0.50"] == y[manual_order[:2]].sum()
    reversed_got = ScoreOrder(y[::-1], scores[::-1], ids[::-1]).evaluate(fractions=(.5,))
    assert got == reversed_got


def test_priority_is_label_independent_and_constant_scores_retained() -> None:
    ids = np.arange(6)
    y = np.array([1, 0, 0, 1, 0, 0])
    a = ScoreOrder(y, np.ones(6), ids)
    b = ScoreOrder(1 - y, np.ones(6), ids)
    np.testing.assert_array_equal(a.order, b.order)
    got = a.evaluate(fractions=(.5,))
    assert got["constant_scores"]
    assert got["ap"] == pytest.approx(2 / 6)
    assert got["tie_tp_min@0.50"] == 0 and got["tie_tp_max@0.50"] == 2


def literal(y: np.ndarray, scores: np.ndarray, ids: np.ndarray, weight: np.ndarray, q: float) -> tuple[float, int]:
    expanded = np.repeat(np.arange(len(y)), weight)
    copy_order = np.arange(len(expanded))
    order = np.lexsort((copy_order, ids[expanded], priority(ids[expanded]), -scores[expanded]))
    tp = int(y[expanded][order[:capacity(len(expanded), q)]].sum())
    ap = float(average_precision_score(y[expanded], scores[expanded])) if y[expanded].sum() else float("nan")
    return ap, tp


@pytest.mark.parametrize("weight", [np.array([2, 0, 1, 3, 1, 0]), np.array([0, 2, 2, 0, 0, 1]), np.array([1, 1, 1, 1, 1, 1])])
def test_optimized_metrics_equal_literal_row_replication(weight: np.ndarray) -> None:
    y = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([.9, .8, .8, .8, .2, .1])
    ids = np.array([8, 3, 9, 2, 7, 5])
    for q in (.03, .4, .5, 1.):
        got = ScoreOrder(y, scores, ids).evaluate(weight, fractions=(q,))
        ap, tp = literal(y, scores, ids, weight, q)
        assert got["ap"] == pytest.approx(ap)
        assert got[f"tp@{q:.2f}"] == tp
        assert got[f"k@{q:.2f}"] == capacity(int(weight.sum()), q)


def test_resample_changes_capacity_and_whole_steps_stay_together() -> None:
    steps = np.array([0, 1, 1, 2, 2, 2, 2])
    weights = multiplicities(steps, np.array([0, 0, 2]), np.arange(3))
    np.testing.assert_array_equal(weights, [2, 0, 0, 1, 1, 1, 1])
    order = ScoreOrder(np.array([1, 0, 1, 0, 1, 0, 0]), np.arange(7.), np.arange(7))
    assert order.evaluate(fractions=(.5,))["k@0.50"] == 4
    assert order.evaluate(weights, fractions=(.5,))["k@0.50"] == 3


def test_no_positives_invalid_weights_and_nonfinite_scores() -> None:
    order = ScoreOrder(np.zeros(4), np.ones(4), np.arange(4))
    result = order.evaluate()
    assert result["tp@0.03"] == 0 and result["precision@0.03"] == 0
    assert math.isnan(result["ap"]) and math.isnan(result["recall@0.03"])
    for bad in (np.array([1, -1, 1, 1]), np.array([1, .5, 1, 1]), np.zeros(4)):
        with pytest.raises(ProtocolError):
            order.evaluate(bad)
    with pytest.raises(ProtocolError):
        ScoreOrder(np.zeros(2), np.array([np.nan, 1]), np.arange(2))


def test_seed_mean_is_not_score_ensemble_and_missing_seed_propagates() -> None:
    y, ids = np.array([1, 0, 0]), np.arange(3)
    score_a, score_b = np.array([1., .9, 0]), np.array([0., .9, 1.])
    rows = {}
    for family in FAMILIES:
        for cell in CELLS:
            for seed, scores in ((42, score_a), (52, score_b)):
                rows[f"{family}|{cell}|{seed}"] = ScoreOrder(y, scores, ids).evaluate()
    summary = paired_summary(rows, (42, 52))
    assert summary["LightGBM/B0/tp@0.03"] == .5
    assert ScoreOrder(y, (score_a + score_b) / 2, ids).evaluate()["tp@0.03"] == 0
    rows["LightGBM|B0H|42"]["tp@0.03"] = 3
    rows["LightGBM|B1H|42"]["tp@0.03"] = 2
    summary = paired_summary(rows, (42, 52))
    assert summary["LightGBM/H0/tp@0.03"] == 1
    assert summary["LightGBM/H1/tp@0.03"] == .5
    assert summary["LightGBM/D/tp@0.03"] == -.5
    del rows["LightGBM|B0H|52"]
    assert math.isnan(paired_summary(rows, (42, 52))["LightGBM/H0/tp@0.03"])


def test_draws_are_replayable_and_intervals_do_not_replenish() -> None:
    a = moving_blocks(np.arange(54), 7, 20, 20260925)
    np.testing.assert_array_equal(a, moving_blocks(np.arange(54), 7, 20, 20260925))
    assert a.shape == (20, 54) and a.min() >= 0 and a.max() < 54
    for start in range(0, 49, 7):
        assert np.all(np.diff(a[:, start:start+7], axis=1) == 1)
    got = interval(np.array([1, 2, np.nan]), min_valid=3)
    assert got["status"] == "insufficient_valid_draws" and got["invalid"] == 1
    assert got["lower"] is None
    got = interval(np.array([-2, -1, 0, 1, 2]), min_valid=5)
    assert got["lower"] < 0 < got["upper"]


def test_small_paired_bootstrap_matches_direct_replay(tmp_path: Path) -> None:
    ids = np.arange(14)
    steps = np.repeat(np.arange(7), 2)
    y = np.tile([1, 0], 7)
    arrays = {f"{family}|{cell}|{seed}": np.sin(ids + c + seed) for family in FAMILIES for c, cell in enumerate(CELLS) for seed in (42, 52)}
    orders = {k: ScoreOrder(y, scores, ids) for k, scores in arrays.items()}
    result = bootstrap_orders(orders, steps, tmp_path, draws=12, seeds=(42, 52), lengths=(3,))
    assert result["status"] == "complete" and len(result["multiplicity_family"]) == 4
    draws = np.load(tmp_path / "step_sequences_L3.npy", allow_pickle=False)
    saved = np.load(tmp_path / "draw_effects_L3.npy", allow_pickle=False)
    import json
    names = json.loads((tmp_path / "estimand_names.json").read_text())
    for i, seq in enumerate(draws):
        weights = multiplicities(steps, seq, np.arange(7))
        rows = {}
        for key, scores in arrays.items():
            ap, tp = literal(y, scores, ids, weights, .03)
            rows[key] = {"ap": ap, "tp@0.03": tp}
        expected = paired_summary(rows, (42, 52))
        for name in ("LightGBM/H1/tp@0.03", "LightGBM/D/tp@0.03", "XGBoost-minus-LightGBM/H0/ap"):
            assert saved[i, names.index(name)] == pytest.approx(expected[name])
