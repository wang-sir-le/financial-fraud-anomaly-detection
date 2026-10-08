"""Scientific boundary tests for the new LR preprocessing and paired analysis."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraudx.cross_model.analysis import summarize
from fraudx.cross_model.replay import replay
from fraudx.cross_model.statistics import (
    calculate,
    components,
    paired_difference,
    positive_level_view,
)
from fraudx.cross_model.training import FrozenPreprocessor
from fraudx.q2_extension.common import read_json, write_json


def test_train_only_missing_and_unknown() -> None:
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0], "empty": [np.nan] * 3, "kind": ["A", "B", None]})
    prep = FrozenPreprocessor(list(train), ["kind"]).fit(train)
    before = prep.audit()
    future = pd.DataFrame(
        {"a": [100000.0, np.nan], "empty": [4.0, np.nan], "kind": ["unseen", None]}
    )
    result = prep.transform(future)
    assert result.format == "csr"
    assert np.isfinite(result.data).all()
    assert prep.audit() == before
    assert before["medians"] == [2.0, 0.0]
    assert before["missing_indicator_indices"] == [0, 1]
    category = result.toarray()[:, -3:]
    assert category[0].sum() == 0  # No future vocabulary fitting.
    assert category[1].sum() == 1  # Reserved missing level remains distinct.


def test_category_missing_token_cannot_collide() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "kind": ["missing:", None]})
    prep = FrozenPreprocessor(list(frame), ["kind"]).fit(frame)
    categories = prep.transform(frame).toarray()[:, -2:]
    assert not np.array_equal(categories[0], categories[1])


def test_positive_levels_match_expanded_bootstrap_with_ties() -> None:
    rng = np.random.default_rng(704)
    ids = np.arange(80)[::-1]
    labels = (rng.random(80) < 0.2).astype(np.int8)
    scores = rng.integers(0, 12, 80) / 12
    positions = rng.integers(0, 6, 80)
    multiplicities = rng.integers(0, 5, size=(30, 6), dtype=np.int16)
    view = positive_level_view(ids, labels, scores, positions, 6)
    result = calculate(view, positions, labels, multiplicities)
    for draw, row in enumerate(multiplicities):
        expanded = np.repeat(np.arange(80), row[positions])
        y, s, identity = labels[expanded], scores[expanded], ids[expanded]
        order = np.lexsort((identity, -s))
        k = int(np.ceil(0.03 * len(y)))
        assert result.iloc[draw].tp == y[order[:k]].sum()
        assert result.iloc[draw].ap == pytest.approx(average_precision_score(y, s), abs=1e-14)


def test_effect_difference_preserves_pairing_and_invalid_draws() -> None:
    baseline = pd.DataFrame(
        {
            "replicate_id": [1, 2],
            "n": [100, 100],
            "frauds": [2, 0],
            "alerts": [3, 3],
            "tp": [0, 0],
            "ap": [0.1, np.nan],
        }
    )
    candidate = baseline.copy()
    candidate.loc[0, ["tp", "ap"]] = [1, 0.2]
    delta = components(baseline, candidate)
    assert delta.delta_recall.iloc[0] == 0.5
    assert np.isnan(delta.delta_recall.iloc[1])
    old = [np.array([x, np.nan]) for x in [1.0, 2.0, 3.0, 4.0, 5.0]]
    actual = paired_difference(np.array([4.0, np.nan]), old)
    assert actual[0] == 1
    assert np.isnan(actual[1])
    candidate.loc[1, "replicate_id"] = 3
    with pytest.raises(ValueError, match="Unpaired"):
        components(baseline, candidate)


def test_failed_models_cannot_create_or_hide_inference(tmp_path) -> None:
    folder = tmp_path / "analysis/ieee_cis"
    folder.mkdir(parents=True)
    pd.DataFrame(columns=["family", "seed", "fold", "model", "score_stage", "q"]).to_csv(
        folder / "model_points.csv", index=False
    )
    protocol = tmp_path / "protocol.json"
    write_json(
        protocol,
        {
            "output_root": str(tmp_path),
            "datasets": {
                "ieee_cis": {
                    "models": {"M0": {}, "M1": {}},
                    "stages": ["raw_probability"],
                    "core_contrasts": [{"id": "I_HISTORY", "baseline": "M0", "candidate": "M1"}],
                    "auxiliary_contrasts": [],
                }
            },
        },
    )
    for fold in (1, 2, 3):
        for model in ("M0", "M1"):
            write_json(
                tmp_path / "training/ieee_cis" / f"ieee_cis_{fold}_{model}/COMPLETED.json",
                {"converged": False},
            )
    summarize(protocol, "ieee_cis")
    assert pd.read_csv(folder / "effect_intervals.csv").empty
    assert not pd.read_csv(folder / "contrast_coverage.csv").estimable.any()
    replay(tmp_path / "analysis", tmp_path / "replay", ("ieee_cis",))
    assert read_json(tmp_path / "replay/REPLAY_REPORT.json")["total_interval_rows"] == 0
    write_json(tmp_path / "training/ieee_cis/ieee_cis_1_M0/COMPLETED.json", {"converged": True})
    with pytest.raises(ValueError, match="lacks bootstrap"):
        summarize(protocol, "ieee_cis")
