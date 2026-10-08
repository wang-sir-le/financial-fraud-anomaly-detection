from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from banksim.common import HISTORY, WINDOWS, ProtocolError
from banksim.data import assign_splits, build_history, parse_cents
from banksim.preprocess import MISSING, Preprocessor, check_common_rules, fit_cells


def fixture_frame() -> pd.DataFrame:
    # Rows 0/1 share time and merchant, and cannot see one another.
    frame = pd.DataFrame({"source_row_id": [0, 1, 2, 3, 4, 5, 6, 7], "step": [0, 0, 1, 2, 8, 31, 1, 2], "merchant": ["M", "M", "M", "M", "M", "M", "N", "N"], "customer": ["a", "b", "a", "c", "a", "d", "x", "x"], "amount_cents": [100, 300, 500, 0, 800, 1000, 0, 200]})
    frame["amount"] = frame["amount_cents"] / 100
    for column in ("category", "age", "gender", "zipcodeOri", "zipMerchant"):
        frame[column] = ["a", "b"] * 4
    frame["split"] = "train"
    return frame


def brute_force(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        output = {"source_row_id": row.source_row_id}
        past = frame[(frame.merchant == row.merchant) & (frame.step < row.step)]
        for w in WINDOWS:
            history = past[past.step >= row.step - w]
            cents = [int(v) for v in history.amount_cents]
            n = len(cents)
            mean = sum(cents) / (100 * n) if n else 0
            std = math.sqrt((n * sum(c * c for c in cents) - sum(cents)**2) / (n * (n - 1))) / 100 if n > 1 else 0
            stats = [n, history.customer.nunique(), mean, max(cents) / 100 if n else 0, std, row.amount / mean if mean else 0]
            output.update(dict(zip([f"h{w}_{s}" for s in ("count", "customers", "mean", "max", "std", "ratio")], stats, strict=True)))
        output["h_available"] = int(not past.empty)
        output["h_recency"] = row.step - past.step.max() if not past.empty else -1
        rows.append(output)
    return pd.DataFrame(rows).sort_values("source_row_id").set_index("source_row_id")


def test_history_matches_direct_definition_and_hand_values() -> None:
    source = fixture_frame()
    actual = build_history(source).set_index("source_row_id").sort_index()
    expected = brute_force(source)
    np.testing.assert_array_equal(actual[HISTORY], expected[HISTORY])
    assert actual.loc[0, "h1_count"] == actual.loc[1, "h1_count"] == 0
    assert actual.loc[2, "h1_mean"] == 2
    assert actual.loc[2, "h1_std"] == math.sqrt(2)
    assert actual.loc[2, "h1_ratio"] == 2.5
    assert actual.loc[7, "h1_mean"] == actual.loc[7, "h1_ratio"] == 0
    assert actual.loc[5, "h1_count"] == 0 and actual.loc[5, "h_recency"] == 23


def test_order_invariance_duplicates_and_future_mutation() -> None:
    original = fixture_frame()
    duplicate = original.iloc[[0]].copy()
    duplicate["source_row_id"] = 8
    source = pd.concat([original, duplicate], ignore_index=True)
    baseline = build_history(source).set_index("source_row_id").sort_index()
    reversed_result = build_history(source.iloc[::-1]).set_index("source_row_id").sort_index()
    pd.testing.assert_frame_equal(baseline, reversed_result)
    assert baseline.loc[2, "h1_count"] == 3
    changed = source.copy()
    changed.loc[changed.step > 2, "amount_cents"] = 999999999
    changed["amount"] = changed.amount_cents / 100
    altered = build_history(changed).set_index("source_row_id").sort_index()
    pd.testing.assert_frame_equal(baseline.loc[baseline.step <= 2, HISTORY], altered.loc[altered.step <= 2, HISTORY])


@pytest.mark.parametrize("raw,expected", [("1.23", 123), ("-1.20", -120), ("0", 0), ("1.2300", 123), ("9e2", 90000)])
def test_decimal_exactness(raw: str, expected: int) -> None:
    assert parse_cents(raw) == expected


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "0.001", "1.00000000000000000000000000001", "99999999999999999999", "invalid"])
def test_invalid_money_stops(raw: str) -> None:
    with pytest.raises((ProtocolError, ValueError)):
        parse_cents(raw)


def test_labels_and_duplicate_source_ids_rejected() -> None:
    frame = fixture_frame()
    with pytest.raises(ProtocolError):
        build_history(frame.assign(fraud=0))
    frame.loc[1, "source_row_id"] = 0
    with pytest.raises(ProtocolError):
        build_history(frame)


def test_complete_step_split_boundaries() -> None:
    frame = pd.DataFrame({"step": [0, 107, 107, 108, 125, 126, 179]})
    got = assign_splits(frame, production=False)
    assert list(got.split) == ["train", "train", "train", "validation", "validation", "test", "test"]
    with pytest.raises(ProtocolError):
        assign_splits(pd.DataFrame({"step": [180]}), production=False)


def test_train_only_median_vocab_and_constant_outputs() -> None:
    train = pd.DataFrame({"split": ["train"] * 3, "amount": [1., np.nan, 3.], "category": ["a", "b", "a"], "age": ["x"] * 3})
    proc = Preprocessor(["amount", "category", "age"]).fit(train)
    before = proc.to_dict()
    test = pd.DataFrame({"split": ["test", "test"], "amount": [np.nan, 1000.], "category": ["unseen", "a"], "age": ["new", "x"]})
    result = proc.transform(test)
    assert before == proc.to_dict()
    assert proc.feature_names() == ["num:amount", "missing:amount", "cat:category=a", "cat:category=b"]
    np.testing.assert_array_equal(result[0], [2, 1, 0, 0])
    np.testing.assert_array_equal(result[1], [1000, 0, 1, 0])
    np.testing.assert_array_equal(result, Preprocessor.from_dict(before).transform(test))
    with pytest.raises(ProtocolError):
        Preprocessor(["amount"]).fit(test)


def test_all_empty_numeric_and_category_collision() -> None:
    train = pd.DataFrame({"split": ["train", "train"], "amount": [np.nan, np.nan], "category": ["", "a"]})
    proc = Preprocessor(["amount", "category"]).fit(train)
    assert proc.rules[0]["median"] == 0 and not proc.rules[0]["keep_value"]
    assert len(proc.feature_names()) == 2
    with pytest.raises(ProtocolError):
        proc.transform(train.assign(category=MISSING))


def test_common_columns_identical_across_cells() -> None:
    frame = build_history(fixture_frame())
    processors = fit_cells(frame)
    check_common_rules(processors)
    columns = {}
    for proc in processors.values():
        values = proc.transform(frame)
        for j, name in enumerate(proc.feature_names()):
            if name in columns:
                np.testing.assert_array_equal(columns[name], values[:, j])
            columns[name] = values[:, j]
