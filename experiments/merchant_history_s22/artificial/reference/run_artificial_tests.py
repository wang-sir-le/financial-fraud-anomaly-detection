"""Run artificial-only six-cell contract acceptance; no real inputs are accepted."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
import time
import traceback
import unittest
from dataclasses import asdict, replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parent
RUN_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from contract_kernel import (  # noqa: E402
    CELLS,
    HISTORY,
    MISSING,
    STATS,
    WINDOWS,
    X_OFF,
    X_ON,
    ArtificialPreprocessor,
    ContractError,
    Row,
    artificial_contrasts,
    artificial_metrics,
    capacity,
    cell_frame,
    float32,
    history_stream,
    input_columns,
    parse_cents,
    priority,
    rows_from_raw,
    split_for,
)


def one(identity: int, step: int, cents: int = 100, merchant: str = "synthetic-m0",
        customer: str = "synthetic-c0", **kwargs: Any) -> Row:
    return Row(identity, step, merchant, customer, cents, **kwargs)


def fixture() -> list[Row]:
    rows: list[Row] = []
    for step in (0, 1, 2, 7, 8, 9, 29, 30, 31, 40, 80, 100, 107, 108, 109, 125, 126, 127, 160, 179):
        for merchant_id in range(2):
            i = len(rows)
            rows.append(one(i, step, (1 + i % 7) * 101,
                            merchant=f"synthetic-m{merchant_id}",
                            customer=f"synthetic-c{i % 3}",
                            category=f"synthetic-cat{i % 2}",
                            age=str(i % 3), gender="F" if i % 2 else "M"))
        if step in (0, 7, 108, 126):
            prior = rows[-2]
            rows.append(replace(prior, source_row_id=len(rows)))
    rows.append(one(len(rows), 108, 808, merchant="synthetic-unseen",
                    category="synthetic-unseen-category", age=None))
    return rows


def batch_oracle(rows: list[Row], mode: str) -> dict[int, dict[str, float]]:
    """Independent explicit-set oracle and centered rational variance."""
    if mode == "H0":
        return {r.source_row_id: {} for r in rows}
    output = {}
    for current in rows:
        allowed = [r for r in rows if r.merchant == current.merchant
                   and r.step < current.step
                   and (mode != "HS" or r.step >= max(0, current.step - 7))]
        values: dict[str, float] = {}
        for window in WINDOWS:
            past = [r for r in allowed if r.step >= current.step - window]
            n = len(past)
            exact_mean = Fraction(sum(r.cents for r in past), n) if n else Fraction(0)
            variance = (sum((Fraction(r.cents) - exact_mean) ** 2 for r in past)
                        / (n - 1)) if n > 1 else Fraction(0)
            numbers = (
                n, len({r.customer for r in past}), float(exact_mean) / 100,
                max((r.cents for r in past), default=0) / 100,
                math.sqrt(float(variance)) / 100,
                float(Fraction(current.cents) / exact_mean) if exact_mean else 0.0,
            )
            values.update({f"h{window}_{stat}": value
                           for stat, value in zip(STATS, numbers, strict=True)})
        values["h_recency"] = current.step - max(r.step for r in allowed) if allowed else -1
        values["h_available"] = int(bool(allowed))
        output[current.source_row_id] = values
    return output


def prepared(rows: list[Row], cell: str) -> tuple[list[dict[str, Any]], ArtificialPreprocessor]:
    values, _ = history_stream(rows, CELLS[cell][1])
    frame = cell_frame(rows, cell, values)
    processor = ArtificialPreprocessor(input_columns(cell))
    processor.fit([r for r in frame if r["split"] == "train"])
    return frame, processor


class AuditGuard:
    def __init__(self) -> None:
        self.denials: list[dict[str, str]] = []
        self.opens: list[dict[str, str]] = []

    def __call__(self, event: str, args: tuple[Any, ...]) -> None:
        reason = None
        if event == "open" and isinstance(args[0], (str, bytes)):
            raw_path = args[0].decode(sys.getfilesystemencoding()) if isinstance(args[0], bytes) else args[0]
            path = Path(raw_path).resolve()
            if not path.is_relative_to(RUN_ROOT):
                reason = "file access outside independent preparation directory"
            else:
                self.opens.append({"path": str(path), "mode": str(args[1])})
        elif event == "import":
            root = str(args[0]).split(".")[0]
            if root not in sys.stdlib_module_names and root != "contract_kernel":
                reason = "non-stdlib import prohibited"
        elif event.startswith(("socket.", "subprocess.", "os.spawn", "os.system")):
            reason = "network or process creation prohibited"
        if reason:
            self.denials.append({"event": event, "target": str(args[0]), "reason": reason})
            raise PermissionError(reason)


GUARD = AuditGuard()


class InputContractTests(unittest.TestCase):
    def test_six_cells_only(self) -> None:
        """Exactly six IDs; presentation names do not reuse old B0 input semantics."""
        self.assertEqual(tuple(CELLS), ("A0", "B0", "A1", "B1", "A2", "B2"))
        self.assertEqual({v[1] for v in CELLS.values()}, {"H0", "HS", "HL"})

    def test_direct_merchant_is_only_raw_column_change(self) -> None:
        self.assertEqual(tuple(c for c in X_ON if c != "merchant"), X_OFF)
        self.assertEqual(len(X_ON) - len(X_OFF), 1)
        self.assertTrue({"category", "zipMerchant", "age", "gender"} <= set(X_OFF))

    def test_history_schema_twenty(self) -> None:
        self.assertEqual(len(HISTORY), 20)
        self.assertEqual(len(set(HISTORY)), 20)
        self.assertEqual(input_columns("A0"), X_OFF)
        self.assertEqual(input_columns("B2"), X_ON + HISTORY)

    def test_forbidden_predictors_rejected(self) -> None:
        for column in ("fraud", "customer", "source_row_id", "merchant_risk", "old_score"):
            with self.subTest(column=column), self.assertRaises(ContractError):
                ArtificialPreprocessor((column,))

    def test_labels_rejected_before_history(self) -> None:
        for label_column in ("fraud", "label"):
            raw = {"step": "0", "merchant": "synthetic-m", "customer": "synthetic-c",
                   "amount": "1.00", label_column: 1}
            with self.subTest(column=label_column), self.assertRaises(ContractError):
                rows_from_raw([raw])

    def test_unknown_raw_field_rejected(self) -> None:
        with self.assertRaises(ContractError):
            rows_from_raw([{"step": "0", "merchant": "m", "customer": "c",
                            "amount": "1.00", "old_state": 3}])

    def test_required_key_missing_rejected(self) -> None:
        for field in ("merchant", "customer", "amount", "step"):
            raw = {"step": "0", "merchant": "m", "customer": "c", "amount": "1.00"}
            raw[field] = ""
            with self.subTest(field=field), self.assertRaises(ContractError):
                rows_from_raw([raw])

    def test_step_text_and_range_rejected(self) -> None:
        for bad in ("-1", "1.0", " 1", "180", "１"):
            with self.subTest(step=bad), self.assertRaises(ContractError):
                rows_from_raw([{"step": bad, "merchant": "m", "customer": "c", "amount": "1"}])

    def test_exact_cent_parse(self) -> None:
        self.assertEqual(parse_cents("0.29"), 29)
        self.assertEqual(parse_cents("-1.01"), -101)
        self.assertEqual(parse_cents("1e2"), 10000)

    def test_bad_amount_rejected(self) -> None:
        for value in ("0.001", "NaN", "Infinity", "not-decimal", "92233720368547758.08"):
            with self.subTest(value=value), self.assertRaises(ContractError):
                parse_cents(value)

    def test_duplicate_identity_rejected(self) -> None:
        with self.assertRaises(ContractError):
            history_stream([one(0, 0), one(0, 1)], "HL")

    def test_duplicate_transactions_remain_distinct_rows(self) -> None:
        rows = [one(0, 0), one(1, 0), one(2, 1)]
        history, _ = history_stream(rows, "HL")
        self.assertEqual(history[2]["h1_count"], 2)
        self.assertEqual(len(cell_frame(rows, "A1", history)), 3)

    def test_split_boundaries(self) -> None:
        self.assertEqual([split_for(t) for t in (0, 107, 108, 125, 126, 179)],
                         ["train", "train", "validation", "validation", "review", "review"])

    def test_wrong_history_schema_rejected(self) -> None:
        with self.assertRaises(ContractError):
            cell_frame([one(0, 0)], "A1", {0: {}})


class HistoryTests(unittest.TestCase):
    def test_cold_start_empty_values(self) -> None:
        for mode in ("HS", "HL"):
            values, _ = history_stream([one(0, 0)], mode)
            self.assertEqual(values[0], {c: (-1 if c == "h_recency" else 0) for c in HISTORY})

    def test_h0_allocates_no_state_or_history_columns(self) -> None:
        values, audit = history_stream(fixture(), "H0")
        self.assertEqual(audit["state_allocations"], 0)
        self.assertTrue(all(not v for v in values.values()))

    def test_single_past_row(self) -> None:
        values, _ = history_stream([one(0, 0, 123), one(1, 1, 246)], "HL")
        self.assertEqual([values[1][f"h1_{s}"] for s in STATS], [1, 1, 1.23, 1.23, 0, 2])

    def test_sample_std_and_distinct_customers(self) -> None:
        rows = [one(0, 0, 100, customer="c"), one(1, 0, 300, customer="c"), one(2, 1)]
        values, _ = history_stream(rows, "HL")
        self.assertEqual(values[2]["h1_count"], 2)
        self.assertEqual(values[2]["h1_customers"], 1)
        self.assertAlmostEqual(values[2]["h1_std"], math.sqrt(2))
        self.assertEqual(values[2]["h1_mean"], 2)

    def test_zero_history_mean_ratio(self) -> None:
        values, _ = history_stream([one(0, 0, 0), one(1, 1, 100)], "HL")
        self.assertEqual(values[1]["h1_ratio"], 0)

    def test_negative_amount_max_and_exact_variance(self) -> None:
        values, _ = history_stream([one(0, 0, -100), one(1, 0, -300), one(2, 1)], "HL")
        self.assertEqual(values[2]["h1_max"], -1)
        self.assertAlmostEqual(values[2]["h1_std"], math.sqrt(2))

    def test_large_cents_do_not_cancel_variance(self) -> None:
        base = 10**15
        values, _ = history_stream([one(0, 0, base), one(1, 0, base + 2), one(2, 1)], "HL")
        self.assertAlmostEqual(values[2]["h1_std"], math.sqrt(2) / 100)

    def test_seven_step_left_endpoint_included(self) -> None:
        values, _ = history_stream([one(0, 3), one(1, 10)], "HS")
        self.assertEqual(values[1]["h7_count"], 1)
        self.assertEqual(values[1]["h_recency"], 7)

    def test_eight_step_old_event_excluded_from_every_hs_feature(self) -> None:
        values, audit = history_stream([one(0, 2), one(1, 10)], "HS")
        self.assertEqual(values[1], {c: (-1 if c == "h_recency" else 0) for c in HISTORY})
        self.assertFalse(audit["all_past_last_state"])

    def test_one_step_left_endpoint(self) -> None:
        values, _ = history_stream([one(0, 8), one(1, 9), one(2, 10)], "HL")
        self.assertEqual(values[2]["h1_count"], 1)
        self.assertEqual(values[2]["h7_count"], 2)

    def test_thirty_step_left_endpoint(self) -> None:
        values, _ = history_stream([one(0, 9), one(1, 10), one(2, 40)], "HL")
        self.assertEqual(values[2]["h30_count"], 1)
        self.assertEqual(values[2]["h7_count"], 0)

    def test_hl_recency_reaches_beyond_thirty(self) -> None:
        values, _ = history_stream([one(0, 0), one(1, 40)], "HL")
        self.assertEqual(values[1]["h30_count"], 0)
        self.assertEqual(values[1]["h_recency"], 40)
        self.assertEqual(values[1]["h_available"], 1)

    def test_hs_seven_and_thirty_slots_match(self) -> None:
        values, _ = history_stream(fixture(), "HS")
        for value in values.values():
            self.assertEqual([value[f"h7_{s}"] for s in STATS],
                             [value[f"h30_{s}"] for s in STATS])

    def test_same_step_rows_never_see_each_other(self) -> None:
        values, _ = history_stream([one(0, 7), one(1, 7, 10000), one(2, 8)], "HL")
        self.assertEqual(values[0]["h7_count"], 0)
        self.assertEqual(values[1]["h7_count"], 0)
        self.assertEqual(values[2]["h1_count"], 2)

    def test_same_step_modification_does_not_change_other_current_row(self) -> None:
        original = [one(0, 0), one(1, 1), one(2, 1, 999)]
        changed = original[:2] + [replace(original[2], cents=10**8)]
        a, _ = history_stream(original, "HL")
        b, _ = history_stream(changed, "HL")
        self.assertEqual(a[1], b[1])

    def test_merchant_history_isolation_not_customer_binding(self) -> None:
        rows = [one(0, 0, merchant="m0"), one(1, 1, merchant="m1"),
                one(2, 1, merchant="m0", customer="other")]
        values, _ = history_stream(rows, "HL")
        self.assertEqual(values[1]["h7_count"], 0)
        self.assertEqual(values[2]["h7_count"], 1)

    def test_future_append_invariance(self) -> None:
        rows = [one(0, 0), one(1, 8)]
        for mode in ("HS", "HL"):
            before, _ = history_stream(rows, mode)
            after, _ = history_stream(rows + [one(2, 179, 999999)], mode)
            self.assertEqual(before, {i: after[i] for i in before})

    def test_row_permutation_invariance(self) -> None:
        rows = fixture()
        for mode in ("HS", "HL"):
            a, _ = history_stream(rows, mode)
            b, _ = history_stream(list(reversed(rows)), mode)
            self.assertEqual(a, b)

    def test_history_continues_across_split_boundaries(self) -> None:
        rows = [one(0, 107), one(1, 108), one(2, 125), one(3, 126)]
        values, _ = history_stream(rows, "HS")
        self.assertEqual(values[1]["h1_count"], 1)
        self.assertEqual(values[3]["h1_count"], 1)

    def test_empty_histories_not_filtered(self) -> None:
        rows = [one(0, 0, merchant="m0"), one(1, 100, merchant="m1")]
        for cell in CELLS:
            values, _ = history_stream(rows, CELLS[cell][1])
            self.assertEqual([r["source_row_id"] for r in cell_frame(rows, cell, values)], [0, 1])

    def test_inactive_merchant_state_expires_without_new_transaction(self) -> None:
        rows = [one(0, 0, merchant="inactive"), one(1, 8, merchant="active")]
        _, audit = history_stream(rows, "HS")
        self.assertEqual(audit["retained_state_after_step"]["8"]["inactive"],
                         {"1": [], "7": [], "30": []})

    def test_fixture_matches_independent_set_oracle(self) -> None:
        for mode in ("HS", "HL"):
            actual, _ = history_stream(fixture(), mode)
            expected = batch_oracle(fixture(), mode)
            for identity, values in expected.items():
                for feature, target in values.items():
                    self.assertTrue(math.isclose(actual[identity][feature], target,
                                                 rel_tol=1e-12, abs_tol=1e-12))

    def test_random_small_worlds_against_independent_oracle(self) -> None:
        for seed in range(24):
            with self.subTest(artificial_seed=seed):
                # Reproducible artificial fixture generation; no cryptographic use.
                rng = random.Random(2026100600 + seed)  # nosec B311
                rows = [one(i, rng.randrange(0, 50), rng.randrange(-500, 5000),
                            merchant=f"synthetic-m{rng.randrange(3)}",
                            customer=f"synthetic-c{rng.randrange(4)}") for i in range(45)]
                for mode in ("HS", "HL"):
                    actual, audit = history_stream(rows, mode)
                    expected = batch_oracle(rows, mode)
                    for identity, values in expected.items():
                        for name, target in values.items():
                            self.assertTrue(math.isclose(actual[identity][name], target,
                                                         rel_tol=1e-12, abs_tol=1e-12))
                    if mode == "HS":
                        by_id = {r.source_row_id: r for r in rows}
                        for identity, sources in audit["used_history"].items():
                            current = by_id[int(identity)]
                            used = [j for w in ("h1", "h7", "h30") for j in sources[w]]
                            if sources["recency_source"] is not None:
                                used.append(sources["recency_source"])
                            self.assertTrue(all(current.step - 7 <= by_id[j].step < current.step
                                                and by_id[j].merchant == current.merchant
                                                for j in used))
                        for step, entities in audit["retained_state_after_step"].items():
                            for windows in entities.values():
                                for window, identities in windows.items():
                                    self.assertTrue(all(int(step) - min(int(window), 7)
                                                        <= by_id[j].step <= int(step)
                                                        for j in identities))


class EncodingTests(unittest.TestCase):
    def test_only_training_rows_can_fit(self) -> None:
        for split in ("validation", "review"):
            with self.subTest(split=split), self.assertRaises(ContractError):
                ArtificialPreprocessor(("amount",)).fit([{"split": split, "amount": 1}])

    def test_empty_fit_rejected(self) -> None:
        with self.assertRaises(ContractError):
            ArtificialPreprocessor(("amount",)).fit([])

    def test_unseen_categories_all_zero(self) -> None:
        proc = ArtificialPreprocessor(("merchant",)).fit([
            {"split": "train", "merchant": "synthetic-a"},
            {"split": "train", "merchant": "synthetic-b"},
        ])
        self.assertEqual(proc.transform([{"merchant": "synthetic-new"}]), [[0, 0]])

    def test_future_categories_do_not_extend_vocabulary(self) -> None:
        frame, proc = prepared(fixture(), "B0")
        rule = next(r for r in proc.rules if r["column"] == "merchant")
        self.assertNotIn("synthetic-unseen", rule["vocabulary"])
        before = json.dumps(proc.rules, sort_keys=True)
        proc.transform(frame)
        self.assertEqual(json.dumps(proc.rules, sort_keys=True), before)

    def test_constant_training_dummy_removed_even_if_future_differs(self) -> None:
        proc = ArtificialPreprocessor(("merchant",)).fit([{"split": "train", "merchant": "x"}])
        self.assertEqual(proc.feature_names(), [])
        self.assertEqual(proc.transform([{"merchant": "new"}]), [[]])

    def test_missing_marker_collision_rejected_fit_and_transform(self) -> None:
        with self.assertRaises(ContractError):
            ArtificialPreprocessor(("age",)).fit([{"split": "train", "age": MISSING}])
        proc = ArtificialPreprocessor(("age",)).fit([{"split": "train", "age": "2"}])
        with self.assertRaises(ContractError):
            proc.transform([{"age": MISSING}])

    def test_strings_not_normalized(self) -> None:
        proc = ArtificialPreprocessor(("merchant",)).fit([
            {"split": "train", "merchant": "'synthetic-a'"},
            {"split": "train", "merchant": "synthetic-a"},
        ])
        self.assertEqual(len(proc.feature_names()), 2)
        self.assertEqual(proc.transform([{"merchant": "'synthetic-a'"}]), [[1, 0]])

    def test_missing_category_uses_train_token(self) -> None:
        proc = ArtificialPreprocessor(("age",)).fit([
            {"split": "train", "age": None}, {"split": "train", "age": "2"},
        ])
        self.assertIn(f"cat:age={MISSING}", proc.feature_names())
        self.assertEqual(proc.transform([{"age": ""}]), [[0, 1]])

    def test_numeric_train_median_and_missing_indicator(self) -> None:
        proc = ArtificialPreprocessor(("amount",)).fit([
            {"split": "train", "amount": 1}, {"split": "train", "amount": 3},
            {"split": "train", "amount": None},
        ])
        self.assertEqual(proc.rules[0]["median"], 2)
        self.assertEqual(proc.transform([{"amount": None}]), [[2, 1]])

    def test_all_missing_numeric_column_dropped(self) -> None:
        proc = ArtificialPreprocessor(("amount",)).fit([{"split": "train", "amount": None}])
        self.assertEqual(proc.rules[0]["median"], 0)
        self.assertEqual(proc.feature_names(), [])

    def test_numeric_constant_removed_without_evaluation_peeking(self) -> None:
        proc = ArtificialPreprocessor(("amount",)).fit([
            {"split": "train", "amount": 5}, {"split": "train", "amount": 5},
        ])
        self.assertEqual(proc.transform([{"amount": 999}]), [[]])

    def test_float32_constant_rule(self) -> None:
        proc = ArtificialPreprocessor(("amount",)).fit([
            {"split": "train", "amount": 1}, {"split": "train", "amount": 1 + 1e-9},
        ])
        self.assertEqual(proc.feature_names(), [])
        self.assertEqual(float32(1 + 1e-9), 1)

    def test_nonfinite_or_overflow_numeric_rejected(self) -> None:
        for amount in (math.inf, -math.inf, 1e99):
            with self.subTest(amount=amount), self.assertRaises(ContractError):
                ArtificialPreprocessor(("amount",)).fit([{"split": "train", "amount": amount}])

    def test_m_switch_changes_only_all_merchant_encoded_columns(self) -> None:
        rows = fixture()
        for off, on in (("A0", "B0"), ("A1", "B1"), ("A2", "B2")):
            fa, pa = prepared(rows, off)
            fb, pb = prepared(rows, on)
            a_names, b_names = pa.feature_names(), pb.feature_names()
            added = set(b_names) - set(a_names)
            self.assertTrue(added)
            self.assertTrue(all(name.startswith("cat:merchant=") for name in added))
            self.assertEqual(set(a_names), {n for n in b_names if not n.startswith("cat:merchant=")})
            am, bm = pa.transform(fa), pb.transform(fb)
            for left, right in zip(am, bm, strict=True):
                self.assertEqual(dict(zip(a_names, left, strict=True)),
                                 {name: value for name, value in zip(b_names, right, strict=True)
                                  if not name.startswith("cat:merchant=")})

    def test_common_raw_and_rules_match_only_corresponding_history_conditions(self) -> None:
        rows = fixture()
        for off, on in (("A0", "B0"), ("A1", "B1"), ("A2", "B2")):
            fa, pa = prepared(rows, off)
            fb, pb = prepared(rows, on)
            self.assertEqual(pa.rules, [r for r in pb.rules if r["column"] != "merchant"])
            for a, b in zip(fa, fb, strict=True):
                self.assertEqual(a, {k: v for k, v in b.items() if k != "merchant"})

    def test_six_cells_preserve_all_identities_and_empty_history_rows(self) -> None:
        rows = fixture()
        expected = [r.source_row_id for r in sorted(rows, key=lambda r: (r.step, r.source_row_id))]
        for cell in CELLS:
            frame, proc = prepared(rows, cell)
            self.assertEqual([r["source_row_id"] for r in frame], expected)
            self.assertEqual(len(proc.transform(frame)), len(rows))
            self.assertTrue({"source_row_id", "customer", "fraud"}.isdisjoint(proc.columns))
            # h*_customers is the allowed historical distinct count, not a raw ID.
            self.assertTrue(all(not name.startswith(("cat:customer=", "num:customer",
                                                     "cat:source_row_id=", "num:source_row_id",
                                                     "cat:fraud=", "num:fraud"))
                                for name in proc.feature_names()))

    def test_m0_retains_category_location_and_history_retrieval_key(self) -> None:
        rows = [one(0, 0, merchant="m"), one(1, 1, merchant="m"),
                one(2, 1, merchant="other")]
        values, _ = history_stream(rows, "HS")
        frame = cell_frame(rows, "A1", values)
        self.assertIn("category", frame[1])
        self.assertIn("zipMerchant", frame[1])
        self.assertNotIn("merchant", frame[1])
        self.assertEqual(frame[1]["h7_count"], 1)
        self.assertEqual(frame[2]["h7_count"], 0)


class SyntheticMetricTests(unittest.TestCase):
    def test_capacity_ceiling_exact(self) -> None:
        self.assertEqual([capacity(n) for n in (0, 1, 33, 34, 100, 198311)],
                         [0, 1, 1, 2, 3, 5950])

    def test_capacity_invalid_rejected(self) -> None:
        for value in ("0", "-0.1", "1.1"):
            with self.subTest(value=value), self.assertRaises(ContractError):
                capacity(10, value)

    def test_original_hash_priority_format(self) -> None:
        expected = hashlib.sha256(b"banksim-capacity-v1|12").hexdigest()
        self.assertEqual(priority(12), expected)

    def test_ties_use_shared_identity_priority_and_permutation_invariance(self) -> None:
        ids = list(range(60))
        labels = [i % 2 for i in ids]
        scores = [0.5] * 60
        expected_ids = sorted(ids, key=lambda i: (priority(i), i))[:2]
        a = artificial_metrics(labels, scores, ids)
        b = artificial_metrics(list(reversed(labels)), list(reversed(scores)), list(reversed(ids)))
        self.assertEqual(a, b)
        self.assertEqual(a["selected_ids"], expected_ids)

    def test_ap_groups_ties_instead_of_priority_order(self) -> None:
        values = artificial_metrics([1, 0, 1], [1.0, 1.0, 0.0], [0, 1, 2])
        self.assertAlmostEqual(values["AP"], 7 / 12)

    def test_no_positive_rows_kept_and_ratios_undefined(self) -> None:
        values = artificial_metrics([0, 0], [0.1, 0.2], [0, 1])
        self.assertEqual((values["N"], values["P"], values["K"], values["TP"]), (2, 0, 1, 0))
        self.assertIsNone(values["Recall"])
        self.assertIsNone(values["AP"])
        self.assertEqual(values["Precision"], 0)

    def test_empty_evaluation_boundary(self) -> None:
        values = artificial_metrics([], [], [])
        self.assertEqual((values["N"], values["K"]), (0, 0))
        self.assertIsNone(values["Precision"])

    def test_duplicate_ids_and_nonfinite_scores_rejected(self) -> None:
        with self.assertRaises(ContractError):
            artificial_metrics([0, 1], [0, 1], [0, 0])
        with self.assertRaises(ContractError):
            artificial_metrics([1], [math.nan], [0])

    def test_global_completed_window_is_not_daily_budget(self) -> None:
        self.assertEqual(capacity(66), 2)
        self.assertEqual(capacity(33) + capacity(33), 2)
        self.assertEqual(capacity(34), 2)
        self.assertEqual(capacity(17) + capacity(17), 2)
        # Different ceiling at another artificial size explicitly distinguishes rules.
        self.assertEqual(capacity(100), 3)
        self.assertEqual(capacity(50) + capacity(50), 4)

    def test_micro_recall_uses_total_positive_denominator(self) -> None:
        self.assertAlmostEqual((1 + 1) / (1 + 9), 0.2)
        self.assertNotEqual((1 + 1) / (1 + 9), (1 / 1 + 1 / 9) / 2)

    def test_difference_of_differences_hand_example(self) -> None:
        counts = {"A0": 0, "A1": 1, "A2": 3, "B0": 1, "B1": 2, "B2": 3}
        measures = {c: {"N": 100, "P": 4, "K": 3, "TP": tp} for c, tp in counts.items()}
        self.assertEqual(artificial_contrasts(measures),
                         {"delta0": 0.5, "delta1": 0.25, "I": -0.25})

    def test_contrasts_reject_population_or_budget_mismatch(self) -> None:
        measures = {c: {"N": 100, "P": 4, "K": 3, "TP": 1} for c in CELLS}
        measures["B2"]["K"] = 2
        with self.assertRaises(ContractError):
            artificial_contrasts(measures)

    def test_zero_positive_contrasts_undefined(self) -> None:
        measures = {c: {"N": 100, "P": 0, "K": 3, "TP": 0} for c in CELLS}
        self.assertEqual(artificial_contrasts(measures),
                         {"delta0": None, "delta1": None, "I": None})


class IsolationTests(unittest.TestCase):
    def test_guard_blocks_external_file_before_open(self) -> None:
        with self.assertRaises(PermissionError):
            Path("D:/forbidden-artificial-sentinel.csv").read_text(encoding="utf-8")

    def test_guard_blocks_model_import_event(self) -> None:
        for module in ("lightgbm", "xgboost", "numpy", "pandas", "banksim"):
            with self.subTest(module=module), self.assertRaises(PermissionError):
                GUARD("import", (module,))

    def test_guard_blocks_network_and_worker_creation_events(self) -> None:
        for event in ("socket.connect", "subprocess.Popen", "os.system"):
            with self.subTest(event=event), self.assertRaises(PermissionError):
                GUARD(event, ("artificial-denied-operation",))

    def test_no_real_library_loaded(self) -> None:
        forbidden = ("lightgbm", "xgboost", "numpy", "pandas", "sklearn", "banksim")
        self.assertTrue(all(name not in sys.modules for name in forbidden))


class RecordingResult(unittest.TextTestResult):
    def __init__(self, stream: Any, descriptions: bool, verbosity: int) -> None:
        super().__init__(stream, descriptions, verbosity)
        self.records: list[dict[str, Any]] = []
        self.subtests: list[dict[str, Any]] = []
        self.started: dict[str, float] = {}

    def startTest(self, test: unittest.TestCase) -> None:
        self.started[test.id()] = time.perf_counter()
        super().startTest(test)

    def _record(self, test: unittest.TestCase, status: str, detail: str | None = None) -> None:
        self.records.append({"id": test.id(), "description": test.shortDescription(),
                             "status": status, "seconds": time.perf_counter() - self.started[test.id()],
                             "detail": detail})

    def addSuccess(self, test: unittest.TestCase) -> None:
        self._record(test, "PASS")
        super().addSuccess(test)

    def addFailure(self, test: unittest.TestCase, err: Any) -> None:
        self._record(test, "FAIL", "".join(traceback.format_exception(*err)))
        super().addFailure(test, err)

    def addError(self, test: unittest.TestCase, err: Any) -> None:
        self._record(test, "ERROR", "".join(traceback.format_exception(*err)))
        super().addError(test, err)

    def addSubTest(self, test: unittest.TestCase, subtest: Any, err: Any) -> None:
        self.subtests.append({"parent": test.id(), "case": str(subtest),
                              "status": "PASS" if err is None else "FAIL",
                              "detail": None if err is None else "".join(traceback.format_exception(*err))})
        super().addSubTest(test, subtest, err)


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="Fresh child directory under this preparation's results/")
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to(RUN_ROOT / "results"):
        parser.error("output must be inside this independent results directory")
    if output.exists():
        parser.error("output exists; preserve every old attempt")
    output.mkdir(parents=True)
    sys.addaudithook(GUARD)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in CODE_ROOT.glob("*.py")}
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    console = io.StringIO()
    runner = unittest.TextTestRunner(stream=console, verbosity=2, resultclass=RecordingResult)
    result = runner.run(suite)
    if not isinstance(result, RecordingResult):
        raise TypeError("unexpected test result type")
    rows = fixture()
    histories: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    cells: dict[str, Any] = {}
    for mode in ("H0", "HS", "HL"):
        histories[mode], audits[mode] = history_stream(rows, mode)
    for cell in CELLS:
        frame, proc = prepared(rows, cell)
        cells[cell] = {
            "columns": input_columns(cell), "feature_names": proc.feature_names(),
            "synthetic_training_ids": [r["source_row_id"] for r in frame if r["split"] == "train"],
            "rules": proc.rules, "raw_model_frame": frame, "encoded_matrix": proc.transform(frame),
        }
    elapsed = time.perf_counter() - started
    write_json(output / "synthetic_fixture.json", {
        "origin": "constructed inside this source; no external input",
        "rows": [asdict(r) for r in rows], "total_rows": len(rows),
        "split_rows": {s: sum(r.split == s for r in rows)
                       for s in ("train", "validation", "review")},
        "contains_real_labels": False,
    })
    write_json(output / "six_cell_trace.json", {
        "origin": "artificial only", "histories": histories, "history_sources": audits, "cells": cells,
    })
    with (output / "unittest_console.txt").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(console.getvalue())
    summary = {
        "status": "ARTIFICIAL_CONTRACT_PASS" if result.wasSuccessful() else "ARTIFICIAL_CONTRACT_FAIL",
        "started_utc": started_utc, "seconds": elapsed, "python_version": sys.version,
        "unit_tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "parameter_cases": len(result.subtests),
        "parameter_failures": sum(r["status"] != "PASS" for r in result.subtests),
        "tests": result.records, "parameter_results": result.subtests, "source_sha256": source_hashes,
        "guard": {"allowed_file_access": GUARD.opens, "expected_denials": GUARD.denials},
        "execution": {"real_dataset_reads": 0, "real_label_reads": 0, "real_model_loads": 0,
                      "real_model_fit_calls": 0, "real_preprocessor_fits": 0,
                      "real_feature_generation": 0, "real_predictions": 0,
                      "real_score_cache_reads": 0, "selection_supplement_fits": 0,
                      "real_statistical_replays": 0},
        "limits": ["stdlib artificial kernel, not native adapter parity",
                   "No training/resource-supervisor integration acceptance",
                   "No scientific efficacy, noninferiority, retention decision, or independent validation"],
        "training_authorized": False,
    }
    write_json(output / "test_results.json", summary)
    print(console.getvalue(), end="")
    print(json.dumps({k: summary[k] for k in ("status", "unit_tests", "parameter_cases",
                                           "failures", "errors", "seconds")}, ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
