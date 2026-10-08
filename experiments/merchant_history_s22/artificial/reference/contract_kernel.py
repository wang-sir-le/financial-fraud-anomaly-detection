"""Small artificial contract kernel; no dataset loader, learner, or external dependency."""

from __future__ import annotations

import hashlib
import math
import struct
from collections import Counter, deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from itertools import groupby
from statistics import median
from typing import Any

WINDOWS = (1, 7, 30)
STATS = ("count", "customers", "mean", "max", "std", "ratio")
HISTORY = tuple(f"h{w}_{s}" for w in WINDOWS for s in STATS) + (
    "h_recency", "h_available",
)
X_ON = ("step", "category", "amount", "age", "gender", "merchant", "zipcodeOri", "zipMerchant")
X_OFF = tuple(c for c in X_ON if c != "merchant")
CATEGORICAL = frozenset(("category", "age", "gender", "merchant", "zipcodeOri", "zipMerchant"))
MISSING = "__BANKSIM_MISSING_V1__"
CELLS = {
    "A0": (False, "H0"), "B0": (True, "H0"),
    "A1": (False, "HS"), "B1": (True, "HS"),
    "A2": (False, "HL"), "B2": (True, "HL"),
}


class ContractError(ValueError):
    """Artificial contract violation; never silently repair a row."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def parse_cents(text: str) -> int:
    require(isinstance(text, str), "amount must be source text")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ContractError("amount is not a decimal") from exc
    require(value.is_finite(), "nonfinite amount")
    numerator, denominator = value.as_integer_ratio()
    require((100 * numerator) % denominator == 0, "sub-cent precision")
    cents = 100 * numerator // denominator
    require(-(2**63) < cents < 2**63, "cents outside original int64 domain")
    return cents


def split_for(step: int) -> str:
    require(type(step) is int and 0 <= step <= 179, "step outside contract")
    if step <= 107:
        return "train"
    return "validation" if step <= 125 else "review"


@dataclass(frozen=True)
class Row:
    source_row_id: int
    step: int
    merchant: str
    customer: str
    cents: int
    category: str | None = "cat"
    age: str | None = "2"
    gender: str | None = "F"
    zipcodeOri: str | None = "z0"
    zipMerchant: str | None = "z1"

    @property
    def split(self) -> str:
        return split_for(self.step)


def rows_from_raw(raw: list[dict[str, Any]]) -> list[Row]:
    """In-memory artificial projection. Row labels are rejected before construction."""
    answer = []
    allowed = {
        "source_row_id", "step", "merchant", "customer", "amount", "category",
        "age", "gender", "zipcodeOri", "zipMerchant",
    }
    for position, item in enumerate(raw):
        require("fraud" not in item and "label" not in item, "history rejects labels")
        require(set(item) <= allowed, "unexpected raw field")
        step_text = item.get("step")
        if not isinstance(step_text, str):
            raise ContractError("step must be source text")
        require(step_text.isascii() and step_text.isdigit(),
                "step must be nonnegative source integer text")
        step = int(step_text)
        split_for(step)
        for name in ("merchant", "customer"):
            require(isinstance(item.get(name), str) and bool(item[name]), f"missing {name}")
        identity = item.get("source_row_id", position)
        require(type(identity) is int and identity >= 0, "invalid source identity")
        amount_text = item.get("amount")
        if not isinstance(amount_text, str):
            raise ContractError("amount must be source text")
        answer.append(Row(
            identity, step, item["merchant"], item["customer"], parse_cents(amount_text),
            **{c: item.get(c, getattr(Row, c)) for c in
               ("category", "age", "gender", "zipcodeOri", "zipMerchant")},
        ))
    validate_rows(answer)
    return answer


def validate_rows(rows: list[Row]) -> None:
    require(len({r.source_row_id for r in rows}) == len(rows), "duplicate source identity")
    for row in rows:
        split_for(row.step)
        require(type(row.source_row_id) is int and row.source_row_id >= 0, "invalid identity")
        require(type(row.cents) is int and -(2**63) < row.cents < 2**63, "invalid cents")
        require(isinstance(row.merchant, str) and bool(row.merchant), "missing merchant")
        require(isinstance(row.customer, str) and bool(row.customer), "missing customer")


class WindowState:
    def __init__(self) -> None:
        self.rows: deque[Row] = deque()
        self.maxima: deque[Row] = deque()
        self.customers: Counter[str] = Counter()
        self.total = 0
        self.square = 0

    def expire(self, lower: int) -> None:
        while self.rows and self.rows[0].step < lower:
            row = self.rows.popleft()
            self.total -= row.cents
            self.square -= row.cents * row.cents
            self.customers[row.customer] -= 1
            if not self.customers[row.customer]:
                del self.customers[row.customer]
        while self.maxima and self.maxima[0].step < lower:
            self.maxima.popleft()

    def add(self, row: Row) -> None:
        self.rows.append(row)
        self.total += row.cents
        self.square += row.cents * row.cents
        self.customers[row.customer] += 1
        while self.maxima and self.maxima[-1].cents <= row.cents:
            self.maxima.pop()
        self.maxima.append(row)

    def stats(self) -> tuple[float, ...]:
        n = len(self.rows)
        if not n:
            return (0, 0, 0.0, 0.0, 0.0)
        numerator = n * self.square - self.total * self.total
        require(numerator >= 0, "negative exact variance")
        std = math.sqrt(numerator / (n * (n - 1))) / 100 if n > 1 else 0.0
        return (n, len(self.customers), self.total / (100 * n),
                self.maxima[0].cents / 100, std)


def history_stream(rows: list[Row], mode: str) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    """Sliding integer moments. HS never allocates the all-past previous dictionary."""
    require(mode in ("H0", "HS", "HL"), "unknown history condition")
    validate_rows(rows)
    canonical = sorted(rows, key=lambda r: (r.step, r.source_row_id))
    output: dict[int, dict[str, float]] = {}
    audit: dict[str, Any] = {"mode": mode, "state_allocations": 0, "used_history": {},
                             "retained_state_after_step": {}}
    if mode == "H0":
        return {r.source_row_id: {} for r in canonical}, audit
    states: dict[str, dict[int, WindowState]] = {}
    # None is deliberate: a capped condition must not retain an all-past last event.
    previous: dict[str, Row] | None = {} if mode == "HL" else None
    for step, step_rows_iterator in groupby(canonical, key=lambda r: r.step):
        current = list(step_rows_iterator)
        merchants = sorted({r.merchant for r in current})
        for merchant in merchants:
            if merchant not in states:
                states[merchant] = {w: WindowState() for w in WINDOWS}
                audit["state_allocations"] += 3
        # Eager expiration also covers inactive merchants; no excluded row remains
        # in a capped window merely because that entity had no new transaction.
        for local in states.values():
            for w in WINDOWS:
                local[w].expire(step - (min(w, 7) if mode == "HS" else w))
        # Emit the entire step before committing any current transaction.
        for row in current:
            local = states[row.merchant]
            features: dict[str, float] = {}
            used: dict[str, Any] = {}
            for w in WINDOWS:
                stats = local[w].stats()
                for stat_name, value in zip(STATS[:-1], stats, strict=True):
                    features[f"h{w}_{stat_name}"] = value
                features[f"h{w}_ratio"] = (row.cents / 100 / stats[2]) if stats[2] else 0.0
                used[f"h{w}"] = [r.source_row_id for r in local[w].rows]
            last = (previous.get(row.merchant) if previous is not None else
                    (local[7].rows[-1] if local[7].rows else None))
            features["h_recency"] = row.step - last.step if last else -1
            features["h_available"] = int(last is not None)
            used["recency_source"] = last.source_row_id if last else None
            require(all(math.isfinite(v) for v in features.values()), "nonfinite history")
            output[row.source_row_id] = features
            audit["used_history"][str(row.source_row_id)] = used
        for row in current:
            for state in states[row.merchant].values():
                state.add(row)
            if previous is not None:
                previous[row.merchant] = row
        audit["retained_state_after_step"][str(step)] = {
            merchant: {str(w): [r.source_row_id for r in state.rows]
                       for w, state in local.items()} for merchant, local in states.items()
        }
    audit["all_past_last_state"] = previous is not None
    return output, audit


def input_columns(cell: str) -> tuple[str, ...]:
    require(cell in CELLS, "unknown six-cell identifier")
    merchant_on, mode = CELLS[cell]
    return (X_ON if merchant_on else X_OFF) + (() if mode == "H0" else HISTORY)


def cell_frame(rows: list[Row], cell: str, history: dict[int, dict[str, float]]) -> list[dict[str, Any]]:
    mode = CELLS[cell][1]
    result = []
    for row in sorted(rows, key=lambda r: (r.step, r.source_row_id)):
        history_values = history[row.source_row_id]
        require(set(history_values) == (set() if mode == "H0" else set(HISTORY)),
                "history schema differs from condition")
        values = {"step": float(row.step), "amount": row.cents / 100,
                  **{c: getattr(row, c) for c in CATEGORICAL}, **history_values}
        result.append({
            "source_row_id": row.source_row_id, "split": row.split,
            **{c: values[c] for c in input_columns(cell)},
        })
    return result


def float32(value: float) -> float:
    try:
        answer = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ContractError("float32 overflow") from exc
    require(math.isfinite(answer), "float32 nonfinite")
    return answer


def category_value(value: Any) -> str:
    require(value != MISSING, "reserved missing marker collision")
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return MISSING
    require(isinstance(value, str), "category must be unmodified source string")
    return value


def numeric_values(frame: list[dict[str, Any]], column: str) -> list[float]:
    try:
        values = [float(r[column]) if r[column] is not None else math.nan for r in frame]
    except (TypeError, ValueError) as exc:
        raise ContractError("invalid numeric input") from exc
    require(not any(math.isinf(v) for v in values), "infinite numeric input")
    return values


class ArtificialPreprocessor:
    """A stdlib mirror of the documented train-only rules, not the native adapter."""

    def __init__(self, columns: tuple[str, ...]) -> None:
        require(len(columns) == len(set(columns)), "duplicate columns")
        require(set(columns) <= (set(X_ON) | set(HISTORY)), "forbidden predictor")
        self.columns = columns
        self.rules: list[dict[str, Any]] = []
        self.fitted = False

    def fit(self, frame: list[dict[str, Any]]) -> ArtificialPreprocessor:
        require(bool(frame) and all(r["split"] == "train" for r in frame), "training rows only")
        self.rules = []
        for column in self.columns:
            if column in CATEGORICAL:
                values = [category_value(r[column]) for r in frame]
                vocab = sorted(set(values))
                keep = [v for v in vocab if not all(x == v for x in values)]
                self.rules.append({"column": column, "kind": "categorical",
                                   "vocabulary": vocab, "keep": keep})
            else:
                numeric = numeric_values(frame, column)
                valid = [v for v in numeric if not math.isnan(v)]
                fill = float(median(valid)) if valid else 0.0
                missing = [math.isnan(v) for v in numeric]
                filled = [float32(fill if absent else v)
                          for v, absent in zip(numeric, missing, strict=True)]
                self.rules.append({"column": column, "kind": "numeric", "median": fill,
                                   "keep_value": len(set(filled)) > 1,
                                   "keep_missing": len(set(missing)) > 1})
        self.fitted = True
        self.feature_names()
        return self

    def feature_names(self) -> list[str]:
        require(self.fitted, "preprocessor not fitted")
        names: list[str] = []
        for rule in self.rules:
            column = rule["column"]
            if rule["kind"] == "categorical":
                names.extend(f"cat:{column}={v}" for v in rule["keep"])
            else:
                if rule["keep_value"]:
                    names.append(f"num:{column}")
                if rule["keep_missing"]:
                    names.append(f"missing:{column}")
        require(len(names) == len(set(names)), "encoded name collision")
        return names

    def transform(self, frame: list[dict[str, Any]]) -> list[list[float]]:
        require(self.fitted, "preprocessor not fitted")
        blocks: list[list[float]] = []
        for rule in self.rules:
            column = rule["column"]
            if rule["kind"] == "categorical":
                values = [category_value(r[column]) for r in frame]
                blocks.extend([float(x == v) for x in values] for v in rule["keep"])
            else:
                numeric = numeric_values(frame, column)
                missing = [math.isnan(v) for v in numeric]
                if rule["keep_value"]:
                    blocks.append([float32(rule["median"] if absent else value)
                                   for value, absent in zip(numeric, missing, strict=True)])
                if rule["keep_missing"]:
                    blocks.append([float(absent) for absent in missing])
        result = [[block[i] for block in blocks] for i in range(len(frame))]
        require(all(math.isfinite(v) for row in result for v in row), "nonfinite encoded matrix")
        return result


def capacity(n: int, fraction: str = "0.03") -> int:
    require(type(n) is int and n >= 0, "invalid population")
    q = Fraction(fraction)
    require(0 < q <= 1, "invalid capacity")
    return (n * q.numerator + q.denominator - 1) // q.denominator


def priority(source_id: int) -> str:
    return hashlib.sha256(f"banksim-capacity-v1|{source_id}".encode()).hexdigest()


def artificial_metrics(labels: list[int], scores: list[float], ids: list[int]) -> dict[str, Any]:
    """Only hand-generated scores/labels are supplied by this test harness."""
    require(len(labels) == len(scores) == len(ids), "length mismatch")
    require(len(set(ids)) == len(ids), "duplicate evaluation identity")
    require(all(type(y) is int and y in (0, 1) for y in labels), "nonbinary labels")
    require(all(math.isfinite(s) for s in scores), "nonfinite synthetic score")
    n = len(labels)
    k = capacity(n)
    positives = sum(labels)
    order = sorted(range(n), key=lambda j: (-scores[j], priority(ids[j]), ids[j]))
    tp = sum(labels[j] for j in order[:k])
    ap = None
    if positives:
        accumulated_tp = 0
        accumulated_n = 0
        ap = 0.0
        for _, group in groupby(order, key=lambda j: scores[j]):
            indices = list(group)
            group_tp = sum(labels[j] for j in indices)
            accumulated_tp += group_tp
            accumulated_n += len(indices)
            ap += group_tp / positives * (accumulated_tp / accumulated_n)
    return {"N": n, "P": positives, "K": k, "TP": tp,
            "Recall": tp / positives if positives else None,
            "Precision": tp / k if k else None, "AP": ap,
            "selected_ids": [ids[j] for j in order[:k]]}


def artificial_contrasts(measures: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    require(set(measures) == set(CELLS), "six results required")
    require(len({(m["N"], m["P"], m["K"]) for m in measures.values()}) == 1,
            "populations and budgets differ")
    p = measures["A0"]["P"]
    if not p:
        return {"delta0": None, "delta1": None, "I": None}
    delta0 = (measures["A2"]["TP"] - measures["A1"]["TP"]) / p
    delta1 = (measures["B2"]["TP"] - measures["B1"]["TP"]) / p
    return {"delta0": delta0, "delta1": delta1, "I": delta1 - delta0}
