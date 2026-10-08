"""Final pandas/numpy feature, encoding and ranking path for both phases."""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter, deque
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd
from native_common import (
    CATEGORICAL,
    CELL_COLUMNS,
    HEADER,
    HISTORY,
    WINDOWS,
    require,
)
from sklearn.metrics import average_precision_score

MISSING = "__BANKSIM_MISSING_V1__"
COUNTERS = {"artificial_preprocessor_fits": 0, "project_preprocessor_fits": 0}


def parse_cents(text: str) -> int:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal amount") from exc
    require(value.is_finite(), "nonfinite amount")
    a, b = value.as_integer_ratio()
    require((100 * a) % b == 0, "sub-cent precision")
    cents = 100 * a // b
    require(-(2**63) < cents < 2**63, "cents outside original domain")
    return cents


def normalize_public(raw: pd.DataFrame, *, purpose: str) -> pd.DataFrame:
    require(purpose in ("artificial", "project"), "purpose not declared")
    require("fraud" not in raw and "label" not in raw, "labels entered feature projection")
    require(
        set(raw) == set(HEADER[:-1]) or set(raw) == set(HEADER[:-1] + ["source_row_id"]),
        "unexpected public input columns",
    )
    frame = raw.copy()
    if "source_row_id" not in frame:
        frame.insert(0, "source_row_id", np.arange(len(frame), dtype=np.int64))
    require(frame["source_row_id"].is_unique, "duplicate source identity")
    require(bool(frame["source_row_id"].ge(0).all()), "negative source identity")
    for field in ("step", "merchant", "customer", "amount"):
        require(
            not bool(frame[field].isna().any()) and not bool(frame[field].eq("").any()),
            f"required field missing: {field}",
        )
    require(bool(frame["step"].map(lambda v: isinstance(v, str)).all()), "step must be source text")
    require(bool(frame["step"].str.fullmatch(r"[0-9]+").all()), "invalid step syntax")
    frame["step"] = frame["step"].astype(np.int64)
    require(bool(frame["step"].between(0, 179).all()), "unassigned time region")
    require(
        bool(frame["merchant"].map(lambda v: isinstance(v, str)).all()), "merchant string changed"
    )
    frame["amount_cents"] = frame["amount"].map(parse_cents).astype(np.int64)
    frame["amount"] = frame["amount_cents"].astype(np.float64) / 100
    frame = frame.sort_values(["step", "source_row_id"], kind="stable").reset_index(drop=True)
    frame["split"] = np.where(
        frame["step"] <= 107, "train", np.where(frame["step"] <= 125, "validation", "review")
    )
    frame.attrs["purpose"] = purpose
    require("fraud" not in frame, "label projection failed")
    return frame


class WindowState:
    def __init__(self) -> None:
        self.rows: deque[tuple[int, str, int, int]] = deque()
        self.maxima: deque[tuple[int, int]] = deque()
        self.customers: Counter[str] = Counter()
        self.total = 0
        self.square = 0

    def expire(self, lower: int) -> int:
        expired = 0
        while self.rows and self.rows[0][0] < lower:
            _, customer, cents, _ = self.rows.popleft()
            self.customers[customer] -= 1
            if not self.customers[customer]:
                del self.customers[customer]
            self.total -= cents
            self.square -= cents * cents
            expired += 1
        while self.maxima and self.maxima[0][0] < lower:
            self.maxima.popleft()
        return expired

    def add(self, step: int, customer: str, cents: int, identity: int) -> None:
        self.rows.append((step, customer, cents, identity))
        self.customers[customer] += 1
        self.total += cents
        self.square += cents * cents
        while self.maxima and self.maxima[-1][1] <= cents:
            self.maxima.pop()
        self.maxima.append((step, cents))

    def stats(self) -> tuple[int, int, float, float, float]:
        n = len(self.rows)
        if not n:
            return (0, 0, 0.0, 0.0, 0.0)
        numerator = n * self.square - self.total * self.total
        require(numerator >= 0, "negative exact variance numerator")
        std = math.sqrt(numerator / (n * (n - 1))) / 100 if n > 1 else 0.0
        return n, len(self.customers), self.total / (100 * n), self.maxima[0][1] / 100, std


def build_history(
    frame: pd.DataFrame, mode: str, *, trace: bool = False
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require("fraud" not in frame and "label" not in frame, "history cannot receive labels")
    require(mode in ("H0", "HS", "HL"), "invalid history condition")
    require(frame["source_row_id"].is_unique, "duplicate history source identity")
    started = time.perf_counter()
    result = (
        frame.sort_values(["step", "source_row_id"], kind="stable").reset_index(drop=True).copy()
    )
    audit: dict[str, Any] = {
        "mode": mode,
        "entity": "merchant",
        "expired_state_rows": 0,
        "all_past_previous_allocated": mode == "HL",
    }
    if mode == "H0":
        audit["seconds"] = time.perf_counter() - started
        return result, audit
    values = np.zeros((len(result), len(HISTORY)), dtype=np.float64)
    states: dict[str, dict[int, WindowState]] = {}
    previous: dict[str, tuple[int, int]] | None = {} if mode == "HL" else None
    sources: dict[str, Any] = {}
    for step_value, step_indices in result.groupby("step", sort=True).groups.items():
        step = int(step_value)
        step_frame = result.loc[step_indices]
        for merchant in sorted(set(step_frame["merchant"])):
            states.setdefault(merchant, {w: WindowState() for w in WINDOWS})
        # Expire inactive merchants too; HS never keeps excluded aggregate state.
        for local in states.values():
            for window, state in local.items():
                audit["expired_state_rows"] += state.expire(
                    step - (min(window, 7) if mode == "HS" else window)
                )
        for merchant, indices in step_frame.groupby("merchant", sort=True).groups.items():
            idx = np.asarray(indices, dtype=np.int64)
            local = states[merchant]
            used: dict[str, Any] = {}
            for j, window in enumerate(WINDOWS):
                n, customers, mean, maximum, std = local[window].stats()
                values[idx, j * 6 : j * 6 + 5] = [n, customers, mean, maximum, std]
                values[idx, j * 6 + 5] = (
                    result.loc[idx, "amount"].to_numpy() / mean if mean else 0.0
                )
                if trace:
                    used[f"h{window}"] = [r[3] for r in local[window].rows]
            last = (
                previous.get(merchant)
                if previous is not None
                else ((local[7].rows[-1][0], local[7].rows[-1][3]) if local[7].rows else None)
            )
            values[idx, -2] = step - last[0] if last is not None else -1
            values[idx, -1] = int(last is not None)
            if trace:
                used["recency_source"] = last[1] if last else None
                for identity in result.loc[idx, "source_row_id"]:
                    sources[str(int(identity))] = dict(used)
        # No merchant/time current row can affect another row emitted above.
        for merchant, customer, cents, identity in step_frame[
            ["merchant", "customer", "amount_cents", "source_row_id"]
        ].itertuples(index=False, name=None):
            for state in states[merchant].values():
                state.add(step, str(customer), int(cents), int(identity))
            if previous is not None:
                previous[merchant] = (step, int(identity))
    require(bool(np.isfinite(values).all()), "nonfinite history")
    for j, name in enumerate(HISTORY):
        result[name] = values[:, j]
    if trace:
        audit["used_history"] = sources
    audit["seconds"] = time.perf_counter() - started
    return result, audit


def category_values(series: pd.Series) -> np.ndarray:
    vals = series.to_numpy(dtype=object).copy()
    require(not bool(np.any(vals == MISSING)), "reserved category marker occurs in source")
    vals[pd.isna(vals) | (vals == "")] = MISSING
    require(all(isinstance(x, str) for x in vals), "category must be unmodified string")
    return vals


class NativePreprocessor:
    def __init__(self, columns: list[str]) -> None:
        require(len(columns) == len(set(columns)), "duplicate predictor")
        require(set(columns) <= set(CELL_COLUMNS["B2"]), "unknown predictor")
        self.columns = list(columns)
        self.rules: list[dict[str, Any]] = []
        self.fitted = False

    def fit(self, train: pd.DataFrame) -> NativePreprocessor:
        require("fraud" not in train and "label" not in train, "labels entered preprocessing")
        require(
            not train.empty and bool(train["split"].eq("train").all()), "fit accepts train only"
        )
        purpose = train.attrs.get("purpose")
        require(purpose in ("artificial", "project"), "preprocessing purpose missing")
        COUNTERS[f"{purpose}_preprocessor_fits"] += 1
        require(COUNTERS["project_preprocessor_fits"] <= 6, "project preprocessing budget exceeded")
        self.rules = []
        for column in self.columns:
            if column in CATEGORICAL:
                values = category_values(train[column])
                vocab = sorted(set(values))
                keep = [v for v in vocab if not bool(np.all(values == v))]
                self.rules.append(
                    {"column": column, "kind": "categorical", "vocabulary": vocab, "keep": keep}
                )
            else:
                numeric = train[column].to_numpy(dtype=np.float64)
                require(not bool(np.isinf(numeric).any()), "infinite numeric input")
                missing = np.isnan(numeric)
                median = float(np.median(numeric[~missing])) if (~missing).any() else 0.0
                filled = np.where(missing, median, numeric).astype(np.float32)
                require(bool(np.isfinite(filled).all()), "float32 overflow")
                self.rules.append(
                    {
                        "column": column,
                        "kind": "numeric",
                        "median": median,
                        "keep_value": not bool(np.all(filled == filled[0])),
                        "keep_missing": not bool(np.all(missing == missing[0])),
                    }
                )
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

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        require(self.fitted, "preprocessor not fitted")
        require("fraud" not in frame and "label" not in frame, "labels entered transform")
        blocks: list[Any] = []
        for rule in self.rules:
            column = rule["column"]
            if rule["kind"] == "categorical":
                vals = category_values(frame[column])
                blocks.extend((vals == v).astype(np.float32) for v in rule["keep"])
            else:
                vals = frame[column].to_numpy(dtype=np.float64)
                require(not bool(np.isinf(vals).any()), "infinite numeric input")
                missing = np.isnan(vals)
                if rule["keep_value"]:
                    blocks.append(np.where(missing, rule["median"], vals).astype(np.float32))
                if rule["keep_missing"]:
                    blocks.append(missing.astype(np.float32))
        matrix = (
            np.column_stack(blocks).astype(np.float32)
            if blocks
            else np.empty((len(frame), 0), dtype=np.float32)
        )
        require(bool(np.isfinite(matrix).all()), "nonfinite encoded matrix")
        return matrix

    def snapshot(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "rules": self.rules,
            "names": self.feature_names(),
            "fit_split": "train",
            "dtype": "float32",
            "missing_token": MISSING,
        }

    @classmethod
    def restore(cls, value: dict[str, Any]) -> NativePreprocessor:
        result = cls(value["columns"])
        result.rules, result.fitted = value["rules"], True
        require(result.feature_names() == value["names"], "preprocessor reload schema changed")
        return result


def metrics(labels: np.ndarray, scores: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    labels, scores, ids = np.asarray(labels), np.asarray(scores), np.asarray(ids)
    require(labels.ndim == scores.ndim == ids.ndim == 1, "metric dimensional mismatch")
    require(len(labels) == len(scores) == len(ids), "metric row mismatch")
    require(len(np.unique(ids)) == len(ids), "duplicate evaluation source")
    require(
        bool(np.isin(labels, [0, 1]).all()) and bool(np.isfinite(scores).all()),
        "invalid metric inputs",
    )
    n, p = len(labels), int(labels.sum())
    k = (3 * n + 99) // 100
    priorities = np.array(
        [hashlib.sha256(f"banksim-capacity-v1|{int(i)}".encode()).hexdigest() for i in ids]
    )
    order = np.lexsort((ids, priorities, -scores))
    tp = int(labels[order[:k]].sum())
    boundary: dict[str, Any] = {}
    if k:
        cut = float(scores[order[k - 1]])
        tied = scores == cut
        above = scores > cut
        slots = k - int(above.sum())
        tie_p = int(labels[tied].sum())
        tie_n = int(tied.sum())
        above_tp = int(labels[above].sum())
        boundary = {
            "margin": cut,
            "tied_rows": tie_n,
            "tied_positives": tie_p,
            "slots_from_tie": slots,
            "tp_min": above_tp + max(0, slots - (tie_n - tie_p)),
            "tp_max": above_tp + min(slots, tie_p),
        }
    return {
        "N": n,
        "P": p,
        "K": k,
        "TP": tp,
        "Recall": tp / p if p else None,
        "Precision": tp / k if k else None,
        "AP": float(average_precision_score(labels, scores)) if p else None,
        "boundary_tie": boundary,
        "selected_ids": ids[order[:k]].astype(int).tolist(),
    }


def contrasts(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    require(set(table) == set(CELL_COLUMNS), "six complete conditions required")
    require(
        len({(v["N"], v["P"], v["K"]) for v in table.values()}) == 1,
        "common rows/positive denominator/capacity mismatch",
    )
    p = table["A0"]["P"]
    d0 = table["A2"]["TP"] - table["A1"]["TP"]
    d1 = table["B2"]["TP"] - table["B1"]["TP"]
    j = d1 - d0
    return {
        "common_P": p,
        "delta0_TP": d0,
        "delta1_TP": d1,
        "J": j,
        "delta0": d0 / p if p else None,
        "delta1": d1 / p if p else None,
        "I": j / p if p else None,
        "history_vs_none": {
            "A1_minus_A0": table["A1"]["TP"] - table["A0"]["TP"],
            "A2_minus_A0": table["A2"]["TP"] - table["A0"]["TP"],
            "B1_minus_B0": table["B1"]["TP"] - table["B0"]["TP"],
            "B2_minus_B0": table["B2"]["TP"] - table["B0"]["TP"],
        },
        "HS_headroom": {
            c: min(table[c]["P"], table[c]["K"]) - table[c]["TP"] for c in ("A1", "B1")
        },
        "history_vs_none_recall": {
            "A1_minus_A0": (table["A1"]["TP"] - table["A0"]["TP"]) / p if p else None,
            "A2_minus_A0": (table["A2"]["TP"] - table["A0"]["TP"]) / p if p else None,
            "B1_minus_B0": (table["B1"]["TP"] - table["B0"]["TP"]) / p if p else None,
            "B2_minus_B0": (table["B2"]["TP"] - table["B0"]["TP"]) / p if p else None,
        },
    }
