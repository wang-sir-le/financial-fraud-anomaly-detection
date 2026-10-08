"""Transparent per-column train-only transforms, serialized without pickle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import CATEGORICAL, CELL_COLUMNS, require

MISSING = "__BANKSIM_MISSING_V1__"


def category_values(series: pd.Series) -> np.ndarray:
    vals = series.to_numpy(dtype=object).copy()
    require(not bool(np.any(vals == MISSING)), "Reserved category marker occurs in source")
    vals[pd.isna(vals) | (vals == "")] = MISSING
    require(all(isinstance(x, str) for x in vals), "Category must be an unmodified string")
    return vals


class Preprocessor:
    def __init__(self, columns: list[str]) -> None:
        self.columns = list(columns)
        self.rules: list[dict[str, Any]] = []
        self.fitted = False

    def fit(self, train: pd.DataFrame) -> Preprocessor:
        require(not train.empty and train["split"].eq("train").all(), "Fit accepts training rows only")
        self.rules = []
        for column in self.columns:
            if column in CATEGORICAL:
                values = category_values(train[column])
                vocab = sorted(set(values))
                # Remove constant output dummy columns, not the source before encoding.
                keep = [v for v in vocab if not bool(np.all(values == v))]
                self.rules.append({"column": column, "kind": "categorical", "vocabulary": vocab, "keep": keep})
            else:
                values = train[column].to_numpy(dtype=np.float64)
                require(not bool(np.isinf(values).any()), "Infinite numeric input")
                missing = np.isnan(values)
                median = float(np.median(values[~missing])) if (~missing).any() else 0.0
                filled = np.where(missing, median, values).astype(np.float32)
                require(bool(np.isfinite(filled).all()), "float32 overflow")
                self.rules.append({"column": column, "kind": "numeric", "median": median, "keep_value": not bool(np.all(filled == filled[0])), "keep_missing": not bool(np.all(missing == missing[0]))})
        self.fitted = True
        return self

    def feature_names(self) -> list[str]:
        require(self.fitted, "Preprocessor not fitted")
        names = []
        for rule in self.rules:
            c = rule["column"]
            if rule["kind"] == "categorical":
                names += [f"cat:{c}={value}" for value in rule["keep"]]
            else:
                if rule["keep_value"]:
                    names.append(f"num:{c}")
                if rule["keep_missing"]:
                    names.append(f"missing:{c}")
        require(len(names) == len(set(names)), "Encoded feature names collide")
        return names

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        require(self.fitted, "Preprocessor not fitted")
        blocks: list[np.ndarray] = []
        for rule in self.rules:
            column = rule["column"]
            if rule["kind"] == "categorical":
                values = category_values(frame[column])
                blocks.extend((values == value).astype(np.float32) for value in rule["keep"])
            else:
                values = frame[column].to_numpy(dtype=np.float64)
                require(not bool(np.isinf(values).any()), "Infinite numeric input")
                missing = np.isnan(values)
                if rule["keep_value"]:
                    blocks.append(np.where(missing, rule["median"], values).astype(np.float32))
                if rule["keep_missing"]:
                    blocks.append(missing.astype(np.float32))
        result = np.column_stack(blocks).astype(np.float32) if blocks else np.empty((len(frame), 0), dtype=np.float32)
        require(bool(np.isfinite(result).all()), "Transformed matrix contains nonfinite values")
        return result

    def to_dict(self) -> dict[str, Any]:
        require(self.fitted, "Preprocessor not fitted")
        return {"columns": self.columns, "rules": self.rules, "names": self.feature_names(), "fit_split": "train", "dtype": "float32", "missing_token": MISSING}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Preprocessor:
        result = cls(value["columns"])
        result.rules = value["rules"]
        result.fitted = True
        require(result.feature_names() == value["names"], "Preprocessor schema changed")
        return result


def fit_cells(train: pd.DataFrame) -> dict[str, Preprocessor]:
    return {cell: Preprocessor(columns).fit(train) for cell, columns in CELL_COLUMNS.items()}


def check_common_rules(processors: dict[str, Preprocessor]) -> None:
    known: dict[str, dict[str, Any]] = {}
    for proc in processors.values():
        for rule in proc.rules:
            c = rule["column"]
            if c in known:
                require(rule == known[c], f"Common transform differs: {c}")
            known[c] = rule
