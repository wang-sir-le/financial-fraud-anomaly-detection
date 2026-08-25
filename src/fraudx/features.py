"""Causal temporal features computed with bounded SQL range windows."""

from __future__ import annotations

import re
from collections.abc import Sequence

import duckdb
import numpy as np
import pandas as pd

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

BASE_FEATURES = ("step", "type", "amount")
TEMPORAL_CONTEXT_FEATURES = ("hour_of_day", "day_index")


def feature_columns_for_group(frame: pd.DataFrame, group: str) -> list[str]:
    """Return an ordered, leakage-safe feature subset for an ablation group."""
    available = set(frame.columns)
    base = list(BASE_FEATURES)
    temporal = base + [
        column for column in TEMPORAL_CONTEXT_FEATURES if column in available
    ]
    frequency = temporal + _matching_columns(
        frame,
        ("time_since_last_dest", "dest_count_", "unique_origins_"),
    )
    amount = temporal + _matching_columns(
        frame,
        (
            "dest_amount_mean_",
            "dest_amount_max_",
            "dest_amount_std_",
            "dest_amount_ratio_",
        ),
    )
    full = list(dict.fromkeys(frequency + amount))
    groups = {
        "raw_safe": base,
        "temporal_context": temporal,
        "recipient_frequency": frequency,
        "recipient_amount": amount,
        "recipient_full": full,
        "recipient_full_with_pair": full + _matching_columns(frame, ("pair_count_",)),
    }
    if group not in groups:
        raise ValueError(f"Unknown feature group: {group}")
    columns = groups[group]
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"Feature group {group} is missing columns: {missing}")
    return columns


def _matching_columns(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [column for column in frame.columns if column.startswith(prefixes)]


def build_behavioral_features(
    frame: pd.DataFrame,
    windows: Sequence[int] = (1, 6, 24),
) -> pd.DataFrame:
    """Build recipient and transaction-pair history from strictly earlier steps.

    SQL RANGE frames end at one step before the current step. Transactions sharing
    the current step therefore cannot leak into each other's feature values.
    """
    clean_windows = tuple(sorted({int(window) for window in windows}))
    if not clean_windows or clean_windows[0] <= 0:
        raise ValueError("All windows must be positive")
    required = {"step", "nameOrig", "nameDest", "amount"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("transactions", frame.reset_index(names="_row_id"))
        expressions: list[str] = []
        for window in clean_windows:
            expressions.extend(_window_expressions(window))
        # SQL fragments are generated only from validated positive integers; no user
        # string is interpolated into identifiers or predicates.
        query = f"""
            WITH history AS (
                SELECT
                    *,
                    {", ".join(expressions)},
                    MAX(step) OVER (
                        PARTITION BY nameDest ORDER BY step
                        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    ) AS previous_dest_step
                FROM transactions
            )
            SELECT
                * EXCLUDE (previous_dest_step),
                COALESCE(step - previous_dest_step, -1) AS time_since_last_dest,
                ((step - 1) % 24)::INTEGER AS hour_of_day,
                FLOOR((step - 1) / 24)::INTEGER AS day_index
            FROM history
            ORDER BY _row_id
        """  # nosec B608
        result = connection.execute(query).df().drop(columns="_row_id")
    finally:
        connection.close()
    for window in clean_windows:
        mean_column = f"dest_amount_mean_{window}h"
        result[f"dest_amount_ratio_{window}h"] = np.divide(
            result["amount"],
            result[mean_column] + 1e-6,
        )
    numeric = result.select_dtypes(include=["number"]).columns
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result


def _window_expressions(window: int) -> list[str]:
    if window <= 0:
        raise ValueError("Window must be positive")
    suffix = f"{window}h"
    frame = f"RANGE BETWEEN {window} PRECEDING AND 1 PRECEDING"
    return [
        f"COUNT(*) OVER (PARTITION BY nameDest ORDER BY step {frame}) AS dest_count_{suffix}",
        "AVG(amount) OVER "
        f"(PARTITION BY nameDest ORDER BY step {frame}) AS dest_amount_mean_{suffix}",
        "MAX(amount) OVER "
        f"(PARTITION BY nameDest ORDER BY step {frame}) AS dest_amount_max_{suffix}",
        "STDDEV_SAMP(amount) OVER "
        f"(PARTITION BY nameDest ORDER BY step {frame}) AS dest_amount_std_{suffix}",
        "COUNT(DISTINCT nameOrig) OVER "
        f"(PARTITION BY nameDest ORDER BY step {frame}) AS unique_origins_{suffix}",
        "COUNT(*) OVER "
        f"(PARTITION BY nameOrig, nameDest ORDER BY step {frame}) AS pair_count_{suffix}",
    ]
