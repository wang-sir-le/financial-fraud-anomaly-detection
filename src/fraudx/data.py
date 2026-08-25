"""PaySim schema checks and leakage-resistant temporal splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}


@dataclass(frozen=True)
class TemporalSplit:
    """Three ordered, non-overlapping temporal partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def rolling_temporal_splits(
    frame: pd.DataFrame,
    folds: int = 3,
    initial_train_fraction: float = 0.4,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    time_column: str = "step",
) -> list[TemporalSplit]:
    """Create expanding-window walk-forward folds on complete time steps.

    The first fold uses the requested initial training, validation, and test
    fractions. Later folds move forward at equal intervals; observations from
    earlier validation/test windows may then enter the expanding training set.
    """
    if folds < 2:
        raise ValueError("Rolling validation requires at least two folds")
    fractions = (initial_train_fraction, validation_fraction, test_fraction)
    if any(value <= 0 for value in fractions):
        raise ValueError("Rolling split fractions must be positive")
    first_end = sum(fractions)
    if first_end > 1:
        raise ValueError("Initial rolling split fractions cannot exceed one")
    times = pd.Index(sorted(frame[time_column].unique()))
    if len(times) < folds * 3:
        raise ValueError("Not enough distinct time steps for rolling validation")
    counts = frame.groupby(time_column, sort=True).size().reindex(times).to_numpy()
    cumulative_rows = np.cumsum(counts)
    shift = (1 - first_end) / (folds - 1)
    ordered = frame.sort_values(time_column, kind="stable").reset_index(drop=True)
    splits: list[TemporalSplit] = []
    for fold in range(folds):
        train_end_fraction = initial_train_fraction + fold * shift
        validation_end_fraction = train_end_fraction + validation_fraction
        test_end_fraction = min(validation_end_fraction + test_fraction, 1.0)
        train_end = _nearest_group_boundary(
            cumulative_rows,
            len(frame) * train_end_fraction,
            minimum=1,
            maximum=len(times) - 2,
        )
        validation_end = _nearest_group_boundary(
            cumulative_rows,
            len(frame) * validation_end_fraction,
            minimum=train_end + 1,
            maximum=len(times) - 1,
        )
        test_end = _nearest_group_boundary(
            cumulative_rows,
            len(frame) * test_end_fraction,
            minimum=validation_end + 1,
            maximum=len(times),
        )
        split = TemporalSplit(
            train=ordered[ordered[time_column].isin(times[:train_end])].copy(),
            validation=ordered[
                ordered[time_column].isin(times[train_end:validation_end])
            ].copy(),
            test=ordered[
                ordered[time_column].isin(times[validation_end:test_end])
            ].copy(),
        )
        _assert_temporal_order(split, time_column)
        splits.append(split)
    return splits


def validation_period_masks(
    frame: pd.DataFrame,
    time_column: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Split validation chronologically into calibration and threshold periods."""
    times = np.asarray(sorted(frame[time_column].unique()))
    if len(times) < 4:
        raise ValueError("Validation needs at least four time steps")
    counts = frame.groupby(time_column, sort=True).size().reindex(times).to_numpy()
    cumulative = np.cumsum(counts)
    boundaries = np.arange(1, len(times), dtype=int)
    ordered_boundaries = boundaries[
        np.argsort(np.abs(cumulative[boundaries - 1] - len(frame) / 2))
    ]
    for boundary in ordered_boundaries:
        calibration = frame[time_column].isin(times[:boundary]).to_numpy()
        threshold = ~calibration
        if (
            frame.loc[calibration, target].nunique() == 2
            and frame.loc[threshold, target].nunique() == 2
        ):
            return calibration, threshold
    raise ValueError("Validation cannot be split into two temporal intervals with both classes")


def validate_paysim(frame: pd.DataFrame) -> None:
    """Fail fast on schema, label, time, and numerical integrity problems."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing PaySim columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("PaySim data is empty")
    labels = set(frame["isFraud"].dropna().unique())
    if not labels.issubset({0, 1}) or len(labels) < 2:
        raise ValueError("isFraud must contain both binary classes")
    if frame["step"].isna().any() or (frame["step"] < 0).any():
        raise ValueError("step must be non-negative and non-null")
    if frame["amount"].isna().any() or (frame["amount"] < 0).any():
        raise ValueError("amount must be non-negative and non-null")


def temporal_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    time_column: str = "step",
    split_mode: str = "time_fraction",
) -> TemporalSplit:
    """Create ordered partitions without allowing one timestamp to cross a boundary.

    ``row_fraction`` places boundaries near the requested cumulative row fractions.
    ``time_fraction`` keeps equal fractions of distinct timestamps and is useful as a
    stronger late-period drift stress test.
    """
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Temporal test split must be non-empty")
    times = pd.Index(sorted(frame[time_column].unique()))
    if len(times) < 5:
        raise ValueError("At least five distinct time steps are required")
    if split_mode == "time_fraction":
        train_end_index = max(1, int(len(times) * train_fraction))
        validation_end_index = max(
            train_end_index + 1,
            int(len(times) * (train_fraction + validation_fraction)),
        )
        validation_end_index = min(validation_end_index, len(times) - 1)
    elif split_mode == "row_fraction":
        counts = frame.groupby(time_column, sort=True).size().reindex(times).to_numpy()
        cumulative_rows = np.cumsum(counts)
        train_end_index = _nearest_group_boundary(
            cumulative_rows,
            target_rows=len(frame) * train_fraction,
            minimum=1,
            maximum=len(times) - 2,
        )
        validation_end_index = _nearest_group_boundary(
            cumulative_rows,
            target_rows=len(frame) * (train_fraction + validation_fraction),
            minimum=train_end_index + 1,
            maximum=len(times) - 1,
        )
    else:
        raise ValueError("split_mode must be 'row_fraction' or 'time_fraction'")
    train_times = set(times[:train_end_index])
    validation_times = set(times[train_end_index:validation_end_index])
    test_times = set(times[validation_end_index:])
    ordered = frame.sort_values(time_column, kind="stable").reset_index(drop=True)
    split = TemporalSplit(
        train=ordered[ordered[time_column].isin(train_times)].copy(),
        validation=ordered[ordered[time_column].isin(validation_times)].copy(),
        test=ordered[ordered[time_column].isin(test_times)].copy(),
    )
    _assert_temporal_order(split, time_column)
    return split


def _nearest_group_boundary(
    cumulative_rows: np.ndarray,
    target_rows: float,
    minimum: int,
    maximum: int,
) -> int:
    """Return a group-count boundary nearest to a cumulative row target."""
    candidates = np.arange(minimum, maximum + 1, dtype=int)
    distances = np.abs(cumulative_rows[candidates - 1] - target_rows)
    return int(candidates[np.argmin(distances)])


def _assert_temporal_order(split: TemporalSplit, time_column: str) -> None:
    if split.train.empty or split.validation.empty or split.test.empty:
        raise ValueError("All temporal partitions must be non-empty")
    if split.train[time_column].max() >= split.validation[time_column].min():
        raise AssertionError("Training and validation times overlap")
    if split.validation[time_column].max() >= split.test[time_column].min():
        raise AssertionError("Validation and test times overlap")
