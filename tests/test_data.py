import numpy as np
import pandas as pd
import pytest

from fraudx.data import rolling_temporal_splits, temporal_split, validate_paysim
from fraudx.synthetic import make_synthetic_paysim


def test_schema_and_temporal_split_are_ordered() -> None:
    frame = make_synthetic_paysim(rows=500, steps=20)
    validate_paysim(frame)
    split = temporal_split(frame)
    assert split.train["step"].max() < split.validation["step"].min()
    assert split.validation["step"].max() < split.test["step"].min()


def test_schema_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing PaySim columns"):
        validate_paysim(pd.DataFrame({"step": [1], "isFraud": [0]}))


def test_row_fraction_split_is_near_requested_size_without_crossing_steps() -> None:
    counts = np.array([5, 10, 20, 30, 45, 60, 80, 100, 130, 160])
    frame = pd.DataFrame({"step": np.repeat(np.arange(len(counts)), counts)})
    split = temporal_split(frame, split_mode="row_fraction")
    total = len(frame)
    largest_step_fraction = counts.max() / total
    assert abs(len(split.train) / total - 0.6) <= largest_step_fraction
    assert abs(len(split.validation) / total - 0.2) <= largest_step_fraction
    assert split.train["step"].max() < split.validation["step"].min()
    assert split.validation["step"].max() < split.test["step"].min()


def test_rolling_splits_expand_training_and_preserve_order() -> None:
    frame = make_synthetic_paysim(rows=1200, steps=60)
    splits = rolling_temporal_splits(frame, folds=3)
    assert len(splits) == 3
    assert len(splits[0].train) < len(splits[1].train) < len(splits[2].train)
    for split in splits:
        assert split.train["step"].max() < split.validation["step"].min()
        assert split.validation["step"].max() < split.test["step"].min()
