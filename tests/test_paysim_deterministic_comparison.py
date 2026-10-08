"""Comparison ledgers must preserve estimand keys and incomplete inventories."""

import pandas as pd
import pytest

from fraudx.paysim_deterministic.comparison import table_difference


def test_interval_estimand_key_is_preserved() -> None:
    old = pd.DataFrame({"fold": [1, 1], "metric": ["delta_tp", "delta_ap"], "estimate": [2., -.1]})
    new = old.assign(estimate=[-1., .2])
    result = table_difference(old, new, ["fold", "metric"], "intervals")
    assert result.metric.tolist() == ["delta_ap", "delta_tp"]
    assert result.field.tolist() == ["estimate", "estimate"]
    assert result.v2_minus_v1.tolist() == pytest.approx([.3, -3.])


def test_incomplete_version_grid_is_rejected() -> None:
    old = pd.DataFrame({"fold": [1, 2], "estimate": [1., 2.]})
    with pytest.raises(ValueError, match="Unmatched"):
        table_difference(old, old.iloc[:1], ["fold"], "primary")
