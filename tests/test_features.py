import pandas as pd
from pandas.testing import assert_frame_equal

from fraudx.features import build_behavioral_features


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 2, 2, 3],
            "type": ["PAYMENT"] * 4,
            "amount": [10.0, 20.0, 30.0, 40.0],
            "nameOrig": ["A", "A", "A", "A"],
            "oldbalanceOrg": [100.0] * 4,
            "newbalanceOrig": [90.0, 80.0, 70.0, 60.0],
            "nameDest": ["X", "X", "X", "Z"],
            "oldbalanceDest": [0.0] * 4,
            "newbalanceDest": [10.0, 20.0, 30.0, 40.0],
            "isFraud": [0, 0, 1, 1],
            "isFlaggedFraud": [0] * 4,
        }
    )


def test_same_step_and_future_rows_do_not_leak() -> None:
    original = _rows()
    before = build_behavioral_features(original.iloc[:3], windows=(1, 24))
    after = build_behavioral_features(original, windows=(1, 24)).iloc[:3]
    feature_columns = [column for column in before.columns if column not in original.columns]
    assert_frame_equal(before[feature_columns], after[feature_columns], check_dtype=False)
    same_step = after[after["step"] == 2]
    assert same_step["dest_count_1h"].nunique() == 1
    assert same_step["dest_count_1h"].iloc[0] == 1
    assert "origin_balance_error" not in after.columns
    assert "dest_balance_error" not in after.columns
