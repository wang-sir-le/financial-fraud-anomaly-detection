import numpy as np
import pandas as pd
import pytest

from fraudx.models import PlattCalibrator, build_lightgbm_pipeline


def test_platt_calibration_preserves_score_order() -> None:
    scores = np.asarray([0.01, 0.1, 0.3, 0.7, 0.9])
    labels = np.asarray([0, 0, 1, 0, 1])
    calibrated = PlattCalibrator().fit(scores, labels).transform(scores)
    assert np.all(np.diff(calibrated) >= 0)


def test_model_builder_accepts_explicit_safe_features_only() -> None:
    frame = pd.DataFrame(
        {
            "step": [1, 2],
            "type": ["PAYMENT", "TRANSFER"],
            "amount": [10.0, 20.0],
            "isFraud": [0, 1],
        }
    )
    _, columns = build_lightgbm_pipeline(
        frame,
        seed=42,
        feature_columns=["step", "type", "amount"],
    )
    assert columns == ["step", "type", "amount"]
    with pytest.raises(ValueError, match="leakage"):
        build_lightgbm_pipeline(frame, seed=42, feature_columns=["isFraud"])
