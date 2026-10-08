from __future__ import annotations

from typing import Any

import lightgbm
import pytest
import xgboost


@pytest.fixture(autouse=True)
def forbid_real_model_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Real model fitting forbidden in pre-freeze verification")
    monkeypatch.setattr(lightgbm.LGBMModel, "fit", forbidden)
    monkeypatch.setattr(lightgbm.LGBMClassifier, "fit", forbidden)
    monkeypatch.setattr(lightgbm, "train", forbidden)
    monkeypatch.setattr(xgboost.XGBModel, "fit", forbidden)
    monkeypatch.setattr(xgboost.XGBClassifier, "fit", forbidden)
    monkeypatch.setattr(xgboost, "train", forbidden)
    monkeypatch.setattr(xgboost.Booster, "update", forbidden)
