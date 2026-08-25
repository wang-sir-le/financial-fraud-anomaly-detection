import numpy as np
import pandas as pd

from fraudx.features import build_behavioral_features, feature_columns_for_group
from fraudx.models import build_lightgbm_pipeline
from fraudx.pair_audit import (
    AuditModelArtifacts,
    build_feature_audit,
    feature_distribution_stats,
    feature_importance_rows,
    model_inclusion_rows,
    pair_repetition_profile,
    prediction_comparison_row,
)
from fraudx.synthetic import make_synthetic_paysim


def test_pair_key_and_temporal_window_exclude_current_and_future_rows() -> None:
    frame = pd.DataFrame(
        {
            "step": [1, 2, 2, 3, 4],
            "type": ["TRANSFER"] * 5,
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            "nameOrig": ["A"] * 5,
            "nameDest": ["X"] * 5,
            "isFraud": [0, 0, 1, 0, 1],
        }
    )
    through_step_three = build_behavioral_features(frame.iloc[:4], windows=(1, 6))
    with_future = build_behavioral_features(frame, windows=(1, 6)).iloc[:4]
    assert through_step_three[["pair_count_1h", "pair_count_6h"]].equals(
        with_future[["pair_count_1h", "pair_count_6h"]]
    )
    same_step = with_future.loc[with_future["step"] == 2]
    assert (same_step["pair_count_1h"] == 1).all()
    step_three = with_future.loc[with_future["step"] == 3].iloc[0]
    assert step_three["pair_count_1h"] == 2
    assert step_three["pair_count_6h"] == 3


def test_pair_feature_list_adds_only_expected_fields() -> None:
    behavior = build_behavioral_features(make_synthetic_paysim(rows=500), windows=(1, 6, 24))
    recipient = feature_columns_for_group(behavior, "recipient_full")
    pair = feature_columns_for_group(behavior, "recipient_full_with_pair")
    added = sorted(set(pair) - set(recipient))
    assert added == ["pair_count_1h", "pair_count_24h", "pair_count_6h"]
    audit = build_feature_audit(recipient, pair)
    pair_rows = audit.loc[audit["is_pair_feature"]]
    assert len(pair_rows) == 3
    assert not pair_rows["recipient_full_contains"].any()
    assert pair_rows["recipient_full_with_pair_contains"].all()


def test_pair_repetition_and_distribution_statistics_are_correct() -> None:
    frame = pd.DataFrame(
        {
            "nameOrig": ["A", "A", "A", "B"],
            "nameDest": ["X", "X", "Y", "X"],
        }
    )
    profile = pair_repetition_profile(frame).iloc[0]
    assert profile["unique_pairs"] == 3
    assert profile["repeated_pair_count"] == 1
    assert profile["repeated_pair_transaction_count"] == 2
    assert profile["maximum_pair_frequency"] == 2
    stats = feature_distribution_stats(pd.Series([0, 0, 1, 2]))
    assert stats["zero_count"] == 2
    assert stats["nonzero_count"] == 2
    assert stats["unique_count"] == 3
    assert stats["probability_zero"] == 0.5
    assert stats["probability_one"] == 0.25
    assert stats["probability_greater_than_one"] == 0.25


def test_model_inclusion_and_importance_extraction_track_pair_features() -> None:
    behavior = build_behavioral_features(make_synthetic_paysim(rows=600), windows=(1, 6, 24))
    columns = feature_columns_for_group(behavior, "recipient_full_with_pair")
    pair_features = [column for column in columns if column.startswith("pair_count_")]
    model, columns = build_lightgbm_pipeline(
        behavior,
        seed=42,
        n_estimators=10,
        feature_columns=columns,
    )
    model.fit(behavior[columns], behavior["isFraud"])
    inclusion = pd.DataFrame(
        model_inclusion_rows(
            fold=1,
            model=model,
            feature_columns=columns,
            train=behavior,
            pair_features=pair_features,
        )
    )
    assert len(inclusion) == 3
    assert inclusion[
        [
            "dataframe_present",
            "preprocessing_present",
            "transformed_present",
            "model_feature_present",
        ]
    ].all().all()
    importance = pd.DataFrame(
        feature_importance_rows(fold=1, model=model, pair_features=pair_features)
    )
    assert len(importance) == 3
    assert set(importance.columns) >= {"split_count", "gain", "gain_fraction"}


def test_prediction_comparison_reports_exact_and_decision_differences() -> None:
    behavior = build_behavioral_features(make_synthetic_paysim(rows=300), windows=(1,))
    columns = feature_columns_for_group(behavior, "raw_safe")
    model, columns = build_lightgbm_pipeline(
        behavior,
        seed=42,
        n_estimators=5,
        feature_columns=columns,
    )
    model.fit(behavior[columns], behavior["isFraud"])
    recipient = AuditModelArtifacts(
        model=model,
        feature_columns=columns,
        raw_test_scores=np.array([0.1, 0.9]),
        calibrated_test_scores=np.array([0.2, 0.8]),
        threshold=0.5,
        decisions=np.array([0, 1]),
    )
    pair = AuditModelArtifacts(
        model=model,
        feature_columns=columns,
        raw_test_scores=np.array([0.1, 0.7]),
        calibrated_test_scores=np.array([0.2, 0.4]),
        threshold=0.5,
        decisions=np.array([0, 0]),
    )
    comparison = prediction_comparison_row(fold=1, recipient=recipient, pair=pair)
    assert comparison["raw_number_different"] == 1
    assert comparison["calibrated_number_different"] == 1
    assert comparison["decision_disagreement_count"] == 1
