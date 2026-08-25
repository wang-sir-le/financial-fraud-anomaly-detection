"""Acceptance tests for the frozen IEEE-CIS result-analysis phase."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraudx.ieee_cis_experiment import MODEL_ORDER, SEEDS, file_hash
from fraudx.ieee_cis_result_analysis import (
    ANALYSIS_OUTPUTS,
    COMPARISONS,
    _directional_classification,
    audit_frozen_result_structure,
    load_paysim_evidence,
    make_fixed_capacity_dependency_audit,
    make_fold_effect_summary,
    make_fold_equal_weight_summary,
    make_paired_effects,
    make_ranking_decision_alignment,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "ieee_cis" / "model_results"
PAYSIM_DIR = ROOT / "outputs" / "optimized_first_round" / "robustness"
PAIR_STATUS = (
    ROOT
    / "outputs"
    / "optimized_first_round"
    / "pair_audit"
    / "pair_audit_status.json"
)
SOURCE = ROOT / "src" / "fraudx" / "ieee_cis_result_analysis.py"
FROZEN_RESULTS_AVAILABLE = all(
    path.is_file()
    for path in (
        MODEL_DIR / "experiment_config.json",
        MODEL_DIR / "run_metadata.json",
        MODEL_DIR / "seed_metrics.csv",
        MODEL_DIR / "prediction_scores.parquet",
        PAYSIM_DIR / "rolling_multiseed_results.csv",
        PAIR_STATUS,
    )
)
requires_frozen_results = pytest.mark.skipif(
    not FROZEN_RESULTS_AVAILABLE,
    reason="requires locally regenerated frozen result assets that are not redistributed",
)


def _frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    config = json.loads((MODEL_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((MODEL_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(MODEL_DIR / "seed_metrics.csv")
    return config, metadata, metrics


@requires_frozen_results
def test_frozen_hashes_grid_roles_and_fit_count() -> None:
    config, metadata, metrics = _frozen_inputs()
    assert file_hash(MODEL_DIR / "experiment_config.json") == metadata["config_hash"]
    assert file_hash(MODEL_DIR / "prediction_scores.parquet") == metadata["output_hashes"][
        "prediction_scores.parquet"
    ]
    audit = audit_frozen_result_structure(config, metadata, metrics)
    assert audit["passed"].all()
    assert metadata["actual_official_fit_count"] == 60
    assert len(metrics) == 3 * 4 * 5
    assert set(metrics["model_id"]) == set(MODEL_ORDER)
    assert set(metrics["seed"]) == set(SEEDS)


@requires_frozen_results
def test_exact_comparisons_pairing_and_continuous_effects() -> None:
    _, _, metrics = _frozen_inputs()
    paired = make_paired_effects(metrics)
    expected = {
        "C01": ("M0", "M1"),
        "C12": ("M1", "M2"),
        "C23": ("M2", "M3"),
        "C02": ("M0", "M2"),
    }
    assert len(paired) == 60
    assert not paired[["comparison_id", "fold", "seed"]].duplicated().any()
    for comparison, baseline, candidate, _ in COMPARISONS:
        group = paired.loc[paired["comparison_id"] == comparison]
        assert expected[comparison] == (baseline, candidate)
        assert set(group["baseline_model"]) == {baseline}
        assert set(group["comparison_model"]) == {candidate}
        assert len(group) == 15
        assert group["alert_count_equal"].all()
    assert np.issubdtype(paired["delta_pr_auc"].dtype, np.floating)
    assert np.issubdtype(paired["delta_recall_at_3pct"].dtype, np.floating)


@requires_frozen_results
def test_fixed_capacity_identity_and_fold_equal_aggregation() -> None:
    _, _, metrics = _frozen_inputs()
    paired = make_paired_effects(metrics)
    dependency = make_fixed_capacity_dependency_audit(paired, metrics)
    assert len(dependency) == 60
    assert dependency["passed"].all()
    np.testing.assert_allclose(
        dependency["observed_cost_reduction"],
        101 * dependency["delta_tp"],
        rtol=0,
        atol=1e-12,
    )
    fold_summary = make_fold_effect_summary(paired)
    overall = make_fold_equal_weight_summary(fold_summary)
    assert len(fold_summary) == 12
    assert len(overall) == 4 * 8
    c02_cost = overall.loc[
        (overall["comparison_id"] == "C02") & (overall["metric"] == "cost_reduction")
    ].iloc[0]
    expected = np.mean(
        [c02_cost["fold1_effect"], c02_cost["fold2_effect"], c02_cost["fold3_effect"]]
    )
    assert c02_cost["equal_weight_overall_effect"] == expected
    assert c02_cost["strict_fold_signs"] == "negative/positive/positive"


@requires_frozen_results
def test_c02_expected_values_are_derived_from_frozen_metrics() -> None:
    _, _, metrics = _frozen_inputs()
    folds = make_fold_effect_summary(make_paired_effects(metrics))
    c02 = folds.loc[folds["comparison_id"] == "C02"].sort_values("fold")
    np.testing.assert_allclose(
        c02["delta_pr_auc_mean"],
        [-0.00386527, -0.00193796, 0.0188298],
        rtol=0,
        atol=5e-8,
    )
    np.testing.assert_allclose(
        c02["cost_reduction_mean"], [-1999.8, 161.6, 1878.6], rtol=0, atol=1e-12
    )


@requires_frozen_results
def test_ranking_decision_alignment_keeps_strict_nonzero_signs() -> None:
    _, _, metrics = _frozen_inputs()
    folds = make_fold_effect_summary(make_paired_effects(metrics))
    alignment = make_ranking_decision_alignment(folds)
    assert len(alignment) == 12
    assert set(alignment["strict_pr_auc_sign"]) <= {"positive", "zero", "negative"}
    assert set(alignment["strict_decision_sign"]) <= {"positive", "zero", "negative"}
    exact_zero_mismatch = (alignment["delta_pr_auc"] != 0) & (
        alignment["strict_pr_auc_sign"] == "zero"
    )
    assert not exact_zero_mismatch.any()


@requires_frozen_results
def test_paysim_evidence_is_loaded_from_frozen_files() -> None:
    evidence = load_paysim_evidence(PAYSIM_DIR, PAIR_STATUS)
    assert len(evidence["results"]) == 60
    assert len(evidence["pairwise"]) == 15
    assert set(evidence["effects"]) == {"C01", "C12", "C02"}
    for path in evidence["source_paths"]:
        assert evidence["source_hashes"][str(path.resolve())] == file_hash(path)
    assert evidence["pair_status"]["classification"] == (
        "B: Pair features degenerate under PaySim"
    )


def test_replication_classification_is_deterministic_and_mixed_is_preserved() -> None:
    cases = [
        ([1.0, 2.0, 3.0], [0.5, 0.4, 0.3]),
        ([1.0, 2.0, 3.0], [-0.5, -0.4, -0.3]),
        ([1.0, -2.0, 3.0], [-0.5, 0.4, -0.3]),
        ([1.0, 2.0, 3.0], [-0.5, 0.4, 0.3]),
    ]
    first = [_directional_classification(left, right) for left, right in cases]
    second = [_directional_classification(left, right) for left, right in cases]
    assert first == second
    assert first == [
        "Replicated",
        "Not Replicated",
        "Dataset-Specific / Time-Dependent",
        "Partially Replicated",
    ]


def test_analysis_source_contains_no_model_execution_or_forbidden_inference() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    invoked_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {"fit", "predict", "predict_proba"} & invoked_attributes
    assert "significant temporal heterogeneity" not in source.lower()
    assert "statistically significant" not in source.lower()
    assert "abs(delta)" not in source.lower()


def test_output_contract_avoids_self_referential_checksum() -> None:
    assert len(ANALYSIS_OUTPUTS) == 14
    source = SOURCE.read_text(encoding="utf-8")
    assert 'name != "checksums.sha256"' in source
    assert "output_hashes_excluding_metadata_and_checksum_manifest" in source
    assert '"overall_external_validation_pass_fail": False' in source
