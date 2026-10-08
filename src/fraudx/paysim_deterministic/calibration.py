"""Recompute all original calibration diagnostics from the new frozen score grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from fraudx.calibration import (
    IdentityCalibrator,
    _compare_decisions,
    _compare_reliability,
    _evaluate_bayes_branch,
    _evaluate_branch,
)
from fraudx.calibration_reporting import summarize_calibration, validate_calibration_outputs
from fraudx.models import PlattCalibrator
from fraudx.paysim_deterministic.analysis import context
from fraudx.q2_extension.common import (
    FOLDS,
    SEEDS,
    array_hash,
    load_frame,
    prediction_path,
    read_json,
    require,
)


def run(config: Path) -> None:
    ctx = context(config)
    folder = ctx.start("calibration")
    results, curves, invariance, bayes, pairs, constants = [], [], [], [], [], []
    for fold in FOLDS:
        for model, original, feature, weight in [("P0", "raw_safe", "raw_safe", 1.),
                ("PTHW", "recipient_full_weight_2", "recipient_full", 2.)]:
            for seed in SEEDS:
                val = load_frame(ctx, "paysim", fold, model, seed, "validation")
                test = load_frame(ctx, "paysim", fold, model, seed)
                mask = val.calibration_part.to_numpy(bool)
                path = prediction_path(ctx, "paysim", fold, model, seed)
                meta = read_json(path.with_name(path.name.replace("_test.parquet", ".json")))
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0., y_max=1.)
                iso.fit(val.raw.to_numpy()[mask], val.isFraud.to_numpy()[mask])
                require(np.array_equal(iso.transform(test.raw), test.isotonic.to_numpy()),
                        "Isotonic score reconstruction mismatch")
                calibrators: dict[str, Any] = {"Uncalibrated": IdentityCalibrator(),
                    "Platt": PlattCalibrator(**meta["calibrator"]), "Isotonic": iso}
                branches = {}
                for method, transform in calibrators.items():
                    vr, tr = val.raw.to_numpy(), test.raw.to_numpy()
                    vs, ts = transform.transform(vr), transform.transform(tr)
                    branch, curve = _evaluate_branch(fold=fold, scheme=("Raw" if model == "P0" else "Recipient"),
                        model_id=original, feature_group=feature, weight=weight, seed=seed, method=method,
                        calibration_labels=val.isFraud.to_numpy()[mask], calibration_raw=vr[mask],
                        calibration_scores=vs[mask], validation_labels=val.isFraud.to_numpy()[~mask],
                        validation_raw=vr[~mask], validation_scores=vs[~mask], test_labels=test.isFraud.to_numpy(),
                        test_raw=tr, test_scores=ts, raw_hash=array_hash(tr), calibrator=transform,
                        fn_cost=100., fp_cost=1., capacity=.03, bins=10,
                        boundaries=ctx.config["fold_boundaries"][fold - 1])
                    if method == "Platt":
                        primary = next(r for r in meta["rows"] if r["capacity"] == .03 and r["fn_cost"] == 100)
                        require(branch.result["tp"] == primary["tp"] and branch.result["fp"] == primary["fp"],
                                "Calibration diagnostic decision differs from primary fit")
                    branches[method] = branch
                    results.append(branch.result)
                    curves.extend(curve)
                    bayes.append(_evaluate_bayes_branch(fold=fold, scheme=branch.result["scheme"],
                        model_id=original, feature_group=feature, weight=weight, seed=seed, method=method,
                        labels=test.isFraud.to_numpy(), scores=ts, raw_scores=tr, bayes_threshold=1/101,
                        capacity=.03, fn_cost=100., fp_cost=1., stable_indices=test.source_row_id.to_numpy()))
                invariance.extend(_compare_decisions(fold, original, seed, branches))
                pairs.extend(_compare_reliability(fold, original, seed, branches))
                if model == "PTHW":
                    p0, pi = float(val.loc[mask, "isFraud"].mean()), float(test.isFraud.mean())
                    constants.append({"fold": fold, "seed": seed, "constant_probability": p0,
                        "constant_brier": pi*(1-p0)**2+(1-pi)*p0**2,
                        "constant_log_loss": -pi*np.log(p0)-(1-pi)*np.log1p(-p0),
                        "platt_brier": branches["Platt"].result["test_brier"],
                        "platt_log_loss": branches["Platt"].result["test_log_loss"]})
                print(f"Calibration diagnostics fold={fold} model={model} seed={seed}", flush=True)
    result, curve_frame = pd.DataFrame(results), pd.concat(curves, ignore_index=True)
    decision_frame, bayes_frame = pd.DataFrame(invariance), pd.DataFrame(bayes)
    checks = validate_calibration_outputs(result, curve_frame, decision_frame, bayes_frame,
                                          expected_model_trainings=30, expected_branch_rows=90)
    for filename, data in {
        "calibration_ablation_results": result, "calibration_ablation_summary": summarize_calibration(result),
        "calibration_curve_points": curve_frame, "calibration_decision_invariance": decision_frame,
        "calibration_bayes_decision_results": bayes_frame, "calibration_pairwise_reliability": pd.DataFrame(pairs),
        "constant_reference_seed": pd.DataFrame(constants),
        "constant_reference_fold": pd.DataFrame(constants).groupby("fold", as_index=False).mean(numeric_only=True),
    }.items():
        data.to_csv(folder / f"{filename}.csv", index=False)
    ctx.finish(folder, {"branches": len(result), "detector_refits": 0, "checks": checks,
                        "isotonic_parameter_replay": "30 exact score checks"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
