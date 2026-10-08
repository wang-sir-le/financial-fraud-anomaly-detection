"""Full frozen PaySim grid, with source-identity ledgers and safe resumption."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_curve

from fraudx.data import validation_period_masks
from fraudx.models import PlattCalibrator, build_lightgbm_pipeline
from fraudx.paysim_deterministic.preprocess import file_hash
from fraudx.q2_extension.common import array_hash, require, write_json


def decision_metrics(labels: Any, decisions: Any, cost: int) -> dict[str, float | int]:
    y, d = np.asarray(labels, dtype=np.int64), np.asarray(decisions, dtype=bool)
    tp, alerts, frauds = int(y[d].sum()), int(d.sum()), int(y.sum())
    fp, fn = alerts - tp, frauds - tp
    return {"n": len(y), "frauds": frauds, "tp": tp, "fp": fp, "fn": fn,
            "alerts": alerts, "recall": tp / frauds,
            "precision": tp / alerts if alerts else 0, "alert_rate": alerts / len(y),
            "cost": fp + cost * fn, "cost_per_10000": (fp + cost * fn) * 10000 / len(y)}


def threshold_grid(labels: Any, scores: Any, capacities: list[float],
                   costs: list[int]) -> dict[tuple[float, int], dict[str, float]]:
    """Identical exact-integer objective/tie rules to the frozen 2026-09-20 correction."""
    y, s = np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=float)
    require(bool(np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()), "Bad scores")
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    tp = np.rint(tpr * y.sum()).astype(np.int64)
    fp = np.rint(fpr * (len(y) - y.sum())).astype(np.int64)
    fn, alerts = y.sum() - tp, tp + fp
    output = {}
    for q in capacities:
        eligible = np.flatnonzero(alerts / len(y) <= q)
        for c in costs:
            objective = fp + c * fn
            tied = eligible[objective[eligible] == objective[eligible].min()]
            tied = tied[tp[tied] == tp[tied].max()]
            tied = tied[alerts[tied] == alerts[tied].min()]
            finite = tied[np.isfinite(thresholds[tied])]
            idx = int(finite[np.argmax(thresholds[finite])] if finite.size else tied[0])
            threshold = float(thresholds[idx])
            if not np.isfinite(threshold):
                threshold = float(np.nextafter(s.max(), np.inf))
            output[(q, c)] = {"threshold": threshold,
                              "validation_alert_rate": float(alerts[idx] / len(y))}
    return output


def source_fingerprint() -> dict[str, str]:
    import fraudx.data
    import fraudx.models
    return {p.name: file_hash(p) for p in [Path(__file__), Path(fraudx.data.__file__),
                                         Path(fraudx.models.__file__)]}


def train(protocol: Path, features: Path, destination: Path) -> None:
    config = json.loads(protocol.read_text(encoding="utf-8"))
    signature = {"protocol_sha256": file_hash(protocol), "features_sha256": file_hash(features),
                 "code": source_fingerprint()}
    destination.mkdir(parents=True, exist_ok=True)
    status = destination / "TRAINING_INPUTS.json"
    if status.exists():
        require(json.loads(status.read_text()) == signature, "Resume inputs/code changed")
    else:
        require(not list(destination.iterdir()), "Nonempty unsigned model destination")
        write_json(status, signature)
    frame = pd.read_parquet(features)
    require(not frame.source_row_id.duplicated().any(), "Duplicate source identity")
    require(frame[["step", "source_row_id"]].equals(frame[["step", "source_row_id"]]
            .sort_values(["step", "source_row_id"], kind="stable")), "Noncanonical input order")
    all_rows = []
    for boundary in config["folds"]:
        fold = boundary["fold"]
        train_data = frame[frame.step.between(boundary["train_min_step"], boundary["train_max_step"])]
        val = frame[frame.step.between(boundary["validation_min_step"], boundary["validation_max_step"])]
        test = frame[frame.step.between(boundary["test_min_step"], boundary["test_max_step"])]
        calmask, thmask = validation_period_masks(val, "step", "isFraud")
        require(int(val.loc[calmask, "step"].max()) == config["calibration_max_steps"][fold - 1],
                "Calibration boundary drift")
        for name, specification in config["models"].items():
            columns = specification["columns"]
            require(not {"source_row_id", "row_id"}.intersection(columns), "Identity in features")
            for seed in config["seeds"]:
                prefix = f"paysim_{fold}_{name}_{seed}"
                checkpoint = destination / (prefix + ".json")
                if checkpoint.exists():
                    meta = json.loads(checkpoint.read_text())
                    require(meta["input_signature"] == signature, "Checkpoint signature mismatch")
                    for item in meta["files"]:
                        require(file_hash(destination / item["name"]) == item["sha256"],
                                "Checkpoint artifact changed")
                    all_rows.extend(meta["rows"])
                    print(f"Verified completed {prefix}", flush=True)
                    continue
                require(not list(destination.glob(prefix + "_*")), "Preserve incomplete fit artifacts")
                started = time.monotonic()
                settings = config["training"]
                model, selected = build_lightgbm_pipeline(
                    train_data, seed, scale_pos_weight=specification["weight"],
                    feature_columns=columns, **{k: settings[k] for k in ["n_estimators",
                    "learning_rate", "num_leaves", "subsample", "colsample_bytree"]})
                require(selected == columns, "Feature order mismatch")
                model.named_steps["classifier"].set_params(
                    n_jobs=settings["n_jobs"], **dict.fromkeys(settings["seed_fields"], seed))
                model.fit(train_data[columns], train_data.isFraud)
                valraw = model.predict_proba(val[columns])[:, 1]
                testraw = model.predict_proba(test[columns])[:, 1]
                calibrator = PlattCalibrator().fit(valraw[calmask], val.isFraud.to_numpy()[calmask])
                valscore, testscore = calibrator.transform(valraw), calibrator.transform(testraw)
                thresholds = threshold_grid(val.isFraud.to_numpy()[thmask], valscore[thmask],
                                            config["decisions"]["capacities"], config["decisions"]["fn_costs"])
                ap = float(average_precision_score(test.isFraud, testscore))
                rows = []
                for (q, c), item in thresholds.items():
                    if name not in config["decisions"]["threshold_grid_models"] and (q, c) != (.03, 100):
                        continue
                    rows.append({"fold": fold, "model": name, "seed": seed, "capacity": q,
                                 "fn_cost": c, "ap": ap, **item,
                                 **decision_metrics(test.isFraud, testscore >= item["threshold"], c)})
                isotonic = None
                if name in config["calibration"]["isotonic_models"]:
                    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0., y_max=1.)
                    isotonic.fit(valraw[calmask], val.isFraud.to_numpy()[calmask])
                    write_json(destination / (prefix + "_isotonic.json"), {
                        "x": isotonic.X_thresholds_.tolist(), "y": isotonic.y_thresholds_.tolist()})
                for split, data, raw, platt in [("validation", val, valraw, valscore),
                                                 ("test", test, testraw, testscore)]:
                    scores = data[["source_row_id", "step", "isFraud"]].reset_index(drop=True)
                    scores["row_id"] = scores.source_row_id
                    scores["raw"], scores["platt"] = raw, platt
                    if split == "validation":
                        scores["calibration_part"] = calmask
                    if isotonic is not None:
                        scores["isotonic"] = isotonic.transform(raw)
                    scores.to_parquet(destination / f"{prefix}_{split}.parquet", index=False)
                booster = model.named_steps["classifier"].booster_
                booster.save_model(str(destination / (prefix + "_booster.txt")))
                write_json(checkpoint, {"rows": rows, "columns": columns,
                    "input_signature": signature, "seconds": time.monotonic() - started,
                    "effective_num_threads": booster.params.get("num_threads"),
                    "calibrator": {"slope": calibrator.slope, "intercept": calibrator.intercept},
                    "raw_array_sha256": array_hash(testraw), "platt_array_sha256": array_hash(testscore),
                    "files": [{"name": p.name, "sha256": file_hash(p)}
                              for p in sorted(destination.glob(prefix + "_*"))]})
                all_rows.extend(rows)
                print(f"Completed {prefix}: {time.monotonic() - started:.1f}s", flush=True)
                del model
                gc.collect()
    require(len(all_rows) == 660, "Incomplete scenario grid")
    pd.DataFrame(all_rows).to_csv(destination / "all_seed_results.csv", index=False)
    write_json(destination / "COMPLETED.json", {"fits": 90, "scenario_rows": len(all_rows),
               "platt_fits": 90, "isotonic_fits": 30, "input_signature": signature})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.protocol, args.features, args.output)
