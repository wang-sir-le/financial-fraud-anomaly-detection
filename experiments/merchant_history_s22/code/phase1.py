"""Native integration checks; real source reading and every model fit are forbidden here."""

from __future__ import annotations

import copy
import csv
import importlib.metadata
import io
import math
import sys
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import Any

from native_common import ROOT, lock_code, read_json, require, sha, utc, write_json


def run_phase1(config: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "status": "STARTED",
        "start_utc": utc(),
        "checks": checks,
        "project_fit_calls": 0,
        "project_preprocessor_fits": 0,
    }
    path = ROOT / "artificial/NATIVE_ACCEPTANCE.json"

    def check(name: str, condition: bool, **details: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), **details})
        write_json(path, report)
        require(bool(condition), f"native integration failed: {name}")

    try:
        check(
            "new runtime ROOT and output paths",
            ROOT.resolve() == Path(config["execution_root"]).resolve()
            and ROOT.resolve() != Path(config["restores_from_read_only_attempt"]).resolve(),
            root=str(ROOT),
            outputs="ROOT-relative code, config, artificial, audit and six-cell artifacts",
        )
        versions = {
            name: importlib.metadata.version(name)
            for name in config["versions"]
            if name != "python"
        }
        versions["python"] = sys.version.split()[0]
        write_json(
            ROOT / "artificial/ENVIRONMENT_METADATA.json",
            {"python_executable": sys.executable, "prefix": sys.prefix, "versions": versions},
        )
        check("installed versions exact", versions == config["versions"], actual=versions)
        import lightgbm
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        from lightgbm import LGBMClassifier
        from native_batch import (
            COUNTERS,
            NativePreprocessor,
            build_history,
            contrasts,
            metrics,
            normalize_public,
        )
        from native_common import CELL_COLUMNS, CELLS, HEADER, HISTORY
        from supervision import directory_bytes, run_supervised

        imported = {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        }
        check(
            "imported runtime versions",
            all(config["versions"][k] == v for k, v in imported.items()),
            actual=imported,
        )
        params = read_json(ROOT / "config/C07_PARAMETERS.json")
        constructor = LGBMClassifier(**params)
        actual_params = constructor.get_params(deep=False)
        write_json(ROOT / "artificial/NATIVE_C07_PARAMETERS.json", actual_params)
        check("complete native C07 constructor", actual_params == params)
        del constructor
        fixture = read_json(Path(config["reference"]["fixture"]))
        trace = read_json(Path(config["reference"]["trace"]))
        previous_results = read_json(Path(config["reference"]["results"]))
        check(
            "accepted artificial source receipts",
            previous_results["status"] == "ARTIFICIAL_CONTRACT_PASS"
            and previous_results["failures"] == previous_results["errors"] == 0,
        )
        for source_name, expected in previous_results["source_sha256"].items():
            check(
                f"unchanged artificial reference {source_name}",
                sha(ROOT / "artificial/reference" / source_name) == expected,
            )
        sys.path.insert(0, str(ROOT / "artificial/reference"))
        import contract_kernel as reference
        import run_artificial_tests as reference_tests

        rows = [reference.Row(**r) for r in fixture["rows"]]
        raw_rows = [
            {
                **{c: getattr(r, c) for c in HEADER[:-1] if c not in ("step", "amount")},
                "step": str(r.step),
                "amount": format(Decimal(r.cents) / 100, ".2f"),
                "fraud": "__ARTIFICIAL_UNUSED_LABEL__",
            }
            for r in rows
        ]
        check(
            "artificial CSV source ordinals match fixture identities",
            [r.source_row_id for r in rows] == list(range(len(rows))),
        )
        from source_io import CountedRaw, read_public_projection

        csv_text = io.StringIO(newline="")
        writer = csv.DictWriter(csv_text, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(raw_rows)
        artificial_archive = ROOT / "artificial/NATIVE_FIXTURE.zip"
        with zipfile.ZipFile(artificial_archive, "x", compression=zipfile.ZIP_DEFLATED) as package:
            package.writestr("artificial_fixture.csv", csv_text.getvalue().encode("utf-8"))
        with (
            zipfile.ZipFile(artificial_archive) as package,
            package.open("artificial_fixture.csv") as member,
        ):
            counted = CountedRaw(member)
            with io.BufferedReader(counted) as stream:
                raw = read_public_projection(stream)
        check(
            "final native CSV projection omits unused synthetic fraud fields",
            list(raw) == HEADER[:-1] and "fraud" not in raw,
            returned_logical_bytes=counted.returned_bytes,
        )
        public = normalize_public(raw, purpose="artificial")
        check(
            "final source identities retained",
            len(public) == len(rows)
            and public["source_row_id"].is_unique
            and set(public["source_row_id"]) == {r.source_row_id for r in rows},
        )
        frames: dict[str, Any] = {}
        native_trace: dict[str, Any] = {
            "public_rows": public.to_dict(orient="records"),
            "history": {},
            "cells": {},
        }
        for mode in ("H0", "HS", "HL"):
            frame, audit = build_history(public, mode, trace=True)
            frames[mode] = frame
            native_trace["history"][mode] = {
                "audit": audit,
                "rows": frame[
                    ["source_row_id", "step"] + ([] if mode == "H0" else HISTORY)
                ].to_dict(orient="records"),
            }
            oracle = reference_tests.batch_oracle(rows, mode)
            valid = True
            for _, row in frame.iterrows():
                identity = int(row["source_row_id"])
                for feature, expected in oracle[identity].items():
                    valid = valid and math.isclose(
                        float(row[feature]), expected, rel_tol=1e-12, abs_tol=1e-12
                    )
            check(f"{mode} final history matches independent set oracle", valid)
            if mode != "H0":
                expected = trace["history_sources"][mode]["used_history"]
                check(f"{mode} actual history source identities", audit["used_history"] == expected)
            if mode == "HS":
                by_id = {r.source_row_id: r for r in rows}
                valid = True
                for identity, sources in audit["used_history"].items():
                    current = by_id[int(identity)]
                    used = [j for key in ("h1", "h7", "h30") for j in sources[key]]
                    if sources["recency_source"] is not None:
                        used.append(sources["recency_source"])
                    valid = valid and all(
                        current.step - 7 <= by_id[j].step < current.step
                        and by_id[j].merchant == current.merchant
                        for j in used
                    )
                check(
                    "HS seven-step history and recency isolation",
                    valid and not audit["all_past_previous_allocated"],
                )
                check(
                    "HS capped h30 equals h7",
                    np.array_equal(
                        frame[[c for c in HISTORY if c.startswith("h7_")]].to_numpy(),
                        frame[[c for c in HISTORY if c.startswith("h30_")]].to_numpy(),
                    ),
                )
        processors: dict[str, Any] = {}
        encoded: dict[str, Any] = {}
        for cell in config["cells"]:
            frame = frames[CELLS[cell][1]]
            train = frame.loc[frame["split"] == "train"].copy()
            train.attrs["purpose"] = "artificial"
            processor = NativePreprocessor(CELL_COLUMNS[cell]).fit(train)
            matrix = processor.transform(frame)
            snapshot = processor.snapshot()
            expected = trace["cells"][cell]
            check(f"{cell} final prediction columns", snapshot["columns"] == expected["columns"])
            check(
                f"{cell} train vocabulary, median and constant rules",
                snapshot["rules"] == expected["rules"],
            )
            check(
                f"{cell} exact float32 encoding and identity",
                snapshot["names"] == expected["feature_names"]
                and np.array_equal(matrix, np.asarray(expected["encoded_matrix"], dtype=np.float32))
                and frame["source_row_id"].astype(int).tolist()
                == [r["source_row_id"] for r in expected["raw_model_frame"]],
            )
            check(
                f"{cell} preprocessing serialize/restore without fit",
                np.array_equal(matrix, NativePreprocessor.restore(snapshot).transform(frame)),
            )
            processors[cell], encoded[cell] = processor, matrix
            native_trace["cells"][cell] = {
                "preprocessor": snapshot,
                "encoded_matrix": matrix.tolist(),
                "source_row_ids": frame["source_row_id"].astype(int).tolist(),
                "raw_predictor_rows": frame[["source_row_id", *CELL_COLUMNS[cell]]].to_dict(
                    orient="records"
                ),
            }
        write_json(ROOT / "artificial/FINAL_NATIVE_TRACE.json", native_trace, exclusive=True)
        for off, on in (("A0", "B0"), ("A1", "B1"), ("A2", "B2")):
            off_names, on_names = processors[off].feature_names(), processors[on].feature_names()
            selected = [
                j for j, name in enumerate(on_names) if not name.startswith("cat:merchant=")
            ]
            check(
                f"{off}/{on} only direct merchant encoding removed",
                [on_names[j] for j in selected] == off_names
                and np.array_equal(encoded[off], encoded[on][:, selected])
                and processors[off].rules
                == [r for r in processors[on].rules if r["column"] != "merchant"],
            )
        # Boundary scenarios independently generated, without changing the old tests.
        edge = pd.DataFrame(
            [
                {"step": "0", "customer": "same", "merchant": "old", "amount": "1.00"},
                {"step": "40", "customer": "same", "merchant": "old", "amount": "2.00"},
                {"step": "40", "customer": "same", "merchant": "other", "amount": "3.00"},
            ]
        )
        for field in ("age", "gender", "zipcodeOri", "zipMerchant", "category"):
            edge[field] = "synthetic-constant"
        edge_public = normalize_public(edge, purpose="artificial")
        hs, _ = build_history(edge_public, "HS")
        hl, _ = build_history(edge_public, "HL")
        check(
            "HL global recency retained beyond 30 and merchant entity",
            hl.loc[1, "h30_count"] == 0
            and hl.loc[1, "h_recency"] == 40
            and hl.loc[1, "h_available"] == 1
            and hl.loc[2, "h_available"] == 0,
        )
        check(
            "HS expired aggregate and availability",
            hs.loc[1, "h_recency"] == -1
            and hs.loc[1, "h_available"] == 0
            and float(hs.loc[1, HISTORY[:-2]].sum()) == 0,
        )
        from identity_check import PUBLIC_COLUMNS, aligned_history, bound_public
        from native_common import ProtocolError

        identity_trace: dict[str, Any] = {"permutations": {}, "expected_rejections": {}}
        variants = {
            "original": public.copy(),
            "reversed": public.iloc[::-1].copy(),
            "shuffled_seed20261006": public.iloc[
                np.random.default_rng(20261006).permutation(len(public))
            ].copy(),
        }
        for mode in ("HS", "HL"):
            original_audit = native_trace["history"][mode]["audit"]
            for variant_name, candidate in variants.items():
                bound_public(public, candidate)
                changed, changed_audit = build_history(candidate, mode, trace=True)
                aligned = aligned_history(
                    public, frames[mode], original_audit, changed, changed_audit, mode
                )
                key = f"{mode}_{variant_name}"
                identity_trace["permutations"][key] = {
                    "input_IDs": candidate["source_row_id"].astype(int).tolist(),
                    "output_IDs": changed["source_row_id"].astype(int).tolist(),
                    "alignment": aligned,
                    "sources": changed_audit["used_history"],
                }
                check(f"{mode} permutation independence {variant_name}", True, alignment=aligned)
            # Each deliberately wrong binding must fail the same comparator.
            bad_reindexed = public.iloc[::-1].copy()
            bad_reindexed["source_row_id"] = np.arange(len(public), dtype=np.int64)
            duplicate = public.copy()
            duplicate.loc[duplicate.index[1], "source_row_id"] = int(
                public.iloc[0]["source_row_id"]
            )
            missing = public.iloc[:-1].copy()
            wrong_features = frames[mode].copy()
            first, later = wrong_features.index[0], wrong_features.index[3]
            swapped = wrong_features.loc[[later, first], HISTORY].to_numpy().copy()
            wrong_features.loc[[first, later], HISTORY] = swapped
            wrong_sources = copy.deepcopy(original_audit)
            source_map = wrong_sources["used_history"]
            source_map["0"], source_map["3"] = source_map["3"], source_map["0"]
            mutants = {
                "renumbered_transactions": (bad_reindexed, frames[mode], original_audit),
                "duplicate_ID": (duplicate, frames[mode], original_audit),
                "omitted_ID": (missing, frames[mode], original_audit),
                "misassigned_features": (public, wrong_features, original_audit),
                "misassigned_provenance": (public, frames[mode], wrong_sources),
            }
            for mutant_name, (input_rows, output_rows, output_audit) in mutants.items():
                rejected = False
                reason = None
                try:
                    bound_public(public, input_rows)
                    aligned_history(
                        public, frames[mode], original_audit, output_rows, output_audit, mode
                    )
                except ProtocolError as exc:
                    rejected, reason = True, str(exc)
                key = f"{mode}_{mutant_name}"
                identity_trace["expected_rejections"][key] = {
                    "rejected": rejected,
                    "reason": reason,
                }
                check(f"{mode} rejects {mutant_name}", rejected, reason=reason)
            duplicate_business = public[PUBLIC_COLUMNS].duplicated(keep=False)
            duplicate_ids = set(public.loc[duplicate_business, "source_row_id"].astype(int))
            check(
                f"{mode} duplicate business events keep distinct source IDs",
                len(duplicate_ids) >= 2
                and duplicate_ids <= set(frames[mode]["source_row_id"].astype(int)),
            )
        write_json(ROOT / "artificial/IDENTITY_REGRESSION.json", identity_trace, exclusive=True)
        rejected = False
        try:
            normalize_public(raw.assign(fraud=1), purpose="artificial")
        except RuntimeError:
            rejected = True
        check("labels cannot enter final feature path", rejected)
        numeric_train = pd.DataFrame({"split": ["train"] * 3, "amount": [1, 3, np.nan]})
        numeric_train.attrs["purpose"] = "artificial"
        numeric = NativePreprocessor(["amount"]).fit(numeric_train)
        check(
            "native train-only numeric median and missing rule",
            numeric.rules[0]["median"] == 2
            and np.array_equal(
                numeric.transform(pd.DataFrame({"amount": [np.nan]})),
                np.array([[2, 1]], dtype=np.float32),
            ),
        )
        rejected = False
        try:
            numeric.fit(pd.DataFrame({"split": ["review"], "amount": [1]}))
        except RuntimeError:
            rejected = True
        check("nontraining preprocessing fit rejected", rejected)
        synthetic = [
            ([1, 0, 1], [1.0, 1.0, 0.0], [0, 1, 2]),
            ([i % 2 for i in range(60)], [0.5] * 60, list(range(60))),
            ([0, 0], [0.2, 0.1], [0, 1]),
            ([], [], []),
        ]
        for i, (labels, scores, ids) in enumerate(synthetic):
            actual = metrics(
                np.asarray(labels, dtype=np.int64),
                np.asarray(scores, dtype=float),
                np.asarray(ids, dtype=np.int64),
            )
            expected = reference.artificial_metrics(labels, scores, ids)
            valid = True
            for key in ("N", "P", "K", "TP", "Recall", "Precision", "AP", "selected_ids"):
                a, b = actual[key], expected[key]
                valid = valid and (
                    math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                    if isinstance(a, float) and b is not None
                    else a == b
                )
            check(f"native capacity, SHA tie and grouped AP artificial case {i}", valid)
        counts = {"A0": 0, "A1": 1, "A2": 3, "B0": 1, "B1": 2, "B2": 3}
        measures = {c: {"N": 100, "P": 4, "K": 3, "TP": tp} for c, tp in counts.items()}
        effects = contrasts(measures)
        expected_effects = reference.artificial_contrasts(measures)
        check(
            "native six-cell contrasts and integer denominator",
            all(effects[k] == expected_effects[k] for k in ("delta0", "delta1", "I"))
            and effects["I"] == effects["J"] / effects["common_P"],
        )
        from tail_budget import budget_decision

        baseline_tail = {
            "wall_seconds": 10.0,
            "delivery_seconds": 1.0,
            "directory_bytes": directory_bytes(ROOT),
            "rss_bytes": 32 * 1024**2,
        }
        tail_cases = {
            "normal_complete": (True, baseline_tail, "COMPLETED", 0),
            "late_global_wall": (
                True,
                {**baseline_tail, "wall_seconds": 1200.01},
                "STOPPED_NO_RETRY",
                1,
            ),
            "late_delivery_wall": (
                True,
                {**baseline_tail, "delivery_seconds": 60.01},
                "STOPPED_NO_RETRY",
                1,
            ),
            "late_disk": (
                True,
                {**baseline_tail, "directory_bytes": config["caps"]["disk_bytes"] + 1},
                "STOPPED_NO_RETRY",
                1,
            ),
            "late_RSS": (
                True,
                {**baseline_tail, "rss_bytes": config["caps"]["rss_bytes"]},
                "STOPPED_NO_RETRY",
                1,
            ),
            "incomplete_pipeline": (False, baseline_tail, "STOPPED_NO_RETRY", 1),
        }
        tail_trace: dict[str, Any] = {}
        for name, (
            candidate_complete,
            observed,
            expected_status,
            expected_code,
        ) in tail_cases.items():
            decision = budget_decision(candidate_complete, observed, config["caps"])
            tail_trace[name] = decision
            check(
                f"final budget decision {name}",
                decision["status"] == expected_status and decision["exit_code"] == expected_code,
                decision=decision,
            )
        write_json(ROOT / "artificial/TAIL_BUDGET_REGRESSION.json", tail_trace, exclusive=True)
        # Real tiny processes execute the same gated launch/job supervision used by fits.
        resources = []
        for action in ("toy_normal", "toy_descendant", "toy_memory", "toy_disk"):
            payload = ROOT / "artificial" / f"{action}.json"
            write_json(payload, {"action": action}, exclusive=True)
            resource = run_supervised(
                action,
                payload,
                f"phase1_{action}",
                seconds=1.5 if action == "toy_descendant" else 10,
                rss_limit=1 if action == "toy_memory" else config["caps"]["rss_bytes"],
                disk_limit=(directory_bytes(ROOT) + 64 * 1024)
                if action == "toy_disk"
                else config["caps"]["disk_bytes"],
                artificial_resource_arm=(ROOT / "artificial/toy_memory.ready")
                if action == "toy_memory"
                else None,
            )
            resources.append(resource)
            expected_stop = {
                "toy_normal": "NORMAL_EXIT",
                "toy_descendant": "TIME_LIMIT",
                "toy_memory": "RSS_LIMIT",
                "toy_disk": "DISK_LIMIT",
            }[action]
            check(
                f"real tiny subprocess {action}",
                resource["stop_reason"] == expected_stop
                and not resource["cleanup_pid_survivors"]
                and resource["gate_released"],
                resource=resource,
            )
            if action == "toy_descendant":
                check(
                    "timeout terminates actual descendant tree",
                    len(resource["terminated_pids"]) >= 2,
                )
        check(
            "no model fit and no project preprocessing during native acceptance",
            COUNTERS["project_preprocessor_fits"] == 0,
        )
        report.update(
            {
                "status": "PASS",
                "end_utc": utc(),
                "artificial_preprocessor_fit_calls": COUNTERS["artificial_preprocessor_fits"],
                "real_project_data_read": False,
                "real_model_loaded": False,
                "project_fit_calls": 0,
                "project_preprocessor_fits": 0,
                "operating_system_isolation_claim": False,
                "limitations": [
                    "sampled RSS and directory bytes, not instantaneous strict peaks",
                    "Python audit/controlled code path is not full OS isolation",
                ],
            }
        )
        write_json(path, report)
        write_json(
            ROOT / "config/PHASE1_LOCK.json",
            {
                "status": "NATIVE_ACCEPTANCE_PASS_BEFORE_REAL_DATA",
                "time_utc": utc(),
                "code_sha256": lock_code(),
                "config_sha256": sha(ROOT / "config/EXECUTION_CONFIG.json"),
                "parameter_sha256": sha(ROOT / "config/C07_PARAMETERS.json"),
                "expected_source": config["source"],
                "metrics": "fixed whole-window q=.03 and fixed segments",
                "train_sample_selection": config["sample_check"],
            },
            exclusive=True,
        )
        return report
    except BaseException as exc:
        if "native_batch" in sys.modules:
            report["artificial_preprocessor_fit_calls"] = sys.modules["native_batch"].COUNTERS[
                "artificial_preprocessor_fits"
            ]
        report.update(
            {
                "status": "FAILED_STOP_NO_RETRY",
                "end_utc": utc(),
                "error": f"{type(exc).__name__}:{exc}",
            }
        )
        write_json(path, report)
        raise
