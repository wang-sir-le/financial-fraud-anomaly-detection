"""Phase barriers: explicit CLI intent, frozen bytes and append-only access receipts."""

from __future__ import annotations

import csv
import importlib.metadata
import io
import platform
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    CELLS,
    FAMILIES,
    HEADER,
    MEMBER,
    MEMBER_SHA,
    SEEDS,
    append_event,
    object_hash,
    read_json,
    require,
    sha256,
    utc_now,
    write_json,
)

CORE_PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm", "xgboost", "psutil")


def environment() -> dict[str, Any]:
    return {"python": platform.python_version(), "implementation": platform.python_implementation(), "packages": {p: importlib.metadata.version(p) for p in CORE_PACKAGES}, "platform": platform.platform(), "machine": platform.machine()}


def verify_freeze(root: Path) -> dict[str, Any]:
    path = root / "protocol/FREEZE.json"
    require(path.exists(), "Training/test stage requires a completed pre-training freeze")
    freeze = read_json(path)
    require(freeze["status"] == "FROZEN_BEFORE_TRAINING", "Protocol not frozen")
    for item in freeze["files"]:
        target = root / item["path"]
        require(target.is_file() and sha256(target) == item["sha256"], f"Frozen artifact changed: {item['path']}")
    actual, expected = environment(), read_json(root / "reproduction/environment.json")
    for field in ("python", "implementation", "packages"):
        require(actual[field] == expected[field], f"Runtime mismatch: {field}")
    return freeze


@dataclass(frozen=True)
class Permit:
    phase: str
    freeze_sha256: str


def training_permit(root: Path, execute: bool) -> Permit:
    require(execute, "BLOCKED: real training requires --execute-real-training in a subsequent authorized phase")
    verify_freeze(root)
    require(not (root / "protocol/TEST_RELEASE.json").exists(), "Training cannot resume after test release")
    return Permit("training", sha256(root / "protocol/FREEZE.json"))


def test_permit(root: Path, open_test: bool) -> Permit:
    require(open_test, "BLOCKED: test access requires --open-test-once in a subsequent authorized phase")
    verify_freeze(root)
    training = read_json(root / "protocol/TRAINING_COMPLETE.json")
    selection = read_json(root / "protocol/SELECTION.json")
    expected = {(f, c, s) for f in FAMILIES for c in CELLS for s in SEEDS}
    actual = {(r["family"], r["cell"], r["seed"]) for r in training["formal_conditions"]}
    require(training["status"] == "success" and actual == expected and len(training["formal_conditions"]) == 40, "All forty formal conditions required before test release")
    require(all(r["status"] == "success" for r in training["formal_conditions"]), "Failed formal condition keeps test closed")
    require(training["freeze_sha256"] == sha256(root / "protocol/FREEZE.json"), "Training uses another freeze")
    require(training["selection_sha256"] == sha256(root / "protocol/SELECTION.json"), "Selection changed")
    require(selection["freeze_sha256"] == training["freeze_sha256"], "Selection uses another freeze")
    for record in training["artifacts"]:
        require(sha256(root / record["path"]) == record["sha256"], "Training artifact changed before test release")
    require(not (root / "protocol/TEST_RELEASE.json").exists(), "Test already released; use cached analysis, do not rescore/reselect")
    write_json(root / "protocol/TEST_RELEASE.json", {"time_utc": utc_now(), "freeze_sha256": training["freeze_sha256"], "selection_sha256": training["selection_sha256"], "training_complete_sha256": sha256(root / "protocol/TRAINING_COMPLETE.json"), "status": "OPENED_ONCE"}, exclusive=True)
    append_event(root, "test_release", freeze_sha256=training["freeze_sha256"])
    return Permit("test", training["freeze_sha256"])


def labels_for_ids(root: Path, ids: np.ndarray, permit: Permit) -> np.ndarray:
    """Only accesses row['fraud'] for requested IDs in the permitted time region."""
    require(permit.phase in ("training", "test"), "No real label access in preparation")
    require(permit.freeze_sha256 == sha256(root / "protocol/FREEZE.json"), "Permit/freeze mismatch")
    ids = np.asarray(ids, dtype=np.int64)
    require(len(np.unique(ids)) == len(ids), "Duplicate requested original identities")
    requested = {int(v): i for i, v in enumerate(ids)}
    result = np.full(len(ids), -1, dtype=np.int8)
    with zipfile.ZipFile(root / "inputs/banksim_v1.zip") as archive, archive.open(MEMBER) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
        require(reader.fieldnames == HEADER, "Label source schema changed")
        for source_id, row in enumerate(reader):
            if source_id not in requested:
                continue
            step = int(row["step"])
            require((0 <= step <= 125) if permit.phase == "training" else (126 <= step <= 179), "Label request crosses phase boundary")
            text = row["fraud"]
            require(text in ("0", "1"), "Invalid binary label in permitted region")
            result[requested[source_id]] = int(text)
    require(bool((result >= 0).all()), "Some requested identities missing from label source")
    append_event(root, "labels_accessed", phase=permit.phase, rows=len(ids), source_member_sha256=MEMBER_SHA, identity_hash=object_hash(ids))
    return result
