"""Full lifecycle with a hand-written CSV and fake models, never real estimators."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from banksim import data, guards, models, workflow
from banksim.common import HEADER, MEMBER, ProtocolError, read_json, sha256, write_json


def test_artificial_lifecycle_reuse_selection_and_one_time_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = Path(__file__).resolve().parents[1]
    for folder in ("protocol", "inputs", "features", "preprocessing", "models", "predictions", "analysis", "audit"):
        (tmp_path / folder).mkdir()
    shutil.copyfile(project / "protocol/draft_source.json", tmp_path / "protocol/draft_source.json")
    lines = [",".join(HEADER)]
    for step in range(180):
        for j in range(2):
            lines.append(f"{step},customer{j},age{j},gender{j},zip,merchant{step % 3},zip,category{j},{j + 1}.00,{j}")
    with zipfile.ZipFile(tmp_path / "inputs/banksim_v1.zip", "w") as archive:
        archive.writestr(MEMBER, "\n".join(lines) + "\n")
    monkeypatch.setattr(data, "inspect_archive", lambda p: {"artificial": True})
    monkeypatch.setattr(workflow, "acquire", lambda root: {"artificial": True})
    monkeypatch.setattr(workflow, "assign_splits", lambda frame: data.assign_splits(frame, production=False))
    prepared = workflow.prepare(tmp_path)
    assert prepared["fraud_column_loaded"] is False
    assert not (tmp_path / "preprocessing/B0_test.npy").exists()
    workflow.configure(tmp_path)
    # Minimal fixture freeze: the production freeze implementation is tested separately.
    write_json(tmp_path / "protocol/FREEZE.json", {"status": "FROZEN_BEFORE_TRAINING", "files": []})
    fake_calls: list[tuple[str, int, int]] = []
    class FakeModel:
        def predict(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
            assert x[:, 0].max() <= 125, "No test scoring during model selection"
            return x[:, -1].astype(float) + x[:, 0] / 1000
    def fake_fit(family: str, candidate: dict[str, Any], seed: int, x: np.ndarray, y: np.ndarray, path: Path) -> FakeModel:
        assert x[:, 0].max() <= 107
        assert len(np.unique(y)) == 2
        fake_calls.append((family, seed, len(x)))
        write_json(path, {"artificial_model": True, "family": family, "candidate": candidate, "seed": seed})
        write_json(path.with_suffix(path.suffix + ".fit.json"), {"artificial": True})
        return FakeModel()
    monkeypatch.setattr(models, "fit_save", fake_fit)
    result = workflow.train_select(tmp_path, True)
    assert result["status"] == "success"
    assert len(fake_calls) == 54
    training = read_json(tmp_path / "protocol/TRAINING_COMPLETE.json")
    assert len(training["formal_conditions"]) == 40
    assert sum(r["reused_tuning_fit"] for r in training["formal_conditions"]) == 2
    assert not (tmp_path / "protocol/TEST_RELEASE.json").exists()
    assert not (tmp_path / "predictions/test_labels.npy").exists()
    events = [json.loads(line) for line in (tmp_path / "audit/events.jsonl").read_text().splitlines()]
    assert {r["phase"] for r in events if r["event"] == "labels_accessed"} == {"training"}
    # Incomplete formal set cannot open the test, even if all its entries succeeded.
    altered = dict(training)
    altered["formal_conditions"] = training["formal_conditions"][:-1]
    write_json(tmp_path / "protocol/TRAINING_COMPLETE.json", altered)
    with pytest.raises(ProtocolError, match="forty"):
        guards.test_permit(tmp_path, True)
    assert not (tmp_path / "protocol/TEST_RELEASE.json").exists()
    write_json(tmp_path / "protocol/TRAINING_COMPLETE.json", training)
    before = sha256(tmp_path / "protocol/SELECTION.json")
    def fake_load_predict(path: Path, family: str, x: np.ndarray) -> np.ndarray:
        assert read_json(path)["artificial_model"] is True
        assert x[:, 0].min() >= 126
        return x[:, -1].astype(float) + x[:, 0] / 1000
    monkeypatch.setattr(models, "load_predict", fake_load_predict)
    result = workflow.evaluate(tmp_path, True)
    assert result["models_scored"] == 40 and result["status"] == "evaluated"
    assert sha256(tmp_path / "protocol/SELECTION.json") == before
    with pytest.raises(ProtocolError, match="already released"):
        workflow.evaluate(tmp_path, True)
    with pytest.raises(ProtocolError, match="after test release"):
        guards.training_permit(tmp_path, True)
    evaluation = read_json(tmp_path / "protocol/EVALUATION_COMPLETE.json")
    workflow.verify_records(tmp_path, evaluation["artifacts"])
    assert len(evaluation["score_files"]) == 40


def test_freeze_rejects_source_change_after_validation(tmp_path: Path) -> None:
    for folder in ("src", "tests", "audit", "protocol"):
        (tmp_path / folder).mkdir()
    for name in ("run.py", "verify.py", "pyproject.toml"):
        (tmp_path / name).write_text("# artificial fixture\n")
    records = workflow.source_records(tmp_path)
    write_json(tmp_path / "audit/VALIDATION.json", {"status": "PASS", "source_records": records})
    (tmp_path / "run.py").write_text("# changed after validation\n")
    with pytest.raises(ProtocolError, match="Code changed"):
        workflow.freeze(tmp_path)
