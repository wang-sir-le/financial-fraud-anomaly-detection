from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from banksim import data, guards, models, workflow
from banksim.common import HEADER, MEMBER, ProtocolError, sha256, write_json


def artificial_zip(root: Path) -> None:
    path = root / "inputs/banksim_v1.zip"
    path.parent.mkdir()
    # Invalid test label must remain unparsed during training-only label access.
    body = ",".join(HEADER) + "\n"
    body += "126,c,4,F,z,M,z,cat,1.00,FORBIDDEN_TEST_LABEL\n"
    body += "0,a,3,M,z,N,z,cat,2.00,1\n"
    body += "0,a,3,M,z,N,z,cat,2.00,0\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(MEMBER, body)


def test_source_identity_before_sort_and_label_column_excluded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artificial_zip(tmp_path)
    monkeypatch.setattr(data, "inspect_archive", lambda p: {})
    frame = data.load_unlabelled(tmp_path / "inputs/banksim_v1.zip")
    assert frame.source_row_id.tolist() == [1, 2, 0]
    assert "fraud" not in frame.columns
    assert len(frame) == 3


def test_label_access_only_requests_permitted_rows(tmp_path: Path) -> None:
    artificial_zip(tmp_path)
    write_json(tmp_path / "protocol/FREEZE.json", {"artificial": True})
    permit = guards.Permit("training", sha256(tmp_path / "protocol/FREEZE.json"))
    y = guards.labels_for_ids(tmp_path, np.array([2, 1]), permit)
    np.testing.assert_array_equal(y, [0, 1])
    with pytest.raises(ProtocolError, match="phase boundary"):
        guards.labels_for_ids(tmp_path, np.array([0]), permit)
    with pytest.raises(ProtocolError):
        guards.labels_for_ids(tmp_path, np.array([1]), guards.Permit("prepare", permit.freeze_sha256))


def test_real_phases_block_before_any_file_or_label_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def bomb(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Label access attempted before guard")
    monkeypatch.setattr(workflow, "labels_for_ids", bomb)
    with pytest.raises(ProtocolError, match="execute-real-training"):
        workflow.train_select(tmp_path, False)
    with pytest.raises(ProtocolError, match="open-test-once"):
        workflow.evaluate(tmp_path, False)
    with pytest.raises(ProtocolError, match="freeze"):
        workflow.train_select(tmp_path, True)


def test_freeze_detects_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "source.txt"
    target.write_text("original")
    write_json(tmp_path / "protocol/FREEZE.json", {"status": "FROZEN_BEFORE_TRAINING", "files": [{"path": "source.txt", "sha256": sha256(target)}]})
    env = {"python": "test", "implementation": "test", "packages": {}}
    write_json(tmp_path / "reproduction/environment.json", env)
    monkeypatch.setattr(guards, "environment", lambda: env)
    guards.verify_freeze(tmp_path)
    target.write_text("changed")
    with pytest.raises(ProtocolError, match="changed"):
        guards.verify_freeze(tmp_path)


def test_candidate_order_selection_tiebreak_and_failure() -> None:
    configs = models.candidates("LightGBM")
    assert len(configs) == 8 and configs[0]["id"] == "C01"
    records = [{"candidate": c, "status": "success", "validation": {"tp@0.03": 5, "ap": .4}} for c in configs]
    assert models.choose_candidate(records)["candidate"]["id"] == "C01"
    records[-1]["validation"]["tp@0.03"] = 6
    assert models.choose_candidate(records)["candidate"]["id"] == "C08"
    records[-1]["status"] = "failed"
    with pytest.raises(ProtocolError):
        models.choose_candidate(records)


def test_model_catalog_does_not_fit_and_is_pinned() -> None:
    catalog = models.catalog()
    assert len(catalog["LightGBM"]) == len(catalog["XGBoost"]) == 8
    left = catalog["LightGBM"][0]["constructor_parameters_seed42"]
    right = catalog["XGBoost"][0]["constructor_parameters_seed42"]
    assert left["subsample_freq"] == 1 and left["boost_from_average"] is False
    assert right["base_score"] == .5 and right["early_stopping_rounds"] is None
    assert left["n_jobs"] == right["n_jobs"] == 4


def test_failed_fit_is_recorded_and_cannot_auto_retry(tmp_path: Path) -> None:
    write_json(tmp_path / "protocol/FREEZE.json", {})
    write_json(tmp_path / "preprocessing/B0.json", {})
    (tmp_path / "features").mkdir()
    np.save(tmp_path / "features/train_identity.npy", np.arange(3), allow_pickle=False)
    class FakeFailure:
        @staticmethod
        def fit_save(*args: Any) -> None:
            raise RuntimeError("Artificial injected failure; no estimator fit")
    result = workflow.attempt(tmp_path, "formal/artificial", "LightGBM", "B0", models.candidates("LightGBM")[0], 42, np.ones((3, 1)), np.array([0, 1, 0]), None, adapter=FakeFailure)
    assert result["status"] == "failed" and "Artificial" in result["error"]
    with pytest.raises(ProtocolError, match="auto-retry"):
        workflow.attempt(tmp_path, "formal/artificial", "LightGBM", "B0", models.candidates("LightGBM")[0], 42, np.ones((3, 1)), np.array([0, 1, 0]), None, adapter=FakeFailure)


def test_raw_margin_adapter_is_explicit() -> None:
    class Fake:
        def predict(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
            assert kwargs in ({"raw_score": True}, {"output_margin": True})
            return np.array([-50., 50.])
    for family in ("LightGBM", "XGBoost"):
        np.testing.assert_array_equal(models.predict_margin(Fake(), family, np.zeros((2, 1))), [-50, 50])


def test_invalid_archive_does_not_pass_hash_check(tmp_path: Path) -> None:
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as archive:
        archive.writestr(MEMBER, io.StringIO(",".join(HEADER) + "\n").getvalue())
    with pytest.raises(ProtocolError, match="hash mismatch"):
        data.inspect_archive(p)
