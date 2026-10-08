"""Explicit paths, input contracts and auditable stage outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

SEEDS = (42, 52, 62, 72, 82)
FOLDS = (1, 2, 3)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@dataclass
class Context:
    config_path: Path
    config: dict[str, Any]
    root: Path
    output: Path
    resource: Path

    @classmethod
    def load(cls, path: Path) -> Context:
        config = read_json(path)
        root = Path(config["workspace_root"]).resolve()
        output = Path(config["output_root"]).resolve()
        allowed = root / "experiment/outputs/q2_extension_v1"
        require(output.is_relative_to(allowed), "Output must be a new extension run")
        resource = root / config["baseline"]["root"] / "reproducibility"
        require(not resource.is_relative_to(output), "Output cannot contain the baseline")
        require(config == read_json(output / "protocol/protocol_effective.json"),
                "Runtime configuration differs from the execution protocol")
        return cls(path.resolve(), config, root, output, resource)

    def spec(self, dataset: str) -> dict[str, Any]:
        return dict(self.config["datasets"][dataset])

    def contrasts(self, dataset: str, core: bool = False) -> list[dict[str, Any]]:
        spec = self.spec(dataset)
        return list(spec["core_contrasts"] + ([] if core else spec["auxiliary_contrasts"]))

    def start(self, stage: str) -> Path:
        folder = self.output / stage
        folder.mkdir(exist_ok=True, parents=True)
        status = folder / "STATUS.json"
        require(not status.exists(), f"Stage already attempted; preserve it: {folder}")
        write_json(status, {"status": "RUNNING", "started_utc": datetime.now(UTC).isoformat(),
                            "config_sha256": file_hash(self.config_path)})
        return folder

    def finish(self, folder: Path, details: dict[str, Any]) -> None:
        report = read_json(folder / "STATUS.json")
        report.update(status="COMPLETED", completed_utc=datetime.now(UTC).isoformat(),
                      details=details)
        report["files"] = [{"name": p.relative_to(folder).as_posix(),
                            "sha256": file_hash(p), "bytes": p.stat().st_size}
                           for p in sorted(folder.rglob("*")) if p.is_file()
                           and p.name != "STATUS.json"]
        write_json(folder / "STATUS.json", report)


def prediction_path(ctx: Context, dataset: str, fold: int, model: str,
                    seed: int, split: str = "test") -> Path:
    path: Path = ctx.root / str(ctx.spec(dataset)["predictions"])
    if dataset == "paysim":
        return path / f"paysim_{fold}_{model}_{seed}_{split}.parquet"
    require(split == "test", "IEEE ledger contains only the frozen Test scores")
    return path


def load_frame(ctx: Context, dataset: str, fold: int, model: str,
               seed: int, split: str = "test") -> pd.DataFrame:
    spec = ctx.spec(dataset)
    path = prediction_path(ctx, dataset, fold, model, seed, split)
    if dataset == "paysim":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_parquet(path, filters=[("fold", "=", fold), ("model_id", "=", model),
                                              ("seed", "=", seed)])
    require(len(frame) > 0, f"Empty vector: {dataset}/{fold}/{model}/{seed}")
    require(not frame[spec["id"]].duplicated().any(), "Duplicate transaction ID")
    frame = frame.sort_values(spec["id"], kind="stable").reset_index(drop=True)
    require(bool(frame[spec["target"]].isin([0, 1]).all()), "Invalid binary labels")
    for stage in spec["stages"]:
        if stage == "clipped":
            continue
        values = frame[stage].to_numpy(dtype=np.float64)
        require(bool(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()),
                "Invalid probabilities")
    return frame


def assert_aligned(reference: pd.DataFrame, other: pd.DataFrame,
                   columns: list[str]) -> None:
    require(len(reference) == len(other), "Different paired sample sizes")
    for column in columns:
        require(np.array_equal(reference[column].to_numpy(), other[column].to_numpy()),
                f"Pairing mismatch: {column}")


def scores_for(frame: pd.DataFrame, stage: str) -> np.ndarray:
    if stage == "clipped":
        return cast(np.ndarray, np.clip(frame.raw.to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6))
    return cast(np.ndarray, frame[stage].to_numpy(dtype=np.float64))


def inventory(ctx: Context) -> None:
    folder = ctx.start("audit/inputs")
    rows: list[dict[str, Any]] = []
    for item in read_json(ctx.root / ctx.spec("paysim")["manifest"]):
        name = item["path"].replace("\\", "/").split("/")[-1]
        path = ctx.root / ctx.spec("paysim")["predictions"] / name
        digest = file_hash(path)
        require(digest == item["sha256"], f"Frozen score mismatch: {name}")
        rows.append({"path": str(path), "sha256": digest})
    ieee = ctx.root / ctx.spec("ieee_cis")["predictions"]
    expected = read_json(ctx.root / ctx.spec("ieee_cis")["manifest"])
    require(file_hash(ieee) == expected["output_hashes"]["prediction_scores.parquet"],
            "IEEE score hash mismatch")
    rows.append({"path": str(ieee), "sha256": file_hash(ieee)})
    alignments = []
    calibrators = []
    for dataset in ctx.config["datasets"]:
        spec = ctx.spec(dataset)
        keys = [spec["id"], spec["time"], spec["target"]]
        for fold in FOLDS:
            for split in (["test", "validation"] if dataset == "paysim" else ["test"]):
                reference = None
                for model in spec["models"]:
                    for seed in SEEDS:
                        data = load_frame(ctx, dataset, fold, model, seed, split)
                        if reference is None:
                            reference = data[keys].copy()
                        else:
                            assert_aligned(reference, data, keys)
                        alignments.append({"dataset": dataset, "fold": fold, "model": model,
                                           "seed": seed, "split": split, "rows": len(data),
                                           "key_hash": array_hash(data[keys].to_numpy(np.int64))})
                        if dataset == "paysim" and split == "test":
                            path = prediction_path(ctx, dataset, fold, model, seed)
                            meta = path.with_name(path.name.replace("_test.parquet", ".json"))
                            checkpoint = read_json(meta)
                            require(checkpoint["calibrator"]["slope"] > 0,
                                    "Unexpected non-positive Platt slope")
                            calibrators.append({"fold": fold, "model": model, "seed": seed,
                                                "path": str(meta), "sha256": file_hash(meta),
                                                **checkpoint["calibrator"]})
                print(f"Paired {dataset} fold={fold} split={split}", flush=True)
    pd.DataFrame(alignments).to_csv(folder / "alignment.csv", index=False)
    pd.DataFrame(calibrators).to_csv(folder / "calibrator_parameters.csv", index=False)
    write_json(folder / "prediction_hashes.json", rows)
    ctx.finish(folder, {"hashed_files": len(rows), "paired_vectors": len(alignments),
                        "labels_times_ids_aligned": True})
