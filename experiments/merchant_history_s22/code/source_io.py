"""Only the declared BankSim archive; feature and label projections remain separate."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from native_batch import normalize_public
from native_common import HEADER, ROOT, require, sha, utc, verify_lock, write_json


class CountedRaw(io.RawIOBase):
    def __init__(self, stream: Any) -> None:
        super().__init__()
        self.stream = stream
        self.returned_bytes = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        block = self.stream.read(len(buffer))
        buffer[: len(block)] = block
        self.returned_bytes += len(block)
        return len(block)


def receipt(operation: str, source: dict[str, Any], started: float, **details: Any) -> None:
    record = {
        "time_utc": utc(),
        "operation": operation,
        "archive": source["archive"],
        "member": source["member"],
        "seconds": time.perf_counter() - started,
        **details,
    }
    with (ROOT / "audit/DATA_READS.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def read_public_projection(stream: Any) -> pd.DataFrame:
    """Identical native CSV public-column path for artificial and project inputs."""
    return pd.read_csv(
        stream,
        usecols=HEADER[:-1],
        dtype=str,
        keep_default_na=False,
        engine="c",
        on_bad_lines="error",
    )


def read_public(config: dict[str, Any]) -> pd.DataFrame:
    verify_lock()
    source = config["source"]
    archive = Path(source["archive"])
    started = time.perf_counter()
    actual = sha(archive)
    receipt(
        "archive_sha256",
        source,
        started,
        returned_logical_bytes=archive.stat().st_size,
        actual_sha256=actual,
        includes_all_region_label_bytes_in_compressed_archive=True,
        label_values_used=False,
    )
    require(actual == source["archive_sha256"], "raw archive hash mismatch")
    with zipfile.ZipFile(archive) as package:
        info = package.getinfo(source["member"])
        started = time.perf_counter()
        digest = hashlib.sha256()
        returned = 0
        with package.open(info) as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                returned += len(block)
                digest.update(block)
        receipt(
            "member_sha256",
            source,
            started,
            returned_logical_bytes=returned,
            actual_sha256=digest.hexdigest(),
            all_fraud_column_bytes_traversed=True,
            label_values_decoded=False,
            label_values_used=False,
        )
        require(digest.hexdigest() == source["member_sha256"], "CSV member hash mismatch")
        started = time.perf_counter()
        with package.open(info) as member:
            counted = CountedRaw(member)
            with io.TextIOWrapper(io.BufferedReader(counted), encoding="utf-8", newline="") as text:
                header = next(csv.reader(text))
        require(header == HEADER, "CSV header does not match the locked native projection")
        receipt(
            "header_projection_check",
            source,
            started,
            returned_logical_bytes=counted.returned_bytes,
            header=header,
            read_ahead_may_contain_label_bytes=True,
            label_values_used=False,
        )
        started = time.perf_counter()
        with package.open(info) as member:
            counted = CountedRaw(member)
            with io.BufferedReader(counted) as stream:
                raw = read_public_projection(stream)
        receipt(
            "public_feature_projection",
            source,
            started,
            returned_logical_bytes=counted.returned_bytes,
            output_columns=list(raw.columns),
            returned_rows=len(raw),
            all_fraud_column_bytes_traversed=True,
            fraud_column_retained_in_dataframe=False,
            label_values_used=False,
            parser_tokenization_is_not_OS_column_selective_IO=True,
        )
    started = time.perf_counter()
    public = normalize_public(raw, purpose="project")
    del raw
    expected = {
        "train": config["splits"]["train"][2],
        "validation": config["splits"]["validation_unlabelled"][2],
        "review": config["splits"]["review"][2],
    }
    actual_rows = {str(k): int(v) for k, v in public["split"].value_counts().items()}
    require(
        actual_rows == expected and len(public) == sum(expected.values()),
        "source split counts disagree with fixed input contract",
    )
    require(
        public["source_row_id"].is_unique and set(public["step"].unique()) == set(range(180)),
        "source coverage/identity mismatch",
    )
    np.save(
        ROOT / "features/all_identity.npy",
        public[["source_row_id", "step"]].to_numpy(dtype=np.int64),
        allow_pickle=False,
    )
    write_json(
        ROOT / "audit/SOURCE_ALIGNMENT.json",
        {
            "archive_sha256": actual,
            "member_sha256": source["member_sha256"],
            "counts": actual_rows,
            "total_rows": len(public),
            "identity_assignment": "original CSV ordinal before sorting",
            "canonical_order": "step ascending, source_row_id ascending; no filtered transactions",
            "public_columns": list(public.columns),
            "labels_in_public_features": False,
            "normalization_seconds": time.perf_counter() - started,
            "identity_sha256": sha(ROOT / "features/all_identity.npy"),
            "validation_label_values_used": False,
        },
        exclusive=True,
    )
    return public


def read_labels(config: dict[str, Any], identity: np.ndarray, region: str) -> np.ndarray:
    verify_lock()
    require(region in ("train", "review"), "label region not authorized")
    if region == "review":
        locked = read_models_lock()
        require(
            locked["all_six_models_and_preprocessors_locked"], "review labels before model lock"
        )
    lower, upper = (0, 107) if region == "train" else (126, 179)
    require(
        identity.ndim == 2
        and identity.shape[1] == 2
        and bool(((identity[:, 1] >= lower) & (identity[:, 1] <= upper)).all()),
        "label identity outside declared region",
    )
    source = config["source"]
    started = time.perf_counter()
    write_json(
        ROOT / "audit" / f"{region.upper()}_LABEL_ACCESS_ENTERED.json",
        {
            "time_utc": utc(),
            "region": region,
            "authorized_numeric_label_steps": [lower, upper],
            "state": "PROJECTION_ENTERED",
            "other_region_labels_not_used": True,
            "completed_count": (
                "completion receipt will record count; partial interruption is unknown"
            ),
        },
        exclusive=True,
    )
    parsed: dict[int, tuple[int, int]] = {}
    all_rows = 0
    with zipfile.ZipFile(source["archive"]) as package, package.open(source["member"]) as member:
        counted = CountedRaw(member)
        with io.TextIOWrapper(io.BufferedReader(counted), encoding="utf-8", newline="") as text:
            reader = csv.reader(text)
            require(next(reader) == HEADER, "label reader header changed")
            for source_id, row in enumerate(reader):
                require(len(row) == len(HEADER), "label reader malformed source row")
                step = int(row[0])
                all_rows += 1
                if lower <= step <= upper:
                    require(row[-1] in ("0", "1"), "authorized label is not binary")
                    parsed[source_id] = (step, int(row[-1]))
    receipt(
        f"{region}_label_projection",
        source,
        started,
        returned_logical_bytes=counted.returned_bytes,
        source_rows_traversed=all_rows,
        all_fraud_column_bytes_traversed=True,
        csv_reader_transiently_decodes_all_field_strings=True,
        label_numeric_conversion_retention_and_use_steps=[lower, upper],
        other_region_label_strings_not_converted_retained_or_used=True,
        label_rows_used=len(parsed),
    )
    require(len(parsed) == len(identity), "label/source region counts differ")
    labels = np.empty(len(identity), dtype=np.int8)
    for j, (source_id, step) in enumerate(identity):
        actual_step, label = parsed[int(source_id)]
        require(actual_step == int(step), "source step changed during label projection")
        labels[j] = label
    require(
        all_rows
        == sum(config["splits"][k][2] for k in ("train", "validation_unlabelled", "review")),
        "label reader full source count changed",
    )
    destination = ROOT / "labels" / f"{region}_only.npy"
    destination.parent.mkdir(exist_ok=True)
    np.save(destination, labels, allow_pickle=False)
    write_json(
        ROOT / "audit" / f"{region.upper()}_LABEL_SCOPE.json",
        {
            "region": region,
            "used_steps": [lower, upper],
            "N": len(labels),
            "P": int(labels.sum()),
            "aligned_identity_sha256": sha(ROOT / "features" / f"{region}_identity.npy"),
            "label_projection_sha256": sha(destination),
            "feature_preprocessing_modules_receive_labels": False,
            "physical_byte_traversal_contains_labels_from_all_regions": True,
            "CSV_transient_field_strings_vs_numeric_label_use": "see DATA_READS.jsonl",
            "time_utc": utc(),
        },
        exclusive=True,
    )
    return labels


def read_models_lock() -> Any:
    from native_common import read_json

    return read_json(ROOT / "config/MODELS_LOCK.json")
