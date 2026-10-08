"""Strict JSON records, immutable receipts and shared constants."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MEMBER = "bs140513_032310.csv"
MEMBER_SHA = "e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377"
ZIP_SHA = "3bfd0c2bfbdec83ad36eed499eaf38c6951c36be864ecbe30f3bcb7d84895f66"
URL = "https://www.kaggle.com/api/v1/datasets/download/ealaxi/banksim1?datasetVersionNumber=1"
HEADER = ["step", "customer", "age", "gender", "zipcodeOri", "merchant", "zipMerchant", "category", "amount", "fraud"]
WINDOWS = (1, 7, 30)
SEEDS = (42, 52, 62, 72, 82)
FAMILIES = ("LightGBM", "XGBoost")
CELLS = ("B0", "B0H", "B1", "B1H")
SPLITS = {"train": (0, 107, 332592), "validation": (108, 125, 63740), "test": (126, 179, 198311)}
HISTORY = [f"h{w}_{stat}" for w in WINDOWS for stat in ("count", "customers", "mean", "max", "std", "ratio")] + ["h_recency", "h_available"]
B0 = ["step", "category", "amount"]
B1 = B0 + ["age", "gender", "merchant", "zipcodeOri", "zipMerchant"]
CELL_COLUMNS = {"B0": B0, "B0H": B0 + HISTORY, "B1": B1, "B1H": B1 + HISTORY}
CATEGORICAL = {"category", "age", "gender", "merchant", "zipcodeOri", "zipMerchant"}


class ProtocolError(RuntimeError):
    """A protocol invariant failed; never silently repair scientific inputs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    if isinstance(value, np.ndarray):
        return normalize(value.tolist())
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(normalize(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def append_event(root: Path, event: str, **details: Any) -> None:
    path = root / "audit/events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(normalize({"time_utc": utc_now(), "event": event, **details}), ensure_ascii=False, allow_nan=False) + "\n")
