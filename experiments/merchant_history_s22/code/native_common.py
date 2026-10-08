"""Owned run constants and receipts; import does not read project data."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
HEADER = [
    "step",
    "customer",
    "age",
    "gender",
    "zipcodeOri",
    "merchant",
    "zipMerchant",
    "category",
    "amount",
    "fraud",
]
WINDOWS = (1, 7, 30)
STATS = ("count", "customers", "mean", "max", "std", "ratio")
HISTORY = [f"h{w}_{s}" for w in WINDOWS for s in STATS] + ["h_recency", "h_available"]
BASE_ON = ["step", "category", "amount", "age", "gender", "merchant", "zipcodeOri", "zipMerchant"]
BASE_OFF = [c for c in BASE_ON if c != "merchant"]
CELLS = {
    "A0": (False, "H0"),
    "B0": (True, "H0"),
    "A1": (False, "HS"),
    "B1": (True, "HS"),
    "A2": (False, "HL"),
    "B2": (True, "HL"),
}
CELL_COLUMNS = {
    c: (BASE_ON if on else BASE_OFF) + ([] if mode == "H0" else HISTORY)
    for c, (on, mode) in CELLS.items()
}
CATEGORICAL = {"category", "age", "gender", "merchant", "zipcodeOri", "zipMerchant"}


class ProtocolError(RuntimeError):
    """Stop the one authorized task; never repair scientific inputs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def utc() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def events(event: str, **details: Any) -> None:
    with (ROOT / "audit/events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                {"time_utc": utc(), "event": event, **details}, ensure_ascii=False, allow_nan=False
            )
            + "\n"
        )


def lock_code() -> dict[str, str]:
    files = list((ROOT / "code").glob("*.py")) + list((ROOT / "artificial/reference").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def verify_lock() -> None:
    frozen = read_json(ROOT / "config/PHASE1_LOCK.json")
    require(lock_code() == frozen["code_sha256"], "final code changed after phase-one lock")
    require(
        sha(ROOT / "config/EXECUTION_CONFIG.json") == frozen["config_sha256"],
        "execution config changed after phase-one lock",
    )
    require(
        sha(ROOT / "config/C07_PARAMETERS.json") == frozen["parameter_sha256"],
        "C07 parameters changed after phase-one lock",
    )
