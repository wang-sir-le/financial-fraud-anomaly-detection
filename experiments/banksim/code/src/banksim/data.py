"""Input verification and strictly-past, label-free history construction."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import shutil
import urllib.request
import zipfile
from collections import Counter, deque
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    HEADER,
    HISTORY,
    MEMBER,
    MEMBER_SHA,
    SPLITS,
    URL,
    WINDOWS,
    ZIP_SHA,
    append_event,
    require,
    sha256,
    write_json,
)


def inspect_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist().count(MEMBER) == 1, "Missing or duplicated transaction member")
        h = hashlib.sha256()
        with archive.open(MEMBER) as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        require(h.hexdigest() == MEMBER_SHA, "Transaction member hash mismatch; stop, do not replace version")
        with archive.open(MEMBER) as raw:
            header = next(csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline="")))
        require(header == HEADER, "Source header/order mismatch")
    archive_hash = sha256(path)
    return {"archive_sha256": archive_hash, "original_container_match": archive_hash == ZIP_SHA, "member": MEMBER, "member_sha256": h.hexdigest(), "header": header, "row_labels_read": False}


def acquire(root: Path, source: Path | None = None) -> dict[str, Any]:
    target = root / "inputs/banksim_v1.zip"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if source is not None:
            inspect_archive(source)
            shutil.copyfile(source, target)
        else:
            # Fixed public HTTPS endpoint only; no caller-controlled URL.
            with urllib.request.urlopen(URL, timeout=120) as response, target.open("xb") as stream:  # nosec B310
                shutil.copyfileobj(response, stream)
    record = inspect_archive(target)
    write_json(root / "audit/input_verification.json", record)
    append_event(root, "input_verified", **record)
    return record


def parse_cents(text: str) -> int:
    try:
        amount = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Amount is not an exact decimal") from exc
    require(amount.is_finite(), "Nonfinite amount")
    numerator, denominator = amount.as_integer_ratio()
    require((numerator * 100) % denominator == 0, "Amount has sub-cent precision")
    cents = numerator * 100 // denominator
    require(-(2**63) < cents < 2**63, "Amount outside signed int64 cents domain")
    return cents


def load_unlabelled(path: Path) -> pd.DataFrame:
    inspect_archive(path)
    with zipfile.ZipFile(path) as archive, archive.open(MEMBER) as stream:
        frame = pd.read_csv(stream, usecols=HEADER[:-1], dtype=str, keep_default_na=False)
    require("fraud" not in frame.columns, "Labels must not enter preparation")
    frame.insert(0, "source_row_id", np.arange(len(frame), dtype=np.int64))
    for field in ("step", "merchant", "customer", "amount"):
        require(not frame[field].eq("").any(), f"Required input missing: {field}")
    require(frame["step"].str.fullmatch(r"[0-9]+").all(), "Step is not a nonnegative integer")
    frame["step"] = frame["step"].astype(np.int64)
    frame["amount_cents"] = frame["amount"].map(parse_cents).astype(np.int64)
    frame["amount"] = frame["amount_cents"].astype(np.float64) / 100
    frame = frame.sort_values(["step", "source_row_id"], kind="stable").reset_index(drop=True)
    return frame


def assign_splits(frame: pd.DataFrame, *, production: bool = True) -> pd.DataFrame:
    frame = frame.copy()
    parts = np.full(len(frame), "", dtype=object)
    for name, (first, last, expected) in SPLITS.items():
        mask = frame["step"].between(first, last).to_numpy()
        parts[mask] = name
        if production:
            require(int(mask.sum()) == expected, f"Row count mismatch: {name}")
    require(bool(np.all(parts != "")), "Unassigned time step")
    if production:
        require(np.array_equal(np.unique(frame["step"]), np.arange(180)), "Step coverage changed")
    frame["split"] = parts
    return frame


class WindowState:
    """Exact integer moments; sliding max and customer multiplicities."""

    def __init__(self) -> None:
        self.rows: deque[tuple[int, str, int]] = deque()
        self.maxima: deque[tuple[int, int]] = deque()
        self.customers: Counter[str] = Counter()
        self.total = 0
        self.square = 0

    def expire(self, lower: int) -> None:
        while self.rows and self.rows[0][0] < lower:
            _, customer, cents = self.rows.popleft()
            self.customers[customer] -= 1
            if self.customers[customer] == 0:
                del self.customers[customer]
            self.total -= cents
            self.square -= cents * cents
        while self.maxima and self.maxima[0][0] < lower:
            self.maxima.popleft()

    def add(self, step: int, customer: str, cents: int) -> None:
        self.rows.append((step, customer, cents))
        self.customers[customer] += 1
        self.total += cents
        self.square += cents * cents
        while self.maxima and self.maxima[-1][1] <= cents:
            self.maxima.pop()
        self.maxima.append((step, cents))

    def stats(self) -> tuple[int, int, float, float, float]:
        n = len(self.rows)
        if n == 0:
            return 0, 0, 0.0, 0.0, 0.0
        # Python integers are arbitrary precision: no fixed-width moment overflow.
        numerator = n * self.square - self.total * self.total
        require(numerator >= 0, "Negative exact variance numerator")
        std = math.sqrt(numerator / (n * (n - 1))) / 100 if n > 1 else 0.0
        mean = self.total / (100 * n)
        require(math.isfinite(mean) and math.isfinite(std), "Moment conversion overflow")
        return n, len(self.customers), mean, self.maxima[0][1] / 100, std


def build_history(frame: pd.DataFrame) -> pd.DataFrame:
    require("fraud" not in frame.columns, "History cannot accept labels")
    require(frame["source_row_id"].is_unique, "Source identity must be unique")
    frame = frame.sort_values(["step", "source_row_id"], kind="stable").reset_index(drop=True).copy()
    values = np.zeros((len(frame), len(HISTORY)), dtype=np.float64)
    states: dict[str, dict[int, WindowState]] = {}
    previous: dict[str, int] = {}
    for (step_value, merchant), indices in frame.groupby(["step", "merchant"], sort=True).groups.items():
        step = int(step_value)
        idx = np.asarray(indices, dtype=np.int64)
        merchant_states = states.setdefault(str(merchant), {w: WindowState() for w in WINDOWS})
        for j, window in enumerate(WINDOWS):
            state = merchant_states[window]
            state.expire(step - window)
            n, customers, mean, maximum, std = state.stats()
            values[idx, j * 6:j * 6 + 5] = [n, customers, mean, maximum, std]
            values[idx, j * 6 + 5] = frame.loc[idx, "amount"].to_numpy() / mean if mean != 0 else 0.0
        last = previous.get(str(merchant))
        values[idx, -2] = step - last if last is not None else -1
        values[idx, -1] = int(last is not None)
        # Emit every row of this merchant/time group before any update.
        for customer, cents in frame.loc[idx, ["customer", "amount_cents"]].itertuples(index=False, name=None):
            for state in merchant_states.values():
                state.add(step, str(customer), int(cents))
        previous[str(merchant)] = step
    require(bool(np.isfinite(values).all()), "Nonfinite history feature")
    for j, name in enumerate(HISTORY):
        frame[name] = values[:, j]
    return frame
