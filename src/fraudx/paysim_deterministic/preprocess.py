"""Stable source identities and exact-cent history aggregation for PaySim v2."""

from __future__ import annotations

import hashlib
import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

RAW_COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def literal(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def ingest_csv(source: Path, destination: Path, block_size: int = 1048576) -> dict[str, Any]:
    """Assign CSV record ordinals before sorting; reject amounts needing rounding."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    types = {"step": pa.int32(), "type": pa.string(), "amount": pa.decimal128(20, 2),
             "nameOrig": pa.string(), "nameDest": pa.string(), "isFraud": pa.int8()}
    reader = pacsv.open_csv(source, read_options=pacsv.ReadOptions(
        use_threads=False, block_size=block_size), convert_options=pacsv.ConvertOptions(
            include_columns=RAW_COLUMNS, column_types=types, strings_can_be_null=False))
    writer = None
    count, frauds, maximum = 0, 0, 0
    try:
        for batch in reader:
            if any(column.null_count for column in batch.columns):
                raise ValueError("Missing required source value")
            table = pa.Table.from_batches([batch])
            cents = pc.cast(pc.multiply(table["amount"], pa.scalar(
                Decimal(100), type=pa.decimal128(3, 0))), pa.int64(), safe=True)
            labels = table["isFraud"].to_numpy()
            if not np.isin(labels, [0, 1]).all():
                raise ValueError("Non-binary source label")
            if any(pc.equal(table[col], "").to_numpy().any()
                   for col in ["type", "nameOrig", "nameDest"]):
                raise ValueError("Empty source category/entity")
            maximum = max(maximum, int(pc.max(pc.abs(cents)).as_py()))
            frauds += int(labels.sum())
            table = table.drop(["amount"]).append_column("amount_cents", cents)
            table = table.add_column(0, "source_row_id", pa.array(
                np.arange(count, count + len(table), dtype=np.int64)))
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=250000)
            count += len(table)
    finally:
        if writer is not None:
            writer.close()
    if not count or not 0 < frauds < count:
        raise ValueError("Expected nonempty source with both classes")
    if count * count * maximum * maximum >= 2**127:
        raise OverflowError("Exact moment numerator can overflow signed HUGEINT")
    return {"rows": count, "frauds": frauds, "max_abs_cents": maximum,
            "source_sha256": file_hash(source), "source_id_min": 0,
            "source_id_max": count - 1, "csv_block_size": block_size}


def history_query(source: Path, windows: tuple[int, ...] = (1, 6, 24)) -> str:
    """Integer moments commute across thread/scan order; same-step peers are excluded."""
    if not windows or any(type(w) is not int or w < 1 for w in windows):
        raise ValueError("Positive integer windows required")
    aggregates, statistics, output = [], [], []
    for window in windows:
        suffix = f"{window}h"
        frame = f"ORDER BY step RANGE BETWEEN {window} PRECEDING AND 1 PRECEDING"
        over = f"OVER (PARTITION BY nameDest {frame})"
        aggregates.extend([
            f"COUNT(*) {over} AS n_{suffix}",
            f"SUM(cents) {over} AS s_{suffix}",
            f"SUM(cents*cents) {over} AS ss_{suffix}",
            f"MAX(cents) {over} AS mx_{suffix}",
            f"COUNT(DISTINCT nameOrig) {over} AS unique_origins_{suffix}",
            f"COUNT(*) OVER (PARTITION BY nameOrig,nameDest {frame}) AS pair_count_{suffix}",
        ])
        n, total, squared = f"n_{suffix}", f"s_{suffix}", f"ss_{suffix}"
        statistics.extend([
            f"{n} AS dest_count_{suffix}",
            f"CASE WHEN {n}>0 THEN {total}::DOUBLE/(100.0*{n}) ELSE 0.0 END "
            f"AS dest_amount_mean_{suffix}",
            f"COALESCE(mx_{suffix}::DOUBLE/100.0,0.0) AS dest_amount_max_{suffix}",
            f"CASE WHEN {n}>1 THEN sqrt(({n}::HUGEINT*{squared}-{total}*{total})::DOUBLE"
            f"/({n}::HUGEINT*({n}-1))::DOUBLE)/100.0 ELSE 0.0 END AS dest_amount_std_{suffix}",
        ])
        output.extend([f"dest_count_{suffix}", f"dest_amount_mean_{suffix}",
                       f"dest_amount_max_{suffix}", f"dest_amount_std_{suffix}",
                       f"unique_origins_{suffix}", f"pair_count_{suffix}"])
    ratios = [f"COALESCE(amount/NULLIF(dest_amount_mean_{w}h,0),0.0) "
              f"AS dest_amount_ratio_{w}h" for w in windows]
    return f"""
      WITH source AS (
        SELECT *, amount_cents::HUGEINT AS cents, amount_cents::DOUBLE/100.0 AS amount
        FROM read_parquet({literal(source)})
      ), history AS (
        SELECT *, {', '.join(aggregates)},
          MAX(step) OVER (PARTITION BY nameDest ORDER BY step
            RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_dest_step
        FROM source
      ), statistics AS (SELECT *, {', '.join(statistics)} FROM history)
      SELECT source_row_id, step, type, amount, isFraud,
        ((step-1)%24)::INTEGER AS hour_of_day,
        floor((step-1)/24)::INTEGER AS day_index,
        COALESCE(step-previous_dest_step,-1)::INTEGER AS time_since_last_dest,
        {', '.join(output)}, {', '.join(ratios)}
      FROM statistics ORDER BY step,source_row_id
    """  # nosec B608


def prepare(source: Path, folder: Path, *, threads: int, reverse_scan: bool,
            block_size: int, expected_hash: str | None = None) -> dict[str, Any]:
    if folder.exists():
        raise FileExistsError(folder)
    if expected_hash is not None and file_hash(source) != expected_hash:
        raise ValueError("Raw input SHA-256 mismatch")
    if threads not in [1, 16]:
        raise ValueError("Only the frozen thread settings are allowed")
    folder.mkdir(parents=True)
    start = time.perf_counter()
    profile = ingest_csv(source, folder / "source.parquet", block_size)
    scan = folder / "source.parquet"
    with duckdb.connect() as connection:
        connection.execute(f"SET threads={threads}")
        connection.execute("SET preserve_insertion_order=true")
        connection.execute("SET memory_limit='8GB'")
        connection.execute(f"SET temp_directory={literal(folder / 'duckdb_tmp')}")
        if reverse_scan:
            scan = folder / "reverse_scan.parquet"
            connection.execute(
                f"COPY (SELECT * FROM read_parquet({literal(folder / 'source.parquet')}) "  # nosec B608
                f"ORDER BY source_row_id DESC) TO {literal(scan)} (FORMAT PARQUET)"
            )  # nosec B608
        output = folder / "features.parquet"
        connection.execute(f"COPY ({history_query(scan)}) TO {literal(output)} "
                           "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)")  # nosec B608
    profile.update(threads=threads, reverse_scan=reverse_scan,
                   feature_sha256=file_hash(output), seconds=time.perf_counter() - start)
    (folder / "BUILD_REPORT.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def compare_parquets(left: Path, right: Path) -> dict[str, Any]:
    """Check every typed value in canonical order without loading both full tables."""
    first, second = pq.ParquetFile(left), pq.ParquetFile(right)
    if first.schema_arrow != second.schema_arrow:
        raise ValueError("Feature schemas differ")
    total = 0
    for a, b in zip(first.iter_batches(batch_size=100000),
                    second.iter_batches(batch_size=100000), strict=True):
        if a.num_rows != b.num_rows:
            raise ValueError("Mismatched record batch sizes")
        for name, x, y in zip(a.schema.names, a.columns, b.columns, strict=True):
            if x.null_count or y.null_count:
                raise ValueError(f"Unexpected null: {name}")
            xx, yy = x.to_numpy(zero_copy_only=False), y.to_numpy(zero_copy_only=False)
            if not np.array_equal(xx, yy):
                raise ValueError(f"Value mismatch: {name}, batch starting {total}")
            if xx.dtype.kind in "fiu" and xx.tobytes() != yy.tobytes():
                raise ValueError(f"Bit mismatch: {name}, batch starting {total}")
        total += a.num_rows
    return {"status": "PASS_EXACT_VALUES", "rows": total,
            "columns": first.schema_arrow.names, "left_sha256": file_hash(left),
            "right_sha256": file_hash(right), "same_bytes": file_hash(left) == file_hash(right)}
