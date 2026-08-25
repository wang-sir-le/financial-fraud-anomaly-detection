"""Out-of-core, leakage-resistant PaySim preprocessing with DuckDB."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from fraudx.data import REQUIRED_COLUMNS


@dataclass(frozen=True)
class PreparedDataProfile:
    """Integrity and suitability statistics for a prepared PaySim dataset."""

    rows: int
    fraud_rows: int
    fraud_rate: float
    min_step: int
    max_step: int
    core_null_rows: int
    invalid_label_rows: int
    feature_columns: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def prepare_paysim_csv(
    source_path: Path,
    output_path: Path,
    windows: tuple[int, ...] = (1, 6, 24),
    force: bool = False,
) -> PreparedDataProfile:
    """Create a compact Parquet table without balances or contemporaneous leakage."""
    source = source_path.resolve()
    output = output_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PaySim CSV not found: {source}")
    clean_windows = tuple(sorted({int(window) for window in windows}))
    if not clean_windows or clean_windows[0] <= 0:
        raise ValueError("All windows must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        return profile_prepared_data(output)

    temp_output = output.with_suffix(".tmp.parquet")
    if temp_output.exists():
        temp_output.unlink()
    temp_directory = output.parent / ".duckdb_tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(f"SET temp_directory={_sql_literal(temp_directory)}")  # nosec B608
        connection.execute("SET preserve_insertion_order=false")
        columns = _source_columns(connection, source)
        missing = REQUIRED_COLUMNS.difference(columns)
        if missing:
            raise ValueError(f"Missing PaySim columns: {sorted(missing)}")
        query = _prepared_query(source, clean_windows)
        copy_query = (
            f"COPY ({query}) TO {_sql_literal(temp_output)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
        )  # nosec B608
        connection.execute(copy_query)
    finally:
        connection.close()
    temp_output.replace(output)
    return profile_prepared_data(output)


def load_prepared_frame(path: Path) -> Any:
    """Load the compact prepared table after checking its integrity."""
    profile_prepared_data(path)
    connection = duckdb.connect(database=":memory:")
    try:
        return connection.execute(
            f"SELECT * FROM read_parquet({_sql_literal(path.resolve())})"  # nosec B608
        ).df()
    finally:
        connection.close()


def profile_prepared_data(path: Path) -> PreparedDataProfile:
    """Validate row-level integrity without materializing the table in memory."""
    parquet = path.resolve()
    if not parquet.is_file():
        raise FileNotFoundError(f"Prepared PaySim data not found: {parquet}")
    connection = duckdb.connect(database=":memory:")
    try:
        relation = f"read_parquet({_sql_literal(parquet)})"
        row = connection.execute(
            f"""
            SELECT
                COUNT(*) AS rows,
                SUM(isFraud)::BIGINT AS fraud_rows,
                AVG(isFraud)::DOUBLE AS fraud_rate,
                MIN(step)::INTEGER AS min_step,
                MAX(step)::INTEGER AS max_step,
                SUM(CASE WHEN step IS NULL OR type IS NULL OR amount IS NULL
                    OR isFraud IS NULL THEN 1 ELSE 0 END)::BIGINT AS core_null_rows,
                SUM(CASE WHEN isFraud NOT IN (0, 1) THEN 1 ELSE 0 END)::BIGINT
                    AS invalid_label_rows
            FROM {relation}
            """  # nosec B608
        ).fetchone()
        column_count = len(
            connection.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(parquet)]).description
        )
    finally:
        connection.close()
    if row is None or int(row[0]) == 0:
        raise ValueError("Prepared PaySim data is empty")
    profile = PreparedDataProfile(
        rows=int(row[0]),
        fraud_rows=int(row[1]),
        fraud_rate=float(row[2]),
        min_step=int(row[3]),
        max_step=int(row[4]),
        core_null_rows=int(row[5]),
        invalid_label_rows=int(row[6]),
        feature_columns=column_count,
    )
    if profile.fraud_rows == 0 or profile.fraud_rows == profile.rows:
        raise ValueError("Prepared data must contain both classes")
    if profile.core_null_rows or profile.invalid_label_rows:
        raise ValueError(f"Prepared data failed integrity checks: {profile.to_dict()}")
    return profile


def write_profile(profile: PreparedDataProfile, path: Path) -> None:
    """Persist an auditable preprocessing summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _source_columns(connection: duckdb.DuckDBPyConnection, source: Path) -> set[str]:
    rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto({_sql_literal(source)})"  # nosec B608
    ).fetchall()
    return {str(row[0]) for row in rows}


def _prepared_query(source: Path, windows: tuple[int, ...]) -> str:
    expressions: list[str] = []
    ratio_expressions: list[str] = []
    for window in windows:
        frame = f"RANGE BETWEEN {window} PRECEDING AND 1 PRECEDING"
        suffix = f"{window}h"
        expressions.extend(
            [
                "COUNT(*) OVER "
                f"(PARTITION BY nameDest ORDER BY step {frame}) AS dest_count_{suffix}",
                "AVG(amount) OVER "
                f"(PARTITION BY nameDest ORDER BY step {frame}) AS dest_amount_mean_{suffix}",
                "MAX(amount) OVER "
                f"(PARTITION BY nameDest ORDER BY step {frame}) AS dest_amount_max_{suffix}",
                "STDDEV_SAMP(amount) OVER "
                f"(PARTITION BY nameDest ORDER BY step {frame}) AS dest_amount_std_{suffix}",
                "COUNT(DISTINCT nameOrig) OVER "
                f"(PARTITION BY nameDest ORDER BY step {frame}) AS unique_origins_{suffix}",
                "COUNT(*) OVER "
                f"(PARTITION BY nameOrig, nameDest ORDER BY step {frame}) AS pair_count_{suffix}",
            ]
        )
        ratio_expressions.append(
            f"COALESCE(amount / NULLIF(dest_amount_mean_{suffix}, 0), 0) "
            f"AS dest_amount_ratio_{suffix}"
        )
    template = """
        WITH source AS (
            SELECT
                step::INTEGER AS step,
                type::VARCHAR AS type,
                amount::DOUBLE AS amount,
                nameOrig::VARCHAR AS nameOrig,
                nameDest::VARCHAR AS nameDest,
                isFraud::INTEGER AS isFraud
            FROM read_csv_auto({source}, header=true)
        ), history AS (
            SELECT
                *,
                {expressions},
                MAX(step) OVER (
                    PARTITION BY nameDest ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS previous_dest_step
            FROM source
        )
        SELECT
            step,
            type,
            amount,
            isFraud,
            ((step - 1) % 24)::INTEGER AS hour_of_day,
            FLOOR((step - 1) / 24)::INTEGER AS day_index,
            COALESCE(step - previous_dest_step, -1)::INTEGER AS time_since_last_dest,
            {features},
            {ratios}
        FROM history
        ORDER BY step
    """
    # All fragments come from validated positive integers; source is one escaped
    # local path literal and never supplies SQL syntax.
    return template.format(  # nosec B608
        source=_sql_literal(source),
        expressions=", ".join(expressions),
        features=", ".join(_coalesced_feature_names(windows)),
        ratios=", ".join(ratio_expressions),
    )


def _coalesced_feature_names(windows: tuple[int, ...]) -> list[str]:
    names: list[str] = []
    for window in windows:
        suffix = f"{window}h"
        for prefix in (
            "dest_count",
            "dest_amount_mean",
            "dest_amount_max",
            "dest_amount_std",
            "unique_origins",
            "pair_count",
        ):
            name = f"{prefix}_{suffix}"
            names.append(f"COALESCE({name}, 0) AS {name}")
    return names


def _sql_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"
