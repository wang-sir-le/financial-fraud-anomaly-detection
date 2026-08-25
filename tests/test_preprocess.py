from pathlib import Path

import duckdb

from fraudx.preprocess import prepare_paysim_csv
from fraudx.synthetic import make_synthetic_paysim


def test_prepared_data_excludes_leakage_and_same_step_history(tmp_path: Path) -> None:
    source = tmp_path / "paysim.csv"
    output = tmp_path / "prepared.parquet"
    frame = make_synthetic_paysim(rows=500, steps=20)
    frame.to_csv(source, index=False)
    profile = prepare_paysim_csv(source, output, windows=(1, 6))
    assert profile.rows == 500
    connection = duckdb.connect()
    try:
        columns = {
            row[0]
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{output.as_posix()}')"
            ).fetchall()
        }
        first_step_max = connection.execute(
            f"SELECT MAX(dest_count_1h) FROM read_parquet('{output.as_posix()}') WHERE step=0"
        ).fetchone()
    finally:
        connection.close()
    assert first_step_max is not None
    assert first_step_max[0] == 0
    assert not {
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
    }.intersection(columns)
