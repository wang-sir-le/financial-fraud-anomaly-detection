"""Identity, leakage and arithmetic checks for the new explicitly versioned route."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudx.paysim_deterministic.preprocess import compare_parquets, prepare


def source(path: Path, rows: list[tuple[object, ...]]) -> Path:
    pd.DataFrame(rows, columns=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]).to_csv(path, index=False)
    return path


def test_exact_cent_history_and_scan_thread_invariance(tmp_path: Path) -> None:
    raw = source(tmp_path / "raw.csv", [
        (1, "TRANSFER", 10.01, "A", "Z", 0),
        (1, "TRANSFER", 20.03, "B", "Z", 1),
        (2, "TRANSFER", 30.05, "A", "Z", 0),
        (2, "TRANSFER", 40.07, "C", "Z", 1),
        (8, "TRANSFER", 50.09, "D", "Z", 0),
    ])
    prepare(raw, tmp_path / "a", threads=1, reverse_scan=False, block_size=128)
    prepare(raw, tmp_path / "b", threads=16, reverse_scan=True, block_size=256)
    assert compare_parquets(tmp_path / "a/features.parquet", tmp_path / "b/features.parquet")["rows"] == 5
    frame = pd.read_parquet(tmp_path / "a/features.parquet")
    assert frame.source_row_id.tolist() == list(range(5))
    assert frame.dest_count_1h.tolist() == [0, 0, 2, 2, 0]
    assert frame.dest_count_6h.tolist() == [0, 0, 2, 2, 2]
    assert frame.unique_origins_24h.tolist() == [0, 0, 2, 2, 3]
    np.testing.assert_allclose(frame.loc[[2, 3], "dest_amount_mean_1h"], 15.02)
    np.testing.assert_allclose(frame.loc[[2, 3], "dest_amount_std_1h"], np.std([10.01, 20.03], ddof=1))
    assert frame.pair_count_1h.tolist() == [0, 0, 1, 0, 0]
    assert frame.time_since_last_dest.tolist() == [-1, -1, 1, 1, 6]


def test_future_and_label_mutations_do_not_change_past_features(tmp_path: Path) -> None:
    rows = [(1, "TRANSFER", 10.0, "A", "Z", 0), (2, "TRANSFER", 20.0, "B", "Z", 1),
            (2, "TRANSFER", 99.0, "C", "Z", 0), (9, "TRANSFER", 100.0, "D", "Z", 1)]
    changed = [(s, t, 9876.0 if s == 9 else a, o, d, 1-y) for s, t, a, o, d, y in rows]
    for name, values in [("a", rows), ("b", changed)]:
        prepare(source(tmp_path / f"{name}.csv", values), tmp_path / name,
                threads=1, reverse_scan=False, block_size=256)
    a, b = [pd.read_parquet(tmp_path / name / "features.parquet") for name in ["a", "b"]]
    pd.testing.assert_frame_equal(a.iloc[:3].drop(columns="isFraud"), b.iloc[:3].drop(columns="isFraud"))
    # Both step-2 rows see only step1; changing the other peer cannot enter history.
    assert a.dest_amount_mean_1h.tolist()[:3] == [0.0, 10.0, 10.0]


def test_duplicate_records_retain_source_identity(tmp_path: Path) -> None:
    raw = source(tmp_path / "raw.csv", [(2, "TRANSFER", 1.0, "A", "Z", 0),
        (1, "TRANSFER", 2.0, "B", "Z", 1), (2, "TRANSFER", 1.0, "A", "Z", 0)])
    prepare(raw, tmp_path / "out", threads=1, reverse_scan=False, block_size=256)
    frame = pd.read_parquet(tmp_path / "out/features.parquet")
    assert frame.source_row_id.tolist() == [1, 0, 2]
    assert len(frame) == 3


def test_fractional_cent_rejected(tmp_path: Path) -> None:
    raw = source(tmp_path / "raw.csv", [(1, "TRANSFER", 1.001, "A", "Z", 0),
                                         (2, "TRANSFER", 2.0, "B", "Z", 1)])
    with pytest.raises(Exception, match="Rescaling|rescaling|precision|scale"):
        prepare(raw, tmp_path / "out", threads=1, reverse_scan=False, block_size=256)
