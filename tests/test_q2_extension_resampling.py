"""Random stream identities and direct processing difference checks."""

import numpy as np

from fraudx.ieee_cis_timeblock_bootstrap import moving_block_positions as ieee_positions
from fraudx.q2_extension.bootstrap import interval_row
from fraudx.q2_extension.resampling import draw_batches
from fraudx.timeblock_bootstrap import moving_block_positions as pay_positions


def test_pay_rng_identity() -> None:
    child = np.random.SeedSequence(20260820).spawn(3)[1]
    expected = pay_positions(31, 3, 101, np.random.default_rng(child))
    np.testing.assert_array_equal(next(draw_batches("paysim", 2, 31, 101)), expected)


def test_ieee_rng_identity_across_32_draw_boundary() -> None:
    child = np.random.SeedSequence(20260822).spawn(3)[0]
    seed = int(child.generate_state(1, dtype=np.uint64)[0])
    rng = np.random.default_rng(seed)
    expected = np.concatenate([ieee_positions(27, 3, n, rng) for n in [32, 32, 1]])
    actual = np.concatenate(list(draw_batches("ieee_cis", 1, 27, 65)))
    np.testing.assert_array_equal(actual, expected)


def test_interval_retains_undefined_count_and_is_pointwise() -> None:
    row = interval_row(np.array([1., 2., 3., np.nan]), 2., {"metric": "x"})
    assert row["draws"] == 4 and row["valid"] == 3 and row["invalid"] == 1
    assert row["ci_lower"] == 1.05 and row["ci_upper"] == 2.95
    assert "pointwise" in row["ci_type"]
