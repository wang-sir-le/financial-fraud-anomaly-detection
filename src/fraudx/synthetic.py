"""PaySim-shaped synthetic data for fast integration tests only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_paysim(
    rows: int = 2_000,
    steps: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate repeated accounts and a late fraud-pattern drift."""
    if rows < 200 or steps < 10:
        raise ValueError("Synthetic smoke data must be large enough for temporal evaluation")
    rng = np.random.default_rng(seed)
    step = np.repeat(np.arange(steps), int(np.ceil(rows / steps)))[:rows]
    origin_index = rng.integers(0, 80, size=rows)
    destination_index = rng.integers(0, 45, size=rows)
    transaction_type = rng.choice(
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"],
        size=rows,
        p=[0.60, 0.12, 0.23, 0.05],
    )
    amount = rng.lognormal(mean=5.0, sigma=1.0, size=rows)
    late = step >= int(steps * 0.7)
    fraud_score = (
        1.4 * np.isin(transaction_type, ["TRANSFER", "CASH_OUT"])
        + 1.1 * (amount > np.quantile(amount, 0.92))
        + 1.2 * late * np.isin(transaction_type, ["PAYMENT", "DEBIT"])
        + rng.normal(0, 0.5, size=rows)
    )
    cutoff = np.quantile(fraud_score, 0.94)
    fraud = (fraud_score >= cutoff).astype(int)
    old_origin = rng.uniform(500, 20_000, size=rows)
    new_origin = np.maximum(0, old_origin - amount)
    old_dest = rng.uniform(0, 30_000, size=rows)
    new_dest = old_dest + amount
    fraud_mask = fraud.astype(bool)
    new_origin[fraud_mask] = old_origin[fraud_mask]
    new_dest[fraud_mask] = old_dest[fraud_mask]
    return pd.DataFrame(
        {
            "step": step,
            "type": transaction_type,
            "amount": amount,
            "nameOrig": [f"C{value:04d}" for value in origin_index],
            "oldbalanceOrg": old_origin,
            "newbalanceOrig": new_origin,
            "nameDest": [f"M{value:04d}" for value in destination_index],
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": fraud,
            "isFlaggedFraud": np.zeros(rows, dtype=int),
        }
    )

