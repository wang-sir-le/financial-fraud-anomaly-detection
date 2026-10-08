"""Artificial identity-bound comparator; never used to alter production features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from native_common import HEADER, HISTORY, WINDOWS, ProtocolError, require

PUBLIC_COLUMNS = [*HEADER[:-1], "amount_cents", "split"]


def indexed(frame: pd.DataFrame, expected_ids: set[int], expected_n: int) -> pd.DataFrame:
    require(len(frame) == expected_n, "row count changed or a source row was omitted")
    require("source_row_id" in frame, "bound source identity missing")
    require(pd.api.types.is_integer_dtype(frame["source_row_id"]), "source identity is not integer")
    require(frame["source_row_id"].is_unique, "duplicate source identity")
    require(set(frame["source_row_id"].astype(int)) == expected_ids, "source ID set changed")
    return frame.set_index("source_row_id", verify_integrity=True).sort_index()


def bound_public(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    require(reference["source_row_id"].is_unique, "reference identities not unique")
    ids = set(reference["source_row_id"].astype(int))
    expected = indexed(reference, ids, len(reference))
    actual = indexed(candidate, ids, len(reference))
    require(set(PUBLIC_COLUMNS) <= set(actual), "public source fields missing")
    try:
        pd.testing.assert_frame_equal(
            expected[PUBLIC_COLUMNS],
            actual[PUBLIC_COLUMNS],
            check_exact=True,
            check_dtype=True,
            check_names=True,
        )
    except AssertionError as exc:
        raise ProtocolError("an ID is bound to different public transaction content") from exc
    return actual


def aligned_history(
    public: pd.DataFrame,
    reference: pd.DataFrame,
    reference_audit: dict[str, Any],
    actual: pd.DataFrame,
    audit: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    require(mode in ("HS", "HL"), "unexpected history comparison mode")
    expected = bound_public(public, reference)
    candidate = bound_public(public, actual)
    require(
        set(HISTORY) <= set(reference) and set(HISTORY) <= set(actual),
        "history feature outputs omitted",
    )
    left, right = expected[HISTORY].to_numpy(), candidate[HISTORY].to_numpy()
    require(
        bool(np.isfinite(left).all())
        and bool(np.isfinite(right).all())
        and bool(np.allclose(left, right, rtol=1e-12, atol=1e-12)),
        "features are not aligned with the same source identities",
    )
    sources = audit["used_history"]
    original_sources = reference_audit["used_history"]
    expected_keys = {str(int(i)) for i in expected.index}
    require(
        set(sources) == set(original_sources) == expected_keys,
        "history provenance omitted or added source identities",
    )
    require(sources == original_sources, "history provenance is assigned to different transactions")
    for identity, record in sources.items():
        current = expected.loc[int(identity)]
        step = int(current["step"])
        require(
            set(record) == {"h1", "h7", "h30", "recency_source"},
            "history provenance schema differs",
        )
        for window in WINDOWS:
            used = record[f"h{window}"]
            require(len(used) == len(set(used)), "history source identity duplicated")
            span = min(window, 7) if mode == "HS" else window
            for source in used:
                require(int(source) in expected.index, "unknown history source identity")
                past = expected.loc[int(source)]
                require(
                    past["merchant"] == current["merchant"]
                    and max(0, step - span) <= int(past["step"]) < step,
                    "source violates merchant, current-step isolation or window",
                )
        recency = record["recency_source"]
        if recency is not None:
            require(int(recency) in expected.index, "unknown recency source identity")
            past = expected.loc[int(recency)]
            lower = max(0, step - 7) if mode == "HS" else 0
            require(
                past["merchant"] == current["merchant"] and lower <= int(past["step"]) < step,
                "recency source violates access or strict-past rules",
            )
    return {
        "rows": len(actual),
        "unique_IDs": True,
        "same_ID_set": True,
        "public_content_exact": True,
        "one_to_one_output_alignment": True,
        "features_rtol": 1e-12,
        "features_atol": 1e-12,
        "provenance_exact": True,
    }
