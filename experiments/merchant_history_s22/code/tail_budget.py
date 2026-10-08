"""A shared final-state decision, including work performed after report generation."""

from __future__ import annotations

from typing import Any


def budget_decision(
    candidate_complete: bool, observed: dict[str, Any], caps: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "global_wall": observed["wall_seconds"] <= caps["global_wall_seconds"],
        "delivery_wall": observed["delivery_seconds"] <= caps["delivery_seconds"],
        "final_artifacts": observed["directory_bytes"] <= caps["disk_bytes"],
        "final_RSS": observed["rss_bytes"] < caps["rss_bytes"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    complete = candidate_complete and not failures
    return {
        "status": "COMPLETED" if complete else "STOPPED_NO_RETRY",
        "exit_code": 0 if complete else 1,
        "limits_ok": not failures,
        "checks": checks,
        "failures": failures,
        "observed": observed,
    }
