"""One authorized external entry, including artificial acceptance and final delivery."""

import time

SCRIPT_START_PERF = time.perf_counter()

import os  # noqa: E402
import sys  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import psutil  # noqa: E402
from native_common import ROOT, read_json, require, sha, utc, write_json  # noqa: E402
from supervision import directory_bytes, run_supervised, startup_resources  # noqa: E402
from tail_budget import budget_decision  # noqa: E402

DEV_START_UTC = "2026-10-06 10:49:52 UTC"


def main() -> int:
    process_creation = psutil.Process(os.getpid()).create_time()
    initialized = time.perf_counter()
    entry_start = initialized - max(0.0, time.time() - process_creation)
    config = read_json(ROOT / "config/EXECUTION_CONFIG.json")
    require(
        ROOT.resolve() == Path(config["execution_root"]).resolve(),
        "execution ROOT is not the new authorized run",
    )
    require(
        ROOT.resolve() != Path(config["restores_from_read_only_attempt"]).resolve(),
        "execution targets the old read-only attempt",
    )
    require(not (ROOT / "audit/ENTRY_PLAN.json").exists(), "this entry has already been attempted")
    startup = startup_resources()
    clocks: dict[str, Any] = {
        "entry_process_creation_utc": datetime.fromtimestamp(process_creation, UTC).isoformat(),
        "entry_clock": (
            "native outer process creation through final shutdown; process startup included"
        ),
        "script_first_instruction_perf": SCRIPT_START_PERF,
        "entry_initialization_seconds": initialized - entry_start,
        "global_wall_stop_seconds": config["caps"]["global_wall_seconds"],
        "worker_clock": (
            "before Popen through exit, termination and cleanup; imports and IO included"
        ),
        "sample_target_seconds": config["caps"]["interval_seconds"],
        "delivery_reserved_seconds": config["caps"]["delivery_seconds"],
        "scope": (
            "development outside entry separately disclosed; phase1 and real work share this entry"
        ),
    }
    dev_start = datetime.strptime(DEV_START_UTC, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=UTC)
    clocks["development_seconds"] = process_creation - dev_start.timestamp()
    write_json(
        ROOT / "audit/DEVELOPMENT_COST.json",
        {
            "start_utc": dev_start.isoformat(),
            "end_utc": clocks["entry_process_creation_utc"],
            "external_elapsed_seconds_including_tool_waits": clocks["development_seconds"],
            "work": (
                "source identity permutation comparator and negative regressions; "
                "final delivery budget/status repair; static checks"
            ),
            "excluded_from_execution_cap_but_not_free": True,
            "not_measured": [
                "human labour money",
                "LLM token cost",
                "development whole-system CPU/RSS",
            ],
        },
        exclusive=True,
    )
    write_json(
        ROOT / "audit/ENTRY_PLAN.json",
        {
            "time_utc": utc(),
            "clocks": clocks,
            "startup": startup,
            "caps": config["caps"],
            "sampling_blind_spots": [
                "50ms target is not a strict real-time or instantaneous memory/disk guarantee",
                (
                    "short-lived processes between samples can miss CPU; "
                    "IO counters are not physical sector reads"
                ),
                (
                    "gated acknowledgement and final cleanup are recorded, "
                    "not full OS security isolation"
                ),
                (
                    "outer native process and job members counted; "
                    "venv launcher startup handoff is a brief blind spot"
                ),
                (
                    "delivery after child closure is sampled at endpoints, "
                    "with a reserved 60-second budget"
                ),
            ],
            "final_code_sha256_preflight": {
                str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "code").glob("*.py"))
            },
        },
        exclusive=True,
    )
    resource: dict[str, Any] = {
        "stop_reason": "NOT_LAUNCHED",
        "seconds": 0.0,
        "sampled_peak_rss": 0,
        "max_new_directory_bytes": directory_bytes(ROOT),
        "samples": 0,
    }
    try:
        require(
            startup["available_memory"] >= config["caps"]["start_free_memory_bytes"],
            "startup available memory below 8 GiB",
        )
        require(
            startup["D_free"] >= config["caps"]["start_free_disk_bytes"],
            "startup D disk free space below 10 GiB",
        )
        deadline = entry_start + config["caps"]["global_wall_seconds"]
        controlled_seconds = deadline - time.perf_counter() - config["caps"]["delivery_seconds"]
        require(controlled_seconds > 0, "entry initialization exhausted execution budget")
        clocks["controlled_phase_allocated_seconds"] = controlled_seconds
        resource = run_supervised(
            "pipeline",
            ROOT / "config/EXECUTION_CONFIG.json",
            "global_execution",
            seconds=controlled_seconds,
            rss_limit=config["caps"]["rss_bytes"],
            disk_limit=config["caps"]["disk_bytes"],
            include_self=True,
            global_caps=config["caps"],
        )
    except BaseException as exc:
        resource["stop_reason"] = f"ENTRY_ERROR:{type(exc).__name__}:{exc}"
        write_json(
            ROOT / "audit/ENTRY_FAILURE.json",
            {"time_utc": utc(), "error": resource["stop_reason"], "retries": 0},
            exclusive=True,
        )
    delivery_start = time.perf_counter()
    clocks["external_elapsed_at_delivery_start_seconds"] = delivery_start - entry_start
    from delivery import write_delivery

    final = write_delivery(resource, clocks, startup)
    # All large reads/hashes and the initial report precede the final budget checks.
    manifest = {
        "time_utc": utc(),
        "root": str(ROOT),
        "final_status": "PENDING_FINAL_BUDGET",
        "sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for pattern in (
                "code/*.py",
                "config/*.json",
                "preprocessing/*.json",
                "models/*/model.txt",
                "results/*.json",
                "results/*.csv",
                "features/*identity.npy",
                "labels/*.npy",
            )
            for p in sorted(ROOT.glob(pattern))
        },
        "feature_matrix_hashes": "config/MODELS_LOCK.json and audit/COSTS.json",
        "resource_samples": "audit/*/RSS_CPU_DISK.jsonl",
        "no_automatic_followup": True,
    }
    write_json(ROOT / "ARTIFACT_MANIFEST.json", manifest, exclusive=True)

    def observed_tail() -> dict[str, Any]:
        return {
            "wall_seconds": time.perf_counter() - entry_start,
            "delivery_seconds": time.perf_counter() - delivery_start,
            "directory_bytes": directory_bytes(ROOT),
            "rss_bytes": psutil.Process().memory_info().rss,
        }

    def seal(decision: dict[str, Any]) -> None:
        final["status"] = decision["status"]
        if not decision["limits_ok"]:
            final.update(stopped_stage="delivery_limit", delivery_limit_failed=True)
            final["final_budget_failures_seen"] = sorted(
                set(final.get("final_budget_failures_seen", []) + decision["failures"])
            )
        final["final_budget_decision"] = decision
        clocks.update(
            {
                "delivery_seconds": decision["observed"]["delivery_seconds"],
                "external_entry_elapsed_before_exit_seconds": decision["observed"]["wall_seconds"],
                "new_directory_bytes_at_delivery_end": decision["observed"]["directory_bytes"],
                "limits_ok_at_exit": decision["limits_ok"],
                "end_utc": utc(),
            }
        )
        final["clocks"] = clocks
        report = ROOT / "六组回顾开发结果与下一步投入判断.md"
        content = report.read_text(encoding="utf-8").split("\n<!-- ENTRY_BUDGET_FOOTER -->")[0]
        lines = content.splitlines()
        lines[2] = (
            "**全部完成。**"
            if decision["status"] == "COMPLETED"
            else f"**在{final['stopped_stage']}阶段停止；不重试。**"
        ) + (
            f" 原生接入：{final['native_acceptance_status']}；"
            f"阶段二启动：{final['real_execution_started']}；"
            f"真实执行：{final['real_execution_status']}。"
        )
        footer = (
            "\n<!-- ENTRY_BUDGET_FOOTER -->\n"
            f"最终状态：{decision['status']}；退出码：{decision['exit_code']}；"
            f"外部墙钟{decision['observed']['wall_seconds']:.6f}秒；"
            f"新增目录{decision['observed']['directory_bytes']}字节；"
            f"尾部停止项：{decision['failures']}。\n"
        )
        report.write_text("\n".join(lines) + footer, encoding="utf-8")
        manifest["final_status"] = final["status"]
        write_json(ROOT / "ARTIFACT_MANIFEST.json", manifest)
        write_json(ROOT / "audit/ENTRY_CLOCK.json", clocks)
        write_json(
            ROOT / "audit/EXIT_CLOCK.json",
            {
                "time_utc": utc(),
                "external_entry_before_process_exit_seconds": decision["observed"]["wall_seconds"],
                "total_wall_within_1200": decision["checks"]["global_wall"],
                "fit_retries": 0,
                "final_status": final["status"],
                "exit_code": decision["exit_code"],
            },
        )
        write_json(ROOT / "FINAL_STATUS.json", final)

    candidate_complete = final["candidate_complete"]
    seal(budget_decision(candidate_complete, observed_tail(), config["caps"]))
    candidate_complete = candidate_complete and not final.get("delivery_limit_failed", False)
    # Includes final report, manifest and status writes, not merely a recorded Boolean.
    tail_decision = budget_decision(candidate_complete, observed_tail(), config["caps"])
    seal(tail_decision)
    candidate_complete = candidate_complete and not final.get("delivery_limit_failed", False)
    last_decision = budget_decision(candidate_complete, observed_tail(), config["caps"])
    if not last_decision["limits_ok"]:
        seal(last_decision)  # Only a downgrade; no execution retry.
    print(
        f"FINAL_STATUS={final['status']}; "
        f"external_wall_before_exit={last_decision['observed']['wall_seconds']:.6f}s",
        flush=True,
    )
    return 0 if final["status"] == "COMPLETED" and last_decision["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
