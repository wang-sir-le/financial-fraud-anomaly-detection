"""Windows job membership and sampled process supervision, not full OS isolation."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

import psutil
from native_common import ROOT, read_json, require, utc, write_json

KERNEL = ctypes.WinDLL("kernel32", use_last_error=True)


class BasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_longlong),
        ("job_time", ctypes.c_longlong),
        ("flags", wintypes.DWORD),
        ("min_ws", ctypes.c_size_t),
        ("max_ws", ctypes.c_size_t),
        ("active_processes", wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority", wintypes.DWORD),
        ("scheduling", wintypes.DWORD),
    ]


class IOCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "read_ops",
            "write_ops",
            "other_ops",
            "read_bytes",
            "write_bytes",
            "other_bytes",
        )
    ]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", BasicLimits),
        ("io", IOCounters),
        ("process_memory", ctypes.c_size_t),
        ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


class ProcessList(ctypes.Structure):
    _fields_ = [
        ("assigned", wintypes.DWORD),
        ("in_list", wintypes.DWORD),
        ("pids", ctypes.c_size_t * 256),
    ]


KERNEL.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
KERNEL.CreateJobObjectW.restype = wintypes.HANDLE
KERNEL.SetInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]
KERNEL.SetInformationJobObject.restype = wintypes.BOOL
KERNEL.QueryInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.c_void_p,
]
KERNEL.QueryInformationJobObject.restype = wintypes.BOOL
KERNEL.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
KERNEL.AssignProcessToJobObject.restype = wintypes.BOOL
KERNEL.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
KERNEL.TerminateJobObject.restype = wintypes.BOOL
KERNEL.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
KERNEL.OpenProcess.restype = wintypes.HANDLE
KERNEL.CloseHandle.argtypes = [wintypes.HANDLE]
KERNEL.CloseHandle.restype = wintypes.BOOL
KERNEL.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
KERNEL.IsProcessInJob.restype = wintypes.BOOL


def checked(ok: Any) -> None:
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())


class Job:
    def __init__(self) -> None:
        self.handle = KERNEL.CreateJobObjectW(None, None)
        checked(self.handle)
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        checked(
            KERNEL.SetInformationJobObject(
                self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            )
        )

    def assign_pid(self, pid: int) -> None:
        handle = KERNEL.OpenProcess(0x0100 | 0x0001 | 0x0400, False, pid)
        checked(handle)
        try:
            already = wintypes.BOOL()
            checked(KERNEL.IsProcessInJob(handle, self.handle, ctypes.byref(already)))
            if not already.value:
                checked(KERNEL.AssignProcessToJobObject(self.handle, handle))
        finally:
            KERNEL.CloseHandle(handle)

    def pids(self) -> list[int]:
        record = ProcessList()
        checked(
            KERNEL.QueryInformationJobObject(
                self.handle, 3, ctypes.byref(record), ctypes.sizeof(record), None
            )
        )
        require(record.in_list <= 256, "job process list exceeds fixed bound")
        return [int(record.pids[i]) for i in range(record.in_list)]

    def terminate(self) -> None:
        checked(KERNEL.TerminateJobObject(self.handle, 91))

    def close(self) -> None:
        checked(KERNEL.CloseHandle(self.handle))


def directory_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                pass
    return total


def process_sample(pids: list[int]) -> dict[str, Any]:
    rss = read_bytes = write_bytes = 0
    cpu: float = 0
    per_pid = []
    for pid in sorted(set(pids)):
        try:
            process = psutil.Process(pid)
            memory = process.memory_info()
            times = process.cpu_times()
            io = process.io_counters()
            rss += memory.rss
            cpu += times.user + times.system
            read_bytes += io.read_bytes
            write_bytes += io.write_bytes
            per_pid.append(
                {
                    "pid": pid,
                    "create_time": process.create_time(),
                    "rss": memory.rss,
                    "cpu": times.user + times.system,
                    "read_bytes": io.read_bytes,
                    "write_bytes": io.write_bytes,
                    "os_peak_working_set": int(getattr(memory, "peak_wset", 0)),
                }
            )
        except psutil.NoSuchProcess:
            continue
    return {
        "rss": rss,
        "cpu": cpu,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "per_pid": per_pid,
    }


def cumulative_observed_cpu(samples: list[dict[str, Any]]) -> float:
    maxima: dict[tuple[int, float], float] = {}
    for sample in samples:
        for process in sample["per_pid"]:
            identity = (process["pid"], process["create_time"])
            maxima[identity] = max(maxima.get(identity, 0.0), process["cpu"])
    return sum(maxima.values())


def run_supervised(
    action: str,
    payload: Path,
    name: str,
    *,
    seconds: float,
    rss_limit: int,
    disk_limit: int,
    include_self: bool = False,
    global_caps: dict[str, Any] | None = None,
    artificial_resource_arm: Path | None = None,
) -> dict[str, Any]:
    """One launch, no retries. Gate binds native PID and launcher to a job before work."""
    folder = ROOT / "audit" / name
    require(not folder.exists(), "supervised attempt already exists")
    folder.mkdir(parents=True)
    ack, gate = folder / "ack.json", folder / "gate.txt"
    command = [
        read_json(ROOT / "config/EXECUTION_CONFIG.json")["python"],
        "-B",
        "-I",
        str(ROOT / "code/entry.py"),
        "--action",
        action,
        "--payload",
        str(payload),
        "--ack",
        str(ack),
        "--gate",
        str(gate),
    ]
    job = Job()
    started = time.perf_counter()
    start_utc = utc()
    samples: list[dict[str, Any]] = []
    reason = None
    process = None
    last_sample = started
    cleanup_pids: list[int] = []
    cpu_maxima: dict[tuple[int, float], float] = {}
    with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(  # nosec B603: fixed owned command, no shell
                command,
                stdout=stdout,
                stderr=stderr,
                cwd=str(ROOT / "code"),
                creationflags=subprocess.CREATE_NO_WINDOW,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "MPLCONFIGDIR": str(ROOT / "audit/mpl"),
                },
            )
            # Assign launcher if still alive; acknowledgement covers the actual native PID.
            try:
                job.assign_pid(process.pid)
            except OSError:
                require(process.poll() is not None and ack.exists(), "launcher could not enter job")
            gate_released = False
            while True:
                now = time.perf_counter()
                pids = job.pids()
                if ack.exists() and not gate_released:
                    native_pid = int(read_json(ack)["pid"])
                    job.assign_pid(native_pid)
                    pids = job.pids()
                    require(native_pid in pids, "native worker not recorded in job")
                    gate.write_text("GO\n", encoding="ascii")
                    gate_released = True
                sample = process_sample(pids + ([os.getpid()] if include_self else []))
                sample.update(
                    {
                        "seconds": now - started,
                        "interval_since_previous": now - last_sample,
                        "disk_bytes": directory_bytes(ROOT),
                        "available_memory": psutil.virtual_memory().available,
                    }
                )
                last_sample = now
                samples.append(sample)
                for observed in sample["per_pid"]:
                    identity = (observed["pid"], observed["create_time"])
                    cpu_maxima[identity] = max(cpu_maxima.get(identity, 0.0), observed["cpu"])
                with (folder / "RSS_CPU_DISK.jsonl").open("a", encoding="utf-8") as stream:
                    import json

                    stream.write(json.dumps(sample, allow_nan=False) + "\n")
                if now - started >= seconds:
                    reason = "TIME_LIMIT"
                elif sample["rss"] >= rss_limit and (
                    artificial_resource_arm is None or artificial_resource_arm.exists()
                ):
                    reason = "RSS_LIMIT"
                elif sample["disk_bytes"] > disk_limit:
                    reason = "DISK_LIMIT"
                elif (
                    global_caps is not None
                    and sample["available_memory"] < global_caps["stop_free_memory_bytes"]
                ):
                    reason = "AVAILABLE_MEMORY_LIMIT"
                elif (
                    global_caps is not None
                    and sum(cpu_maxima.values()) >= global_caps["cpu_seconds"]
                ):
                    reason = "CPU_LIMIT"
                if reason is not None:
                    cleanup_pids = job.pids()
                    job.terminate()
                    process.wait(timeout=5)
                    break
                code = process.poll()
                if code is not None and not job.pids():
                    reason = "NORMAL_EXIT" if code == 0 else "NONZERO_EXIT"
                    break
                # Launcher can exit while the actual gated native worker is still alive.
                if code is not None and not gate_released and not ack.exists():
                    reason = "STARTUP_EXIT_BEFORE_ACK"
                    break
                time.sleep(0.05)
        except BaseException as exc:
            reason = f"SUPERVISOR_ERROR:{type(exc).__name__}:{exc}"
            cleanup_pids = job.pids()
            job.terminate()
            if process is not None:
                process.wait(timeout=5)
        finally:
            remaining = job.pids()
            if remaining:
                cleanup_pids.extend(remaining)
                job.terminate()
            cleanup_deadline = time.perf_counter() + 3.0
            while any(psutil.pid_exists(pid) for pid in set(cleanup_pids)):
                if time.perf_counter() >= cleanup_deadline:
                    break
                time.sleep(0.02)
            job.close()
    ended = time.perf_counter()
    survivors = [pid for pid in set(cleanup_pids) if psutil.pid_exists(pid)]
    record = {
        "action": action,
        "start_utc": start_utc,
        "end_utc": utc(),
        "seconds": ended - started,
        "stop_reason": reason,
        "exitcode": process.returncode if process is not None else None,
        "sampled_peak_rss": max((s["rss"] for s in samples), default=0),
        "max_sample_interval": max((s["interval_since_previous"] for s in samples), default=0),
        "observed_cumulative_cpu": cumulative_observed_cpu(samples),
        "max_new_directory_bytes": max((s["disk_bytes"] for s in samples), default=0),
        "cleanup_pid_survivors": survivors,
        "strict_instantaneous_peak_guarantee": False,
        "terminated_pids": sorted(set(cleanup_pids)),
        "ack_native_pid": int(read_json(ack)["pid"]) if ack.exists() else None,
        "gate_released": gate.exists(),
        "artificial_resource_arm": str(artificial_resource_arm)
        if artificial_resource_arm
        else None,
        "os_isolation_claim": False,
        "samples": len(samples),
    }
    write_json(folder / "RESOURCE.json", record, exclusive=True)
    return record


def startup_resources() -> dict[str, Any]:
    return {
        "time_utc": utc(),
        "available_memory": psutil.virtual_memory().available,
        "physical_memory": psutil.virtual_memory().total,
        "logical_processors": psutil.cpu_count(),
        "D_free": shutil.disk_usage("D:/").free,
    }
