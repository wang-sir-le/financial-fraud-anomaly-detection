"""Gated process entry; no native library or project source is read before the gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--ack", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    temporary = args.ack.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"pid": os.getpid(), "parent_pid": os.getppid(), "executable": sys.executable}),
        encoding="utf-8",
    )
    temporary.replace(args.ack)
    waiting = time.perf_counter()
    while not args.gate.exists():
        if time.perf_counter() - waiting > 15:
            raise RuntimeError("gate not released")
        time.sleep(0.01)
    sys.path.insert(0, str(CODE))
    if args.action.startswith("toy_"):
        (ROOT / "artificial" / f"{args.action}.started").write_text(
            str(os.getpid()), encoding="ascii"
        )
        if args.action == "toy_normal":
            return 0
        if args.action == "toy_descendant":
            process = subprocess.Popen(  # nosec B603: fixed tiny sleeping child, no model
                [sys.executable, "-B", "-I", "-c", "import time; time.sleep(10)"],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            (ROOT / "artificial/descendant_pid.json").write_text(
                json.dumps({"pid": process.pid}), encoding="utf-8"
            )
            time.sleep(10)
        elif args.action == "toy_memory":
            held = bytearray(1024 * 1024)
            (ROOT / "artificial/toy_memory.ready").write_text(str(len(held)), encoding="ascii")
            time.sleep(10)
        elif args.action == "toy_disk":
            (ROOT / "artificial/toy_disk_payload.bin").write_bytes(b"0" * (128 * 1024))
            time.sleep(10)
        else:
            raise RuntimeError("unknown toy action")
        return 0
    if args.action == "pipeline":
        from pipeline import run_pipeline

        run_pipeline()
        return 0
    if args.action in ("train", "predict"):
        from model_worker import run_model_worker

        run_model_worker(args.action, args.payload)
        return 0
    raise RuntimeError("unknown authorized action")


if __name__ == "__main__":
    raise SystemExit(main())
