"""Command entry point; hazardous phases require explicit intent switches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from banksim.common import ProtocolError, normalize  # noqa: E402
from banksim.data import acquire  # noqa: E402
from banksim.guards import verify_freeze  # noqa: E402
from banksim.workflow import (  # noqa: E402
    analyse,
    configure,
    evaluate,
    freeze,
    prepare,
    train_select,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("acquire", help="Verify or obtain the fixed public archive; no labels analysed")
    get.add_argument("--source", type=Path)
    for name in ("prepare", "configure", "freeze", "verify-freeze", "status"):
        commands.add_parser(name)
    replay = commands.add_parser("analyse")
    replay.add_argument("--replay-name")
    train = commands.add_parser("train-select")
    train.add_argument("--execute-real-training", action="store_true")
    test = commands.add_parser("evaluate")
    test.add_argument("--open-test-once", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.command == "acquire":
            result = acquire(root, args.source)
        elif args.command == "prepare":
            result = prepare(root)
        elif args.command == "configure":
            result = configure(root)
        elif args.command == "freeze":
            result = freeze(root)
        elif args.command == "verify-freeze":
            result = {"status": verify_freeze(root)["status"]}
        elif args.command == "train-select":
            result = train_select(root, args.execute_real_training)
        elif args.command == "evaluate":
            result = evaluate(root, args.open_test_once)
        elif args.command == "analyse":
            result = analyse(root, args.replay_name)
        else:
            result = {name: (root / "protocol" / name).exists() for name in ("FREEZE.json", "SELECTION.json", "TRAINING_COMPLETE.json", "TEST_RELEASE.json", "EVALUATION_COMPLETE.json")}
        print(json.dumps(normalize(result), ensure_ascii=False, allow_nan=False))
        return 0
    except (ProtocolError, FileNotFoundError) as exc:
        print(json.dumps({"status": "STOPPED", "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
