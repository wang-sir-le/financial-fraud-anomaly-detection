"""Portable entry points: acquire, prepare, train and compare-features."""

import argparse
import json
from pathlib import Path

from fraudx.paysim_deterministic.acquisition import CSV_SHA256, acquire
from fraudx.paysim_deterministic.preprocess import compare_parquets, prepare
from fraudx.paysim_deterministic.training import train
from fraudx.q2_extension.common import read_json, require, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("acquire")
    get.add_argument("--output", type=Path, required=True)
    get.add_argument("--archive", type=Path)
    build = sub.add_parser("prepare")
    build.add_argument("--csv", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--threads", type=int, choices=[1, 16], default=1)
    build.add_argument("--reverse-scan", action="store_true")
    build.add_argument("--block-size", type=int, default=1048576)
    fit = sub.add_parser("train")
    fit.add_argument("--protocol", type=Path, required=True)
    fit.add_argument("--features", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare-features")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    configure = sub.add_parser("configure-analysis")
    configure.add_argument("--template", type=Path, required=True)
    configure.add_argument("--scores", type=Path, required=True)
    configure.add_argument("--output", type=Path, required=True)
    configure.add_argument("--write", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        print(json.dumps(acquire(args.output, args.archive), indent=2))
    elif args.command == "prepare":
        print(json.dumps(prepare(args.csv, args.output, threads=args.threads,
            reverse_scan=args.reverse_scan, block_size=args.block_size,
            expected_hash=CSV_SHA256), indent=2))
    elif args.command == "train":
        train(args.protocol, args.features, args.output)
    elif args.command == "compare-features":
        print(json.dumps(compare_parquets(args.left, args.right), indent=2))
    else:
        require(not args.write.exists(), "Configuration must be a new file")
        config = read_json(args.template)
        config["workspace_root"] = str(Path.cwd())
        config["datasets"]["paysim"]["predictions"] = str(args.scores.resolve())
        config["output_root"] = str(args.output.resolve())
        write_json(args.write, config)


if __name__ == "__main__":
    main()
