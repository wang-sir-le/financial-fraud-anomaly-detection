"""Reconstruct the frozen PaySim prediction ledger and run its paired Bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

from fraudx.config import load_config
from fraudx.prediction_reconstruction import run_prediction_reconstruction
from fraudx.timeblock_bootstrap import run_timeblock_bootstrap


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paysim_frozen.yaml"),
        help="Frozen PaySim YAML configuration",
    )
    return parser


def main() -> None:
    """Run reconstruction first and Bootstrap only after the audit passes."""
    config = load_config(build_parser().parse_args().config)
    run_prediction_reconstruction(config)
    run_timeblock_bootstrap(config)


if __name__ == "__main__":
    main()
