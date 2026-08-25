"""Command-line entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from fraudx.calibration import run_calibration_ablation
from fraudx.config import load_config
from fraudx.explain import run_shap_analysis
from fraudx.multiseed import run_multiseed_validation
from fraudx.pair_audit import run_pair_feature_audit
from fraudx.pipeline import prepare_real_data, run_experiment
from fraudx.robustness import run_rolling_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run temporal financial fraud experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Experiment YAML path",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and prepare the real PaySim CSV without training",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Rebuild the prepared Parquet cache",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run with synthetic PaySim-shaped data",
    )
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Run expanding-window temporal robustness experiments",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Run TreeSHAP importance and failure-case analysis",
    )
    parser.add_argument(
        "--pair-audit",
        action="store_true",
        help="Audit Pair feature generation, model inclusion, importance, and predictions",
    )
    parser.add_argument(
        "--multiseed",
        action="store_true",
        help="Run the Pair-gated five-seed primary-scenario stability experiment",
    )
    parser.add_argument(
        "--calibration-ablation",
        action="store_true",
        help="Run shared-score Uncalibrated/Platt/Isotonic calibration analysis",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.prepare_only:
        prepare_real_data(config, force=args.force_prepare)
        print(f"Prepared PaySim data: {config.processed_path}")
        return
    if args.rolling:
        results = run_rolling_validation(config, smoke=args.smoke)
        print(
            results[
                [
                    "fold",
                    "model_id",
                    "fn_cost",
                    "target_capacity",
                    "pr_auc",
                    "recall",
                    "cost_saving_rate",
                ]
            ]
        )
        return
    if args.pair_audit:
        outputs = run_pair_feature_audit(config, smoke=args.smoke)
        print({name: len(frame) for name, frame in outputs.items()})
        return
    if args.multiseed:
        outputs = run_multiseed_validation(config, smoke=args.smoke)
        print({name: len(frame) for name, frame in outputs.items()})
        return
    if args.calibration_ablation:
        outputs = run_calibration_ablation(config, smoke=args.smoke)
        print({name: len(frame) for name, frame in outputs.items()})
        return
    if args.explain:
        importance = run_shap_analysis(config, smoke=args.smoke)
        print(importance.head(15))
        return
    results = run_experiment(config, smoke=args.smoke)
    print(results[["experiment", "test_pr_auc", "test_recall", "test_business_cost"]])


if __name__ == "__main__":
    main()
