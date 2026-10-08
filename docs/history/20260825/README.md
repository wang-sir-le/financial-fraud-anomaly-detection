# Financial Fraud Detection under Temporal and Capacity Constraints

This repository contains the research code for a temporal fraud-detection study on PaySim and IEEE-CIS. The implementation separates global ranking, probability reliability, and capacity-constrained decision utility while preserving dataset-specific evaluation protocols.

The repository is a reproducibility package, not a data distribution. Raw datasets, processed transaction-level data, prediction ledgers, fitted models, and local manuscript-audit tooling are intentionally excluded.

## Scientific scope

- Strictly past-only entity-history features with same-timestamp isolation.
- Expanding-window temporal validation for both datasets.
- PaySim recipient-history comparisons under a validation-selected threshold constrained to at most a 3% alert rate.
- IEEE-CIS M0-M3 LightGBM comparisons under exact Test top-3% ranking.
- Five fixed training seeds: 42, 52, 62, 72, and 82.
- An analytical false-negative/false-positive cost ratio of 100:1.
- PaySim-only calibration analysis.
- Paired moving time-block Bootstrap analyses within frozen future windows.

The PaySim and IEEE-CIS protocols are intentionally not described as identical replications. Their time variables, entity meanings, capacity implementations, and uncertainty units differ.

## Repository layout

```text
configs/                 Portable frozen PaySim configuration
data/                    Local-only data locations; data are ignored by Git
docs/                    Data access, protocol, and Supplement material
outputs/                 Generated artifacts; contents are ignored by Git
scripts/                 Dataset audit, feature-building, and Bootstrap entrypoints
src/fraudx/              Core scientific implementation
tests/                   Tests for the public scientific code
```

## Installation

Python 3.11 or later is required.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

The same dependencies are also listed in `requirements.txt` for environments that do not use editable installation.

## Smoke test

The smoke test uses synthetic PaySim-shaped records only to verify execution. It is not a paper result.

```bash
python -m fraudx.cli --config configs/paysim_frozen.yaml --smoke
pytest
```

## PaySim workflow

Obtain PaySim from its original source and place the CSV at:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

Then run:

```bash
python -m fraudx.cli --config configs/paysim_frozen.yaml --prepare-only
python -m fraudx.cli --config configs/paysim_frozen.yaml --rolling
python -m fraudx.cli --config configs/paysim_frozen.yaml --pair-audit
python -m fraudx.cli --config configs/paysim_frozen.yaml --multiseed
python -m fraudx.cli --config configs/paysim_frozen.yaml --calibration-ablation
python scripts/run_paysim_timeblock_bootstrap.py --config configs/paysim_frozen.yaml
```

The Bootstrap entrypoint first reconstructs the frozen paired prediction ledger and requires the reconstruction audit to pass. It does not select a new model, threshold, capacity, or block length.

## IEEE-CIS workflow

Obtain `train_transaction.csv` and `train_identity.csv` through the official IEEE-CIS Fraud Detection Kaggle page and place them under `data/raw/ieee_cis/`. The following commands retain separate output directories for every stage:

```bash
python scripts/run_ieee_cis_audit.py \
  --transaction_path data/raw/ieee_cis/train_transaction.csv \
  --identity_path data/raw/ieee_cis/train_identity.csv \
  --output_dir outputs/ieee_cis/audit

python scripts/build_ieee_history_features.py \
  --transaction_path data/raw/ieee_cis/train_transaction.csv \
  --identity_path data/raw/ieee_cis/train_identity.csv \
  --output_dir outputs/ieee_cis/history

python build_ieee_temporal_protocol.py \
  --transaction_path data/raw/ieee_cis/train_transaction.csv \
  --feature_path outputs/ieee_cis/history/ieee_cis_entity_history_features.parquet \
  --output_dir outputs/ieee_cis/temporal_protocol

python run_ieee_cis_experiment.py \
  --transaction_path data/raw/ieee_cis/train_transaction.csv \
  --history_path outputs/ieee_cis/history/ieee_cis_entity_history_features.parquet \
  --history_metadata_path outputs/ieee_cis/history/history_feature_metadata.json \
  --protocol_path outputs/ieee_cis/temporal_protocol/ieee_cis_temporal_protocol.json \
  --output_dir outputs/ieee_cis/model_results
```

Directional cross-dataset synthesis and the IEEE-CIS time-block Bootstrap require outputs from the preceding frozen stages. Their command-line arguments are available through:

```bash
python analyze_ieee_cis_results.py --help
python run_ieee_cis_timeblock_bootstrap.py --help
```

## Data and reporting boundaries

See `docs/DATA_ACCESS.md` for provenance and access conditions. `docs/SUPPLEMENT.md` records the venue-neutral frozen protocol details, and `docs/SUPPLEMENT_ASSET_MANIFEST.md` maps each Supplement section to public code or locally generated artifacts. The analytical 100:1 cost ratio is a predefined evaluation scenario, not an observed monetary-loss estimate or a universal operational optimum.

## License

The research code is released under the MIT License. The current public author identifier is `wang-sir-le`; it may be replaced with the author's preferred publication name before a tagged release.
