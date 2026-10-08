# Financial Fraud Detection under Temporal and Capacity Constraints

Research code, frozen protocol specifications, compact recorded results, and standalone figures for PaySim, IEEE-CIS and BankSim. The October 8, 2026 update adds the completed component/capacity analyses, deterministic PaySim reconstruction, prescribed logistic comparison, BankSim validation, post-hoc capture diagnostics and retrospective merchant-history supplement S22.

The stage index is [docs/UPDATES_20261008.md](docs/UPDATES_20261008.md). It identifies the source stage and the evidence boundary for each directory. The original August base implementation and its historical documents remain available in Git history and [docs/history/20260825](docs/history/20260825).

## Layout

```text
src/fraudx/              Base implementation and q2_extension, paysim_deterministic, cross_model
configs/                Original portable PaySim configurations
scripts/                Original dataset audit and Bootstrap entrypoints
tests/                  Public-code regression tests using artificial records
experiments/            Six completed extension snapshots and compact summary tables
figures/main/           Current Figures 1-6, including October 8 revisions of Figures 3-5
figures/supplement/     Figures S1-S8 with scripts and adjacent aggregate JSON inputs
docs/                   Access, reconstruction limits, stage map and historical documents
```

## Installation and checks

Python 3.11 or later is required. For numerical reconstruction, use the stage-specific Windows/Python 3.11 dependency records rather than assuming that unpinned versions produce identical scores.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
python -m fraudx.cli --config configs/paysim_frozen.yaml --smoke
```

The smoke run and unit tests use artificial records and are execution checks, not paper results. BankSim has separate tests and an environment guide in [experiments/banksim](experiments/banksim).

## Base workflows

Obtain PaySim from its original source and place its CSV under `data/raw/`. Obtain the labelled IEEE-CIS training files through the official Kaggle competition and place them under `data/raw/ieee_cis/`. The original base workflow is documented in [docs/history/20260825/README.md](docs/history/20260825/README.md). The added reconstruction modules accept explicit reader input/output paths; consult the current stage README before using a historical full-bundle guide.

## Scientific and distribution boundaries

Global ranking, calibration reliability and capacity-constrained capture are distinct outcomes. The original PaySim primary analysis transfers a validation-selected threshold; its later exact-capacity component comparisons are different estimands. IEEE-CIS uses exact Test top-3% ranking. BankSim and S22 retain their own frozen designs. Five training seeds are not five independent datasets. Conditional intervals, descriptive post-hoc diagnostics and the retrospective S22 findings retain their original inferential status; negative and uncertain results are included.

Raw transactions, row-level labels/scores/features, fitted models, preprocessing states, binary score caches and large Bootstrap replicate files are excluded. Compact CSV/JSON tables and figure inputs are included. Author-local paths in copied JSON and documentation are explicit placeholders; scientific numeric fields are preserved. Historical lock digests refer to the original author artifacts, not to these path-adapted public copies.

This update publishes recorded evidence and code. Publication and local regression checks do not establish independent external raw-to-model reconstruction. Read [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the actual available routes and missing inputs.

## License

Research code remains under the [MIT License](LICENSE). Dataset access and use remain subject to the original providers' terms; datasets are not redistributed here.
