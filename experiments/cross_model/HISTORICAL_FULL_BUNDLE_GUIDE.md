# Cross-model extension input access and execution

The C extension tests model dependence using a fixed logistic-regression pipeline. It uses the deterministic PaySim v2 features and unchanged IEEE-CIS features, splits and feature groups. It is an additional analysis of previously examined windows, not independent blind validation. The master scientific protocol is `protocol/protocol_effective.json`; later IEEE job configurations change task scope and output paths only. No hyperparameter search is performed.

The inherited `baseline/` and `v2/` directories remain historical evidence. Current C code and evidence are in `c1/`. Use the single version 0.3.1 wheel in `c1/dist/`. The development package in `c1/history/` is archival. Its subsequent reporting fix handles nonconverged conditions and empty estimand sets; it does not change numerical estimands, the trainer or the completed PaySim numerical results.

## 1. Install the pinned environment

The commands use PowerShell, Windows x64 and Python 3.11.9. Start in the directory containing the extracted `online_resource_2/` folder. The rebuild output and replay output must be new directories. Replace the example raw-data directory with your actual location.

```powershell
$resource = (Resolve-Path ./online_resource_2).Path
$work = Join-Path (Get-Location) 'c-reconstruction'
New-Item -ItemType Directory -Path $work | Out-Null
py -3.11 -m venv "$work/env"
$python = (Resolve-Path "$work/env/Scripts/python.exe").Path
& $python -I -c "import sys; assert sys.version_info[:3] == (3,11,9), sys.version; print(sys.version)"
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
& $python -m pip install --no-deps --only-binary=:all: --use-deprecated=legacy-resolver -r "$resource/c1/requirements-full-lock-windows-py311.txt"
& $python -m pip install --no-deps --no-index (Get-ChildItem "$resource/c1/dist/*.whl").FullName
& $python -m pip check
& $python -I -c "import fraudx.cross_model.training as m; print(m.__file__)"
```

If the version check fails, recreate the environment using an explicitly located Python 3.11.9 executable rather than the launcher's default 3.11 installation. The printed import must resolve inside the new environment's `site-packages`. `-I` prevents a source checkout or `PYTHONPATH` from satisfying the import. The pinned dependency file contains the scientific dependency closure; installing the wheel requires no editable checkout. Cross-platform numerical identity has not been established.

## 2. Replay C intervals without original transactions

```powershell
& $python -I -m fraudx.cross_model.replay --evidence "$resource/c1/evidence" --output "$work/interval-replay"
```

This rebuilds every reported C interval from the supplied draw-level effects and observed point estimates. It checks complete draw identities, finite/undefined counts, percentile endpoints and the published values. Each interval is pointwise and conditional on the fixed models and observed windows. LR-minus-LightGBM effects compare one fitted LR effect with the mean of five paired LightGBM effects inside each shared draw; they do not compare LR with an averaged prediction ensemble.

Nonconverged fits do not have formal prediction or interval outputs. Their missing contrasts remain explicitly recorded in `contrast_coverage.csv`; an empty interval table is not evidence of a zero effect. Replay verifies reporting and aggregation, not model refitting or independent external reproduction.

## 3. Obtain the original datasets

PaySim comes from the [PaySim dataset page](https://www.kaggle.com/datasets/ealaxi/paysim1). The following command downloads and verifies the exact expected CSV:

```powershell
& $python -I -m fraudx.paysim_deterministic acquire --output "$work/inputs/paysim"
```

If source access or an interrupted transfer prevents that command, download the ZIP from the source page and give it to the verified extractor in a new output directory:

```powershell
& $python -I -m fraudx.paysim_deterministic acquire --archive C:/data/paysim1.zip --output "$work/inputs/paysim-manual"
```

Use the successful output path below. The expected PaySim CSV is `PS_20174392719_1491204439457_log.csv`, 493,534,783 bytes, SHA-256 `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`.

IEEE-CIS comes from the [IEEE fraud detection competition data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data). Readers must satisfy the platform's account and data-access requirements. No credentials are supplied, and no authenticated external acquisition has been tested in this extension. After obtaining the competition archive, extract these two labelled files to a directory such as `C:/data/ieee-fraud-detection/`:

| File | Bytes | SHA-256 |
|---|---:|---|
| train_transaction.csv | 683351067 | 3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642 |
| train_identity.csv | 26529680 | b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c |

The model uses labelled transaction data partitioned into the paper's temporal windows, not the competition's unlabelled test files. The identity file is required by the inherited history-builder audit but is not joined into the 392-feature model input. The local source files were hash verified; this extension did not perform a new network download. A mismatch stops the exact-input route. Do not edit hashes or silently use another dataset release.

## 4. Rebuild the required features and LightGBM reference scores

```powershell
& $python -I -m fraudx.paysim_deterministic prepare --csv "$work/inputs/paysim/PS_20174392719_1491204439457_log.csv" --output "$work/paysim-features" --threads 1
& $python -I -m fraudx.paysim_deterministic train --protocol "$resource/v2/protocol/protocol_effective.json" --features "$work/paysim-features/features.parquet" --output "$work/paysim-lightgbm"
```

The expected feature Parquet SHA-256 is `1e241c5e46e1d8323fe021c4185ce2ab3cb22f739d224384ebcb248490ed568f`. This route rebuilds the full preserved v2 reference grid, including branches outside the narrower C comparisons. Do not rebuild historical v1 PaySim rows or use their positional `row_id` as a substitute for v2 `source_row_id`.

```powershell
$ieeeRaw = 'C:/data/ieee-fraud-detection'
& $python -I -m fraudx.ieee_cis_history --transaction_path "$ieeeRaw/train_transaction.csv" --identity_path "$ieeeRaw/train_identity.csv" --output_dir "$work/ieee-history"
& $python -I -m fraudx.ieee_cis_experiment --transaction_path "$ieeeRaw/train_transaction.csv" --history_path "$work/ieee-history/ieee_cis_entity_history_features.parquet" --history_metadata_path "$resource/c1/protocol/ieee_history_metadata.json" --protocol_path "$resource/c1/protocol/ieee_temporal_protocol.json" --output_dir "$work/ieee-lightgbm"
```

The expected IEEE history Parquet SHA-256 is `7e3f4f28d787bc6e31318fbd148833c939bf78a6c231fbe920a749f96ed89e35`. The supplied metadata and temporal protocol retain the original boundaries and expected data hashes. Original absolute paths inside them are provenance; explicit command arguments select the actual inputs. The IEEE LightGBM command runs M0–M3 across three folds and five seeds. Native thread settings and operating-system differences can affect reconstruction; subsequent scientific-vector checks stop if results differ.

Source transactions, feature matrices, fitted models and transaction-level prediction files are not redistributed in this package. These rebuild commands are the concrete route to the inputs needed for C. Existing verified local inputs can be supplied directly instead.

## 5. Configure the C run against the verified references

```powershell
& $python -I -m fraudx.cross_model.portable --resource "$resource/c1" --output "$work/c-run" --paysim-features "$work/paysim-features/features.parquet" --paysim-lgbm-predictions "$work/paysim-lightgbm" --ieee-transaction-path "$ieeeRaw/train_transaction.csv" --ieee-history-path "$work/ieee-history/ieee_cis_entity_history_features.parquet" --ieee-lgbm-predictions "$work/ieee-lightgbm/prediction_scores.parquet"
```

This compares all consumed reference score arrays, sample identities, timestamps, labels and PaySim threshold rows with the released scientific manifest. It allows metadata containers to have new paths or run durations while requiring identical scientific values. It regenerates all six frozen time-block multiplicity files and verifies their hashes. It also verifies the source-feature inputs and writes a new, path-adapted C protocol. Settings, feature roles, splits, calibration boundaries and estimands remain unchanged.

The supplied `reference_replicates/` files contain aggregate reference bootstrap records, not individual transactions. Their provenance comes from the preserved LightGBM analyses. The package-wide manifest covers these files. Recomputing the inherited reference bootstrap records from source predictions is described in the v2 and baseline guides.

## 6. Execute all C fits and analyses

```powershell
$protocol = "$work/c-run/protocol/protocol_effective.json"
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $python -I -u -m fraudx.cross_model.training --protocol $protocol --dataset paysim
& $python -I -u -m fraudx.cross_model.training --protocol $protocol --dataset ieee_cis
& $python -I -u -m fraudx.cross_model.analysis --protocol $protocol --dataset paysim
& $python -I -u -m fraudx.cross_model.analysis --protocol $protocol --dataset ieee_cis
& $python -I -m fraudx.cross_model.replay --evidence "$work/c-run/analysis" --output "$work/rebuilt-interval-replay"
```

There are 15 PaySim and 12 IEEE-CIS formal detector conditions, each starting with one deterministic logistic fit and contributing at most one converged model. PaySim has 15 calibrators when all its detector fits converge. The fixed solver uses a 3,000-iteration limit and at most one new fit at 10,000 iterations, with all other settings unchanged. Both attempts are retained. A failure after the fixed retry is reported, not replaced with a favourable model or accepted as a converged result. High-dimensional IEEE-CIS fits can be expensive.

Completed checkpoints can be resumed only when input, code and artifact signatures match. Incomplete attempts are retained and not silently reused. The author run used independently scoped IEEE model jobs to overlap computation; `IEEE_DISPATCH.json`, scoped protocols and preserved source checkpoint metadata document that execution arrangement. Later-window concurrency was reduced to two after measured preprocessing-memory peaks; the completed M2 first-window checkpoint was retained for resumption. The scheduling and boundary records remain in c1/protocol/. This does not add independent observations or modify the scientific conditions. The straightforward sequential commands above implement the same scientific grid.

## 7. Recheck predictions from saved C model states

```powershell
& $python -I "$resource/c1/operators/verify_cross_model_checkpoints.py" --protocol $protocol --dataset paysim --output "$work/checkpoint-check-paysim"
& $python -I "$resource/c1/operators/verify_cross_model_checkpoints.py" --protocol $protocol --dataset ieee_cis --output "$work/checkpoint-check-ieee"
```

The checker restores coefficients, the train-fitted preprocessing state and PaySim calibration parameters, then recomputes Validation/Test score columns. It checks exact values without refitting a detector. The frozen command-line trainer serialized its preprocessing class under `__main__`; the checker explicitly exposes the identical installed class to read those locally produced checkpoints. Do not load unrelated or untrusted pickle files.

The reports distinguish ledger replay, reference-input verification, checkpoint-to-score reconstruction and detector refitting. Successful local checks do not constitute third-party reproduction, model-independent validity, or evidence of likely journal acceptance.
