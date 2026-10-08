# PaySim v2 input access and reconstruction

This version supplies a route from a verified raw CSV to all affected predictions. It creates a new deterministic baseline; it does not claim to recover the historical PaySim processed-row ordering. Scientific verification status is recorded separately in the accompanying reports. Commands below use PowerShell and a new writable working directory.

## Required inputs

PaySim is available from the [source dataset page](https://www.kaggle.com/datasets/ealaxi/paysim1). The included acquisition command either downloads the public archive or accepts a manually downloaded archive, extracts only the expected CSV member to a fixed filename, and verifies its length and SHA-256. Source transactions are not redistributed.

| File | Bytes | SHA-256 |
|---|---:|---|
| PS_20174392719_1491204439457_log.csv | 493534783 | 16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b |
| IEEE train_transaction.csv | 683351067 | 3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642 |
| IEEE train_identity.csv | 26529680 | b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c |

The PaySim public download was verified on 2026-09-23. The first network transfer was truncated; the retained acquisition record documents its completion and CRC/SHA-256 verification. The v2 extraction command was tested using that verified archive. This does not establish the original historical download date or guarantee that the source website will retain the same version.

IEEE-CIS access is through the [competition data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data). An anonymous request returned HTTP 401. Readers must complete the platform's account/access requirements themselves; no authenticated acquisition was tested here. The labelled transaction file supplies the temporal training and test windows; the competition's unlabelled test files are not the paper's Test windows. The identity file is retained for the original audit route, not merged into the 392-feature model input. Existing local CSVs matched the recorded hashes and the unchanged IEEE reconstruction was previously checked. Its original commands and complete reports remain in the preserved baseline resource.

## Install the versioned scientific environment

Use Windows x64 and Python 3.11.9 for the tested reconstruction. Cross-platform bit identity is not claimed. From the extracted v2 package, create a new environment:

```powershell
py -3.11 -m venv .venv-v2
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
& .venv-v2/Scripts/python.exe -m pip install --no-deps --only-binary=:all: --use-deprecated=legacy-resolver -r requirements-full-lock-windows-py311.txt
& .venv-v2/Scripts/python.exe -m pip install --no-deps --no-index (Get-ChildItem dist/*.whl).FullName
& .venv-v2/Scripts/python.exe -m pip check
```

The full dependency closure is pinned. The legacy pip resolver option avoids the metadata-resolution stall observed with the Python 3.11 environment's bundled pip; it does not change the specified package versions. Only one distribution wheel should be present in the final package's `dist` directory.

## Acquire and prepare PaySim

```powershell
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic acquire --output inputs/paysim
```

If the public transfer is interrupted or the website requires browser access, download the archive from the source page, preserve the failed attempt, and use a new output directory:

```powershell
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic acquire --archive downloads/paysim1.zip --output inputs/paysim-manual
```

Use the successfully verified CSV path in subsequent commands. A hash mismatch stops the exact-input route; do not edit the expected hash or silently substitute another dataset release.

```powershell
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic prepare --csv inputs/paysim/PS_20174392719_1491204439457_log.csv --output rebuild/features --threads 1
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic prepare --csv inputs/paysim/PS_20174392719_1491204439457_log.csv --output rebuild/features-check --threads 16 --reverse-scan --block-size 4194304
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic compare-features rebuild/features/features.parquet rebuild/features-check/features.parquet
```

The source identity is the zero-based data-record ordinal of the exact CSV, excluding its header. It is assigned before sorting; duplicate records retain distinct ordinals. Model processing order is `(step, source_row_id)` and the ID is excluded from model inputs. History excludes every same-step record. Integer-cent sums and squared sums remove order-dependent floating accumulation. The checked feature-file SHA-256 is `1e241c5e46e1d8323fe021c4185ce2ab3cb22f739d224384ebcb248490ed568f` in the tested environment.

## Rebuild all affected models

```powershell
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic train --protocol protocol/protocol_effective.json --features rebuild/features/features.parquet --output rebuild/models
```

This runs 90 LightGBM fits, 90 Platt fits and 30 Isotonic fits using the preserved time boundaries, five seeds and six schemes, including the weight-five branch. The output contains 180 validation/test score ledgers, model records, calibrator parameters and 660 scenario rows. Completed models can be resumed only if input/code signatures and all retained artifact hashes match. Incomplete artifacts are preserved and rejected for silent reuse. The absolute provenance paths in the frozen protocol are historical records: the command explicitly receives the current feature and output paths.

## Recalculate decisions and intervals

```powershell
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic configure-analysis --template protocol/analysis_config.json --scores rebuild/models --output rebuild/analysis --write rebuild/analysis_config.json
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic.analysis --config rebuild/analysis_config.json audit
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic.analysis --config rebuild/analysis_config.json draws
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic.analysis --config rebuild/analysis_config.json points
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic.analysis --config rebuild/analysis_config.json bootstrap
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic.analysis --config rebuild/analysis_config.json priorities
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic.primary --config rebuild/analysis_config.json
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic.calibration --config rebuild/analysis_config.json
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic.score_audit --config rebuild/analysis_config.json
```

Stage outputs must be new. These commands retain the three score stages, four capacities, 100 shared label-free priorities and 5,000 paired time-block draws. Threshold-transfer intervals and exact-capacity intervals answer different questions. The package does not need unavailable v1 transaction identities to rebuild v2. Direct v1/v2 comparisons use preserved aggregate/replicate evidence and common time-group draws, not an invented transaction mapping.

## Replay intervals without raw transactions

```powershell
& .venv-v2/Scripts/python.exe -I -m fraudx.paysim_deterministic.replay --evidence evidence --output replay-check
```

This checks component and processing intervals, primary threshold-transfer intervals, original block-length sensitivities and the existing secondary equal-fold summary from supplied seed ledgers. It verifies aggregation and percentile calculation, not raw-data acquisition or detector fitting. Ledger replay, prediction reanalysis, local raw-to-score reconstruction and independent external reproduction remain distinct levels. No third-party reproduction is claimed.
