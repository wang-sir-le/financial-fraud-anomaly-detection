# External input access and reconstruction status

Audit date: 2026-09-23. The extension archive supports interval-ledger replay without transaction data. Prediction reanalysis and raw-data reconstruction require different inputs and have different verification status.

## Obtain source transactions

The official access routes are the [PaySim dataset page](https://www.kaggle.com/datasets/ealaxi/paysim1) and the [IEEE-CIS competition data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data). Use an acquisition environment separate from the scientific dependency lock. The current [official Kaggle CLI instructions](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md) describe account authentication. Do not put credentials in the manuscript or archive.

Example PowerShell commands after installing Python 3.11 or later:

```powershell
py -3.11 -m venv C:/fraudx-acquisition-env
& C:/fraudx-acquisition-env/Scripts/python.exe -m pip install kaggle
& C:/fraudx-acquisition-env/Scripts/kaggle.exe auth login
& C:/fraudx-acquisition-env/Scripts/kaggle.exe datasets download -d ealaxi/paysim1 -p C:/fraudx-inputs/paysim
& C:/fraudx-acquisition-env/Scripts/kaggle.exe competitions download ieee-fraud-detection -f train_transaction.csv -p C:/fraudx-inputs/ieee
& C:/fraudx-acquisition-env/Scripts/kaggle.exe competitions download ieee-fraud-detection -f train_identity.csv -p C:/fraudx-inputs/ieee
```

Complete any account/access/rules steps shown by the source website yourself. The competition's unlabelled test files are not needed for this paper's temporal Test windows. Extract downloaded ZIPs into new folders. Preserve CSV bytes: do not resave through a spreadsheet application. Command syntax is supported by the [official dataset](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md) and [competition](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) documentation; the authenticated commands were not executed in this audit.

| CSV | Expected bytes | Expected SHA-256 |
|---|---:|---|
| PS_20174392719_1491204439457_log.csv | 493534783 | 16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b |
| train_transaction.csv | 683351067 | 3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642 |
| train_identity.csv | 26529680 | b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c |

```powershell
Get-FileHash -Algorithm SHA256 C:/fraudx-inputs/paysim/PS_20174392719_1491204439457_log.csv
Get-FileHash -Algorithm SHA256 C:/fraudx-inputs/ieee/train_transaction.csv
Get-FileHash -Algorithm SHA256 C:/fraudx-inputs/ieee/train_identity.csv
```

Stop the exact-input route if a hash differs. Do not edit expected hashes or select a replacement dataset to improve results.

## What was actually accessed

`input_acquisition_record.json` retains the attempts. The public PaySim API returned ZIP content. The first transfer ended prematurely after 156572227 bytes; an HTTP range request completed the remaining bytes. The completed archive has 186385561 bytes. ZIP integrity and the decompressed CSV SHA-256 were verified; the CSV exactly matches the historical input above. This establishes a working fresh-download route on the audit date, not the historical acquisition date or a persistent version guarantee.

The unauthenticated IEEE transaction-file request returned HTTP 401. No authenticated download was attempted and no account credentials were read. Both existing local IEEE CSV files match the frozen hashes. Access to those local inputs does not demonstrate that an unauthenticated external reader can download them.

## Frozen predictions versus reconstruction

The exact PaySim prediction set comprises 150 files named `paysim_{fold}_{model}_{seed}_{test|validation}.parquet` and 75 associated JSON calibration/checkpoint records. Models are P0, PT, PH, PTH and PTHW; folds are 1–3 and seeds are 42, 52, 62, 72 and 82. IEEE uses one `prediction_scores.parquet` containing 60 fixed model vectors. Their historical hashes are preserved in the baseline resource and the current input audit. No public distribution location for those exact prediction files has been established. The extension archive supplies the seed ledgers needed for its new intervals, but does not silently substitute those ledgers for transaction-level predictions.

Raw-data commands remain in the frozen resource's `INPUT_ACCESS_AND_REBUILD.md`. Their actual outcome in this execution is recorded in `REBUILD_REPORT.json`; consult that report before treating a command as a verified reconstruction route. In particular, the PaySim raw route stopped at its first fitted-model discrepancy. The preprocessor sorts only by step and does not preserve a stable source-row identity within a step. A separate single-model control using the historical processed feature table exactly recovered the frozen P0 predictions in the current environment. That control is not raw-data reconstruction.

Two follow-up routes have different meanings. Making the historical processed input or exact predictions accessible, subject to the source's distribution terms, would enable replay of the frozen study. Rebuilding with a newly specified stable source-row identity and deterministic aggregation would create a new experimental baseline that must be evaluated and reported in full. The second route cannot be labelled a byte-identical recovery of the frozen results. No such replacement of the baseline was made in this execution.

## Interpretation of the archive

L1 means recalculating intervals from the supplied replicate ledgers. L2 means reanalysis of hash-matched predictions. L3 means rebuilding models and scores from verified raw data. L4 means an external reader independently completing a stated route. Local isolated installation is evidence for L1 portability, not L4. Raw data, transaction-level scores and the diagnostic downloads are not included in the extension release ZIP.
