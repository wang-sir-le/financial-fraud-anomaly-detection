# BankSim: complete evidence and fresh-directory reconstruction

This package accompanies Section S20. It copies audited results and the original
implementation; no new training, test scoring or bootstrap was executed to make it.
It is a local delivery, not a claim of public repository publication or independent
third-party reproduction. No source transaction CSV, transaction-level predictions,
model binaries or dependency wheels are included.

## Existing evidence

`results/` contains all 40 conditions, three capacities, descriptive segments,
paired effects, all 480 interval rows, tie/ceiling records, and the original saved
bootstrap sequences and effects. `reference_protocol/` and `audit/` are historical
receipts, not receipts for a new experiment. Absolute paths inside them identify
the original author run and will not resolve on another machine. The package
manifest maps every copied file to its original path and digest.

The original run already opened the test. Never delete its receipts, retrain it,
or invoke its one-time evaluation command again. This archive omits score caches,
so it does not promise direct execution of `analyse` from this evidence-only folder.
Saved effect arrays and sequence records remain inspectable without model execution.

## Obtain the exact input

Uploader page: https://www.kaggle.com/datasets/ealaxi/banksim1

Version-1 endpoint:
https://www.kaggle.com/api/v1/datasets/download/ealaxi/banksim1?datasetVersionNumber=1

If the endpoint requires authentication, use the same uploader/version through the
platform. Do not substitute another upload or a fresh simulation. The input member is
`bs140513_032310.csv`, SHA256
`e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377`.
Archive SHA256:
`3bfd0c2bfbdec83ad36eed499eaf38c6951c36be864ecbe30f3bcb7d84895f66`.
Recorded metadata identifies CC BY-NC-SA 4.0; check the source's current terms and
retain provenance before redistribution. No data redistribution is performed here.

## Prepare an independent new run

Copy the contents of `code/` into a NEW empty directory; preserve relative paths.
Do not copy `reference_protocol/` stage receipts into it. Create output folders
`audit`, `features`, `preprocessing`, `models`, `predictions`, `analysis`.
Use Windows x64 CPython 3.11.9. The lock contains exact platform-specific wheel
hashes. `reproduction/install_report.json` retains each original PyPI wheel URL and
archive hash. Download those files into a new `reproduction/wheels/` directory and
check their hashes, or obtain the original verified wheels locally. Other Python
versions/platforms are not claimed equivalent.

From the NEW run directory (replace only the example Python/input paths):

```powershell
& 'C:/reader/Python311/python.exe' -m venv './reproduction/env'
& './reproduction/env/Scripts/python.exe' -m pip install --no-index --find-links './reproduction/wheels' --require-hashes -r './reproduction/requirements-lock.txt'
& './reproduction/env/Scripts/python.exe' -m pip check
& './reproduction/env/Scripts/python.exe' run.py acquire --source 'C:/reader/input/banksim_v1.zip'
& './reproduction/env/Scripts/python.exe' run.py prepare
& './reproduction/env/Scripts/python.exe' run.py configure
& './reproduction/env/Scripts/python.exe' verify.py
& './reproduction/env/Scripts/python.exe' run.py freeze
& './reproduction/env/Scripts/python.exe' run.py verify-freeze
```

`acquire` without `--source` uses the fixed endpoint above. The implementation assigns
source identities before sorting. `configure` reconstructs the effective protocol
and full catalog from the copied original code and draft source. Compare its scientific
settings with `reference_protocol/effective.json`; timestamps and local receipt hashes
belong to the new run and need not equal the historical receipts.

## Separate gated execution stages (not executed for this manuscript revision)

After the input, hand-built checks and freeze are valid:

```powershell
& './reproduction/env/Scripts/python.exe' run.py train-select --execute-real-training
```

This batches the 16 candidates and 40 formal conditions (54 actual fits with exact
reuse), not a free pilot. It does not automatically pause after its first fit for
resource approval. Resource monitoring and the retained first-fit records must be
reviewed; preserve failed/interrupted records without automatic retries or added
candidates. Check two-class training/validation feasibility, every candidate,
selection rule, backend parameters, warnings and complete formal models before
allowing test access. Never refit on combined training and validation or alter the
protocol in response to test performance.

Only after all training checks pass, in this same NEW run:

```powershell
& './reproduction/env/Scripts/python.exe' run.py evaluate --open-test-once
```

Inspect the complete evaluation receipt and all 40 aligned prediction caches before:

```powershell
& './reproduction/env/Scripts/python.exe' run.py analyse
```

This completes the frozen L=7/3/14, 5,000-draw analyses. It is not part of this editing
stage. A completed original run must not be repurposed as a new independent validation.
Do not reduce draws, discard conditions, select a favourable block length, or tune
after test access. Missing inputs/hash mismatches, invalid labels, required model
failures, non-finite predictions, identity mismatch or insufficient resources require
stopping and retaining the failure record. An interrupted test cannot be silently
reset to unopened. A no-positive test retains its defined/undefined outcomes and stops
inference without moving boundaries. Intervals crossing zero and saturation are
reported outcomes, not reasons to modify the protocol.

## Practical limits

The command names and required source files were inspected during this revision;
this newly assembled package was not executed end to end. Availability of external
downloads and exact reconstruction on a third-party machine remain unverified.
The historical scoring-worker RSS gap and reporting-helper/exposure notes are
retained in `audit/test_review/`. Engineering replay and additional seeds do not
constitute independent scientific evidence.
