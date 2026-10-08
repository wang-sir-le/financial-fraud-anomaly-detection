# Online Resource 4 — finite editorial revision and cache replay

This package retains all previously reported diagnostics unchanged and adds an
explicit-path interface to completed BankSim outputs from Online Resource 3.
It does not supply excluded transaction identities, labels, margins or models.
There is no fitting, prediction or new bootstrap entry in the bridge.

## Inputs and entry

Extract ESM_3_BankSim.zip and this archive to obtain `banksim/` and
`posthoc_revision/`. Follow `banksim/README_BankSim_rebuild.md` to reconstruct
the original input and all frozen models in a separate new run if caches are
unavailable. Do not copy historical completion receipts into a fresh run.
The historical run is already test-opened and must not be trained/scored again.
Its receipts are never reset. The bridge requires a completed evaluated run.

After reconstruction, use the same isolated environment (Python >=3.11, NumPy):

```powershell
& 'C:/reader/new-run/reproduction/env/Scripts/python.exe' 'C:/reader/posthoc_revision/bridge/replay_banksim.py' --run 'C:/reader/new-run' --esm3 'C:/reader/banksim' --esm4 'C:/reader/posthoc_revision' --out 'C:/reader/replay-check-01'
```

The example directories are placeholders for the reader's own paths. The output
directory must not exist and must be outside all inputs. No author-local working
directory or Chinese-named review CSV is needed. Relative paths in receipts must
remain inside the run. The bridge checks delivered package hashes, scientific
protocol fields, selected configuration, each run's own freeze/release links,
all evaluated artifact hashes, reference source identity and preprocessing,
label identity, all 40 score keys and finite aligned margin vectors. Historical
and new-run freeze digests may differ: scientific settings must agree, and each
run must satisfy its own receipt links. This is a reproducibility check of the
reported configuration, not permission to tune until reference values match.

Required completed-run paths are `protocol/effective.json`, `SELECTION.json`,
`FREEZE.json`, `TEST_RELEASE.json`, `EVALUATION_COMPLETE.json` under protocol;
`audit/PREPARED.json`; `features/test_identity.npy` (source row, step);
`preprocessing/B0.json`, `B0H.json`, `B1.json`, `B1H.json`; and all artifacts
listed by EVALUATION_COMPLETE, including the test label and 40 margin vectors.
Score stage is raw margin. Ordering is descending score, ascending hexadecimal
SHA256 of UTF8 `banksim-capacity-v1|<decimal source_row_id>`, then ascending ID.
The entire completed test window is ranked separately for every seed; exact
ceil(qN) is used at q=1%,3%,5%. No score ensemble is formed.

The adapter compares 120 absolute TP values with
`banksim/results/absolute_all_conditions.csv`, all 60 R/G/L/U partitions with
`derived/posthoc_capture_exchange/all_60_pairs.csv`, 12 means with `means_12.csv`,
and 2,160 global-selection step counts with
`derived/time_structure/banksim_step_capture_all_seeds.csv`.
Integer agreement must be exact; mean absolute tolerance is fixed at 1e-9.
All results are compared to the existing reported values; no additional
scientific analysis or interval is generated. A mismatch stops with a failure
receipt and is not repaired by resampling, selection, deletion or relaxed rules.

## Other diagnostic paths

- S21.1: `derived/headroom.csv`.
- S21.2: `derived/posthoc_capture_exchange/all_60_pairs.csv` and `means_12.csv`.
- S21.3: `derived/existing_draw_diagnostics/paysim_existing_5000.csv` and
  `SUMMARY.json`; the original draw-level source is in Online Resource 2's c1
  archive, `online_resource_2/c1/evidence/paysim/effect_replicates.parquet`.
  No new draws are needed. The bridge does not diagnose the cause of the wide CI.
- S21.4: `derived/time_structure/ieee_fold*_all_block_spans.csv`,
  `ieee_block_span_summary.csv`, `IEEE_RECEIPT.json` and BankSim time CSVs.
  IEEE-CIS exact time reconstruction still requires the frozen input identified
  by IEEE_RECEIPT and the unchanged Online Resource 2 access guide.
- S21.5: preservation and sign conventions are in Online Resource 1.

Author-layout scripts are preserved under reproduction_notes for provenance;
see README_original_author_layout.md. They are not the portable bridge entry.
The original dependency/environment record describes the earlier diagnostics;
bridge_validation/REPLAY_RECEIPT.json records the adapter's actual runtime.

## Validation boundary

Local cache interoperability was checked using the already completed original
run. No independent reader's complete raw-to-model reconstruction, external
download, cross-platform equivalence or public publication is claimed.
ESM3 code and scientific outputs are unchanged. Any freshly rebuilt margins
that lead to different reported captures fail this check; such differences
must be investigated and disclosed, never silently accepted as reproduction.
All capture-exchange quantities are descriptive and post-hoc. They provide no
new environment, G/L confidence interval, causal attribution or equivalence.
