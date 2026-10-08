# Component interval and decision stability extension

This archive replays 132 component and 54 processing-effect interval rows from 825,000 fixed-seed replicate records. It also supplies the analysis code, point estimates, full capacity/tie diagnostics, original draw adapters and regression references. It is an extension of the frozen KAIS study, not a replacement of its original experiments.

## Install in an isolated environment

Tested: Windows x64, Python 3.11.9. Run from a location outside this extracted directory, using an absolute path for `$bundle`. The install is a wheel-based ordinary install; do not use `pip install -e`. The package name is `fraudx-q2-extension` but its Python namespace is `fraudx.q2_extension`; use a separate environment from the original full fraudx package.

```powershell
$bundle = 'C:/fraudx-extension'
py -3.11 -m venv C:/fraudx-extension-env
$python = 'C:/fraudx-extension-env/Scripts/python.exe'
& $python -m pip --disable-pip-version-check install -r "$bundle/code/requirements-lock-windows-py311.txt"
& $python -m pip --disable-pip-version-check install --no-deps "$bundle/dist/fraudx_q2_extension-0.1.0+20260923-py3-none-any.whl"
& $python -m pip --disable-pip-version-check check
& $python -I -c "import fraudx.q2_extension.bundle_replay as m; print(m.__file__)"
```

The printed module path must be inside the new environment's `site-packages`. Dependencies must be available through the package index or a verified local wheel cache. Source is also supplied under `code`; a source build was tested using setuptools 80.9.0 and wheel 0.45.1, which are separate from the pinned scientific dependencies.

## Recompute the new intervals

Choose a nonexistent output folder outside the input archive:

```powershell
& $python -I -m fraudx.q2_extension.bundle_replay --bundle $bundle --output C:/fraudx-results/extension-interval-replay
```

The command verifies the archive manifest, reads the full seed ledger, recomputes five-seed means and linear percentile endpoints, and compares all 186 rows (372 endpoints) with the saved reference. It also writes the fold ledgers and invalid-draw inventory. Read `REPLAY_REPORT.json`, not just the process exit code. This route does not need Kaggle transactions, transaction-level predictions or trained models. It does not generate the original predictions or independently verify every resample from transactions.

## Reanalyse frozen transaction scores

This optional, more expensive route requires the 150 hash-matched PaySim prediction Parquets and 75 adjacent checkpoint JSON files, plus the exact IEEE `prediction_scores.parquet`. These inputs are not included or available through an established public download. `INPUT_ACCESS.md` distinguishes those missing frozen inputs from raw datasets. Raw-data recipes are in the separately retained baseline archive, and the current `REBUILD_REPORT.json` records their actual outcome. Do not relabel newly rebuilt files as the historical scores by changing the expected hashes.

The archive's `baseline_references/reproducibility` directory supplies only the four manifests/ledgers needed for old-effect regression, not the entire original training resource. If the frozen scores are available, stage a new execution directory and change only the following path fields in a copy of the supplied protocol:

```powershell
$workspace = 'C:/fraudx-work'
$run = "$workspace/experiment/outputs/q2_extension_v1/run_external_01"
$legacy = "$bundle/baseline_references/reproducibility"
if (Test-Path -LiteralPath $run) { throw 'Use a new run directory' }
$cfg = Get-Content -LiteralPath "$bundle/protocol_effective.json" -Raw | ConvertFrom-Json
$cfg.workspace_root = $workspace
$cfg.output_root = $run
$cfg.baseline.root = Split-Path -Parent $legacy
$cfg.datasets.paysim.predictions = 'C:/fraudx-inputs/frozen-paysim-checkpoints'
$cfg.datasets.paysim.manifest = "$legacy/audit/prediction_input_manifest.json"
$cfg.datasets.ieee_cis.predictions = 'C:/fraudx-inputs/frozen-ieee/prediction_scores.parquet'
$cfg.datasets.ieee_cis.manifest = "$legacy/evidence/ieee_run_metadata.json"
New-Item -ItemType Directory -Path "$run/protocol" | Out-Null
$runJson = $cfg | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText("$run/protocol/protocol_effective.json", $runJson, [System.Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath "$run/protocol/protocol_effective.json" -Destination "$run/runtime.json"
foreach ($step in @('inventory','points','priorities','draws','benchmark','bootstrap','replay')) {
    & $python -I -m fraudx.q2_extension.cli $step --config "$run/runtime.json"
    if ($LASTEXITCODE -ne 0) { throw "Stopped at $step; preserve its output and diagnostic" }
}
```

The JSON write explicitly avoids a UTF-8 BOM. Each stage refuses to overwrite an attempted stage. Do not delete a failure status and call that a fresh successful run. The main implementation uses four independent model workers and metric microbatches of four; these share frozen draws and do not change the estimand. Timing depends on hardware. Full calculation generates large draw-multiplicity arrays and 285 model/stage metric ledgers. The diagnostic priority fractions in `evidence/assessment/priority_sign_summary.csv` can be obtained by averaging paired seed TP differences within each trial and classifying their integer sums; they are not inferential probabilities.

## Interpretation

- Component intervals are conditional on fixed fitted models and observed windows. The five training seeds are averaged within each replicate, not treated as five independent datasets.
- Capture intervals use the exact 3% completed-window ranking rule. PaySim's old primary recall intervals use validation-selected thresholds; the two estimands must not be interchanged.
- AP remains sklearn's grouped-score average precision. Cost reduction at equal capacity is exactly 101 times the TP difference and is not independent confirmation.
- All intervals are pointwise. Clipping/Platt stage duplication is deliberate, not additional independent evidence. No new Overall inference is added.
- Modelwise tie ranges, conservative paired envelopes and 100 shared-priority trials describe fixed-score sensitivity. They are not confidence intervals.
- Local isolated replay is L1 portability. The current IEEE raw reconstruction reproduces scientific values; PaySim raw reconstruction failed at the first fit. No third-party L4 reproduction or publication acceptance is claimed.

The original protocol JSON is preserved for provenance, including its planning-time status fields. Actual acquisition and reconstruction outcomes are in `INPUT_ACCESS.md` and `REBUILD_REPORT.json`. No source dataset, frozen manuscript, model setting, split or expected historical hash was changed to improve results.
