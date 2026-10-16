# Figure 6: current five-panel source

Run `python fig6.py` from any working directory. This preserves the existing script entry point and writes `fig6.pdf`, `fig6.svg`, `fig6.png` (600 dpi), `fig6_preview.png`, `fig6_plot_data.csv` and `fig6_validation.json` beside the script. Install `fig6_requirements.txt` in an isolated environment when the plotting dependencies are unavailable.

The default artwork matches the five-panel image embedded in the current manuscript: the original 170 × 166 mm canvas is exported with its standalone footer removed and the same native bottom crop (170 × 150.728 mm). The full editable figure caption is saved separately in `fig6_caption.txt` and `CAPTIONS.md`. `--with-caption` exports the approved 170 × 166 mm standalone version. `--output-dir DIRECTORY` and `--data-dir DIRECTORY` support isolated reproduction without changing source files. No manuscript is modified.

## Compact frozen inputs

All inputs required for reaggregation and drawing are in `data/fig6/`; the directory can travel with this script independently of the rest of the repository. `source_manifest.json` retains the original source locations, selection rules and SHA256 digests from the checked 2026-10-14 package. These original locations describe provenance; the script reads only the adjacent snapshots and verifies their copied digests. Existing `data/fig6_scenarios.csv` and the shared main-figure data files remain unchanged for compatibility.

- `paysim_seed_scenarios.csv`: 660 saved seed-level scenario records. Recall differences are paired within seed and averaged across five seeds; Test alert rates are five-seed means. Cost reduction is the mean of seedwise `100 × (C_P0 - C_PTHW) / C_P0`, with `C = cFN + FP`, rather than a ratio of mean costs.
- `capacity_seed_points.csv`: the 60 PTHW/Platt records used at exact 0.1%, 1%, 3% and 5% completed-Test-window capacities. The script validates `K = ceil(qN)` and reaggregates `100 × TP / frauds` for each of the three windows.
- `fig6_scenarios.csv`: the previously frozen IEEE-CIS five-seed budget means. The corresponding full LightGBM predictions are not in this compact package; their rankings are not independently regenerated.
- `primary_effects.csv` and `paysim_bootstrap_summary.csv`: the saved PaySim reference recall estimates and conditional interval endpoints, cross-checked in their original units.
- `ieee_reference_effects.csv` and `ieee_provenance.json`: the reported Table 9 IEEE-CIS reference estimates and intervals, with the original extraction and precision notes.
- `capacity_contract.json`: the saved exact-budget decision contract.
- `expected_plot_data.csv`: the approved 111-point coordinate/interval reference, used as a regression check.

The script checks 660 scenario-record identities, five-seed pairing, 60 exact-budget records, all 111 plotted coordinates, six saved intervals, 24 line series and visible text bounds. Panel (e) reuses previously reported pointwise conditional 95% intervals and uses separate PaySim and IEEE-CIS horizontal scales. It creates no pooled effect, model fits, predictions or bootstrap draws. The glow is decorative fixed-width styling and encodes no uncertainty. Validation-threshold transfer and exact completed-Test-window ranking remain distinct decisions.

`fig6_reproduction_check.json` records the 2026-10-15 independent temporary-directory redraw and manuscript-image comparison. Reproducibility here means reaggregation of saved records and figure export; it does not mean rerunning raw transactions through training, ranking or bootstrap interval construction.
