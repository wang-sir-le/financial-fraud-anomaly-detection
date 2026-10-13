# Current standalone figures

`main/` follows the six-figure manuscript numbering adopted on October 13, 2026. `supplement/` retains Figures S1-S8 from the previously published snapshot. The older six-main-figure public presentation remains available at `b52524c`; its numbering is distinct from the current six figures. Supplementary exports are unchanged.

| Figure | Content | Published presentation |
| --- | --- | --- |
| 1 | Temporal designs, workflows and later analysis roles | Adopted PNG, SVG and PDF |
| 2 | Study-specific experimental framework | Adopted browser PNG, native editable SVG and PDF |
| 3 | PaySim transferred-threshold recall/AP effects and exact-capacity profiles | Adopted PNG, SVG and PDF |
| 4 | BankSim paired capture effects and post-hoc capture exchanges | Adopted PNG, SVG and PDF; merges the earlier manuscript Figures 4 and 5 |
| 5 | IEEE-CIS combined effects, component effects and directly paired workflow differences | Adopted six-panel PNG, SVG and PDF; merges the intermediate manuscript Figures 5 and 6 |
| 6 | Capacity, analytical cost and realised workload sensitivity | Adopted PNG, SVG and PDF; intermediate manuscript Figure 7 |

`main/NUMBERING.json` records both the earlier eight-figure and intermediate seven-figure manuscript mappings. `main/CAPTIONS.md` contains the current adopted annotations. Only individual figure media are published; manuscript Word/PDF files and whole-page manuscript screenshots are excluded.

Install the optional drawing dependencies with `pip install -e ".[figures]"`. Run scripts in a copy of `main/` to preserve the delivered exports. Figures 2 and 4 use native SVG/PyMuPDF rendering; the other scripts use Matplotlib. Figure 2 selects paired regular/bold Arial, Liberation Sans or DejaVu Sans fonts, or an explicit `FIGURE_FONT_DIR`; fonts are not redistributed. Its bounding checks reject wider metrics that do not fit the frozen layout (DejaVu Sans was correctly rejected locally). Use Arial or a compatible paired font for that layout. Fonts and renderers can affect appearance; manifests authenticate recorded bytes, not universal cross-platform raster identity.

Figure 3 reads eight frozen primary effect rows and six 501-point capacity profiles. The accompanying seed-capacity CSV contains aggregate counts and recalls, not transaction-level observations. Profiles were calculated in the earlier revision stage from saved Platt predictions, with seed-wise ranking and source identity breaking ties. Drawing reads the saved aggregates without repeating ranking, training, prediction, calibration or resampling. Panel (c) has no intervals and uses a different decision rule from panels (a-b).

Figure 4 retains the original 95% H0 and 98.75% H1/D intervals. G/L heatmaps are post-hoc descriptive paired means; negative D, zero loss, unrounded original endpoints and observed-point arithmetic bounds are retained. Repeated H1/net estimates provide no independent evidence.

Figure 5 reads nine combined-metric and eighteen component-effect rows extracted at the reported decimal precision of Tables 9 and 10. Original higher-precision experimental records remain in the aggregate archives. All 81 reported numerical values are retained. Only delta TP shares a position/colour scale; Recall and AP have separate units. Direct component intervals come from the paired workflow differences, not endpoint subtraction. Colours encode direction, not significance. Different preprocessing prevents isolation of an algorithm-family effect. No new intervals are computed.

The historical scientific stage index is [UPDATES_20261008.md](../docs/UPDATES_20261008.md). Current presentation and publication adaptations are in [UPDATES_20261013.md](../docs/UPDATES_20261013.md).
