# Current standalone figures

`main/` follows the six-figure manuscript numbering adopted on October 13, 2026, with the final Figure 1/5 layout refinements and the five-panel Figure 6. `supplement/` contains Figures S1-S9. S1 retains the original scatter layout; S2 uses the compact forest style; S3 uses decorative soft halos and reduced plot areas; S7 uses the selected three-row layout; S9 uses six condition tiles and four I (pp) bars. The [October 16 update](../docs/UPDATES_20261016.md) records the final adopted versions. The earlier public presentation remains available at `b52524c`; its numbering is distinct from the current six figures.

| Figure | Content | Published presentation |
| --- | --- | --- |
| 1 | Temporal designs, workflows and later analysis roles | Adopted PNG/SVG and matching-layout PDF; Decisions text fitted within its card |
| 2 | Study-specific experimental framework | Adopted browser PNG, native editable SVG and PDF |
| 3 | PaySim transferred-threshold recall/AP effects and exact-capacity profiles | Adopted PNG, SVG and PDF |
| 4 | BankSim paired capture effects and post-hoc capture exchanges | Adopted PNG, SVG and PDF; merges the earlier manuscript Figures 4 and 5 |
| 5 | IEEE-CIS combined effects, component effects and directly paired workflow differences | Adopted six-panel blue/orange PNG, SVG and PDF; three-column numerical rows and horizontal intervals |
| 6 | Capacity, analytical cost, realised workload and reference recall effects | Adopted five-panel PNG, SVG and PDF, matching the current manuscript; intermediate manuscript Figure 7 |

`main/NUMBERING.json` records both the earlier eight-figure and intermediate seven-figure manuscript mappings. `main/CAPTIONS.md` contains the current adopted annotations. Only individual figure media are published; manuscript Word/PDF files and whole-page manuscript screenshots are excluded.

Install the optional drawing dependencies with `pip install -e ".[figures]"`. Run scripts with an explicit temporary `--output-dir` where supported, or use a copy of the figure directory to preserve adopted exports. Main Figures 2 and 4 use native SVG/PyMuPDF rendering; the other main scripts use Matplotlib. S2 uses Matplotlib; S3/S7/S9 use ReportLab with PyMuPDF rasterization. S1 retains its original entry point. See `main/fig6_requirements.txt` for the recorded Figure 6 environment. Figure 2 selects paired regular/bold Arial, Liberation Sans or DejaVu Sans fonts, or an explicit `FIGURE_FONT_DIR`; fonts are not redistributed. Its bounding checks reject wider metrics that do not fit the frozen layout (DejaVu Sans was correctly rejected locally). Use Arial or a compatible paired font for that layout. Fonts and renderers can affect appearance; manifests authenticate recorded bytes rather than universal cross-platform raster identity.

Figure 3 reads eight frozen primary effect rows and six 501-point capacity profiles. The accompanying seed-capacity CSV contains aggregate counts and recalls, not transaction-level observations. Profiles were calculated in the earlier revision stage from saved Platt predictions, with seed-wise ranking and source identity breaking ties. Drawing reads the saved aggregates without repeating ranking, training, prediction, calibration or resampling. Panel (c) has no intervals and uses a different decision rule from panels (a-b).

Figure 4 retains the original 95% H0 and 98.75% H1/D intervals. G/L heatmaps are post-hoc descriptive paired means; negative D, zero loss, unrounded original endpoints and observed-point arithmetic bounds are retained. Repeated H1/net estimates provide no independent evidence.

Figure 5 reads nine combined-metric and eighteen component-effect rows extracted at the reported decimal precision of Tables 9 and 10. Original higher-precision experimental records remain in the aggregate archives. All 81 reported numerical values are retained. The adopted layout uses single-line numerical rows, a right-aligned CI column, horizontal TP intervals and a signed blue/orange palette. Only delta TP shares a position/colour scale; Recall and AP have separate units. Direct component intervals come from the paired workflow differences, not endpoint subtraction. Colours encode direction, not significance. Different preprocessing prevents isolation of an algorithm-family effect. No new intervals are computed.

Figure 6 is independently reproducible from its adjacent compact snapshots in `main/data/fig6/`, through `python main/fig6.py --output-dir <directory>`. It reaggregates 660 saved scenario rows and 60 exact-budget seed rows, retains all 111 plotted points and six saved conditional intervals, and verifies the frozen source hashes. Panel (e) reuses the 3% reference effects on separate dataset scales; it adds no pooled or between-window inference. The default PNG matches the manuscript image byte for byte in the recorded environment. See [its source guide](main/fig6_README.md) for the full-caption option and reproduction limits.

Figure S1 retains the original two scatter panels for PaySim and IEEE-CIS and the original short caption. Figure S8 retains the separate heatmap of the same 12 component differences; no additional S1 heatmap panel is included. The original S1 PNG, PDF, drawing script and source JSON are preserved.

Figure S2 retains the three original M2-minus-M0 effects and conditional intervals, including the Fold 2 interval spanning zero. Its compact forest layout changes presentation only. Figure S7 retains the original descriptive values in the three-row layout selected after the landscape comparison.

Figure S9 reads adjacent `supplement/figS9_data.json` and checks its values against the preserved S22 result CSVs. The six whole-window condition tiles retain TP, recall and uncaptured frauds. Four bars show I=100J/P in percentage points, with J and denominator labels. The same latest PNG is embedded in the local supplement. No fitting or uncertainty analysis is repeated. Segment selections use local budgets and do not decompose the whole-window list; the four contrasts are not independent replications. `supplement/figS9_verification.json` records source hashes and arithmetic checks. Earlier reconstruction and paired-point layouts are superseded.

`MANIFEST.json` authenticates the current scripts, compact inputs, captions and exports. Figure reproduction checks concern saved aggregates and drawing only; no raw-to-model scientific reproduction is claimed.

The historical scientific stage index is [UPDATES_20261008.md](../docs/UPDATES_20261008.md). The earlier presentation update is [UPDATES_20261013.md](../docs/UPDATES_20261013.md); the current update is [UPDATES_20261016.md](../docs/UPDATES_20261016.md).
