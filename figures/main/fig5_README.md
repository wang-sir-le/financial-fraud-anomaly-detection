# Figure 5: adopted blue/orange three-column layout

The adopted PNG, editable SVG and vector PDF preserve the author exports from October 16, 2026. Upper cards use one row per metric with Metric / Estimate / 95% CI columns, right-aligned intervals and a heading centred over their fixed column. Horizontal ΔTP strips are 12 pt high, markers are 6 pt, card borders are lighter, and title bands are shallower. Orange means a negative point estimate; blue means a positive point estimate. Colour does not indicate significance.

```bash
python fig5.py --out-dir /tmp/fig5-redraw
```

Python 3.11+ and Matplotlib are required. The script reads only the adjacent frozen `data/fig5_combined_metrics.csv`, `data/fig5_component_effects.csv` and provenance JSON. It has no manuscript, author-workspace, network or private-template dependency. Install the repository drawing dependencies with `pip install -e ".[figures]"` when needed.

All nine combined-metric rows and 18 component rows are retained, including 81 numerical point/endpoint values. The script checks sequential point-estimate identities and directly paired workflow-difference directions. It retains original conditional intervals and the −40 to +40 ΔTP and symmetric ±120 colour scales. No predictions, fits, bootstrap draws or new intervals are computed.

A redraw creates PNG, editable SVG, vector PDF, preview and `fig5_validation.json`, which includes input hashes, source counts, typography and canvas checks. All 126 visible adopted SVG text labels are retained. `fig5_sync_verification.json` records the publication checks. The complete within-workflow component intervals remain in the frozen CSV; only direct component intervals are printed in the figure.

Adopted media are the delivered publication artifacts. The public Matplotlib script regenerates the same content and layout; font and renderer differences can change export bytes. The adopted PNG used the author SVG raster renderer and the adopted PDF used the author vector renderer. Byte identity across renderers or environments is not claimed. Use `--out-dir` to preserve adopted files. Font files are not redistributed.
