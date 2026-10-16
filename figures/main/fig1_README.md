# Figure 1: adopted Decisions text fit

The adopted PNG and editable SVG preserve the author exports from the October 16, 2026 Decisions-card correction. The 8 pt wording remains intact; “Completed Test ranking” occupies two lines and all eight Decisions labels fit inside the inner white card. The standalone PDF was generated from the updated public drawing script because that revision did not supply a PDF.

```bash
python fig1.py --out-dir /tmp/fig1-redraw
```

Run from any directory using the path to `fig1.py`. Python 3.10+, Matplotlib and the adjacent `data/fig1_protocol.json` are required; the source JSON is authenticated before drawing. There is no dependency on a manuscript, author workspace, network service or privately held template. No model is fitted and no outcome is generated.

The script creates PNG, editable SVG, vector PDF, a preview and `fig1_validation.json`. It checks a minimum 8 pt font, canvas/cell bounds and the tighter Decisions-card bounds. The redraw retains all 98 visible SVG text labels of the adopted artwork. `fig1_sync_verification.json` records the publication checks.

Adopted media are the delivered publication artifacts. Freshly redrawn PNG/SVG/PDF bytes can differ because the adopted PNG retained the earlier raster outside the edited Decisions region, and fonts or renderers vary. The local redraw has matching text and geometry; byte identity is not claimed across environments. Font files are not redistributed. Use `--out-dir` to preserve adopted files.
