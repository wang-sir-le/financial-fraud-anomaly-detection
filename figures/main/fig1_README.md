# Figure 1: named data fields, metric reports and saved outputs

The adopted 2026-10-24 revision replaces the decorative feature/readout/S21 matrices with defined schemas: prior count `n`, prior amount mean `μ`, recency `Δt`; AP/Recall/Cost report fields; and saved Scores/Draws files. These icons identify fields and existing artifacts without inventing measurements. The dataset-specific temporal splits, calibration/decision rules, original and post-hoc model scopes, and S21/S22 boundaries remain explicit. The authoritative submission caption is `fig1_caption_submission.txt`.

```bash
python fig1.py --out-dir /tmp/fig1-redraw
python fig1.py --out-dir /tmp/fig1-redraw --font-family "Liberation Sans"
```

Python 3.10+ and Matplotlib are required. Run the script by its path from any directory. The adjacent `data/fig1_protocol.json` is authenticated against its unchanged SHA-256 before drawing. No manuscript, author workspace, private dependency, model fit, prediction, resampling or network service is used.

Install Arial for the adopted letterforms. The script finds installed fonts through Matplotlib; metric-compatible Liberation Sans is the default fallback. `--font-family` can select another installed family. Both regular/bold glyph coverage and full rendered geometry are checked, and an incompatible wider/taller font is rejected instead of exporting clipped text. Fonts are not redistributed. The adopted exports use Arial. No cross-platform byte identity is claimed.

The publication width is exactly **170.0 mm**, height **115.287356321839 mm**. The drawing retains its 174×118 coordinate geometry; figure size, rather than a scaled raster, sets the physical dimensions. Lettering remains **8–11 pt** at the adopted size and strokes are at least **0.4 pt**.

Exports:

- `fig1.png`: 600 dpi, 4015×2723 pixels; the identical PNG is used in Word.
- `fig1.svg`: pure vector with outlined text, independent of installed viewer fonts.
- `fig1_editable.svg`: pure vector with editable text; needs a compatible font when edited.
- `fig1.pdf`: vector PDF with embedded TrueType letterforms.
- `fig1.eps`: vector EPS for journal production.
- `fig1_preview.png`: 300 dpi review preview.
- `fig1_validation.json`: glyph, font, stroke, canvas/cell/overlap checks, output dimensions and hashes.
- `fig1_sync_verification.json`: checks on the adopted publication exports, including PDF fonts and absence of raster content in vector files.

The PNG and an independently rendered PDF were visually checked. All 113 labels pass their assigned bounds and pairwise overlap checks at the final physical size. No experiment protocol or scientific result was modified.
