"""Redraw Fig. S2 in the supplied table-and-forest visual style.

All three estimates and their intervals come unchanged from the existing
aggregate Fig. S2 JSON. No new inference or pooled estimate is calculated.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

from _figure_io import output_directory, write_verification

HERE = Path(__file__).resolve().parent
OUT = output_directory(HERE, "Draw the adopted compact S2 table and forest plot.")
SOURCE = HERE / "figS2_data.json"
PDF = OUT / "figS2.pdf"
STEM = "figS2"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == "78158681ae24e0c82878e5c4936bf77f0b4472bb88654c8064e96bf804233358"

raw = SOURCE.read_bytes()
data = json.loads(raw)
rows = data["rows"]
assert [r["fold"] for r in rows] == [1, 2, 3]
assert all(r["lower_pp"] < r["estimate_pp"] < r["upper_pp"] for r in rows)
assert rows[0]["upper_pp"] < 0 < rows[2]["lower_pp"]
assert rows[1]["lower_pp"] < 0 < rows[1]["upper_pp"]

INK = "#111111"
BLUE = "#15558A"
CI = "#949494"
GRID = "#EFEFEF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.6,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": True,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

# Physical coordinates make the 14 pt row rhythm match the supplied reference.
WIDTH_PT, HEIGHT_PT = 160 / 25.4 * 72, 43 / 25.4 * 72
fig = plt.figure(figsize=(WIDTH_PT / 72, HEIGHT_PT / 72), dpi=150)

def text(x, y, s, **kw):
    return fig.text(x / WIDTH_PT, y / HEIGHT_PT, s, va="center", **kw)

# Thin table rules and aligned numeric columns follow the supplied reference.
cols = {"fold": 7, "estimate": 113.5, "ci": 222.5}
head_y = 110.7
for y in (117.5, 103.9):
    fig.add_artist(Line2D([7 / WIDTH_PT, 294.5 / WIDTH_PT],
                         [y / HEIGHT_PT, y / HEIGHT_PT],
                         transform=fig.transFigure, color=INK, lw=.50))
text(cols["fold"], head_y, "Contrast / fold")
text(cols["estimate"], head_y, "Estimate", ha="center")
text(cols["ci"], head_y, "95% CI", ha="center")
text(7, 92, "IEEE-CIS · C02 = M2 − M0 · Exact 3% review capacity",
     fontsize=8.6, fontweight="bold")

row_y = [78, 64, 50]
positions = ["Below zero", "Crosses zero", "Above zero"]

def num(x):
    return f"{x:.3f}".replace("-", "−")

for r, y, position in zip(rows, row_y, positions):
    text(cols["fold"], y, f"Fold {r['fold']}")
    text(cols["estimate"], y, num(r["estimate_pp"]), ha="center")
    text(cols["ci"], y,
         f"({num(r['lower_pp'])}, {num(r['upper_pp'])})", ha="center")

# The axes use page points as y data so every interval is level with its table row.
bottom, top = 43, 117.5
ax = fig.add_axes([317.5 / WIDTH_PT, bottom / HEIGHT_PT,
                   (WIDTH_PT - 7 - 317.5) / WIDTH_PT, (top - bottom) / HEIGHT_PT])
ax.set_ylim(bottom, top)
ax.set_xlim(-2.0, 2.0)
ax.set_xticks([-2, -1, 0, 1, 2])
ax.tick_params(axis="x", labelsize=8.0, length=2.7, width=.5,
               color="#777777", pad=2.6)
ax.set_yticks([])
ax.spines[["left", "top", "right"]].set_visible(False)
ax.spines["bottom"].set_color("#777777")
ax.spines["bottom"].set_linewidth(.6)
ax.set_axisbelow(True)
for x in [-2, -1, 1, 2]:
    ax.axvline(x, color=GRID, lw=.65, linestyle=(0, (3.5, 3.5)), zorder=0)
ax.axvline(0, color="#777777", lw=.75, linestyle=(0, (3.5, 3.5)), zorder=1)

rendered = []
for r, y in zip(rows, row_y):
    estimate, lower, upper = (r[k] for k in ("estimate_pp", "lower_pp", "upper_pp"))
    artist = ax.errorbar(
        estimate, y,
        xerr=[[estimate - lower], [upper - estimate]],
        fmt="s", color=BLUE, markeredgewidth=.75,
        markersize=4.4, ecolor=CI, elinewidth=.85,
        capsize=1.8, capthick=.65, zorder=3,
    )
    # An independent extraction from the plotted line collection verifies endpoints.
    segment = artist.lines[2][0].get_segments()[0]
    plotted = {"fold": r["fold"], "estimate_pp": float(artist.lines[0].get_xdata()[0]),
               "lower_pp": float(segment[0][0]), "upper_pp": float(segment[1][0])}
    assert all(abs(plotted[k] - r[k]) < 1e-12
               for k in ("estimate_pp", "lower_pp", "upper_pp"))
    rendered.append(plotted)
ax.set_xlabel("ΔRecall@3% (pp)", fontsize=8.5, labelpad=4)

text(7, 26,
     "Five-seed means; pointwise conditional 95% paired moving-block percentile CIs.", fontsize=7.0)
text(7, 14,
     "Crossing zero does not establish equivalence.", fontsize=7.0)

# Check that all visible text is inside the figure and does not overlap other text.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = []
outside = []
for t in fig.findobj(match=Text):
    if not t.get_visible() or not t.get_text().strip():
        continue
    box = t.get_window_extent(renderer)
    if box.width == 0 or box.height == 0:
        continue
    texts.append((t.get_text(), box))
    if (box.x0 < -.5 or box.y0 < -.5 or
            box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5):
        outside.append(t.get_text())
assert not outside, outside
overlaps = []
for i, (label, box) in enumerate(texts):
    for other_label, other_box in texts[i + 1:]:
        if box.overlaps(other_box):
            overlaps.append([label, other_label])
assert not overlaps, overlaps

PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / f"{STEM}.png", dpi=600)
fig.savefig(OUT / f"{STEM}.svg")
fig.savefig(PDF)
plt.close(fig)

with (OUT / "figS2_plot_data.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["fold", "seeds", "estimate_pp", "lower_pp", "upper_pp", "ci_position"])
    writer.writeheader()
    for r, position in zip(rows, positions):
        writer.writerow({**r, "seeds": 5, "ci_position": position})

validation = {
    "figure": "S2",
    "source_path": SOURCE.name,
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "source_evidence": data["sources"],
    "source_numeric_values_preserved": True,
    "rendered_endpoints_checked": rendered,
    "points": 3,
    "intervals": 3,
    "point_marker": "same-size dark-blue square for every fold",
    "pooled_estimate_added": False,
    "p_values_added": False,
    "new_statistical_inference": False,
    "interval_label": "pointwise conditional 95% percentile intervals from paired moving-block draws",
    "fold_2_crosses_zero": True,
    "text_outside_canvas": outside,
    "overlapping_text": overlaps,
    "size_mm": [160, 43],
    "row_spacing_pt": 14,
    "body_font_size_pt": 8.6,
    "layout_revision": "Single compact contrast row, three table columns, 14 pt data rows, shorter notes; removed large title, Seeds column and CI-position column.",
    "png_dpi": 600,
    "pdf_path": PDF.name,
    "svg_text_editable": True,
    "style_reference": "First user-supplied image: aligned table, thin rules, white background, blue squares, grey horizontal intervals, vertical zero reference.",
}
write_verification(OUT / "figS2_verification.json", validation, [OUT / "figS2.png", OUT / "figS2.pdf", OUT / "figS2.svg"])
(OUT / "figS2_statistical_notes.txt").write_text(
    "IEEE-CIS, C02 = M2 - M0, exact 3% review capacity.\n"
    "The point estimates are five-seed means (seeds 42, 52, 62, 72, 82).\n"
    "Intervals are pointwise conditional 95% percentile intervals from 5,000 paired moving-block draws.\n"
    "Intervals condition on fixed fitted models and observed Test windows.\n"
    "Recall differences are in percentage points (pp).\n"
    "An interval crossing zero does not establish equivalence.\n"
    "No pooled effect, P value, new resampling or new statistical inference was added.\n",
    encoding="utf-8",
)
