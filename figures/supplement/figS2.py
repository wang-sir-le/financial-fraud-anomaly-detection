"""Independent redraw using only the neighbouring aggregate data JSON.
No project raw records, labels, scores, models, or resampling arrays are read.
Run with an existing Python environment containing matplotlib and numpy.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem
D = json.loads((HERE / (STEM + "_data.json")).read_text(encoding="utf-8"))
NAVY = "#234E70"
TEAL = "#008C95"
SLATE = "#7096AC"
INK = "#25343D"
GRAY = "#7C8B93"
GRID = "#E8EEF1"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": GRAY,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "lines.linewidth": 1.15,
    "xtick.major.width": 0.55, "ytick.major.width": 0.55,
    "xtick.major.size": 2.7, "ytick.major.size": 2.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.unicode_minus": True,
})

def clean(ax, grid_axis=None):
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.set_axisbelow(True)
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)

def panel(ax, label, title):
    ax.set_title(label + "  " + title, loc="left", pad=7, fontweight="semibold")

def finish(fig, verification):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    outside = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer=renderer)
        if box.width == 0 or box.height == 0:
            continue
        if box.x0 < canvas.x0-.5 or box.y0 < canvas.y0-.5 or box.x1 > canvas.x1+.5 or box.y1 > canvas.y1+.5:
            outside.append({"text": artist.get_text(), "box": list(box.bounds)})
    assert not outside, outside
    fig.savefig(HERE / (STEM + ".png"), dpi=450)
    fig.savefig(HERE / (STEM + ".pdf"))
    assert tuple(round(x, 6) for x in fig.get_size_inches()) == tuple(round(x, 6) for x in D["figsize_inches"])
    (HERE / (STEM + "_verification.json")).write_text(json.dumps({"figure": STEM, "source_hashes": D["sources"], "original_extent_emu": D["original_extent_emu"], "dpi": 450, "rendered_inches": list(fig.get_size_inches()), "font_size_pt": "Main labels 8–9 pt; S2 title 9.5; S8 colorbar 7.5", "text_canvas_bounds": "PASS", "new_model_or_statistical_calls": 0, **verification}, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.close(fig)

fig, ax = plt.subplots(figsize=D["figsize_inches"])
fig.subplots_adjust(left=.13, right=.98, top=.80, bottom=.25)
clean(ax, "y")
ax.axhline(0, color=GRAY, linewidth=.8, linestyle=(0, (3, 3)))
position_text = ["CI below zero", "CI crosses zero", "CI above zero"]
for x, r in enumerate(D["rows"], 1):
    ax.errorbar(x, r["estimate_pp"], yerr=[[r["estimate_pp"] - r["lower_pp"]], [r["upper_pp"] - r["estimate_pp"]]], fmt="o", color=NAVY, ecolor=NAVY, markersize=4.5, capsize=4, elinewidth=1.3, capthick=.8, zorder=3)
    ax.annotate(f'{r["estimate_pp"]:+.3f}', (x, r["estimate_pp"]), xytext=(10, 0), textcoords="offset points", fontsize=8.5, va="center", color=NAVY)
    ax.text(x, .99, position_text[x-1], transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=INK)
ax.set(xlim=(.45, 3.55), ylim=(-1.82, 1.78), xticks=[1, 2, 3], xticklabels=["Fold 1", "Fold 2", "Fold 3"], yticks=[-1.5, -1, -.5, 0, .5, 1, 1.5])
ax.set_ylabel("Recall@3% difference (pp)")
ax.set_title("IEEE-CIS · C02 = M2 − M0", loc="left", fontsize=9.5, fontweight="semibold", pad=28)
fig.text(.5, .085, "Five-seed means · Pointwise 95% paired moving-block intervals", ha="center", fontsize=8)
fig.text(.5, .033, "An interval crossing zero does not establish equivalence.", ha="center", fontsize=8, color=GRAY)
finish(fig, {"points": 3, "intervals": 3, "percent_to_pp_factor": 100, "inference_categories_color_coded": False, "fold_2_zero_crossing_preserved": True})
