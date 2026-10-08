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

fig, axs = plt.subplots(1, 3, figsize=D["figsize_inches"], sharey=True)
fig.subplots_adjust(left=.095, right=.985, bottom=.31, top=.83, wspace=.2)
styles = {"P_H_BASE": (NAVY, "o", "PH − P0"), "P_H_TIME": (TEAL, "s", "PTH − PT"), "P_WEIGHT": (SLATE, "^", "PTHW − PTH")}
stages = ["raw", "clipped", "platt"]
for ax, fold, letter in zip(axs, [1, 2, 3], ["(a)", "(b)", "(c)"]):
    clean(ax, "y")
    ax.axhline(0, color=GRAY, linestyle=(0, (3, 3)), linewidth=.7)
    for contrast, (color, marker, label) in styles.items():
        rows = [next(r for r in D["rows"] if r["fold"] == fold and r["contrast"] == contrast and r["score_stage"] == s) for s in stages]
        ax.plot(range(3), [r["delta_ap"] for r in rows], color=color, marker=marker, markersize=4, linewidth=1.1, label=label)
    panel(ax, letter, "PaySim · Fold " + str(fold))
    ax.set(xlim=(-.12, 2.12), ylim=(-.075, .092), xticks=[0, 1, 2], xticklabels=["Raw", "Clipped", "Platt"], yticks=[-.05, 0, .05])
axs[0].set_ylabel("AP difference")
fig.legend(handles=axs[0].get_legend_handles_labels()[0], labels=axs[0].get_legend_handles_labels()[1], loc="lower center", bbox_to_anchor=(.5, .015), ncol=3, frameon=False, handletextpad=.4, columnspacing=1.5)
finish(fig, {"paired_five_seed_mean_points": 27, "series": 9, "score_stages": stages, "no_continuous_optimisation": True, "intervals": 0})
