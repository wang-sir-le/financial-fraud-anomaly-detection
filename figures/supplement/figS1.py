"""Independent redraw using only the neighbouring aggregate data JSON.
No project raw records, labels, scores, models, or resampling arrays are read.
Run with an existing Python environment containing matplotlib and numpy.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

fig, axs = plt.subplots(1, 2, figsize=D["figsize_inches"])
fig.subplots_adjust(left=.09, right=.99, bottom=.17, top=.84, wspace=.33)
styles = {"History": (NAVY, "o"), "Weighting": (TEAL, "s")}
offsets = {
    ("paysim", "History", 1): (6, 5), ("paysim", "History", 2): (6, 5),
    ("paysim", "History", 3): (6, 5), ("paysim", "Weighting", 1): (-7, 13),
    ("paysim", "Weighting", 2): (-7, 8), ("paysim", "Weighting", 3): (6, -12),
    ("ieee_cis", "History", 1): (11, 13), ("ieee_cis", "History", 2): (-27, -15),
    ("ieee_cis", "History", 3): (6, 5), ("ieee_cis", "Weighting", 1): (-25, -14),
    ("ieee_cis", "Weighting", 2): (-27, 13), ("ieee_cis", "Weighting", 3): (6, 5),
}
for ax, dataset, title, letter in zip(axs, ["paysim", "ieee_cis"], ["PaySim", "IEEE-CIS"], ["(a)", "(b)"]):
    clean(ax)
    ax.axhline(0, color=GRAY, linestyle=(0, (3, 3)), linewidth=.7)
    ax.axvline(0, color=GRAY, linestyle=(0, (3, 3)), linewidth=.7)
    for r in D["rows"]:
        if r["dataset"] != dataset:
            continue
        color, marker = styles[r["component"]]
        ax.scatter(r["delta_ap"], r["delta_recall_pp"], c=color, marker=marker, s=27, zorder=3)
        dx, dy = offsets[(dataset, r["component"], r["fold"])]
        ax.annotate("F" + str(r["fold"]), (r["delta_ap"], r["delta_recall_pp"]), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": color, "linewidth": .4, "shrinkA": 1, "shrinkB": 4} if dataset == "ieee_cis" and r["fold"] < 3 else None)
    panel(ax, letter, title)
    ax.set_xlabel("AP difference")
    ax.set_ylabel("Recall difference (pp)")
axs[0].set(xlim=(-.17, .19), ylim=(-9, 68), xticks=[-.1, 0, .1], yticks=[0, 20, 40, 60])
axs[1].set(xlim=(-.025, .045), ylim=(-5, 6), xticks=[-.02, 0, .02, .04], yticks=[-4, -2, 0, 2, 4, 6])
handles = [Line2D([], [], color=c, marker=m, linestyle="None", markersize=4.5, label=k) for k, (c, m) in styles.items()]
fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(.5, .985), frameon=False, handletextpad=.4, columnspacing=2)
finish(fig, {"points": 12, "intervals": 0, "zero_lines": 4, "independent_dataset_axes": True, "retained_full_rounded_annotations_in_S8": True})
