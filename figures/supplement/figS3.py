"""Independent redraw using only the neighbouring aggregate data JSON.
No project raw records, labels, scores, models, or resampling arrays are read.
Run with an existing Python environment containing matplotlib and numpy.
"""
from typing import Any
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

fig, axs = plt.subplots(3, 3, figsize=D["figsize_inches"], sharex=True, sharey=True)
fig.subplots_adjust(left=.095, right=.965, bottom=.13, top=.935, wspace=.17, hspace=.23)
methods = ["Uncalibrated", "Platt", "Isotonic"]
seeds = [42, 52, 62, 72, 82]
colors = [NAVY, TEAL, "#7096AC", "#305E60", "#95A5B0"]
markers = ["o", "s", "^", "D", "v"]
lines: list[Any] = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]
series_count = 0
point_count = 0
for i, fold in enumerate([1, 2, 3]):
    for j, method in enumerate(methods):
        ax = axs[i, j]
        clean(ax, "both")
        ax.plot([0, 1], [0, 1], color=GRAY, linestyle=(0, (3, 3)), linewidth=.7, zorder=1)
        for seed, color, marker, ls in zip(seeds, colors, markers, lines):
            rows = sorted([r for r in D["rows"] if r["fold"] == fold and r["calibration_method"] == method and r["seed"] == seed], key=lambda r: r["bin_id"])
            assert rows and all(r["sample_count"] > 0 for r in rows)
            ax.plot([r["mean_predicted_probability"] for r in rows], [r["observed_fraud_rate"] for r in rows], color=color, marker=marker, linestyle=ls, markersize=3.2, linewidth=.9, markeredgewidth=.35, zorder=2, clip_on=False)
            series_count += 1
            point_count += len(rows)
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1])
        if i == 0:
            ax.set_title(method, fontsize=9, fontweight="semibold", pad=8)
        if j == 0:
            ax.set_ylabel(f"Fold {fold}\nObserved fraud fraction", fontsize=8)
        if i == 2:
            ax.set_xlabel("Mean predicted probability", fontsize=8)
        ax.tick_params(labelsize=8)
legend = [Line2D([], [], color=c, marker=m, linestyle=line_style, markersize=3.5, linewidth=.9, label=str(s)) for s, c, m, line_style in zip(seeds, colors, markers, lines)]
fig.legend(handles=legend, title="Seed", title_fontsize=8, loc="lower center", bbox_to_anchor=(.53, .016), ncol=5, frameon=False, handlelength=2, columnspacing=1.2, handletextpad=.4)
assert series_count == 45 and point_count == 299
assert sum(r["sample_count"] < 10 for r in D["rows"]) == 50
finish(fig, {"series": series_count, "nonempty_bins": point_count, "sparse_nonempty_bins_below_10": 50, "fixed_partition_total_bins": 450, "empty_bins_omitted": 151, "pooled_bins": False, "intervals": 0, "probability_axes_0_to_1": True})
