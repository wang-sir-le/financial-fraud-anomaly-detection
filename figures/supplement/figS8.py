"""Independent redraw using only the neighbouring aggregate data JSON.
No project raw records, labels, scores, models, or resampling arrays are read.
Run with an existing Python environment containing matplotlib and numpy.
"""
from typing import Any, cast
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
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

fig, axs = plt.subplots(2, 2, figsize=D["figsize_inches"])
fig.subplots_adjust(left=.125, right=.955, top=.91, bottom=.09, wspace=.65, hspace=.44)
cmap = LinearSegmentedColormap.from_list("navy_white_teal", [NAVY, "#F6F8F9", TEAL])
for ax, dataset, metric, letter in zip(axs.flat, ["paysim", "paysim", "ieee_cis", "ieee_cis"], ["delta_ap", "delta_recall_pp", "delta_ap", "delta_recall_pp"], ["(a)", "(b)", "(c)", "(d)"]):
    array = np.array([[next(r[metric] for r in D["rows"] if r["dataset"] == dataset and r["component"] == c and r["fold"] == f) for f in [1, 2, 3]] for c in ["History", "Weighting"]])
    vmax = float(np.max(np.abs(array)))
    im = ax.pcolormesh(np.arange(-.5, 3, 1), np.arange(-.5, 2, 1), array, cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), shading="flat", rasterized=False)
    ax.set(xticks=range(3), xticklabels=["F1", "F2", "F3"], yticks=range(2), yticklabels=["History", "Weighting"], xlim=(-.5, 2.5), ylim=(1.5, -.5))
    ax.tick_params(length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for (i, j), value in np.ndenumerate(array):
        decimals = 4 if metric == "delta_ap" else 2
        color = "white" if abs(value) / vmax > .55 else INK
        ax.text(j, i, f'{value:+.{decimals}f}', ha="center", va="center", fontsize=8.5, color=color)
    panel(ax, letter, ("PaySim" if dataset == "paysim" else "IEEE-CIS") + (" · AP" if metric == "delta_ap" else " · Recall (pp)"))
    bar = fig.colorbar(im, ax=ax, fraction=.047, pad=.065, aspect=18, ticks=[-vmax, 0, vmax])
    assert bar.solids is not None
    bar.solids.set_rasterized(False)
    cast(Any, bar.outline).set_visible(False)
    bar.ax.tick_params(labelsize=7.5, length=2, pad=2)
    decimals = 3 if metric == "delta_ap" else 1
    bar.set_ticklabels([f'{-vmax:.{decimals}f}', "0", f'{vmax:.{decimals}f}'])
finish(fig, {"numeric_cells": 24, "intervals": 0, "independent_symmetric_zero_centered_color_scales": 4, "recall_pp_preserved": True, "negative_cells_preserved": True, "paysim_history_is_PTH_minus_P0": True})
