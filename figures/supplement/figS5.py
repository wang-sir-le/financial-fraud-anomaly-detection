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

fig = plt.figure(figsize=D["figsize_inches"])
gs = fig.add_gridspec(2, 2, height_ratios=[9, 6], left=.17, right=.985, bottom=.14, top=.935, wspace=.76, hspace=.30)
axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
stage_styles = {"raw": (NAVY, "o", .20, "Raw"), "clipped": (TEAL, "s", 0, "Clipped"), "platt": (SLATE, "^", -.20, "Platt")}
contrast_labels = {"P_H_BASE": "PH − P0", "P_H_TIME": "PTH − PT", "P_WEIGHT": "PTHW − PTH", "I_HISTORY": "M1 − M0", "I_WEIGHT": "M2 − M1"}
interval_count = 0
for ax, dataset, metric, letter in zip(axs, ["paysim", "paysim", "ieee_cis", "ieee_cis"], ["delta_ap", "delta_recall", "delta_ap", "delta_recall"], ["(a)", "(b)", "(c)", "(d)"]):
    clean(ax, "x")
    ax.axvline(0, color=GRAY, linewidth=.7, linestyle=(0, (3, 3)))
    contrasts = ["P_H_BASE", "P_H_TIME", "P_WEIGHT"] if dataset == "paysim" else ["I_HISTORY", "I_WEIGHT"]
    row_keys = [(c, f) for c in contrasts for f in [1, 2, 3]]
    for index, (contrast, fold) in enumerate(row_keys):
        y = len(row_keys)-1-index
        rows = [r for r in D["rows"] if r["dataset"] == dataset and r["metric"] == metric and r["contrast"] == contrast and r["fold"] == fold]
        assert len(rows) == (3 if dataset == "paysim" else 1)
        for r in rows:
            stage = "raw" if dataset == "ieee_cis" else r["score_stage"]
            color, marker, offset, _ = stage_styles[stage]
            if dataset == "ieee_cis":
                offset = 0
            factor = 100 if metric == "delta_recall" else 1
            est, low, high = [r[k]*factor for k in ["estimate", "ci_lower", "ci_upper"]]
            ax.errorbar(est, y+offset, xerr=[[est-low], [high-est]], fmt=marker, markersize=3.5, color=color, ecolor=color, elinewidth=.9, capsize=0, zorder=3)
            interval_count += 1
        if index in [2, 5] and index < len(row_keys)-1:
            ax.axhline(y-.5, color=GRID, linewidth=.7, zorder=0)
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([contrast_labels[c] + "  F" + str(f) for c, f in reversed(row_keys)], fontsize=8)
    ax.set_ylim(-.65, len(row_keys)-.35)
    xlabel = "AP difference" if metric == "delta_ap" else "Recall difference (pp)"
    ax.set_xlabel(xlabel)
    panel(ax, letter, ("PaySim" if dataset == "paysim" else "IEEE-CIS") + (" · AP" if metric == "delta_ap" else " · Recall"))
axs[0].set(xlim=(-.17, .14), xticks=[-.1, 0, .1])
axs[1].set(xlim=(-5, 44), xticks=[0, 20, 40])
axs[2].set(xlim=(-.022, .044), xticks=[-.02, 0, .02, .04])
axs[3].set(xlim=(-5, 5.5), xticks=[-5, 0, 5])
handles = [Line2D([], [], color=c, marker=m, linestyle="-", markersize=3.5, linewidth=.9, label=label) for c, m, _, label in stage_styles.values()]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .012), ncol=3, frameon=False, handletextpad=.4, columnspacing=2)
assert interval_count == 66
for r in D["rows"]:
    if r["dataset"] == "paysim" and r["score_stage"] == "clipped":
        other = next(x for x in D["rows"] if x["dataset"] == r["dataset"] and x["fold"] == r["fold"] and x["contrast"] == r["contrast"] and x["metric"] == r["metric"] and x["score_stage"] == "platt")
        assert all(abs(r[k]-other[k]) < 1e-12 for k in ["estimate", "ci_lower", "ci_upper"])
finish(fig, {"intervals": interval_count, "AP_intervals": 33, "recall_intervals": 33, "pointwise_interval_level": .95, "clipped_and_platt_coincidences_separately_displayed": 18, "percent_to_pp_factor": 100, "independent_axes": True})
