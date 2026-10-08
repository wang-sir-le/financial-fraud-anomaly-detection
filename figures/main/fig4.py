"""Redraw BankSim Fig. 4 from the adjacent frozen aggregate JSON only.

Design components: grouped horizontal effects and a zero reference (gallery
oa-bio-plos-pbio-3002856-f06), point/interval hierarchy and redundant shapes
(oa-tech-pone-0258309-f05), the user's blue/teal palette. No densities,
significance categories or values from those references are imported.
Adapted from the figure-generation template's Agg/rcParams/export pipeline.
No fits, predictions, ranking, AP calculation or interval estimation.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "fig4.json"
COLORS = ["#244F72", "#078A8F"]
MARKERS = ["o", "s"]
FAMILIES = ["LightGBM", "XGBoost"]
INK, MUTED, GRID = "#24323C", "#60707B", "#DCE5EA"


def fmt(value, digits=1, sign=False):
    return (f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}").replace("-", "−")


def row_for(data, effect, family):
    rows = [r for r in data["points"] if r["effect"] == effect and r["family"] == family]
    assert len(rows) == 1
    row = rows[0]
    assert row["confidence"] == (.95 if effect == "H0" else .9875)
    assert row["lower"] <= row["estimate"] <= row["upper"]
    return row


def point(ax, row, y, family_index):
    ax.errorbar(row["estimate"], y,
                xerr=np.array([[row["estimate"] - row["lower"]], [row["upper"] - row["estimate"]]]),
                fmt=MARKERS[family_index], color=COLORS[family_index],
                markersize=5.1, markeredgewidth=.7, capsize=2.7,
                elinewidth=1.35, capthick=.8, zorder=4)


def axes_style(ax):
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#96A8B3")
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=8, length=2.8, width=.6, colors=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=.45)
    ax.axvline(0, color="#7A8A94", lw=.85, ls=(0, (3, 2)), zorder=2)


def header(fig):
    fig.text(.018, .982, "BankSim · whole-window 3% · L = 7", fontsize=9.0, va="top", fontweight="bold")
    handles = [Line2D([], [], color=COLORS[i], marker=MARKERS[i], lw=1.2,
                      markersize=4.7, label=family) for i, family in enumerate(FAMILIES)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.992, 1.003),
               frameon=False, ncol=2, fontsize=8, handlelength=1.5,
               columnspacing=1.4, handletextpad=.5)


def arithmetic_footer(fig, data):
    fig.add_artist(plt.Rectangle((.018, .009), .964, .227,
                                transform=fig.transFigure, facecolor="#F1F5F7", edgecolor="none", zorder=-10))
    fig.text(.030, .216, "Observed-point arithmetic", fontsize=8.2, fontweight="bold", va="top")
    fig.text(.976, .216, "Not confidence limits", fontsize=8, va="top", ha="right", color=MUTED)
    fig.text(.030, .147, "Model", fontsize=8, fontweight="bold", va="center")
    fig.text(.499, .147, "B1 capture space", fontsize=8, fontweight="bold", ha="center", va="center")
    fig.text(.852, .147, "Bound on D", fontsize=8, fontweight="bold", ha="center", va="center")
    fig.add_artist(plt.Line2D([.03, .970], [.126, .126], transform=fig.transFigure, color="#CBD8E0", lw=.55))
    for index, family in enumerate(FAMILIES):
        y = [.093, .040][index]
        fig.text(.030, float(y), family, fontsize=8, va="center", color=COLORS[index])
        fig.text(.499, float(y), fmt(data["arithmetic_footer"]["B1_headroom"][index]),
                 fontsize=8.5, ha="center", va="center", color=COLORS[index])
        fig.text(.852, float(y), "≤ " + fmt(data["arithmetic_footer"]["D_upper_bound"][index]),
                 fontsize=8.5, ha="center", va="center", color=COLORS[index])


def common_scale(data):
    fig = plt.figure(figsize=(6.6, data["height_in"]))
    header(fig)
    fig.text(.018, .895, "a", fontsize=10, fontweight="bold", va="center")
    fig.text(.055, .895, "Effects on a common scale", fontsize=8.8, va="center")
    fig.text(.705, .895, "b", fontsize=10, fontweight="bold", va="center")
    fig.text(.740, .895, "H1 enlarged", fontsize=8.8, va="center")
    ax = fig.add_axes((.170, .442, .445, .398))
    axes_style(ax)
    ax.set_xlim(-280, 280)
    ax.set_ylim(-.12, 6.55)
    ax.set_xticks([-250, -125, 0, 125, 250])
    ax.set_xticklabels(["−250", "−125", "0", "125", "250"])
    ax.set_xlabel("ΔTP (five-seed mean)", fontsize=8, labelpad=4)
    ys = {"H0": [5.83, 5.03], "H1": [3.52, 2.72], "D": [1.21, .41]}
    for effect, meaning, confidence in [("H0", "B0H − B0", "95%"), ("H1", "B1H − B1", "98.75%"), ("D", "H1 − H0", "98.75%")]:
        group_center = float(np.mean(ys[effect]))
        group_y = .442 + .398 * ((group_center + .12) / 6.67)
        fig.text(.020, group_y + .036, effect, fontsize=9, fontweight="bold", va="center")
        fig.text(.020, group_y - .010, meaning, fontsize=8, va="center")
        fig.text(.020, group_y - .054, confidence + " CI", fontsize=8, va="center", color=MUTED)
        for i, family in enumerate(FAMILIES):
            row = row_for(data, effect, family)
            point(ax, row, ys[effect][i], i)
            ax.annotate(fmt(row["estimate"], sign=True), (row["upper"], ys[effect][i]),
                        xytext=(5, 0), textcoords="offset points", va="center", fontsize=8,
                        color=COLORS[i], fontweight="bold", annotation_clip=False)
    for y in [4.31, 2.0]:
        ax.axhline(y, color=GRID, lw=.5)
    ax.axhspan(2.0, 4.31, color="#F2F7F7", zorder=-1)
    # The inset repeats H1 only; its endpoints and confidence level are unchanged.
    zoom = fig.add_axes((.713, .442, .265, .398))
    axes_style(zoom)
    zoom.set_facecolor("#F2F7F7")
    zoom.set_xlim(-14, 23)
    zoom.set_ylim(-.65, 1.65)
    zoom.set_xticks([-10, 0, 10, 20])
    zoom.set_xticklabels(["−10", "0", "10", "20"])
    zoom.set_xlabel("ΔTP · magnified scale", fontsize=8, labelpad=4)
    for i, family in enumerate(FAMILIES):
        row = row_for(data, "H1", family)
        y = 1 - i
        point(zoom, row, y, i)
        label = fmt(row["estimate"], sign=True)
        zoom.text(.5, (y + .65 + .32) / 2.3, label, transform=zoom.transAxes,
                  fontsize=8.5, fontweight="bold", ha="center", va="center", color=COLORS[i])
        zoom.text(.5, (y + .65 - .30) / 2.3,
                  f"[{fmt(row['lower'], 2)}, {fmt(row['upper'], 2)}]", transform=zoom.transAxes,
                  fontsize=8, ha="center", va="center", color=COLORS[i])
    fig.text(.018, .270, "Intervals: H0 95% pointwise; H1/D 98.75% Bonferroni marginal (4 comparisons).",
             fontsize=8, color=MUTED, va="bottom")
    arithmetic_footer(fig, data)
    return fig


def facets(data):
    fig = plt.figure(figsize=(6.6, data["height_in"]))
    header(fig)
    for col, (effect, title, limits, ticks) in enumerate([
            ("H0", "History above B0", (150, 267), [160, 200, 240]),
            ("H1", "History above B1", (-15, 24), [-10, 0, 10, 20]),
            ("D", "Difference: H1 − H0", (-280, 10), [-250, -125, 0])]):
        left = .050 + col * .323
        fig.text(left, .892, f"{'abc'[col]}  {effect}: {title}", fontsize=8.2, fontweight="bold", va="center")
        fig.text(left, .830, "95% pointwise CI" if effect == "H0" else "98.75% adjusted CI", fontsize=8, color=MUTED)
        ax = fig.add_axes((left, .442, .273, .339))
        axes_style(ax)
        ax.set_xlim(limits)
        ax.set_ylim(-.7, 1.6)
        ax.set_xticks(ticks)
        for i, family in enumerate(FAMILIES):
            row = row_for(data, effect, family)
            point(ax, row, 1-i, i)
            ax.text(.5, (1-i + .7 + .34)/2.3, fmt(row["estimate"], sign=True),
                    transform=ax.transAxes, ha="center", fontsize=8.5, fontweight="bold", color=COLORS[i])
        ax.set_xlabel("ΔTP (five-seed mean)", fontsize=8, labelpad=4)
    fig.text(.018, .270, "Horizontal scales differ. Adjusted intervals: Bonferroni marginal, four comparisons.",
             fontsize=8, color=MUTED, va="bottom")
    arithmetic_footer(fig, data)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", action="store_true")
    args = parser.parse_args()
    assert sha256(DATA_PATH.read_bytes()).hexdigest() == "5fa3633f2c70f55bfe4d54f6ba56e827d47dcbd4fa151e6f98a60cf3850ad801"
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    assert data["point_count"] == len(data["points"]) == 6
    assert data["arithmetic_footer"]["not_confidence_limits"]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "text.color": INK, "axes.labelcolor": INK, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    styles = [("draft_a", common_scale), ("draft_b", facets)] if args.drafts else [("fig4", common_scale)]
    for name, builder in styles:
        fig = builder(data)
        fig.canvas.draw()
        for artist in fig.findobj(match=Text):
            if artist.get_visible() and artist.get_text():
                assert artist.get_fontsize() >= 8, artist.get_text()
        extensions = ["png"] if args.drafts else ["png", "pdf", "svg"]
        for extension in extensions:
            fig.savefig(HERE / f"{name}.{extension}", dpi=240 if args.drafts else 450, facecolor="white")
        plt.close(fig)
    if not args.drafts:
        verification = {"source_sha256": sha256(DATA_PATH.read_bytes()).hexdigest(),
                        "unique_effects": data["points"], "h1_repeated_in_magnified_panel": True,
                        "arithmetic": data["arithmetic_footer"], "smallest_font_pt": 8,
                        "new_model_statistical_calls": 0, "visual_review": "PENDING"}
        (HERE / "fig4_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BankSim figure generated from six existing effects; interval definitions retained.")


if __name__ == "__main__":
    main()
