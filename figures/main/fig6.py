"""Reproduce Figure 7 from the adjacent, full-precision Fig6_data.csv.

Usage: python fig6.py
Requires Python >= 3.10 and matplotlib >= 3.8.
"""

import csv
import json
from pathlib import Path
from typing import Any, Iterable, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parent
COLORS = {1: "#AD6345", 2: "#8A7195", 3: "#667D56"}
MARKERS = {1: "o", 2: "s", 3: "^"}
FONT = "Times New Roman" if any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist) else "DejaVu Serif"
plt.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": "#292C2D",
        "axes.labelcolor": "#292C2D",
        "xtick.color": "#55595A",
        "ytick.color": "#55595A",
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)

with (ROOT / "data/fig6_scenarios.csv").open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 63
assert all(int(row["seed_count"]) == 5 for row in rows)
assert all(row["uncertainty"].startswith("not supplied") for row in rows)

fig = plt.figure(figsize=(170 / 25.4, 170 / 25.4), facecolor="white")
positions = {
    "a": [0.090, 0.640, 0.365, 0.250],
    "b": [0.595, 0.640, 0.365, 0.250],
    "c": [0.090, 0.285, 0.365, 0.210],
    "d": [0.595, 0.285, 0.365, 0.210],
}
axes = {key: fig.add_axes((pos[0], pos[1], pos[2], pos[3])) for key, pos in positions.items()}
headers = {
    "a": ("IEEE-CIS · review budget", "M2 − M0  |  c = 100"),
    "b": ("PaySim · threshold transfer", "PTHW − P0  |  c = 100"),
    "c": ("PaySim · cost sensitivity", "PTHW − P0  |  Validation cap = 3%"),
    "d": ("PaySim · Test workload", "c = 100  |  solid PTHW; dashed P0"),
}

for key, ax in axes.items():
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#B9BCBC")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3, pad=3)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7E8E6", linewidth=0.5)
    ax.axhline(0, color="#666B6C", linewidth=0.9, zorder=2)
    x, y, _, h = positions[key]
    title, subtitle = headers[key]
    fig.add_artist(
        Rectangle(
            (x - 0.029, y + h + 0.035),
            0.405,
            0.040,
            transform=fig.transFigure,
            facecolor="#F0F2F1",
            edgecolor="none",
            zorder=0,
        )
    )
    fig.text(x - 0.021, y + h + 0.047, f"({key})", size=10, weight="bold")
    fig.text(x + 0.024, y + h + 0.047, title, size=9.5, weight="bold")
    fig.text(x + 0.024, y + h + 0.011, subtitle, size=8, color="#5D6263")
    reference_x = 100 if key == "c" else 3
    ax.axvline(reference_x, color="#A5ABA9", linestyle=(0, (2, 3)), linewidth=0.8)
    if key != "c":
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks([1, 3, 5, 10])
        ax.set_xlabel(
            "Exact Test budget (%)" if key == "a" else "Validation alert cap (%)",
            labelpad=5,
        )
    else:
        ax.set_xscale("log")
        ax.set_xlim(7, 1400)
        ax.xaxis.set_major_locator(FixedLocator([10, 50, 100, 500, 1000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("False-negative cost c (log scale)", labelpad=5)

plotted = 0
for key, ax in axes.items():
    models = ["PTHW", "P0"] if key == "d" else [None]
    for model in models:
        for fold in [1, 2, 3]:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["panel"] == key and int(row["fold"]) == fold and (model is None or row["comparison_or_model"] == model)
                ],
                key=lambda row: float(row["x"]),
            )
            assert len(selected) == (5 if key == "c" else 4)
            xs = [float(row["x"]) for row in selected]
            ys = [float(row["value"]) for row in selected]
            baseline = model == "P0"
            line = ax.plot(
                xs,
                ys,
                color=COLORS[fold],
                marker=MARKERS[fold],
                markersize=4.2,
                markeredgewidth=0.8,
                markeredgecolor=COLORS[fold],
                markerfacecolor="white" if baseline else COLORS[fold],
                linewidth=1.1,
                linestyle=(0, (4, 2.5)) if baseline else "-",
                zorder=4 if baseline else 3,
            )[0]
            assert list(cast(Iterable[float], line.get_xdata())) == xs
            assert list(cast(Iterable[float], line.get_ydata())) == ys
            plotted += len(xs)

axes["a"].set(ylim=(-1.5, 6.5), yticks=[-1, 0, 2, 4, 6])
axes["b"].set(ylim=(0, 65), yticks=[0, 20, 40, 60])
axes["c"].set(ylim=(-10, 70), yticks=[0, 20, 40, 60])
axes["d"].set(ylim=(-0.15, 4.1), yticks=[0, 1, 2, 3, 4])
axes["a"].set_ylabel("Recall difference (pp)", labelpad=6)
axes["b"].set_ylabel("Recall difference (pp)", labelpad=6)
axes["c"].set_ylabel("Cost reduction (%)", labelpad=6)
axes["d"].set_ylabel("Test alert rate (%)", labelpad=6)

legend = [
    Line2D(
        [],
        [],
        color=COLORS[fold],
        marker=MARKERS[fold],
        markersize=4.5,
        linewidth=1.1,
        label=f"Fold {fold}",
    )
    for fold in [1, 2, 3]
]
fig.legend(
    handles=legend,
    loc="upper left",
    bbox_to_anchor=(0.066, 1.002),
    ncol=3,
    frameon=False,
    handlelength=1.8,
    columnspacing=2.0,
    fontsize=9,
)
fig.text(0.96, 0.971, "Five-seed means", ha="right", size=8.5, color="#5D6263")
fig.text(0.065, 0.192, "Reference scenarios", size=9.5, weight="bold")
fig.text(0.96, 0.192, "Capacity = 3%  |  c = 100  |  vertical guides", ha="right", size=8)
table_ax = fig.add_axes((0.065, 0.062, 0.90, 0.112))
table_ax.axis("off")


def reference_value(panel, fold, model=None):
    result = [
        row
        for row in rows
        if row["panel"] == panel
        and int(row["fold"]) == fold
        and float(row["capacity"]) == 0.03
        and float(row["fn_cost"]) == 100
        and (model is None or row["comparison_or_model"] == model)
    ]
    assert len(result) == 1
    return float(result[0]["value"])


table_rows = []
for fold in [1, 2, 3]:
    table_rows.append(
        [
            f"Fold {fold}",
            f"{reference_value('a', fold):+.3f}",
            f"{reference_value('b', fold):+.3f}",
            f"{reference_value('c', fold):+.3f}",
            f"{reference_value('d', fold, 'PTHW'):.3f} / {reference_value('d', fold, 'P0'):.3f}",
        ]
    )
table = table_ax.table(
    cellText=table_rows,
    colLabels=[
        "Window",
        "(a) ΔRecall\n(pp)",
        "(b) ΔRecall\n(pp)",
        "(c) Cost reduction\n(%)",
        "(d) Alerts (%)\nPTHW / P0",
    ],
    colWidths=[0.12, 0.20, 0.20, 0.20, 0.28],
    cellLoc="center",
    # Matplotlib accepts [left, bottom, width, height]; its stub is narrower.
    bbox=cast(Any, [0, 0, 1, 1]),
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
for (row_index, column), cell in table.get_celld().items():
    cell.set_height(0.37 if row_index == 0 else 0.21)
    cell.set_edgecolor("white")
    cell.set_linewidth(1.4)
    cell.set_facecolor("#F0F2F1" if row_index == 0 else ("#F7F8F7" if row_index % 2 else "white"))
    if row_index == 0:
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_fontsize(8)
    elif column == 0:
        cell.get_text().set_color(COLORS[row_index])
        cell.get_text().set_fontweight("bold")
fig.text(
    0.065,
    0.036,
    "Points: evaluated scenarios; lines: guides only. No uncertainty intervals supplied.",
    size=8,
    color="#55595A",
)
fig.text(
    0.065,
    0.015,
    "PaySim Validation caps do not impose exact Test workloads. Recall panels use different scales.",
    size=8,
    color="#55595A",
)
assert plotted == 63
fig.canvas.draw()
for text in fig.findobj(matplotlib.text.Text):
    if text.get_visible() and text.get_text():
        assert float(text.get_fontsize()) >= 8
for extension in ["pdf", "svg", "png"]:
    fig.savefig(ROOT / f"fig6.{extension}", dpi=600)
fig.savefig(ROOT / "fig6_preview.png", dpi=144)
(ROOT / "fig6_validation.json").write_text(
    json.dumps(
        {
            "plotted_points": plotted,
            "all_coordinates_match_csv": True,
            "figure_width_mm": 170,
            "figure_height_mm": 170,
            "minimum_font_pt": 8,
            "font": FONT,
            "png_dpi": 600,
            "intervals_drawn": 0,
        },
        indent=2,
    ),
    encoding="utf-8",
)
plt.close(fig)
print("Exported fig6.pdf / .svg / .png; 63 verified coordinates.")
