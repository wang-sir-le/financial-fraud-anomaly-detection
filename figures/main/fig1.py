"""Figure 1 with moderately reduced text; original three-panel layout preserved.

Usage: python fig1.py [--out-dir PATH]
Requires Python 3.10+, matplotlib, and the adjacent data/fig1_protocol.json.
Source manuscript hashes are metadata only; redraw opens no manuscript file.
No model training, synthetic outcomes, score distributions, or experiment calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

PROTOCOL_HASH = "3e5863cd422e6a6263d616e9e3b7df6d5f4ce10278023691619f41a8293c8c7a"

W, H = 167.64, 138.0  # millimetres, intended full-width manuscript placement
INK = "#25333D"
MUTED = "#536775"
EDGE = "#B7C9D4"
BLUE = "#E5F0FA"
GREEN = "#E2F0EA"
PINK = "#F8EBEB"
TRAIN = "#A9CBE0"
VAL = "#DCE9F1"
TEST = "#E9C5C3"
FONT = "Arial"


def draw(protocol: dict, dest: Path) -> dict:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "DejaVu Sans"],
            "font.size": 8,
            "svg.fonttype": "none",
            "svg.hashsalt": "fig01-text-reduced-v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.set_axis_off()
    labels = []

    def label(x, y, value, *, size=8, weight="normal", color=INK, align="left", bounds=None):
        obj = ax.text(x, y, value, fontsize=size, fontweight=weight, color=color, va="center", ha=align, linespacing=1.22, zorder=8)
        labels.append((obj, bounds))
        return obj

    def card(x, y, w, h, fill, radius=1.7, line=0):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=fill,
                edgecolor=EDGE if line else "none",
                linewidth=line,
                zorder=1,
            )
        )

    def line(x1, y1, x2, y2, color=EDGE, width=0.7):
        ax.plot([x1, x2], [y1, y2], color=color, lw=width, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6.5, linewidth=0.7, color=MUTED, zorder=5))

    def database(cx, cy):
        ax.add_patch(Rectangle((cx - 3.0, cy - 2.2), 6.0, 4.4, facecolor="white", edgecolor=INK, lw=0.85, zorder=3))
        for dy in (-2.2, 0.0):
            ax.add_patch(Arc((cx, cy + dy), 6, 2, theta1=180, theta2=360, edgecolor=INK, lw=0.85, zorder=4))
        ax.add_patch(Ellipse((cx, cy + 2.2), 6, 2, facecolor="white", edgecolor=INK, lw=0.85, zorder=4))

    def mini_tree(cx, cy):
        for a, b in [
            ((cx, cy + 3), (cx - 4, cy)),
            ((cx, cy + 3), (cx + 4, cy)),
            ((cx - 4, cy), (cx - 6, cy - 3)),
            ((cx - 4, cy), (cx - 2, cy - 3)),
            ((cx + 4, cy), (cx + 2, cy - 3)),
            ((cx + 4, cy), (cx + 6, cy - 3)),
        ]:
            line(*a, *b, color=MUTED, width=0.65)
        for dx, dy in [(0, 3), (-4, 0), (4, 0), (-6, -3), (-2, -3), (2, -3), (6, -3)]:
            ax.add_patch(Circle((cx + dx, cy + dy), 0.9, edgecolor=MUTED, facecolor=TRAIN if dy >= 0 else "white", lw=0.7, zorder=4))

    def review(cx, cy):
        ax.add_patch(Rectangle((cx - 4.5, cy - 4), 8, 9, edgecolor=INK, facecolor="white", lw=0.95, zorder=3))
        for dy in (2.8, 0.6, -1.6):
            line(cx - 3, cy + dy, cx + 1.8, cy + dy, INK, 0.85)
        ax.add_patch(Circle((cx + 3.3, cy - 2.8), 2.4, edgecolor=INK, facecolor="white", lw=1.05, zorder=5))
        line(cx + 4.9, cy - 4.4, cx + 7.2, cy - 6.6, INK, 1.05)

    def split_table(x, rows):
        positions = [x + 14.75, x + 28.75, x + 42.75]
        for cx, value in zip(positions, ("Train", "Val", "Test")):
            label(cx, 106.3, value, size=8, color=MUTED, align="center")
        for row_index, (name, values) in enumerate(rows):
            cy = 101.2 - row_index * 4.6
            label(x + 4.2, cy, name, size=8, weight="bold", align="center")
            for cx, value, fill in zip(positions, values, (TRAIN, VAL, TEST)):
                card(cx - 6.5, cy - 1.8, 13, 3.6, fill, radius=0.5)
                label(cx, cy, value, size=8, align="center", bounds=(cx - 6.5, cy - 1.8, cx + 6.5, cy + 1.8))

    # (a): three parallel study environments, no cross-dataset transfer arrows.
    label(3, 134, "a", size=12, weight="bold")
    label(8, 134, "Separate temporal designs", size=10, weight="bold")
    label(8, 129.2, "Steps: PaySim / BankSim; IEEE-CIS: % of all labelled rows.", size=8, color=MUTED)
    panel_w = (W - 6 - 6) / 3
    xs = [3 + i * (panel_w + 3) for i in range(3)]
    configs = [
        ("PaySim", "Mobile-money simulation", protocol["pay_sim"]["total_rows"], BLUE),
        ("IEEE-CIS", "Anonymised transactions", protocol["ieee_cis"]["total_labelled_rows"], BLUE),
        ("BankSim", "External payment simulation", protocol["bank_sim"]["total_rows"], GREEN),
    ]
    for x, (name, description, count, fill) in zip(xs, configs):
        card(x, 78, panel_w, 48, fill)
        card(x + 1.5, 79.5, panel_w - 3, 45, "#FFFFFF", radius=1.2)
        label(x + 3, 121.2, name, size=10, weight="bold")
        label(x + 3, 117.0, description, size=8, color=MUTED, bounds=(x + 1.5, 114.8, x + panel_w - 1.5, 119))
        database(x + 6.5, 111.5)
        label(x + 12, 111.5, f"{count:,} rows", size=8.5, weight="bold", bounds=(x + 10, 109.4, x + panel_w - 1.5, 114.0))

    def ranges(v):
        return f"{v[0]}–{v[1]}"
    pay_rows = [(f"F{f['fold']}", [ranges(f[k]) for k in ("train", "validation", "test")]) for f in protocol["pay_sim"]["folds"]]
    split_table(xs[0], pay_rows)
    label(xs[0] + 3, 86.2, "LightGBM (5 seeds)", size=8)
    label(xs[0] + 3, 82.0, "Later logistic", size=8, color=MUTED)

    ieee_rows = []
    for f in protocol["ieee_cis"]["folds"]:
        ieee_rows.append((f"F{f['fold']}", [f"{f[k + '_percent_of_full_labelled_rows']:.0f}%" for k in ("train", "validation", "test")]))
    split_table(xs[1], ieee_rows)
    label(xs[1] + 3, 86.2, "Test: 59,054 rows/fold", size=8)
    label(xs[1] + 3, 82.0, "LightGBM; later logistic", size=8, color=MUTED)

    bsplit = protocol["bank_sim"]["split"]
    split_table(xs[2], [("", [ranges(bsplit[k]["steps"]) for k in ("train", "validation", "test")])])
    label(xs[2] + 3, 95.6, "LightGBM / XGBoost (5 seeds)", size=8)
    label(xs[2] + 3, 91.0, "Freeze before fits / Test access", size=8, color=MUTED)
    label(xs[2] + 3, 86.5, "B1 / seed 42: Val selection", size=8, color=MUTED)
    label(xs[2] + 3, 82.0, "Selected settings held fixed", size=8, color=MUTED)

    # (b): the visual pathway is schematic; decision rules remain dataset specific.
    label(3, 72.3, "b", size=12, weight="bold")
    label(8, 72.3, "Study-specific workflows", size=10, weight="bold")
    label(8, 67.6, "PaySim: later exact-3% contrasts.", size=8, color=MUTED)
    cw = (W - 6 - 4 * 3) / 5
    flow_x = [3 + i * (cw + 3) for i in range(5)]
    titles = ("Inputs", "Features", "Models", "Decisions", "Readouts")
    for i, (x, title, fill) in enumerate(zip(flow_x, titles, (BLUE, PINK, BLUE, GREEN, BLUE))):
        card(x, 26, cw, 39, fill)
        label(x + cw / 2, 61.2, title, size=8.5, weight="bold", align="center", bounds=(x + 0.7, 59.2, x + cw - 0.7, 63.2))
        card(x + 1.3, 27.3, cw - 2.6, 30.8, "#FFFFFF", radius=1.0)
        if i < 4:
            arrow(x + cw + 0.15, 46.3, x + cw + 2.85, 46.3)

    x = flow_x[0]
    table_x, table_y, table_w, table_h = x + 2.8, 35.5, cw - 5.6, 20.5
    ax.add_patch(Rectangle((table_x, table_y), table_w, table_h, facecolor="white", edgecolor=EDGE, lw=0.7, zorder=3))
    for j, name in enumerate(("Attributes", "Time group", "Entity key", "Prior events")):
        cy = table_y + table_h - (j + 0.5) * table_h / 4
        label(table_x + 1.3, cy, name, size=8)
        if j:
            line(table_x, table_y + table_h - j * table_h / 4, table_x + table_w, table_y + table_h - j * table_h / 4)
    label(x + cw / 2, 30.9, "Schematic fields", size=8, color=MUTED, align="center")

    x = flow_x[1]
    for j, name in enumerate(("Current context", "Past history", "Entity aggregates")):
        cy = 53.0 - j * 7.1
        card(x + 2.4, cy - 2.6, cw - 4.8, 5.2, "#FCF3F3", radius=0.8, line=0.55)
        label(x + cw / 2, cy, name, size=8, align="center", bounds=(x + 2.4, cy - 2.6, x + cw - 2.4, cy + 2.6))
    label(x + cw / 2, 30.9, "Strictly earlier events", size=8, color=MUTED, align="center")

    x = flow_x[2]
    mini_tree(x + cw / 2, 52.5)
    label(x + cw / 2, 44.2, "Original", size=8, weight="bold", align="center")
    label(x + cw / 2, 39.7, "LightGBM; XGBoost*", size=8, align="center")
    line(x + 3, 36.8, x + cw - 3, 36.8)
    label(x + cw / 2, 34, "Later logistic", size=8, align="center")
    label(x + cw / 2, 30, "*BankSim only", size=8, color=MUTED, align="center")

    x = flow_x[3]
    label(x + cw / 2, 54.9, "PaySim", size=8, weight="bold", align="center")
    label(x + cw / 2, 50.7, "Val alert cap ≤3%", size=8, align="center")
    label(x + cw / 2, 46.5, "Threshold → Test", size=8, align="center")
    label(x + cw / 2, 42.3, "Test workload varies", size=8, color=MUTED, align="center")
    line(x + 3, 39.6, x + cw - 3, 39.6)
    label(x + cw / 2, 36.8, "IEEE-CIS / BankSim", size=8, weight="bold", align="center")
    label(x + cw / 2, 32.8, "Completed Test ranking", size=8, align="center")
    label(x + cw / 2, 28.8, "k = ceil(0.03N)", size=8, align="center")

    x = flow_x[4]
    review(x + cw / 2 - 1, 51.2)
    for cy, name in zip((42.3, 38.2, 34.1, 30.0), ("AP by stage", "Recall / capture", "Test workload", "Analytical cost")):
        label(x + cw / 2, cy, name, size=8, align="center")

    # (c): keep existing-output analysis distinct from later model development.
    label(3, 21.4, "c", size=12, weight="bold")
    label(8, 21.4, "Later analyses and scope", size=10, weight="bold")
    small_w = (W - 6 - 3) / 2
    for x, fill in ((3, "#EEF3F6"), (6 + small_w, "#EDF4F0")):
        card(x, 4.4, small_w, 12.9, fill, radius=1.2)
    label(6, 14.1, "S21 | Existing outputs", size=8.5, weight="bold")
    label(6, 10.2, "Headroom / exchanges / time structure", size=8)
    label(6, 6.5, "No new fits / scores / bootstrap draws", size=8, color=MUTED)
    right_x = 9 + small_w
    label(right_x, 14.1, "S22 | Retrospective development", size=8.5, weight="bold")
    label(right_x, 10.2, "Exposed BankSim; 6 LightGBM C07 fits", size=8)
    label(right_x, 6.5, "Seed 42; no independent validation", size=8, color=MUTED)
    label(W / 2, 1.85, "Icons and split cells are schematic; studies are not pooled.", size=8, color=MUTED, align="center")

    # Physical-size QA: every label is at least 8 pt, unclipped and in its cell.
    fig.canvas.draw()
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
    inverse = ax.transData.inverted()
    problems = []
    text_records = []
    panel_regions: list[tuple[float, float, float, float]] = [(x, 78, x + panel_w, 126) for x in xs]
    panel_regions += [(x, 26, x + cw, 65) for x in flow_x]
    panel_regions += [(3, 4.4, 3 + small_w, 17.3), (6 + small_w, 4.4, 6 + 2 * small_w, 17.3)]
    for obj, bounds in labels:
        bbox = obj.get_window_extent(renderer=renderer).transformed(inverse)
        box = [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]
        text_records.append({"text": obj.get_text(), "font_pt": obj.get_fontsize(), "bbox_mm": box})
        if float(obj.get_fontsize()) < 8:
            problems.append("Font below 8 pt: " + obj.get_text())
        if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > W or bbox.y1 > H:
            problems.append("Outside canvas: " + obj.get_text())
        px, py = obj.get_position()
        if bounds is None:
            for region in panel_regions:
                if region[0] <= px <= region[2] and region[1] <= py <= region[3]:
                    bounds = region
                    break
        if bounds and (bbox.x0 < bounds[0] or bbox.y0 < bounds[1] or bbox.x1 > bounds[2] or bbox.y1 > bounds[3]):
            problems.append("Outside assigned cell: " + obj.get_text())
    if problems:
        raise AssertionError("\n".join(problems))

    dest.mkdir(parents=True, exist_ok=True)
    stem = dest / "fig1"
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None, "Creator": "Reproducible Matplotlib schematic"})
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None, "Creator": "Reproducible Matplotlib schematic"})
    fig.savefig(dest / "fig1.png", dpi=600)
    fig.savefig(dest / "fig1_preview.png", dpi=240)
    plt.close(fig)
    svg = stem.with_suffix(".svg").read_text(encoding="utf-8")
    root_start = svg.index("<svg")
    root_end = svg.index(">", root_start) + 1
    accessibility = (
        "\n <title>Separate study designs, decision pathways and later analysis roles</title>"
        "\n <desc>Protocol facts verified against the current main manuscript and supplement. "
        "PaySim and BankSim show exact step bounds. IEEE-CIS percentages use all 590540 labelled rows "
        "as denominator. Split cells are schematic, not proportional bars. "
        "No empirical curves or outcomes are simulated. Studies are not pooled.</desc>"
    )
    stem.with_suffix(".svg").write_text(svg[:root_end] + accessibility + svg[root_end:], encoding="utf-8")
    report = {
        "canvas_mm": [W, H],
        "minimum_font_pt": min(t["font_pt"] for t in text_records),
        "text_count": len(text_records),
        "all_text_inside_canvas_and_assigned_cells": True,
        "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "text": text_records,
    }
    (dest / "fig1_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {k: v for k, v in report.items() if k != "text"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parent / "data" / "fig1_protocol.json")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    raw_protocol = args.protocol.read_bytes()
    assert hashlib.sha256(raw_protocol).hexdigest() == PROTOCOL_HASH, "Frozen protocol changed"
    protocol = json.loads(raw_protocol.decode("utf-8-sig"))
    assert protocol["pay_sim"]["total_rows"] == 6362620
    assert protocol["ieee_cis"]["total_labelled_rows"] == 590540
    assert protocol["bank_sim"]["split"]["validation"]["steps"] == [108, 125]
    assert protocol["bank_sim"]["split"]["train"]["steps"] == [0, 107]
    for f in protocol["ieee_cis"]["folds"]:
        assert f["test_rows"] == 59054
        assert f["train_percent_of_full_labelled_rows"] == f["train_rows"] / 590540 * 100
    print(json.dumps(draw(protocol, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
