"""IEEE-CIS: unchanged saved point estimates/95% intervals; no raw data/statistics.
Styling adapts figure-generation's Agg/rcParams/export template.
Vertical point/interval and shapes: gallery oa-tech-pone-0258309-f05.
Aligned labels/zero line: oa-bio-plos-pbio-3002856-f06 (no densities copied).
Colors distinguish folds, never significance. No boxplots or connected trend.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
COLORS = ["#244F72", "#078A8F", "#738F9F"]
MARKERS = ["o", "s", "^"]
INK, MUTED, GRID = "#24323C", "#60707B", "#DCE5EA"


def fmt(x, sign=False):
    return (f"{x:+.3f}" if sign else f"{x:.3f}").replace("-", "−")


def base(data):
    fig = plt.figure(figsize=(6.6, data["height_in"]))
    fig.text(0.026, 0.971, "IEEE-CIS  |  C02: M2 − M0", weight="bold", size=10, va="top")
    fig.text(0.98, 0.971, "Whole-window 3% capacity", size=8.4, ha="right", va="top", color=MUTED)
    fig.text(0.026, 0.018, "Points: five-seed paired means · Lines: 95% within-window intervals", size=8, color=MUTED, va="bottom")
    return fig


def vertical_axis(ax):
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#A4B4BD")
    ax.set_ylim(-1.9, 1.9)
    ax.set_yticks([-1.5, -0.75, 0, 0.75, 1.5])
    ax.set_yticklabels(["−1.50", "−0.75", "0", "0.75", "1.50"])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.axhline(0, color="#7A8A94", lw=0.9, ls=(0, (3, 2)))
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8.5, pad=5, colors=MUTED)


def vpoint(ax, x, row, i):
    ax.errorbar(
        x,
        row["estimate"],
        yerr=[[row["estimate"] - row["lower"]], [row["upper"] - row["estimate"]]],
        fmt=MARKERS[i],
        color=COLORS[i],
        markersize=6.4,
        elinewidth=1.45,
        capsize=4,
        capthick=1,
        zorder=4,
    )


def vertical_table(data):
    fig = base(data)
    ax = fig.add_axes((0.105, 0.213, 0.447, 0.62))
    vertical_axis(ax)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks([0, 1, 2], ["Fold 1", "Fold 2", "Fold 3"])
    ax.set_ylabel("Recall difference (pp)", fontsize=9, labelpad=7)
    for i, row in enumerate(data["points"]):
        vpoint(ax, i, row, i)
    fig.add_artist(plt.Line2D([0.587, 0.587], [0.205, 0.837], transform=fig.transFigure, color=GRID, lw=0.7))
    fig.text(0.633, 0.815, "Window", size=8, weight="bold")
    fig.text(0.976, 0.815, "Estimate [95% CI], pp", size=8, weight="bold", ha="right")
    for i, (row, y) in enumerate(zip(data["points"], [0.665, 0.455, 0.245])):
        fig.text(0.633, y, row["label"], size=8.5, va="center", color=COLORS[i])
        fig.text(0.974, y + 0.038, fmt(row["estimate"], True), size=10, weight="bold", ha="right", va="center", color=COLORS[i])
        fig.text(0.974, y - 0.046, f"[{fmt(row['lower'])}, {fmt(row['upper'])}]", size=8.3, ha="right", va="center", color=MUTED)
    return fig


def three_panels(data):
    fig = base(data)
    for i, row in enumerate(data["points"]):
        ax = fig.add_axes((0.105 + i * 0.296, 0.270, 0.245, 0.49))
        vertical_axis(ax)
        ax.set_xlim(-1, 1)
        ax.set_xticks([])
        if i == 0:
            ax.set_ylabel("Recall difference (pp)", fontsize=9)
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
        vpoint(ax, 0, row, i)
        fig.text(0.2275 + i * 0.296, 0.818, row["label"], size=9, weight="bold", ha="center")
        fig.text(0.2275 + i * 0.296, 0.174, fmt(row["estimate"], True), size=10, weight="bold", color=COLORS[i], ha="center")
        fig.text(0.2275 + i * 0.296, 0.099, f"[{fmt(row['lower'])}, {fmt(row['upper'])}]", size=8.5, color=MUTED, ha="center")
    return fig


def horizontal_table(data):
    fig = base(data)
    ax = fig.add_axes((0.15, 0.265, 0.47, 0.51))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#A4B4BD")
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.set_xticks([-1.5, -0.75, 0, 0.75, 1.5])
    ax.tick_params(labelsize=8, length=3, colors=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.axvline(0, color=MUTED, lw=0.9, ls=(0, (3, 2)))
    ax.set_xlabel("Recall difference (pp)", size=9)
    fig.text(0.974, 0.827, "Estimate [95% CI], pp", ha="right", weight="bold", size=8.5)
    for i, row in enumerate(data["points"]):
        y = 2 - i
        fy = 0.265 + 0.51 * (y + 0.5) / 3
        ax.errorbar(
            row["estimate"],
            y,
            xerr=[[row["estimate"] - row["lower"]], [row["upper"] - row["estimate"]]],
            fmt=MARKERS[i],
            color=COLORS[i],
            capsize=3,
            markersize=6,
            elinewidth=1.4,
        )
        fig.text(0.026, fy, row["label"], size=9, va="center")
        fig.text(0.974, fy + 0.028, fmt(row["estimate"], True), size=10, weight="bold", color=COLORS[i], ha="right", va="center")
        fig.text(0.974, fy - 0.042, f"[{fmt(row['lower'])}, {fmt(row['upper'])}]", size=8.3, color=MUTED, ha="right", va="center")
    return fig


def diverging_bars(data):
    fig = plt.figure(figsize=(6.6, data["height_in"]))
    fig.text(0.026, 0.971, "IEEE-CIS  |  C02: M2 − M0", weight="bold", size=10, va="top")
    fig.text(0.98, 0.971, "Whole-window 3% capacity", size=8.4, ha="right", va="top", color=MUTED)
    fig.text(0.026, 0.806, "Window", size=8.5, weight="bold")
    fig.text(0.405, 0.806, "Recall difference (pp)", size=8.5, weight="bold", ha="center")
    fig.text(0.733, 0.806, "Estimate", size=8.5, weight="bold", ha="center")
    fig.text(0.903, 0.806, "95% CI", size=8.5, weight="bold", ha="center")
    ax = fig.add_axes((0.153, 0.275, 0.476, 0.46))
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["−1.0", "−0.5", "0", "0.5", "1.0"])
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#A4B4BD")
    ax.tick_params(length=3, labelsize=8, colors=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#697E8A", lw=0.95, zorder=3)
    for i, row in enumerate(data["points"]):
        y = 2 - i
        fy = 0.275 + 0.46 * (y + 0.5) / 3
        ax.barh(y, row["estimate"], height=0.36, color=COLORS[i], edgecolor="none", zorder=4)
        fig.text(0.026, fy, row["label"], size=9, va="center")
        fig.text(0.733, fy, fmt(row["estimate"], True), size=10, weight="bold", color=COLORS[i], ha="center", va="center")
        fig.text(0.903, fy, f"[{fmt(row['lower'])}, {fmt(row['upper'])}]", size=8, ha="center", va="center", color=MUTED)
    fig.text(0.153, 0.143, "Lower recall for M2", size=8, color=MUTED, ha="left")
    fig.text(0.629, 0.143, "Higher recall for M2", size=8, color=MUTED, ha="right")
    fig.add_artist(plt.Line2D([0.664, 0.664], [0.265, 0.846], transform=fig.transFigure, color=GRID, lw=0.7))
    fig.text(
        0.026, 0.018, "Bars: five-seed paired means · 95% within-window intervals reported numerically", size=8, color=MUTED, va="bottom"
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", action="store_true")
    parser.add_argument("--style", choices=["A", "B", "C", "D"], default="D")
    args = parser.parse_args()
    assert sha256((HERE / "fig5.json").read_bytes()).hexdigest() == "4e9577138777b161bc18533ebb715a2f0bd9c26b605718da884cbd69fc26b115"
    data = json.loads((HERE / "fig5.json").read_text(encoding="utf-8"))
    assert len(data["points"]) == data["point_count"] == 3
    for i, row in enumerate(data["points"]):
        assert row["label"] == f"Fold {i + 1}" and row["confidence"] == 0.95
        assert row["lower"] <= row["estimate"] <= row["upper"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    builders = {"A": vertical_table, "B": three_panels, "C": horizontal_table, "D": diverging_bars}
    for key in builders if args.drafts else [args.style]:
        fig = builders[key](data)
        fig.canvas.draw()
        assert all(t.get_fontsize() >= 8 for t in fig.findobj(match=Text) if t.get_visible() and t.get_text())
        for ext in ["png"] if args.drafts else ["png", "pdf", "svg"]:
            fig.savefig(HERE / f"{'draft_' + key if args.drafts else 'fig5'}.{ext}", dpi=240 if args.drafts else 450, facecolor="white")
        plt.close(fig)
    if args.drafts:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 25)
        names = ["before", "draft_A", "draft_B", "draft_C"]
        labels = [
            "现图｜保留作对照",
            "A｜我的推荐：纵向点区间＋数值栏",
            "B｜备选：三个窗口分面，共同纵轴",
            "C｜贴近既有风格：横向点区间＋数值栏",
        ]
        imgs = [Image.open(HERE / f"{n}.png").convert("RGB") for n in names]
        width = 1300
        height = round(width * data["height_in"] / 6.6)
        sheet = Image.new("RGB", (width, (height + 48) * 4), "#e9eff3")
        draw = ImageDraw.Draw(sheet)
        for i, (im, label) in enumerate(zip(imgs, labels)):
            draw.text((16, i * (height + 48) + 8), label, font=font, fill=INK)
            sheet.paste(im.resize((width, height), Image.Resampling.LANCZOS), (0, i * (height + 48) + 48))
        sheet.save(HERE / "style_comparison.png")
    else:
        (HERE / "fig5_verification.json").write_text(
            json.dumps(
                {
                    "source_sha256": sha256((HERE / "fig5.json").read_bytes()).hexdigest(),
                    "original_points": data["points"],
                    "chosen_style": args.style,
                    "minimum_font_pt": 8,
                    "new_statistical_model_calls": 0,
                    "visual_review": "PENDING",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print("Three frozen effects retained; no model or statistical computation.")


if __name__ == "__main__":
    main()
