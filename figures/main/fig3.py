"""Reproduce the corrected Fig. 3 from the adjacent frozen aggregate CSVs.

Usage: python fig3.py --output-dir output
No model fitting, score averaging, resampling, curve smoothing or interpolation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
COLORS = {1: "#AD6345", 2: "#8A7195", 3: "#667D56"}
MARKERS = {1: "o", 2: "s", 3: "^"}
INK = "#27343E"
MUTED = "#60717B"
GRAY = "#7B878E"
STEPS = {1: "239–281", 2: "324–354", 3: "399–743"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def signed(value: float, decimals: int) -> str:
    return f"{value:+.{decimals}f}".replace("-", "−")


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#B3BDC2")
        ax.spines[name].set_linewidth(0.55)
    ax.tick_params(length=3, width=0.55, color="#89969D", pad=4)
    ax.set_axisbelow(True)


def draw_effects(ax: plt.Axes, rows: list[dict[str, str]], metric: str) -> None:
    selected = [row for row in rows if row["metric"] == metric]
    order = ["1", "2", "3", "equal_fold"]
    assert len(selected) == 4
    by_fold = {row["fold"]: row for row in selected}
    assert set(by_fold) == set(order)
    for y, key in zip([3, 2, 1, 0], order, strict=True):
        row = by_fold[key]
        mean = float(row["estimate"])
        lo, hi = float(row["ci_lower"]), float(row["ci_upper"])
        assert lo <= mean <= hi
        color = GRAY if key == "equal_fold" else COLORS[int(key)]
        marker = "D" if key == "equal_fold" else MARKERS[int(key)]
        ax.axhline(y, color="#EEF1F2", lw=0.55, zorder=0)
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - lo], [hi - mean]],
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=5.2,
            elinewidth=1.05,
            capsize=2.8,
            capthick=0.85,
            zorder=4,
        )
        ax.annotate(
            signed(mean, 3 if metric == "recall" else 4),
            (mean, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="bold",
            color=color,
            zorder=5,
            bbox={"facecolor": "#F3F5F6" if key == "equal_fold" else "white", "edgecolor": "none", "boxstyle": "square,pad=0.13"},
        )
    ax.axhspan(-0.48, 0.48, color="#F3F5F6", zorder=-2)
    ax.axhline(0.51, color="#D3DADF", lw=0.6, zorder=0)
    ax.axvline(0, color="#83919A", lw=0.85, zorder=2)
    ax.set_ylim(-0.58, 3.65)
    ax.set_yticks([3, 2, 1, 0])
    if metric == "recall":
        ax.set_yticklabels([f"Fold {fold}\n{STEPS[fold]}" for fold in [1, 2, 3]] + ["Equal-fold\nsecondary"])
        ax.set_xlim(-5, 76)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_xlabel("Recall difference (pp)", labelpad=6)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(-0.16, 0.40)
        ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-12 else f"{value:.1f}".replace("-", "−")))
        ax.set_xlabel("AP difference (0–1 scale)", labelpad=6)


def make_figure(output_dir: Path) -> dict:
    effects_path = ROOT / "data" / "primary_effects.csv"
    curves_path = ROOT / "data" / "capacity_mean_points.csv"
    effects = read_csv(effects_path)
    curves = read_csv(curves_path)
    assert len(effects) == 8
    assert len(curves) == 3 * 2 * 501
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = "Arial" if "Arial" in available_fonts else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.unicode_minus": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )
    fig = plt.figure(figsize=(167.64 / 25.4, 180 / 25.4), facecolor="white")
    fig.text(0.045, 0.975, "PaySim | combined-scheme effects and capture capacity", fontsize=11.1, weight="bold", va="top")
    fig.text(0.045, 0.944, "LightGBM · PTHW versus P0 · five-seed paired means", fontsize=8.5, color=MUTED, va="top")
    fig.text(0.045, 0.921, "PTHW = current-time context + history + class weighting", fontsize=8, color=MUTED, va="top")

    ax_a = fig.add_axes((0.17, 0.585, 0.355, 0.265))
    ax_b = fig.add_axes((0.635, 0.585, 0.320, 0.265))
    for ax in [ax_a, ax_b]:
        style_axes(ax)
    draw_effects(ax_a, effects, "recall")
    draw_effects(ax_b, effects, "ap")
    fig.text(0.045, 0.887, "a", fontsize=12, weight="bold")
    fig.text(0.17, 0.887, "Recall effect", fontsize=10, weight="bold")
    fig.text(0.17, 0.866, "Validation threshold → Test", fontsize=8, color=MUTED)
    fig.text(0.565, 0.887, "b", fontsize=12, weight="bold")
    fig.text(0.635, 0.887, "Average precision effect", fontsize=10, weight="bold")
    fig.text(0.635, 0.866, "Platt score stage", fontsize=8, color=MUTED)
    fig.text(0.17, 0.516, "Whiskers: pointwise 95% paired moving-block intervals.", fontsize=8, color=MUTED)

    fig.text(0.045, 0.462, "c", fontsize=12, weight="bold")
    fig.text(0.17, 0.462, "Capture at fixed Test capacity", fontsize=10, weight="bold")
    fig.text(0.17, 0.441, "Completed-window ranking · Platt scores · K = ceil(qN)", fontsize=8, color=MUTED)
    ax_c = fig.add_axes((0.17, 0.151, 0.785, 0.265))
    style_axes(ax_c)
    ax_c.set_xlim(0, 5)
    ax_c.set_ylim(0, 100)
    ax_c.set_xticks([0, 1, 2, 3, 4, 5])
    ax_c.set_yticks([0, 20, 40, 60, 80, 100])
    ax_c.set_xlabel("Exact Test review capacity (%)", labelpad=6)
    ax_c.set_ylabel("Recall (%)", labelpad=7)
    ax_c.grid(axis="y", color="#E8ECEE", lw=0.55)
    ax_c.axvline(3, color="#B0B9BE", lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax_c.text(3.055, 96, "3%", fontsize=8, color=MUTED, va="top")
    plot_audit = []
    for fold in [1, 2, 3]:
        for scheme in ["P0", "PTHW"]:
            selected = sorted(
                [r for r in curves if int(r["fold"]) == fold and r["scheme"] == scheme],
                key=lambda r: float(r["capacity_pct"]),
            )
            assert len(selected) == 501
            x = np.array([float(r["capacity_pct"]) for r in selected])
            y = np.array([100 * float(r["mean_recall"]) for r in selected])
            assert np.allclose(x, np.arange(501) / 100, rtol=0, atol=1e-12)
            assert np.all(np.diff(y) >= -1e-10)
            assert y[0] == 0 and np.all((y >= 0) & (y <= 100))
            assert all(int(r["seed_count"]) == 5 for r in selected)
            ax_c.step(
                x,
                y,
                where="post",
                color=COLORS[fold],
                lw=1.25,
                ls="-" if scheme == "PTHW" else (0, (4, 2.5)),
                alpha=1.0 if scheme == "PTHW" else 0.87,
                zorder=4 if scheme == "PTHW" else 3,
            )
            if scheme == "PTHW":
                ax_c.plot(
                    [x[300]],
                    [y[300]],
                    marker=MARKERS[fold],
                    color=COLORS[fold],
                    markersize=4.5,
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    zorder=5,
                )
            plot_audit.append(
                {
                    "fold": fold,
                    "scheme": scheme,
                    "points": len(x),
                    "recall_at_3_pct": float(y[300]),
                    "recall_at_5_pct": float(y[500]),
                    "monotonic": True,
                }
            )
    handles = [Line2D([], [], color=COLORS[fold], marker=MARKERS[fold], lw=0, markersize=4.5, label=f"Fold {fold}") for fold in [1, 2, 3]]
    handles += [Line2D([], [], color=INK, lw=1.25, label="PTHW"), Line2D([], [], color=INK, lw=1.25, ls=(0, (4, 2.5)), label="P0")]
    legend = ax_c.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.96,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.6,
        borderpad=0.45,
        labelspacing=0.45,
    )
    legend.set_zorder(8)
    fig.text(0.17, 0.079, "Panels a–b: paired effects. Panel c: a separate exact-capacity readout.", fontsize=8, color=MUTED)
    fig.text(0.17, 0.056, "Curve points are calculated from saved predictions; no smoothing or curve intervals.", fontsize=8, color=MUTED)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig3"
    fig.canvas.draw()
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
    bounds = fig.bbox
    clipped = []
    for label in fig.findobj(matplotlib.text.Text):
        if not label.get_visible() or not label.get_text():
            continue
        box = label.get_window_extent(renderer)
        if (
            box.width > 0
            and box.height > 0
            and (box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1)
        ):
            clipped.append(label.get_text())
    assert not clipped, f"Text extends beyond the figure: {clipped}"
    fig.savefig(
        stem.with_suffix(".pdf"),
        metadata={"Title": "PaySim combined-scheme effects and capture capacity", "Author": "", "Creator": "Matplotlib"},
    )
    fig.savefig(
        stem.with_suffix(".svg"), metadata={"Title": "PaySim combined-scheme effects and capture capacity", "Creator": "Matplotlib"}
    )
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(output_dir / "fig3_preview.png", dpi=180)
    plt.close(fig)
    audit = {
        "status": "PASS",
        "size_mm": [167.64, 180],
        "font": font,
        "minimum_text_pt": 8,
        "effect_records": len(effects),
        "curve_records": len(curves),
        "curve_groups": plot_audit,
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in [effects_path, curves_path]},
        "output_hashes": {path.name: sha256(path) for path in output_dir.glob("fig3*")},
        "text_clipped": clipped,
        "independent_effect_axes": True,
        "curve_intervals": "not computed; not shown",
        "smoothing": False,
    }
    (output_dir / "fig3_validation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    audit = make_figure(args.output_dir)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output_dir": str(args.output_dir),
                "effect_records": audit["effect_records"],
                "curve_records": audit["curve_records"],
            }
        )
    )


if __name__ == "__main__":
    main()
