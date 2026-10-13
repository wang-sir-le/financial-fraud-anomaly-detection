"""Recreate a six-panel IEEE-CIS figure using reported manuscript results.

Run this script next to its data directory. Matplotlib and Pillow are required.
No predictions, fits, bootstrap draws, or confidence intervals are recomputed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT
QA = ROOT / "fig5_qa"
NAME = "fig5"
INK = "#26323D"
MUTED = "#667480"
BORDER = "#ABB5BE"
BAND = "#F1F3F5"
NEGATIVE = "#FA7A6F"
POSITIVE = "#5DC050"
NEGATIVE_INK = "#B7443A"
POSITIVE_INK = "#2F7E2B"
WHITE = "#FFFFFF"

CAPTION = (
    "IEEE-CIS combined and component effects across three temporal Test windows at "
    "the exact 3% completed-window review budget using raw scores. Columns correspond "
    "to Folds 1–3. (a–c) LightGBM combined effects, M2−M0: captured-fraud difference "
    "(ΔTP), recall difference (percentage points), and average-precision difference "
    "(0–1 scale), with their original pointwise 95% intervals from Table 9. The "
    "vertical point–interval plots display ΔTP only and share the same −40 to +40 "
    "range. (d–f) History increments (H, M1−M0) and incremental weighting above history "
    "(W, M2−M1) from Table 10. Numerical columns and heatmaps show ΔTP point estimates "
    "for LightGBM, Logistic and directly paired Logistic-minus-LightGBM effects. "
    "All three heatmaps share a symmetric zero-centred ±120 ΔTP scale. The two "
    "intervals printed beneath each component table belong to the direct workflow "
    "effect differences; the full within-workflow intervals remain in Table 10 "
    "and the data export. LightGBM averages five paired seed effects; Logistic uses "
    "one fixed fit per condition. Intervals are the original pointwise conditional "
    "95% percentile intervals from 5,000 paired moving-block draws per window, "
    "without multiple-comparison adjustment, and condition on fixed workflows and "
    "observed windows. Colours encode signed point estimates, not significance. "
    "Different preprocessing prevents isolation of an algorithm-family effect. "
    "Sequential component point estimates sum to the combined point estimate; "
    "interval endpoints are not added or subtracted."
)


@dataclass(frozen=True)
class Effect:
    contrast: str
    fold: int
    workflow: str
    metric: str
    estimate: float
    lower: float
    upper: float
    digits: int


def load_data(path: Path) -> list[Effect]:
    records: list[Effect] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            metric = row.get("metric", "ΔTP")
            digits = {"ΔTP": 1, "ΔRecall": 3, "ΔAP": 4}[metric]
            record = Effect(
                row["contrast"],
                int(row["fold"]),
                row["workflow"],
                metric,
                float(row["estimate"]),
                float(row["ci_lower"]),
                float(row["ci_upper"]),
                digits,
            )
            if not all(math.isfinite(v) for v in (record.estimate, record.lower, record.upper)):
                raise ValueError("Nonfinite source value")
            if not record.lower <= record.estimate <= record.upper:
                raise ValueError("Unordered source interval")
            records.append(record)
    return records


def number(value: float, digits: int = 1, *, sign: bool = False) -> str:
    pattern = f"+.{digits}f" if sign else f".{digits}f"
    return format(value, pattern).replace("-", "−")


def effect_color(value: float) -> str:
    if value < 0:
        return NEGATIVE_INK
    if value > 0:
        return POSITIVE_INK
    return MUTED


def ci(record: Effect) -> str:
    return f"[{number(record.lower, record.digits)}, {number(record.upper, record.digits)}]"


def text(
    fig: Figure,
    x: float,
    y: float,
    label: str,
    *,
    size: float = 8,
    weight: str = "normal",
    color: str = INK,
    align: str = "left",
) -> None:
    fig.text(x, y, label, fontsize=size, fontweight=weight, color=color, ha=align, va="center")


def box(fig: Figure, x: float, y: float, w: float, h: float, color: Any) -> None:
    fig.add_artist(
        Rectangle(
            (x, y), w, h, transform=fig.transFigure, facecolor=color, edgecolor="none", zorder=-1
        )
    )


def separator(fig: Figure, x: float, y: float, w: float, h: float) -> None:
    box(fig, x, y, w, h, "#E4E8EC")


def panel(fig: Figure, x: float, y: float, letter: str, label: str) -> None:
    width, height = 0.300, 0.364
    fig.add_artist(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.002,rounding_size=0.005",
            transform=fig.transFigure,
            facecolor=WHITE,
            edgecolor=BORDER,
            linewidth=0.6,
            zorder=-2,
        )
    )
    box(fig, x + 0.001, y + height - 0.041, width - 0.002, 0.039, BAND)
    text(fig, x + 0.011, y + height - 0.022, letter, size=9.6, weight="bold")
    text(fig, x + 0.034, y + height - 0.022, label, size=8.7, weight="bold")


def validate_data(totals: list[Effect], components: list[Effect]) -> None:
    if len(totals) != 9 or len(components) != 18:
        raise ValueError("Expected 9 combined metrics and 18 component effects")
    lookup = {(r.contrast, r.fold, r.workflow): r for r in components}
    totals_lookup = {(r.fold, r.metric): r for r in totals}
    for fold in (1, 2, 3):
        history = lookup[("M1-M0", fold, "LightGBM")]
        weighting = lookup[("M2-M1", fold, "LightGBM")]
        total = totals_lookup[(fold, "ΔTP")]
        if not math.isclose(history.estimate + weighting.estimate, total.estimate):
            raise ValueError("Incorrect sequential point-estimate identity")
        for contrast in ("M1-M0", "M2-M1"):
            lgb = lookup[(contrast, fold, "LightGBM")]
            logistic = lookup[(contrast, fold, "Logistic")]
            direct = lookup[(contrast, fold, "Logistic minus LightGBM")]
            if not math.isclose(logistic.estimate - lgb.estimate, direct.estimate):
                raise ValueError("Incorrect direct-difference direction")


def draw(totals: list[Effect], components: list[Effect]) -> Figure:
    names = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "Arial" if "Arial" in names else "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
            "savefig.facecolor": WHITE,
        }
    )
    fig = plt.figure(figsize=(7.2, 4.7), facecolor=WHITE)
    text(
        fig,
        0.026,
        0.964,
        "IEEE-CIS | Combined, component and workflow effects",
        size=10.6,
        weight="bold",
    )
    text(
        fig,
        0.026,
        0.921,
        "Exact 3% completed-Test-window budget · raw scores · fixed workflows",
        size=8.3,
        color=MUTED,
    )
    totals_lookup = {(r.fold, r.metric): r for r in totals}
    lookup = {(r.contrast, r.fold, r.workflow): r for r in components}
    cmap = LinearSegmentedColormap.from_list("signed_delta_tp", (NEGATIVE, "#FAF7F6", POSITIVE))
    norm = TwoSlopeNorm(vmin=-120, vcenter=0, vmax=120)
    column_x = (0.026, 0.349, 0.672)
    top_y, bottom_y = 0.519, 0.134
    for fold, x in enumerate(column_x, start=1):
        panel(fig, x, top_y, chr(96 + fold), f"Fold {fold} | Combined effect")
        panel(fig, x, bottom_y, chr(99 + fold), f"Fold {fold} | Component effects")
        text(fig, x + 0.011, top_y + 0.300, "LightGBM · M2 − M0", size=8, color=MUTED)
        metric_y = (top_y + 0.240, top_y + 0.164, top_y + 0.088)
        labels = ("ΔTP", "ΔRecall (pp)", "ΔAP (0–1)")
        for metric, label, y in zip(("ΔTP", "ΔRecall", "ΔAP"), labels, metric_y, strict=True):
            record = totals_lookup[(fold, metric)]
            text(fig, x + 0.012, y, label, size=8)
            text(
                fig,
                x + 0.188,
                y,
                number(record.estimate, record.digits, sign=True),
                size=8.4,
                weight="bold",
                align="right",
                color=effect_color(record.estimate),
            )
            text(fig, x + 0.188, y - 0.028, ci(record), size=8, color=MUTED, align="right")
        separator(fig, x + 0.202, top_y + 0.045, 0.0008, 0.259)
        text(fig, x + 0.2325, top_y + 0.300, "ΔTP", size=8, weight="bold", align="center")
        ax = fig.add_axes((x + 0.211, top_y + 0.061, 0.043, 0.220))
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-40, 40)
        ax.set_yticks((-40, 0, 40))
        ax.set_xticks([])
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", length=2.5, width=0.5, labelsize=8, pad=2, colors=MUTED)
        ax.spines[["left", "top", "bottom"]].set_visible(False)
        ax.spines["right"].set_color(BORDER)
        ax.spines["right"].set_linewidth(0.5)
        ax.axhline(0, color="#8B96A1", linewidth=0.7, linestyle=(0, (3, 2)))
        total = totals_lookup[(fold, "ΔTP")]
        ax.errorbar(
            0,
            total.estimate,
            yerr=[[total.estimate - total.lower], [total.upper - total.estimate]],
            fmt="o",
            markersize=4.8,
            color=effect_color(total.estimate),
            markerfacecolor=NEGATIVE if total.estimate < 0 else POSITIVE,
            ecolor=effect_color(total.estimate),
            elinewidth=1.0,
            capsize=3.5,
            capthick=0.9,
            markeredgecolor=WHITE,
            markeredgewidth=0.6,
        )
        text(fig, x + 0.2325, top_y + 0.028, "95% CI", size=8, color=MUTED, align="center")

        # Component card: aligned numerical columns and a small heatmap, like the reference.
        text(fig, x + 0.011, bottom_y + 0.296, "Workflow", size=8, weight="bold")
        text(fig, x + 0.111, bottom_y + 0.296, "H", size=8, weight="bold", align="center")
        text(fig, x + 0.172, bottom_y + 0.296, "W", size=8, weight="bold", align="center")
        text(fig, x + 0.231, bottom_y + 0.296, "H", size=8, weight="bold", align="center")
        text(fig, x + 0.270, bottom_y + 0.296, "W", size=8, weight="bold", align="center")
        separator(fig, x + 0.201, bottom_y + 0.107, 0.0008, 0.202)
        row_y = (bottom_y + 0.245, bottom_y + 0.190, bottom_y + 0.135)
        workflows = ("LightGBM", "Logistic", "Logistic minus LightGBM")
        for workflow, label, y in zip(
            workflows, ("LightGBM", "Logistic", "Direct"), row_y, strict=True
        ):
            text(fig, x + 0.011, y, label, size=8)
            for contrast, numeric_x, cell_x in (("M1-M0", 0.111, 0.213), ("M2-M1", 0.172, 0.252)):
                record = lookup[(contrast, fold, workflow)]
                text(
                    fig,
                    x + numeric_x,
                    y,
                    number(record.estimate, sign=True),
                    size=8.2,
                    weight="bold",
                    align="center",
                )
                box(fig, x + cell_x, y - 0.0225, 0.036, 0.045, cmap(norm(record.estimate)))
        separator(fig, x + 0.011, bottom_y + 0.099, 0.277, 0.0008)
        text(fig, x + 0.011, bottom_y + 0.080, "Direct 95% intervals", size=8, weight="bold")
        for contrast, label, y in (
            ("M1-M0", "H", bottom_y + 0.049),
            ("M2-M1", "W", bottom_y + 0.021),
        ):
            direct = lookup[(contrast, fold, "Logistic minus LightGBM")]
            text(fig, x + 0.011, y, f"{label}: {ci(direct)}", size=8, color=MUTED)

    cax = fig.add_axes((0.721, 0.074, 0.247, 0.010))
    bar = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor("face")
    cax.set_frame_on(False)
    bar.set_ticks((-120, 0, 120))
    bar.ax.tick_params(labelsize=8, width=0.5, length=2.5, pad=2, colors=MUTED)
    text(fig, 0.705, 0.079, "ΔTP", size=8, align="right", color=MUTED)
    text(
        fig,
        0.026,
        0.098,
        "H: history (M1 − M0)   W: weighting above history (M2 − M1)",
        size=8,
        color=MUTED,
    )
    text(
        fig,
        0.026,
        0.065,
        "Direct = Logistic effect − LightGBM effect; all component values are ΔTP.",
        size=8,
        color=MUTED,
    )
    text(
        fig,
        0.026,
        0.026,
        "Pointwise conditional 95% intervals; colours encode direction, not significance.",
        size=8,
        color=MUTED,
    )
    return fig


def render_report(fig: Figure) -> dict[str, Any]:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    width, height = canvas.get_width_height()
    clipped: list[str] = []
    boxes: list[dict[str, Any]] = []
    sizes: list[float] = []
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        extent = artist.get_window_extent(renderer)
        sizes.append(float(artist.get_fontsize()))
        boxes.append({"label": artist.get_text(), "bounds": list(extent.bounds)})
        if (
            extent.x0 < -0.5
            or extent.y0 < -0.5
            or extent.x1 > width + 0.5
            or extent.y1 > height + 0.5
        ):
            clipped.append(artist.get_text())
    if clipped or min(sizes) < 8:
        raise ValueError(f"Typography or bounds failure: {clipped}")
    return {
        "minimum_font_pt": min(sizes),
        "clipped_text": clipped,
        "text_boxes": boxes,
        "figure_width_mm": 182.88,
        "figure_height_mm": 119.38,
        "top_delta_tp_axis_ranges": [[-40, 40]] * 3,
        "heatmap_scale": [-120, 0, 120],
        "new_intervals_computed": False,
        "manuscript_modified_by_this_script": False,
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    totals = load_data(DATA / "fig5_combined_metrics.csv")
    components = load_data(DATA / "fig5_component_effects.csv")
    validate_data(totals, components)
    fig = draw(totals, components)
    report = render_report(fig)
    (QA / "render_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig.savefig(FIGURES / f"{NAME}.svg", metadata={"Creator": ""})
    fig.savefig(FIGURES / f"{NAME}.pdf", metadata={"Creator": "", "Producer": ""})
    fig.savefig(FIGURES / f"{NAME}.png", dpi=600, metadata={"Software": ""})
    fig.savefig(FIGURES / f"{NAME}_preview.png", dpi=240, metadata={"Software": ""})
    plt.close(fig)
    with Image.open(FIGURES / f"{NAME}_preview.png") as image:
        image.convert("L").save(QA / "grayscale_check.png")
    (QA / "Figure_caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    (QA / "README.md").write_text(
        "# Reference-style six-panel IEEE-CIS figure\n\n"
        "Columns are Folds 1–3. The top row shows combined effects and original "
        "Table 9 intervals; the bottom row shows Table 10 component effects and "
        "directly paired workflow differences. All 81 reported numerical values "
        "are included in the data exports. Within-workflow component intervals "
        "remain in Table 10 and component_effects.csv. Only directly paired "
        "component intervals are printed on the figure.\n\n"
        "Top-row estimates use red for negative values and green for positive "
        "values, evaluated separately for each metric. Marker fills match the "
        "heatmap endpoint hues; darker related colours keep text and whiskers "
        "legible. Colours encode direction, not statistical significance.\n\n"
        "Physical size: 182.88 × 119.38 mm; all text is at least 8 pt. Do not "
        "shrink to a single column without redesigning the typography.\n\n"
        "All cards share fixed inner margins, row centres and column widths. "
        "Combined estimates and intervals share a right edge; ΔTP plots have "
        "an independent column. Component numbers and heatmap cells share row "
        "centres. Detailed evidence boundaries remain in Figure_caption.txt.\n\n"
        "Run `python recreate_template_style.py` beside `data/`; dependencies: "
        "Matplotlib and Pillow. SVG text is editable and PDF/SVG contain no "
        "raster images. Output includes 600-DPI PNG and a short preview.\n\n"
        "No manuscript is edited. Current source content is Fig. 5 + Fig. 6 "
        "(previously discussed as Fig. 6 + Fig. 7). The six panels are an "
        "organisation of existing evidence, not six new experiments.\n",
        encoding="utf-8",
    )
    checksums = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(FIGURES.iterdir())
        if p.is_file()
    }
    (QA / "figure_checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    print(f"Created {FIGURES}")


if __name__ == "__main__":
    main()
