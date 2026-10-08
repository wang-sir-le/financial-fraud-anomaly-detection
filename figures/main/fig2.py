"""Draw the preserved original comparison map, using manuscript text only.

No experiment inputs, predictions, labels, or model files are read.
Default outputs are written beside this script; all SVG labels remain text.
Physical geometry exactly matches the existing Fig. 2 drawing in the DOCX.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, cast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


WIDTH = 6.6
HEIGHT = 3060020 / 914400
DPI = 400
INK = "#243D49"
BLUE = "#B0CEDD"
LIGHTBLUE = "#EEF5F9"
BORDER = "#8EA8B3"
GREEN = "#EAF2ED"
GREENINK = "#4E7865"
GOLD = "#FAEBC7"
GOLDBORDER = "#D9B66E"


def draw(out_dir: Path, audit_path: Path | None = None) -> dict:
    """Render and reject clipped, region-overflowing, or overlapping labels."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.3,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    font_path = font_manager.findfont("Arial", fallback_to_default=False)
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    text_records = []

    def box(x, y, w, h, face="white", edge=BORDER, rounded=True, lw=0.65):
        artist: Any
        if rounded:
            artist = FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0,rounding_size=0.032",
                linewidth=lw, edgecolor=edge, facecolor=face,
            )
        else:
            artist = Rectangle((x, y), w, h, linewidth=lw, edgecolor=edge, facecolor=face)
        ax.add_patch(artist)
        return artist

    def label(x, y, s, size=8.3, bold=False, ha="left", color=INK, region=None):
        artist = ax.text(
            x, y, s, fontsize=size, fontweight="bold" if bold else "normal",
            ha=ha, va="center", color=color, linespacing=1.2,
        )
        text_records.append((artist, region))
        return artist

    def arrow(x1, x2, y):
        ax.add_patch(FancyArrowPatch(
            (x1, y), (x2, y), arrowstyle="-|>", mutation_scale=6.5,
            linewidth=0.8, color=GREENINK, shrinkA=0, shrinkB=0,
        ))

    def pill(x, y, w, text):
        region = (x, y, w, 0.23)
        box(*region, face="white", edge=GOLDBORDER, lw=0.55)
        label(x + w / 2, y + 0.115, text, size=8.6, ha="center", region=region)

    # Reference-derived layout: grouped blue question lanes, gold comparison
    # nodes, green readouts, restrained outlines, and connected horizontal flow.
    label(0.07, HEIGHT - 0.17, "Original LightGBM comparison map", size=10.0, bold=True)
    label(6.53, HEIGHT - 0.17, "PaySim / IEEE-CIS", size=8.1, ha="right", color=GREENINK)

    left = (0.07, 1.48)
    middle = (1.73, 2.93)
    right = (4.84, 1.69)
    header_y = 2.805
    for (x, w), title in (
        (left, "Original question"),
        (middle, "Within-pipeline comparison"),
        (right, "Readouts and scope"),
    ):
        region = (x, header_y, w, 0.265)
        box(*region, face=BLUE, rounded=False)
        label(x + w / 2, header_y + 0.1325, title, size=8.5, bold=True, ha="center", region=region)

    rows = [2.02, 1.245, 0.47]
    row_h = 0.675
    for y in rows:
        box(left[0], y, left[1], row_h, face=LIGHTBLUE)
        box(middle[0], y, middle[1], row_h, face=GOLD, edge=GOLDBORDER)
        box(right[0], y, right[1], row_h, face=GREEN)
        arrow(left[0] + left[1] + 0.025, middle[0] - 0.025, y + row_h / 2)
        arrow(middle[0] + middle[1] + 0.025, right[0] - 0.025, y + row_h / 2)

    y = rows[0]
    label(0.81, y + 0.54, "RQ1 · Representation", size=8.5, bold=True, ha="center", region=(left[0], y, left[1], row_h))
    label(0.81, y + 0.335, "Recipient history +\ncurrent-time context", size=8.3, ha="center", region=(left[0], y, left[1], row_h))
    # Past-state records connected to a distinct current-time context node.
    for k in range(3):
        box(0.35 + k * 0.105, y + 0.095, 0.075, 0.07, face=BLUE, rounded=False, lw=0.45)
    ax.plot([0.67, 0.9], [y + 0.13, y + 0.13], color=BORDER, linewidth=0.7)
    box(0.94, y + 0.09, 0.18, 0.08, face=GOLD, edge=GOLDBORDER, rounded=False, lw=0.5)
    label(1.865, y + 0.466, "PaySim", size=8.3, bold=True)
    pill(2.56, y + 0.35, 0.70, "PH − P0")
    label(3.40, y + 0.465, "and", size=8.0, ha="center")
    pill(3.56, y + 0.35, 0.94, "PTH − PT")
    label(1.865, y + 0.18, "IEEE-CIS", size=8.3, bold=True)
    pill(2.77, y + 0.065, 1.26, "M1 − M0")
    label(5.685, y + 0.415, "AP by score stage", size=8.3, ha="center", region=(right[0], y, right[1], row_h))
    label(5.685, y + 0.225, "Capture at matched budgets", size=8.0, ha="center", region=(right[0], y, right[1], row_h))

    y = rows[1]
    label(0.81, y + 0.54, "RQ2 · Weighting", size=8.5, bold=True, ha="center", region=(left[0], y, left[1], row_h))
    label(0.81, y + 0.345, "Fixed feature sets", size=8.3, ha="center", region=(left[0], y, left[1], row_h))
    box(0.38, y + 0.075, 0.31, 0.155, face="white", rounded=False, lw=0.5)
    label(0.535, y + 0.1525, "w = 1", size=8.0, ha="center", region=(0.38, y + 0.075, 0.31, 0.155))
    arrow(0.73, 0.89, y + 0.15)
    box(0.94, y + 0.075, 0.31, 0.155, face=GOLD, edge=GOLDBORDER, rounded=False, lw=0.5)
    label(1.095, y + 0.1525, "w = 2", size=8.0, ha="center", region=(0.94, y + 0.075, 0.31, 0.155))
    label(1.865, y + 0.466, "PaySim", size=8.3, bold=True)
    pill(2.77, y + 0.35, 1.50, "PTHW − PTH")
    label(1.865, y + 0.18, "IEEE-CIS", size=8.3, bold=True)
    pill(2.77, y + 0.065, 1.26, "M2 − M1")
    label(5.685, y + 0.415, "AP and capture", size=8.3, ha="center", region=(right[0], y, right[1], row_h))
    label(5.685, y + 0.225, "Workload + analytical cost", size=8.0, ha="center", region=(right[0], y, right[1], row_h))

    y = rows[2]
    label(0.81, y + 0.54, "RQ3 · Conditions", size=8.5, bold=True, ha="center", region=(left[0], y, left[1], row_h))
    label(0.81, y + 0.29, "Future windows\nCapacity and cost", size=8.3, ha="center", region=(left[0], y, left[1], row_h))
    # Three equally sized glyphs represent the three declared windows, not
    # a fabricated result or an assertion that their time durations are equal.
    for k in range(3):
        x = 1.94 + k * 0.17
        box(x, y + 0.235, 0.13, 0.20, face="white", edge=GOLDBORDER, rounded=False, lw=0.5)
        label(x + 0.065, y + 0.335, str(k + 1), size=8.0, ha="center", region=(x, y + 0.235, 0.13, 0.20))
    label(2.59, y + 0.43, "Three windows per dataset", size=8.3, region=(middle[0], y, middle[1], row_h))
    label(2.59, y + 0.225, "All declared scenario cells", size=8.3, region=(middle[0], y, middle[1], row_h))
    label(5.685, y + 0.415, "Window-specific intervals", size=8.0, ha="center", region=(right[0], y, right[1], row_h))
    label(5.685, y + 0.225, "Descriptive scenario grids", size=8.0, ha="center", region=(right[0], y, right[1], row_h))

    footer = (0.07, 0.05, 6.46, 0.315)
    box(*footer, face=GREEN, edge="#BED0C6", lw=0.55)
    label(3.30, 0.252, "Primary and core-component intervals are reported separately.", size=7.8, ha="center", color=GREENINK, region=footer)
    label(3.30, 0.109, "Inference is conditional on fixed models and windows; no model-independent effect.", size=7.8, ha="center", color=GREENINK, region=footer)

    fig.canvas.draw()
    renderer = cast(Any, fig.canvas).get_renderer()
    issues = []
    text_bounds = []
    for artist, region in text_records:
        bbox = artist.get_window_extent(renderer)
        inches = ax.transData.inverted().transform(bbox.get_points())
        text_bounds.append((artist.get_text(), inches.tolist(), bbox))
        x0, y0 = inches[0]
        x1, y1 = inches[1]
        if x0 < 0 or y0 < 0 or x1 > WIDTH or y1 > HEIGHT:
            issues.append(f"Canvas clipping: {artist.get_text()}")
        if region:
            rx, ry, rw, rh = region
            if x0 < rx + 0.015 or y0 < ry + 0.008 or x1 > rx + rw - 0.015 or y1 > ry + rh - 0.008:
                issues.append(f"Label escapes region: {artist.get_text()}")
    for i, (label_a, _, bbox_a) in enumerate(text_bounds):
        for label_b, _, bbox_b in text_bounds[i + 1:]:
            if bbox_a.overlaps(bbox_b):
                issues.append(f"Text collision: {label_a!r} / {label_b!r}")
    audit = {
        "figure": "Fig. 2",
        "width_inches": WIDTH,
        "height_inches": HEIGHT,
        "source_docx_extent_emu": {"cx": 6035040, "cy": 3060020},
        "font_path": font_path,
        "minimum_font_pt": min(a.get_fontsize() for a, _ in text_records),
        "png_dpi": DPI,
        "text_count": len(text_records),
        "text_bounds_inches": [{"text": s, "bbox": b} for s, b, _ in text_bounds],
        "automated_geometry_issues": issues,
        "original_comparison_tokens": ["RQ1 · Representation", "RQ2 · Weighting", "RQ3 · Conditions", "PH − P0", "PTH − PT", "M1 − M0", "PTHW − PTH", "M2 − M1"],
        "no_experimental_inputs_read": True,
        "style_reference": "User-supplied grouped blue/gold/green workflow diagram",
        "icon_scope": "Schematic history/time, fixed weight 1 to 2, and three declared-window glyphs; not results",
    }
    if audit_path:
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if issues:
        plt.close(fig)
        raise AssertionError("; ".join(issues))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"fig2.{ext}", dpi=DPI, transparent=False)
    plt.close(fig)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    audit = draw(args.out_dir, args.audit)
    print(json.dumps({k: audit[k] for k in ("figure", "width_inches", "height_inches", "minimum_font_pt", "text_count", "automated_geometry_issues")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
