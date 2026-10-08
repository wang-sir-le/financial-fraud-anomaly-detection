"""PaySim Fig. 3: paired effects with a numeric evidence column.

Input: the adjacent fig3.json, copied byte-for-byte from the preceding redraw.
Only saved fold and equal-fold aggregates are used. No new model, score,
rank, AP, bootstrap distribution, confidence interval or observation is made.

Design: aligned rows from gallery oa-bio-plos-pbio-3002856-f06; restrained
point/interval hierarchy from oa-tech-pone-0258309-f05. Their densities,
significance categories, odds-ratio baseline and data are not copied.
The blue/teal palette is the user's already chosen palette. Test step ranges
are transcribed from Table 1 of the current manuscript. The equal-fold
summary remains secondary, with its own saved interval unchanged.
"""
from hashlib import sha256
import json
from typing import Any, cast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

HERE = Path(__file__).resolve().parent
INPUT = HERE / "fig3.json"
EXPECTED_INPUT_SHA256 = "2a14d977bef5f55c9dc5aff8303870bc324ae15847b1d3c23e24517abc6ebe29"
BLUE, TEAL, INK, MUTED = "#244F72", "#078A8F", "#24323C", "#60707B"
PALE, LINE = "#F1F5F7", "#D8E1E6"
STEP_LABELS = ["steps 239–281", "steps 324–354", "steps 399–743", "secondary"]


def signed(value, digits):
    return f"{value:+.{digits}f}".replace("-", "−")


def interval_text(row, digits):
    return f"[{row['lower']:.{digits}f}, {row['upper']:.{digits}f}]".replace("-", "−")


def main():
    data_bytes = INPUT.read_bytes()
    assert sha256(data_bytes).hexdigest() == EXPECTED_INPUT_SHA256, "Input aggregates changed"
    data = json.loads(data_bytes)
    assert data["contrast"] == "PTHW - P0" and data["point_count"] == 8
    assert [panel["metric"] for panel in data["panels"]] == ["delta_recall", "delta_pr_auc"]
    expected_labels = ["Fold 1", "Fold 2", "Fold 3", "Equal-fold"]
    for panel in data["panels"]:
        assert [row["label"] for row in panel["points"]] == expected_labels
        assert panel["points"][-1]["summary"] == "secondary"
        for row in panel["points"]:
            assert row["confidence"] == .95
            assert row["lower"] <= row["estimate"] <= row["upper"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.2, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(data["width_in"], data["height_in"]))
    bottom, body_height, n = .22, .57, 4
    positions = [(.158, .249, .420, .569), (.597, .221, .829, .990)]
    fig.text(.015, .98, "PaySim  |  PTHW − P0", fontsize=10.2, fontweight="bold", va="top")
    fig.text(.985, .97, "Paired effects", ha="right", fontsize=8.3, color=MUTED, va="top")
    fig.text(.015, .85, "Test window", fontsize=8, fontweight="bold")
    fig.add_artist(plt.Rectangle((.012, bottom), .979, body_height / n,
                                transform=fig.transFigure, color=PALE, zorder=-10))
    row_centers = [bottom + body_height * (n - .5 - i) / n for i in range(n)]
    for i, label in enumerate(expected_labels):
        label = "Equal-fold*" if i == n - 1 else label
        fig.text(.015, row_centers[i] + .023, label, fontsize=8.3, va="center",
                 fontweight="bold" if i == n - 1 else "normal")
        fig.text(.015, row_centers[i] - .033, STEP_LABELS[i], fontsize=7.2, va="center", color=MUTED)
    audit_points, numeric_artists = [], []
    for metric, (panel, position, heading, color) in enumerate(zip(
            data["panels"], positions, ["(a) Recall", "(b) Average precision"], [BLUE, TEAL], strict=True)):
        x, plot_width, text_left, text_right = position
        ax = fig.add_axes((x, bottom, plot_width, body_height))
        ax.set_ylim(3.5, -.5)
        ax.set_yticks([])
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#9BAAB3")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2.7, width=.55, labelsize=7.6)
        ax.grid(axis="x", color="#E6ECEF", linewidth=.45)
        ax.axvline(0, color="#778892", lw=.8, ls=(0, (3, 2)), zorder=2)
        ax.set_facecolor((1, 1, 1, 0))
        ax.axhspan(2.5, 3.5, color=PALE, zorder=-1)
        if metric == 0:
            ax.set_xlim(-4, 74)
            ax.set_xticks([0, 20, 40, 60])
        else:
            ax.set_xlim(-.17, .39)
            ax.set_xticks([-.1, 0, .1, .2, .3])
            ax.set_xticklabels(["−0.1", "0", "0.1", "0.2", "0.3"])
        fig.text(x, .85, heading, fontsize=8.5, fontweight="bold", color=color)
        fig.text((text_left + text_right) / 2, .85, "Estimate [95% CI]", fontsize=7.1,
                 fontweight="bold", ha="center")
        digits = 2 if metric == 0 else 4
        for index, (row_y, row) in enumerate(zip(row_centers, panel["points"], strict=True)):
            point_color = MUTED if index == 3 else color
            xerr = np.array([[row["estimate"] - row["lower"]], [row["upper"] - row["estimate"]]])
            ax.errorbar(row["estimate"], index, xerr=xerr,
                        fmt=["o", "s", "^", "D"][index], color=point_color,
                        markersize=4.9, markeredgewidth=.65, capsize=2.6,
                        elinewidth=1.25, capthick=.75, zorder=4)
            numeric_artists.append((
                fig.text((text_left + text_right) / 2, row_y + .024, signed(row["estimate"], digits),
                         fontsize=8.6, fontweight="bold", va="center", ha="center", color=point_color),
                text_left, text_right))
            numeric_artists.append((
                fig.text((text_left + text_right) / 2, row_y - .034, interval_text(row, digits),
                         fontsize=7.5, va="center", ha="center", color=MUTED),
                text_left, text_right))
            assert ax.get_xlim()[0] < row["lower"] <= row["upper"] < ax.get_xlim()[1]
            audit_points.append({"metric": panel["metric"], **row,
                                 "display_estimate": signed(row["estimate"], digits),
                                 "display_interval": interval_text(row, digits)})
        ax.set_xlabel("Recall difference (pp)" if metric == 0 else "AP difference", fontsize=7.9, labelpad=4.4)
    fig.add_artist(plt.Line2D([.012, .990], [.825, .825], transform=fig.transFigure, color=LINE, lw=.65))
    fig.text(.015, .012,
             "Points: five-seed paired means  ·  Bars: 95% within-window intervals  ·  *Secondary summary",
             fontsize=7.1, color=MUTED, va="bottom")
    fig.canvas.draw()
    renderer = cast(Any, fig.canvas).get_renderer()
    numeric_checks = []
    for artist, left, right in numeric_artists:
        bounds = artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert bounds.x0 >= left - .002 and bounds.x1 <= right + .002, f"Numeric column overflow: {artist.get_text()}"
        numeric_checks.append({"text": artist.get_text(), "bbox": list(bounds.bounds)})
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text() or artist.axes is not None:
            continue
        bounds = artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert bounds.x0 >= -.002 and bounds.x1 <= 1.002 and bounds.y0 >= -.002 and bounds.y1 <= 1.002, artist.get_text()
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(HERE / f"fig3.{extension}", dpi=450, facecolor="white")
    plt.close(fig)
    audit = {
        "input_sha256": EXPECTED_INPUT_SHA256,
        "unchanged_estimates_and_interval_pairs": len(audit_points),
        "points": audit_points, "numeric_column_checks": numeric_checks,
        "matplotlib_version": matplotlib.__version__,
        "no_transactions_labels_predictions_or_bootstrap_draws_opened": True,
        "no_new_statistics_or_significance_encoding": True,
        "same_document_frame_width_and_height": True,
        "visual_review": "PENDING",
    }
    (HERE / "fig3_verification.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Fig. 3: all 8 saved effects/intervals retained; PNG, vector PDF, SVG exported.")


if __name__ == "__main__":
    main()
