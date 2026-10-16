"""Reproduce the current five-panel Figure 6 from compact, frozen source records.

Usage: python fig6.py [--output-dir DIRECTORY] [--with-caption]
The default exports match the manuscript artwork, with a separate editable caption.
Only saved aggregate/seed records are used; no models or bootstrap draws are created.
Requires Python >= 3.10 and matplotlib >= 3.8.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parent
OUT = ROOT
DATA = ROOT / "data" / "fig6"
PALETTE = {1: "#AD6345", 2: "#8A7195", 3: "#667D56"}
MARKERS = {1: "o", 2: "s", 3: "^"}
INK, MUTED = "#101827", "#65707C"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def rebuild_inputs():
    manifest = json.loads((DATA / "source_manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == info["copied_sha256"], name
    seeds = (42, 52, 62, 72, 82)
    runs = read_csv(DATA / "paysim_seed_scenarios.csv")
    lookup = {(int(r["fold"]), r["model"], int(r["seed"]), float(r["capacity"]), float(r["fn_cost"])): r for r in runs}
    assert len(lookup) == len(runs) == 660
    for r in runs:
        assert int(r["tp"]) + int(r["fn"]) == int(r["frauds"])
        assert int(r["tp"]) + int(r["fp"]) == int(r["alerts"])
        assert math.isclose(float(r["cost"]), float(r["fn_cost"]) * int(r["fn"]) + int(r["fp"]), abs_tol=1e-8)
        assert math.isclose(float(r["recall"]), int(r["tp"]) / int(r["frauds"]), abs_tol=1e-14)
        assert math.isclose(float(r["alert_rate"]), int(r["alerts"]) / int(r["n"]), abs_tol=1e-14)
    frozen = read_csv(DATA / "fig6_scenarios.csv")
    records = [dict(panel="a", fold=int(r["fold"]), model=r["comparison_or_model"],
                    x=float(r["x"]), value=float(r["value"])) for r in frozen if r["panel"] == "a"]
    assert len(records) == 12
    for fold in (1, 2, 3):
        for q in (.01, .03, .05, .10):
            paired = [(lookup[(fold, "PTHW", seed, q, 100)], lookup[(fold, "P0", seed, q, 100)]) for seed in seeds]
            assert all(p["n"] == b["n"] and p["frauds"] == b["frauds"] for p, b in paired)
            difference = statistics.mean(100 * (float(p["recall"]) - float(b["recall"])) for p, b in paired)
            records.append(dict(panel="b", fold=fold, model="PTHW-P0", x=100*q, value=difference))
            for model in ("PTHW", "P0"):
                alerts = statistics.mean(100 * float(lookup[(fold, model, s, q, 100)]["alert_rate"]) for s in seeds)
                records.append(dict(panel="c", fold=fold, model=model, x=100*q, value=alerts))
        for cost in (10, 50, 100, 500, 1000):
            paired = [(lookup[(fold, "PTHW", s, .03, cost)], lookup[(fold, "P0", s, .03, cost)]) for s in seeds]
            reduction = statistics.mean(100 * (float(b["cost"]) - float(p["cost"])) / float(b["cost"]) for p, b in paired)
            records.append(dict(panel="e", fold=fold, model="PTHW-P0", x=cost, value=reduction))
            for model in ("PTHW", "P0"):
                alerts = statistics.mean(100 * float(lookup[(fold, model, s, .03, cost)]["alert_rate"]) for s in seeds)
                records.append(dict(panel="f", fold=fold, model=model, x=cost, value=alerts))
    assert len(records) == 93
    capacity = []
    seed_points = read_csv(DATA / "capacity_seed_points.csv")
    assert len(seed_points) == 60
    for fold in (1, 2, 3):
        for q in (.001, .01, .03, .05):
            rows = [r for r in seed_points if int(r["fold"]) == fold and abs(float(r["q"]) - q) < 1e-12]
            assert len(rows) == 5 and sorted(int(r["seed"]) for r in rows) == list(seeds)
            for r in rows:
                assert r["model"] == "PTHW" and r["score_stage"] == "platt"
                assert int(r["k"]) == math.ceil(float(r["q"]) * int(r["n"]))
                assert math.isclose(float(r["recall"]), int(r["tp"]) / int(r["frauds"]), abs_tol=1e-14)
            value = statistics.mean(100 * int(r["tp"]) / int(r["frauds"]) for r in rows)
            capacity.append(dict(model="PTHW", score_stage="platt", fold=str(fold),
                                 q=str(q), seed_count="5", mean_recall_pct=str(value)))
    effects = read_csv(DATA / "primary_effects.csv")
    bootstrap = read_csv(DATA / "paysim_bootstrap_summary.csv")
    for fold in (1, 2, 3):
        point = next(r for r in effects if r["metric"] == "recall" and r["fold"] == str(fold))
        source = next(r for r in bootstrap if r["metric"] == "delta_recall" and r["fold"] == str(fold))
        for dest, origin in (("estimate", "observed_effect"), ("ci_lower", "ci_lower_95"), ("ci_upper", "ci_upper_95")):
            assert math.isclose(float(point[dest]), 100 * float(source[origin]), abs_tol=1e-12)
        assert int(source["valid_replicates"]) == 5000
    return records, capacity, effects, read_csv(DATA / "ieee_reference_effects.csv")


def main(with_caption=False):
    OUT.mkdir(parents=True, exist_ok=True)
    verified, capacity, effects, ieee = rebuild_inputs()
    sans, serif = "DejaVu Sans", "DejaVu Serif"
    plt.rcParams.update({"font.family": sans, "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.unicode_minus": True,
                         "pdf.fonttype": 42, "svg.fonttype": "path", "svg.hashsalt": "fig6-checked-20261014"})
    fig = plt.figure(figsize=(170 / 25.4, 166 / 25.4), facecolor="white")
    fig.legend(handles=[Line2D([], [], color=PALETTE[f], marker=MARKERS[f],
                               linewidth=1, markersize=4, label=f"Fold {f}") for f in (1, 2, 3)],
               loc="upper center", bbox_to_anchor=(.50, .999), ncol=3,
               fontsize=8, frameon=True, fancybox=True, facecolor="#EEF0F2", edgecolor="none",
               handlelength=1.8, columnspacing=1.7, handletextpad=.6, borderpad=.35, borderaxespad=.2)
    headings = ((.009, .948, "(a) Recall sensitivity across temporal windows"),
                (.009, .680, "(b) Cost sensitivity under asymmetric loss"),
                (.530, .680, "(c) Detection versus review capacity"),
                (.009, .410, "(d) PaySim Test alert rate"),
                (.530, .410, "(e) Reference recall effects"))
    for x, y, title in headings:
        fig.text(x, y, title, fontsize=9.1, fontweight="bold", ha="left", va="center")

    specs = {
        "a_pay": ((.077, .757, .407, .143), "PaySim (PTHW − P0) | c = 100"),
        "a_ieee": ((.583, .757, .405, .143), "IEEE-CIS (M2 − M0) | raw scores"),
        "b": ((.077, .487, .407, .143), "PaySim (PTHW − P0) | Validation cap = 3%"),
        "c": ((.583, .487, .405, .143), "PaySim (PTHW) | exact budgets | Platt scores"),
        "d_cap": ((.068, .168, .178, .173), "Validation cap"),
        "d_cost": ((.326, .168, .178, .173), "False-negative cost"),
    }
    axes, plotted, glow_series = {}, [], []
    for key, (rect, title) in specs.items():
        ax = fig.add_axes(rect)
        axes[key] = ax
        fig.text(rect[0] + rect[2] / 2, rect[1] + rect[3] + (.043 if key.startswith("d_") else .020), title,
                 fontsize=8.0, fontweight="bold" if key.startswith("d_") else "normal",
                 ha="center", va="center")
        if key.startswith("d_"):
            fig.text(rect[0] + rect[2] / 2, rect[1] + rect[3] + .020,
                     "c = 100" if key == "d_cap" else "Validation cap = 3%",
                     fontsize=8, color=MUTED, ha="center", va="center")
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(.6)
            ax.spines[side].set_color("#59616B")
        ax.tick_params(length=2.5, width=.5, pad=3)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#C8CDD3", linestyle=(0, (6, 2)), linewidth=.5, alpha=.8)
        ax.grid(axis="x", color="#DCE0E5", linestyle=(0, (5, 3)), linewidth=.4, alpha=.58)
        if key in ("a_pay", "a_ieee", "d_cap"):
            ax.set_xlim(.5, 10.5)
            ax.set_xticks((1, 3, 5, 10))
            ax.axvline(3, linewidth=.6, linestyle=(0, (4, 3)), color="#9AA3AF")
            ax.set_xlabel("Exact Test budget (%)" if key == "a_ieee" else
                          "Validation cap (%)" if key == "d_cap" else "Validation alert cap (%)", labelpad=4)
        elif key in ("b", "d_cost"):
            ax.set_xscale("log")
            ax.set_xlim(7, 1400)
            ax.xaxis.set_major_locator(FixedLocator((10, 100, 1000) if key == "d_cost" else (10, 50, 100, 500, 1000)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.axvline(100, linewidth=.6, linestyle=(0, (4, 3)), color="#9AA3AF")
            ax.set_xlabel("c (log scale)" if key == "d_cost" else "False-negative cost c (log scale)", labelpad=4)
        else:
            ax.set_xlim(0, 5.2)
            ax.set_xticks((.1, 1, 3, 5), labels=("0.1", "1", "3", "5"))
            ax.set_xlabel("Exact Test review capacity (%)", labelpad=4)
        if key == "a_pay":
            ax.set(ylim=(0, 65), yticks=(0, 20, 40, 60), ylabel="Recall difference (pp)")
        elif key == "a_ieee":
            ax.set(ylim=(-1.5, 6.5), yticks=(-1, 0, 2, 4, 6), ylabel="Recall difference (pp)")
            ax.axhline(0, linewidth=.7, linestyle=(0, (6, 2)), color="#919AA6")
        elif key == "b":
            ax.set(ylim=(-10, 70), yticks=(0, 20, 40, 60), ylabel="Cost reduction (%)")
            ax.axhline(0, linewidth=.7, linestyle=(0, (6, 2)), color="#919AA6")
        elif key == "c":
            ax.set(ylim=(0, 100), yticks=(0, 20, 40, 60, 80, 100), ylabel="Recall@K (%)")
        else:
            ax.set(ylim=(-.15, 5), yticks=(0, 1, 2, 3, 4, 5), ylabel="Test alert rate (%)")

        source_panel = {"a_pay": "b", "a_ieee": "a", "b": "e", "d_cap": "c", "d_cost": "f"}.get(key)
        models = ("PTHW", "P0") if key.startswith("d_") else ("M2-M0",) if key == "a_ieee" else ("PTHW-P0",)
        if key == "c":
            models = ("PTHW",)
        for model in models:
            for fold in (1, 2, 3):
                if key == "c":
                    chosen = []
                    for q in (.001, .01, .03, .05):
                        match = [r for r in capacity if r["model"] == "PTHW" and r["score_stage"] == "platt"
                                 and int(r["fold"]) == fold and abs(float(r["q"]) - q) < 1e-12]
                        assert len(match) == 1 and int(match[0]["seed_count"]) == 5
                        chosen.append((100 * float(match[0]["q"]), float(match[0]["mean_recall_pct"])))
                    source = "capacity_seed_points.csv; mean(100 TP/frauds)"
                else:
                    rows = sorted((r for r in verified if r["panel"] == source_panel
                                   and int(r["fold"]) == fold and r["model"] == model), key=lambda r:float(r["x"]))
                    assert len(rows) == (5 if key in ("b", "d_cost") else 4)
                    chosen = [(float(r["x"]), float(r["value"])) for r in rows]
                    source = "fig6_scenarios.csv (frozen IEEE means)" if key == "a_ieee" else "paysim_seed_scenarios.csv; five-seed aggregation"
                xs, ys = zip(*chosen)
                baseline = model == "P0"
                # Width is in print points, independent of the numeric axes.
                # This is a visual highlight, not a statistical interval.
                strokes = ((5.5, .045), (3.8, .065), (2.2, .045)) if key.startswith("d_") else (
                    (8.5, .055), (5.7, .075), (3.4, .055))
                for stroke_width, opacity in strokes:
                    # Matplotlib scales dash lengths by linewidth; compensate so
                    # the soft strokes follow the printed dashes of the .95pt line.
                    glow_style = (0, (5 * .95 / stroke_width, 3 * .95 / stroke_width)) if baseline else "-"
                    halo, = ax.plot(xs, ys, color=PALETTE[fold], linewidth=stroke_width,
                                    alpha=opacity, solid_capstyle="round", solid_joinstyle="round",
                                    dash_capstyle="round", dash_joinstyle="round", linestyle=glow_style,
                                    zorder=2, label="_nolegend_")
                    assert list(halo.get_xdata()) == list(xs) and list(halo.get_ydata()) == list(ys)
                glow_series.append(dict(panel=key, fold=fold, model=model, dash_matched=baseline))
                line, = ax.plot(xs, ys, color=PALETTE[fold], marker=MARKERS[fold],
                                markersize=3.5 if key.startswith("d_") else 4,
                                linewidth=.95, markeredgewidth=.7,
                                markerfacecolor="white" if baseline else PALETTE[fold],
                                linestyle=(0, (5, 3)) if baseline else "-", zorder=4)
                assert list(line.get_xdata()) == list(xs) and list(line.get_ydata()) == list(ys)
                plotted.extend(dict(panel=key, fold=fold, model=model, x=x, value=y,
                                    ci_lower="", ci_upper="", source=source) for x, y in chosen)
    fig.legend(handles=[Line2D([], [], color="#202833", linewidth=1, label="PTHW"),
                        Line2D([], [], color="#202833", linewidth=1, linestyle=(0, (5, 3)), label="P0")],
               loc="center right", bbox_to_anchor=(.504, .410), ncol=2,
               frameon=False, fontsize=8, handlelength=1.4, handletextpad=.4,
               columnspacing=.8, borderaxespad=0)

    fig.text(.6245, .387, "95% CI · separate scales", fontsize=8, color=MUTED, ha="left", va="center")
    fig.text(.6245, .366, "PaySim · PTHW − P0", fontsize=8, color=MUTED, ha="left", va="center")
    fig.text(.6245, .251, "IEEE-CIS · M2 − M0", fontsize=8, color=MUTED, ha="left", va="center")
    forest_rects = {"PaySim": (.666088, .285924, .283824, .064152), "IEEE-CIS": (.666088, .166924, .283824, .064152)}
    for dataset, rect in forest_rects.items():
        ax = fig.add_axes(rect)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#59616B")
        ax.spines["bottom"].set_linewidth(.6)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.tick_params(axis="x", length=2.5, width=.5, pad=2)
        ax.set_ylim(2.5, -.5)
        ax.set_yticks((0, 1, 2), labels=[f"W{f}" if dataset == "PaySim" else f"F{f}" for f in (1, 2, 3)])
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#DFE2E6", linewidth=.4, alpha=.7)
        if dataset == "PaySim":
            ax.set_xlim(25, 75)
            ax.set_xticks((30, 50, 70))
        else:
            ax.set_xlim(-2, 2)
            ax.set_xticks((-2, -1, 0, 1, 2))
            ax.axvline(0, color="#77818E", linewidth=.6, linestyle=(0, (5, 3)))
            ax.set_xlabel("Recall difference (pp)", labelpad=3)
        for fold in (1, 2, 3):
            rows = [r for r in effects if r["metric"] == "recall" and r["fold"] == str(fold)] if dataset == "PaySim" else [
                r for r in ieee if r["contrast"] == "M2-M0" and r["workflow"] == "LightGBM"
                and r["metric"] == "ΔRecall" and r["fold"] == str(fold)]
            assert len(rows) == 1
            row = rows[0]
            value, lo, hi = (float(row[k]) for k in ("estimate", "ci_lower", "ci_upper"))
            assert lo <= value <= hi
            reference = next(r for r in plotted if r["panel"] == ("a_pay" if dataset == "PaySim" else "a_ieee")
                             and r["fold"] == fold and r["x"] == 3)
            assert abs(value - reference["value"]) < .0006
            ax.errorbar(value, fold - 1, xerr=[[value - lo], [hi - value]], fmt=MARKERS[fold],
                        color=PALETTE[fold], markersize=3.4, elinewidth=.87, capsize=1.8, capthick=.67)
            plotted.append(dict(panel="e", fold=fold, model=dataset, x=fold, value=value,
                                ci_lower=lo, ci_upper=hi, source="primary_effects.csv" if dataset == "PaySim" else "ieee_reference_effects.csv"))
    caption = [
        (True, "Fig. 6 | Capacity, cost sensitivity and reference effects across temporal Test windows."),
        (False, "(a) Recall differences; (b) cost reduction; (c) exact-budget recall; (d) PaySim Test alert rates."),
        (False, "PaySim transfers Validation thresholds in (a, b, d); (c) ranks completed-Test Platt scores at exact budgets."),
        (False, "(e) PaySim: PTHW − P0, Val. cap 3%, c = 100; IEEE-CIS: LightGBM M2 − M0, exact Test budget 3%."),
    ]
    for i, (bold, text) in enumerate(caption):
        fig.text(.012, .078 - i * .018, text, fontsize=8, fontfamily=serif,
                 fontweight="bold" if bold else "normal", ha="left", va="baseline")
    fig.text(.012, .006, "Five-seed means; lines guide reading; glow is decorative. CI are conditional on fixed workflows and windows.",
             fontsize=8, fontfamily=serif, ha="left", va="baseline")
    # The manuscript uses a native caption. Retain the approved plot geometry
    # and remove only the standalone footer before the matching native crop.
    if not with_caption:
        for artist in list(fig.texts):
            if artist.get_position()[1] < .1:
                artist.remove()
    assert len(plotted) == 111
    expected = read_csv(DATA / "expected_plot_data.csv")
    lookup = {(r["panel"], int(r["fold"]), r["model"], round(float(r["x"]), 10)): r for r in expected}
    assert len(lookup) == len(expected) == 111
    differences = []
    for row in plotted:
        reference = lookup[(row["panel"], row["fold"], row["model"], round(row["x"], 10))]
        differences.append(abs(row["value"] - float(reference["value"])))
        assert differences[-1] < 1e-10
        for k in ("ci_lower", "ci_upper"):
            assert row[k] == "" if reference[k] == "" else row[k] == float(reference[k])
    assert len(glow_series) == 24 and {r["panel"] for r in glow_series} == set(specs)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    clipped, boxes, sizes = [], [], []
    for artist in fig.findobj(match=matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            extent = artist.get_window_extent(renderer)
            sizes.append(artist.get_fontsize())
            boxes.append(dict(label=artist.get_text(), bounds=list(extent.bounds)))
            if extent.x0 < -.5 or extent.y0 < -.5 or extent.x1 > width + .5 or extent.y1 > height + .5:
                clipped.append(artist.get_text())
    assert not clipped, clipped
    assert min(sizes) >= 8
    with (OUT / "fig6_plot_data.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plotted[0]))
        writer.writeheader()
        writer.writerows(plotted)
    save_options = {} if with_caption else {
        "bbox_inches": Bbox.from_bounds(0, .092 * fig.get_size_inches()[1],
                                       fig.get_size_inches()[0], .908 * fig.get_size_inches()[1]),
        "pad_inches": 0,
    }
    fig.savefig(OUT / "fig6.png", dpi=600, metadata={"Software": ""}, **save_options)
    fig.savefig(OUT / "fig6_preview.png", dpi=240, metadata={"Software": ""}, **save_options)
    fig.savefig(OUT / "fig6.svg", metadata={"Creator": "", "Date": None}, **save_options)
    fig.savefig(OUT / "fig6.pdf", metadata={"Creator": "", "Producer": "", "CreationDate": None, "ModDate": None}, **save_options)
    plt.close(fig)
    report = {"plotted_points": len(plotted), "source_values_preserved": True,
              "physical_size_mm": [170, 166 if with_caption else 166 * .908],
              "canvas_size_mm": [170, 166], "standalone_caption_in_artwork": with_caption,
              "font": sans, "minimum_font_pt": min(sizes),
              "maximum_point_difference_from_previous": max(differences),
              "paysim_seed_rows_validated": 660, "capacity_seed_rows_validated": 60,
              "clipped_text": clipped, "manuscript_modified": False,
              "curve_intervals_added": False, "summary_intervals": "saved conditional pointwise 95% intervals",
              "curve_glow": "decorative fixed-print-width strokes, independent of axis values",
              "glow_strokes_pt_alpha": [[8.5, .055], [5.7, .075], [3.4, .055]],
              "small_panel_glow_strokes_pt_alpha": [[5.5, .045], [3.8, .065], [2.2, .045]],
              "all_six_line_panels_have_glow": True, "glow_series_count": len(glow_series),
              "glow_series": glow_series,
              "summary_uses_separate_x_scales": True, "text_boxes": boxes,
              "forest_axis_rectangles": forest_rects,
              "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.iterdir()) if p.is_file()}}
    (OUT / "fig6_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Independent figure reproduced: 111 verified points, saved summary intervals, all 24 curves with glow; no clipped text.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="Directory for the figure and validation outputs (default: beside this script).")
    parser.add_argument("--data-dir", type=Path, default=DATA,
                        help="Directory containing the frozen Figure 6 input snapshots.")
    parser.add_argument("--with-caption", action="store_true",
                        help="Include the approved standalone footer; default matches manuscript artwork.")
    options = parser.parse_args()
    OUT, DATA = options.output_dir.resolve(), options.data_dir.resolve()
    main(with_caption=options.with_caption)
