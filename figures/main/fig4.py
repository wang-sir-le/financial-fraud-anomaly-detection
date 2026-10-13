"""Five-panel BankSim figure adapted to the user's heatmap/effect layout.

Uses the existing vector helpers and frozen-data workflow from the earlier
four-panel merge. No fitting, predictions, ranking, pooling or resampling.
The SVG uses editable text and vector geometry; no raster is embedded.
Public redraw uses adjacent frozen inputs; external author-source hashes are
provenance only. No manuscript file is required or inspected.
"""

import ast
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Any

from lxml import etree as ET
import pymupdf

HERE = Path(__file__).resolve().parent
STEM = HERE / "fig4"
EFFECTS = HERE / "fig4_effects.json"
EXCHANGE = HERE / "fig4_exchange.csv"
CAPTION = HERE / "fig4_caption.txt"
CAPTION_HASH = "b6fbbc65ceef4bf671453cc05946215c6eb005dfa3d4202d171e55d208b37273"
EFFECTS_HASH = "630b15e40da083d189f68bc78237ec7baf0d686e7adc4e17a58ac80d5b601ca3"
EXCHANGE_HASH = "e58d75633cb281db33b56f38fa98c90a2eb8fee951844d194b20ddf6d3a24ee8"
SVG = "http://www.w3.org/2000/svg"
WIDTH, HEIGHT = 475.2, 304
FAMILIES = ["LightGBM", "XGBoost"]
CONTRASTS = ["H0", "H1"]
COLORS = {"LightGBM": "#4C72B0", "XGBoost": "#DD8452"}
PALETTES = {
    "G": [(0, "#F0F9E8"), (0.5, "#7BCCC4"), (1, "#0868AC")],
    "L": [(0, "#FFF7EC"), (0.5, "#FDBB84"), (1, "#B30000")],
    "Net": [(0, "#B2182B"), (0.5, "#F7F7F7"), (1, "#2166AC")],
}
RANGES = {"G": (0, 220), "L": (0, 20), "Net": (-220, 220)}


def node(parent, tag, **attrs):
    return ET.SubElement(parent, "{" + SVG + "}" + tag, {key.replace("_", "-"): str(value) for key, value in attrs.items()})


def text(parent, x, y, value, size=8.5, color="#111111", anchor="start", bold=False):
    assert size >= 8
    item = node(
        parent,
        "text",
        x=x,
        y=y,
        font_size=size,
        fill=color,
        font_family="Arial, Helvetica, sans-serif",
        text_anchor=anchor,
        font_weight="bold" if bold else "normal",
    )
    item.text = value
    return item


def line(parent, x1, y1, x2, y2, color="#222222", width=0.8, dash=None):
    item = node(parent, "line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, stroke_width=width)
    if dash:
        item.set("stroke-dasharray", dash)
    return item


def number(value, digits=1, sign=False):
    if value == 0:
        return "0.0" if digits else "0"
    return (f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}").replace("-", "−")


def rgb(color):
    return tuple(int(color[index : index + 2], 16) for index in [1, 3, 5])


def color_at(metric, value):
    low, high = RANGES[metric]
    assert low <= value <= high
    fraction = (value - low) / (high - low)
    stops = PALETTES[metric]
    for (start, start_color), (end, end_color) in zip(stops[:-1], stops[1:], strict=True):
        if start <= fraction <= end:
            amount = (fraction - start) / (end - start)
            mixed = [round(a + (b - a) * amount) for a, b in zip(rgb(start_color), rgb(end_color), strict=True)]
            return "#" + "".join(f"{channel:02X}" for channel in mixed)
    raise AssertionError(value)


def foreground(fill):
    linear = []
    for channel in rgb(fill):
        value = channel / 255
        linear.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    luminance = sum(a * b for a, b in zip(linear, [0.2126, 0.7152, 0.0722], strict=True))
    black_ratio = (luminance + 0.05) / 0.05
    white_ratio = 1.05 / (luminance + 0.05)
    return ("#111111" if black_ratio >= white_ratio else "#FFFFFF"), max(black_ratio, white_ratio)


def colorbar(svg, metric, top, height):
    low, high = RANGES[metric]
    # Vector strips render identically in SVG/PDF; small overlaps remove seams.
    for index in range(256):
        value = high - (index + 0.5) * (high - low) / 256
        y = top + index * height / 256
        strip_height = min(height / 256 + 0.25, top + height - y)
        node(svg, "rect", x=185, y=y, width=7, height=strip_height, fill=color_at(metric, value))
    node(svg, "rect", x=185, y=top, width=7, height=height, fill="none", stroke="#333333", stroke_width=0.55)
    text(svg, 188.5, top - 4, "ΔTP" if metric == "Net" else metric, size=8, anchor="middle")
    ticks = [low, (low + high) / 2, high]
    for value in ticks:
        y = top + (high - value) * height / (high - low)
        line(svg, 192, y, 194, y, "#333333", 0.55)
        text(svg, 197, y + 2.5, number(value, 0), size=8)


def heatmap(svg, exchange, metric, title_y, panel, title, recorded):
    text(svg, 7, title_y, f"{panel}  {title}", size=9.3, bold=True)
    top, cell_width, cell_height = title_y + 17, 55, 18
    for column, contrast in enumerate(CONTRASTS):
        text(svg, 67 + (column + 0.5) * cell_width, top - 4, contrast, size=8.5, anchor="middle", bold=True)
    for row_index, family in enumerate(FAMILIES):
        text(svg, 60, top + (row_index + 0.5) * cell_height + 3, family, size=8.5, anchor="end")
        for column, contrast in enumerate(CONTRASTS):
            row = next(r for r in exchange if r["family"] == family and r["contrast"] == contrast)
            value = float(row["delta_tp"]) if metric == "Net" else float(row[metric])
            fill = color_at(metric, value)
            label_color, ratio = foreground(fill)
            x, y = 67 + column * cell_width, top + row_index * cell_height
            group = node(
                svg, "g", data_panel=panel, data_metric=metric, data_family=family, data_contrast=contrast, data_value=value, data_fill=fill
            )
            node(group, "rect", x=x, y=y, width=cell_width, height=cell_height, fill=fill, stroke="#FFFFFF", stroke_width=0.65)
            label = number(value, sign=metric == "Net")
            text(group, x + cell_width / 2, y + cell_height / 2 + 3, label, size=9.5, anchor="middle", color=label_color, bold=True)
            recorded.append(
                {
                    "panel": panel,
                    "metric": metric,
                    "family": family,
                    "contrast": contrast,
                    "value": value,
                    "label": label,
                    "cell_xy": [x, y],
                    "fill": fill,
                    "label_contrast_ratio": ratio,
                }
            )
    node(svg, "rect", x=67, y=top, width=110, height=36, fill="none", stroke="#333333", stroke_width=0.7)
    colorbar(svg, metric, top, 36)


def axis(svg, left, right, top, bottom, low, high, ticks):
    def scale(value):
        return left + (value - low) * (right - left) / (high - low)

    line(svg, left, top, left, bottom)
    line(svg, left, bottom, right, bottom)
    line(svg, scale(0), top, scale(0), bottom, "#808B95", 0.85, "3 2")
    for tick in ticks:
        x = scale(tick)
        line(svg, x, bottom, x, bottom + 3)
        text(svg, x, bottom + 13, number(tick, 0), size=8, anchor="middle")
    return scale


def effect_point(svg, row, y, transform, panel, recorded):
    locations = [transform(row[key]) for key in ["lower", "estimate", "upper"]]
    low, center, high = locations
    assert low <= center <= high
    color = COLORS[row["family"]]
    group = node(
        svg,
        "g",
        data_panel=panel,
        data_family=row["family"],
        data_effect=row["effect"],
        data_estimate=row["estimate"],
        data_lower=row["lower"],
        data_upper=row["upper"],
        data_confidence=row["confidence"],
    )
    line(group, low, y, high, y, color, 1.3)
    for endpoint in [low, high]:
        line(group, endpoint, y - 2.5, endpoint, y + 2.5, color, 0.9)
    node(group, "circle", cx=center, cy=y, r=2.3, fill=color)
    recorded.append({"panel": panel, **row, "svg_x": locations, "svg_y": y})


def build(effects, exchange):
    svg = ET.Element(
        "{" + SVG + "}svg", nsmap={None: SVG}, width=f"{WIDTH}pt", height=f"{HEIGHT}pt", viewBox=f"0 0 {WIDTH} {HEIGHT}", version="1.1"
    )
    node(svg, "title").text = "BankSim history capture effects and exchanges"
    node(
        svg, "desc"
    ).text = "Three left heatmaps show frozen newly captured, lost and net fraud counts for two models and H0/H1; right plots show six original paired effects and a magnified repeat of H1. Captured-fraud counts are not cost or monetary units. Heatmap colours are descriptive; observed arithmetic is not uncertainty."
    node(svg, "rect", x=0, y=0, width=WIDTH, height=HEIGHT, fill="white")
    text(svg, 7, 14, "BankSim history capture effects and exchanges", size=11.5, bold=True)
    text(svg, 7, 28, "Exact 3% completed-window budget (K = 5,950) · five-seed paired means", size=8.5, color="#555555")
    line(svg, 223, 37, 223, 263, "#DFE4E8", 0.6)
    heat_cells: list[dict[str, Any]] = []
    heatmap(svg, exchange, "G", 47, "a", "Newly captured frauds (G)", heat_cells)
    heatmap(svg, exchange, "L", 119, "b", "Lost frauds (L)", heat_cells)
    heatmap(svg, exchange, "Net", 191, "c", "Net capture change (G − L)", heat_cells)
    for y in [112, 184]:
        line(svg, 7, y, 213, y, "#E0E4E8", 0.5)
    text(svg, 7, 255, "a–c: descriptive paired means.", size=8, color="#555555")
    text(svg, 7, 266, "G and L have no separate intervals.", size=8, color="#555555")
    text(svg, 238, 47, "d  Paired net capture effects", size=9.3, bold=True)
    for x, family in zip([249, 364], FAMILIES, strict=True):
        line(svg, x, 61, x + 13, 61, COLORS[family], 1.4)
        node(svg, "circle", cx=x + 6.5, cy=61, r=2, fill=COLORS[family])
        text(svg, x + 18, 64, family, size=8.5)
    common = axis(svg, 266, 468, 74, 145, -280, 280, [-250, -125, 0, 125, 250])
    paired: list[dict[str, Any]] = []
    for y, effect in zip([84, 109, 134], ["H0", "H1", "D"], strict=True):
        text(svg, 258, y + 3, effect, size=8.5, bold=True, anchor="end")
        for index, family in enumerate(FAMILIES):
            row = next(r for r in effects["points"] if r["effect"] == effect and r["family"] == family)
            point_y = y + (-4 if index == 0 else 4)
            effect_point(svg, row, point_y, common, "d", paired)
            if effect == "D":
                text(svg, 313, point_y + 3, number(row["estimate"], sign=True), size=8, color=COLORS[family])
    text(svg, 367, 172, "Captured-fraud difference (ΔTP)", size=8.5, anchor="middle")
    text(svg, 238, 184, "H0: 95% pointwise; H1/D: 98.75% Bonferroni", size=8, color="#555555")
    text(svg, 238, 202, "e  H1 enlarged (same ΔTP intervals)", size=9.3, bold=True)
    zoom = axis(svg, 266, 468, 215, 251, -14, 23, [-10, 0, 10, 20])
    for y, family in zip([225, 244], FAMILIES, strict=True):
        row = next(r for r in effects["points"] if r["effect"] == "H1" and r["family"] == family)
        effect_point(svg, row, y, zoom, "e", paired)
        text(
            svg,
            zoom(row["estimate"]),
            y - 8,
            number(row["estimate"], sign=True),
            size=8.5,
            color=COLORS[family],
            bold=True,
            anchor="middle",
        )
    node(svg, "rect", x=7, y=273, width=461, height=29, rx=3, fill="#F3F5F7")
    text(svg, 13, 284, "Observed-point arithmetic", size=8, bold=True)
    text(svg, 461, 284, "Not confidence limits", size=8, anchor="end", color="#555555")
    headroom = effects["arithmetic_footer"]["B1_headroom"]
    bounds = effects["arithmetic_footer"]["D_upper_bound"]
    text(
        svg,
        13,
        297,
        f"LightGBM / XGBoost: B1 space {headroom[0]:.1f} / {headroom[1]:.1f}; D ≤ {number(bounds[0])} / ≤ {number(bounds[1])}",
        size=8.5,
    )
    return svg, heat_cells, paired


def main():
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    raw_effects, raw_exchange = EFFECTS.read_bytes(), EXCHANGE.read_bytes()
    assert sha256(raw_effects).hexdigest() == EFFECTS_HASH
    assert sha256(raw_exchange).hexdigest() == EXCHANGE_HASH
    raw_caption = CAPTION.read_bytes()
    assert sha256(raw_caption).hexdigest() == CAPTION_HASH
    effects = json.loads(raw_effects)
    exchange = list(csv.DictReader(io.StringIO(raw_exchange.decode("utf-8-sig"))))
    assert len(effects["points"]) == 6 and len(exchange) == 4
    # External source hashes are provenance only; redraw reads adjacent inputs.
    for row in exchange:
        assert float(row["K"]) == 5950 and float(row["N"]) == 198311 and float(row["q"]) == 0.03
        gain, loss, net = [float(row[key]) for key in ["G", "L", "delta_tp"]]
        assert loss >= 0 and abs(gain - loss - net) < 1e-10
        effect = next(r for r in effects["points"] if r["effect"] == row["contrast"] and r["family"] == row["family"])
        assert effect["estimate"] == net
    for family in FAMILIES:
        selected = {r["effect"]: r for r in effects["points"] if r["family"] == family}
        assert abs(selected["H1"]["estimate"] - selected["H0"]["estimate"] - selected["D"]["estimate"]) < 1e-10
    svg, heat_cells, paired = build(effects, exchange)
    assert len(heat_cells) == 12 and len(paired) == 8
    assert len([r for r in paired if r["panel"] == "d"]) == 6
    assert len([r for r in paired if r["panel"] == "e"]) == 2
    assert all(r["confidence"] == (0.95 if r["effect"] == "H0" else 0.9875) for r in paired)
    assert all(r["estimate"] < 0 and r["upper"] < 0 for r in paired if r["effect"] == "D")
    assert all(r["label_contrast_ratio"] >= 4.5 for r in heat_cells)
    ns = {"s": SVG}
    assert not svg.findall(".//s:image", ns)
    svg_bytes = ET.tostring(svg, encoding="UTF-8", xml_declaration=True, pretty_print=True)
    STEM.with_suffix(".svg").write_bytes(svg_bytes)
    with pymupdf.open("svg", svg_bytes) as document:
        pdf_bytes = document.convert_to_pdf()
    STEM.with_suffix(".pdf").write_bytes(pdf_bytes)
    with pymupdf.open("pdf", pdf_bytes) as document:
        page = document[0]
        for block in page.get_text("dict")["blocks"]:
            for entry in block.get("lines", []):
                for span in entry["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    assert x0 >= -1 and y0 >= -1 and x1 <= WIDTH + 1 and y1 <= HEIGHT + 1, span["text"]
        output = page.get_pixmap(matrix=pymupdf.Matrix(450 / 72, 450 / 72), alpha=False)
        output.set_dpi(450, 450)
        output.save(str(STEM.with_suffix(".png")))
        page.get_pixmap(matrix=pymupdf.Matrix(2.8, 2.8), alpha=False).save(str(HERE / "fig4_preview.png"))
    caption = raw_caption.decode("utf-8-sig").strip()
    verification = {
        "status": "NUMERIC_AND_GEOMETRY_PASS",
        "source_sha256": {"effects": EFFECTS_HASH, "exchange": EXCHANGE_HASH},
        "heatmap_cells": heat_cells,
        "effect_intervals": paired,
        "heatmap_ranges": RANGES,
        "unrounded_interval_endpoints_preserved": True,
        "negative_D_preserved": True,
        "zero_L_preserved": True,
        "H0_H1_values_match_between_panels": True,
        "shared_model_colors": COLORS,
        "no_new_fitting_or_resampling": True,
        "captured_fraud_counts_not_cost_units": True,
        "no_pooled_datasets": True,
        "minimum_font_pt": 8,
        "width_in": WIDTH / 72,
        "height_in": HEIGHT / 72,
        "png_dpi": 450,
        "svg_embedded_images": 0,
        "svg_editable_text_count": len(svg.findall(".//s:text", ns)),
        "caption": caption,
        "external_source_provenance": effects["sources"],
        "visual_review": "PENDING",
    }
    (HERE / "fig4_validation.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Five panels exported: 12 frozen heatmap cells and 8 original effect intervals preserved.")


if __name__ == "__main__":
    main()
