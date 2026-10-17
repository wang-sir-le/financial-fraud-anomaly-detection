"""Editable, evidence-aligned SVG based on the supplied four-panel template.

Miniature matrices, timeline, calibration reference and block glyphs are
conceptual schematics. They contain no experiment measurements or sample rows.
Dataset counts and protocol labels come from the 2026-10-11 manuscript.
Run with Python with Pillow, lxml and PyMuPDF installed; no network is used.
Outputs are written beside this script. Set FIGURE_FONT_DIR to use an installed
Arial, Liberation Sans or DejaVu Sans regular/bold TTF pair. Font files are not
distributed. Width checks remain mandatory with every selected font family.
"""

from __future__ import annotations

import json
import math
import os
from html import escape
from pathlib import Path
from typing import TypedDict

import pymupdf
from lxml import etree as ET
from PIL import ImageFont

OUT = Path(__file__).resolve().parent
STEM = "fig2"
W, H = 1920, 1120
C = {
    "ink": "#142330",
    "muted": "#526474",
    "blue": "#2384c4",
    "header": "#48a3d0",
    "line": "#c8ddea",
    "pale": "#f5faff",
    "blue_pale": "#e4f2fb",
    "green": "#669b46",
    "orange": "#ec8128",
    "purple": "#7965ad",
    "history": "#fae9d5",
    "red": "#d96b5e",
}
parts: list[str] = []


class TextAudit(TypedDict):
    text: str
    x: float
    y: float
    size: int
    width_px: float
    max_width_px: float | None


text_audit: list[TextAudit] = []


class FontSelection(TypedDict):
    family: str
    regular: Path
    bold: Path
    source: str


def font_directories() -> list[tuple[Path, str]]:
    """Find installed fonts; never download or redistribute font files."""
    configured = os.environ.get("FIGURE_FONT_DIR")
    if configured:
        directory = Path(configured).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"FIGURE_FONT_DIR is not a directory: {directory}")
        return [(directory, "FIGURE_FONT_DIR")]
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    user_data = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return [
        (windows_fonts, "Windows installed fonts"),
        (Path("/Library/Fonts"), "macOS installed fonts"),
        (Path("/System/Library/Fonts/Supplemental"), "macOS system fonts"),
        (Path.home() / "Library" / "Fonts", "macOS user fonts"),
        (Path("/usr/share/fonts/truetype/msttcorefonts"), "Linux installed fonts"),
        (Path("/usr/share/fonts/truetype/liberation2"), "Linux installed fonts"),
        (Path("/usr/share/fonts/truetype/liberation"), "Linux installed fonts"),
        (Path("/usr/share/fonts/truetype/dejavu"), "Linux installed fonts"),
        (Path("/usr/share/fonts/TTF"), "Linux installed fonts"),
        (Path("/usr/local/share/fonts"), "locally installed fonts"),
        (user_data / "fonts", "user-installed fonts"),
        (Path.home() / ".fonts", "user-installed fonts"),
    ]


def resolve_font_pair() -> FontSelection:
    """Select a readable regular/bold pair, in declared family preference order."""
    directories = font_directories()
    families = [
        ("Arial", ("arial.ttf", "Arial.ttf"), ("arialbd.ttf", "Arial Bold.ttf")),
        ("Liberation Sans", ("LiberationSans-Regular.ttf",), ("LiberationSans-Bold.ttf",)),
        ("DejaVu Sans", ("DejaVuSans.ttf",), ("DejaVuSans-Bold.ttf",)),
    ]
    unreadable = []
    for family, regular_names, bold_names in families:
        for directory, source in directories:
            regular = next((directory / name for name in regular_names if (directory / name).is_file()), None)
            bold = next((directory / name for name in bold_names if (directory / name).is_file()), None)
            if regular is None or bold is None:
                continue
            try:
                ImageFont.truetype(str(regular), 18)
                ImageFont.truetype(str(bold), 18)
            except OSError as error:
                unreadable.append(f"{family}: {error}")
                continue
            return {"family": family, "regular": regular, "bold": bold, "source": source}
    searched = ", ".join(str(directory) for directory, _ in directories)
    details = "; unreadable pairs: " + "; ".join(unreadable) if unreadable else ""
    raise FileNotFoundError(
        "No readable regular/bold TTF pair found. Set FIGURE_FONT_DIR to a directory "
        "containing Arial (arial.ttf/arialbd.ttf), Liberation Sans "
        "(LiberationSans-Regular.ttf/LiberationSans-Bold.ttf), or DejaVu Sans "
        "(DejaVuSans.ttf/DejaVuSans-Bold.ttf). No font files are bundled. "
        f"Searched: {searched}{details}"
    )


FONT_SELECTION = resolve_font_pair()


def emit(tag: str, attrs: dict[str, object], body: str = "") -> None:
    att = " ".join(f'{k.replace("_", "-")}="{escape(str(v), quote=True)}"' for k, v in attrs.items())
    parts.append(f"<{tag} {att}>{body}</{tag}>")


def rect(x: float, y: float, w: float, h: float, fill: str = "white", stroke: str = C["line"], radius: float = 3, sw: float = 1.5) -> None:
    emit("rect", {"x": x, "y": y, "width": w, "height": h, "rx": radius, "fill": fill, "stroke": stroke, "stroke_width": sw})


def line(x1: float, y1: float, x2: float, y2: float, color: str = C["blue"], sw: float = 2.2, dash: str = "") -> None:
    a: dict[str, object] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "stroke": color, "stroke_width": sw}
    if dash:
        a["stroke_dasharray"] = dash
    emit("line", a)


def arrow(points: list[tuple[float, float]], color: str = C["blue"], sw: float = 2.8, size: float = 9) -> None:
    emit(
        "polyline",
        {
            "points": " ".join(f"{x},{y}" for x, y in points),
            "fill": "none",
            "stroke": color,
            "stroke_width": sw,
            "stroke_linejoin": "round",
        },
    )
    px, py = points[-2]
    x, y = points[-1]
    ang = math.atan2(y - py, x - px)
    p1 = (x - size * math.cos(ang) + size * 0.55 * math.sin(ang), y - size * math.sin(ang) - size * 0.55 * math.cos(ang))
    p2 = (x - size * math.cos(ang) - size * 0.55 * math.sin(ang), y - size * math.sin(ang) + size * 0.55 * math.cos(ang))
    emit("polygon", {"points": f"{x},{y} {p1[0]},{p1[1]} {p2[0]},{p2[1]}", "fill": color})


def txt(
    x: float,
    y: float,
    value: str,
    size: int = 25,
    bold: bool = False,
    anchor: str = "start",
    color: str = C["ink"],
    maxw: float | None = None,
    rotate: float | None = None,
) -> None:
    fontpath = FONT_SELECTION["bold" if bold else "regular"]
    font = ImageFont.truetype(str(fontpath), size)
    width = font.getlength(value)
    a: dict[str, object] = {
        "x": x,
        "y": y,
        "font_family": "Arial, Helvetica, sans-serif",
        "font_size": size,
        "font_weight": "700" if bold else "400",
        "text_anchor": anchor,
        "fill": color,
    }
    if rotate is not None:
        a["transform"] = f"rotate({rotate} {x} {y})"
    emit("text", a, escape(value))
    text_audit.append({"text": value, "x": x, "y": y, "size": size, "width_px": round(width, 2), "max_width_px": maxw})


def multi(
    x: float,
    y: float,
    values: list[str],
    size: int = 25,
    gap: int = 31,
    bold: bool = False,
    anchor: str = "start",
    color: str = C["ink"],
    maxw: float | None = None,
) -> None:
    for n, val in enumerate(values):
        txt(x, y + n * gap, val, size, bold, anchor, color, maxw)


def panel(x: float, y: float, w: float, h: float, letter: str, title: str) -> None:
    txt(x, y - 12, letter, 34, bold=True)
    rect(x, y, w, h, C["pale"], radius=4)
    rect(x, y, w, 50, C["header"], C["header"], radius=3)
    txt(x + w / 2, y + 35, title, 31, True, "middle", "white", w - 30)


def grid(x: float, y: float, w: float, h: float, cols: int, rows: int, history: bool = False, scores: bool = False) -> None:
    cw, rh = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            fill = "white"
            if history and (r + 2 * c) % 5 == 0:
                fill = C["history"]
            elif scores and c == 0:
                fill = ["#83bde3", "#a3cdea", "#bfddf0", "#d8eaf7", "#edf5fb"][r]
            elif not history and not scores and (r + c) % 4 == 0:
                fill = C["blue_pale"]
            rect(x + c * cw, y + r * rh, cw, rh, fill, "#718393", 0, 0.8)
    if scores:
        txt(x + cw / 2, y - 10, "score", 18, anchor="middle")
        txt(x + cw * 1.5, y - 10, "rank", 18, anchor="middle")


def badge(x: float, y: float, w: float, h: float, title: str, color: str, size: int = 27) -> None:
    rect(x, y, w, h, color, color, 4)
    txt(x + w / 2, y + h / 2 + size * 0.34, title, size, True, "middle", "white", w - 20)


def subcard(x: float, y: float, w: float, h: float, title: str, size: int = 25) -> None:
    rect(x, y, w, h, "white")
    txt(x + 13, y + 32, title, size, True, maxw=w - 26)


def draw() -> None:
    parts.clear()
    text_audit.clear()
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="260mm" height="{260 * H / W}mm" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">'
    )
    parts.append('<title id="title">Experimental framework: methods, datasets, metrics and evaluation design</title>')
    parts.append(
        '<desc id="desc">Four-panel research overview aligned with the 2026-10-11 manuscript. All miniature graphics are schematics, not measured results. PaySim, IEEE-CIS and BankSim use dataset-specific protocols.</desc>'
    )
    rect(0, 0, W, H, "white", "white", 0)
    panel(24, 44, 1872, 470, "a", "Methods and decision workflow")
    panel(24, 570, 564, 536, "b", "Datasets")
    panel(604, 570, 600, 536, "c", "Metrics")
    panel(1220, 570, 676, 536, "d", "Evaluation design")

    # A: the two branches are feature/model construction and score-based decisions.
    txt(414, 129, "Features & supervised learning", 29, True, "middle", maxw=735)
    txt(971, 129, "Datasets", 28, True, "middle")
    txt(1501, 129, "Scores & review decisions", 29, True, "middle", maxw=735)
    rect(44, 151, 746, 304, "white", C["line"], 4)
    rect(1135, 151, 741, 304, "white", C["line"], 4)
    multi(139, 176, ["Transaction", "features"], 22, 27, True, "middle", maxw=170)
    multi(322, 176, ["Past-only", "history"], 22, 27, True, "middle", maxw=170)
    grid(76, 219, 126, 105, 5, 5)
    grid(259, 219, 126, 105, 5, 5, history=True)
    txt(230, 282, "+", 36, True, "middle", C["blue"])
    arrow([(397, 269), (431, 269)])
    rect(441, 225, 148, 88, C["blue_pale"], C["blue"], 4)
    multi(515, 260, ["Supervised", "model"], 25, 31, True, "middle", maxw=137)
    arrow([(600, 269), (623, 269)])
    rect(632, 225, 127, 88, C["blue_pale"], C["blue"], 4)
    multi(695.5, 260, ["Risk", "score"], 26, 31, True, "middle")
    txt(600, 340, "Training labels used for fitting", 21, anchor="middle", color=C["muted"], maxw=366)
    multi(417, 395, ["PaySim / IEEE-CIS: LightGBM, logistic regression", "BankSim: LightGBM, XGBoost"], 24, 33, True, "middle", maxw=720)
    for yy, name, color in [(194, "PaySim", C["green"]), (252, "IEEE-CIS", C["orange"]), (310, "BankSim", C["purple"])]:
        badge(846, yy, 252, 40, name, color, 25)
        line(846, yy + 20, 823, yy + 20)
    line(823, 214, 823, 330)
    arrow([(823, 269), (809, 269), (809, 153), (57, 153), (57, 269), (76, 269)])
    arrow([(759, 269), (791, 269), (791, 376), (1178, 376), (1178, 324)])
    rect(904, 360, 162, 30, C["pale"], C["pale"], 0)
    txt(985, 382, "Model scores", 23, True, "middle", C["blue"])
    grid(1153, 219, 126, 105, 2, 5, scores=True)
    arrow([(1290, 269), (1316, 269)])
    rect(1326, 225, 151, 88, C["blue_pale"], C["blue"], 4)
    multi(1401.5, 260, ["Review", "selection"], 26, 31, True, "middle")
    arrow([(1488, 269), (1515, 269)])
    rect(1525, 225, 131, 88, "white", C["blue"], 4)
    multi(1590.5, 253, ["Review", "list"], 25, 30, True, "middle", maxw=125)
    line(1547, 298, 1634, 298, "#89badb", 3)
    arrow([(1667, 269), (1694, 269)])
    rect(1704, 225, 151, 88, C["blue_pale"], C["blue"], 4)
    multi(1779.5, 259, ["TP / FP / FN", "and metrics"], 23, 30, True, "middle", maxw=146)
    txt(1779, 340, "Held-out labels", 21, anchor="middle", color=C["muted"])
    multi(1505, 395, ["Threshold transfer or exact top-k", "Decision rule specified per dataset"], 25, 33, True, "middle", maxw=710)
    txt(
        984,
        482,
        "Past-only history is constructed without fraud labels; all miniature graphics are schematics.",
        23,
        anchor="middle",
        color=C["muted"],
        maxw=1770,
    )

    # B: source dataset sizes, not model-development sample counts.
    dataset_rows = [
        (651, "PaySim", C["green"], ["6,362,620 synthetic", "mobile-money transactions", "Recipient history"]),
        (766, "IEEE-CIS", C["orange"], ["590,540 labeled", "transactions", "Card-related identifier history"]),
        (881, "BankSim", C["purple"], ["594,643 simulated", "transactions", "Merchant history"]),
    ]
    for yy, name, color, labels in dataset_rows:
        badge(44, yy, 208, 64, name, color, 29)
        multi(272, yy + 19, labels[:2], 23, 30, maxw=295)
        txt(272, yy + 81, labels[2], 21, color=C["muted"], maxw=295)
    rect(43, 1003, 526, 83, C["blue_pale"])
    multi(58, 1036, ["Three dataset-specific evidence settings", "Source dataset sizes shown above"], 24, 30, color=C["ink"], maxw=494)

    # C: ranking is distinct from probability reliability and review utility.
    subcard(624, 640, 560, 212, "Ranking & probability quality", 27)
    txt(643, 721, "AP (primary ranking)", 25, True, maxw=285)
    txt(643, 767, "Brier · log loss · ECE", 23, True, maxw=280)
    multi(643, 799, ["Calibration analysis:", "PaySim only"], 22, 28, color=C["muted"], maxw=280)
    # Ideal-calibration diagonal only: no fictitious observed curve or bins.
    gx, gy, gw, gh = 980, 699, 179, 98
    line(gx, gy + gh, gx + gw, gy + gh, C["muted"], 1.5)
    line(gx, gy + gh, gx, gy, C["muted"], 1.5)
    line(gx, gy + gh, gx + gw, gy, C["blue"], 2)
    txt(gx + 75, gy + 20, "Ideal", 21, color=C["blue"])
    txt(gx + gw / 2, gy + gh + 30, "Predicted risk", 20, anchor="middle", maxw=190)
    txt(gx - 16, gy + gh / 2, "Observed rate", 20, anchor="middle", rotate=-90)
    subcard(624, 870, 560, 216, "Review utility & analytical cost", 26)
    multi(643, 941, ["TP · Recall", "Precision", "Actual alert rate"], 24, 32, True, maxw=245)
    txt(1023, 935, "Top-k review list", 23, True, "middle", maxw=285)
    for i in range(10):
        rect(901 + i * 26, 956, 26, 35, "#a4d2ee" if i < 3 else "#edf5fa", "#5a9bc7", 0, 1)
    line(901, 948, 979, 948, C["blue"], 2)
    line(901, 943, 901, 950)
    line(979, 943, 979, 950)
    multi(643, 1040, ["Analytical cost = FP + c × FN", "Primary cost setting: c = 100 (PaySim / IEEE-CIS)"], 22, 29, maxw=524)

    # D: scope-aware protocols, no artificial increasing performance bars.
    subcard(1240, 640, 308, 202, "Strictly past history", 25)
    x0, x2, ty = 1267, 1517, 728
    arrow([(x0, ty), (x2, ty)], C["muted"], 2, 7)
    for xx, color in [(1300, C["blue"]), (1389, C["orange"]), (1464, "#9ba7b1")]:
        emit("circle", {"cx": xx, "cy": ty, "r": 6, "fill": color})
    txt(1300, 707, "Past", 21, anchor="middle", color=C["blue"])
    txt(1389, 707, "Current", 21, anchor="middle", color=C["orange"])
    txt(1464, 707, "Future", 21, anchor="middle", color=C["muted"])
    multi(1254, 774, ["Earlier time groups only", "Exclude the current group", "No fraud-label history"], 22, 27, maxw=282)
    subcard(1562, 640, 314, 202, "Temporal validation", 25)
    txt(1576, 699, "PaySim / IEEE-CIS", 23, True, maxw=283)
    for r in range(3):
        yy = 718 + r * 23
        txt(1577, yy + 14, str(r + 1), 17, color=C["muted"])
        rect(1600, yy, 83 + 22 * r, 16, "#c9e2f3", "#84b5d7", 0, 0.8)
        rect(1683 + 22 * r, yy, 24, 16, "#e5d7b6", "#bca779", 0, 0.8)
        rect(1707 + 22 * r, yy, 44, 16, "#7fb2d9", "#3a82bb", 0, 0.8)
    for lx, label, fill in [(1580, "Train", "#c9e2f3"), (1666, "Val", "#e5d7b6"), (1741, "Test", "#7fb2d9")]:
        rect(lx, 791, 13, 13, fill, "#9bafbd", 0, 0.7)
        txt(lx + 18, 804, label, 18, color=C["muted"])
    txt(1576, 831, "BankSim: one test window", 22, maxw=285)
    subcard(1240, 860, 308, 226, "Review decision rules", 25)
    txt(1254, 924, "PaySim threshold rule:", 23, True, maxw=282)
    txt(1254, 953, "Validation threshold → Test", 21, maxw=282)
    txt(1254, 981, "Component audit: exact 3%", 21, color=C["muted"], maxw=282)
    txt(1254, 1022, "IEEE-CIS / BankSim:", 23, True, maxw=282)
    txt(1254, 1052, "Exact top-k (3% primary)", 21, maxw=282)
    subcard(1562, 860, 314, 226, "Uncertainty & sensitivity", 24)
    txt(1576, 926, "Paired block bootstrap", 23, True, maxw=284)
    for r in range(2):
        for i, cc in enumerate(["#b1d5ed", "#e6d8b8", "#b1d5ed", "#c2b5df", "#e6d8b8"]):
            rect(1580 + i * 53, 944 + r * 20, 46, 13, cc, "#9bafbd", 0, 0.7)
    txt(1576, 1011, "Fitted models held fixed", 22, maxw=284)
    txt(1576, 1042, "Conditional intervals", 22, maxw=284)
    txt(1576, 1071, "Capacity / cost sensitivity", 21, maxw=284)
    parts.append("</svg>")


def save() -> None:
    draw()
    failures = [a for a in text_audit if a["max_width_px"] is not None and float(a["width_px"]) > float(a["max_width_px"])]
    if failures:
        raise ValueError(
            f"Text exceeds allocated widths with {FONT_SELECTION['family']} fonts: " + json.dumps(failures, ensure_ascii=False)
        )
    svg = "\n".join(parts)
    root = ET.fromstring(svg.encode("utf-8"), parser=ET.XMLParser(resolve_entities=False, no_network=True))
    ns = {"s": "http://www.w3.org/2000/svg"}
    if root.findall(".//s:image", ns):
        raise ValueError("Raster image unexpectedly embedded in editable SVG")
    svg_path = OUT / f"{STEM}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    doc = pymupdf.open(stream=svg.encode("utf-8"), filetype="svg")
    pdf_data = doc.convert_to_pdf()
    pdf = pymupdf.open(stream=pdf_data, filetype="pdf")
    pdf.save(OUT / f"{STEM}.pdf", garbage=4, deflate=True)
    # Export at 600 dpi at the final 6.60-inch manuscript width.
    page = pdf[0]
    target_width = 3960
    scale = target_width / page.rect.width
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    pixmap.set_dpi(600, 600)
    pixmap.save(OUT / f"{STEM}.png")
    page.get_pixmap(matrix=pymupdf.Matrix(1920 / page.rect.width, 1920 / page.rect.width), alpha=False).save(OUT / f"{STEM}_preview.png")
    audit = {
        "canvas": [W, H],
        "png_width_px": target_width,
        "manuscript_width_in": 6.60,
        "effective_manuscript_dpi": 600,
        "panel_labels": ["a", "b", "c", "d"],
        "svg_text_elements": len(root.findall(".//s:text", ns)),
        "embedded_images": 0,
        "text_width_checks": "PASS",
        "miniature_graphics": "schematics only",
        "source_manuscript": "Private manuscript snapshot dated 2026-10-11 (not distributed); protocol labels only",
        "text_measurement_fonts": {
            "family": FONT_SELECTION["family"],
            "regular": str(FONT_SELECTION["regular"]),
            "bold": str(FONT_SELECTION["bold"]),
            "source": FONT_SELECTION["source"],
        },
        "svg_font_family": "Arial, Helvetica, sans-serif",
        "text": text_audit,
        "caption": "Experimental framework. (a) Dataset-specific past-only history and supervised models produce scores for review selection and evaluation using held-out labels. (b) Source dataset sizes and entity-history scopes. (c) AP assesses ranking; probability quality is analyzed only for PaySim. TP, recall, precision, actual alert rate, and analytical cost describe review decisions. (d) Dataset-specific temporal validation, threshold transfer or completed-window exact top-k, paired moving-block bootstrap conditional on fixed fitted models, and sensitivity analysis. The original PaySim primary threshold is selected under a Validation alert cap of 3% and transferred to Test, whose alert rate can differ. The PaySim component audit separately uses an exact 3% budget. IEEE-CIS and BankSim use whole-window exact top-k, with 3% as primary. The PaySim/IEEE-CIS sensitivity grids are q = {1, 3, 5, 10}% and c = {10, 50, 100, 500, 1000}; BankSim reports capacities {1, 3, 5}%. The primary analytical-cost condition is FP cost 1 and FN cost 100 for PaySim/IEEE-CIS. Miniature graphics are schematic, not observed results.",
    }
    (OUT / f"{STEM}_validation.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {svg_path}; {audit['svg_text_elements']} editable text elements; no embedded raster images")


if __name__ == "__main__":
    save()
