"""Earlier-study chronology from the preserved manuscript, without new analyses.

Run with an existing matplotlib installation. Only this script's directory is
written. Icons represent protocol structure, not measured data or model output.
"""
from typing import Any, cast
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent
WIDTH = 6.6
HEIGHT = 3819049 / 914400
YMAX = HEIGHT / WIDTH * 100
C = dict(ink='#243D49', muted='#617580', blue='#B0CEDD', pale='#EEF5F9',
         border='#8EA8B3', green='#EAF2ED', greenink='#4E7865',
         gold='#FAEBC7', goldborder='#D9B66E', white='#FFFFFF')
plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.unicode_minus': False})
fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor='white')
ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 100), ylim=(0, YMAX))
ax.axis('off')
CHECKS: list[Any] = []


def panel(x, y, w, h, fill, edge=None, radius=0.45, lw=0.65, dashed=False):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=fill, edgecolor=edge or C['border'], linewidth=lw,
        linestyle=(0, (3, 2)) if dashed else '-')
    ax.add_patch(patch)
    return (x, y, w, h)


def text(x, y, value, size=8.1, weight='normal', color=None, ha='center',
         va='center', region=None, rotation=0):
    artist = ax.text(x, y, value, fontsize=size, fontweight=weight,
        color=color or C['ink'], ha=ha, va=va, linespacing=1.18,
        rotation=rotation)
    CHECKS.append((artist, region))
    return artist


def arrow(x1, y1, x2, y2, color=None, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle='-|>', mutation_scale=7, linewidth=0.8,
        color=color or C['muted'], linestyle=(0, (3, 2)) if dashed else '-'))


def timeline(x, y, width=17.4):
    widths = [width * .48, width * .22, width * .30]
    colors = ['#8CB4C9', '#BFD6E1', '#DFEBF1']
    labels = ['Train', 'Val', 'Test']
    start = x
    for w, color, label in zip(widths, colors, labels):
        ax.add_patch(Rectangle((start, y), w - .25, 2.65,
                     facecolor=color, edgecolor=C['white'], linewidth=.45))
        text(start + (w - .25) / 2, y + 1.32, label, size=7.35)
        start += w
    arrow(x, y - 1.25, x + width, y - 1.25)


def history_glyph(x, y):
    for j, width in enumerate([4.0, 6.4, 8.8]):
        ax.add_patch(Rectangle((x + 8.8 - width, y + j * 1.6), width, 1.05,
            facecolor=C['blue'], edgecolor=C['border'], linewidth=.45))
    ax.plot([x + 10, x + 10], [y - .3, y + 5], color=C['muted'], lw=.65,
            linestyle=(0, (2, 2)))
    text(x + 10, y - 1.45, 't', size=7.2)


def document_glyph(x, y):
    for j in [2, 1, 0]:
        ax.add_patch(Rectangle((x + j * .7, y + j * .6), 7.6, 5.7,
            facecolor=C['white'], edgecolor=C['border'], linewidth=.65))
    for j in range(3):
        ax.plot([x + 1.2, x + 6.4], [y + 1.4 + j * 1.2] * 2,
                color=C['border'], lw=.65)


def block_glyph(x, y):
    # Schematic retained resampling blocks; no numerical draw values.
    for j in range(8):
        ax.add_patch(Rectangle((x + j * 1.2, y), 1, 2.0,
            facecolor=C['blue'] if j in [0, 1, 4, 5, 6] else C['white'],
            edgecolor=C['border'], linewidth=.45))
    ax.plot([x, x + 9.4], [y - .9] * 2, color=C['muted'], lw=.65)


text(.8, YMAX - 2.1, 'Earlier studies and analysis chronology', size=10.4,
     weight='bold', ha='left')
text(99, YMAX - 2.15, 'Separate designs', size=7.5, color=C['muted'], ha='right')

panel(.7, 54.7, 59.7, 4, C['blue'], radius=0)
panel(64.1, 54.7, 35.2, 4, '#BED5C8', radius=0)
text(30.55, 56.7, 'Temporal design and fixed references', size=8.8, weight='bold')
text(81.7, 56.7, 'Evaluation and analysis roles', size=8.8, weight='bold')

# Lane 1: the two preserved original workflows, followed by post-hoc contrasts.
panel(.7, 37.6, 98.6, 16.1, C['pale'], edge='#D0E0E8')
panel(.7, 37.6, 7.1, 16.1, '#D1E3ED', edge='#D0E0E8')
text(4.25, 45.65, 'PaySim / IEEE-CIS', size=8.5, weight='bold', rotation=90)
r = panel(9.3, 38.6, 20.1, 14.1, C['white'])
text(19.35, 50.5, 'Original time splits', size=8.2, weight='bold', region=r)
timeline(10.6, 45.4, 17.5)
text(19.35, 41.5, 'Three future windows\nIntact time groups', size=7.8, region=r)
arrow(29.5, 45.65, 32, 45.65)
r = panel(32.6, 38.6, 27.8, 14.1, C['gold'], edge=C['goldborder'])
text(46.5, 50.5, 'Fixed LightGBM studies', size=8.2, weight='bold', region=r)
history_glyph(34.5, 44.0)
text(53.0, 45.5, 'Strictly\npast history', size=7.8, region=r)
text(46.5, 40.65, 'Train-only fit · five paired seeds', size=7.4, region=r)
arrow(60.7, 45.65, 63.3, 45.65)
r = panel(64.1, 38.6, 34.5, 14.1, C['green'], edge='#A8BFB1')
text(81.35, 50.7, 'Original future-window evaluation', size=8.05,
     weight='bold', color=C['greenink'], region=r)
text(81.35, 47.6, 'PaySim: transferred Val threshold\nIEEE-CIS: $k=\\lceil0.03N\\rceil$', size=7.6, region=r)
arrow(81.35, 45.3, 81.35, 43.8, color=C['greenink'])
text(81.35, 41.5, 'Later post-hoc component /\nLightGBM–logistic contrasts', size=7.8, region=r)

# Lane 2: the earlier BankSim validation, explicitly a separate local freeze.
panel(.7, 22.0, 98.6, 14.6, C['pale'], edge='#D0E0E8')
panel(.7, 22.0, 7.1, 14.6, '#D1E3ED', edge='#D0E0E8')
text(4.25, 29.3, 'BankSim', size=8.6, weight='bold', rotation=90)
r = panel(9.3, 23.0, 20.1, 12.6, C['white'])
text(19.35, 33.3, 'Separate benchmark', size=8.2, weight='bold', region=r)
timeline(10.6, 28.35, 17.5)
text(19.35, 25.0, 'External simulated input', size=7.6, region=r)
arrow(29.5, 29.3, 32, 29.3)
r = panel(32.6, 23.0, 27.8, 12.6, C['gold'], edge=C['goldborder'])
text(46.5, 33.3, 'Local protocol freeze', size=8.2, weight='bold', region=r)
text(46.5, 30.1, 'Before fitting and Test access', size=7.8, region=r)
arrow(46.5, 28.9, 46.5, 27.5)
text(46.5, 25.7, 'B1-selected configurations', size=7.9, region=r)
arrow(60.7, 29.3, 63.3, 29.3)
r = panel(64.1, 23.0, 34.5, 12.6, C['green'], edge='#A8BFB1')
text(81.35, 33.3, 'Fixed-model reference comparisons', size=8.05,
     weight='bold', color=C['greenink'], region=r)
text(81.35, 29.8, 'B0 / B0H / B1 / B1H', size=8.3, region=r)
text(81.35, 25.9, 'AP and whole-window capture\nConditional on the selected configurations',
     size=7.55, region=r)

# Lane 3: diagnostics reuse known outputs; no fit-free label for later new fits.
panel(.7, 7.0, 98.6, 14.0, '#F4F7F6', edge='#D1DDD7')
panel(.7, 7.0, 7.1, 14.0, '#D5E4DD', edge='#D1DDD7')
text(4.25, 14.0, 'Revision\ndiagnostics', size=8.0, weight='bold', rotation=90,
     region=(.7, 7.0, 7.1, 14.0))
r = panel(9.3, 8.0, 20.1, 12.0, C['white'])
text(19.35, 18.0, 'Results already known', size=8.0, weight='bold', region=r)
document_glyph(11.0, 9.7)
text(24.25, 12.8, 'Retained\noutputs', size=7.6, region=r)
arrow(29.5, 14.0, 32, 14.0)
r = panel(32.6, 8.0, 27.8, 12.0, C['white'])
text(46.5, 18.0, 'Existing outputs and saved draws', size=7.8,
     weight='bold', region=r)
block_glyph(41.6, 13.0)
text(46.5, 10.3, 'No new fits or bootstrap draws', size=7.6, region=r)
arrow(60.7, 14.0, 63.3, 14.0)
r = panel(64.1, 8.0, 34.5, 12.0, C['green'], edge='#A8BFB1')
text(81.35, 17.8, 'Descriptive diagnostics', size=8.2,
     weight='bold', color=C['greenink'], region=r)
text(81.35, 13.9, 'Capture headroom / exchanges\nExisting draws / time structure', size=8.0, region=r)
text(81.35, 10.3, 'After the original results were known', size=7.5, region=r)

text(50, 4.65, 'Shared boundaries: strictly past history · intact time groups · fixed models for inference',
     size=7.65, color=C['greenink'], weight='bold')
text(50, 2.05, 'Distinct designs; no common prospective protocol. Later six-condition development: §3.6.1.',
     size=7.6, color=C['muted'])

fig.canvas.draw()
renderer = cast(Any, fig.canvas).get_renderer()
violations = []
for artist, region in CHECKS:
    bbox = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
    if bbox.x0 < 0 or bbox.x1 > 100 or bbox.y0 < 0 or bbox.y1 > YMAX:
        violations.append({'text': artist.get_text(), 'reason': 'outside canvas'})
    if region:
        x, y, w, h = region
        if bbox.x0 < x + .3 or bbox.x1 > x + w - .3 or bbox.y0 < y + .3 or bbox.y1 > y + h - .3:
            violations.append({'text': artist.get_text(), 'reason': 'outside parent node'})
if violations:
    raise RuntimeError(json.dumps(violations, ensure_ascii=False))
for suffix in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / f'fig1.{suffix}', dpi=400, facecolor='white')
plt.close(fig)
print('Fig1 saved: exact manuscript extent, 400 dpi PNG, vector PDF/SVG; text bounds passed.')
