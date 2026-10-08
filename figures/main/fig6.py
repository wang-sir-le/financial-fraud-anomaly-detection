"""Redraw Fig. 6 from saved scenario means, with no smoothing or new intervals."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

HERE = Path(__file__).resolve().parent
DATA = json.loads(HERE.joinpath('fig6.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': .65,
    'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'text.color': '#26343F', 'axes.labelcolor': '#26343F',
    'xtick.color': '#44545E', 'ytick.color': '#44545E'})
COLORS = ['#234E70', '#008C95', '#718B9A']
MARKERS = ['o', 's', '^']
fig, axes = plt.subplots(2, 2, figsize=(DATA['width_in'], DATA['height_in']))
fig.subplots_adjust(left=.11, right=.975, bottom=.13, top=.865, wspace=.40, hspace=.62)
titles = ['(a) IEEE-CIS: M2 − M0', '(b) PaySim: PTHW − P0', '(c) PaySim: q = 3%', '(d) PaySim: workload']
point_count = 0
for ax, key, title in zip(axes.flat, ['a', 'b', 'c', 'd'], titles, strict=True):
    ax.set_axisbelow(True)
    for series in DATA['panels'][key]:
        i = series['fold'] - 1
        assert len(series['x']) == len(series['y'])
        style = '--' if series.get('model') == 'P0' else '-'
        ax.plot(series['x'], series['y'], color=COLORS[i], marker=MARKERS[i], markersize=3.7,
                linewidth=1.15, linestyle=style, markeredgewidth=.55, zorder=3)
        point_count += len(series['x'])
    ax.axhline(0, color='#7C8992', lw=.7, zorder=2)
    ax.set_title(title, loc='left', pad=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#A5B0B8')
    ax.tick_params(length=3, width=.6, pad=3)
    ax.grid(axis='y', color='#E7ECEF', linewidth=.55, zorder=0)
    if key in ['a', 'b', 'd']:
        ax.set_xticks([1, 3, 5, 10])
        ax.set_xlim(.45, 10.55)
assert point_count == DATA['point_count'] == 63
axes[0, 0].set(xlabel='Capacity (%)', ylabel='Recall difference (pp)', ylim=(-1.5, 6.3))
axes[0, 1].set(xlabel='Capacity (%)', ylabel='Recall difference (pp)', ylim=(-3, 65))
axes[0, 1].set_yticks([0, 20, 40, 60])
axes[1, 0].set(xlabel='False-negative cost', ylabel='Cost reduction (%)', xscale='log', ylim=(-8, 69))
axes[1, 0].set_xticks([10, 50, 100, 500, 1000], ['10', '50', '100', '500', '1000'])
axes[1, 0].xaxis.set_minor_locator(NullLocator())
axes[1, 0].set_yticks([0, 20, 40, 60])
axes[1, 1].set(xlabel='Validation cap (%)', ylabel='Test alerts (%)', ylim=(-.15, 4.2))
axes[1, 1].set_yticks([0, 1, 2, 3, 4])
fold_handles = [Line2D([0], [0], color=COLORS[i], marker=MARKERS[i], markersize=4, linewidth=1.1, label=f'Fold {i + 1}') for i in range(3)]
fig.legend(handles=fold_handles, loc='upper center', bbox_to_anchor=(.5, .985), ncol=3, frameon=False, columnspacing=2.3, handlelength=2)
style_handles = [Line2D([0], [0], color='#536573', lw=1.1, ls=ls, label=label) for ls, label in [('-', 'PTHW'), ('--', 'P0')]]
axes[1, 1].legend(handles=style_handles, loc='upper left', frameon=False, handlelength=2, ncol=2, columnspacing=1, borderaxespad=.3)
fig.text(.11, .032, 'Five-seed means; lines connect calculated scenarios. No intervals or continuous optima.', fontsize=8, va='bottom')
for ext in ['png', 'pdf']:
    fig.savefig(HERE / f'fig6.{ext}', dpi=450, facecolor='white')
plt.close(fig)
print('Fig. 6: 63 unchanged scenario-mean points; four original panels; exported PNG/PDF.')
