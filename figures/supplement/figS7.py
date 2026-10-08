"""Redraw existing BankSim aggregate time structure, with no model or statistical replay.

Run using Python with matplotlib installed. Reads only the adjacent aggregate JSON.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

BASE = Path(__file__).resolve().with_suffix('')
DATA = json.loads(BASE.with_suffix('.data.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.labelsize': 9, 'axes.titlesize': 9.5,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'axes.linewidth': .6, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'savefig.facecolor': 'white'})
COLORS = {'LightGBM': '#234E70', 'XGBoost': '#008C95'}
groups = {family: sorted((r for r in DATA['rows'] if r['family'] == family), key=lambda r: r['step'])
          for family in COLORS}
assert len(DATA['rows']) == 108
assert [(r['step'], r['N'], r['F']) for r in groups['LightGBM']] == [
    (r['step'], r['N'], r['F']) for r in groups['XGBoost']]
fig, axes = plt.subplots(2, 2, figsize=(6.6, 4005651 / 914400), sharex=True)
fig.subplots_adjust(left=.105, right=.98, top=.93, bottom=.20, wspace=.32, hspace=.48)
steps = [r['step'] for r in groups['LightGBM']]
axes[0, 0].plot(steps, [r['N'] for r in groups['LightGBM']], color='#234E70', lw=1.35)
axes[0, 0].set_title('(a) Transaction volume', loc='left', fontweight='bold', pad=7)
axes[0, 0].set_ylabel('Transactions')
axes[0, 0].set_ylim(3500, 3830)
axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
axes[0, 1].plot(steps, [r['F'] for r in groups['LightGBM']], color='#627D8C', lw=1.35)
axes[0, 1].set_title('(b) Fraud count', loc='left', fontweight='bold', pad=7)
axes[0, 1].set_ylabel('Frauds')
axes[0, 1].set_ylim(0, 50)
for ax, effect, label in [(axes[1, 0], 'H0', '(c) History above $B_0$'),
                         (axes[1, 1], 'H1', '(d) History above $B_1$')]:
    for family, style in [('LightGBM', '-'), ('XGBoost', '--')]:
        ax.plot(steps, [r[effect] for r in groups[family]], color=COLORS[family],
                linestyle=style, lw=1.2, label=family)
    ax.axhline(0, color='#7D858B', lw=.65, zorder=0)
    ax.set_title(label, loc='left', fontweight='bold', pad=7)
    ax.set_ylabel('Δ captured frauds')
    ax.set_xlabel('Simulation step')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
for ax in axes.flat:
    ax.set_xlim(125, 180)
    ax.set_xticks([126, 140, 153, 166, 179])
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['bottom', 'left']:
        ax.spines[side].set_color('#8D969D')
    ax.tick_params(width=.6, length=3, color='#8D969D')
    ax.grid(axis='y', color='#E4E9EC', lw=.45)
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.54, .065),
           ncol=2, frameon=False, fontsize=8.5, handlelength=2.8)
fig.text(.54, .025, 'Whole-window top-3% selections; no step-wise reranking.',
         ha='center', fontsize=8, color='#505C65')
fig.savefig(BASE.with_suffix('.png'), dpi=400)
fig.savefig(BASE.with_suffix('.pdf'))
plt.close(fig)
print('Saved figS7 PNG and vector PDF from 108 existing aggregate rows.')
