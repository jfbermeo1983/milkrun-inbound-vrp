# -*- coding: utf-8 -*-
"""
Figures 3, 4 and 5 of the manuscript, as vector PDF and as PNG at 600 dpi.

Design decisions:
  - No dual axes. The two quantities are shown in separate panels and indexed
    to S = 2 -> 100, which is what allows them to be compared on a single
    scale without implying a spurious common unit.
  - The case-study instance is drawn in the accent colour while the seven
    benchmark instances stay light grey. Identity is not carried by colour
    alone: line weight and a direct label distinguish it as well, so the
    figures remain readable in greyscale and to colour-deficient readers.
  - Validated palette (blue #2a78d6 on a light surface).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

ACCENT = '#2a78d6'
GREY = '#b9b8b4'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e3e2df'
MM = 1 / 25.4

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.edgecolor': INK2,
    'axes.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.labelcolor': INK,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})

CASE = 'VW-n40'
S_ORDER = ['2', '3', '4', '5', 'sin limite']
S_LABEL = ['2', '3', '4', '5', 'unrestr.']

raw = pd.read_csv(os.path.join(HERE, 'results', 'results_raw.csv')); raw['S'] = raw.S.astype(str)
ext = pd.read_csv(os.path.join(HERE, 'results', 'results_extra_S_open.csv')); ext['S'] = ext.S.astype(str)
op = raw[raw.outbound == 0]


def style(ax):
    ax.grid(True, axis='y', color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def panel(ax, obj, col, title, ylabel):
    d = op[op.objective == obj]
    for inst, g in d.groupby('instance'):
        g = g.set_index('S')
        if not all(s in g.index for s in S_ORDER):
            continue
        y = [100 * g.loc[s, col] / g.loc['2', col] for s in S_ORDER]
        case = inst == CASE
        ax.plot(range(len(S_ORDER)), y, marker='o',
                markersize=3.6 if case else 2.6,
                linewidth=2.0 if case else 1.0,
                color=ACCENT if case else GREY,
                zorder=3 if case else 2,
                label=CASE if case else None)
        if case:
            ax.annotate(CASE, xy=(len(S_ORDER) - 1, y[-1]), xytext=(-2, 7),
                        textcoords='offset points', ha='right',
                        fontsize=7.5, color=ACCENT, fontweight='bold')
    ax.set_xticks(range(len(S_ORDER))); ax.set_xticklabels(S_LABEL)
    ax.set_xlabel('Maximum stops per route, S')
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', color=INK)
    style(ax)


# ---------------- Figure 3: effect of the stop limit ----------------
fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM))
panel(axes[0], 'emis', 'co2', '(a) CO₂e emissions', 'Index, S = 2 → 100')
panel(axes[1], 'emis', 'distance', '(b) Distance travelled', 'Index, S = 2 → 100')
h = [plt.Line2D([], [], color=ACCENT, lw=2.0, marker='o', ms=3.6),
     plt.Line2D([], [], color=GREY, lw=1.0, marker='o', ms=2.6)]
axes[1].legend(h, ['Case study (VW-n40)', 'Benchmark instances'],
               frameon=False, loc='upper right')
fig.tight_layout()
for e in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'figure3.{e}'), dpi=600, bbox_inches='tight')
plt.close(fig)

# ---------------- Figure 4: marginal benefit of one additional stop -------
# The extended runs (S = 6, 7, 8) are appended to the main sweep so that the
# marginal-improvement curve can be followed past the point where it flattens.
em = pd.concat([op[op.objective == 'emis'], ext], ignore_index=True)
seq = ['2', '3', '4', '5', '6', '7', '8']
fig, ax = plt.subplots(figsize=(120 * MM, 78 * MM))
for inst, g in em.groupby('instance'):
    g = g.set_index('S')
    ss = [s for s in seq if s in g.index]
    if len(ss) < 4:
        continue
    xs, ys = [], []
    for a, b in zip(ss, ss[1:]):
        xs.append(int(a)); ys.append(100 * (g.loc[a, 'co2'] - g.loc[b, 'co2']) / g.loc[a, 'co2'])
    case = inst == CASE
    ax.plot(xs, ys, marker='o', markersize=3.6 if case else 2.6,
            linewidth=2.0 if case else 1.0,
            color=ACCENT if case else GREY, zorder=3 if case else 2)
    if case:
        ax.annotate(CASE, xy=(xs[1], ys[1]), xytext=(10, 6), textcoords='offset points',
                    fontsize=7.5, color=ACCENT, fontweight='bold')
ax.axhline(3.0, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
ax.annotate('3% marginal-improvement criterion', xy=(7.0, 3.0), xytext=(0, 5),
            textcoords='offset points', ha='right', fontsize=7, color=INK2)
ax.annotate('S* = 4', xy=(4, 0.15), xytext=(-4, -18), textcoords='offset points',
            ha='center', fontsize=7.5, color=ACCENT, fontweight='bold',
            arrowprops=dict(arrowstyle='-', color=ACCENT, lw=0.7))
ax.set_xticks(range(2, 8))
ax.set_xticklabels([f'{s}→{s+1}' for s in range(2, 8)])
ax.set_xlabel('Increment in the stop limit')
ax.set_ylabel('Marginal reduction in CO₂e (%)')
ax.set_title('Marginal benefit of one additional stop', loc='left', color=INK)
h = [plt.Line2D([], [], color=ACCENT, lw=2.0, marker='o', ms=3.6),
     plt.Line2D([], [], color=GREY, lw=1.0, marker='o', ms=2.6)]
ax.legend(h, ['Case study (VW-n40)', 'Benchmark instances'], frameon=False, loc='upper right')
style(ax)
fig.tight_layout()
for e in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'figure4.{e}'), dpi=600, bbox_inches='tight')
plt.close(fig)

# ---------------- Figure 5: sensitivity to the consolidation threshold ----
# Values summarise the case-study runs: emission intensity per tonne collected
# and the stop limit that binds at each threshold theta.
theta = [0, 2000, 4000, 6000, 8000, 10000]
kgt = [139.8, 138.6, 136.6, 134.2, 134.3, 129.0]
binding = [4, 3, 3, 2, 2, 2]
fig, ax = plt.subplots(figsize=(120 * MM, 78 * MM))
ax.plot(theta, kgt, marker='o', markersize=4.2, linewidth=2.0, color=ACCENT, zorder=3)
for x, y, b in zip(theta, kgt, binding):
    dx = -14 if x == 6000 else 0
    ax.annotate(f'S={b}', xy=(x, y), xytext=(dx, 9), textcoords='offset points',
                ha='center', fontsize=7, color=INK2)
ax.axvline(6000, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
ax.annotate('current guideline', xy=(6000, 142.6), xytext=(5, -2),
            textcoords='offset points', ha='left', va='top', fontsize=7, color=INK2)
ax.set_xlabel('Consolidation threshold θ (kg)')
ax.set_ylabel('Emission intensity (kg CO₂e per tonne collected)')
ax.set_title('Milk-run intensity and binding stop limit against θ', loc='left', color=INK)
ax.set_xticks(theta)
ax.set_xticklabels([f'{t:,}' for t in theta])
ax.set_ylim(126, 143)
style(ax)
fig.tight_layout()
for e in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'figure5.{e}'), dpi=600, bbox_inches='tight')
plt.close(fig)

print('figuras 3, 4 y 5 generadas en', OUT)
