# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Shared look of the repository's PNG charts (scripts/make_charts.py and the research chart scripts).

Every chart is drawn twice, for GitHub's light and dark themes, on a rounded card. Colors are the validated
categorical palette of the charts (slot 1 blue, 2 orange, 3 aqua; light and dark steps), a neutral gray for a
reference or "other" mark, recessive grid and axes, thin bars with 4 px rounded data ends, values written next
to the marks. Needs matplotlib."""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

THEMES = {
    'light': dict(surface='#fcfcfb', border='#e6e5e0', ink='#0b0b0b', ink2='#52514e', muted='#898781',
                  grid='#e1e0d9', base='#c3c2b7', blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', gray='#b4b2aa'),
    'dark': dict(surface='#1a1a19', border='#2f2f2d', ink='#f0efec', ink2='#c3c2b7', muted='#898781',
                 grid='#2c2c2a', base='#383835', blue='#3987e5', orange='#d95926', aqua='#199e70', gray='#5c5b56'),
}
WIDTH_IN, DPI = 8.0, 200


def setup_fonts():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Inter', 'Helvetica Neue', 'Arial', 'DejaVu Sans']
    plt.rcParams['svg.fonttype'] = 'none'


def figure(t, height_in):
    """A transparent figure with the rounded card drawn on it."""
    fig = plt.figure(figsize=(WIDTH_IN, height_in), dpi=DPI)
    fig.patch.set_alpha(0)
    aspect = WIDTH_IN / height_in
    fig.patches.append(FancyBboxPatch((0.004, 0.004 * aspect), 0.992, 1 - 0.008 * aspect, transform=fig.transFigure,
                                      boxstyle='round,pad=0,rounding_size=0.012', mutation_aspect=aspect,
                                      facecolor=t['surface'], edgecolor=t['border'], linewidth=1.0, zorder=-10))
    return fig


def header(fig, t, title, subtitle, height_in):
    top = 1 - 0.30 / height_in
    fig.text(0.045, top, title, fontsize=15, fontweight='bold', color=t['ink'], va='top')
    if subtitle:
        fig.text(0.045, top - 0.34 / height_in, subtitle, fontsize=9.5, color=t['ink2'], va='top', linespacing=1.45)


def footer(fig, t, text, height_in):
    fig.text(0.045, 0.20 / height_in, text, fontsize=7.5, color=t['muted'], va='bottom', linespacing=1.5)


def axes(fig, t, rect, title=None):
    ax = fig.add_axes(rect)
    ax.set_facecolor('none')
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(t['base'])
    ax.tick_params(colors=t['muted'], labelsize=8.5, length=0)
    ax.tick_params(axis='y', labelcolor=t['ink2'], labelsize=9.5, pad=8)
    ax.grid(axis='x', color=t['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc='left', fontsize=10.5, fontweight='bold', color=t['ink'], pad=10)
    return ax


def hbars(ax, labels, values, colors, t, xmax, fmt, height=0.56, radius_px=4, notes=None):
    """Horizontal bars from zero, top to bottom in the given order, with a 4 px rounded data end and the value
    (and an optional note) written after the bar."""
    n = len(values)
    ax.set_xlim(0, xmax)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    fig = ax.figure
    bbox = ax.get_position()
    wpx, hpx = bbox.width * fig.get_figwidth() * fig.dpi, bbox.height * fig.get_figheight() * fig.dpi
    xpp, ypp = xmax / wpx, n / hpx            # data units per pixel
    r = radius_px * xpp
    for i, (v, c) in enumerate(zip(values, colors)):
        y0 = i - height / 2
        if v > 2 * r:
            ax.add_patch(FancyBboxPatch((0, y0), v, height, boxstyle=f'round,pad=0,rounding_size={r}',
                                        mutation_aspect=ypp / xpp, facecolor=c, edgecolor='none', zorder=3))
            ax.add_patch(Rectangle((0, y0), r * 1.5, height, facecolor=c, edgecolor='none', zorder=3))
        else:
            ax.add_patch(Rectangle((0, y0), v, height, facecolor=c, edgecolor='none', zorder=3))
        mask = dict(facecolor=t['surface'], edgecolor='none', pad=1.5)   # hides a gridline behind the text
        label = ax.text(v + 6 * xpp, i, fmt(v), va='center', ha='left', fontsize=9, color=t['ink'], zorder=4,
                        fontweight='bold', bbox=mask)
        if notes and notes[i]:
            wtext = label.get_window_extent(renderer=fig.canvas.get_renderer()).width
            ax.text(v + (6 + wtext + 7) * xpp, i, notes[i], va='center', ha='left', fontsize=8.5, color=t['ink2'],
                    zorder=4, bbox=mask)


def save(fig, out_dir, name, theme):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{name}-{theme}.png')
    fig.savefig(path, dpi=DPI, transparent=True, metadata={'Software': None})
    plt.close(fig)
    return path
