# SPDX-License-Identifier: Apache-2.0
"""Charts of the WNR4 run (light and dark PNG each) in charts/, with the look of scripts/chartstyle.py.

    python research/wnr4-20261121/charts.py      (matplotlib; not -I when it is in the user site)

Ratios are read from results/pack3-*.json (the speed packs) and results/other/summary.md; the kernel speeds are the
rows "same run (bench9)" of docs/reports/wnr4-research-20261121.md section 5 (results/bench9-*.log); the nvCOMP
ratios are the small-batch points of that report's section 6 (its harness is not in this repository)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'scripts'))
import chartstyle as cs  # noqa: E402

RES = os.path.join(HERE, 'results')
PACKS = [('q38mx', 'Qwen3.8 MXFP4'), ('q38nv', 'Qwen3.8 NVFP4'), ('v41', 'DeepSeek V4.1 MXFP4')]
NVCOMP = {'q38mx': 0.8997, 'q38nv': 0.9372, 'v41': 0.8983}     # report section 6, fast at 8-64 records


def pack_ratios(name):
    d = json.load(open(os.path.join(RES, f'pack3-{name}.json'), encoding='utf-8'))
    wnr3 = d.get('wnr3_same_train') or d.get('wnr3nv_same_train_8192_2048')
    return d['wnr4']['128']['ratio'], wnr3['ratio']


def chart_ratio(t):
    h = 4.2
    fig = cs.figure(t, h)
    cs.header(fig, t, 'WNR4 stores experts in fewer bytes',
              'Space saved on the records of the speed packs (1 - stored/raw, higher is better), every record '
              'decoded back\nand compared. NVFP4: byte-exact, WNR3 is WNR3-NV. V4.1: canonical zero (-0 stored as +0), as in production.', h)
    x0, w, gap = 0.14, 0.25, 0.03
    for k, (name, title) in enumerate(PACKS):
        wnr4, wnr3 = pack_ratios(name)
        ax = cs.axes(fig, t, [x0 + k * (w + gap), 0.26, w, 0.37], title)
        labels = ['WNR4 G=128', 'WNR3', 'nvCOMP ANS']
        cs.hbars(ax, labels, [100 * (1 - r) for r in (wnr4, wnr3, NVCOMP[name])], [t['blue'], t['orange'], t['gray']],
                 t, 20.5, lambda v: f'{v:.1f}%', height=0.6)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xticklabels(['0', '5', '10', '15', '20%'])
        if k:
            ax.set_yticklabels(['', '', ''])
    cs.footer(fig, t, 'Source: research/wnr4-20261121/results/pack3-*.json (128 Qwen3.8 and 32 V4.1 records, models '
              'counted on 256 other records).\nnvCOMP 6.0 ANS: the configuration that stays fast at 8-64 records '
              '(64-256 KiB chunks), docs/reports/wnr4-research-20261121.md\nsection 6; its harness is not in this '
              'repository.', h)
    return fig


def chart_gpu(t):
    # docs/reports/wnr4-research-20261121.md section 5, rows "same run (bench9)": kernel only, records in VRAM
    batches = [1, 8, 16, 32, 64, 128]
    q38 = {'WNR4 G=128': [3.6, 29, 58, 92, 164, 209], 'WNR3 kernels': [4.3, 33, 32, 56, 80, 91]}
    v41 = {'WNR4 G=128': [25.6, 146, 187, 187], 'WNR3 kernels': [30.6, 90, 103, 105]}
    h = 4.8
    fig = cs.figure(t, h)
    cs.header(fig, t, 'GPU decode by batch size',
              'Decoded GB/s, kernel time, records in VRAM, both codecs in one run. WNR4: one launch per batch;\n'
              'WNR3: two kernels per record on 8 streams.', h)
    fig.text(0.045, 1 - 1.18 / h, 'Machine: RTX 5090 (32 GB VRAM, PCIe 5.0 x16), AMD Ryzen 9 7950X3D, 32 GB '
             'DDR5-6200, Windows 11, CUDA 13.4.\nNo engine and no drive in these runs: the records were loaded into '
             'VRAM first, so neither the RAM nor the drives limit them.', fontsize=8, color=t['ink2'], va='top',
             linespacing=1.5)
    colors = {'WNR4 G=128': t['blue'], 'WNR3 kernels': t['orange']}
    for k, (title, data, bs) in enumerate((('Qwen3.8 MXFP4 (2.6 MB records)', q38, batches),
                                           ('DeepSeek V4.1 MXFP4 (18.8 MB records)', v41, batches[:4]))):
        ax = cs.axes(fig, t, [0.08 + k * 0.47, 0.2, 0.38, 0.42], title)
        ax.grid(axis='y', color=t['grid'], linewidth=0.8)
        ax.grid(axis='x', visible=False)
        ax.set_xscale('log', base=2)
        ax.set_xticks(bs)
        ax.set_xticklabels([str(b) for b in bs])
        ax.tick_params(axis='y', labelcolor=t['muted'], labelsize=8.5, pad=4)
        ax.set_ylim(0, 230)
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_xlim(bs[0] / 1.3, bs[-1] * 1.9)
        for name, ys in data.items():
            ax.plot(bs, ys, color=colors[name], linewidth=2, marker='o', markersize=5.5, zorder=3,
                    markeredgecolor=t['surface'], markeredgewidth=1.5, solid_capstyle='round')
            ax.text(bs[-1] * 1.12, ys[-1], f'{ys[-1]:.0f}', va='center', ha='left', fontsize=9, fontweight='bold',
                    color=t['ink'])
        ax.set_xlabel('records per batch', fontsize=8.5, color=t['muted'], labelpad=4)
        if k == 0:
            ax.set_ylabel('GB/s', fontsize=8.5, color=t['muted'])
    handles = [cs.plt.Line2D([], [], color=colors[n], linewidth=2, marker='o', markersize=5.5,
                             markeredgecolor=t['surface']) for n in colors]
    leg = fig.axes[0].legend(handles, list(colors), loc='upper left', frameon=False, fontsize=9, handlelength=1.8)
    for txt in leg.get_texts():
        txt.set_color(t['ink2'])
    cs.footer(fig, t, 'Source: docs/reports/wnr4-research-20261121.md section 5 (results/bench9-*.log), median of 5 '
              'runs per point.\nFrom page-locked host memory, upload included, WNR4 reaches 49-54 GB/s on '
              'Qwen3.8: the PCIe 5 upload is the limit there, not the decoder.', h)
    return fig


def chart_models(t):
    rows = []
    for line in open(os.path.join(RES, 'other', 'summary.md'), encoding='utf-8'):
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) == 8 and cells[2][:2] == '0.':
            rows.append((cells[0], float(cells[2]), float(cells[3])))
    rows.sort(key=lambda r: r[1])
    h = 4.9
    fig = cs.figure(t, h)
    cs.header(fig, t, 'Eight MoE exports, 64-expert samples', 'Space saved, byte-exact, models trained in 2 folds ('
              'each fold coded with the other). WNR4 at G=128 against WNR3.', h)
    ax = cs.axes(fig, t, [0.36, 0.17, 0.53, 0.62])
    n = len(rows)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0].replace(' (', '\n(') if len(r[0]) > 34 else r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(6, 13.5)
    ax.set_xticks([6, 8, 10, 12])
    ax.set_xticklabels(['6', '8', '10', '12%'])
    for i, (_, wnr3, wnr4) in enumerate(rows):
        a, b = 100 * (1 - wnr3), 100 * (1 - wnr4)
        ax.plot([a, b], [i, i], color=t['base'], linewidth=2.2, zorder=2, solid_capstyle='round')
        ax.scatter([b], [i], s=46, color=t['blue'], zorder=3, edgecolors=t['surface'], linewidths=1.5)
        ax.scatter([a], [i], s=22, color=t['orange'], zorder=4, edgecolors=t['surface'], linewidths=1.0)
        ax.text(max(a, b) + 0.18, i, f'{b - a:+.2f} pt', va='center', ha='left', fontsize=8.5, color=t['ink2'])
    handles = [cs.plt.Line2D([], [], color=c, linestyle='none', marker='o', markersize=7, markeredgecolor=t['surface'])
               for c in (t['blue'], t['orange'])]
    leg = fig.legend(handles, ['WNR4 G=128', 'WNR3 (WNR3-NV on NVFP4)'], loc='upper left',
                     bbox_to_anchor=(0.355, 0.84), ncol=2, frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t['ink2'])
    cs.footer(fig, t, 'Source: research/wnr4-20261121/results/other/ (WNR4 columns, other_models.py). The WNR3 '
              'columns are copied from a study that is not in this repository.', h)
    return fig


CHARTS = {'wnr4-ratio': chart_ratio, 'wnr4-gpu': chart_gpu, 'wnr4-models': chart_models}


def main():
    cs.setup_fonts()
    for name, make in CHARTS.items():
        for theme, t in cs.THEMES.items():
            print(cs.save(make(t), os.path.join(HERE, 'charts'), name, theme))


if __name__ == '__main__':
    main()
