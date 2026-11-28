# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""The PNG charts of the README, in assets/charts/ (one light and one dark file each).

    python scripts/make_charts.py [OUT_DIR]      (not -I when matplotlib is in the user site)

Needs matplotlib. The measured numbers are copied from the pages named in each chart's footnote (docs/BENCHMARKS.md
sections); the test counts are read from wex4/vectors/MANIFEST.tsv and the CI workflow files when the script runs.
Change a number here only together with its source page."""
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import chartstyle as cs  # noqa: E402

GB = 1e9


# Routed-expert bytes of the published safetensors, summed from their headers by scripts/expert_bytes.py:
# patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4 @ 4ab649b2 (7,680 experts in 40 layers; the 384 MTP experts
# have other names and are not counted) and amd/Qwen3.8-Flash-Next-Quark-MXFP4 @ b71e28af (24,576 experts).
V41_ORIGINAL, Q38_ORIGINAL = 144_388_915_200, 64_172_851_200
# Whole built WEX4 + WNR4 containers (2026-11-21 to 2026-11-26, docs/BENCHMARKS.md):
# DeepSeek V4.1 REAP-192 WNR4-MX (7,680 records), Qwen3.8 MXFP4 WNR4-MX (24,576 records),
# Qwen3.8 INT4-g64 WNR4-I4 (24,576 records: G=4 production & G=32 archive, research/wnr4-i4-20261122),
# and Qwen3.8 UD-IQ1_M WNR4-GG (24,576 records, research/wnr4-gg-20261121).
V41_WNR4, Q38_WNR4 = 122_872_295_424, 56_847_855_616
Q38_I4_RAW, Q38_WNR4_I4_G4, Q38_WNR4_I4_G32 = 67_995_492_352, 53_961_261_056, 53_294_047_232
Q38_GG_RAW, Q38_WNR4_GG = 34_862_370_816, 33_377_181_696


def chart_sizes(t):
    # docs/BENCHMARKS.md: "DeepSeek V4.1 REAP-192: container sizes", "Qwen3.8-Flash-Next MXFP4",
    # and "WNR4 record codec: container sizes, ratio and decode speed".
    v41 = [('original MXFP4 files', V41_ORIGINAL, 'muted', 'published safetensors'),
           ('raw WEX1', 144_389_472_256, 'gray', '+557,056 B header and table'),
           ('WNR2', 129_905_360_896, 'orange', None), ('WNR3-CZ', 123_306_192_896, 'blue', None),
           ('WNR4-MX', V41_WNR4, 'aqua', None)]
    q38 = [('original MXFP4 files', Q38_ORIGINAL, 'muted', 'published safetensors'),
           ('raw WEX4', 64_251_047_936, 'gray', '+78.2 MB index and 4 KiB alignment'),
           ('WNR3', 56_532_672_512, 'blue', None), ('WNR4-MX', Q38_WNR4, 'aqua', None)]
    q38i4 = [('raw int4-g64', Q38_I4_RAW, 'muted', 'uncompressed WEX4 RAW'),
             ('WNR4-I4 (G=4 prod)', Q38_WNR4_I4_G4, 'aqua', '0.794 of original · 0.027 ms GPU'),
             ('WNR4-I4 (G=32 archive)', Q38_WNR4_I4_G32, 'aqua', None)]
    q38gg = [('raw GGUF 2-bit', Q38_GG_RAW, 'muted', 'experts-only WEX4 RAW'),
             ('WNR4-GG', Q38_WNR4_GG, 'aqua', None)]
    h = 8.65
    fig = cs.figure(t, h)
    cs.header(fig, t, 'Same experts, fewer bytes',
              'Routed-expert bytes: as published on Hugging Face, then in the containers. Every record decodes back\n'
              'to the published bytes (WNR3-CZ / V4.1 WNR4-MX: to the same values; -0 becomes +0).', h)
    ax1 = cs.axes(fig, t, [0.26, 0.665, 0.68, 0.165], 'DeepSeek V4.1 REAP-192 MXFP4 · 7,680 experts')
    ax2 = cs.axes(fig, t, [0.26, 0.455, 0.68, 0.140], 'Qwen3.8-Flash-Next MXFP4 · 24,576 experts')
    ax3 = cs.axes(fig, t, [0.26, 0.275, 0.68, 0.110], 'Qwen3.8-Flash-Next INT4-g64 · 24,576 experts')
    ax4 = cs.axes(fig, t, [0.26, 0.125, 0.68, 0.078], 'Qwen3.8-Flash-Next UD-IQ1_M (2-bit GGUF) · 24,576 experts')
    for ax, rows in ((ax1, v41), (ax2, q38), (ax3, q38i4), (ax4, q38gg)):
        orig = rows[0][1]
        cs.hbars(ax, [r[0] for r in rows], [r[1] / GB for r in rows], [t[r[2]] for r in rows], t, 240,
                 lambda v: f'{v:.1f} GB',
                 notes=[r[3] or f'{r[1] / orig:.3f} of original' for r in rows])
        ax.set_xticks(range(0, 161, 40))
    for ax in (ax1, ax2, ax3):
        ax.set_xticklabels([])
    ax4.set_xticklabels([f'{x} GB' for x in range(0, 161, 40)])
    cs.footer(fig, t, 'Original: safetensors expert tensors of patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4 @ 4ab649b2\n'
              'and amd/Qwen3.8-Flash-Next-Quark-MXFP4 @ b71e28af (scripts/expert_bytes.py). Containers: docs/BENCHMARKS.md,\n'
              'research/wnr4-i4-20261122 and research/wnr4-gg-20261121 (whole built WEX4+WNR4 files). GB = 10^9 bytes.', h)
    return fig


def chart_codecs(t):
    # docs/BENCHMARKS.md, "WNR3 against general-purpose codecs on real expert records (2026-11-15, 24 experts)";
    # DeepSeek V4.1 WNR4 ratio (0.8488 on 2,048 held-out records): research/wnr4-20261121/README.md;
    # Qwen3.8 INT4-g64 WNR4-I4 (0.78333) vs zstd 19 (0.8074) on 256 held-out records: research/wnr4-i4-20261122.
    h = 7.9
    fig = cs.figure(t, h)
    cs.header(fig, t, 'WNR3 and WNR4 save more than general-purpose compressors',
              'Real expert records per checkpoint: their bytes as published, then stored by each codec '
              '(shorter is better).', h)
    v41_rec = 18_800_640
    q3_rec = 10_027_008
    mm_rec = 7_520_256
    i4_total = 256 * 2_764_800   # 256 held-out int4-g64 records (2,764,800 B each)
    panels = [
        (f'DeepSeek V4.1 Flash · 8 records of {v41_rec:,} B',
         [('original MXFP4 files', 8 * v41_rec, 'muted'),
          ('WNR4-MX', 8 * v41_rec * 0.8488, 'aqua'),
          ('WNR3', 8 * v41_rec * 0.8796, 'blue'),
          ('brotli', 8 * v41_rec * 0.9222, 'gray'),
          ('zstd 19', 8 * v41_rec * 0.9268, 'gray'),
          ('zlib 9', 8 * v41_rec * 0.9297, 'gray'),
          ('lzma 9e', 8 * v41_rec * 0.9306, 'gray'),
          ('lz4hc 12', 8 * v41_rec * 0.9557, 'gray')]),
        (f'Qwen3-235B MXFP4 · 8 records of {q3_rec:,} B',
         [('original MXFP4 files', 8 * q3_rec, 'muted'),
          ('WNR3', 8 * q3_rec * 0.8919, 'blue'),
          ('brotli', 8 * q3_rec * 0.8951, 'gray'),
          ('zstd 19', 8 * q3_rec * 0.8966, 'gray'),
          ('zlib 9', 8 * q3_rec * 0.8999, 'gray'),
          ('lzma 9e', 8 * q3_rec * 0.9043, 'gray'),
          ('lz4hc 12', 8 * q3_rec * 0.9527, 'gray')]),
        (f'MiniMax-M2.1 MXFP4 · 8 records of {mm_rec:,} B',
         [('original MXFP4 files', 8 * mm_rec, 'muted'),
          ('WNR3', 8 * mm_rec * 0.8856, 'blue'),
          ('brotli', 8 * mm_rec * 0.9043, 'gray'),
          ('zstd 19', 8 * mm_rec * 0.9055, 'gray'),
          ('zlib 9', 8 * mm_rec * 0.9083, 'gray'),
          ('lzma 9e', 8 * mm_rec * 0.9117, 'gray'),
          ('lz4hc 12', 8 * mm_rec * 0.9556, 'gray')]),
        ('Qwen3.8-Flash-Next INT4-g64 · 256 held-out records',
         [('original int4-g64', i4_total, 'muted'),
          ('WNR4-I4', i4_total * 0.78333, 'aqua'),
          ('zstd 19', i4_total * 0.8074, 'gray')]),
    ]
    cs.size_panels(fig, t, panels, 1.25, h, row_in=0.16, gap_in=0.45)
    cs.footer(fig, t, 'Sources: docs/BENCHMARKS.md (2026-11-15, WNR3 byte-exact vs general codecs), '
              'research/wnr4-20261121 (V4.1 WNR4-MX G=128, CZ),\nand research/wnr4-i4-20261122 (Qwen3.8 INT4-g64 '
              'WNR4-I4 byte-exact vs zstd 19). Ratios do not depend on the machine.', h)
    return fig


def chart_wnr4_decode(t):
    # docs/BENCHMARKS.md, "WNR4 record codec: container sizes, ratio and decode speed":
    # RTX 5090 (CUDA 13.4, sm_120), records resident in VRAM, kernel only. The I4 figures are those of the rerun
    # of 2026-11-27 (research/wnr4-gpu-rerun-20261127: batch 32, and one record; where its 3 runs differ, their median).
    peak = [
        ('WNR4-I4 · Qwen3.8 (G=32)', 347.0, 'aqua'),
        ('WNR4-I4 · Qwen3.8 (G=16)', 345.5, 'aqua'),
        ('WNR4-I4 · Qwen3.8 (G=4)', 339.9, 'aqua'),
        ('WNR4-MX · Qwen3.8', 209.0, 'aqua'),
        ('WNR4-MX · V4.1', 188.0, 'aqua'),
        ('WNR4-NV · Qwen3.8', 149.0, 'aqua'),
        ('WNR4-GG · Qwen3.8 whole', 133.0, 'aqua'),
        ('WNR3 · V4.1 (ref)', 105.0, 'blue'),
        ('WNR3 · Qwen3.8 (ref)', 91.0, 'blue'),
    ]
    h = 5.85
    fig = cs.figure(t, h)
    cs.header(fig, t, 'WNR4 GPU decode: throughput and single-record latency',
              '32-lane warp-parallel rANS on RTX 5090 (CUDA 13.4, sm_120, kernel only, records in VRAM).\n'
              'End-to-end PCIe 5 upload + decode from page-locked host memory: 48–60 GB/s (MX); I4: 70–72 GB/s reported, no tracked log.', h)
    ax1 = cs.axes(fig, t, [0.265, 0.18, 0.30, 0.54], 'Peak kernel decode (GB/s)')
    cs.hbars(ax1, [r[0] for r in peak], [r[1] for r in peak], [t[r[2]] for r in peak], t, 445,
             lambda v: f'{v:.0f}' if v == int(v) else f'{v:.1f}', height=0.58)
    ax1.tick_params(axis='y', labelsize=8.1)
    ax1.set_xticks([0, 100, 200, 300])
    ax1.set_xticklabels(['0', '100', '200', '300 GB/s'])

    ax2 = cs.axes(fig, t, [0.645, 0.18, 0.30, 0.54], '1-record latency vs G (ms)')
    ax2.grid(axis='y', color=t['grid'], linewidth=0.8)
    ax2.grid(axis='x', color=t['grid'], linewidth=0.5, linestyle=':')
    series = [
        ('I4 (Qwen3.8)', [4, 8, 16, 32, 64], [0.027, 0.046, 0.079, 0.146, 0.290], t['orange'], 's'),
        ('MX (Qwen3.8)', [32, 64, 128, 256], [0.19, 0.37, 0.75, 1.44], t['aqua'], 'o'),
        ('NV (Qwen3.8)', [32, 64, 128, 256], [0.25, 0.47, 0.90, 1.79], t['blue'], '^'),
        ('GG (whole)', [32, 128], [0.234, 0.872], t['muted'], 'D'),
    ]
    for name, gs, ms, col, mk in series:
        ax2.plot(gs, ms, marker=mk, linewidth=2.0, markersize=5.0, color=col, label=name, zorder=4)
    ax2.annotate('0.027 ms\n(G=4 prod)', xy=(4, 0.027), xytext=(4.8, 0.50),
                 fontsize=7.6, fontweight='bold', color=t['orange'],
                 arrowprops={'arrowstyle': '->', 'color': t['orange'], 'lw': 1.1, 'shrinkA': 2, 'shrinkB': 4})
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax2.set_xticklabels(['4', '8', '16', '32', '64', '128', '256'])
    ax2.set_xlim(3, 320)
    ax2.set_ylim(0, 2.0)
    ax2.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax2.set_yticklabels(['0', '0.5', '1.0', '1.5', '2.0 ms'])
    ax2.set_xlabel('tile group size G (cells/lane)', fontsize=8, color=t['muted'], labelpad=4)
    ax2.tick_params(axis='y', labelsize=8.3, labelcolor=t['muted'], pad=4)
    leg = ax2.legend(loc='upper left', frameon=True, facecolor=t['surface'], edgecolor=t['border'],
                     fontsize=7.6, handlelength=1.6, borderpad=0.5)
    for txt in leg.get_texts():
        txt.set_color(t['ink2'])
    cs.footer(fig, t, 'Sources: docs/BENCHMARKS.md ("WNR4 record codec"), research/wnr4-20261121 (MX/NV/V4.1, bench8/bench9),\n'
              'research/wnr4-gg-20261121 (GG, batch 32), and research/wnr4-gpu-rerun-20261127 (I4 v2 kernel, median of its 3 runs). Kernel only; no RAM or drive.', h)
    return fig


MACHINE = ('Machine: RTX 5090 (32 GB VRAM), AMD Ryzen 9 7950X3D, 32 GB DDR5-6200, Windows 11; engines built with gcc '
           '16.2 and CUDA 13.4.\nDrives: U: = NVMe PCIe 4.0 (Gen4), C: = NVMe PCIe 5.0 (Gen5), a full copy of the '
           'container on each; "U: + C:" splits the expert reads 38/62.\nRAM budget: the cap the engine is given for '
           'its RAM use (15–16 GB in the V4.1 runs, 13 GB in the Qwen3.8 runs), not the 32 GB of the machine.')


def chart_speed(t):
    # V4.1: NOTE-5 (2026-11-10, RAM 16 GB: WEX1 U: 2.02, WNR2 U:+C: 2.99) and r4-r9combo.log (2026-11-13,
    # RAM 15 GB, WEX4+WNR3 U:+C: + V4_WEX_SUBREADS=4 + V4_PREWARM_BATCH=1 + V41_PREFILL_SUFFIX=1: 3.83/3.84 -> 3.84).
    # Qwen3.8 (258-token prompt, RAM 13 GB): QWEN38-MXFP4-20261111.md (FP8 4.39, WNR3 9.48, RAW U: 11.48, RAW U:+C: 12.45)
    # and rdq38dma.cmp.txt (2026-11-12, WEX4 U:+C: + QT_ARENA=1 + QT_GPU_MISS=2: mean of 18.76 / 19.24 = 19.00).
    v41 = [('raw WEX1, U: Gen4', 2.02, 'gray'),
           ('WNR2, U: Gen4 + C: Gen5', 2.99, 'blue'),
           ('WEX4+WNR3, U:+C: + sub-reads', 3.84, 'aqua')]
    q38 = [('FP8 experts, U: Gen4', 4.39, 'gray'),
           ('MXFP4 WNR3, U: Gen4', 9.48, 'blue'),
           ('MXFP4 raw, U: Gen4', 11.48, 'blue'),
           ('MXFP4 raw, U: Gen4 + C: Gen5', 12.45, 'blue'),
           ('MXFP4 WEX4, U:+C: + arena/GPU-miss', 19.00, 'aqua')]
    h = 6.3
    fig = cs.figure(t, h)
    cs.header(fig, t, 'Generation speed in the engine', 'Tokens per second while generating: dual-NVMe mirror '
              'splits, WEX4 sub-reads, and VRAM arena / GPU-miss path.', h)
    fig.text(0.045, 1 - 0.96 / h, MACHINE, fontsize=8, color=t['ink2'], va='top', linespacing=1.5)
    ax1 = cs.axes(fig, t, [0.31, 0.545, 0.63, 0.135],
                  'DeepSeek V4.1 REAP-192 · RAM budget 15–16 GB · ~960–971 experts in VRAM')
    ax2 = cs.axes(fig, t, [0.31, 0.18, 0.63, 0.24],
                  'Qwen3.8-Flash-Next · RAM budget 13 GB · 258-token prompt')
    for ax, rows, xmax in ((ax1, v41, 4.7), (ax2, q38, 23.0)):
        cs.hbars(ax, [r[0] for r in rows], [r[1] for r in rows], [t[r[2]] for r in rows], t, xmax,
                 lambda v: f'{v:.2f} tok/s')
    ax1.set_xticks([0, 1, 2, 3, 4])
    ax2.set_xticks([0, 5, 10, 15, 20])
    cs.footer(fig, t, 'Sources: docs/BENCHMARKS.md ("Machine"). V4.1: NOTE-5 (2026-11-10, 7x64 tok, bit-identical) and '
              'r4-r9combo.log (2026-11-13,\n657-tok prompt, 75 tok, ABBA mean 3.84). Qwen3.8: QWEN38-MXFP4-20261111.md '
              'and rdq38dma.cmp.txt (2026-11-12, ABBA mean 19.00).\nNot comparable across the two models (different sizes, budgets, prompts).', h)
    return fig


def ci_jobs():
    """The jobs that run on every pull request, by runner, from the workflow files."""
    names = {'ubuntu-24.04': 'Linux x86-64', 'ubuntu-24.04-arm': 'Linux arm64', 'macos-15': 'macOS arm64',
             'windows-2022': 'Windows x86-64'}
    jobs = collections.defaultdict(list)
    for f in ('ci.yml', 'fuzz.yml'):
        text = open(os.path.join(ROOT, '.github', 'workflows', f), encoding='utf-8').read().split('\njobs:\n', 1)[1]
        for m in re.finditer(r'(?m)^  ([\w-]+):\n((?:(?:    .*|\s*)\n)*)', text):
            block = m.group(2)
            if re.search(r'(?m)^    if:.*workflow_dispatch', block):
                continue
            runner = re.search(r'runs-on:\s*(\S+)', block).group(1)
            if runner.startswith('$'):   # `${{ matrix.os }}`: the job runs once on each runner of its `os:` list
                runners = re.search(r'(?m)^ +os:\s*\[([^\]]+)\]', block).group(1).split(',')
            else:
                runners = [runner]
            for r in runners:
                jobs[names.get(r.strip(), r.strip())].append(m.group(1))
    assert not any(k.startswith('$') for k in jobs), 'a runs-on expression that ci_jobs() cannot resolve'
    return jobs


def chart_tests(t):
    rows = [line.rstrip('\n').split('\t') for line in open(os.path.join(ROOT, 'wex4', 'vectors', 'MANIFEST.tsv'),
                                                            encoding='utf-8') if not line.startswith('#')]
    verdicts = collections.Counter(r[2] if r[1] == 'reject' else 'accepted' for r in rows if len(r) > 2)
    accepted = verdicts.pop('accepted')
    classes = sorted(verdicts.items(), key=lambda kv: -kv[1])
    jobs = ci_jobs()
    order = sorted(jobs, key=lambda k: -len(jobs[k]))
    h = 5.4
    fig = cs.figure(t, h)
    cs.header(fig, t, 'Tested on every change', f'{len(rows)} conformance files that every reader must accept or '
              f'refuse, and {sum(map(len, jobs.values()))} CI jobs on four platforms for every pull request.', h)
    ax1 = cs.axes(fig, t, [0.235, 0.15, 0.27, 0.64], f'Conformance vectors ({len(rows)})')
    labels = ['accepted'] + [c.replace('WEX4_E_', 'refused: ') for c, _ in classes]
    cs.hbars(ax1, labels, [accepted] + [n for _, n in classes], [t['aqua']] + [t['blue']] * len(classes), t, 31,
             lambda v: f'{v:.0f}', height=0.62)
    ax1.tick_params(axis='y', labelsize=8.5)
    ax1.set_xticks([0, 10, 20, 30])
    ax2 = cs.axes(fig, t, [0.69, 0.47, 0.25, 0.32], 'CI jobs per pull request')
    most = len(jobs[order[0]])
    cs.hbars(ax2, order, [len(jobs[k]) for k in order], [t['blue']] * len(order), t, most * 1.2, lambda v: f'{v:.0f}',
             height=0.58)
    ax2.set_xticks(range(0, most + 1, 2 if most <= 8 else 4))
    ax2.tick_params(axis='y', labelsize=8.5)
    detail = '\n'.join(f'{k}: ' + ', '.join(sorted(jobs[k])) for k in order)
    fig.text(0.535, 0.39, detail, fontsize=7.0, color=t['ink2'], va='top', linespacing=1.6, wrap=True)
    cs.footer(fig, t, 'Source: wex4/vectors/MANIFEST.tsv (for a refused file, the first rule it breaks, as the C '
              'reader c/wex4.h reports it) and .github/workflows/ci.yml, fuzz.yml\n(jobs without a manual-run condition).', h)
    return fig


CHARTS = {
    'sizes': chart_sizes,
    'codecs': chart_codecs,
    'wnr4-decode': chart_wnr4_decode,
    'speed': chart_speed,
    'tests': chart_tests,
}


def main(argv):
    out = argv[1] if len(argv) > 1 else os.path.join(ROOT, 'assets', 'charts')
    cs.setup_fonts()
    for name, make in CHARTS.items():
        for theme, t in cs.THEMES.items():
            print(cs.save(make(t), out, name, theme))


if __name__ == '__main__':
    main(sys.argv)
