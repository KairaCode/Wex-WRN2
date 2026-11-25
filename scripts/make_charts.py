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
# Whole built WEX4 + WNR4 containers on disk (2026-11-25):
# DeepSeek V4.1 REAP-192 WNR4 (7,680 records), Qwen3.8 MXFP4 WNR4 (24,576 records),
# and Qwen3.8 INT4-g64 WNR4-I4 (24,576 records vs 68.0 GB RAW, research/wnr4-i4-20261122).
V41_WNR4, Q38_WNR4 = 122_872_295_424, 56_847_855_616
Q38_I4_RAW, Q38_WNR4_I4 = 68_013_391_872, 53_294_047_232


def chart_sizes(t):
    # docs/BENCHMARKS.md, "DeepSeek V4.1 REAP-192: container sizes" and "Qwen3.8-Flash-Next MXFP4" (container row);
    # the Qwen3.8 WEX4 and WNR4 file sizes are in README.md ("What runs today") and research/wnr4-i4-20261122.
    v41 = [('original MXFP4 files', V41_ORIGINAL, 'muted', 'published safetensors'),
           ('raw WEX1', 144_389_472_256, 'gray', '+557,056 B header and table'),
           ('WNR2', 129_905_360_896, 'orange', None), ('WNR3-CZ', 123_306_192_896, 'blue', None),
           ('WNR4', V41_WNR4, 'aqua', None)]
    q38 = [('original MXFP4 files', Q38_ORIGINAL, 'muted', 'published safetensors'),
           ('raw WEX4', 64_251_047_936, 'gray', '+78.2 MB index and 4 KiB alignment'),
           ('WNR3', 56_532_672_512, 'blue', None), ('WNR4', Q38_WNR4, 'aqua', None)]
    q38i4 = [('raw int4-g64', Q38_I4_RAW, 'muted', 'uncompressed int4-g64 container'),
             ('WNR4-I4', Q38_WNR4_I4, 'aqua', None)]
    h = 6.9
    fig = cs.figure(t, h)
    cs.header(fig, t, 'Same experts, fewer bytes',
              'Routed-expert bytes: as published on Hugging Face, then in the containers. Every record decodes back\n'
              'to the published bytes (WNR3-CZ / V4.1 WNR4: to the same values; -0 becomes +0).', h)
    ax1 = cs.axes(fig, t, [0.25, 0.575, 0.69, 0.195], 'DeepSeek V4.1 REAP-192 MXFP4 · 7,680 experts')
    ax2 = cs.axes(fig, t, [0.25, 0.32, 0.69, 0.16], 'Qwen3.8-Flash-Next MXFP4 · 24,576 experts')
    ax3 = cs.axes(fig, t, [0.25, 0.145, 0.69, 0.085], 'Qwen3.8-Flash-Next INT4-g64 · 24,576 experts')
    for ax, rows in ((ax1, v41), (ax2, q38), (ax3, q38i4)):
        orig = rows[0][1]
        cs.hbars(ax, [r[0] for r in rows], [r[1] / GB for r in rows], [t[r[2]] for r in rows], t, 240,
                 lambda v: f'{v:.1f} GB',
                 notes=[r[3] or f'{r[1] / orig:.3f} of original' for r in rows])
        ax.set_xticks(range(0, 161, 40))
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([f'{x} GB' for x in range(0, 161, 40)])
    cs.footer(fig, t, 'Original: the expert tensors in the safetensors headers of patrickbdevaney/DeepSeek-V4.1-Flash-'
              '1017-REAP192-MXFP4 @ 4ab649b2 and\namd/Qwen3.8-Flash-Next-Quark-MXFP4 @ b71e28af, summed by '
              'scripts/expert_bytes.py. Containers: docs/BENCHMARKS.md, README.md\nand research/wnr4-i4-20261122 '
              '(whole built WEX4+WNR4 files). GB = 10^9 bytes.', h)
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
    i4_total = 708_472_832   # 256 held-out int4-g64 records (2,767,472 B each)
    panels = [
        (f'DeepSeek V4.1 Flash · 8 records of {v41_rec:,} B',
         [('original MXFP4 files', 8 * v41_rec, 'muted'),
          ('WNR4', 8 * v41_rec * 0.8488, 'aqua'),
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
              'research/wnr4-20261121 (V4.1 WNR4 G=128, CZ),\nand research/wnr4-i4-20261122 (Qwen3.8 INT4-g64 '
              'WNR4-I4 byte-exact vs zstd 19). Ratios do not depend on the machine.', h)
    return fig


MACHINE = ('Machine: RTX 5090 (32 GB VRAM), AMD Ryzen 9 7950X3D, 32 GB DDR5-6200, Windows 11; engines built with gcc '
           '16.2 and CUDA 13.4.\nDrives: U: = NVMe PCIe 4.0 (Gen4), C: = NVMe PCIe 5.0 (Gen5), a full copy of the '
           'container on each; "U: + C:" splits the expert reads 38/62.\nRAM budget: the cap the engine is given for '
           'its RAM use (16 GB in the V4.1 runs, 13 GB in the Qwen3.8 runs), not the 32 GB of the machine.')


def chart_speed(t):
    # V4.1: research/wnr2-fastbench-20261109/NOTE-5-mirror-ab-20261110.md (docs/BENCHMARKS.md "containers, drives
    # and decode speed", first row): RAM_GB 16 in both arms, ctx 32768, 7 prompts x 64 tokens, one pair, 971 (A) and
    # 970 (B) experts in VRAM; A read U: only, B U: + C: split 38/62 (OS read bytes C: 61%, U: 39%).
    # Qwen3.8: docs/reports/QWEN38-MXFP4-20261111.md "Speed": RAM 13 GB ("ram_gb": 13 in every speed-*.json),
    # 258-token prompt, 200 tokens, one run each. The FP8 snapshot is on U: (docs/reports/WNR3-VERIFY.md, disk table;
    # the --snap path in research/qwen38-mxfp4-20261111/q_ab.py).
    v41 = [('raw WEX1, U: Gen4', 2.02, 'gray'), ('WNR2, U: Gen4 + C: Gen5', 2.99, 'blue')]
    q38 = [('FP8 experts, U: Gen4', 4.39, 'gray'), ('MXFP4 WNR3, U: Gen4', 9.48, 'blue'),
           ('MXFP4 raw, U: Gen4', 11.48, 'blue'), ('MXFP4 raw, U: Gen4 + C: Gen5', 12.45, 'blue')]
    h = 5.6
    fig = cs.figure(t, h)
    cs.header(fig, t, 'Generation speed in the engine', 'Tokens per second while generating. A second drive '
              'with a copy of the container splits the expert reads.', h)
    fig.text(0.045, 1 - 0.98 / h, MACHINE, fontsize=8, color=t['ink2'], va='top', linespacing=1.5)
    ax1 = cs.axes(fig, t, [0.27, 0.545, 0.67, 0.09],
                  'DeepSeek V4.1 REAP-192 · RAM budget 16 GB · 970-971 experts in VRAM')
    ax2 = cs.axes(fig, t, [0.27, 0.2, 0.67, 0.2],
                  'Qwen3.8-Flash-Next · RAM budget 13 GB · ~25.5 GB of experts in VRAM')
    for ax, rows, xmax in ((ax1, v41, 3.6), (ax2, q38, 14.5)):
        cs.hbars(ax, [r[0] for r in rows], [r[1] for r in rows], [t[r[2]] for r in rows], t, xmax,
                 lambda v: f'{v:.2f} tok/s')
    ax1.set_xticks([0, 1, 2, 3])
    ax2.set_xticks([0, 3, 6, 9, 12])
    cs.footer(fig, t, 'Sources: docs/BENCHMARKS.md ("Machine"). V4.1: research/wnr2-fastbench-20261109 NOTE-5, one '
              'pair (2026-11-10), 7 prompts x 64 tokens, ctx 32768,\ntoken ids and logits bit-identical; format and '
              'drive split change together. Qwen3.8: docs/reports/QWEN38-MXFP4-20261111.md, 258-token prompt,\n200 '
              'tokens, one run each, engine of 2026-11-11 (later builds are faster). Not comparable across the two '
              'models (different sizes, budgets, prompts).', h)
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
            jobs[names.get(runner, runner)].append(m.group(1))
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
    cs.hbars(ax2, order, [len(jobs[k]) for k in order], [t['blue']] * len(order), t, 8.5, lambda v: f'{v:.0f}',
             height=0.58)
    ax2.set_xticks([0, 2, 4, 6, 8])
    ax2.tick_params(axis='y', labelsize=8.5)
    detail = '\n'.join(f'{k}: ' + ', '.join(sorted(jobs[k])) for k in order)
    fig.text(0.535, 0.39, detail, fontsize=7.2, color=t['ink2'], va='top', linespacing=1.6, wrap=True)
    cs.footer(fig, t, 'Source: wex4/vectors/MANIFEST.tsv (for a refused file, the first rule it breaks, as the C '
              'reader c/wex4.h reports it) and .github/workflows/ci.yml, fuzz.yml\n(jobs without a manual-run condition).', h)
    return fig


CHARTS = {'sizes': chart_sizes, 'codecs': chart_codecs, 'speed': chart_speed, 'tests': chart_tests}


def main(argv):
    out = argv[1] if len(argv) > 1 else os.path.join(ROOT, 'assets', 'charts')
    cs.setup_fonts()
    for name, make in CHARTS.items():
        for theme, t in cs.THEMES.items():
            print(cs.save(make(t), out, name, theme))


if __name__ == '__main__':
    main(sys.argv)
