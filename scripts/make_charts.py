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
# WNR4 is not built as a whole container: the WNR3 container scaled by WNR4 / WNR3 on 2,048 held-out records
# (research/wnr4-20261121/README.md, "Results": V4.1 0.8488 / 0.8539, Qwen3.8 MXFP4 0.8753 / 0.8809).
V41_WNR4_EST, Q38_WNR4_EST = round(123_306_192_896 * 0.8488 / 0.8539), round(56_532_672_512 * 0.8753 / 0.8809)


def chart_sizes(t):
    # docs/BENCHMARKS.md, "DeepSeek V4.1 REAP-192: container sizes" and "Qwen3.8-Flash-Next MXFP4" (container row);
    # the Qwen3.8 WEX4 file size is in README.md ("Sizes of the real files").
    v41 = [('original MXFP4 files', V41_ORIGINAL, 'muted', 'published safetensors'),
           ('raw WEX1', 144_389_472_256, 'gray', '+557,056 B header and table'),
           ('WNR2', 129_905_360_896, 'orange', None), ('WNR3-CZ', 123_306_192_896, 'blue', None),
           ('WNR4 (estimate)', V41_WNR4_EST, 'aqua', None)]
    q38 = [('original MXFP4 files', Q38_ORIGINAL, 'muted', 'published safetensors'),
           ('raw WEX4', 64_251_047_936, 'gray', '+78.2 MB index and 4 KiB alignment'),
           ('WNR3', 56_532_672_512, 'blue', None), ('WNR4 (estimate)', Q38_WNR4_EST, 'aqua', None)]
    h = 5.6
    fig = cs.figure(t, h)
    cs.header(fig, t, 'Same experts, fewer bytes',
              'Routed-expert bytes: as published on Hugging Face, then in the containers. Every record decodes back\n'
              'to the published bytes (WNR3-CZ: to the same values; -0 becomes +0).', h)
    ax1 = cs.axes(fig, t, [0.25, 0.505, 0.69, 0.235], 'DeepSeek V4.1 REAP-192 MXFP4 · 7,680 experts')
    ax2 = cs.axes(fig, t, [0.25, 0.195, 0.69, 0.19], 'Qwen3.8-Flash-Next MXFP4 · 24,576 experts')
    for ax, rows in ((ax1, v41), (ax2, q38)):
        orig = rows[0][1]
        cs.hbars(ax, [r[0] for r in rows], [r[1] / GB for r in rows], [t[r[2]] for r in rows], t, 240,
                 lambda v: f'{v:.1f} GB',
                 notes=[r[3] or f'{r[1] / orig:.3f} of original' for r in rows])
        ax.set_xticks(range(0, 161, 40))
    ax1.set_xticklabels([])
    ax2.set_xticklabels([f'{x} GB' for x in range(0, 161, 40)])
    cs.footer(fig, t, 'Original: the expert tensors in the safetensors headers of patrickbdevaney/DeepSeek-V4.1-Flash-'
              '1017-REAP192-MXFP4 @ 4ab649b2 and\namd/Qwen3.8-Flash-Next-Quark-MXFP4 @ b71e28af, summed by '
              'scripts/expert_bytes.py. Containers: docs/BENCHMARKS.md and README.md. WNR4 estimate:\nthe WNR3 '
              'container x WNR4/WNR3 on 2,048 held-out records (research/wnr4-20261121). GB = 10^9 bytes.', h)
    return fig


def chart_codecs(t):
    # docs/BENCHMARKS.md, "WNR3 against general-purpose codecs on real expert records (2026-11-15, 24 experts)":
    # stored / decoded bytes of 8 expert records per checkpoint; brotli is level 11, level 9 on MiniMax.
    codecs = ['WNR3', 'brotli', 'zstd 19', 'zlib 9', 'lzma 9e', 'lz4hc 12']
    ratios = {'DeepSeek V4.1 Flash': [0.8796, 0.9222, 0.9268, 0.9297, 0.9306, 0.9557],
              'Qwen3-235B MXFP4': [0.8919, 0.8951, 0.8966, 0.8999, 0.9043, 0.9527],
              'MiniMax-M2.1 MXFP4': [0.8856, 0.9043, 0.9055, 0.9083, 0.9117, 0.9556]}
    record = {'DeepSeek V4.1 Flash': 18_800_640, 'Qwen3-235B MXFP4': 10_027_008, 'MiniMax-M2.1 MXFP4': 7_520_256}
    h = 6.7
    fig = cs.figure(t, h)
    cs.header(fig, t, 'WNR3 saves more than general-purpose compressors',
              '8 real MXFP4 expert records per checkpoint: their bytes as published, then stored by each codec '
              '(shorter is better).', h)
    cs.size_panels(fig, t, [(f'{name} · 8 records of {record[name]:,} B', [('original MXFP4 files', 8 * record[name],
                    'muted')] + [(c, 8 * record[name] * r, 'blue' if c == 'WNR3' else 'gray') for c, r in zip(codecs, rs)])
                            for name, rs in ratios.items()], 1.3, h)
    cs.footer(fig, t, 'Source: docs/BENCHMARKS.md, "WNR3 against general-purpose codecs on real expert records" '
              '(2026-11-15). WNR3 byte-exact, with a model\ntrained on the same 8 records; brotli at level 11 (9 on '
              'MiniMax), the best general codec measured. The ratios do not depend on the machine.\nDecode speed, '
              'measured on an Intel Core i7-7700K with 32 GB of RAM: zstd, zlib and lz4 decode faster than WNR3; lzma, '
              'and brotli 11 on one thread, slower.', h)
    return fig


MACHINE = ('Machine: RTX 5090 (32 GB VRAM), AMD Ryzen 9 7950X3D, 32 GB DDR5-6200, Windows 11.\n'
           'Drives: U: NVMe PCIe 4.0 and C: NVMe PCIe 5.0, a full copy of the container on each; "U: + C:" splits the '
           'expert reads 38/62.\nRAM budget: the cap the engine is given for its RAM use; the rest of the 32 GB is '
           'left to the system.')


def chart_speed(t):
    # V4.1: research/wnr2-fastbench-20261109/NOTE-5-mirror-ab-20261110.md (docs/BENCHMARKS.md "containers, drives
    # and decode speed", first row): RAM_GB 16, ctx 32768, 7 prompts x 64 tokens, one pair, 971 experts in VRAM.
    # Qwen3.8: docs/reports/QWEN38-MXFP4-20261111.md "Speed": RAM 13 GB, 258-token prompt, 200 tokens, one run each.
    v41 = [('raw WEX1, U: only', 2.02, 'gray'), ('WNR2, U: + C:', 2.99, 'blue')]
    q38 = [('FP8 experts, U: only', 4.39, 'gray'), ('MXFP4 WNR3, U: only', 9.48, 'blue'),
           ('MXFP4 raw, U: only', 11.48, 'blue'), ('MXFP4 raw, U: + C:', 12.45, 'blue')]
    h = 5.6
    fig = cs.figure(t, h)
    cs.header(fig, t, 'Generation speed in the engine', 'Tokens per second while generating. A second drive '
              'with a copy of the container splits the expert reads.', h)
    fig.text(0.045, 1 - 0.98 / h, MACHINE, fontsize=8, color=t['ink2'], va='top', linespacing=1.5)
    ax1 = cs.axes(fig, t, [0.27, 0.545, 0.67, 0.09],
                  'DeepSeek V4.1 REAP-192 · RAM budget 16 GB · 971 experts in VRAM')
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
