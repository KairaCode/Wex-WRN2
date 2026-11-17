# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Tables of this run from the result files.

  python -I summarize.py RESULTS_DIR > tables.md

RESULTS_DIR holds estimates.json, extra.json, nz5.json (analyze.py, extra.py), wnr3-<checkpoint>.json and
glm-container.json (wnr3_real.py), proto.json (run_proto.py), baselines.log (baselines.py) and
speed-*.txt (bench_real.c of research/real-experts-20261115, optional). GB = 10^9 bytes. Standard library only.
"""
import glob
import json
import os
import sys

# routed experts of each full checkpoint (layers with experts x experts per layer, from config.json and the index)
FULL = {'deepseek': (40, 384), 'qwen': (48, 512), 'glm': (42, 288)}
PAIR_OF = {'ds-nvfp4': 'deepseek', 'ds-mxfp4': 'deepseek', 'ds-quark': 'deepseek', 'qwen-nvfp4': 'qwen',
           'qwen-mxfp4': 'qwen', 'glm-nvfp4': 'glm', 'glm-mxfp4': 'glm', 'glm-nz5': 'glm'}
ORDER = ['ds-nvfp4', 'ds-mxfp4', 'ds-quark', 'qwen-nvfp4', 'qwen-mxfp4', 'glm-nvfp4', 'glm-mxfp4']


def f4(x):
    return f'{x:.4f}'


def main(argv):
    d = argv[1]
    est = {r['checkpoint']: r for r in json.load(open(os.path.join(d, 'estimates.json')))['results']}
    ex = json.load(open(os.path.join(d, 'extra.json')))
    nz5 = json.load(open(os.path.join(d, 'nz5.json')))
    proto = json.load(open(os.path.join(d, 'proto.json')))
    wnr3 = {os.path.basename(p)[5:-5]: json.load(open(p)) for p in glob.glob(os.path.join(d, 'wnr3-*.json'))}
    cont = json.load(open(os.path.join(d, 'glm-container.json')))
    base = [json.loads(l[5:]) for l in open(os.path.join(d, 'baselines.log')) if l.startswith('json ')]
    ctx = {r['checkpoint']: r for r in ex['contexts']}
    out = []
    p = out.append

    p('## 1. Raw bytes of the 16 sampled experts\n')
    p('| checkpoint | format | weights B | scales B | global scalars B | total B | bits per weight |')
    p('|---|---|---:|---:|---:|---:|---:|')
    for c in ORDER:
        r = est[c]
        p(f"| {c} | {r['format']} | {r['raw']['weights']:,} | {r['raw']['scales']:,} | {r['raw']['global']:,} | "
          f"{r['raw_total']:,} | {8 * r['raw_total'] / r['nibbles']:.4f} |")
    p(f"| glm-nz5 | nz5 (csf0_v1) | {nz5['weights_bytes']:,} | {nz5['csf0_codes_bytes'] + nz5['csf0_bases_bytes']:,} "
      f"| {nz5['scalars_bytes']:,} | {nz5['raw_total']:,} | {8 * nz5['raw_total'] / nz5['nibbles']:.4f} |")

    p('\n## 2. Properties of the nibbles and scales\n')
    p('| checkpoint | -0 nibbles | share | blocks with max magnitude 6 | even blocks | blocks with max 3 | scale codes used |')
    p('|---|---:|---:|---:|---:|---:|---:|')
    for c in ORDER:
        r = est[c]
        p(f"| {c} | {r['neg0_nibbles']:,} | {r['neg0_fraction']:.4%} | {r['max7_fraction']:.4f} | {r['even_fraction']:.4f} | "
          f"{r['max5_blocks']:,} | {r['scale_codes_used']} |")

    p('\n## 3. Entropy estimates (in-sample, pooled per matrix over the 16 experts)\n')
    p('Nibbles, bits per nibble:\n')
    p('| checkpoint | order 0 | class only | WNR3 contexts (class, pm, mx) | same after CZ | full scale code, pm, mx (CZ) | + position in block (CZ) |')
    p('|---|---:|---:|---:|---:|---:|---:|')
    for c in ORDER:
        w = est[c]['weight_bits_per_nibble']
        k = 'srel7 (WNR3)' if est[c]['format'] == 'mxfp4' else 'exponent srel7'
        p(f"| {c} | {f4(w['order0'])} | {f4(w['| ' + k])} | {f4(w['| ' + k + ', pm, mx'])} | {f4(w['| ' + k + ', pm, mx (CZ)'])} | "
          f"{f4(w['| full code, pm, mx (CZ)'])} | "
          f"{f4(ctx[c]['nibble_bits_wnr3_ctx_plus_pos_cz']) if c in ctx else '-'} |")
    p('\nClass: MXFP4 `srel7` = clamp(scale - anchor, -3, 3) (WNR3); NVFP4 `exponent srel7` = the same on the E4M3 exponent field.\n')
    p('NVFP4 nibbles under other scale classes (bits per nibble, WNR3 contexts with that class, no CZ):\n')
    p('| checkpoint | none | mantissa | exponent | exponent srel7 | code srel25 | full code |')
    p('|---|---:|---:|---:|---:|---:|---:|')
    for c in ('ds-nvfp4', 'qwen-nvfp4', 'glm-nvfp4'):
        w = est[c]['weight_bits_per_nibble']
        p(f"| {c} | " + ' | '.join(f4(w[f'| {k}, pm, mx']) for k in
                                   ('none', 'mantissa', 'exponent', 'exponent srel7', 'code srel25', 'full code')) + ' |')
    p('\nScales, bits per scale:\n')
    p('| checkpoint | order 0 | order 1 (left) | left + parity | delta along row | order 1 (row above) |')
    p('|---|---:|---:|---:|---:|---:|')
    for c in ORDER:
        s = est[c]['scale_bits_per_scale']
        p(f"| {c} | {f4(s['order0'])} | {f4(s['order1 (left)'])} | {f4(ctx[c]['scale_bits_left_parity']) if c in ctx else '-'} | "
          f"{f4(s['delta along row'])} | {f4(s['order1 (row above)'])} |")
    p('\nEstimated stored / raw (nibble entropy + scale entropy + raw scalars; no model, chunk or padding cost):\n')
    p('| checkpoint | WNR3-like (class srel7, scales order 0) | same + CZ | full scale code + best scale model, CZ |')
    p('|---|---:|---:|---:|')
    for c in ORDER:
        e = est[c]['estimated_fraction_of_raw']
        p(f"| {c} | {f4(e['WNR3-like: srel7 contexts, scales order 0'])} | {f4(e['WNR3-like + CZ'])} | "
          f"{f4(e['full scale code, pm, mx (CZ); best scale model'])} |")

    p('\n## 4. WNR3 (the repository encoder) on the MXFP4 samples\n')
    p('| checkpoint | profile | byte-exact, cross-fold | byte-exact, in-sample | CZ, cross-fold | CZ, in-sample |')
    p('|---|---|---:|---:|---:|---:|')
    for c in ('ds-mxfp4', 'ds-quark', 'qwen-mxfp4', 'glm-mxfp4'):
        s = wnr3[c]['summary']
        p(f"| {c} | {wnr3[c]['profile']} | {f4(s['byte-exact']['ratio_cross_fold'])} | {f4(s['byte-exact']['ratio_in_sample'])} | "
          f"{f4(s['cz']['ratio_cross_fold'])} | {f4(s['cz']['ratio_in_sample'])} |")
    flags = sorted({r['wnr3_flags'] for r in cont['records'].values()})
    p(f"| glm-mxfp4, records of the existing container | (container model) | {f4(cont['ratio'])} | | | |")
    p(f"\nContainer records: {len(cont['records'])} of the 16, WNR3 flags {', '.join(flags)}, "
      f"decoded equal to the HF source in {sum(r['equal_to_source'] for r in cont['records'].values())} of "
      f"{len(cont['records'])}.\n")

    p('## 5. Prototype coder (nvwnr.c), cross-fold unless marked in-sample, records padded to 4096 B\n')
    p('| checkpoint | configuration | stored / raw | scale payload B | weight payload B | round trips | decode MB/s, 1 thread |')
    p('|---|---|---:|---:|---:|---:|---:|')
    for c in ORDER:
        for name, v in proto.get(c, {}).items():
            s = v['summary']
            p(f"| {c} | {name} | {f4(s['ratio_total'])} | {s['scale_payload']:,} | {s['weight_payload']:,} | "
              f"{s['round_trips']}/{s['records']} | {s['decode_mb_s']:.1f} |")

    p('\n## 6. General-purpose baselines on the same slabs\n')
    p('| checkpoint | codec | stored / raw | decode MB/s, 1 thread |')
    p('|---|---|---:|---:|')
    for r in base:
        p(f"| {r['dir']} | {r['codec']} | {f4(r['ratio'])} | {r['decode_mb_s_1thread']} |")

    p('\n## 7. Whole checkpoints (projection from the 16 experts: per-expert mean x routed experts)\n')
    p('| model | format | routed experts | raw GB | best coded ratio (source) | coded GB | saved GB |')
    p('|---|---|---:|---:|---|---:|---:|')
    best_nv = {}
    for c in ORDER:
        fmt = est[c]['format']
        n = FULL[PAIR_OF[c]][0] * FULL[PAIR_OF[c]][1]
        raw = est[c]['raw_total'] / 16 * n
        if fmt == 'nvfp4':
            cross = {k: v for k, v in proto[c].items() if 'in-sample' not in k}
            exact = {k: v['summary']['ratio_total'] for k, v in cross.items() if '-cz' not in k}
            k = min(exact, key=exact.get)
            p(f"| {PAIR_OF[c]} | {c} | {n:,} | {raw / 1e9:.2f} | {f4(exact[k])} (prototype {k}, byte-exact) | "
              f"{raw * exact[k] / 1e9:.2f} | {raw * (1 - exact[k]) / 1e9:.2f} |")
            cands = {k: v['summary']['ratio_total'] for k, v in cross.items() if '-cz' in k}
            k = min(cands, key=cands.get)
            ratio, src = cands[k], f'prototype {k}, value-exact'
            best_nv[c] = (k, ratio)
        else:
            s = wnr3[c]['summary']
            cz = est[c]['neg0_nibbles'] > 0
            ratio, src = s['byte-exact']['ratio_cross_fold'], 'WNR3 byte-exact, cross-fold'
            if cz:
                p(f"| {PAIR_OF[c]} | {c} | {n:,} | {raw / 1e9:.2f} | {f4(s['cz']['ratio_cross_fold'])} (WNR3 CZ, cross-fold) | "
                  f"{raw * s['cz']['ratio_cross_fold'] / 1e9:.2f} | {raw * (1 - s['cz']['ratio_cross_fold']) / 1e9:.2f} |")
        p(f"| {PAIR_OF[c]} | {c} | {n:,} | {raw / 1e9:.2f} | {f4(ratio)} ({src}) | {raw * ratio / 1e9:.2f} | "
          f"{raw * (1 - ratio) / 1e9:.2f} |")
    n = 42 * 288
    raw = nz5['raw_total'] / 16 * n
    p(f"| glm | glm-nz5 | {n:,} | {raw / 1e9:.2f} | {f4(nz5['estimated_fraction_of_raw'])} (entropy estimate only) | "
      f"{raw * nz5['estimated_fraction_of_raw'] / 1e9:.2f} | {raw * (1 - nz5['estimated_fraction_of_raw']) / 1e9:.2f} |")
    for f in sorted(glob.glob(os.path.join(d, 'speed-*.txt'))):
        p(f'\n## Decode speed of the repository decoders: {os.path.basename(f)}\n')
        p(''.join(l for l in open(f) if l.startswith('|')))
    print('\n'.join(out))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
