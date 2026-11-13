"""Compact table from ceilbench.py JSON. Usage: python -I ceilsumm.py BENCH_JSON [OUT_TXT]"""
import json, sys

res = json.load(open(sys.argv[1]))
out = []
P = out.append
P(f"nvsmi start/end: {res.get('nvsmi_start')} | {res.get('nvsmi_end')}  rounds {res['rounds']}")
allx = [(c, v, r) for c, d in res['results'].items() for v, r in d.items()]
P('byte-exact (weights+scales vs original) for all non-diagnostic variants: ' + str(all(
    r['byte_exact_vs_original'] and all(r[k]['weight_bad_streams'] == 0 and r[k]['scale_bad_streams'] == 0 and r[k]['errors'] == 0
                                        for k in ('multistream_recheck', 'm1_recheck', 'm3_recheck'))
    for c, v, r in allx if v not in ('CF0', 'CFD0', 'CF3', 'CF4', 'CFDB20'))))
P('pipeline checks all clean: ' + str(all(r[k]['weight_bad_streams'] == 0 and r[k]['scale_bad_streams'] == 0 and r[k]['errors'] == 0
                                          for c, v, r in allx for k in ('pipeline_check_single', 'pipeline_check_multi') if k in r)))
OWN = {'CF': 'CF0', 'CFw': 'CF0', 'CF5': 'CF0', 'CF6': 'CF0', 'CF3': 'CF0', 'CF4': 'CF0',
       'CFD': 'CFD0', 'CFDw': 'CFD0', 'CFD5': 'CFD0', 'CFD6': 'CFD0', 'CFDB2': 'CFDB20', 'CFDB26': 'CFDB20'}
for c, d in res['results'].items():
    P(f'== {c}')
    for v, r in d.items():
        t = r['timing']
        s = f"{v:6s} regs {r['regs']:3d} lmem {r['local_bytes']} smem {r['smem_per_block']:6d} occW {r['occ_warps_per_sm']:2d} exact {str(r['byte_exact_vs_original'])[0]} |"
        for k in ('m1', 'm3', 'm24', '8streams'):
            x = t[k]
            s += f" {k} {x['us_median']:7.1f}"
            if 'ratio_vs_A259' in x: s += f" A{x['ratio_vs_A259']:.3f}"
            if 'ratio_vs_D' in x: s += f" D{x['ratio_vs_D']:.3f}"
            if v in OWN and f'ratio_vs_{OWN[v]}' in x: s += f" own{x['ratio_vs_' + OWN[v]]:.3f}"
            if 'ratio_vs_DB2' in x: s += f" DB2 {x['ratio_vs_DB2']:.3f}"
            s += ' |'
        P(s)
        if 'pipeline_timing' in r:
            t = r['pipeline_timing']
            s = f"{'':6s} PIPE (sdec<32> + wdec, in-place scale patch) |"
            for k in ('m1', 'm3', 'm24', '8streams'):
                x = t[k]
                s += f" {k} {x['us_median']:7.1f}"
                if 'ratio_vs_A259' in x: s += f" A{x['ratio_vs_A259']:.3f}"
                if 'ratio_vs_D' in x: s += f" D{x['ratio_vs_D']:.3f}"
                s += ' |'
            P(s)
txt = '\n'.join(out)
print(txt)
if len(sys.argv) > 2:
    open(sys.argv[2], 'w').write(txt + '\n')
