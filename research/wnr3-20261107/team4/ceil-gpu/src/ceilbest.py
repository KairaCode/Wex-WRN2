"""Best launch config per variant and size from ceilbench JSON files. Usage: python -I ceilbest.py OUT_TXT JSON..."""
import json, sys

rows = {}
for f in sys.argv[2:]:
    res = json.load(open(f))
    tag = f.rsplit('\\', 1)[-1].replace('.json', '')
    for cfg, d in res['results'].items():
        for v, r in d.items():
            for k, x in r['timing'].items():
                rows.setdefault((v, k), []).append((x['us_median'], f'{tag}:{cfg}', x, r))
out = []
for (v, k) in sorted(rows):
    best = min(rows[(v, k)], key=lambda t: t[0])
    us, where, x, r = best
    ratios = ' '.join(f"{kk[9:]}={vv:.3f}" for kk, vv in x.items() if kk.startswith('ratio_vs_') and kk[9:] in ('A259', 'D', 'CF0', 'CFD0', 'DB2', 'CFDB20'))
    out.append(f"{v:7s} {k:9s} best {us:8.1f} us @ {where:24s} regs {r['regs']:3d} smem {r['smem_per_block']:6d} occW {r['occ_warps_per_sm']:2d} | same-config ratios {ratios}")
txt = '\n'.join(out)
print(txt)
open(sys.argv[1], 'w').write(txt + '\n')
