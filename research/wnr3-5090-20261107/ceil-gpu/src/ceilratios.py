"""Collect paired ratios of the ceil-form kernels from ceilbench JSON files.
Usage: python -I ceilratios.py OUT_JSON BENCH_JSON...
For each file/config/size: ratio(X vs Y) = median over rounds of t_X / t_Y (same round, same config)."""
import json, sys

PAIRS = [('CF3', 'CF0'), ('CF5', 'CF0'), ('CF6', 'CF0'), ('CF', 'CF0'), ('CFD5', 'CFD0'), ('CFDB2', 'CFDB20'),
         ('CF5', 'A259'), ('CF', 'A259'), ('CF0', 'A259'), ('CF5', 'D'), ('CFD5', 'D'), ('CFD5', 'A259'),
         ('CFDB2', 'DB2'), ('CFDB2', 'A259'), ('CFDB2', 'D'), ('DB2', 'A259'), ('D', 'A259')]
SIZES = ('m1', 'm3', 'm24', '8streams')
out = {}
for f in sys.argv[2:]:
    res = json.load(open(f))
    tag = f.rsplit('\\', 1)[-1].replace('.json', '')
    for cfg, d in res['results'].items():
        for x, y in PAIRS:
            if x in d and y in d:
                for s in SIZES:
                    r = d[x]['timing'][s].get(f'ratio_vs_{y}')
                    if r is not None:
                        out.setdefault(f'{x}/{y}', {}).setdefault(s, []).append((tag, cfg, round(r, 4)))
                if 'pipeline_timing' in d[x] and 'pipeline_timing' in d[y]:
                    for s in SIZES:
                        r = d[x]['pipeline_timing'][s].get(f'ratio_vs_{y}')
                        if r is not None:
                            out.setdefault(f'PIPE {x}/{y}', {}).setdefault(s, []).append((tag, cfg, round(r, 4)))
summ = {}
for k, v in out.items():
    summ[k] = {}
    for s, lst in v.items():
        rs = sorted(t[2] for t in lst)
        summ[k][s] = {'n': len(rs), 'min': rs[0], 'median': rs[len(rs) // 2], 'max': rs[-1]}
    line = ' | '.join(f"{s} n{summ[k][s]['n']} {summ[k][s]['min']:.3f}/{summ[k][s]['median']:.3f}/{summ[k][s]['max']:.3f}"
                      for s in SIZES if s in summ[k])
    print(f'{k:18s} {line}')
json.dump({'summary_min_median_max': summ, 'all': out}, open(sys.argv[1], 'w'), indent=1)
