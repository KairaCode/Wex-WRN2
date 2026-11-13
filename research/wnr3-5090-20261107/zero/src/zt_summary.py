"""Print markdown tables from zero/out/*.json (zt_run.py results).  python -I zt_summary.py OUT_DIR"""
import json, os, sys

d = sys.argv[1]


def row(name, e):
    c, r, pc, o = e['canon'], e['rerun'], e['posctl'], e['obs']
    neg = f"{o.get('neg_zero_A', 0)}/{o.get('neg_zero_B', 0)}"
    by = ', '.join(f'{k}:{v}' for k, v in e.get('canon_by_act', {}).items()) or '-'
    return (f"| {name} | {c['compared']:,} | {c['diff_values']} / {c['diff_bits']} | {c['diff_signzero_only']} | "
            f"{r['diff_values']} | {pc['diff_values']:,} / {pc['diff_bits']:,} | {o.get('zero_A', 0):,} | {neg} | "
            f"{o.get('nan_A', 0):,} | {o.get('unwritten_A', 0)} | {by} |")


HDR = ('| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | '
       'rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |\n'
       '|---|---|---|---|---|---|---|---|---|---|---|')
for f in sorted(os.listdir(d)):
    if not f.endswith('.json'):
        continue
    j = json.load(open(os.path.join(d, f)))
    m = j['meta']
    print(f"\n### {f}  ({m['mode']}, {m['flavour']}, env {m.get('process_env')}, {j.get('seconds')} s)")
    if 'code8_nibbles' in m:
        print('code-8 nibbles / nibbles:', {k: (m['code8_nibbles'][k], m['nibbles'][k]) for k in m['code8_nibbles']}, 'experts:', len(m.get('experts', [])), 'shapes:', m.get('shapes'))
    res = j['results']
    if m['mode'] == 'mma':
        for op, o in res.items():
            print(f'\n**{op}**\n')
            for k, v in o.items():
                print(f'- {k}: {v}')
        continue
    groups = res if m['mode'] == 'kernels' else {'paths': res}
    for g, ents in groups.items():
        if g == 'not_run':
            print('\nnot run:', {k: sorted(set(v)) for k, v in ents.items()})
            continue
        print(f'\n{g}\n\n{HDR}')
        for name, e in ents.items():
            print(row(name, e))
