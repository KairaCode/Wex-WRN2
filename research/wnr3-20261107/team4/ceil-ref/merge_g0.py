"""Build the g0 (constant anchor 120) held-out set: fg* rows of every test expert whose G anchors are already
[120,120,120] are reused as is; the other experts' fg* rows come from run_fixed_anchor.py output (team4 / ceil-ref).
Usage: python -I merge_g0.py OUT_JSONL FIXED_JSONL HELDOUT_JSONL [HELDOUT_JSONL]"""
import json, sys
out, fixed = sys.argv[1:3]
rows = []
for p in sys.argv[3:]:
    rows += [json.loads(l) for l in open(p) if l.strip()]
fx = {(r['idx'], r['config']): r for r in (json.loads(l) for l in open(fixed) if l.strip())}
need = sorted({r['idx'] for r in rows if r['config'].startswith('fg') and r['anchors'] != [120, 120, 120]})
print('experts whose G anchors are not all 120:', need)
n_rep = 0
with open(out, 'w') as fo:
    for r in rows:
        if r['config'].startswith('fg') and r['idx'] in need:
            r2 = fx[(r['idx'], r['config'])]
            assert r2['anchors'] == [120, 120, 120] and r2['verify']['equal_to_reference'] and not r2['verify']['decode_error']
            print(r['file'], r['config'], 'G', r['sizes']['total'], '-> const120', r2['sizes']['total'], r2['sizes']['total'] - r['sizes']['total'])
            r = r2
            n_rep += 1
        fo.write(json.dumps(r) + '\n')
print('replaced', n_rep)
