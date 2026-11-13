"""Completeness / consistency check of the held-out jsonl files (team4 / ceil-ref, resumed run).
Usage: python -I check_results.py JSONL [JSONL ...]"""
import json, sys
from collections import Counter, defaultdict

TEST = [4 * l + 1 if l % 2 == 0 else 4 * l + 3 for l in range(40)]
T3 = [r'<RESEARCH-PC>\team3\ref-impl\results\heldout_A.jsonl', r'<RESEARCH-PC>\team3\ref-impl\results\heldout_B.jsonl']

rows = []
for p in sys.argv[1:]:
    rows += [json.loads(l) for l in open(p) if l.strip()]
cnt = Counter((r['idx'], r['config']) for r in rows)
cfgs = list(dict.fromkeys(r['config'] for r in rows))
idxs = sorted({r['idx'] for r in rows})
print('rows', len(rows), 'configs', len(cfgs), cfgs)
print('experts', len(idxs), '== team3 test set:', idxs == TEST)
print('duplicates', [k for k, v in cnt.items() if v > 1][:10])
missing = [(i, c) for i in TEST for c in cfgs if (i, c) not in cnt]
print('missing', missing[:20], len(missing))
bad = [(r['file'], r['config'], r['verify']) for r in rows if not r['verify']['equal_to_reference'] or r['verify']['decode_error']]
print('decode failures', len(bad), bad[:5])
print('sha bad', sum(not r['source_sha256_ok'] for r in rows))
print('fallbacks', [(r['file'], r['config'], r['ceil']) for r in rows if r['ceil'] and r['ceil']['fallback']])
print('byte_exact_vs_source by config', {c: sum(r['verify']['byte_exact_vs_source'] for r in rows if r['config'] == c) for c in cfgs})
print('rec_flags by config', {c: sorted({r['rec_flags'] for r in rows if r['config'] == c}) for c in cfgs})
print('record_bytes_check==total', all(r['sizes']['record_bytes_check'] == r['sizes']['total'] for r in rows))
print('models', Counter(r['models'] for r in rows))
# wnr2 vs team 3 real records (same model, same experts): must be identical sizes
t3 = {}
for p in T3:
    for l in open(p):
        r = json.loads(l)
        if r['config'] == 'wnr2':
            t3[r['idx']] = r
mine = {r['idx']: r for r in rows if r['config'] == 'wnr2'}
same = [i for i in mine if i in t3 and mine[i]['sizes']['total'] == t3[i]['sizes']['total']
        and mine[i]['sizes']['used'] == t3[i]['sizes']['used'] and mine[i]['model_id'] == t3[i]['model_id']]
print('wnr2 identical to team3 (total, used, model_id):', len(same), '/', len(mine), 'of', len(t3))
# anchors / per-expert f vs fg anomalies
fd = defaultdict(dict)
for r in rows:
    fd[r['idx']][r['config']] = r
lst = []
for i in idxs:
    a = fd[i]
    lst.append((a['fg']['sizes']['used'] - a['f']['sizes']['used'], a[ 'f']['file'], a['f']['anchors'], a['fg']['anchors']))
lst.sort()
print('fg-f used delta (B), worst 6:', lst[-6:])
print('fg-f used delta (B), best 4:', lst[:4])
