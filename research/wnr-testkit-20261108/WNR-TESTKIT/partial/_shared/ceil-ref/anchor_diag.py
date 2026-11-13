"""G-anchor diagnostics on ceil-form (team4 / ceil-ref). Usage: python -I anchor_diag.py SAMPLE_DIR COUNTS_DIR JSONL [JSONL]"""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr_common as W

sample_dir, counts_dir = sys.argv[1:3]
man = W.load_manifest(sample_dir)
tr = {'wnr2': Counter(), 'f': Counter(), 'fg': Counter()}
for i, e in enumerate(man['experts']):
    if i % 2:
        continue
    z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
    for k in tr:
        for m, a in enumerate(z[k + '_anchor'].tolist()):
            tr[k][(W.MAT[m], a)] += 1
print('train anchors (matrix, value): count', {k: sorted(v.items()) for k, v in tr.items()})
rows = []
for p in sys.argv[3:]:
    rows += [json.loads(l) for l in open(p) if l.strip()]
te = {c: Counter() for c in ('wnr2', 'f', 'fg')}
for r in rows:
    if r['config'] in te:
        for m, a in enumerate(r['anchors']):
            te[r['config']][(W.MAT[m], a)] += 1
print('test anchors', {k: sorted(v.items()) for k, v in te.items()})
for f in ('L39_E156.bin', 'L24_E058.bin', 'L17_E123.bin'):
    e = [x for x in man['experts'] if x['file'] == f][0]
    raw, mats = W.load_expert(sample_dir, e)
    syms, scs, anchors, finfo = W.prepare(mats, True, False, True)
    for m in range(3):
        h = np.bincount(scs[m], minlength=256)
        ho = np.bincount(mats[m]['sc'], minlength=256)
        print(f, W.MAT[m], 'G anchor', anchors[m], 'lower median (transformed)', W.lower_median(scs[m]),
              'transformed hist 117..124', h[117:125].tolist(), 'pair sums 119/120/121', [int(h[v] + h[v + 1]) for v in (119, 120, 121)],
              'orig hist 117..124', ho[117:125].tolist())
