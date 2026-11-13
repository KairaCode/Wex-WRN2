"""Held-out: anchor fix + row-phase bins on (a) c_bm2h and (c) c_z, team-1 protocol, side bits added (held-out).

Usage: python -I phase_combo_an.py SAMPLE_DIR PHASECOMBO_DIR OUT_JSON
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wnrlib import NW, quantize, cost_bits

FL = 0.015625
REF = 3.782638472270624   # team-1 WNR2 reference (pooled K=1 held-out, "3.7826")
sample_dir, pdir, out_json = sys.argv[1:4]
man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
Z = [np.load(os.path.join(pdir, e['file'].replace('.bin', '.npz'))) for e in man['experts']]
n = len(Z); W = n * 3 * NW
split = np.array([i % 2 for i in range(n)])
MERGE = {8: np.arange(8), 4: np.array([0, 0, 0, 1, 2, 3, 3, 3]), 2: np.array([0, 0, 0, 0, 1, 1, 1, 1]), 1: np.zeros(8, int)}


def ev_key(key, rows, nbin):
    ins = ho = 0.0; used = 0
    for m in range(3):
        X = np.stack([z[key][m] for z in Z]).astype(np.int64)
        if key != 'side_fix':
            X = X.reshape(n, rows, 8, 16)
            Y = np.zeros((n, rows, nbin, 16), np.int64)
            for b in range(8):
                Y[:, :, MERGE[nbin][b]] += X[:, :, b]
            X = Y.reshape(n, rows * nbin, 16)
        c = X.sum(0); a = X[split == 0].sum(0); bb = X[split == 1].sum(0)
        ins += cost_bits(c, quantize(c)); ho += cost_bits(bb, quantize(a)) + cost_bits(a, quantize(bb))
        used = max(used, int((c.sum(1) > 0).sum()))
    return ins, ho, used


si, sh, _ = ev_key('side_fix', None, None)
out = {'side_bits_per_w': si / W}
for name, key, rows in [('(a) c_bm2h + anchor fix', 'h_row8', 2268), ('(c) c_z + anchor fix (value-exact)', 'z_row8', 4536)]:
    for nb in (1, 2, 4, 8):
        i, h, used = ev_key(key, rows, nb)
        ho = (h + sh) / W + FL
        out[f'{name} x rowbin{nb}'] = {'insample': (i + si) / W + FL, 'heldout': ho, 'delta_vs_ref_pct': 100 * (ho - REF) / REF,
                                       'rows_used_max': used, 'table_kb_compacted': used * 32 / 1024}
json.dump(out, open(out_json, 'w'), indent=1)
print(json.dumps(out, indent=1))
