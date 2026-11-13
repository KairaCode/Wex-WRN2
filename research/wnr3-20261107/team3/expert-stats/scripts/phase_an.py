"""Held-out value of the row / column phase contexts from phase_count.py (per matrix type shared model, team-1 protocol).

Usage: python -I phase_an.py SAMPLE_DIR PHASE_DIR OUT_JSON
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wnrlib import NW, quantize, cost_bits

FL = 0.015625
sample_dir, pdir, out_json = sys.argv[1:4]
man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
Z = [np.load(os.path.join(pdir, e['file'].replace('.bin', '.npz'))) for e in man['experts']]
n = len(Z)
split = np.array([i % 2 for i in range(n)])
MERGE = {8: np.arange(8), 4: np.array([0, 0, 0, 1, 2, 3, 3, 3]), 2: np.array([0, 0, 0, 0, 1, 1, 1, 1]), 1: np.zeros(8, int)}


def ev(key, nbin=None):
    ins = ho = 0.0
    used = 0
    for m in range(3):
        X = np.stack([z[key][m] for z in Z]).astype(np.int64)
        if nbin is not None:
            X = X.reshape(n, 567, 8, 16)
            Y = np.zeros((n, 567, nbin, 16), np.int64)
            for b in range(8):
                Y[:, :, MERGE[nbin][b]] += X[:, :, b]
            X = Y.reshape(n, 567 * nbin, 16)
        c = X.sum(0); a = X[split == 0].sum(0); bb = X[split == 1].sum(0)
        ins += cost_bits(c, quantize(c)); ho += cost_bits(bb, quantize(a)) + cost_bits(a, quantize(bb))
        used = max(used, int((c.sum(1) > 0).sum()))
    W = n * 3 * NW
    return {'insample': ins / W + FL, 'heldout': ho / W + FL, 'rows_used_max': used}


out = {'base_fix': ev('base_fix')}
for key in ('row8', 'col8'):
    for nb in (1, 2, 4, 8):
        out[f'{key} merged to {nb}'] = ev(key, nb)
ref = out['base_fix']['heldout']
for k, v in out.items():
    v['delta_vs_base_fix_pct'] = 100 * (v['heldout'] - ref) / ref
out['rowbin_hist_total'] = np.sum([z['rowbin_hist'] for z in Z], (0, 1)).tolist()
out['colbin_hist_total'] = np.sum([z['colbin_hist'] for z in Z], (0, 1)).tolist()
out['row_phase_std_median'] = float(np.median([z['row_phase_std'] for z in Z]))
out['col_phase_std_median'] = float(np.median([z['col_phase_std'] for z in Z]))
json.dump(out, open(out_json, 'w'), indent=1)
print(json.dumps(out, indent=1))
