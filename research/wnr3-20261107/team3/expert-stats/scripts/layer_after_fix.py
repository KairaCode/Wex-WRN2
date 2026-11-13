"""Per-layer LOO curve (team-1 definition) with the median anchor vs the densest-pair anchor.

Usage: python -I layer_after_fix.py SAMPLE_DIR COUNTS_DIR OUT_JSON
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wnrlib import NW, quantize, cost_bits, load_all

FL = 0.015625
sample_dir, counts_dir, out_json = sys.argv[1:4]
ex = load_all(sample_dir, counts_dir, keys=('w_srel15', 's_o0', 'meta'))
n = len(ex)
lay = np.array([e['layer'] for e in ex])
out = {}
for name in ('median', 'pair'):
    WB = np.zeros((n, 3, 567, 16), np.int64)
    for i, e in enumerate(ex):
        for m in range(3):
            h = e['s_o0'][m]; med = int(e['meta'][m, 2])
            A = int((h[:-1] + h[1:]).argmax()) if name == 'pair' else med
            X = e['w_srel15'][m].reshape(15, 81, 16)
            Y = np.zeros((7, 81, 16), np.int64)
            for j in range(15):
                Y[int(np.clip(j - 7 - (A - med), -3, 3) + 3)] += X[j]
            WB[i, m] = Y.reshape(567, 16)
    tot = WB.sum(0)
    curve = []; sh_all = ly_all = 0.0
    for L in sorted(set(lay)):
        idx = np.where(lay == L)[0]
        cl = WB[idx].sum(0)
        s = sum(cost_bits(WB[i], quantize(tot - WB[i])) for i in idx)
        l = sum(cost_bits(WB[i], quantize(cl - WB[i])) for i in idx)
        sh_all += s; ly_all += l
        nW = len(idx) * 3 * NW
        curve.append({'layer': int(L), 'shared_loo': round(s / nW, 4), 'layer_loo': round(l / nW, 4), 'gap': round((l - s) / nW, 4)})
    W = n * 3 * NW
    groups = {}
    for G in (2, 4, 8):
        gl = 0.0
        for g in range(G):
            idx = np.where(lay * G // 40 == g)[0]
            cg = WB[idx].sum(0)
            gl += sum(cost_bits(WB[i], quantize(cg - WB[i])) for i in idx)
        groups[G] = gl / W + FL
    out[name] = {'shared_loo_bpw': sh_all / W + FL, 'per_layer_loo_bpw': ly_all / W + FL, 'layer_group_loo_bpw': groups,
                 'max_gap_layer': max(curve, key=lambda c: c['gap']), 'min_gap_layer': min(curve, key=lambda c: c['gap']), 'curve': curve}
    print(name, json.dumps({k: v for k, v in out[name].items() if k != 'curve'}), flush=True)
json.dump(out, open(out_json, 'w'), indent=1)
for c0, c1 in zip(out['median']['curve'], out['pair']['curve']):
    if abs(c0['gap']) > 0.005 or abs(c1['gap']) > 0.005:
        print(c0, c1)
