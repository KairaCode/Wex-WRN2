"""Diagnose the per-layer LOO anomaly (layer 15 and others): per expert/matrix median, upper-share, LOO costs,
and where the per-layer-LOO excess comes from (srel bucket of the TEST data, zero-count cells).

Usage: python -I layer_diag.py SAMPLE_DIR COUNTS_DIR UNITS_JSON OUT_JSON
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wnrlib import NW, quantize, codelen, cost_bits, load_all

sample_dir, counts_dir, units_json, out_json = sys.argv[1:5]
ex = load_all(sample_dir, counts_dir, keys=('w_base', 's_o0', 'meta'))
WB = np.stack([e['w_base'] for e in ex]); SO = np.stack([e['s_o0'] for e in ex])
lay = np.array([e['layer'] for e in ex]); n = len(ex)
tot = WB.sum(0)


def two_val(h, med):
    """share of scales at med-1, med, med+1 and the 'flip' flag: median is the UPPER of the two dominant adjacent values."""
    t = h.sum()
    sh = {d: float(h[med + d] / t) for d in (-2, -1, 0, 1, 2)}
    flip = sh[-1] > sh[1]
    return sh, flip


out = {'layers': {}}
summary = []
for L in sorted(set(lay)):
    idx = [i for i in range(n) if lay[i] == L]
    rows = []
    layer_loo = shared_loo = 0.0
    for i in idx:
        for m in range(3):
            c = WB[i, m]
            others = [j for j in idx if j != i]
            T = WB[others, m].sum(0)
            Ql = quantize(T); Qs = quantize(tot[m] - c)
            ll = -np.log2(Ql / 16384); ls = -np.log2(Qs / 16384)
            bl = (c * ll).sum(); bs = (c * ls).sum()
            layer_loo += bl; shared_loo += bs
            med = int(ex[i]['meta'][m, 2])
            sh, flip = two_val(SO[i, m], med)
            d = (c * (ll - ls)).reshape(7, 81, 16).sum((1, 2)) / NW     # excess bit/w by srel bucket
            unseen = float((c * ll * (T == 0)).sum() / NW)                 # bits on cells never seen in training
            # flips among the training experts for this matrix type
            tflip = [two_val(SO[j, m], int(ex[j]['meta'][m, 2]))[1] for j in others]
            rows.append({'file': ex[i]['file'], 'm': m, 'med': med, 'share_m1': round(sh[-1], 4), 'share_0': round(sh[0], 4),
                         'share_p1': round(sh[1], 4), 'flip': bool(flip), 'train_flips': int(sum(tflip)),
                         'bpw_layer_loo': round(float(bl / NW), 4), 'bpw_shared_loo': round(float(bs / NW), 4),
                         'excess_by_srel': [round(float(x), 4) for x in d], 'bits_on_unseen_cells_bpw': round(unseen, 5)})
    nW = len(idx) * 3 * NW
    out['layers'][int(L)] = rows
    nflip = sum(r['flip'] for r in rows)
    summary.append({'layer': int(L), 'layer_loo': round(layer_loo / nW, 4), 'shared_loo': round(shared_loo / nW, 4),
                    'gap': round((layer_loo - shared_loo) / nW, 4), 'flipped_matrices_of_12': int(nflip)})
out['summary'] = summary
# global flip census
fl = []
for i in range(n):
    for m in range(3):
        sh, flip = two_val(SO[i, m], int(ex[i]['meta'][m, 2]))
        fl.append({'file': ex[i]['file'], 'layer': int(lay[i]), 'm': m, 'flip': bool(flip),
                   'upper_share': sh[0] if flip else sh[1], 'lower_share': sh[-1] if flip else sh[0]})
out['flip_census'] = {'flipped_units': int(sum(f['flip'] for f in fl)), 'of': len(fl),
                      'by_matrix': [int(sum(f['flip'] for f in fl if f['m'] == m)) for m in range(3)],
                      'flipped_list': [(f['file'], f['m'], round(f['upper_share'], 3)) for f in fl if f['flip']]}
up = np.array([f['upper_share'] for f in fl]); lo = np.array([f['lower_share'] for f in fl])
out['upper_share_dist'] = {'min': float(up.min()), 'p5': float(np.percentile(up, 5)), 'median': float(np.median(up)),
                           'p95': float(np.percentile(up, 95)), 'max': float(up.max()),
                           'two_value_coverage_min': float((up + lo).min()), 'two_value_coverage_median': float(np.median(up + lo)),
                           'hist_0.05': np.histogram(up, bins=np.arange(0, 1.0001, 0.05))[0].tolist()}
json.dump(out, open(out_json, 'w'), indent=1)
print(json.dumps(summary))
print(json.dumps(out['flip_census']))
print(json.dumps(out['upper_share_dist']))
for L in [15, 24, 30, 32, 34, 37, 38, 39]:
    print('--- layer', L)
    for r in out['layers'][L]:
        print(r)
