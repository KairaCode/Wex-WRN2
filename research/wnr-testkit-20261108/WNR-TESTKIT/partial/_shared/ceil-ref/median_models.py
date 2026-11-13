"""Estimate (code lengths from own counts, held-out: train even-indexed experts, test odd-indexed):
model chosen by the stored per-matrix median scale (no side info) vs K=4 k-means models (2-bit id per matrix).
Usage: python -I median_models.py SAMPLE_DIR COUNTS_DIR MODELS_DIR"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wnr_common import load_manifest, quantize, P, NACTX

sample_dir, counts_dir, models_dir = sys.argv[1:4]
man = load_manifest(sample_dir)
ex = []
for i, e in enumerate(man['experts']):
    z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
    ex.append({'i': i, **{k: z[k] for k in z.files}})
tr = [e for e in ex if e['i'] % 2 == 0]
te = [e for e in ex if e['i'] % 2 == 1]
Wm = 2304 * 5120


def L(f):
    return -np.log2(f / P)


def costs(models_w, models_s, pick):
    tot = 0.0
    for e in te:
        for m in range(3):
            k = pick(e, m)
            tot += (e['a'][m] * models_w[k]).sum() + (e['side'][m] * models_s[k]).sum()
    return tot


res = {}
# (1) a, per type
wa = {m: L(quantize(sum(e['a'][m] for e in tr))) for m in range(3)}
sa = {m: L(quantize(sum(e['side'][m] for e in tr))) for m in range(3)}
res['a_per_type'] = costs(wa, sa, lambda e, m: m)
# (2) a, per type x (median == 120 or not)
g = lambda e, m: (m, int(e['med'][m] != 120))
wg, sg = {}, {}
for m in range(3):
    for h in (0, 1):
        sel = [e for e in tr if int(e['med'][m] != 120) == h]
        wg[(m, h)] = L(quantize(sum(e['a'][m] for e in sel))) if sel else wa[m]
        sg[(m, h)] = L(quantize(sum(e['side'][m] for e in sel))) if sel else sa[m]
res['a_per_type_x_median'] = costs(wg, sg, g)
# (3) a, pooled types x median group
wp, sp = {}, {}
for h in (0, 1):
    wp[h] = L(quantize(sum(e['a'][m] for e in tr for m in range(3) if int(e['med'][m] != 120) == h)))
    sp[h] = L(quantize(sum(e['side'][m] for e in tr for m in range(3) if int(e['med'][m] != 120) == h)))
res['a_pooled_x_median'] = costs(wp, sp, lambda e, m: int(e['med'][m] != 120))
# (4) K=4 k-means (trained models), best per matrix
Z = np.load(os.path.join(models_dir, 'even.npz'))
wk = L(Z['a1_K4_c0_wf'].reshape(4, NACTX, 16))
sk = L(Z['a1_K4_c0_sf'].reshape(4, 7, 2))
tot = 0.0
for e in te:
    for m in range(3):
        tot += min((e['a'][m] * wk[k]).sum() + (e['side'][m] * sk[k]).sum() for k in range(4))
res['aK4_kmeans_best'] = tot
nW = len(te) * 3 * Wm
base = res['a_per_type']
nmed = sum(int(e['med'][m] != 120) for e in te for m in range(3))
out = {'test_matrices': len(te) * 3, 'test_matrices_median_not_120': nmed,
       'train_matrices_median_not_120': sum(int(e['med'][m] != 120) for e in tr for m in range(3)),
       **{k: {'payload_bits_per_w_no_flush': v / nW, 'delta_vs_a_pct': 100 * (v - base) / base} for k, v in res.items()}}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(models_dir, 'median_models.json'), 'w'), indent=1)
