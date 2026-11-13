"""Why the main-trained WNR2 model fails on MTP experts: srel (scale - matrix median, clamped +-3) histograms
and the per-srel cost under the main model vs an MTP model.  Usage: python -I mtp_srel.py MTP_DATA_DIR MAIN_COUNTS_DIR"""
import json, os, sys, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_an import matrix_counts, quantize, cost_bits

d, cdir = sys.argv[1], sys.argv[2]
man = json.load(open(os.path.join(d, 'manifest.json'), encoding='utf-8'))
by = {it['name']: it for it in man['items']}
keys = sorted({tuple(map(int, re.match(r'mtp\.(\d+)\.ffn\.experts\.(\d+)\.', n).groups())) for n in by})
TW = np.zeros((3, 567, 16), np.int64)
spread = []
for (s, e) in keys:
    for m, w in enumerate(('w1', 'w2', 'w3')):
        wi = by[f'mtp.{s}.ffn.experts.{e}.{w}.weight']; si = by[f'mtp.{s}.ffn.experts.{e}.{w}.scale']
        wb = np.fromfile(os.path.join(d, wi['tag'] + '.bin'), np.uint8).reshape(wi['shape'][0], -1)
        sc = np.fromfile(os.path.join(d, si['tag'] + '.bin'), np.uint8).reshape(si['shape'][0], -1)
        wc, _, inf = matrix_counts(wb, sc)
        TW[m] += wc
        v = sc.astype(int).ravel()
        spread.append((s, e, w, inf['median_scale'], int(v.min()), int(v.max()), float(np.std(v))))
MW = sum(np.load(f)['w_base'].astype(np.int64) for f in sorted(glob.glob(os.path.join(cdir, '*.npz'))))
wq = quantize(MW); tq = quantize(TW)
for nm, C in (('main', MW), ('mtp', TW)):
    h = C.reshape(3, 7, 81, 16).sum((0, 2, 3))
    print(nm, 'srel hist (frac):', np.round(h / h.sum(), 4).tolist())
for k in range(7):
    sl = slice(k * 81, (k + 1) * 81)
    n = TW[:, sl].sum()
    if n:
        print(f'srel {k - 3:+d}: mtp weights {n / TW.sum():.4f}  bits/w main-model {cost_bits(TW[:, sl], wq[:, sl]) / n:.3f}  mtp-model(in-sample) {cost_bits(TW[:, sl], tq[:, sl]) / n:.3f}')
print('per matrix scale spread (stage, expert, w, median, min, max, std):')
for r in spread:
    print(r)
