"""Quick look at ceil-form scale statistics (row/col means) on a few experts to set bucket thresholds."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxcount import load_expert
from mxceil import forward

SD = os.environ.get('WNR_SAMPLE', 'v41-sample')  # kit: dir with the 160-expert sample (manifest.json + L*_E*.bin)
man = json.load(open(os.path.join(SD, 'manifest.json')))
for idx in [0, 41, 80, 159]:
    e = man['experts'][idx]
    t = load_expert(os.path.join(SD, e['file']), e['tensors'])
    for m, w in enumerate(['w1', 'w2', 'w3']):
        wraw, wshape, _ = t[f'{w}.weight']; sraw, _, _ = t[f'{w}.scale']
        R = wshape[0]; wb = wraw.reshape(R, -1); sc = sraw.reshape(R, -1)
        C = wb.shape[1] * 2
        sym = np.empty((R, C), np.uint8); sym[:, 0::2] = wb & 15; sym[:, 1::2] = wb >> 4
        tsym, tsc, even, bmax = forward(sym, sc)
        s = tsc.astype(np.float64)
        rm = s.mean(1) - 120; cm = s.mean(0) - 120
        h = np.bincount(tsc.ravel().astype(int), minlength=256)
        nz = np.nonzero(h)[0]
        print(e['file'], w, 'scale hist', {int(k): round(h[k] / h.sum(), 4) for k in nz},
              'rowmean q', np.round(np.quantile(rm, [0, .125, .25, .375, .5, .625, .75, .875, 1]), 3),
              'colmean q', np.round(np.quantile(cm, [0, .125, .25, .375, .5, .625, .75, .875, 1]), 3))
        mag = (tsym.reshape(R, -1, 32) & 7)
        print('   mag mean', round(float(mag.mean()), 3), 'block sum q', np.quantile(mag.sum(2), [.1, .25, .5, .75, .9]),
              'zeros/block q', np.quantile((mag == 0).sum(2), [.1, .25, .5, .75, .9]))
