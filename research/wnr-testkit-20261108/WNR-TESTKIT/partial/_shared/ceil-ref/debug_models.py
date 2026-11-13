"""Quick debug models from the team-1 counts (v41-counts) to exercise every code path before the real training.
Usage: python -I debug_models.py SAMPLE_DIR NPZ_DIR OUT_NPZ"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wnr_common import quantize, load_manifest, NCTX, NACTX, NDB

sample_dir, npz_dir, out = sys.argv[1:4]
man = load_manifest(sample_dir)
wb = sa = side = s = 0
per = []
for i, e in enumerate(man['experts'][::2]):
    z = np.load(os.path.join(npz_dir, e['file'].replace('.bin', '.npz')))
    b = z['w_bmpos'].reshape(3, NCTX, 8, 4, 16)
    a = np.zeros((3, NCTX, 2, 2, 16), np.int64)
    a[:, :, 0, 0] = b[:, :, :7, 0:2].sum((2, 3)); a[:, :, 0, 1] = b[:, :, :7, 2:4].sum((2, 3))
    a[:, :, 1, 0] = b[:, :, 7, 0:2].sum(2); a[:, :, 1, 1] = b[:, :, 7, 2:4].sum(2)
    bs = z['bm_side']
    sd = np.stack([bs[:, :, :7].sum(2), bs[:, :, 7]], 2)
    wb = wb + z['w_base']; sa = sa + a.reshape(3, NACTX, 16); side = side + sd; s = s + z['s_o0']
    per.append((e['layer'], a.reshape(3, NACTX, 16), sd))
o = {}
etab = np.full((3, NDB), 120 * 256, np.int64)


def mc(x):
    x = x.copy(); x[..., 0] += x[..., 8]; x[..., 8] = 0; return x


scf = quantize(s)
for c in (0, 1):
    W0 = mc(wb) if c else wb
    A0 = mc(sa) if c else sa
    o[f'a0_K0_c{c}_wf'] = quantize(W0).reshape(-1, 16); o[f'a0_K0_c{c}_scf'] = scf; o[f'a0_K0_c{c}_etab'] = etab
    o[f'a1_K0_c{c}_wf'] = quantize(A0).reshape(-1, 16); o[f'a1_K0_c{c}_sf'] = quantize(side).reshape(-1, 2)
    o[f'a1_K0_c{c}_scf'] = scf; o[f'a1_K0_c{c}_etab'] = etab
    for K in (2, 4):
        wk = np.zeros((K, NACTX, 16), np.int64); sk = np.zeros((K, 7, 2), np.int64)
        for lay, a, sd in per:
            for m in range(3):
                k = lay * K // 40
                wk[k] += mc(a[m]) if c else a[m]; sk[k] += sd[m]
        o[f'a1_K{K}_c{c}_wf'] = quantize(wk).reshape(-1, 16); o[f'a1_K{K}_c{c}_sf'] = quantize(sk).reshape(-1, 2)
        o[f'a1_K{K}_c{c}_scf'] = scf; o[f'a1_K{K}_c{c}_etab'] = etab
np.savez_compressed(out, **o)
print('ok', sorted(o)[:4], len(o))
