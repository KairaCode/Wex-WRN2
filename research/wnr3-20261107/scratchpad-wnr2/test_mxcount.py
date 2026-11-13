"""Check mxcount.matrix_stats against a plain loop of WNR2's context rule (wex_neural_rans.h).

Rule: symbols row-major, symbol 2k in the low nibble. At each 32-block start: srel = clamp(s - med, -3, 3) + 3,
pm = 8, mx = 8. Context = (srel*9 + pm)*9 + mx. After symbol x: pm = x&7, mx = max(mx, x&7) where mx=8 means none.
med = lower median of the matrix scales.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxcount import matrix_stats

rng = np.random.default_rng(1)
R, C = 7, 96
wb = rng.integers(0, 256, size=(R, C // 2), dtype=np.uint8)
sc = rng.integers(118, 130, size=(R, C // 32), dtype=np.uint8)
r, *_ = matrix_stats(wb, sc)

flat = sorted(int(v) for v in sc.ravel())
med = flat[(len(flat) - 1) // 2]
ref = np.zeros((567, 16), np.int64)
bm_side = np.zeros((7, 8), np.int64)
for i in range(R):
    syms = []
    for b in wb[i]:
        syms += [int(b) & 15, int(b) >> 4]
    for j in range(C // 32):
        srel = min(max(int(sc[i, j]) - med, -3), 3) + 3
        pm, mx = 8, 8
        for k in range(32):
            x = syms[j * 32 + k]
            ref[(srel * 9 + pm) * 9 + mx, x] += 1
            pm = x & 7
            mx = (x & 7) if mx == 8 else max(mx, x & 7)
        bm_side[srel, mx] += 1
assert (ref == r['w_base']).all(), 'w_base mismatch'
assert (bm_side == r['bm_side']).all(), 'bm_side mismatch'
assert r['meta'][2] == med
# every variant must hold the same total
n = R * C
for k in ['w_base', 'w_srel15', 'w_pos', 'w_pm2', 'w_psign', 'w_bm', 'w_bmpos', 'w_up', 'w_upsort']:
    assert r[k].sum() == n, k
    # marginalising the extra context must give back w_base where the layout allows it
for k, extra in [('w_pos', 32), ('w_pm2', 9), ('w_psign', 3), ('w_bm', 8), ('w_up', 9), ('w_upsort', 9)]:
    if k == 'w_upsort':
        continue  # row-sorted: same multiset of contexts, checked by totals per context below
    assert (r[k].reshape(567, extra, 16).sum(1) == r['w_base']).all(), k
assert (r['w_upsort'].reshape(567, 9, 16).sum(1) == r['w_base']).all(), 'w_upsort marginal'
assert (r['w_bmpos'].reshape(567, 8, 4, 16).sum((1, 2)) == r['w_base']).all(), 'w_bmpos'
for k in ['s_left', 's_up', 's_lu', 's_lusort']:
    assert (r[k].sum(0) == r['s_o0']).all(), k
print('test_mxcount OK')
