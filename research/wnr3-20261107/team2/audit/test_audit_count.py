"""Check audit_count.block_contexts / unpack against a scalar loop of the rule (my own code)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_count import block_contexts, unpack, lower_median

rng = np.random.default_rng(5)
for trial in range(5):
    R, C = 5, 128
    wb = rng.integers(0, 256, size=R * C // 2, dtype=np.uint8)
    if trial == 1:  # skewed: many zeros
        wb = (wb & rng.integers(0, 256, size=wb.size, dtype=np.uint8) & 0x33).astype(np.uint8)
    sc = rng.integers(110, 135, size=R * C // 32).astype(np.uint8)
    lo, hi = lower_median(sc)
    vals = sorted(int(v) for v in sc)
    assert lo == vals[(len(vals) - 1) // 2]
    srel = (np.clip(sc.astype(np.int32) - lo, -3, 3) + 3).astype(np.int32)
    sym = unpack(wb).reshape(-1, 32)
    ctx, pm2, bmax = block_contexts(sym, srel)
    # scalar reference
    flat = []
    for b in wb:
        flat += [int(b) & 15, int(b) >> 4]
    for j in range(R * C // 32):
        s = min(max(int(sc[j]) - lo, -3), 3) + 3
        pm, mx, p2 = 8, 8, 8
        for k in range(32):
            x = flat[j * 32 + k]
            assert sym[j, k] == x
            assert ctx[j, k] == (s * 9 + pm) * 9 + mx, (j, k)
            assert pm2[j, k] == p2
            p2 = pm
            pm = x & 7
            mx = (x & 7) if mx == 8 else max(mx, x & 7)
        assert bmax[j] == mx
print('test_audit_count OK')
