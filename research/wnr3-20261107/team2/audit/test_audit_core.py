import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import (quantize, quantize_scalar, cum_of, encode_vec, decode_vec, inv_table,
                        scalar_encode, scalar_decode, L, P)

rng = np.random.default_rng(3)
# quantizer vs scalar reference, including skewed, tied and empty rows
rows = [rng.integers(0, 1000, 16) for _ in range(200)] + [rng.integers(0, 3, 16) * 7 for _ in range(50)]
rows += [np.zeros(16, np.int64), np.array([5] * 16), np.array([10**9] + [0] * 15), np.array([1, 1] + [0] * 14)]
rows += [rng.integers(0, 10**8, 2) for _ in range(50)] + [rng.integers(0, 50, 256) for _ in range(20)]
for r in rows:
    q = quantize(np.asarray(r)[None, :])[0]
    assert list(q) == quantize_scalar(list(r)), r
# rANS: random model over 567 contexts, real-looking symbol skew
counts = rng.integers(0, 50, size=(567, 16)) ** 2
freq = quantize(counts)
cum = cum_of(freq)
inv = inv_table(freq)
N, n = 64, 2048
srel = rng.integers(0, 7, size=(N, n // 32))
syms = rng.choice(16, size=(N, n), p=np.bincount(rng.integers(0, 16, 400) % 16, minlength=16) / 400)
ctx = np.empty((N, n), np.int64)
for s in range(N):
    for j in range(n // 32):
        pm = mx = 8
        for k in range(32):
            i = j * 32 + k
            ctx[s, i] = (srel[s, j] * 9 + pm) * 9 + mx
            m = syms[s, i] & 7
            pm = m
            mx = m if mx == 8 else max(mx, m)
F = freq[ctx, syms]
C = cum[ctx, syms]
lens, xf, by = encode_vec(F, C, want_bytes=True)
lens2, _, _ = encode_vec(F, C)
assert (lens == lens2).all()
for s in range(4):
    b = scalar_encode(syms[s].tolist(), ctx[s].tolist(), freq.tolist(), cum.tolist())
    assert len(b) == lens[s] and bytes(by[s, :lens[s]]) == b, s
    d = scalar_decode(b, n, srel[s], freq.tolist(), cum.tolist())
    assert d == syms[s].tolist()
valid, ds, x = decode_vec(by, lens, n, srel, freq, cum, inv)
assert valid.all() and (ds == syms).all() and (x == L).all()
# x0 = L + h variant
h = rng.integers(0, 1 << 23, N)
lh, _, byh = encode_vec(F, C, x0=L + h, want_bytes=True)
v, ds2, xh = decode_vec(byh, lh, n, srel, freq, cum, inv)
assert (ds2 == syms).all() and (xh == L + h).all() and (~v).all()  # today's rule rejects L+h streams
print('test_audit_core OK; mean bytes', lens.mean(), 'with hash', lh.mean())
