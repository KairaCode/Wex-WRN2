"""Characterise the accepted single-bit-flip cases: which symbols swap, their frequencies, context rows."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import cum_of, encode_vec, decode_vec, inv_table
from audit_count import read_expert
from audit_corrupt import streams_of

sample_dir, rans_dir = sys.argv[1:3]
ei, m = int(sys.argv[3]), int(sys.argv[4])
with open(os.path.join(sample_dir, 'manifest.json'), 'r', encoding='utf-8') as f:
    man = json.load(f)
wf = np.load(os.path.join(rans_dir, 'model.npz'))['wf']; wc = cum_of(wf); inv = inv_table(wf[m])
tens, _ = read_expert(sample_dir, man['experts'][ei])
sym, ctx, srel = streams_of(tens, m, wf, wc)
sym = sym.astype(np.int64)
N = sym.shape[0]
F = wf[m][ctx, sym]; C = wc[m][ctx, sym]
lens, _, buf = encode_vec(F, C, want_bytes=True)
rng = np.random.default_rng(7)
ar = np.arange(N)
pairs = {}
bits = {}
tot = acc = 0
for rep in range(4):
    b = buf.copy(); j = (rng.random(N) * lens).astype(np.int64); bit = rng.integers(0, 8, N)
    b[ar, j] ^= (1 << bit).astype(np.uint8)
    v, ds, _ = decode_vec(b, lens, 2048, srel, wf[m], wc[m], inv)
    tot += N; acc += int(v.sum())
    for k in np.nonzero(v)[0]:
        d = np.nonzero(ds[k] != sym[k])[0]
        i = int(d[0])
        c = int(ctx[k, i]); s0 = int(sym[k, i]); s1 = int(ds[k, i])
        key = (s0, s1, int(wf[m][c, s0]), int(wf[m][c, s1]))
        pairs[key] = pairs.get(key, 0) + 1
        bits[int(bit[k])] = bits.get(int(bit[k]), 0) + 1
print('accepted', acc, 'of', tot, acc / tot)
print('flipped bit histogram', dict(sorted(bits.items())))
print('(orig sym, decoded sym, f_orig, f_decoded): count')
for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])[:20]:
    print(k, v)
# how many context rows have two symbols s, s^8 (same magnitude, opposite sign) with equal freq that is a power of two <= 128?
f = wf[m]
eq = [(r, s) for r in range(567) for s in range(8) if f[r, s] == f[r, s + 8]]
print('rows x magnitudes with f(+x) == f(-x):', len(eq), 'of', 567 * 8)
