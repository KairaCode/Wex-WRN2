# Is the column magnitude profile shared by the experts of a layer?  Column classes from K other experts
# of the same layer (or of layer L+1) vs in-sample; gain of H(a|cls,colq) over H(a|cls) on held-out records.
import sys, os, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data
def H(keys, a, nk):
    c = np.bincount(keys * 8 + a, minlength=nk * 8).reshape(nk, 8).astype(np.float64)
    t = c.sum(1, keepdims=True); p = np.where(c > 0, c / np.maximum(t, 1), 1)
    return -(c * np.log2(p)).sum()
ds = data.Dataset(sys.argv[1]); B = 16 if ds.nv else 32; R = 6 if ds.nv else 3
NQ = int(sys.argv[3]) if len(sys.argv) > 3 else 8
by = {}
for r in ds.experts: by.setdefault(ds.owner(r) >> 16, []).append(r)
def mags(r):
    coded, anc, fl = ds.coded(ds.source(r), True)
    buf = np.frombuffer(coded, np.uint8); res = []
    for m in range(3):
        so, rows, cols, wo = ds.geo[4*m:4*m+4]
        sc = buf[so:so+rows*cols].astype(np.int32).reshape(rows, cols)
        wb = buf[wo:wo+rows*cols*B//2]
        q = np.empty(wb.size*2, np.uint8); q[0::2] = wb & 15; q[1::2] = wb >> 4
        a = (q & 7).astype(np.int64).reshape(rows, cols*B)
        cls = (np.clip(sc - anc[m], -R, R) + R).repeat(B, axis=1)
        res.append((a, cls))
    return res
rng = random.Random(1)
layers = sorted(by); K = int(sys.argv[2])
for L in [layers[0], layers[len(layers)//2], layers[-1]]:
    pool = rng.sample(by[L], K + 2)
    ref, tests = pool[:K], pool[K:]
    prof = None
    for r in ref:
        mm = [a.astype(np.float64).mean(0) for a, _ in mags(r)]
        prof = mm if prof is None else [p + x for p, x in zip(prof, mm)]
    for r in tests:
        out = []
        for m, (a, cls) in enumerate(mags(r)):
            n = a.size; h0 = H(cls.ravel(), a.ravel(), 2*R+1)
            def gain(cm):
                cq = np.searchsorted(np.quantile(cm, np.linspace(0,1,NQ+1)[1:-1]), cm)
                return (h0 - H((cls*NQ + cq[None,:]).ravel(), a.ravel(), (2*R+1)*NQ)) / n
            out.append(f'm{m} in {gain(a.mean(0)):+.4f} layer{K} {gain(prof[m]):+.4f}')
        print(ds.name, 'L', L, r, ' | '.join(out), flush=True)
