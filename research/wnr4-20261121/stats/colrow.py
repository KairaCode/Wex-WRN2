# In-sample bound: does a per-column / per-row class (two-part, from the record) add information
# beyond the scale class?  H(a|cls) vs H(a|cls,colq) vs H(a|cls,rowq) on magnitudes a (sign raw).
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data
def H(keys, a, nk):
    c = np.bincount(keys * 8 + a, minlength=nk * 8).reshape(nk, 8).astype(np.float64)
    t = c.sum(1, keepdims=True); p = np.where(c > 0, c / np.maximum(t, 1), 1)
    return -(c * np.log2(p)).sum()
ds = data.Dataset(sys.argv[1]); B = 16 if ds.nv else 32; R = 6 if ds.nv else 3
train, test = ds.split(8, int(sys.argv[2]) if len(sys.argv) > 2 else 4)
for r in test:
    coded, anc, fl = ds.coded(ds.source(r), True)
    buf = np.frombuffer(coded, np.uint8)
    out = []
    for m in range(3):
        so, rows, cols, wo = ds.geo[4*m:4*m+4]
        sc = buf[so:so+rows*cols].astype(np.int32).reshape(rows, cols)
        wb = buf[wo:wo+rows*cols*B//2]
        q = np.empty(wb.size*2, np.uint8); q[0::2] = wb & 15; q[1::2] = wb >> 4
        a = (q & 7).astype(np.int64).reshape(rows, cols*B)
        cls = (np.clip(sc - anc[m], -R, R) + R).repeat(B, axis=1)
        n = a.size
        h0 = H(cls.ravel(), a.ravel(), 2*R+1)
        cm = a.mean(0); cq = np.searchsorted(np.quantile(cm, [.25,.5,.75]), cm)   # column quartile
        rm = a.mean(1); rq = np.searchsorted(np.quantile(rm, [.25,.5,.75]), rm)
        hc = H((cls*4 + cq[None,:]).ravel(), a.ravel(), (2*R+1)*4)
        hr = H((cls*4 + rq[:,None]).ravel(), a.ravel(), (2*R+1)*4)
        cm16 = np.searchsorted(np.quantile(cm, np.linspace(0,1,17)[1:-1]), cm)
        hc16 = H((cls*16 + cm16[None,:]).ravel(), a.ravel(), (2*R+1)*16)
        out.append(f'm{m} H(a|cls) {h0/n:.4f} col4 {(h0-hc)/n:+.4f} col16 {(h0-hc16)/n:+.4f} row4 {(h0-hr)/n:+.4f} b/w gain')
    print(ds.name, r, ds.owner(r) >> 16, ' | '.join(out), flush=True)
