# In-sample conditional entropies (bits per nibble / per scale) of candidate contexts, many records pooled.
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data
def H(keys, sym, nsym):
    k = keys.astype(np.int64) * nsym + sym
    u, c = np.unique(k, return_counts=True)
    ctx = u // nsym
    cu, inv = np.unique(ctx, return_inverse=True)
    tot = np.bincount(inv, weights=c)
    return -(c * np.log2(c / tot[inv])).sum()
ds = data.Dataset(sys.argv[1]); B = 16 if ds.nv else 32; R = 6 if ds.nv else 3
cz = int(sys.argv[3]) if len(sys.argv) > 3 else 1
train, test = ds.split(8, int(sys.argv[2]))
acc = {}
def add(k, v): acc[k] = acc.get(k, 0) + v
nib = nsc = 0
for r in test:
    coded, anc, fl = ds.coded(ds.source(r), bool(cz))
    buf = np.frombuffer(coded, np.uint8)
    for m in range(3):
        so, rows, cols, wo = ds.geo[4*m:4*m+4]
        sc = buf[so:so+rows*cols].astype(np.int64)
        wb = buf[wo:wo+rows*cols*B//2]
        q = np.empty(wb.size*2, np.int64); q[0::2] = wb & 15; q[1::2] = wb >> 4
        q = q.reshape(-1, B); a = q & 7
        cls = np.clip(sc - anc[m], -R, R) + R
        nb = q.shape[0]
        pos = np.tile(np.arange(B), nb)
        x = q.ravel()
        # pm, mx running (8 = none)
        prev = np.concatenate([np.full((nb,1), 8), a[:, :-1]], 1)
        cm = np.maximum.accumulate(a, 1)
        mxs = np.concatenate([np.full((nb,1), 8), cm[:, :-1]], 1)
        M = a.max(1)
        c = np.repeat(cls, B)
        base = (c*9 + prev.ravel())*9 + mxs.ravel()
        key = lambda *parts: sum(p * w for p, w in zip(parts, np.cumprod([1] + [100]*(len(parts)-1))))
        add('W wnr3', H(base, x, 16) )
        add('W wnr3+pos', H(base*B + pos, x, 16))
        rem7 = (mxs.ravel() < 7) & (pos >= B - 4)
        add('W wnr3+lastpos4(mx<7)', H(base*5 + np.where(mxs.ravel()<7, np.maximum(pos-(B-5),0), 0), x, 16))
        # M first: cost H(M|cls) + nibbles given cls, M, prev, seen(M so far), remaining==1
        Mr = np.repeat(M, B)
        seen = (mxs.ravel() == Mr) & (mxs.ravel() != 8)
        last = pos == B - 1
        add('W M-first', H(cls, M, 8) + H(((c*8 + Mr)*9 + prev.ravel())*4 + seen*2 + last, x, 16))
        add('W M-first+pos', H(cls, M, 8) + H((((c*8 + Mr)*9 + prev.ravel())*2 + seen)*B + pos, x, 16))
        nib += x.size
        # scales
        col = np.arange(sc.size) % cols
        left = np.where(col > 0, np.concatenate([[0], sc[:-1]]) + 1, 0)
        up = np.where(np.arange(sc.size) >= cols, np.concatenate([np.zeros(cols, np.int64), sc[:-cols]]) + 1, 0)
        add('S order0', H(np.zeros_like(sc), sc, 256))
        add('S left', H(left, sc, 256))
        d = np.clip(up - left, -3, 3) + 3
        add('S left+dup7', H(left*8 + np.where(up > 0, d, 7), sc, 256))
        add('S left+up', H(left*257 + up, sc, 256))
        nsc += sc.size
raw = len(test) * ds.raw * 8
for k, v in acc.items():
    per = v / (nib if k[0] == 'W' else nsc)
    print(f'{ds.name} cz{cz} {k:24s} {per:.4f} bits/{"nib" if k[0]=="W" else "scale"}  {v/raw:.5f} of raw')
