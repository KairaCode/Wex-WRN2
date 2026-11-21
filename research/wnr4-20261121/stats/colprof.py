# Layer-shared column profile of the scales as an extra context (NV/MX): H(s | up-ctx) vs H(s | up-ctx, prof class).
import sys, os, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data
def H(keys, sym):
    k = keys.astype(np.int64) * 256 + sym
    u, c = np.unique(k, return_counts=True)
    cu, inv = np.unique(u // 256, return_inverse=True)
    tot = np.bincount(inv, weights=c)
    return -(c * np.log2(c / tot[inv])).sum()
ds = data.Dataset(sys.argv[1]); K = int(sys.argv[2])
by = {}
for r in ds.experts: by.setdefault(ds.owner(r) >> 16, []).append(r)
rng = random.Random(5); tot = {}
def scales(r):
    coded, anc, fl = ds.coded(ds.source(r), False)
    buf = np.frombuffer(coded, np.uint8)
    return [buf[ds.geo[4*m]:ds.geo[4*m] + ds.geo[4*m+1]*ds.geo[4*m+2]].astype(np.int64).reshape(ds.geo[4*m+1], ds.geo[4*m+2]) for m in range(3)]
layers = sorted(by)
for L in layers[::max(1, len(layers)//6)]:
    pool = rng.sample(by[L], 2*K)
    prof = [np.mean([scales(r)[m] for r in pool[:K]], axis=0) for m in range(3)]   # [rows, cols] mean over experts
    keyA, keyB, keyC, syms = [], [], [], []
    for r in pool[K:]:
        for m, sc in enumerate(scales(r)):
            rows, cols = sc.shape
            up = np.vstack([np.zeros((1, cols), np.int64) - 1, sc[:-1]])
            ul = np.hstack([np.zeros((rows, 1), np.int64) - 1, up[:, :-1]]); ur = np.hstack([up[:, 1:], np.zeros((rows, 1), np.int64) - 1])
            dl = np.where((ul >= 0) & (up >= 0), np.clip(ul - up, -3, 3) + 3, 7); dr = np.where((ur >= 0) & (up >= 0), np.clip(ur - up, -3, 3) + 3, 7)
            base = np.where(up >= 0, 1 + (up >> 2), 0) * 64 + dl * 8 + dr
            pc = prof[m].mean(0)[None, :].repeat(rows, 0)      # column profile (mean over rows and experts)
            pe = prof[m]                                         # element profile (mean over experts, same [row, col])
            dc = np.where(up >= 0, np.clip(np.rint(pc).astype(np.int64) - up, -3, 3) + 3, 7)
            de = np.where(up >= 0, np.clip(np.rint(pe).astype(np.int64) - up, -3, 3) + 3, 7)
            d = (sc - np.where(up >= 0, up, 0)) & 255
            keyA.append((m * 4160 + base).ravel()); keyB.append(((m * 4160 + base) * 8 + dc).ravel()); keyC.append(((m * 4160 + base) * 8 + de).ravel()); syms.append(d.ravel())
    keyA, keyB, keyC, syms = map(np.concatenate, (keyA, keyB, keyC, syms))
    n = syms.size
    a, b, c = H(keyA, syms) / n, H(keyB, syms) / n, H(keyC, syms) / n
    print(f'{ds.name} L{L}: up-ctx {a:.4f} b/scale, +col profile {b - a:+.4f}, +element profile {c - a:+.4f} (in-sample bits; profile from {K} other experts)', flush=True)
