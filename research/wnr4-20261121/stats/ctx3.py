# Held-out cross-entropy (14-bit quantized tables, train records -> test records) of candidate contexts.
#   python -I stats/ctx3.py DATASET NTRAIN NTEST CZ
import sys, os, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data
ds = data.Dataset(sys.argv[1]); B = 16 if ds.nv else 32; R = 6 if ds.nv else 3; NC = 2*R+1
ntr, nte, cz = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
NL = max(ds.owner(r) >> 16 for r in ds.experts) + 1
train, test = ds.split(ntr, nte)
V = {  # name: (nkeys, nsym)
 'W wnr3': (NC*81, 16), 'W wnr3+pos': (NC*81*B, 16), 'W wnr3+pos8': (NC*81*8, 16),
 'W wnr3+col16': (NC*81*16, 16), 'W cls+pos': (NC*B, 16), 'W wnr3 R+4': ((NC+8)*81, 16), 'W wnr3 R+8': ((NC+16)*81, 16), 'W wnr3/layer': (NC*81*NL, 16), 'W wnr3+last4': (NC*81*5, 16), 'W wnr3+first2last2': (NC*81*5, 16), 'W wnr3+last2(mx)': (NC*81*3, 16), 'W wnr3/lgroup6': (NC*81*8, 16), 'S left/lgroup6': (257*8, 256),
 'S order0': (1, 256), 'S up': (257, 256), 'S up+dul7': (257*8, 256), 'S up+dur7': (257*8, 256), 'S up+dul+dur': (257*64, 256), 'S up+dul5+dur5': (257*36, 256), 'S delta|dul,dur,up>>1': (129*64, 256), 'S delta|dul,dur,up>>2': (65*64, 256), 'S delta|dul,dur,up>>3': (33*64, 256), 'S delta|dul,dur,up>>4': (17*64, 256), 'S uprel+dul+dur': (50*64, 256), 'S uprel+dul5+dur5': (50*36, 256), 'S left+dup+dul': (257*64, 256), 'S left': (257, 256), 'S left+up': (257*257, 256), 'S left+dup7': (257*8, 256)}
cnt = {s: {k: np.zeros((3, n, m), np.int64) for k, (n, m) in V.items()} for s in (0, 1)}
def keys(r):
    coded, anc, fl = ds.coded(ds.source(r), bool(cz))
    buf = np.frombuffer(coded, np.uint8); out = []
    L = ds.owner(r) >> 16; LG = L * 8 // NL
    for m in range(3):
        so, rows, cols, wo = ds.geo[4*m:4*m+4]
        sc = buf[so:so+rows*cols].astype(np.int64)
        wb = buf[wo:wo+rows*cols*B//2]
        q = np.empty(wb.size*2, np.int64); q[0::2] = wb & 15; q[1::2] = wb >> 4
        q = q.reshape(-1, B); a = q & 7; nb = q.shape[0]
        prev = np.concatenate([np.full((nb,1), 8), a[:, :-1]], 1).ravel()
        cm = np.maximum.accumulate(a, 1)
        mxs = np.concatenate([np.full((nb,1), 8), cm[:, :-1]], 1).ravel()
        c = np.repeat(np.clip(sc - anc[m], -R, R) + R, B)
        pos = np.tile(np.arange(B), nb)
        colw = (np.arange(nb*B) % (cols*B)) % 16   # input column mod 16
        base = (c*9 + prev)*9 + mxs
        x = q.ravel()
        col = np.arange(sc.size) % cols
        left = np.where(col > 0, np.concatenate([[0], sc[:-1]]) + 1, 0)
        up = np.where(np.arange(sc.size) >= cols, np.concatenate([np.zeros(cols, np.int64), sc[:-cols]]) + 1, 0)
        d = np.where(up > 0, np.clip(up - left, -3, 3) + 3, 7)
        ul = np.where((up > 0) & (col > 0), np.concatenate([[0], up[:-1]]), 0)
        ur = np.where((up > 0) & (col < cols - 1), np.concatenate([up[1:], [0]]), 0)
        dul = np.where(ul > 0, np.clip(ul - up, -3, 3) + 3, 7)
        dur = np.where(ur > 0, np.clip(ur - up, -3, 3) + 3, 7)
        dul5 = np.where(ul > 0, np.clip(ul - up, -2, 2) + 2, 5)
        dur5 = np.where(ur > 0, np.clip(ur - up, -2, 2) + 2, 5)
        urel = np.where(up > 0, np.clip(up - 1 - anc[m], -24, 24) + 25, 0)
        dlu = np.where((left > 0) & (ul > 0), np.clip(ul - left, -3, 3) + 3, 7)
        out.append({'W wnr3': (base, x), 'W wnr3+pos': (base*B + pos, x), 'W wnr3+pos8': (base*8 + pos % 8, x),
                    'W wnr3+col16': (base*16 + colw, x), 'W cls+pos': (c*B + pos, x),
                    'W wnr3 R+4': ((np.repeat(np.clip(sc - anc[m], -R-4, R+4) + R+4, B)*9 + prev)*9 + mxs, x),
                    'W wnr3 R+8': ((np.repeat(np.clip(sc - anc[m], -R-8, R+8) + R+8, B)*9 + prev)*9 + mxs, x), 'W wnr3/layer': (L*NC*81 + base, x),
                    'W wnr3+last4': (base*5 + np.maximum(pos - (B - 5), 0), x),
                    'W wnr3+first2last2': (base*5 + np.where(pos < 2, pos, np.where(pos >= B - 2, pos - (B - 2) + 2, 4)), x),
                    'W wnr3+last2(mx)': (base*3 + np.maximum(pos - (B - 3), 0), x), 'W wnr3/lgroup6': (LG*NC*81 + base, x),
                    'S left/lgroup6': (LG*257 + left, sc),
                    'S order0': (np.zeros_like(sc), sc), 'S up': (up, sc), 'S up+dul7': (up*8 + dul, sc), 'S up+dur7': (up*8 + dur, sc),
                    'S up+dul+dur': (up*64 + dul*8 + dur, sc), 'S up+dul5+dur5': (up*36 + dul5*6 + dur5, sc),
                    'S delta|dul,dur,up>>1': (np.where(up > 0, 1 + ((up - 1) >> 1), 0)*64 + dul*8 + dur, (sc - np.where(up > 0, up - 1, 0)) & 255),
                    'S delta|dul,dur,up>>2': (np.where(up > 0, 1 + ((up - 1) >> 2), 0)*64 + dul*8 + dur, (sc - np.where(up > 0, up - 1, 0)) & 255),
                    'S delta|dul,dur,up>>3': (np.where(up > 0, 1 + ((up - 1) >> 3), 0)*64 + dul*8 + dur, (sc - np.where(up > 0, up - 1, 0)) & 255),
                    'S delta|dul,dur,up>>4': (np.where(up > 0, 1 + ((up - 1) >> 4), 0)*64 + dul*8 + dur, (sc - np.where(up > 0, up - 1, 0)) & 255),
                    'S uprel+dul+dur': (urel*64 + dul*8 + dur, sc), 'S uprel+dul5+dur5': (urel*36 + dul5*6 + dur5, sc), 'S left+dup+dul': (left*64 + d*8 + dlu, sc), 'S left': (left, sc), 'S left+up': (left*257 + up, sc),
                    'S left+dup7': (left*8 + d, sc)})
    return out
def add(s, r):
    ks = keys(r)
    for m, kv in enumerate(ks):
        for k, (key, sym) in kv.items():
            n, ns = V[k]
            cnt[s][k][m] += np.bincount(key*ns + sym, minlength=n*ns).reshape(n, ns)
with ThreadPoolExecutor(6) as p:
    list(p.map(lambda r: add(0, r), train)); list(p.map(lambda r: add(1, r), test))
T = 16384
raw = len(test) * ds.raw * 8
for k, (n, ns) in V.items():
    tr, te = cnt[0][k], cnt[1][k]
    tot = tr.sum(2, keepdims=True)
    f = np.where(tot > 0, 1 + np.floor(tr * (T - ns) / np.maximum(tot, 1)), T / ns)
    f = f * (T / f.sum(2, keepdims=True))   # renormalised (approximate rANS table)
    bits = (te * (np.log2(T) - np.log2(f))).sum()
    print(f'{ds.name} cz{cz} {k:14s} {bits/raw:.5f} of raw', flush=True)
