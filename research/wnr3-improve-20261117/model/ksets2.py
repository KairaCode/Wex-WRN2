"""Table sets = ncls selector-free classes (class = TRAIN quantile bin of f = sum of srel over the stream's 64
blocks, computed by the decoder from the decoded scales) x K encoder-selected sets per class (Lloyd on the
TRAIN weight streams, ideal costs, WNR3 rule or rule-without-symbol-8 at P bits).
Usage: python3 ksets2.py P NCLS K [z8]   -> sets_P<P>_C<ncls>_K<K>[_z8].npz (wf [ncls*K,3,567,16], thr)."""
import sys, ctypes, numpy as np
from multiprocessing import Pool
from common import *
train, test = split()
P, NCLS, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); Z8 = len(sys.argv) > 4 and sys.argv[4] == "z8"

def feat(tag):
    c = coded(tag); e = struct.unpack("<QIII6H6H3I3II", entry(tag)); out = []
    for m in range(3):
        n = e[4 + 2 * m] * e[5 + 2 * m]; s = np.frombuffer(c, np.uint8, n, e[16 + m]).astype(np.int64)
        out.append((np.clip(s - BY[tag]["anchors"][m], -3, 3) + 3).reshape(-1, 64).sum(1))
    return np.concatenate(out)

def quant(acc):
    if not Z8: return q_rule(acc, P)
    keep = [s for s in range(16) if s != 8]; f = np.zeros(acc.shape, np.int64); f[..., keep] = q_rule(acc[..., keep], P); return f

def costs(wf):
    with np.errstate(divide="ignore"):
        return np.ascontiguousarray(np.where(wf > 0, P - np.log2(np.maximum(wf, 1).astype(np.float64)), 99.0))

def step(args):
    tag, cw, thr, K = args
    acc = np.zeros(NCLS * K * 3 * NC * 16, np.uint64); asg = np.zeros(17280, np.uint8)
    tot = LIB.m_lloyd2(entry(tag), coded(tag), 2048, K, NCLS, thr.ctypes.data, cw.ctypes.data, acc.ctypes.data, asg.ctypes.data)
    return tot, acc.astype(np.int64).reshape(NCLS * K, 3, NC, 16), asg

if __name__ == "__main__":
    allf = np.concatenate([feat(t) for t in train])
    thr = np.unique(np.quantile(allf, np.linspace(0, 1, NCLS + 1)[1:-1]).astype(np.int32)) if NCLS > 1 else np.zeros(0, np.int32)
    # quantile thresholds as "f >= thr" boundaries; ceil so that the bins match searchsorted(side=right)
    thr = np.ascontiguousarray(np.unique(np.ceil(np.quantile(allf, np.linspace(0, 1, NCLS + 1)[1:-1]) + 1e-9)).astype(np.int32)) if NCLS > 1 else np.zeros(1, np.int32)
    NCLS = len(thr) + 1 if NCLS > 1 else 1
    pool = Pool(2)
    # class tables (K = 1)
    one = np.zeros((NCLS, 3, NC, 16), np.int64); Kreal = K; K = 1
    flat = np.ones((NCLS, 3, NC, 16), np.int64) * ((1 << P) // 16)
    r = pool.map(step, [(t, costs(flat), thr, 1) for t in train])
    acc = sum(x[1] for x in r); wf = quant(acc)
    r = pool.map(step, [(t, costs(wf), thr, 1) for t in train]); base = sum(x[0] for x in r)
    print("P=%d classes=%d K=1 train bits/w %.6f thr %s" % (P, NCLS, base / (80 * 35389440), thr.tolist()), flush=True)
    K = Kreal
    if K > 1:
        tabs = []
        p1 = wf / float(1 << P)
        for c in range(NCLS):
            for a in np.linspace(-0.6, 0.6, K):
                ql = np.log(np.maximum(p1[c], 1e-9)) * (1 + a); pq = np.exp(ql - ql.max(-1, keepdims=True)); pq /= pq.sum(-1, keepdims=True)
                tabs.append(quant(np.round(pq * 1e9).astype(np.int64)))
        wf = np.stack(tabs); prev = None
        for it in range(14):
            r = pool.map(step, [(t, costs(wf), thr, K) for t in train])
            tot = sum(x[0] for x in r); acc = sum(x[1] for x in r)
            print("P=%d classes=%d K=%d it %d train bits/w %.6f (%+.4f%% of raw vs K=1, before selector)" % (
                P, NCLS, K, it, tot / (80 * 35389440), (tot - base) / 8 / RAW / 80 * 100), flush=True)
            wf = quant(acc)
            if prev is not None and abs(prev - tot) < 1e-6 * tot: break
            prev = tot
    np.savez(W + "/sets_P%d_C%d_K%d%s%s.npz" % (P, int(sys.argv[2]), K, "_z8" if Z8 else "", SFX), wf=wf, thr=thr if NCLS > 1 else np.zeros(0, np.int32))
