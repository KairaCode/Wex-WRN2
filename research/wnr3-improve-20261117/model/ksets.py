"""K weight table sets (567 contexts each) trained by Lloyd iterations on the TRAIN weight streams (2048 symbols):
assign each stream to the set with the lowest ideal cost, re-estimate each set from its streams, quantise with
the WNR3 rule at P bits. Init: streams split into K quantiles of their bits/symbol under the single T80 model.
Writes ksets_K<k>_P<p>.npz (weight freqs [K,3,567,16]) and ksets.log."""
import sys, ctypes, numpy as np
from multiprocessing import Pool
from common import *
train, test = split()
WS = 17280

def lloyd_rec(args):
    tag, K, cw = args
    acc = np.zeros(K * 3 * NC * 16, np.uint64); asg = np.zeros(WS, np.uint8); b0 = np.zeros(WS)
    tot = LIB.m_lloyd(entry(tag), coded(tag), 2048, K, cw.ctypes.data, acc.ctypes.data, asg.ctypes.data, b0.ctypes.data)
    return tot, acc.astype(np.int64).reshape(K, 3, NC, 16), asg, b0

def costs(wf, P):
    return np.ascontiguousarray(P - np.log2(wf.astype(np.float64)))

if __name__ == "__main__":
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    wc, sc = sum_counts(train)
    w1 = q_rule(wc, P)
    pool = Pool(2)
    r = pool.map(lloyd_rec, [(t, 1, costs(w1[None], P)) for t in train])
    base = sum(x[0] for x in r); b0 = np.concatenate([x[3] for x in r])
    print("P=%d K=1 train bits/w %.6f" % (P, base / (80 * 35389440)), flush=True)
    for K in (2, 4, 8):
        # init by quantile of per-stream cost under the single model
        q = np.quantile(b0, np.linspace(0, 1, K + 1)[1:-1]); grp = np.searchsorted(q, b0)
        # counts per group: one pass with per-stream fixed assignment = run lloyd with K identical tables? no:
        # build init tables from a pass where set k = single model biased; simpler: K-means on cost with
        # tables from group counts, computed via lloyd on a one-hot cost table per stream is not possible in C,
        # so initialise each set as the single-model counts plus the counts of its quantile group (approx.).
        accs = np.zeros((K, 3, NC, 16), np.int64)
        # group counts: rerun with K=1 per record and split by stream group using per-record assignment trick
        off = 0
        for t, x in zip(train, r):
            pass
        # practical init: perturbed tables = single model sharpened/flattened in the direction of the group
        alphas = np.linspace(-0.6, 0.6, K)
        p1 = w1 / float(1 << P)
        prev = None
        tabs = []
        for a in alphas:
            ql = np.log(p1) * (1 + a); pq = np.exp(ql - ql.max(-1, keepdims=True)); pq /= pq.sum(-1, keepdims=True)
            tabs.append(q_rule(np.round(pq * 1e9).astype(np.int64), P))
        wf = np.stack(tabs)
        for it in range(12):
            rr = pool.map(lloyd_rec, [(t, K, costs(wf, P)) for t in train])
            tot = sum(x[0] for x in rr); acc = sum(x[1] for x in rr)
            use = np.bincount(np.concatenate([x[2] for x in rr]), minlength=K)
            print("P=%d K=%d it %d train bits/w %.6f (vs K=1 %+.4f%% of raw, before selector)  use %s" % (
                P, K, it, tot / (80 * 35389440), (tot - base) / 8 / RAW / 80 * 100, use.tolist()), flush=True)
            wf = q_rule(acc, P)
            if prev is not None and abs(prev - tot) < 1e-6 * tot: break
            prev = tot
        np.savez(W + "/ksets_K%d_P%d.npz" % (K, P), wf=wf)
