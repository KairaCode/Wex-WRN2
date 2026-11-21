"""Selector-free table sets: the set of a weight stream is a function of its (already decoded) scales, so no
selector bits are stored. Feature per stream = sum over its 64 blocks of srel (or count of blocks with
srel != 3); K classes by TRAIN quantiles; one 567-context table set per class, WNR3 rule at 14 bits.
Ideal code lengths (bound) on TEST vs the single T80 model. Writes classsets.json."""
import ctypes, json, numpy as np
from common import *
train, test = split()
LIB.m_count_by_class.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
def feats(tag):
    c = coded(tag); e = struct.unpack("<QIII6H6H3I3II", entry(tag)); out = {}
    srels = []
    for m in range(3):
        n = e[4 + 2 * m] * e[5 + 2 * m]; s = np.frombuffer(c, np.uint8, n, e[16 + m]).astype(np.int64)
        a = BY[tag]["anchors"][m]; sr = np.clip(s - a, -3, 3) + 3
        srels.append(sr.reshape(-1, 64))   # 64 blocks per 2048-symbol stream
    sr = np.concatenate(srels)
    return {"sum": sr.sum(1), "off": (sr != 3).sum(1), "hi": (sr >= 4).sum(1), "spread": sr.max(1) - sr.min(1)}
F = {t: feats(t) for t in TAGS}
res = {}
wc1, _ = sum_counts(train); f1 = q_rule(wc1, 14)
base = np.array([ideal_bits(counts(t)[0], f1, 14) for t in test])
import sys
KINDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sum", "off", "hi", "spread"]
KS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2, 4, 8]
OUT = sys.argv[3] if len(sys.argv) > 3 else "classsets.json"
for kind in KINDS:
    allf = np.concatenate([F[t][kind] for t in train])
    for K in KS:
        q = np.unique(np.quantile(allf, np.linspace(0, 1, K + 1)[1:-1]))
        KK = len(q) + 1
        def cnt(t):
            cls = np.ascontiguousarray(np.searchsorted(q, F[t][kind], side="right").astype(np.uint8))
            acc = np.zeros(KK * 3 * NC * 16, np.uint64)
            assert LIB.m_count_by_class(entry(t), coded(t), 2048, KK, cls.ctypes.data, acc.ctypes.data) == 0
            return acc.astype(np.int64).reshape(KK, 3, NC, 16)
        tr = sum(cnt(t) for t in train); fq = q_rule(tr, 14)
        bits = np.array([ideal_bits(cnt(t), fq, 14) for t in test])
        d = (bits - base) / 8; no = np.array([t != "39_120" for t in test])
        res["%s_K%d" % (kind, K)] = dict(classes=KK, thresholds=q.tolist(), delta_B=d.tolist())
        print("%-7s K=%d (classes %d): delta vs single T80 model %+.0f B/rec (%+.4f%% of raw), no outlier %+.0f B" % (
            kind, K, KK, d.mean(), d.mean() / RAW * 100, d[no].mean()), flush=True)
json.dump(res, open(W + "/" + OUT, "w"))
