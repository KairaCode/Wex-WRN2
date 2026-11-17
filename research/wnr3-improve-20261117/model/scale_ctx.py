"""Ideal code lengths (bounds) of the scale streams under scale models: absolute (WNR3), anchor-relative,
and anchor-relative with a context from the previous scale of the same stream (free to the decoder).
Train T80, score TEST; also per record. Writes scale_ctx.json."""
import json, numpy as np
from common import *
train, test = split()
def scales(tag):
    c = coded(tag); e = struct.unpack("<QIII6H6H3I3II", entry(tag))
    out = []
    for m in range(3):
        n = e[4 + 2 * m] * e[5 + 2 * m]; off = e[16 + m]
        s = np.frombuffer(c, np.uint8, n, off).astype(np.int64)
        out.append((s, int(BY[tag]["anchors"][m])))
    return out
def ctx_of(s, a, kind, chunk=512):
    n = len(s); first = (np.arange(n) % chunk) == 0
    prev = np.roll(s, 1)
    if kind == "none": return np.zeros(n, np.int64)
    if kind.startswith("prev"):
        R = int(kind[4:]); c = np.clip(prev - a, -R, R) + R + 1; c[first] = 0; return c
    if kind.startswith("dprev"):   # previous relative to its own previous? (two back)
        return None
KINDS = [("abs", "none"), ("rel", "none"), ("rel", "prev1"), ("rel", "prev3"), ("rel", "prev7"), ("abs", "prev3")]
def nctx(kind): return 1 if kind == "none" else 2 * int(kind[4:]) + 2
cache = {t: scales(t) for t in TAGS}
res = {}
for mode, kind in KINDS:
    C = nctx(kind)
    def cnt(t):
        z = np.zeros((3, C, 256), np.int64)
        for m, (s, a) in enumerate(cache[t]):
            sym = s if mode == "abs" else (s - a) & 255
            np.add.at(z[m], (ctx_of(s, a, kind), sym), 1)
        return z
    CT = {t: cnt(t) for t in TAGS}
    tr = sum(CT[t] for t in train)
    for P in (14, 16):
        f = q_rule(tr, P)
        bits = [ideal_bits(CT[t], f, P) for t in test]
        nm = "%s_%s_rule%d" % (mode, kind, P)
        res[nm] = dict(bits=bits, contexts=C, table_bytes=3 * C * 255 * 2)
        b = np.array(bits); no = np.array([t != "39_120" for t in test])
        print("%-22s ctx/matrix %3d  mean %.5f bit/scale (B/rec %.0f)  without L39E120: %.5f bit/scale (B/rec %.0f)" % (
            nm, C, b.mean() / 1105920, b.mean() / 8, b[no].mean() / 1105920, b[no].mean() / 8), flush=True)
json.dump(dict(test=test, res=res), open(W + "/scale_ctx.json", "w"))
