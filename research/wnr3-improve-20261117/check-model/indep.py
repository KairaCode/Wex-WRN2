#!/usr/bin/env python3
"""Checker's independent re-derivation (numpy only, no mlib, no lane code) of the C16 + z8 + sesc gain as an
ideal code-length bound (-log2 p under the quantised 14-bit tables), held out.
- contexts: WNR3 (srel*9 + prev_mag)*9 + running_max, reset per 32-block; anchors from the record header
- class of a 2048-weight stream: number of thresholds <= sum of srel over its 64 blocks; thresholds = ceil of
  TRAIN quantiles (recomputed here)
- quantiser: own implementation of the WNR3 rule (1 + floor(c*spare/tot), leftovers by largest remainder, lower
  symbol first); z8: rule over the 15 symbols != 8, f8 = 0
- scales: production absolute tables vs anchor-relative 10-value alphabet (+escape) seen in TRAIN
Usage: SPLIT=A|B python3 indep.py  -> indep_<SPLIT>.json"""
import json, os, struct, numpy as np
from multiprocessing import Pool
D = "/tmp/wnr3-data"; W = "/tmp/wnr3-research/check-model"; WI = "/tmp/wnr3-research/model"
RAW = 18800640; NC = 567; P = 14; M = 1 << P
SPLIT = os.environ.get("SPLIT", "A")
PREP = json.load(open(W + "/prep.json")); EXTRA = json.load(open(W + "/prep_extra.json"))
BY = {r["tag"]: r for r in PREP + EXTRA}
TABLE = open(D + "/recs/entries.bin", "rb").read(); BLOB = open(D + "/recs/model.bin", "rb").read()

def rule(c):
    c = np.asarray(c, np.int64); n = c.shape[-1]; c2 = c.reshape(-1, n); out = np.empty_like(c2)
    for t in range(len(c2)):
        tot = c2[t].sum(); spare = M - n
        if tot: f = 1 + c2[t] * spare // tot; rem = c2[t] * spare % tot
        else: f = np.full(n, 1 + spare // n); rem = np.full(n, spare % n)
        k = M - f.sum(); order = np.lexsort((np.arange(n), -rem)); f[order[:k]] += 1; out[t] = f
    return out.reshape(c.shape)

def ent(tag):
    r = BY[tag]; i = r["layer"] * 192 + r["expert"]; return struct.unpack("<QIII6H6H3I3II", TABLE[i * 72:(i + 1) * 72])

def feats(tag):
    """per matrix: (ctx [nstreams, 2048], sym [nstreams, 2048], f [nstreams]) and scales"""
    E = ent(tag); c = open(WI + "/coded/%s.bin" % tag, "rb").read(); out = []
    for m in range(3):
        a = BY[tag]["anchors"][m]; n = E[4 + 2 * m] * E[5 + 2 * m]
        s = np.frombuffer(c, np.uint8, n, E[16 + m]).astype(np.int64)
        b = np.frombuffer(c, np.uint8, n * 16, E[19 + m])
        x = np.empty(n * 32, np.int64); x[0::2] = b & 15; x[1::2] = b >> 4
        x = x.reshape(n, 32); mag = x & 7
        pm = np.full((n, 32), 8); pm[:, 1:] = mag[:, :-1]
        mx = np.full((n, 32), 8); mx[:, 1:] = np.maximum.accumulate(mag, 1)[:, :-1]
        srel = np.clip(s - a, -3, 3) + 3
        ctx = (srel[:, None] * 9 + pm) * 9 + mx
        f = srel.reshape(-1, 64).sum(1)
        out.append((ctx.reshape(-1, 2048), x.reshape(-1, 2048), f, s, a))
    return out

def counts(args):
    tag, thr = args; F = feats(tag)
    wc = np.zeros((16, 3, NC, 16), np.int64); sabs = np.zeros((3, 256), np.int64); srel = np.zeros((3, 256), np.int64)
    for m, (ctx, x, f, s, a) in enumerate(F):
        cls = np.searchsorted(thr, f, side="right")
        idx = (cls[:, None] * 3 + m) * NC * 16 + ctx * 16 + x
        wc += np.bincount(idx.ravel(), minlength=16 * 3 * NC * 16).reshape(16, 3, NC, 16)
        sabs[m] = np.bincount(s, minlength=256); srel[m] = np.bincount((s - a) & 255, minlength=256)
    return tag, wc, sabs, srel, np.concatenate([f for (_, _, f, _, _) in F])

def bits(c, f):
    c = np.asarray(c, np.float64); f = np.asarray(f, np.float64)
    assert not np.any((c > 0) & (f == 0))
    return float(np.sum(np.where(c > 0, c * (P - np.log2(np.maximum(f, 1))), 0)))

if __name__ == "__main__":
    tags = [r["tag"] for r in PREP]
    def is_train(t):
        l, e = BY[t]["layer"], BY[t]["expert"]
        tr = ((37 * l + 11) % 192, (37 * l + 101) % 192) if SPLIT == "A" else ((37 * l + 11) % 192, (53 * l + 7) % 192)
        return e in tr and t != "39_120"
    train = [t for t in tags if is_train(t)]; test = [t for t in tags if not is_train(t)] + [r["tag"] for r in EXTRA]
    assert len(train) == 80
    with Pool(2) as pool:
        allf = np.concatenate([x[4] for x in pool.map(counts, [(t, np.zeros(0)) for t in train])])
        thr = np.unique(np.ceil(np.quantile(allf, np.linspace(0, 1, 17)[1:-1]) + 1e-9)).astype(np.int64)
        print("split", SPLIT, "thresholds", thr.tolist(), flush=True)
        res = {r[0]: r for r in pool.map(counts, [(t, thr) for t in train + test])}
    wtr = sum(res[t][1] for t in train); s1tr = sum(res[t][3] for t in train)
    keep = [s for s in range(16) if s != 8]
    wf = np.zeros_like(wtr); wf[..., keep] = rule(wtr[..., keep])          # C16 + z8
    alpha = np.flatnonzero(s1tr.sum(0) > 0); red = np.zeros((3, len(alpha) + 1), np.int64); red[:, :-1] = s1tr[:, alpha]
    sf = rule(red)
    print("sesc alphabet (relative values mod 256):", alpha.tolist(), flush=True)
    pa = np.frombuffer(BLOB, np.uint16, offset=32, count=3 * NC * 17).reshape(3, NC, 17).astype(np.int64)
    pb = np.frombuffer(BLOB, np.uint16, offset=32 + 3 * NC * 34, count=3 * 257).reshape(3, 257).astype(np.int64)
    pwf, psf = np.diff(pa, axis=-1), np.diff(pb, axis=-1)
    rows = []
    for t in test:
        _, wc, sabs, srel, _ = res[t]
        bp = bits(wc.sum(0), pwf) + bits(sabs, psf)
        rs = srel[:, alpha]; esc = srel.sum() - rs.sum()
        bn = bits(wc, wf) + bits(rs, sf[:, :-1]) + sum(esc * 0 for _ in [0])
        if esc: bn += bits(np.c_[np.zeros((3, len(alpha)), np.int64), srel.sum(1) - rs.sum(1)], sf) + 8 * esc
        rows.append(dict(tag=t, prod_bits=bp, new_bits=bn, delta_B=(bn - bp) / 8, escapes=int(esc)))
    json.dump(dict(thr=thr.tolist(), alpha=alpha.tolist(), rows=rows), open(W + "/indep_%s.json" % SPLIT, "w"), indent=1)
    norm = [r["delta_B"] for r in rows if r["tag"] in BY and r["tag"] not in [x["tag"] for x in EXTRA] and r["tag"] != "39_120"]
    shf = [r["delta_B"] for r in rows if r["tag"] in [x["tag"] for x in EXTRA] or r["tag"] == "39_120"]
    print("normal TEST n=%d ideal delta mean %.0f B (%.4f%% raw) median %.0f; shifted n=%d mean %.0f; L39E120 %.0f; escapes %d" % (
        len(norm), np.mean(norm), np.mean(norm) / RAW * 100, np.median(norm), len(shf), np.mean(shf),
        [r["delta_B"] for r in rows if r["tag"] == "39_120"][0], sum(r["escapes"] for r in rows)))
