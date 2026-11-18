"""Can a STATIC-table decoder use the w1<->w3 sibling relation? Train 14-bit tables (wex_nr2_quantize rule) on
the 80 TRAIN records, score the 41 TEST records (held-out), for the w3 nibbles (and w1 with w3's scale only).

Variants (contexts per matrix; sibd = clamp(s_w1 - s_w3, -2, 2) on the coded scales of the same block,
sa = magnitude of w1[i,j] (coded), mx/pm as WNR3):
  V0  base567 (today)                         567
  V1  base567 x sa                            4,536
  V2  base567 x sibd x sa                     22,680
  V3  srel x mx x sibd x sa                   2,520
  V4  srel x sibd x sa                        280
  V5  base567 x sibd (scales only)            2,835
  Vxx_L8 = same tables trained separately for layers 0-7 and 8-39 (2 table sets, chosen by layer)
  python3 static_sib.py JOBS -> out/static_sib.json"""
import sys, json, os
import numpy as np
from multiprocessing import Pool
from load import load, MAN, H as HERE, test_set, train_set

VARS = {"V0": 567, "V1": 567 * 8, "V2": 567 * 40, "V3": 7 * 9 * 40, "V4": 280, "V5": 567 * 5}

def ctxs(s, q, anc, ss, sq):
    R, C = q.shape; nb = C // 32
    a = (q & 7).reshape(R, nb, 32).astype(np.int64)
    srel = (np.clip(s.astype(np.int64) - anc, -3, 3) + 3)[:, :, None]
    pm = np.empty_like(a); pm[:, :, 0] = 8; pm[:, :, 1:] = a[:, :, :-1]
    mx = np.empty_like(a); mx[:, :, 0] = 8; mx[:, :, 1:] = np.maximum.accumulate(a, axis=2)[:, :, :-1]
    base = (srel * 9 + pm) * 9 + mx
    sibd = (np.clip(ss.astype(np.int64) - s.astype(np.int64), -2, 2) + 2)[:, :, None]
    out = {"V0": base, "V5": base * 5 + sibd}
    if sq is not None:
        sa = (sq & 7).reshape(R, nb, 32).astype(np.int64)
        out.update(V1=base * 8 + sa, V2=(base * 5 + sibd) * 8 + sa, V3=((srel * 9 + mx) * 5 + sibd) * 8 + sa,
                   V4=(srel * 5 + sibd) * 8 + sa)
    return {k: v.ravel() for k, v in out.items()}

def counts(k):
    d, anc, _ = load(*k)
    res = {}
    for m, sib in ((2, 0), (0, 2)):
        s, q = d[m]; ss, sq = d[sib]
        x = q.ravel().astype(np.int64)
        cs = ctxs(s, q, anc[m], ss, sq if m == 2 else None)
        for v, c in cs.items():
            res[(m, v)] = np.bincount(c * 16 + x, minlength=VARS[v] * 16).reshape(VARS[v], 16)
    return k, res

def quantize(c):
    """wex_nr2_quantize, vectorised over rows. c: (K,16) int64 -> freq (K,16) summing to 16384."""
    total = c.sum(1, keepdims=True); spare = 16384 - 16
    scaled = np.where(total > 0, c * spare, spare); div = np.where(total > 0, total, 16)
    f = 1 + scaled // div; rem = scaled % div
    extra = 16384 - f.sum(1)
    order = np.lexsort((np.broadcast_to(np.arange(16), c.shape), -rem), axis=1)  # largest remainder, first index
    rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(16)[None, :].repeat(c.shape[0], 0), axis=1)
    return f + (rank < extra[:, None])

if __name__ == "__main__":
    with Pool(int(sys.argv[1])) as p:
        C = dict(p.map(counts, sorted(MAN), chunksize=1))
    tr, te = train_set(), test_set()
    out = {"per_record": [], "summary": {}}
    for m in (2, 0):
        for v in VARS:
            if (m, v) not in C[tr[0]]: continue
            tot = sum(C[k][(m, v)] for k in tr)
            lo = sum(C[k][(m, v)] for k in tr if k[0] < 8); hi = tot - lo
            q = {"g": quantize(tot), "lo": quantize(lo), "hi": quantize(hi)}
            os.makedirs(os.path.join(HERE, "out", "tables"), exist_ok=True)
            for t in q:  # CDF tables, K x 17 u16 LE, for rt.c
                cdf = np.concatenate([np.zeros((q[t].shape[0], 1), np.int64), np.cumsum(q[t], 1)], 1).astype("<u2")
                cdf.tofile(os.path.join(HERE, "out", "tables", f"m{m}_{v}_{t}.u16"))
            for k in te:
                c = C[k][(m, v)]; n = c.sum()
                bits = {t: float(-(c * np.log2(q[t] / 16384.0)).sum()) for t in q}
                out["per_record"].append(dict(layer=k[0], expert=k[1], m=m, var=v, n=int(n), bits_global=bits["g"],
                                              bits_split8=bits["lo"] if k[0] < 8 else bits["hi"]))
    RAW = 18800640
    pr = out["per_record"]
    for m in (2, 0):
        b0 = {(r["layer"], r["expert"]): r for r in pr if r["m"] == m and r["var"] == "V0"}
        for v in VARS:
            rows = [r for r in pr if r["m"] == m and r["var"] == v]
            if not rows: continue
            for t in ("bits_global", "bits_split8"):
                d = np.array([(r[t] - b0[(r["layer"], r["expert"])]["bits_global"]) / 8 / RAW * 100 for r in rows])
                early = np.array([dd for dd, r in zip(d, rows) if r["layer"] < 8])
                out["summary"][f"m{m} {v} {t}"] = dict(bpw=float(np.mean([r[t] / r["n"] for r in rows])),
                    delta_pct_raw_mean=float(d.mean()), min=float(d.min()), max=float(d.max()),
                    delta_pct_raw_layers0_7=float(early.mean()), delta_pct_raw_layers8_39=float(d[[r["layer"] >= 8 for r in rows]].mean()))
                print(f"m{m} {v:3s} {t:12s}", {kk: round(vv, 5) for kk, vv in out["summary"][f"m{m} {v} {t}"].items()})
    json.dump(out, open(os.path.join(HERE, "out", "static_sib.json"), "w"), indent=0)
