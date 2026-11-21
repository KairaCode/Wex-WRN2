# Bounds for per-record scale tables (scales relative to the rule-g anchor), on the TEST records:
#  - per-record per-matrix order-0 entropy (empirical, in-sample) + a table cost estimate
#  - "K global table sets, one chosen per matrix per record" (held-out: sets trained on TRAIN by Lloyd)
#  - real quantised 14-bit CDF built from the record's own counts (what an encoder would transmit)
# All are ideal code lengths (bounds), no stream overhead. usage: python3 scale_adaptive.py
import numpy as np, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
from scale_bounds_lib import quantize, load, meta, is_test

def H(c):
    c = c[c > 0].astype(float); return -(c * np.log2(c / c.sum())).sum()

rows = []
hist = {}
for r in meta:
    mats = load(r)
    hist[(r["layer"], r["expert"])] = [np.bincount((mats[m].ravel() - r["anchors"][m] + 128) & 255, minlength=256) for m in range(3)]

def cost(h, freq): return -(h * np.log2(freq / 16384.0)).sum() / 8

# global rel table
glob = [sum(hist[(r["layer"], r["expert"])][m] for r in meta if not is_test(r)) for m in range(3)]
gq = [quantize(g) for g in glob]
# K sets via Lloyd on TRAIN matrices (per matrix index separately)
def lloyd(hs, K, it=20):
    hs = [h for h in hs]
    # init: sort by entropy, split
    order = np.argsort([H(h) / h.sum() for h in hs]); groups = np.array_split(order, K)
    sets = [quantize(sum(hs[i] for i in g)) for g in groups]
    for _ in range(it):
        assign = [int(np.argmin([cost(h, s) for s in sets])) for h in hs]
        sets = [quantize(sum((hs[i] for i in range(len(hs)) if assign[i] == k), np.zeros(256, np.int64))) for k in range(K)]
    return sets
Ks = [2, 4, 8, 16]
setsK = {K: [lloyd([hist[(r["layer"], r["expert"])][m] for r in meta if not is_test(r)], K) for m in range(3)] for K in Ks}
for r in meta:
    if not is_test(r): continue
    hs = hist[(r["layer"], r["expert"])]
    row = dict(layer=r["layer"], expert=r["expert"], SP=r["SP"])
    row["global_rel"] = sum(cost(hs[m], gq[m]) for m in range(3))
    row["own_entropy"] = sum(H(hs[m]) for m in range(3)) / 8
    row["own_quantised_table"] = sum(cost(hs[m], quantize(hs[m])) for m in range(3))
    # table cost estimate: symbols used span; ~ (span) x 14 bit + 2 bytes
    row["table_cost_est"] = sum(2 + 14 * (np.ptp(np.nonzero(hs[m])[0]) + 1) / 8 for m in range(3))
    for K in Ks:
        row[f"K{K}"] = sum(min(cost(hs[m], s) for s in setsK[K][m]) for m in range(3)) + 3 * np.log2(K) / 8
    rows.append(row)
keys = [k for k in rows[0] if k not in ("layer", "expert", "SP")]
print("mean over TEST (41) / without L39 E120 / L39 E120, bytes per record (bounds, no stream overhead)")
for k in keys:
    a = np.array([x[k] for x in rows]); o = [x[k] for x in rows if (x["layer"], x["expert"]) == (39, 120)][0]
    b = np.array([x[k] for x in rows if (x["layer"], x["expert"]) != (39, 120)])
    print(f"{k:22s} {a.mean():10.0f} {b.mean():10.0f} {o:10.0f}")
json.dump(rows, open(HERE + "/scale_adaptive.json", "w"), indent=0)
