# Held-out IDEAL code lengths (bounds, quantised 14-bit CDFs exactly like wex_nr2_quantize) of context models
# for the coded scale streams. Train = TRAIN records (80), score = TEST records (41 incl. L39 E120).
# Contexts only use symbols of the SAME stream (Cs-chunk) that precede the symbol, or symbols of an earlier
# matrix's stream (sibling), or the record header (anchors), so every model keeps scales decodable before
# weights. Output: scale_bounds.json + a printed table (bytes per record, no stream overhead).
# usage: python3 scale_bounds.py [Cs=512]
import numpy as np, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
CS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
meta = json.load(open(HERE + "/meta.json"))
def is_test(r): return r["expert"] == (53 * r["layer"] + 7) % 192 or (r["layer"], r["expert"]) == (39, 120)
RAW = 18800640

def quantize(counts):
    """wex_nr2_quantize for one row of counts (len n). returns frequencies (sum 16384)."""
    n = len(counts); total = int(counts.sum()); spare = 16384 - n
    if total:
        scaled = counts.astype(object) * spare
        freq = np.array([1 + int(x // total) for x in scaled]); rem = np.array([int(x % total) for x in scaled])
    else:
        freq = np.full(n, 1 + spare // n); rem = np.full(n, spare % n)
    extra = 16384 - int(freq.sum())
    order = sorted(range(n), key=lambda s: (-rem[s], s))
    for s in order[:extra]: freq[s] += 1
    return freq

def load(r):
    s = np.fromfile(f"{HERE}/scales/{r['layer']:02d}_{r['expert']:03d}.bin", np.uint8)
    out = []; off = 0
    for (R, C) in r["scale_shapes"]:
        out.append(s[off:off + R * C].reshape(R, C).astype(np.int32)); off += R * C
    return out

def clampd(x, k): return np.clip(x, -k, k) + k

# context functions: (mats, anchors, m) -> (symbol array int, context array int, ncontexts)
def ctx_features(mats, anchors, m):
    a = mats[m]; R, C = a.shape; A = anchors[m]
    flat = a.ravel(); n = flat.size; idx = np.arange(n)
    pos_in_stream = idx % CS
    row = idx // C; col = idx % C
    rel = flat - A
    f = {}
    # left neighbour in the row, available if col>0 (always same stream unless stream starts mid-row at this index)
    left_ok = (col > 0) & (pos_in_stream > 0)
    left = np.where(left_ok, np.roll(rel, 1), 99)
    up_ok = (row > 0) & (pos_in_stream >= C)
    up = np.where(up_ok, np.roll(rel, C), 99)
    left2_ok = (col > 1) & (pos_in_stream > 1)
    left2 = np.where(left2_ok, np.roll(rel, 2), 99)
    f["left"] = np.where(left_ok, clampd(left, 2), 5)                  # 6
    f["up"] = np.where(up_ok, clampd(up, 2), 5)                         # 6
    f["left2"] = np.where(left2_ok, clampd(left2, 2), 5)
    # running row mean of earlier symbols in the row & stream (count-limited)
    cs = np.cumsum(rel)
    # sum of previous up-to-8 symbols in row within stream
    k = 8
    start = np.maximum(np.maximum(idx - k, row * C), idx - pos_in_stream)
    cnt = idx - start
    s_prev = np.where(idx > 0, cs[idx - 1], 0) - np.where(start > 0, cs[np.maximum(start - 1, 0)], 0)
    s_prev = np.where(start > 0, s_prev, np.where(idx > 0, cs[idx - 1], 0))
    mean8 = np.where(cnt > 0, np.floor(2 * s_prev / np.maximum(cnt, 1) + 0.5) , 99)   # half-steps
    f["mean8"] = np.where(cnt > 0, clampd(mean8.astype(np.int64), 4), 9)        # 10
    if m == 2:
        sib = mats[0].ravel() - anchors[0]
        f["sib"] = clampd(sib, 2)                                         # 5
        sibdiff = mats[0].ravel() - anchors[0]
    return rel, f

MODELS = {
    "abs_order0 (today)": None,
    "rel_order0": [],
    "rel|left": ["left"],
    "rel|up": ["up"],
    "rel|left,up": ["left", "up"],
    "rel|left,left2": ["left", "left2"],
    "rel|mean8": ["mean8"],
    "rel|mean8,up": ["mean8", "up"],
    "rel|sib(m2)": ["sib"],
    "rel|left,sib(m2)": ["left", "sib"],
    "rel|mean8,sib(m2)": ["mean8", "sib"],
}
SIZES = {"left": 6, "up": 6, "left2": 6, "mean8": 10, "sib": 5}

def ctx_of(feats, names, m):
    c = np.zeros_like(next(iter(feats.values()))) if feats else None
    n = 1
    for nm in names:
        if nm not in feats: continue
        c = c * SIZES[nm] + feats[nm]; n *= SIZES[nm]
    return (c if c is not None else 0), n

def main():
    data = []
    for r in meta:
        mats = load(r); anchors = r["anchors"]
        per = []
        for m in range(3):
            rel, feats = ctx_features(mats, anchors, m)
            per.append((mats[m].ravel(), rel, feats))
        data.append((r, per))
    res = {}
    for name, names in MODELS.items():
        # train counts
        tables = {}
        for r, per in data:
            if is_test(r): continue
            for m in range(3):
                flat, rel, feats = per[m]
                sym = flat if names is None else (rel + 128) & 255
                c, n = (0, 1) if names is None else ctx_of(feats, names, m)
                if m not in tables: tables[m] = np.zeros((n, 256), np.int64)
                np.add.at(tables[m], (c, sym), 1) if n > 1 else np.add.at(tables[m][0], sym, 1)
        logp = {m: np.log2(np.array([quantize(row) for row in t]) / 16384.0) for m, t in tables.items()}
        rows = []
        for r, per in data:
            if not is_test(r): continue
            bits = 0.0
            for m in range(3):
                flat, rel, feats = per[m]
                sym = flat if names is None else (rel + 128) & 255
                c, n = (0, 1) if names is None else ctx_of(feats, names, m)
                bits -= logp[m][c, sym].sum() if n > 1 else logp[m][0][sym].sum()
            rows.append(dict(layer=r["layer"], expert=r["expert"], bytes=bits / 8, SP=r["SP"]))
        res[name] = rows
        b = np.array([x["bytes"] for x in rows]); no = np.array([x["bytes"] for x in rows if (x["layer"], x["expert"]) != (39, 120)])
        o = [x["bytes"] for x in rows if (x["layer"], x["expert"]) == (39, 120)][0]
        print(f"{name:24s} mean {b.mean():10.0f} B  w/o L39E120 {no.mean():10.0f}  L39E120 {o:10.0f}", flush=True)
    sp = np.array([x["SP"] for x in res["rel_order0"]])
    print("stored SP mean", sp.mean())
    json.dump(dict(Cs=CS, models=res), open(HERE + f"/scale_bounds_cs{CS}.json", "w"), indent=0)

main()
