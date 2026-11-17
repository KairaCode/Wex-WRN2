"""Independent re-derivation of the w1->w3 sibling static-table gain (floor lane's V3 split at layer 8).
Own context code (not load.py / static_sib.py), own quantiser (wex_nr2_quantize rule), ideal code lengths.
Also: is the variant / split choice reproducible when chosen on TRAIN only (2-fold CV inside TRAIN)?
  python3 sib_check.py JOBS -> out/sib_check.json"""
import os, sys, json, struct
import numpy as np
from multiprocessing import Pool
D = "/tmp/wnr3-data"; H = os.path.dirname(os.path.abspath(__file__)); RAW = 18800640
ENTRY = struct.Struct('<QIII6H6H3I3II')
TABLE = open(D + "/recs/entries.bin", "rb").read()
blob = np.fromfile(D + "/recs/model.bin", np.uint8)
PROD = np.frombuffer(blob[32:32 + 3 * 567 * 34].tobytes(), "<u2").reshape(3, 567, 17).astype(np.int64)
PRODF = np.diff(PROD, axis=2)[2]          # w3 tables (567,16)
NV = {"base": 567, "V1": 567 * 8, "V3": 7 * 9 * 5 * 8, "V3b": 7 * 9 * 8}   # V3b = srel x mx x sa (no sibd)

def mats(l, e):
    ent = ENTRY.unpack(TABLE[(l * 192 + e) * 72:(l * 192 + e + 1) * 72])
    b = np.fromfile(os.path.join(H, "out/coded/rec_%02d_%03d.coded" % (l, e)), np.uint8)
    rec = open(D + "/recs/rec_%02d_%03d.bin" % (l, e), "rb").read(32)
    out = []
    for m in (0, 2):
        R, NB = ent[4 + 2 * m], ent[5 + 2 * m]
        s = b[ent[16 + m]:ent[16 + m] + R * NB].astype(np.int64)
        w = b[ent[19 + m]:ent[19 + m] + R * NB * 16]
        q = np.stack([w & 15, w >> 4], 1).reshape(-1).astype(np.int64)      # nibble i: low then high
        out.append((s, q))
    return out, rec[26]   # anchor of w3

def ctx(l, e):
    ((s1, q1), (s3, q3)), anc3 = mats(l, e)
    n = q3.size; blk = np.arange(n) // 32; k = np.arange(n) % 32
    a3 = q3 & 7; a1 = q1 & 7
    srel = np.clip(s3[blk] - anc3, -3, 3) + 3
    sibd = np.clip(s1[blk] - s3[blk], -2, 2) + 2
    A = a3.reshape(-1, 32)
    pm = np.full_like(A, 8); pm[:, 1:] = A[:, :-1]
    mx = np.full_like(A, 8); mx[:, 1:] = np.maximum.accumulate(A, axis=1)[:, :-1]
    pm = pm.ravel(); mx = mx.ravel()
    base = (srel * 9 + pm) * 9 + mx
    c = {"base": base, "V1": base * 8 + a1, "V3": ((srel * 9 + mx) * 5 + sibd) * 8 + a1, "V3b": (srel * 9 + mx) * 8 + a1}
    return (l, e), {v: np.bincount(c[v] * 16 + q3, minlength=NV[v] * 16).reshape(NV[v], 16) for v in c}

def quant(c):
    c = c.astype(np.int64); f = np.empty_like(c)
    for r in range(c.shape[0]):
        row = c[r]; tot = row.sum(); spare = 16384 - 16
        sc = row * spare if tot else np.full(16, spare); dv = tot if tot else 16
        fr = 1 + sc // dv; rem = sc % dv; extra = 16384 - fr.sum()
        order = sorted(range(16), key=lambda s: (-rem[s], s))
        for s in order[:extra]: fr[s] += 1
        f[r] = fr
    return f

def bits(c, f): return float((c * -np.log2(f / 16384.0)).sum())

if __name__ == "__main__":
    train = [(l, (37 * l + 11) % 192) for l in range(40)] + [(l, (37 * l + 101) % 192) for l in range(40)]
    test = [(l, (53 * l + 7) % 192) for l in range(40)]
    with Pool(int(sys.argv[1])) as p: C = dict(p.starmap(ctx, train + test, chunksize=1))
    res = {}
    # A. held-out TEST: tables trained on all 80 TRAIN, global or split at layer 8
    def tables(keys, v, split):
        if split is None: return {"g": quant(sum(C[k][v] for k in keys))}
        return {"lo": quant(sum(C[k][v] for k in keys if k[0] < split)), "hi": quant(sum(C[k][v] for k in keys if k[0] >= split))}
    def score(keys_tr, keys_te, v, split):
        T = tables(keys_tr, v, split); d = []
        for k in keys_te:
            f = T["g"] if split is None else T["lo" if k[0] < split else "hi"]
            d.append(100 * (bits(C[k][v], f) - bits(C[k]["base"], PRODF)) / 8 / RAW)
        return np.array(d)
    for v in ("base", "V1", "V3", "V3b"):
        for split in (None, 8):
            d = score(train, test, v, split)
            res[f"TEST {v} split{split}"] = dict(mean=float(d.mean()), l0_7=float(d[:8].mean()), l8_39=float(d[8:].mean()), max=float(d.max()))
            print(f"TEST {v:4s} split={split}", {a: round(b, 5) for a, b in res[f'TEST {v} split{split}'].items()}, flush=True)
    # B. choice made on TRAIN only: 2-fold CV (expert set A <-> B), variants x split points
    A = train[:40]; B = train[40:]
    for v in ("V1", "V3", "V3b"):
        for split in (None, 4, 8, 12, 16):
            d = np.concatenate([score(A, B, v, split), score(B, A, v, split)])
            res[f"CV {v} split{split}"] = float(d.mean())
            print(f"CV-in-TRAIN {v:4s} split={split} mean {d.mean():.5f}", flush=True)
    json.dump(res, open(os.path.join(H, "out/sib_check.json"), "w"), indent=0)
