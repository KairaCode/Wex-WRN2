"""Real encode + decode round trips on the 41 TEST records for the model variants. Every record is decoded
(stream hashes, CZ check, ceil inverse) and its SHA-256 compared with Z(source) of the stored record.
Usage: python3 real.py VARIANT [VARIANT...]   -> out/real_<VARIANT>.json
Sizes are unpadded record bytes ('used'); the stored records' used bytes are the baseline."""
import sys, json, ctypes, numpy as np, os
from multiprocessing import Pool
from common import *
train, test = split()
SET = os.environ.get("SET", "test")
if SET == "extra": test = [r["tag"] for r in EXTRA]
os.makedirs(W + "/out", exist_ok=True)
t40a = [t for t in train if BY[t]["expert"] == (37 * BY[t]["layer"] + 11) % 192]
t40b = [t for t in train if t not in t40a]

def q_rule_z8(wc, P):
    """WNR3 rule over the 15 symbols other than 8 (impossible under CZ); f[8] = 0."""
    keep = [s for s in range(16) if s != 8]
    f = np.zeros(wc.shape, np.int64); f[..., keep] = q_rule(wc[..., keep], P); return f

def s1_counts(tags): return sum(np.load(W + "/counts/%s.s1.npy" % t) for t in tags)

def build(v):
    """-> ('lib', blob) for WNM2 blobs coded by the tool library, or ('proto', Model)."""
    wc, sc = sum_counts(train)
    if v == "prod": return "lib", PROD_BLOB
    if v in ("lib_rule14_T80", "lib_rule14_T40a", "lib_rule14_T40b"):
        tg = {"T80": train, "T40a": t40a, "T40b": t40b}[v.split("_")[-1]]
        a, b = sum_counts(tg)
        blob = ctypes.create_string_buffer(59408); mid = ctypes.c_uint32()
        TOOLS.wex_nr3_build_model(np.ascontiguousarray(a.reshape(-1).astype(np.uint64)).ctypes.data_as(ctypes.c_void_p),
                                  np.ascontiguousarray(b.reshape(-1).astype(np.uint64)).ctypes.data_as(ctypes.c_void_p), blob, ctypes.byref(mid))
        return "lib", blob.raw
    if v == "lib_self": return "self", None
    # proto variants: name fields  P<p>_[z8]_[srel]_[K<k>]
    f = v.split("_"); P = int(f[0][1:]); z8 = "z8" in f; rel = "srel" in f; esc = "sesc" in f
    K = int([x for x in f if x.startswith("K")][0][1:]) if any(x.startswith("K") for x in f) else 1
    C = int([x for x in f if x.startswith("C")][0][1:]) if any(x.startswith("C") for x in f) else 0
    sf = q_rule(s1_counts(train) if rel else sc, P); smode = 1 if rel else 0; salpha = None
    if esc:   # anchor-relative, only the values seen in TRAIN + escape (then 8 raw bits)
        tr = s1_counts(train); keep = np.flatnonzero(tr.sum(0) > 0)
        red = np.zeros((3, len(keep) + 1), np.int64); red[:, :-1] = tr[:, keep]
        sf = q_rule(red, P); smode = 2; salpha = keep.tolist()
    if C:
        z = np.load(W + "/sets_P%d_C%d_K%d%s%s.npz" % (P, C, K, "_z8" if z8 else "", SFX))
        return "proto", Model(z["wf"], sf, P, smode=smode, thr=z["thr"], K=K, salpha=salpha)
    if K > 1:
        wf = np.load(W + "/ksets_K%d_P%d.npz" % (K, P))["wf"]
        if z8:   # requantise each set with f8 = 0 is not possible without its counts; keep as trained
            raise SystemExit("z8 with K>1 not supported")
    else:
        wf = q_rule_z8(wc, P) if z8 else q_rule(wc, P)
    return "proto", Model(wf, sf, P, smode=smode, salpha=salpha)

def run(args):
    v, tag = args
    kind, mdl = build(v) if not isinstance(BUILT, dict) else BUILT[v]
    r = BY[tag]; l, e = r["layer"], r["expert"]
    if kind in ("lib", "self"):
        if kind == "self":
            a, b = counts(tag); blob = ctypes.create_string_buffer(59408); mid = ctypes.c_uint32()
            TOOLS.wex_nr3_build_model(np.ascontiguousarray(a.reshape(-1).astype(np.uint64)).ctypes.data_as(ctypes.c_void_p),
                                      np.ascontiguousarray(b.reshape(-1).astype(np.uint64)).ctypes.data_as(ctypes.c_void_p), blob, ctypes.byref(mid))
            mdl = blob.raw
        cap = 2 * RAW; out = ctypes.create_string_buffer(cap); used = ctypes.c_size_t(); tot = ctypes.c_size_t()
        rc = TOOLS.wex_nr3_encode_record(mdl, entry(tag), coded(tag), r["flags"], (l << 16) | e, 2048, 512, out, cap, ctypes.byref(used), ctypes.byref(tot))
        assert rc == 0
        rec = out.raw[:tot.value]; dec = ctypes.create_string_buffer(RAW)
        ent = entry(tag); ent = ent[:8] + struct.pack("<I", tot.value) + ent[12:]   # the entry names the new record size
        ok = TOOLS.wex_nr3_decode(mdl, ent, rec, l, e, dec) == 0 and hashlib.sha256(dec.raw).hexdigest() == r["zsha"]
        return dict(tag=tag, used=used.value, total=tot.value, ok=ok, base_used=r["used"], base_total=r["total"])
    rec, sel = mdl.encode(tag)
    ok = mdl.decode_ok(tag, rec)
    S = 2160
    selhist = np.bincount(np.frombuffer(sel[S:S + 17280], np.uint8), minlength=mdl.wf.shape[0]).tolist()
    return dict(tag=tag, used=len(rec), total=(len(rec) + 4095) // 4096 * 4096, ok=ok, base_used=r["used"],
                base_total=r["total"], sel=selhist, table_bytes=mdl.table_bytes())

BUILT = None
if __name__ == "__main__":
    for v in sys.argv[1:]:
        BUILT = {v: build(v)}
        with Pool(2) as pool: rows = pool.map(run, [(v, t) for t in test])
        d = np.array([x["used"] - x["base_used"] for x in rows]); dt = np.array([x["total"] - x["base_total"] for x in rows])
        no = np.array([x["tag"] != "39_120" for x in rows])
        summ = dict(variant=v, n=len(rows), all_ok=all(x["ok"] for x in rows), mean_delta_used=float(d.mean()),
                    pct_raw=float(d.mean() / RAW * 100), mean_delta_used_no_outlier=float(d[no].mean()),
                    pct_raw_no_outlier=float(d[no].mean() / RAW * 100), outlier_delta=float(d[~no].sum()) if (~no).any() else None,
                    median_delta=float(np.median(d)), min_delta=float(d.min()), max_delta=float(d.max()),
                    mean_delta_padded=float(dt.mean()), mean_used_of_raw=float(np.mean([x["used"] for x in rows]) / RAW),
                    table_bytes=rows[0].get("table_bytes"))
        json.dump(dict(summary=summ, rows=rows), open(W + "/out/real_%s%s%s.json" % (v, "" if SET == "test" else "_" + SET, SFX), "w"), indent=1)
        print(json.dumps(summ), flush=True)
