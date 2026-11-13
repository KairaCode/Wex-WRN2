"""Drive zs_kern.dll: original vs canonicalised (-0 -> +0) MXFP4 expert records, compare output bits.
Usage: python -I zs_drive.py <dll> <sample_dir> <out.json> <full_files_csv> <quick_all:0|1>
"""
import ctypes, json, os, sys, time
import numpy as np

dll_path, sample_dir, out_path, full_csv, quick_all = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
L = ctypes.CDLL(dll_path)
P = ctypes.c_void_p
L.zs_run.argtypes = [ctypes.c_int, P, P, P, P, P, P, ctypes.c_int, ctypes.c_int, ctypes.c_int, P, P]
L.zs_run.restype = ctypes.c_int
L.zs_set_tables.argtypes = [P, P]
L.zs_pack_rows16.argtypes = [P, P, P, P, ctypes.c_int, ctypes.c_int]

KN = ["K1_mxfp4_avx2", "K2_mxfp4_scalar", "K2c_mxfp4_scalar_fma", "K3_idot_avx2", "K4_idot_scalar",
      "K5_rows16order_avx2", "K6_rows16order_scalar", "K7_batch_rows16order_avx2_S4", "K8_rows16_v10_avx2",
      "K9a_cuda_grouped_fma_emul", "K9b_cuda_grouped_nofma_emul", "K10_cuda_v4f_dot_fp4_emul",
      "K11_vk_chain_moe_gv_acc0_emul", "K12_NEGCTRL_vk_firstterm_emul"]
AN = ["gauss", "gauss_25pct+0_25pct-0", "all-0", "all+0", "one_nonzero_rest-0", "denormals_10pct",
      "inf_at_100", "nan_at_200", "huge_1e30"]

v = np.arange(256)
e8lut = np.where(v == 255, np.float32(np.nan), np.ldexp(np.float32(1), (v - 127)).astype(np.float32)).astype(np.float32)
e8cuda = np.ldexp(np.float32(1), np.where(v == 0, -127, v - 127)).astype(np.float32)   # 255 -> 2^128 = inf
assert e8lut[0] == np.float32(2.0 ** -127) and np.isinf(e8cuda[255])
L.zs_set_tables(e8lut.ctypes.data, e8cuda.ctypes.data)

SC, WB = 368640, 5898240
REC = 3 * SC + 3 * WB
rng = np.random.default_rng(20261107)

def canon(a):
    lo = a & 15; hi = a >> 4
    lo = np.where(lo == 8, 0, lo); hi = np.where(hi == 8, 0, hi)
    return (lo | (hi << 4)).astype(np.uint8)

def act(I, k):
    x = rng.standard_normal(I).astype(np.float32)
    if k == 1:
        r = rng.integers(0, 4, I); x[r == 0] = np.float32(0.0); x[r == 1] = np.float32(-0.0)
    elif k == 2: x[:] = np.float32(-0.0)
    elif k == 3: x[:] = np.float32(0.0)
    elif k == 4: x[:] = np.float32(-0.0); x[7] = 1.0
    elif k == 5:
        m = rng.integers(0, 10, I) == 0; x[m] = np.where(rng.integers(0, 2, m.sum()) == 1, np.float32(1e-41), np.float32(-1e-41))
    elif k == 6: x[100] = np.inf
    elif k == 7: x[200] = np.nan
    elif k == 8: x *= np.float32(1e30)
    return np.ascontiguousarray(x, dtype=np.float32)

xq = np.zeros(4 * 5120, np.int8); xsc = np.zeros(4 * 160, np.float32)
def run(k, w, s, pd, ps, x, I, O):
    S = 1
    if k == 7:
        S = 4
        xx = np.empty(4 * I, np.float32)
        xx[:I] = x; xx[I:2 * I] = -x
        t = x.copy(); t[1::2] = np.float32(-0.0); xx[2 * I:3 * I] = t
        xx[3 * I:] = x * np.float32(1e-30)
        x = xx
    y = np.empty(S * O, np.float32)
    r = L.zs_run(k, y.ctypes.data, w.ctypes.data, s.ctypes.data, pd.ctypes.data, ps.ctypes.data, x.ctypes.data,
                 S, I, O, xq.ctypes.data, xsc.ctypes.data)
    assert r == 0
    return y

def pack(w, s, O, I):
    pd = np.empty_like(w); ps = np.empty_like(s)
    L.zs_pack_rows16(pd.ctypes.data, ps.ctypes.data, w.ctypes.data, s.ctypes.data, O, I)
    return pd, ps

def new_stats():
    return {k: dict(compared=0, mismatch_value=0, mismatch_signzero_only=0, mismatch_nan_payload=0,
                    zero_outputs=0, neg_zero_outputs=0, nan_outputs=0) for k in KN}

def compare(st, a, b):
    ua = a.view(np.uint32); ub = b.view(np.uint32)
    d = ua != ub
    both0 = ((ua | ub) & 0x7fffffff) == 0
    bothnan = np.isnan(a) & np.isnan(b)
    st["compared"] += a.size
    st["mismatch_signzero_only"] += int((d & both0).sum())
    st["mismatch_nan_payload"] += int((d & bothnan & ~both0).sum())
    st["mismatch_value"] += int((d & ~both0 & ~bothnan).sum())
    st["zero_outputs"] += int(((ua & 0x7fffffff) == 0).sum())
    st["neg_zero_outputs"] += int((ua == 0x80000000).sum())
    st["nan_outputs"] += int(np.isnan(a).sum())

def test_matrix(st, wA, wB, s, I, O, ks, acts):
    pdA, psA = pack(wA, s, O, I); pdB, psB = pack(wB, s, O, I)
    for a in acts:
        x = act(I, a)
        for k in ks:
            ya = run(k, wA, s, pdA, psA, x, I, O); yb = run(k, wB, s, pdB, psB, x, I, O)
            compare(st[KN[k]], ya, yb)

def load(fn):
    rec = np.fromfile(os.path.join(sample_dir, fn), dtype=np.uint8)
    assert rec.size == REC, (fn, rec.size)
    s1, s2, s3 = rec[0:SC], rec[SC:2 * SC], rec[2 * SC:3 * SC]
    w1, w2, w3 = rec[3 * SC:3 * SC + WB], rec[3 * SC + WB:3 * SC + 2 * WB], rec[3 * SC + 2 * WB:]
    return [(np.ascontiguousarray(w1), np.ascontiguousarray(s1), 5120, 2304),
            (np.ascontiguousarray(w2), np.ascontiguousarray(s2), 2304, 5120),
            (np.ascontiguousarray(w3), np.ascontiguousarray(s3), 5120, 2304)]

res = {"real_full_suite": new_stats(), "synthetic_zero_rows": new_stats(), "real_quick_all_experts": new_stats(),
       "positive_control_one_flipped_weight": new_stats()}
meta = {"full_files": [], "quick_files": 0, "code8": 0, "code0": 0, "nibbles": 0, "scale_codes_seen_min": 255, "scale_codes_seen_max": 0}
t0 = time.time()
full_files = [f for f in full_csv.split(",") if f]
ALLK = list(range(14)); ALLA = list(range(9))
for fi, fn in enumerate(full_files):
    mats = load(fn); meta["full_files"].append(fn)
    for (w, s, I, O) in mats:
        c = canon(w)
        meta["code8"] += int(((w & 15) == 8).sum() + ((w >> 4) == 8).sum())
        meta["code0"] += int(((w & 15) == 0).sum() + ((w >> 4) == 0).sum()); meta["nibbles"] += 2 * w.size
        meta["scale_codes_seen_min"] = min(meta["scale_codes_seen_min"], int(s.min())); meta["scale_codes_seen_max"] = max(meta["scale_codes_seen_max"], int(s.max()))
        test_matrix(res["real_full_suite"], w, c, s, I, O, ALLK, ALLA)
    # synthetic: first 64 rows of w1 and w2, rows 0..31 all-zero codes (random 0/8 mix vs canonical 0)
    for (w, s, I, O) in (mats[0], mats[1]):
        rb = I // 2
        sa = w[:64 * rb].copy(); ss = s[:64 * (I // 32)].copy()
        lo = np.where(rng.integers(0, 2, 32 * rb) == 1, 8, 0); hi = np.where(rng.integers(0, 2, 32 * rb) == 1, 8, 0)
        sa[:32 * rb] = (lo | (hi << 4)).astype(np.uint8)
        test_matrix(res["synthetic_zero_rows"], sa, canon(sa), ss, I, 64, ALLK, ALLA)
    if fi == 0:   # positive control: canonical copy with ONE nonzero weight sign-flipped (code 1 -> 9) in row 0
        w, s, I, O = mats[0]; c = canon(w); c2 = c.copy()
        j = int(np.nonzero((c2[:I // 2] & 15) == 1)[0][0]); c2[j] = (c2[j] & 0xF0) | 9
        test_matrix(res["positive_control_one_flipped_weight"], c, c2, s, I, O, ALLK, [0])
    print("full", fn, round(time.time() - t0, 1), "s", flush=True)

if quick_all:
    m = json.load(open(os.path.join(sample_dir, "manifest.json")))
    files = [e["file"] for e in m["experts"]]
    QK = [0, 5, 8, 9, 11, 12]; QA = [0, 1]
    for fn in files:
        for (w, s, I, O) in load(fn):
            test_matrix(res["real_quick_all_experts"], w, canon(w), s, I, O, QK, QA)
        meta["quick_files"] += 1
        if meta["quick_files"] % 20 == 0: print("quick", meta["quick_files"], round(time.time() - t0, 1), "s", flush=True)

for g in res: res[g] = {k: v for k, v in res[g].items() if v["compared"]}
out = {"meta": meta, "act_sets_full": AN, "quick_kernels": [KN[k] for k in [0, 5, 8, 9, 11, 12]], "quick_act_sets": AN[:2],
       "results": res, "seconds": round(time.time() - t0, 1)}
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps(out, indent=1))
