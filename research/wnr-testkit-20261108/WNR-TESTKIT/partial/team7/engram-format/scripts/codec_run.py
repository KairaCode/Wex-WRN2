"""Train the static model, encode every sample row held-out, verify lossless round trip, time decode.

Usage: python -I codec_run.py ROWS_NPZ ERROW_DLL OUT_JSON OUT_LENS_NPZ
"""
import ctypes as ct, json, sys, time
import numpy as np

rows_npz, dll_path, out_json, out_lens = sys.argv[1:5]
d = np.load(rows_npz)
W = np.ascontiguousarray(d['W']); S = np.ascontiguousarray(d['S']); C = d['C']; L = d['L']; SEG = d['SEG']
R = len(W)
lib = ct.CDLL(dll_path)
P = ct.c_void_p
lib.erow_count.argtypes = [P, P, ct.c_int64, P]
lib.erow_encode.argtypes = [P, P, P, ct.c_int64, ct.c_int, P, P, ct.c_int64]; lib.erow_encode.restype = ct.c_int64
lib.erow_verify.argtypes = [P, P, P, ct.c_int64, ct.c_int, P, P]; lib.erow_verify.restype = ct.c_int64
lib.erow_bench.argtypes = [P, P, P, P, ct.c_int64, ct.c_int, ct.c_int, P, P]; lib.erow_bench.restype = ct.c_uint64
NCTX, CX_M, CX_S = 291, 272, 274


def ptr(a):
    return a.ctypes.data_as(P)


def counts(Wx, Sx):
    c = np.zeros((NCTX, 16), np.int64)
    lib.erow_count(ptr(Wx), ptr(Sx), len(Wx), ptr(c))
    return c


def quantize(c):
    cum = np.zeros((NCTX, 17), np.uint16)
    for k in range(NCTX):
        valid = np.ones(16, bool)
        if CX_M <= k < CX_S:
            valid[8:] = False
        f = np.where(valid, c[k] + 0.5, 0.0)
        q = np.where(valid, np.maximum(1, np.round(f / f.sum() * 4096)), 0).astype(np.int64)
        q[np.argmax(q)] += 4096 - q.sum()
        assert q.sum() == 4096 and (q[valid] >= 1).all()
        cum[k, 1:] = np.cumsum(q)
    return np.ascontiguousarray(cum)


def encode(cum, Wx, Sx, g=1):
    n = len(Wx); ng = -(-n // g)
    out = np.zeros(n * 900 + 4096, np.uint8); lens = np.zeros(ng, np.int32)
    tot = lib.erow_encode(ptr(cum), ptr(Wx), ptr(Sx), n, g, ptr(out), ptr(lens), out.size)
    assert tot > 0
    bad = lib.erow_verify(ptr(cum), ptr(out), ptr(lens), n, g, ptr(Wx), ptr(Sx))
    return out[:tot].copy(), lens, int(bad)

tr = (C % 2) == 0
res = {'rows': int(R), 'model_bytes': NCTX * 17 * 2}
c_even, c_odd, c_all = counts(W[tr].copy(), S[tr].copy()), counts(W[~tr].copy(), S[~tr].copy()), None
c_all = c_even + c_odd
m_even, m_odd, m_all = quantize(c_even), quantize(c_odd), quantize(c_all)
np.save(out_lens.replace('.npz', '_model_all.npy'), m_all)

lens_ho = np.zeros(R, np.int32)
bad_total = 0
for model, sel in ((m_even, ~tr), (m_odd, tr)):
    Wx, Sx = W[sel].copy(), S[sel].copy()
    buf, lens, bad = encode(model, Wx, Sx, 1)
    lens_ho[sel] = lens; bad_total += bad
_, lens_in, bad_in = encode(m_all, W, S, 1)
res['verify_mismatching_rows_heldout'] = bad_total
res['verify_mismatching_rows_insample'] = bad_in
qs = [0, 0.01, 0.1, 1, 10, 50, 90, 99, 99.9, 99.99, 100]
for name, ln in (('heldout', lens_ho), ('insample', lens_in)):
    res[f'row_bytes_{name}'] = {'mean': float(ln.mean()), 'std': float(ln.std()),
                                'percentiles': dict(zip(map(str, qs), np.percentile(ln, qs).tolist())),
                                'max': int(ln.max()), 'min': int(ln.min()),
                                'frac_ge_264_fallback': float((ln >= 264).mean()),
                                **{f'frac_gt_{s}': float((ln > s).mean()) for s in (216, 220, 224, 228, 232, 240)}}
res['row_bytes_heldout_by_layer'] = {int(v): float(lens_ho[L == v].mean()) for v in (1, 14)}
res['row_bytes_heldout_by_ngram'] = {f'L{v}_ng{g + 2}': float(lens_ho[(L == v) & (SEG // 8 == g)].mean()) for v in (1, 14) for g in range(3)}

# group coding (format c): N consecutive sample rows (same 1024/128-row chunk -> adjacent table rows) in one stream
grp = {}
order = np.argsort(C * (1 << 40) + d['R'], kind='stable')
Wo, So = W[order].copy(), S[order].copy()
for g in (1, 2, 4, 8, 16, 32, 64):
    buf, lens, bad = encode(m_all, Wo, So, g)
    grp[g] = {'bytes_per_row': len(buf) / R, 'verify_bad_groups': bad, 'max_group_bytes': int(lens.max())}
res['group_insample'] = grp

# decode timing: 48 random rows per token from a resident buffer of per-row streams (in-sample model)
buf1, lens1, _ = encode(m_all, Wo, So, 1)
offs1 = np.concatenate([[0], np.cumsum(lens1)[:-1]]).astype(np.int64)
raw = np.ascontiguousarray(np.concatenate([Wo, So], 1))  # 264-B raw rows (baseline copy)
rng = np.random.default_rng(1)
NT = 20000
idx = rng.integers(0, R, NT * 48).astype(np.int64)
Wout = np.zeros((48, 256), np.uint8); Sout = np.zeros((48, 8), np.uint8)
bench = {}
for name, g, mode, b, o in (('rans_row', 1, 0, buf1, offs1), ('raw_copy_264', 1, 1, raw.ravel(), offs1)):
    ts = []
    for rep in range(5):
        t0 = time.perf_counter(); lib.erow_bench(ptr(m_all), ptr(b), ptr(o), ptr(idx), len(idx), g, mode, ptr(Wout), ptr(Sout)); ts.append(time.perf_counter() - t0)
    t = min(ts)
    bench[name] = {'us_per_token_48rows': t / NT * 1e6, 'ns_per_row': t / len(idx) * 1e9, 'reps_s': ts}
for g in (4, 16, 64):
    bg, lg, _ = encode(m_all, Wo, So, g)
    og = np.concatenate([[0], np.cumsum(lg)[:-1]]).astype(np.int64)
    ii = idx[: NT * 48 // max(1, g // 4)]
    ts = []
    for rep in range(3):
        t0 = time.perf_counter(); lib.erow_bench(ptr(m_all), ptr(bg), ptr(og), ptr(ii), len(ii), g, 0, ptr(Wout), ptr(Sout)); ts.append(time.perf_counter() - t0)
    t = min(ts)
    bench[f'rans_group{g}'] = {'us_per_token_48rows': t / len(ii) * 48 * 1e6, 'ns_per_row': t / len(ii) * 1e9}
res['decode_bench_single_thread'] = bench
# decoded output check for the bench path (last token)
last = idx[-48:]
res['bench_last_token_exact'] = bool((Wout[(np.arange(len(idx) - 48, len(idx)) % 48)] == Wo[last]).all())
np.savez(out_lens, lens_heldout=lens_ho, lens_insample=lens_in, L=L, SEG=SEG, C=C, R=d['R'])
json.dump(res, open(out_json, 'w'), indent=1)
print(json.dumps(res, indent=1))
