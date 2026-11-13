"""Canonical-zero (WNR2z: MXFP4 code 8 = -0 rewritten to code 0 = +0) GPU bit test on KAIRA's
DeepSeek V4 CUDA backend (RTX 5090, sm_120 / sm_120a). Drives build/zt_generic.dll or build/zt_dg.dll
(zt_harness.cu = the worktree's backend_cuda_dsv4.cu included verbatim + C glue).

For every tested matrix and activation row, each kernel runs on 4 weight versions:
  A  original records          B  canonicalised (every nibble 8 -> 0)
  A2 original again (negative control: determinism)
  P  positive control: in every row the first nonzero code gets its sign flipped (code ^= 8);
     a row without a nonzero code gets code 1 (+0.5) at column 0.
and output bits are compared A vs B, A vs A2, A vs P.

    python -I zt_run.py MODE DLL OUT.json [--per-layer N] [--env K=V ...]
MODE: kernels  every FP4 expert kernel at V4.1 shapes (w1/w3 [2304x5120], w2 [5120x2304]) + synthetic zero rows
      api      dsv4_cuda_expert_group at V4.1 shapes (4 experts/layer) and dsv4_cuda_moe (fused v4f_moe_kernel
               and its launches fallback) on V4.1 experts cropped to V4 shapes (6 x 4096/2048)
      dg       DeepGEMM routed-expert paths (dg_moe<256>, dg_moe_contiguous_fast<256>, dg_moe_batch_contiguous
               grouped + legacy) on V4.1 experts cropped to V4 shapes (DeepGEMM flavour only)
      shipped  like api (expert_group + moe), but through the shipped app DLL's own dsv4_cuda_* exports
               (DLL = app/colibri-optimized/coli_cuda_dsv4*.dll); DSV4_* env must be set before start
      mma      instruction micro-test of the sm_120a block-scaled mma.sync forms (DeepGEMM flavour only)
Reads U:\\models\\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4 read-only; writes only OUT.json (+ .log via caller).
"""
import ctypes as C, json, os, random, struct, sys, time
import numpy as np

MODEL_DIR = r'U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
CUDA_BIN = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64'
LIMIT = 10.0   # swiglu clamp passed to the MoE entry points (same value for every version)


def picks():   # = lab/wnr3-5090-20261107/g0/t2_g0.py picks(): seed 20261107, 4 experts per layer (160)
    rng = random.Random(20261107)
    return [(L, e) for L in range(40) for e in sorted(rng.sample(range(192), 4))]


class Shards:   # minimal copy of t2_g0.py Shards (safetensors reader, read-only)
    def __init__(self):
        self.wm = json.load(open(os.path.join(MODEL_DIR, 'model.safetensors.index.json')))['weight_map']
        self.hdr = {}

    def tensor(self, name):
        f = self.wm[name]
        if f not in self.hdr:
            with open(os.path.join(MODEL_DIR, f), 'rb') as fh:
                n = struct.unpack('<Q', fh.read(8))[0]
                self.hdr[f] = (8 + n, json.loads(fh.read(n)))
        base, h = self.hdr[f]
        t = h[name]
        a, b = t['data_offsets']
        with open(os.path.join(MODEL_DIR, f), 'rb') as fh:
            fh.seek(base + a)
            buf = fh.read(b - a)
        assert len(buf) == b - a
        return np.frombuffer(buf, np.uint8).copy(), t['shape'], t['dtype']

    def expert(self, L, e):
        """{m: (w [O, I/2] uint8, s [O, I/32] uint8)} for m in 1, 2, 3"""
        out = {}
        for m in (1, 2, 3):
            s, ss, sd = self.tensor(f'layers.{L}.ffn.experts.{e}.w{m}.scale')
            w, ws, wd = self.tensor(f'layers.{L}.ffn.experts.{e}.w{m}.weight')
            assert wd == 'I8' and sd == 'F8_E8M0' and ws[0] == ss[0] and ws[1] * 2 == ss[1] * 32, (L, e, m, ws, wd, ss, sd)
            out[m] = (w.reshape(ws), s.reshape(ss))
        return out


# ---- FP4 helpers ----
def codes_of(w):   # [O, I/2] packed -> [O, I] nibble codes, low nibble first
    c = np.empty((w.shape[0], w.shape[1] * 2), np.uint8)
    c[:, 0::2] = w & 15
    c[:, 1::2] = w >> 4
    return c


def pack(c):
    return np.ascontiguousarray((c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8))


def canon(w):
    lo, hi = w & 15, w >> 4
    return np.ascontiguousarray((np.where(lo == 8, 0, lo) | (np.where(hi == 8, 0, hi) << 4)).astype(np.uint8))


def posctl(w):
    c = codes_of(w)
    nz = (c & 7) != 0
    has = nz.any(1)
    j = nz.argmax(1)
    r = np.arange(c.shape[0])
    c[r[has], j[has]] ^= 8
    c[r[~has], 0] = 1
    return pack(c)


F4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def allneg0(c):   # x making every product w_i * x_i of a row with codes c equal to -0 (code 8 positions: x = +1)
    return np.where(c == 8, np.float32(1), np.where(c < 8, np.float32(-0.0), np.float32(0.0))).astype(np.float32)


def cancel(c, s_row):   # allneg0 + one exactly cancelling pair (q, q|8) -> exact-zero partial sum, then -0 products
    x = allneg0(c)
    I = c.size
    for b in range(I // 32):
        blk = c[b * 32:(b + 1) * 32]
        for q in range(1, 8):
            i = np.nonzero(blk == q)[0]
            j = np.nonzero(blk == (q | 8))[0]
            if i.size and j.size:
                x[b * 32 + i[0]] = 1.0
                x[b * 32 + j[0]] = 1.0
                return x
    return x


AN = ['gauss', 'gauss*0.01', 'laplace', 'gauss_25%+0_25%-0', 'all-0', 'all+0', 'one_nonzero_rest-0', 'denormals_10%',
      'inf_at_100', 'nan_at_200', 'gauss*1e30', '-row0', 'row0*1e-30', 'adv_all_products-0(row r0)',
      'adv_exact_cancel_then-0(row r0)', 'adv_all_products-0(row r1)']


def acts(I, codes, scales, rng):
    X = np.empty((16, I), np.float32)
    g = lambda: rng.standard_normal(I).astype(np.float32)
    X[0] = g(); X[1] = g() * np.float32(0.01); X[2] = rng.laplace(size=I).astype(np.float32)
    x = g(); r = rng.integers(0, 4, I); x[r == 0] = 0.0; x[r == 1] = np.float32(-0.0); X[3] = x
    X[4] = np.float32(-0.0); X[5] = 0.0
    X[6] = np.float32(-0.0); X[6][7] = 1.0
    x = g(); m = rng.integers(0, 10, I) == 0
    x[m] = np.where(rng.integers(0, 2, int(m.sum())) == 1, np.float32(1e-41), np.float32(-1e-41)); X[7] = x
    x = g(); x[100] = np.inf; X[8] = x
    x = g(); x[200] = np.nan; X[9] = x
    X[10] = g() * np.float32(1e30)
    X[11] = -X[0]
    X[12] = X[0] * np.float32(1e-30)
    n8 = (codes == 8).sum(1)
    order = np.argsort(-n8, kind='stable')
    r0, r1 = int(order[0]), int(order[1])
    X[13] = allneg0(codes[r0]); X[14] = cancel(codes[r0], scales[r0]); X[15] = allneg0(codes[r1])
    return np.ascontiguousarray(X), (r0, r1)


# ---- comparison ----
POP = np.array([bin(i).count('1') for i in range(256)], np.uint8)


def popcount(u):
    return int(POP[u.view(np.uint8)].sum())


def new_cmp():
    return dict(compared=0, diff_values=0, diff_bits=0, diff_signzero_only=0, diff_nan_payload=0)


def cmp_into(st, a, b):
    ua = a.view(np.uint32 if a.dtype.itemsize == 4 else np.uint16)
    ub = b.view(ua.dtype)
    sign = 0x80000000 if ua.dtype == np.uint32 else 0x8000
    mag = sign - 1
    d = ua != ub
    both0 = ((ua & mag) == 0) & ((ub & mag) == 0)
    if ua.dtype == np.uint32:
        bothnan = np.isnan(a) & np.isnan(b)
    else:   # bf16 NaN: exponent all ones, mantissa nonzero
        isn = lambda u: ((u & 0x7F80) == 0x7F80) & ((u & 0x7F) != 0)
        bothnan = isn(ua) & isn(ub)
    st['compared'] += int(ua.size)
    st['diff_values'] += int(d.sum())
    st['diff_bits'] += popcount(ua ^ ub)
    st['diff_signzero_only'] += int((d & both0).sum())
    st['diff_nan_payload'] += int((d & bothnan & ~both0).sum())


def new_obs():
    return dict(zero_A=0, neg_zero_A=0, zero_B=0, neg_zero_B=0, nan_A=0, unwritten_A=0)


def obs_into(o, a, b):   # float32 outputs, or uint16 bf16 bit patterns
    if a.dtype == np.uint16:
        u, v, sign, unw = a, b, 0x8000, 0xA5A5
        nan = int((((u & 0x7F80) == 0x7F80) & ((u & 0x7F) != 0)).sum())
    else:
        u, v, sign, unw = a.view(np.uint32), b.view(np.uint32), 0x80000000, 0xA5A5A5A5
        nan = int(np.isnan(a).sum())
    o['zero_A'] += int(((u & (sign - 1)) == 0).sum()); o['neg_zero_A'] += int((u == sign).sum())
    o['zero_B'] += int(((v & (sign - 1)) == 0).sum()); o['neg_zero_B'] += int((v == sign).sum())
    o['nan_A'] += nan; o['unwritten_A'] += int((u == unw).sum())


def new_entry():
    return dict(canon=new_cmp(), rerun=new_cmp(), posctl=new_cmp(), obs=new_obs(), canon_by_act={})


def record(ent, ya, yb, ya2, yp, T=None, O=None):
    cmp_into(ent['canon'], ya, yb); cmp_into(ent['rerun'], ya, ya2); cmp_into(ent['posctl'], ya, yp); obs_into(ent['obs'], ya, yb)
    if T and (ya.view(np.uint8) != yb.view(np.uint8)).any():   # per activation set breakdown of canon diffs
        da = (ya.reshape(T, -1) != yb.reshape(T, -1)).sum(1) if ya.dtype == np.uint16 else (ya.reshape(T, -1).view(np.uint32) != yb.reshape(T, -1).view(np.uint32)).sum(1)
        for t in np.nonzero(da)[0]:
            ent['canon_by_act'][AN[t]] = ent['canon_by_act'].get(AN[t], 0) + int(da[t])


# ---- DLL ----
def load(dll):
    os.add_dll_directory(CUDA_BIN)
    lib = C.CDLL(dll)
    P = C.c_void_p
    lib.zt_kernel_name.restype = C.c_char_p
    lib.zt_run.restype = C.c_longlong
    lib.zt_run.argtypes = [C.c_int, C.c_int, P, C.c_int, P]
    lib.zt_set_matrix.argtypes = [C.c_int, P, P, C.c_int, C.c_int]
    lib.zt_putenv.argtypes = [C.c_char_p]
    lib.zt_api_upload.argtypes = [C.c_int, C.c_int, P, P, C.c_int, C.c_int]
    lib.zt_api_expert_group.argtypes = [C.c_int, P, C.c_float, P, P]
    lib.zt_api_shared.argtypes = [C.c_int, P, P, C.c_int, C.c_int]
    lib.zt_api_moe.argtypes = [C.c_int, P, C.c_float, P, P]
    if lib.zt_flavour():
        lib.zt_dg_upload.argtypes = [C.c_int] + [P] * 6
        lib.zt_dg_moe.argtypes = [C.c_int, P, C.c_int, P, P, C.c_float, P, P, P]
        lib.zt_mma.argtypes = [C.c_int, C.c_int] + [P] * 6
    return lib


def p(a):
    return C.c_void_p(a.ctypes.data)


def run_kernels(lib, res, group, mats, rng, kernels, meta):
    """mats: list of (w [O,I/2], s [O,I/32]) (version A). Runs every kernel on A, B, A, P."""
    for w, s in mats:
        O, I = w.shape[0], w.shape[1] * 2
        codes = codes_of(w)
        meta['code8_nibbles'][group] = meta['code8_nibbles'].get(group, 0) + int((codes == 8).sum())
        meta['nibbles'][group] = meta['nibbles'].get(group, 0) + int(codes.size)
        X, _ = acts(I, codes, s, rng)
        T = X.shape[0]
        vers = {0: np.ascontiguousarray(w), 1: canon(w), 2: posctl(w)}
        sv = np.ascontiguousarray(s)
        for slot, wv in vers.items():
            assert lib.zt_set_matrix(slot, p(wv), p(sv), O, I), 'set_matrix'
        for k in kernels:
            name = lib.zt_kernel_name(k).decode()
            out = {}
            for tag, slot in (('A', 0), ('B', 1), ('A2', 0), ('P', 2)):
                y = np.empty(T * O, np.float32)
                n = lib.zt_run(k, slot, p(X), T, p(y))
                if n == -1:
                    break
                assert n == T * O, f'{name} failed ({n})'
                out[tag] = y
            if not out:
                res.setdefault('not_run', {}).setdefault(name, []).append(f'{O}x{I}')
                continue
            ent = res[group].setdefault(name, new_entry())
            record(ent, out['A'], out['B'], out['A2'], out['P'], T, O)


def mode_kernels(lib, args, out):
    sh = Shards()
    rng = np.random.default_rng(20261108)
    kernels = list(range(lib.zt_kernel_count()))
    res = {'real_matrices': {}, 'synthetic_zero_rows': {}}
    meta = out['meta']; meta.update(code8_nibbles={}, nibbles={}, experts=[], shapes={})
    sel = [pe for pe in picks() if pe[0] < args.layers and pe[1] in sorted([q[1] for q in picks() if q[0] == pe[0]])[:args.per_layer]]
    t0 = time.time()
    for i, (L, e) in enumerate(sel):
        ex = sh.expert(L, e)
        meta['experts'].append(f'L{L:02d}_E{e:03d}')
        mats = [ex[1], ex[2], ex[3]]
        for m in (1, 2, 3):
            meta['shapes'][f'w{m}'] = [int(ex[m][0].shape[0]), int(ex[m][0].shape[1] * 2)]
        run_kernels(lib, res, 'real_matrices', mats, rng, kernels, meta)
        if e == min(q[1] for q in sel if q[0] == L):   # first tested expert of each layer: team2-style synthetic zero rows
            syn = []
            for m in (1, 2):
                w, s = ex[m]
                sw = w[:64].copy(); ss = s[:64].copy()
                sw[:32] = (np.where(rng.integers(0, 2, sw[:32].shape) == 1, 8, 0) | (np.where(rng.integers(0, 2, sw[:32].shape) == 1, 8, 0) << 4)).astype(np.uint8)
                syn.append((sw, ss))
            run_kernels(lib, res, 'synthetic_zero_rows', syn, rng, kernels, meta)
        print(f'L{L:02d}_E{e:03d} done {time.time() - t0:.1f}s', flush=True)
    out['results'] = res


def crop_v4(ex):   # V4.1 expert -> V4 shapes: w1/w3 [2048 x 4096], w2 [4096 x 2048] (top-left corner)
    c = {}
    for m in (1, 3):
        w, s = ex[m]; c[m] = (np.ascontiguousarray(w[:2048, :2048]), np.ascontiguousarray(s[:2048, :128]))
    w, s = ex[2]; c[2] = (np.ascontiguousarray(w[:4096, :1024]), np.ascontiguousarray(s[:4096, :64]))
    return c


def version(ex, tag):
    f = {'A': lambda w: w, 'A2': lambda w: w, 'B': canon, 'P': posctl}[tag]
    return {m: (np.ascontiguousarray(f(w)), s) for m, (w, s) in ex.items()}


def fp8_random(rng, O, I):
    w = rng.integers(0, 256, (O, I), dtype=np.uint8)
    w[(w & 0x7F) == 0x7F] = 0x38   # no e4m3 NaN
    s = np.full(((O + 127) // 128, (I + 127) // 128), 120, np.uint8)
    return np.ascontiguousarray(w), s


class HarnessApi:   # zt_api_* glue in the harness DLL (backend compiled from the worktree source)
    toggles = True
    def __init__(s, lib): s.lib = lib
    def setenv(s, kv): assert s.lib.zt_putenv(kv.encode())
    def upload(s, which, k, w, sc): assert s.lib.zt_api_upload(which, k, p(w), p(sc), w.shape[0], w.shape[1] * 2), 'upload'
    def shared(s, which, w, sc): assert s.lib.zt_api_shared(which, p(w), p(sc), w.shape[0], w.shape[1]), 'shared'
    def group(s, n, wts, y, x): return s.lib.zt_api_expert_group(n, p(wts), C.c_float(LIMIT), p(y), p(x))
    def moe(s, n, wts, y, x): return s.lib.zt_api_moe(n, p(wts), C.c_float(LIMIT), p(y), p(x))


class ShippedApi:   # the shipped app DLL's own dsv4_cuda_* exports (static CRT: env only as set before load)
    toggles = False
    def __init__(s, lib):
        P = C.c_void_p; s.lib = lib
        lib.dsv4_cuda_init.argtypes = [P, C.c_int]
        lib.dsv4_cuda_upload_fp4.argtypes = [P, P, P, C.c_int, C.c_int, C.c_int]
        lib.dsv4_cuda_upload_fp8.argtypes = [P, P, P, C.c_int, C.c_int, C.c_int]
        lib.dsv4_cuda_tensor_refill_fp4.argtypes = [P, P, P, C.c_int, C.c_int, C.c_int]
        lib.dsv4_cuda_tensor_free.argtypes = [P]
        lib.dsv4_cuda_expert_group.argtypes = [P, P, P, P, C.c_int, C.c_float, P, P]
        lib.dsv4_cuda_moe.argtypes = [P, P, P, P, C.c_int, P, P, P, C.c_float, P, P]
        d = C.c_int(0); assert lib.dsv4_cuda_init(C.byref(d), 1), 'dsv4_cuda_init'
        s.t = [[C.c_void_p() for _ in range(16)] for _ in range(3)]; s.shape = {}; s.sh = [C.c_void_p() for _ in range(3)]
    def setenv(s, kv): raise RuntimeError('shipped DLL: set env before start')
    def upload(s, which, k, w, sc):
        O, I = w.shape[0], w.shape[1] * 2; t = s.t[which][k]
        if t.value and s.shape.get((which, k)) != (O, I): s.lib.dsv4_cuda_tensor_free(t); t.value = None
        if not t.value: assert s.lib.dsv4_cuda_upload_fp4(C.byref(t), p(w), p(sc), O, I, 0), 'upload'
        else: assert s.lib.dsv4_cuda_tensor_refill_fp4(t, p(w), p(sc), O, I, 1), 'refill'
        s.shape[(which, k)] = (O, I)
    def shared(s, which, w, sc):
        assert s.lib.dsv4_cuda_upload_fp8(C.byref(s.sh[which]), p(w), p(sc), w.shape[0], w.shape[1], 0), 'shared'
    def arr(s, which): return (C.c_void_p * 16)(*[t.value for t in s.t[which]])
    def group(s, n, wts, y, x): return s.lib.dsv4_cuda_expert_group(s.arr(0), s.arr(1), s.arr(2), p(wts), n, C.c_float(LIMIT), p(y), p(x))
    def moe(s, n, wts, y, x): return s.lib.dsv4_cuda_moe(s.arr(0), s.arr(1), s.arr(2), p(wts), n, s.sh[0], s.sh[1], s.sh[2], C.c_float(LIMIT), p(y), p(x))


def mode_api(lib, args, out):
    sh = Shards()
    rng = np.random.default_rng(20261109)
    res = {}; meta = out['meta']; meta['env_argument'] = args.env
    api = ShippedApi(lib) if args.shipped else HarnessApi(lib)
    for kv in args.env:
        api.setenv(kv)
    allp = picks()
    t0 = time.time()
    # 1) dsv4_cuda_expert_group, V4.1 shapes, the 4 picked experts of each layer
    ent = res.setdefault('dsv4_cuda_expert_group(4 experts, H=5120, I=2304)', new_entry())
    for L in range(args.layers):
        exs = [sh.expert(L, e) for (l, e) in allp if l == L]
        X, _ = acts(5120, codes_of(exs[0][1][0]), exs[0][1][1], rng)
        wts = rng.random(4).astype(np.float32) * np.float32(0.5)
        ys = {}
        for tag in ('A', 'B', 'A2', 'P'):
            for k, ex in enumerate(exs):
                v = version(ex, tag)
                for which, m in ((0, 1), (1, 3), (2, 2)):
                    w, s = v[m]
                    api.upload(which, k, w, np.ascontiguousarray(s))
            y = np.empty((16, 5120), np.float32)
            for t in range(16):
                assert api.group(4, wts, y[t], X[t]), 'expert_group'
            ys[tag] = y.reshape(-1)
        record(ent, ys['A'], ys['B'], ys['A2'], ys['P'], 16, 5120)
        print(f'expert_group L{L:02d} {time.time() - t0:.1f}s', flush=True)
    # 2) dsv4_cuda_moe: 6 cropped experts (layer L picks + 2 of layer L+1), fp8 shared experts (random, fixed)
    sg, su, sd = fp8_random(rng, 2048, 4096), fp8_random(rng, 2048, 4096), fp8_random(rng, 4096, 2048)
    for which, (w, s) in enumerate((sg, su, sd)):
        api.shared(which, w, s)
    for fused in (('1', '0') if api.toggles else (os.environ.get('DSV4_CUDA_MOE_FUSED6', '1'),)):
        if api.toggles:
            api.setenv(f'DSV4_CUDA_MOE_FUSED6={fused}')
        name = 'dsv4_cuda_moe fused v4f_moe_kernel (6 x 4096/2048, cropped)' if fused == '1' else 'dsv4_cuda_moe launches path (FUSED6=0, mv_fp4_grouped<8>; 6 x 4096/2048, cropped)'
        ent = res.setdefault(name, new_entry())
        for L in range(args.layers):
            sel = [e for (l, e) in allp if l == L] + [e for (l, e) in allp if l == (L + 1) % 40][:2]
            lays = [L] * 4 + [(L + 1) % 40] * 2
            exs = [crop_v4(sh.expert(l, e)) for l, e in zip(lays, sel)]
            X, _ = acts(4096, codes_of(exs[0][1][0]), exs[0][1][1], rng)
            wts = rng.random(6).astype(np.float32) * np.float32(0.5)
            ys = {}
            for tag in ('A', 'B', 'A2', 'P'):
                for k, ex in enumerate(exs):
                    v = version(ex, tag)
                    for which, m in ((0, 1), (1, 3), (2, 2)):
                        w, s = v[m]
                        api.upload(which, k, w, np.ascontiguousarray(s))
                y = np.empty((16, 4096), np.float32)
                for t in range(16):
                    assert api.moe(6, wts, y[t], X[t]), 'moe'
                ys[tag] = y.reshape(-1)
            record(ent, ys['A'], ys['B'], ys['A2'], ys['P'], 16, 4096)
            print(f'moe fused={fused} L{L:02d} {time.time() - t0:.1f}s', flush=True)
    out['results'] = res


def mode_dg(lib, args, out):
    sh = Shards()
    rng = np.random.default_rng(20261110)
    assert lib.zt_dg_bank(), 'bank'
    res = {}; allp = picks(); t0 = time.time()
    names = {0: 'dg_moe<256> (masked decode)', 1: 'dg_moe_contiguous_fast<256> (decode default)',
             2: 'dg_moe_batch_contiguous grouped (host ids, prefill default)', 3: 'dg_moe_batch_contiguous legacy replicated'}
    for L in range(args.layers):
        lays = [L] * 4 + [(L + 1) % 40] * 4
        es = [e for (l, e) in allp if l == L] + [e for (l, e) in allp if l == (L + 1) % 40]
        if args.slots == 'spaced':     # bank slots 32 apart: no two routed groups within one 64-row masked block
            slots = [32 * k + 5 for k in range(8)]
        elif args.slots == 'adjacent':  # consecutive bank slots: every routed group shares a 64-row masked block with the next
            slots = [100 + k for k in range(8)]
        else:                           # the experts' own ids (collisions moved by +64)
            slots, used = [], set()
            for e in es:
                s_ = e
                while s_ in used: s_ = (s_ + 64) % 256
                used.add(s_); slots.append(s_)
        exs = [crop_v4(sh.expert(l, e)) for l, e in zip(lays, es)]
        X, _ = acts(4096, codes_of(exs[0][1][0]), exs[0][1][1], rng)
        ids = np.array([[slots[(t + j) % 8] for j in range(6)] for t in range(16)], np.int32)
        wts = (rng.random((16, 6)) * 0.5).astype(np.float32)
        ys = {}
        for tag in ('A', 'B', 'A2', 'P'):
            for k, ex in enumerate(exs):
                v = version(ex, tag)
                (gw, gs), (uw, us), (dw, ds) = v[1], v[3], v[2]
                assert lib.zt_dg_upload(slots[k], p(gw), p(np.ascontiguousarray(gs)), p(uw), p(np.ascontiguousarray(us)), p(dw), p(np.ascontiguousarray(ds))), 'upload'
            r = {}
            for kind in (0, 1):
                o = np.empty((16, 4096), np.float32); m1 = np.empty((16, 6, 4096), np.uint16); m2 = np.empty((16, 6, 4096), np.uint16)
                for t in range(16):
                    x = np.ascontiguousarray(X[t]); i_ = np.ascontiguousarray(ids[t]); w_ = np.ascontiguousarray(wts[t])
                    assert lib.zt_dg_moe(kind, p(x), 1, p(i_), p(w_), C.c_float(LIMIT), p(o[t]), p(m1[t]), p(m2[t])), f'dg kind {kind}'
                r[kind] = (o.reshape(-1), m1.reshape(-1), m2.reshape(-1))
            for kind in (2, 3):
                o = np.empty((16, 4096), np.float32)
                assert lib.zt_dg_moe(kind, p(X), 16, p(ids), p(wts), C.c_float(LIMIT), p(o), None, None), f'dg kind {kind}'
                r[kind] = (o.reshape(-1),)
            ys[tag] = r
        per_layer = out['meta'].setdefault('per_layer_kind0_FC1', [])   # masked decode FC1 rows: canon / rerun differing values per layer
        a, b, a2 = (ys[t][0][1] for t in ('A', 'B', 'A2'))
        per_layer.append([L, sorted(slots), int((a != b).sum()), int((a != a2).sum())])
        for kind in (0, 1, 2, 3):
            ent = res.setdefault(names[kind] + ' : routed output (fp32 of bf16 sum)', new_entry())
            record(ent, ys['A'][kind][0], ys['B'][kind][0], ys['A2'][kind][0], ys['P'][kind][0], 16, 4096)
            if kind < 2:
                for j, lab in ((1, 'FC1 GEMM output (bf16, 6 routes x [gate|up] 4096)'), (2, 'FC2 GEMM output (bf16, 6 routes x 4096)')):
                    ent = res.setdefault(names[kind] + ' : ' + lab, new_entry())
                    record(ent, *(ys[t][kind][j] for t in ('A', 'B', 'A2', 'P')), 16)
        print(f'dg L{L:02d} slots {slots} {time.time() - t0:.1f}s', flush=True)
    out['results'] = res
    out['meta']['slots_mode'] = args.slots
    out['meta']['dg_layout'] = 'per layer: 8 cropped experts (4 picks of L + 4 of L+1) in bank slots; 16 x rows (decode kinds: one call per row, ids = 6 of the 8 slots rotated); batch kinds: T=16 tokens'


def e4m3_random(rng, n):
    b = rng.integers(0, 256, n, dtype=np.uint8)
    b[(b & 0x7F) == 0x7F] = 0x38
    return b


def mode_mma(lib, args, out):
    rng = np.random.default_rng(7)
    T = 20000
    n = T * 32
    res = {}
    LUT8 = np.array([0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c, 0x80, 0xb0, 0xb8, 0xbc, 0xc0, 0xc4, 0xc8, 0xcc], np.uint8)   # tc_expand
    def run(op, A, B, sa, sb, Cf):
        D = np.empty(n * 4, np.float32)
        assert lib.zt_mma(op, T, p(A), p(B), p(sa), p(sb), p(Cf), p(D))
        return D
    def bcodes(op, codes):   # B operand bytes from FP4 codes
        return ((codes.astype(np.uint8) << 2) if op == 0 else LUT8[codes]).astype(np.uint8)
    tile_scale = lambda: np.repeat(rng.integers(118, 136, T, dtype=np.uint8), 32)
    Cs = {'C=+0': np.zeros(n * 4, np.float32), 'C=-0': np.full(n * 4, -0.0, np.float32), 'C=random': rng.standard_normal(n * 4).astype(np.float32)}
    for op, opname in ((0, 'mxf8f6f4 e4m3 x e2m1 (DeepGEMM fp8_fp4_mixed_mma_block_scaled)'), (1, 'mxf8f6f4 e4m3 x e4m3 (fp8_mma_block_scaled; B = tc_expand bytes)')):
        o = {}
        A = e4m3_random(rng, n * 16)
        sa, sb = tile_scale(), tile_scale()
        codes = rng.integers(0, 16, n * 8)
        B1, B2 = bcodes(op, codes), bcodes(op, np.where(codes == 8, 0, codes))
        assert (B1 != B2).any()
        for cn, Cf in Cs.items():
            D1, D2 = run(op, A, B1, sa, sb, Cf), run(op, A, B2, sa, sb, Cf)
            u1, u2 = D1.view(np.uint32), D2.view(np.uint32)
            o[f'random A, random B codes (orig vs canon), {cn}: differing D / bits'] = [int((u1 != u2).sum()), popcount(u1 ^ u2)]
            D3 = run(op, A, B1, sa, sb, Cf)
            o[f'random A, random B codes, {cn}: rerun differing D'] = int((u1 != D3.view(np.uint32)).sum())
        # positive control: one nonzero B code per tile-lane sign flipped
        c3 = codes.copy().reshape(n, 8); nz = (c3 & 7) != 0; j = nz.argmax(1); r = np.arange(n); c3[r[nz.any(1)], j[nz.any(1)]] ^= 8
        D1, D3 = run(op, A, B1, sa, sb, Cs['C=+0']), run(op, A, bcodes(op, c3.reshape(-1)), sa, sb, Cs['C=+0'])
        o['positive control (one B code sign-flipped per lane), C=+0: differing D'] = int((D1.view(np.uint32) != D3.view(np.uint32)).sum())
        # B all zero codes with random signs vs all +0
        zc = np.where(rng.integers(0, 2, n * 8) == 1, 8, 0)
        for cn in ('C=+0', 'C=-0'):
            D1, D2 = run(op, A, bcodes(op, zc), sa, sb, Cs[cn]), run(op, A, bcodes(op, np.zeros_like(zc)), sa, sb, Cs[cn])
            u1, u2 = D1.view(np.uint32), D2.view(np.uint32)
            o[f'B all zero (codes 0/8 random) vs all code 0, {cn}: differing D'] = int((u1 != u2).sum())
            o[f'B all zero (codes 0/8 random), {cn}: D==-0 / D==+0'] = [int((u1 == 0x80000000).sum()), int((u1 == 0).sum())]
            o[f'B all code 0, {cn}: D==-0 / D==+0'] = [int((u2 == 0x80000000).sum()), int((u2 == 0).sum())]
        # every product -0 in the original, +0 after canonicalisation: A > 0 finite nonzero, B all code 8
        Apos = rng.integers(0x08, 0x7F, n * 16, dtype=np.uint8)   # positive normal e4m3
        for cn in ('C=+0', 'C=-0'):
            D1 = run(op, Apos, bcodes(op, np.full(n * 8, 8)), sa, sb, Cs[cn]); D2 = run(op, Apos, bcodes(op, np.zeros(n * 8, np.int64)), sa, sb, Cs[cn])
            u1, u2 = D1.view(np.uint32), D2.view(np.uint32)
            o[f'A>0, B all code 8 (every product -0) vs B all code 0 (every product +0), {cn}: differing D'] = int((u1 != u2).sum())
            o[f'A>0, B all code 8, {cn}: D==-0 / D==+0'] = [int((u1 == 0x80000000).sum()), int((u1 == 0).sum())]
            o[f'A>0, B all code 0, {cn}: D==-0 / D==+0'] = [int((u2 == 0x80000000).sum()), int((u2 == 0).sum())]
        # A all -0 (e4m3 0x80), B nonzero positive codes: every product -0
        Bp = bcodes(op, rng.integers(1, 8, n * 8))
        for cn in ('C=+0', 'C=-0'):
            u1 = run(op, np.full(n * 16, 0x80, np.uint8), Bp, sa, sb, Cs[cn]).view(np.uint32)
            o[f'A all -0, B > 0 (every product -0), {cn}: D==-0 / D==+0'] = [int((u1 == 0x80000000).sum()), int((u1 == 0).sum())]
        res[opname] = o
        print(opname, 'done', flush=True)
    out['results'] = res
    out['meta']['mma'] = f'{T} m16n8k32 tiles (1 warp each) per case; per-tile uniform ue8m0 scales in [118,135]'


def main():
    mode, dll, outp = sys.argv[1:4]
    rest = sys.argv[4:]
    class A: pass
    args = A(); args.per_layer = 4; args.layers = 40; args.env = []; args.shipped = mode == 'shipped'; args.slots = 'expert'
    i = 0
    while i < len(rest):
        if rest[i] == '--per-layer': args.per_layer = int(rest[i + 1]); i += 2
        elif rest[i] == '--layers': args.layers = int(rest[i + 1]); i += 2
        elif rest[i] == '--slots': args.slots = rest[i + 1]; i += 2
        elif rest[i] == '--env': args.env.append(rest[i + 1]); i += 2
        else: raise SystemExit('bad arg ' + rest[i])
    if args.shipped:
        os.add_dll_directory(CUDA_BIN); lib = C.CDLL(dll); flav = 'shipped app DLL'
    else:
        lib = load(dll); assert lib.zt_init(0), 'dsv4_cuda_init'
        flav = 'deepgemm sm_120a' if lib.zt_flavour() else 'generic sm_120'
    out = {'meta': {'mode': mode, 'dll': dll, 'flavour': flav, 'process_env': {k: v for k, v in os.environ.items() if k.startswith('DSV4_')},
                    'act_sets_T16': AN, 'started': time.strftime('%Y-%m-%dT%H:%M:%S')}}
    t0 = time.time()
    {'kernels': mode_kernels, 'api': mode_api, 'shipped': mode_api, 'dg': mode_dg, 'mma': mode_mma}[mode](lib, args, out)
    out['seconds'] = round(time.time() - t0, 1)
    json.dump(out, open(outp, 'w'), indent=1)
    print('wrote', outp, out['seconds'], 's')


if __name__ == '__main__':
    main()
