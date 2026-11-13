"""zero_bittest: canonical-zero GPU bit test (WNR3 draft 5.6 test 1, CUDA part that needs no KAIRA build).

  python -I tests/gpu/run_zero_bittest.py --root <kit> [--data <dir>] [--out <dir>] [--quick] [--max-experts N]

Original MXFP4 records vs canonicalised records (every E2M1 code 8 = -0 replaced by 0 = +0) through every CUDA kernel
variant of team 2's harness (zero-safety/scripts/zs_gpu.py + zs_gpu_kernels.cu = colibri backend_cuda_dsv4.cu FP4
paths: mv_fp4_grouped<1,2,4,8>, mv_fp4_expert_major<4> (4 tokens), mv<4>, mv_rows<4,4>, mv_rows<4,16>, mm_batch<4>
(4 tokens), v4f_dot_fp4 warp/row), same activations for both (9 adversarial sets: gauss, mixed +-0, all -0, all +0,
one nonzero rest -0, denormals, inf, NaN, 1e30; 4-token batch x, -x, x with odd -0, x*1e-30); output bits compared
(value / sign-of-zero-only / NaN-payload mismatches) and every run repeated once (determinism).
Plus (team 2 zero-refuter/gpu_probe.py, zs_mma.py): FP16 tensor-core GEMM (mma.sync m16n8k16, full K) on real
matrices, and the f16/bf16 mma tile micro test incl. all-zero A tiles. On sm_120 also an EXPERIMENTAL block-scaled FP4
tensor-core GEMM (mma kind::mxf4, needs -arch=sm_120a), reported but not part of the verdict.
Controls: positive control (one nonzero weight changed -> every kernel must differ); negative control kernel
negctl_seeded (accumulator seeded with the first product) must show sign-only mismatches on the synthetic zero rows;
A/A determinism (each run repeated) must give 0 differences; unwritten-output sentinel must never survive.
PASS = 0 differing output bits for every real kernel on real and synthetic data, both controls fire, reruns identical.
The arch is detected from the device (-arch=compute_<cc> by default as in team 2's run; --arch-mode sm for a cubin).
Full run = all 160 sample experts (team 2 ran the same on the RTX 3070: ref/zero_3070/gpu_all160.json); --quick = the
8 GPU-bench experts. The KAIRA-build paths (DeepGEMM sm120 FP4, cuBLASLt MXFP8, Vulkan, per-step logits memcmp) are
in CHECKLIST-kaira-gpu-paths.md.
"""
# read by setup/kitenv.py with ast (runner discovery for run_all.ps1); never executed there
KIT_META = {'name': 'zero_bittest', 'order': 60, 'needs': ['gpu', 'data'], 'quick_minutes': 1, 'full_minutes': 10,
            'description': 'canonical-zero GPU bit test, original vs canonicalised records through colibri FP4 kernels'}
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpukit_common as K  # noqa: E402

TEST = 'zero_bittest'
KN = ["mv_fp4_grouped<1>", "mv_fp4_grouped<2>", "mv_fp4_grouped<4>", "mv_fp4_grouped<8>", "mv_fp4_expert_major<4>_T4",
      "mv<4>", "mv_rows<4,4>", "mv_rows<4,16>", "mm_batch<4>_T4", "v4f_dot_fp4(warp/row)", "NEGCTL_seeded"]
REAL_KN = KN[:-1]
TPL = ["mv_fp4_grouped<1>", "mv_fp4_grouped<2>", "mv_fp4_grouped<4>", "mv_fp4_grouped<8>",
       "mv_fp4_expert_major<4>", "mv<4>", "mv_rows<4,4>", "mv_rows<4,16>", "mm_batch<4>"]
AN = ["gauss", "gauss_25pct+0_25pct-0", "all-0", "all+0", "one_nonzero_rest-0", "denormals_10pct", "inf_at_100",
      "nan_at_200", "huge_1e30"]
SENTINEL = np.uint32(0x7FC0DEAD)


def canon(a):
    lo = a & 15; hi = a >> 4
    return (np.where(lo == 8, 0, lo) | (np.where(hi == 8, 0, hi) << 4)).astype(np.uint8)


def parse_args():
    p = K.base_parser(TEST, __doc__)
    p.add_argument('--max-experts', type=int, default=None, help='use only the first N experts of the run order')
    p.add_argument('--arch-mode', choices=('sm', 'compute'), default='compute',
                   help='compute = PTX for compute_<cc> + driver JIT (team 2 default); sm = cubin for sm_<cc>')
    p.add_argument('--no-mxf4', action='store_true', help='skip the experimental sm_120 block-scaled FP4 MMA part')
    return p.parse_args()


def main():
    args = parse_args()
    K.resolve_paths(args, TEST)
    log = K.Log(os.path.join(args.out, TEST + '.log'))
    return K.run_main(TEST, lambda: body(args, log), args, args.out)


def body(args, log):
    import cupy as cp
    cp.cuda.Device(args.device).use()
    env = K.env_info(args.device)
    log('env', json.dumps(env))
    arch, cc = K.device_arch(args.device, args.arch_mode)
    order = list(K.PICK8) if args.quick else list(range(160))
    if args.max_experts:
        order = order[:args.max_experts]
    man, paths, _ = K.resolve_experts(args, order, log)
    opts = (arch, '-std=c++17', '--ftz=false', '--fmad=true')      # nvcc defaults + colibri Makefile -ftz=false
    with open(os.path.join(HERE, 'src_zero', 'zs_gpu_kernels.cu')) as f:
        src = f.read()
    mod = cp.RawModule(code=src, options=opts, name_expressions=TPL + ['v4f_rows'])   # extern "C" listed too
    mod.compile()
    with open(os.path.join(HERE, 'src_zero', 'zs_extra_kernels.cu')) as f:
        xsrc = f.read()
    xmod = cp.RawModule(code=xsrc, options=opts)
    xmod.compile()
    KF = {n: mod.get_function(n) for n in TPL}
    KF['v4f_rows'] = mod.get_function('v4f_rows')
    for n in ('tc_gemm', 'mma_f16', 'mma_bf16', 'negctl_seeded'):
        KF[n] = xmod.get_function(n)
    e8g = mod.get_global('e8_table')
    v = np.arange(256)
    e8 = np.ldexp(np.float32(1), np.where(v == 0, -127, v - 127)).astype(np.float32)   # dsv4_cuda_init (:1605)
    cp.ndarray((256,), cp.float32, e8g).set(e8)
    meta = {'device': env.get('gpu', {}).get('name'), 'cc': cc, 'nvrtc_options': list(opts), 'nvrtc': env.get('nvrtc'),
            'files': []}
    log('compiled', opts)
    rng = np.random.default_rng(20261107)

    def act(I, k):
        x = rng.standard_normal(I).astype(np.float32)
        if k == 1:
            r = rng.integers(0, 4, I); x[r == 0] = 0.0; x[r == 1] = np.float32(-0.0)
        elif k == 2:
            x[:] = np.float32(-0.0)
        elif k == 3:
            x[:] = 0.0
        elif k == 4:
            x[:] = np.float32(-0.0); x[7] = 1.0
        elif k == 5:
            m = rng.integers(0, 10, I) == 0
            x[m] = np.where(rng.integers(0, 2, m.sum()) == 1, np.float32(1e-41), np.float32(-1e-41))
        elif k == 6:
            x[100] = np.inf
        elif k == 7:
            x[200] = np.nan
        elif k == 8:
            x *= np.float32(1e30)
        return x

    def batch4(x):
        t = x.copy(); t[1::2] = np.float32(-0.0)
        return np.ascontiguousarray(np.stack([x, -x, t, x * np.float32(1e-30)]).astype(np.float32))

    def new_stats():
        return {k: dict(compared=0, mismatch_value=0, mismatch_signzero_only=0, mismatch_nan_payload=0, zero_outputs=0,
                        neg_zero_outputs=0, nan_outputs=0, rerun_nondeterministic=0, unwritten=0) for k in KN}

    def compare(st, a, b, a2):
        ua, ub = a.view(np.uint32), b.view(np.uint32); d = ua != ub
        both0 = ((ua | ub) & 0x7fffffff) == 0; bothnan = np.isnan(a) & np.isnan(b)
        st["compared"] += a.size; st["mismatch_signzero_only"] += int((d & both0).sum())
        st["mismatch_nan_payload"] += int((d & bothnan & ~both0).sum()); st["mismatch_value"] += int((d & ~both0 & ~bothnan).sum())
        st["zero_outputs"] += int(((ua & 0x7fffffff) == 0).sum()); st["neg_zero_outputs"] += int((ua == 0x80000000).sum())
        st["nan_outputs"] += int(np.isnan(a).sum()); st["rerun_nondeterministic"] += int((a2.view(np.uint32) != ua).sum())
        st["unwritten"] += int((ua == SENTINEL).sum() + (ub == SENTINEL).sum())

    def outbuf(n):
        y = cp.empty(n, cp.uint32); y.fill(SENTINEL)
        return y

    def run_all(w, s, I, O, x1, x4):
        """kernel -> output (np.float32) for one weight version (team 2 zs_gpu.py run_all, CuPy launches)."""
        dw, ds, dx1, dx4 = cp.asarray(w), cp.asarray(s), cp.asarray(x1), cp.asarray(x4)
        out = {}
        for R, k in ((1, 0), (2, 1), (4, 2), (8, 3)):
            dy = outbuf(O)
            dd = cp.asarray(np.array([dw.data.ptr, ds.data.ptr, dx1.data.ptr, dy.data.ptr], np.uint64))
            KF[f"mv_fp4_grouped<{R}>"](((O + R - 1) // R, 1), (256,), (dd, np.int32(1), np.int32(O), np.int32(I)))
            out[KN[k]] = dy.get().view(np.float32)
        dex = cp.asarray(np.array([0], np.int32)); doff = cp.asarray(np.array([0, 4], np.int32))
        drt = cp.asarray(np.arange(4, dtype=np.int32))
        dy = outbuf(4 * O)
        KF["mv_fp4_expert_major<4>"](((O + 7) // 8, 1), (256,),
                                     (dw, ds, np.uint64(w.nbytes), np.uint64(s.nbytes), dex, doff, drt, dx4, np.int64(I),
                                      np.int32(1), dy, dy, np.int32(O), np.int64(O), np.int32(O), np.int32(I)))
        out[KN[4]] = dy.get().view(np.float32)
        dy = outbuf(O)
        KF["mv<4>"]((O, 1), (256,), (dw, ds, dx1, dy, np.int32(O), np.int32(I), np.int32(1)))
        out[KN[5]] = dy.get().view(np.float32)
        for R, k in ((4, 6), (16, 7)):
            dy = outbuf(O)
            KF[f"mv_rows<4,{R}>"](((O + R - 1) // R, 1), (256,), (dw, ds, dx1, dy, np.int32(O), np.int32(I), np.int32(1)))
            out[KN[k]] = dy.get().view(np.float32)
        dy = outbuf(4 * O)
        KF["mm_batch<4>"]((O, 1), (256,), (dw, ds, dx4, dy, np.int32(O), np.int32(I), np.int32(4)))
        out[KN[8]] = dy.get().view(np.float32)
        dy = outbuf(O)
        KF["v4f_rows"](((O + 7) // 8, 1), (256,), (dw, ds, dx1, dy, np.int32(I), np.int32(O)))
        out[KN[9]] = dy.get().view(np.float32)
        dy = outbuf(O)
        KF["negctl_seeded"](((O + 127) // 128,), (128,), (dw, ds, dx1, dy, np.int32(O), np.int32(I)))
        out[KN[10]] = dy.get().view(np.float32)
        cp.cuda.runtime.deviceSynchronize()
        return out

    res = {"real_matrices": new_stats(), "synthetic_zero_rows": new_stats(), "positive_control": new_stats()}
    t0 = time.time()

    def test(group, wA, wB, s, I, O, acts):
        for a in acts:
            x1 = np.ascontiguousarray(act(I, a)); x4 = batch4(x1)
            oa = run_all(wA, s, I, O, x1, x4); ob = run_all(wB, s, I, O, x1, x4); oa2 = run_all(wA, s, I, O, x1, x4)
            for k in KN:
                compare(res[group][k], oa[k], ob[k], oa2[k])

    # ---------------------------------------------------------------- tensor-core FP16 GEMM on real matrices
    F16 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0.0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float16).view(np.uint16)

    def tc_pack(codes, nblk, nk):
        h = F16[codes[:nblk * 16, :nk * 16]].reshape(nblk, 16, nk, 16).transpose(0, 2, 1, 3)
        lane = np.arange(32); g = lane >> 2; t = lane & 3
        rows = [g, g, g + 8, g + 8, g, g, g + 8, g + 8]
        cols = [2 * t, 2 * t + 1, 2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9, 2 * t + 8, 2 * t + 9]
        halves = np.stack([h[:, :, rows[j], cols[j]] for j in range(8)], axis=-1).astype(np.uint32)
        regs = halves[..., 0::2] | (halves[..., 1::2] << 16)
        return np.ascontiguousarray(regs.astype(np.uint32))

    def tc_bpack(xh, nk):
        lane = np.arange(32); g = lane >> 2; t = lane & 3
        xb = xh.reshape(nk, 16, 8)
        ks = [2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9]
        halves = np.stack([xb[:, ks[j], g] for j in range(4)], axis=-1).astype(np.uint32)
        regs = halves[..., 0::2] | (halves[..., 1::2] << 16)
        return np.ascontiguousarray(regs.astype(np.uint32))

    def tc_run(A, B, nblk, nk, cinit=0):
        dD = cp.zeros(nblk * 128, cp.float32)
        KF['tc_gemm'](((nblk + 7) // 8,), (256,), (cp.asarray(A), cp.asarray(B), dD, np.int32(nblk), np.int32(nk), np.uint32(cinit)))
        return dD.get()

    tc = {'matrices': 0, 'outputs': 0, 'bitdiff': 0, 'positive_control_bitdiff': None, 'layout_max_rel_err': 0.0}
    trng = np.random.default_rng(7)

    def tc_matrix(wb2d, first):
        O, I = wb2d.shape[0], wb2d.shape[1] * 2
        codes = np.empty((O, I), np.uint8); codes[:, 0::2] = wb2d & 15; codes[:, 1::2] = wb2d >> 4
        cz = codes.copy(); cz[cz == 8] = 0
        nblk, nk = min(O // 16, 32), I // 16
        xh = trng.standard_normal((nk * 16, 8)).astype(np.float16).view(np.uint16)
        B = tc_bpack(xh, nk)
        d0 = tc_run(tc_pack(codes, nblk, nk), B, nblk, nk)
        d1 = tc_run(tc_pack(cz, nblk, nk), B, nblk, nk)
        tc['matrices'] += 1; tc['outputs'] += d0.size
        tc['bitdiff'] += int((d0.view(np.uint32) != d1.view(np.uint32)).sum())
        wv = F16[codes].view(np.float16).astype(np.float32)
        ref = wv[:nblk * 16, :nk * 16] @ xh.view(np.float16).astype(np.float32)
        lane = np.arange(32); gg = lane >> 2; tt = lane & 3
        D = d0.reshape(nblk, 32, 4); got = np.empty((nblk, 16, 8), np.float32)
        got[:, gg, 2 * tt] = D[:, :, 0]; got[:, gg, 2 * tt + 1] = D[:, :, 1]
        got[:, gg + 8, 2 * tt] = D[:, :, 2]; got[:, gg + 8, 2 * tt + 1] = D[:, :, 3]
        tc['layout_max_rel_err'] = max(tc['layout_max_rel_err'], float(np.max(np.abs(got.reshape(-1, 8) - ref)) / np.max(np.abs(ref))))
        if first:   # positive control: one nonzero code changed
            c2 = cz.copy(); j = int(np.nonzero(c2[0, :16] != 0)[0][0]); c2[0, j] ^= 1 if (c2[0, j] & 7) != 7 else 2
            d2 = tc_run(tc_pack(c2, nblk, nk), B, nblk, nk)
            tc['positive_control_bitdiff'] = int((d1.view(np.uint32) != d2.view(np.uint32)).sum())

    # ---------------------------------------------------------------- EXPERIMENTAL sm_120 block-scaled FP4 MMA
    mx = {'status': 'skipped (device cc %s is not 12.x)' % cc}
    mxfn = None
    if cc.startswith('12') and not args.no_mxf4:
        try:
            with open(os.path.join(HERE, 'src_zero', 'zs_sm120_mxf4.cu')) as f:
                msrc = f.read()
            mmod = cp.RawModule(code=msrc, options=('-arch=sm_%sa' % cc, '-std=c++17'))
            mmod.compile()
            mxfn = mmod.get_function('mxf4_gemm')
            mx = {'status': 'EXPERIMENTAL, ran', 'arch': 'sm_%sa' % cc, 'matrices': 0, 'outputs': 0, 'bitdiff': 0,
                  'positive_control_bitdiff': None, 'nan_outputs': 0}
        except Exception as ex:  # noqa
            mx = {'status': 'EXPERIMENTAL, compile/load failed: %s' % str(ex)[:300]}

    def mx_run(A, B, SA, SB, ntile, nk, cinit=0):
        dD = cp.zeros(ntile * 128, cp.float32)
        mxfn(((ntile + 7) // 8,), (256,), (cp.asarray(A), cp.asarray(B), cp.asarray(SA), cp.asarray(SB), dD,
                                          np.int32(ntile), np.int32(nk), np.uint32(cinit)))
        return dD.get()

    def mx_matrix(wb2d, sc2d, first):
        O, Ib = wb2d.shape                # Ib = I/2 bytes per row
        ntile, nk = min(O // 16, 32), Ib // 32
        sub = wb2d[:ntile * 16, :nk * 32].reshape(ntile, 16, nk, 32).transpose(0, 2, 1, 3)   # tile, k, row, 32 bytes
        Aorig = np.ascontiguousarray(sub).reshape(ntile, nk, 512).view(np.uint32).reshape(-1)
        Acan = np.ascontiguousarray(canon(np.ascontiguousarray(sub))).reshape(ntile, nk, 512).view(np.uint32).reshape(-1)
        scs = sc2d[:ntile * 16, :2 * nk].reshape(ntile, 16, nk, 2).transpose(0, 2, 1, 3)       # tile, k, row, 2
        sa = np.ascontiguousarray(np.concatenate([scs, scs], axis=-1)).view(np.uint32)[..., 0]  # tile, k, row
        SA = np.ascontiguousarray(np.repeat(sa, 2, axis=2)).reshape(-1)                         # 32 lanes
        B = trng.integers(0, 256, (nk, 32, 8), dtype=np.uint8).view(np.uint32).reshape(-1)
        SB = np.full(nk * 32, 0x7F7F7F7F, np.uint32)
        for cinit in (0, 0x80000000):
            d0 = mx_run(Aorig, B, SA, SB, ntile, nk, cinit); d1 = mx_run(Acan, B, SA, SB, ntile, nk, cinit)
            mx['matrices'] += 1; mx['outputs'] += d0.size
            mx['bitdiff'] += int((d0.view(np.uint32) != d1.view(np.uint32)).sum())
            mx['nan_outputs'] += int(np.isnan(d0).sum())
        if first:
            A2 = Acan.copy().view(np.uint8)
            j = int(np.nonzero((A2 & 15) != 0)[0][0]); A2[j] = (A2[j] & 0xF0) | ((A2[j] & 15) ^ 1)
            d2 = mx_run(A2.view(np.uint32), B, SA, SB, ntile, nk)
            d1 = mx_run(Acan, B, SA, SB, ntile, nk)
            mx['positive_control_bitdiff'] = int((d1.view(np.uint32) != d2.view(np.uint32)).sum())

    # ---------------------------------------------------------------- main loop over experts
    for fi, ei in enumerate(order):
        e = man['experts'][ei]
        t = K.load_record(paths[e['file']], e)
        meta['files'].append(e['file'])
        mats = []
        for wn in ('w1', 'w2', 'w3'):
            w, shp, _ = t[wn + '.weight']; s, _, _ = t[wn + '.scale']
            O, I = shp[0], shp[1] * 2
            mats.append((w.copy(), s.copy(), I, O))
        for (w, s, I, O) in mats:
            test("real_matrices", w, canon(w), s, I, O, range(9))
        for (w, s, I, O) in mats[:2]:
            rb = I // 2; sa = w[:64 * rb].copy(); ss = s[:64 * (I // 32)].copy()
            sa[:32 * rb] = (np.where(rng.integers(0, 2, 32 * rb) == 1, 8, 0) |
                            (np.where(rng.integers(0, 2, 32 * rb) == 1, 8, 0) << 4)).astype(np.uint8)
            test("synthetic_zero_rows", sa, canon(sa), ss, I, 64, range(9))
        if fi == 0:
            w, s, I, O = mats[0]; c = canon(w); c2 = c.copy()
            j = int(np.nonzero((c2[:I // 2] & 15) == 1)[0][0]); c2[j] = (c2[j] & 0xF0) | 9
            test("positive_control", c, c2, s, I, O, [0])
        for mi, (w, s, I, O) in enumerate(mats[:2]):
            tc_matrix(w.reshape(O, I // 2), fi == 0 and mi == 0)
            if mxfn is not None:
                mx_matrix(w.reshape(O, I // 2), s.reshape(O, I // 32), fi == 0 and mi == 0)
        log('expert', fi + 1, '/', len(order), e['file'], '%.1f s' % (time.time() - t0))

    # ---------------------------------------------------------------- mma tile micro test (zs_mma.py)
    mrng = np.random.default_rng(7)
    T = 20000
    E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0.0, -.5, -1, -1.5, -2, -3, -4, -6])

    def to16(vals, kind):
        if kind == "mma_f16":
            return vals.astype(np.float16).view(np.uint16)
        f = vals.astype(np.float32).view(np.uint32)
        return ((f + 0x7fff + ((f >> 16) & 1)) >> 16).astype(np.uint16)

    def mma(kind, A16, B16, Cf):
        dD = cp.empty(Cf.shape, cp.float32)
        KF[kind]((Cf.shape[0],), (32,), (cp.asarray(A16), cp.asarray(B16), cp.asarray(Cf), dD))
        return dD.get()

    micro = {}
    for kind in ("mma_f16", "mma_bf16"):
        codes = mrng.integers(0, 16, (T, 256))
        Aorig = E2M1[codes]; Acan = E2M1[np.where(codes == 8, 0, codes)]
        Bv = mrng.standard_normal((T, 128)) * mrng.choice([1e-3, 1, 30], (T, 1))
        a1 = to16(Aorig, kind).view(np.uint32).reshape(T, 128).copy(); a2 = to16(Acan, kind).view(np.uint32).reshape(T, 128).copy()
        b = to16(Bv, kind).view(np.uint32).reshape(T, 64).copy()
        o = {}
        for cname, Cf in (("C=+0", np.zeros((T, 128), np.float32)), ("C=-0", np.full((T, 128), -0.0, np.float32)),
                          ("C=random", mrng.standard_normal((T, 128)).astype(np.float32))):
            D1 = mma(kind, a1, b, Cf); D2 = mma(kind, a2, b, Cf)
            o[f"random A ({T} tiles), {cname}: differing D"] = int((D1.view(np.uint32) != D2.view(np.uint32)).sum())
        zs = np.where(mrng.integers(0, 2, (T, 256)) == 1, -0.0, 0.0); zp = np.zeros((T, 256))
        az1 = to16(zs, kind).view(np.uint32).reshape(T, 128).copy(); az2 = to16(zp, kind).view(np.uint32).reshape(T, 128).copy()
        for cname, Cf in (("C=+0", np.zeros((T, 128), np.float32)), ("C=-0", np.full((T, 128), -0.0, np.float32))):
            D1 = mma(kind, az1, b, Cf); D2 = mma(kind, az2, b, Cf)
            u1, u2 = D1.view(np.uint32), D2.view(np.uint32)
            o[f"all-zero A (random signs vs +0), {cname}: differing D"] = int((u1 != u2).sum())
            o[f"all-zero A (random signs), {cname}: D==-0 count / D==+0 count"] = [int((u1 == 0x80000000).sum()), int((u1 == 0).sum())]
            o[f"all-zero A (+0), {cname}: D==-0 count / D==+0 count"] = [int((u2 == 0x80000000).sum()), int((u2 == 0).sum())]
        micro[kind] = o
    log('mma micro', json.dumps(micro))

    # ---------------------------------------------------------------- verdict
    for g in res:
        res[g] = {k: v for k, v in res[g].items() if v["compared"]}
    real_ok = all(res[g][k][m] == 0 for g in ('real_matrices', 'synthetic_zero_rows') for k in REAL_KN
                  for m in ('mismatch_value', 'mismatch_signzero_only', 'mismatch_nan_payload'))
    determ = all(v['rerun_nondeterministic'] == 0 for g in res for v in res[g].values())
    written = all(v['unwritten'] == 0 for g in res for v in res[g].values())
    pos = all(res['positive_control'][k]['mismatch_value'] > 0 for k in KN) and (tc['positive_control_bitdiff'] or 0) > 0
    neg = res['synthetic_zero_rows']['NEGCTL_seeded']['mismatch_signzero_only'] > 0
    tc_ok = tc['bitdiff'] == 0 and tc['layout_max_rel_err'] < 1e-2
    micro_ok = all(v == 0 for o in micro.values() for k, v in o.items() if k.endswith('differing D'))
    checks = {'real_kernels_0_differing_bits': real_ok, 'reruns_deterministic': determ, 'all_outputs_written': written,
              'positive_controls_fire': pos, 'negative_control_fires': neg, 'tc_fp16_gemm_0_diff': tc_ok,
              'mma_micro_0_diff': micro_ok}
    worst = {k: sum(res[g][k]['mismatch_value'] + res[g][k]['mismatch_signzero_only'] + res[g][k]['mismatch_nan_payload']
                    for g in ('real_matrices', 'synthetic_zero_rows')) for k in REAL_KN}
    out = {'meta': meta, 'act_sets': AN, 'batch4_rows': ["x", "-x", "x with odd entries -0", "x*1e-30"], 'results': res,
           'tc_fp16_gemm': tc, 'mma_micro': micro, 'sm120_mxf4_EXPERIMENTAL': mx, 'seconds': round(time.time() - t0, 1)}
    with open(os.path.join(args.out, 'zero_bittest.json'), 'w') as f:
        json.dump(out, f, indent=1)
    ref_cmp = compare_ref(res)
    summary = {'env': env, 'checks': checks, 'experts': len(order), 'differing_bits_per_real_kernel': worst,
               'negctl_synthetic_signzero_mismatches': res['synthetic_zero_rows']['NEGCTL_seeded']['mismatch_signzero_only'],
               'tc_fp16_gemm': tc, 'sm120_mxf4_EXPERIMENTAL': mx, 'compare_3070_team2': ref_cmp,
               'files': ['zero_bittest.json', TEST + '.log'],
               'not_covered_here': 'DeepGEMM sm120 FP4, cuBLASLt MXFP8 (tc_fp4_matvec), Vulkan, end-to-end per-step '
                                   'logits memcmp: see tests/gpu/CHECKLIST-kaira-gpu-paths.md'}
    ok = all(checks.values())
    short = '%d experts, %d real kernels: %s differing bits; pos/neg controls %s/%s; tc f16 diff %d; mxf4 %s (%s)' % (
        len(order), len(REAL_KN), sum(worst.values()), pos, neg, tc['bitdiff'],
        mx.get('bitdiff', mx['status'][:20]), meta['device'])
    return K.finish(TEST, args.out, 'PASS' if ok else 'FAIL', short, summary, log)


def compare_ref(res):
    """Outputs compared per kernel here vs team 2's RTX 3070 run over all 160 experts (ref/zero_3070/gpu_all160.json)."""
    try:
        with open(os.path.join(K.REF, 'zero_3070', 'gpu_all160.json')) as f:
            ref = json.load(f)['results']
    except Exception as ex:  # noqa
        return 'n/a (%s)' % ex
    out = {}
    for g in res:
        for k, v in res[g].items():
            r = ref.get(g, {}).get(k)
            if r:
                out[f'{g}/{k}'] = {'compared_here': v['compared'], 'compared_3070': r['compared'],
                                   'mismatch_here': v['mismatch_value'] + v['mismatch_signzero_only'] + v['mismatch_nan_payload'],
                                   'mismatch_3070': r['mismatch_value'] + r['mismatch_signzero_only'] + r['mismatch_nan_payload'],
                                   'zero_outputs_here': v['zero_outputs'], 'zero_outputs_3070': r['zero_outputs']}
    return out


if __name__ == '__main__':
    sys.exit(main())
