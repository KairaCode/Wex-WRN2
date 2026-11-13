"""Diagnostics/extended sweep: kernel structure (unrolled vs rolled), payload load cache flavour,
occupancy limited by shared-memory padding. Every configuration is verified byte-exact.

Usage: python -I diag.py ENC_DIR OUT_JSON CONFIGSET [runs]
CONFIGSET: 'diag' (A-focused diagnostics) or 'final' (best structure for all variants).
"""
import json, os, sys, time
import numpy as np
import cupy as cp
from cupy_backends.cuda.api import driver as drv

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {  # payload file, table file, ALPHA, CTX, ILV
    'A': ('A', 'A', 16, 0, 0), 'A259': ('A', 'A259', 16, 1, 0),
    'B2': ('B2', 'B2', 16, 0, 2), 'B2c': ('B2', 'A259', 16, 1, 2),
    'C': ('C', 'C', 16, 2, 0), 'CB2': ('CB2', 'CB2', 16, 2, 2),
    'D': ('D', 'D', 8, 1, 0), 'DB2': ('DB2', 'DB2', 8, 1, 2),
}
ROWS = {0: 567, 1: 259, 2: 1036}
OUT_PER_MATRIX = 5760 * 1024


def smem_bytes(alpha, ctx):
    return 16 + ROWS[ctx] * alpha * 2 + (16 if ctx == 2 else 0)


def configs(which):
    cf = []
    if which == 'diag':
        for ldm in (0, 1):
            for nt in (32, 64, 128, 256):
                cf.append(('A', 'wdec', nt, ldm, 0, None))
                cf.append(('A', 'wdec2r', nt, ldm, 0, None))
        # occupancy sweep: rolled A, nt=64, target blocks/SM via padding
        for blk in (1, 2, 3, 4):
            cf.append(('A', 'wdec2r', 64, 0, 0, blk))
            cf.append(('A', 'wdec', 64, 0, 0, blk))
    elif which == 'final':
        for v in VARIANTS:
            for nt in (32, 64, 128, 256):
                cf.append((v, 'wdec2r', nt, 0, 0, None))
                cf.append((v, 'wdec2r', nt, 1, 0, None))
    return cf


def kname(kind, alpha, ctx, ilv, nt):
    if kind == 'wdec':
        return f'wdec<{alpha},{ctx},{ilv},{nt}>'
    if kind == 'wdec2r':
        return f'wdec2<{alpha},{ctx},{ilv},{nt},1>'
    if kind == 'wdec2u':
        return f'wdec2<{alpha},{ctx},{ilv},{nt},4>'
    raise ValueError(kind)


def main():
    enc_dir, out_json, which = sys.argv[1], sys.argv[2], sys.argv[3]
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    cf = configs(which)
    com = np.load(os.path.join(enc_dir, 'common.npz'))
    raw = com['raw']; spm = int(com['spm']); nmat = com['mtype'].size
    nstreams = nmat * spm
    d_raw = cp.asarray(raw); d_scales = cp.asarray(com['scales'])
    d_mtype = cp.asarray(com['mtype']); d_med = cp.asarray(com['med'])
    src = open(os.path.join(HERE, 'wnr2dec.cu')).read()
    mods = {}
    t0 = time.time()
    for ldm in sorted({c[3] for c in cf}):
        names = sorted({kname(c[1], *VARIANTS[c[0]][2:], c[2]) for c in cf if c[3] == ldm})
        mods[ldm] = cp.RawModule(code=src, options=('-arch=sm_86', '-std=c++17', f'-DLDM={ldm}'), name_expressions=names)
        mods[ldm].compile()
    clk = mods[min(mods)].get_function('clkprobe')
    d_clk = cp.zeros(3, cp.int64)
    res = {'compile_s': time.time() - t0, 'results': {}}
    d_out = cp.zeros(nstreams * 1024, cp.uint8)
    d_err = cp.zeros(1, cp.int32)
    loaded = {}

    def load(vname):
        if vname in loaded:
            return loaded[vname]
        z = np.load(os.path.join(enc_dir, f'var_{vname}.npz'))
        pay = z['payload']
        payp = np.concatenate([pay, np.zeros((-pay.size) % 4 + 8192, np.uint8)])
        offs = np.zeros(nstreams + 1, np.uint32); offs[1:] = np.cumsum(z['lengths'].astype(np.int64))
        d = {'pay': cp.asarray(payp.view(np.uint32)), 'offs': cp.asarray(offs), 'tabs': cp.asarray(z['tables'].reshape(-1))}
        bt = np.zeros((3, 8), np.uint16)
        if 'bittab' in z.files:
            bt[:, :7] = z['bittab']
        d['bittab'] = cp.asarray(bt.reshape(-1))
        d['signs'] = cp.asarray(z['signs']) if 'signs' in z.files else cp.zeros(1, cp.uint32)
        loaded[vname] = d
        return d

    def sm_mhz():
        clk((46,), (32,), (d_clk, np.int32(1 << 21)))
        cp.cuda.runtime.deviceSynchronize()
        c, ns, _ = d_clk.get()
        return float(c) / float(ns) * 1000.0

    for (v, kind, nt, ldm, _, blk_target) in cf:
        pv, tv, alpha, ctx, ilv = VARIANTS[v]
        dp, dt = load(pv), load(tv)
        name = kname(kind, alpha, ctx, ilv, nt)
        fn = mods[ldm].get_function(name)
        bpm = -(-spm // nt)
        smem = smem_bytes(alpha, ctx)
        if blk_target:
            # pad so that exactly blk_target blocks fit per SM (1 KB per-block reservation on sm_86)
            smem = max(smem, (102400 // blk_target) - 1024 - 16)
            if smem > 48 * 1024:
                fn.max_dynamic_shared_size_bytes = smem
        key = f'{v}/{kind}/nt{nt}/ldm{ldm}' + (f'/blk{blk_target}' if blk_target else '')

        def args(m0):
            return (dp['pay'], dp['offs'], d_scales, dp['signs'], dt['tabs'], dt['bittab'], d_mtype, d_med,
                    np.int32(m0), np.int32(spm), np.int32(bpm), d_out, d_err)

        d_out.fill(0); d_err.fill(0)
        fn((nmat * bpm,), (nt,), args(0), shared_mem=smem)
        cp.cuda.runtime.deviceSynchronize()
        errs = int(d_err.get()[0])
        bad = int((d_out.reshape(nstreams, 1024) != d_raw.reshape(nstreams, 1024)).any(1).sum())
        occ = drv.occupancyMaxActiveBlocksPerMultiprocessor(fn.kernel.ptr, nt, smem)
        r = {'variant': v, 'kernel': name, 'ldm': ldm, 'nt': nt, 'smem': smem, 'regs': fn.num_regs,
             'local_bytes': fn.local_size_bytes, 'occ_blocks': occ, 'occ_warps': occ * nt // 32,
             'byte_exact': bad == 0 and errs == 0, 'bad_streams': bad, 'errors': errs}
        st = cp.cuda.Stream(non_blocking=True)
        for mc in (1, 3, nmat):
            with st:
                tw = time.time()
                while time.time() - tw < 0.3:
                    for _ in range(5):
                        fn((mc * bpm,), (nt,), args(0), shared_mem=smem)
                    st.synchronize()
                evs = [(cp.cuda.Event(), cp.cuda.Event()) for _ in range(runs)]
                for a_, b_ in evs:
                    a_.record(st)
                    fn((mc * bpm,), (nt,), args(0), shared_mem=smem)
                    b_.record(st)
                st.synchronize()
            ts = np.array([cp.cuda.get_elapsed_time(a_, b_) * 1000.0 for a_, b_ in evs])
            med = float(np.median(ts))
            r[f'm{mc}'] = {'us_median': med, 'us_min': float(ts.min()), 'us_p90': float(np.percentile(ts, 90)),
                           'us_per_matrix': med / mc, 'decoded_GiB_s': mc * OUT_PER_MATRIX / (med * 1e-6) / 2**30,
                           'sm_mhz_after': sm_mhz()}
        res['results'][key] = r
        print(f"{key:34s} regs {r['regs']:3d} lmem {r['local_bytes']} occW {r['occ_warps']:2d} exact {r['byte_exact']} "
              f"m1 {r['m1']['us_median']:7.1f} m3 {r['m3']['us_median']:7.1f} m{nmat} {r[f'm{nmat}']['us_median']:8.1f} us "
              f"({r[f'm{nmat}']['decoded_GiB_s']:5.1f} GiB/s, {r[f'm{nmat}']['sm_mhz_after']:.0f} MHz)", flush=True)
        json.dump(res, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
