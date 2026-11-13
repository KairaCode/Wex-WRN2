"""GPU decode prototype: verify every variant byte-exact and time it with CUDA events.

Usage: python -I gpudec.py ENC_DIR OUT_JSON [variants] [nts] [runs]
"""
import json, os, sys, time
import numpy as np
import cupy as cp

HERE = os.path.dirname(os.path.abspath(__file__))
# variant -> (payload file, table file, ALPHA, CTX, ILV)
VARIANTS = {
    'A':    ('A', 'A', 16, 0, 0),
    'A259': ('A', 'A259', 16, 1, 0),
    'B1':   ('B1', 'B1', 16, 0, 1),
    'B1c':  ('B1', 'A259', 16, 1, 1),
    'B2':   ('B2', 'B2', 16, 0, 2),
    'B2c':  ('B2', 'A259', 16, 1, 2),
    'C':    ('C', 'C', 16, 2, 0),
    'CB2':  ('CB2', 'CB2', 16, 2, 2),
    'D':    ('D', 'D', 8, 1, 0),
    'DB2':  ('DB2', 'DB2', 8, 1, 2),
}
ROWS = {0: 567, 1: 259, 2: 1036}
OUT_PER_MATRIX = 5760 * 1024  # bytes of packed FP4 per matrix


def smem_bytes(alpha, ctx):
    return 16 + ROWS[ctx] * alpha * 2 + (16 if ctx == 2 else 0)


def main():
    enc_dir, out_json = sys.argv[1], sys.argv[2]
    variants = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] not in ('', 'all') else list(VARIANTS)
    nts = [int(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 and sys.argv[4] else [32, 64, 128, 256]
    runs = int(sys.argv[5]) if len(sys.argv) > 5 else 25
    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    info = {'gpu': props['name'].decode(), 'sms': props['multiProcessorCount'], 'clock_khz_max': props['clockRate'],
            'smem_per_sm': props['sharedMemPerMultiprocessor'], 'smem_optin_per_block': props['sharedMemPerBlockOptin'],
            'regs_per_sm': props['regsPerMultiprocessor'], 'max_threads_per_sm': props['maxThreadsPerMultiProcessor'],
            'l2_bytes': props['l2CacheSize'], 'driver': cp.cuda.runtime.driverGetVersion(),
            'runtime': cp.cuda.runtime.runtimeGetVersion(), 'cupy': cp.__version__}
    try:
        from cupy_backends.cuda.libs import nvrtc
        info['nvrtc'] = nvrtc.getVersion()
    except Exception as e:  # noqa
        info['nvrtc'] = str(e)
    print(json.dumps(info), flush=True)

    com = np.load(os.path.join(enc_dir, 'common.npz'))
    raw = com['raw']; spm = int(com['spm']); nmat = com['mtype'].size
    d_raw = cp.asarray(raw)
    d_scales = cp.asarray(com['scales'])
    d_mtype = cp.asarray(com['mtype'])
    d_med = cp.asarray(com['med'])
    nstreams = nmat * spm

    src = open(os.path.join(HERE, 'wnr2dec.cu')).read()
    names = []
    for v in variants:
        _, _, a, c, i = VARIANTS[v]
        for nt in nts:
            names.append(f'wdec<{a},{c},{i},{nt}>')
    names = sorted(set(names))
    t0 = time.time()
    mod = cp.RawModule(code=src, options=('-arch=sm_86', '-std=c++17'), name_expressions=names)
    mod.compile()
    info['compile_s'] = time.time() - t0
    from cupy_backends.cuda.api import driver as drv
    clk = mod.get_function('clkprobe')
    d_clk = cp.zeros(3, cp.int64)

    def sm_mhz():
        clk((46,), (32,), (d_clk, np.int32(1 << 21)))
        cp.cuda.runtime.deviceSynchronize()
        c, ns, _ = d_clk.get()
        return float(c) / float(ns) * 1000.0 if ns > 0 else -1.0

    def warm(fn, grid, nt, a, smem, secs=0.4):
        t = time.time()
        while time.time() - t < secs:
            for _ in range(10):
                fn(grid, (nt,), a, shared_mem=smem)
            cp.cuda.runtime.deviceSynchronize()
    res = {'info': info, 'results': {}}
    d_out = cp.zeros(nstreams * 1024, cp.uint8)
    d_err = cp.zeros(1, cp.int32)
    loaded = {}

    def load(vname):
        if vname in loaded:
            return loaded[vname]
        z = np.load(os.path.join(enc_dir, f'var_{vname}.npz'))
        pay = z['payload']
        padn = (-pay.size) % 4 + 8192
        payp = np.concatenate([pay, np.zeros(padn, np.uint8)])
        offs = np.zeros(nstreams + 1, np.uint32)
        offs[1:] = np.cumsum(z['lengths'].astype(np.int64))
        assert offs[-1] == pay.size
        d = {'pay': cp.asarray(payp.view(np.uint32)), 'offs': cp.asarray(offs),
             'tabs': cp.asarray(z['tables'].reshape(-1)), 'bytes': int(pay.size), 'lengths': int(2 * z['lengths'].size)}
        if 'bittab' in z.files:
            bt = np.zeros((3, 8), np.uint16); bt[:, :7] = z['bittab']
            d['bittab'] = cp.asarray(bt.reshape(-1))
        else:
            d['bittab'] = cp.zeros(24, cp.uint16)
        if 'signs' in z.files:
            d['signs'] = cp.asarray(z['signs']); d['sign_bytes'] = int(z['signs'].size * 4)
        else:
            d['signs'] = cp.zeros(1, cp.uint32); d['sign_bytes'] = 0
        loaded[vname] = d
        return d

    for v in variants:
        pv, tv, alpha, ctx, ilv = VARIANTS[v]
        dp = load(pv); dt = load(tv)
        for nt in nts:
            fn = mod.get_function(f'wdec<{alpha},{ctx},{ilv},{nt}>')
            bpm = -(-spm // nt)
            sm = smem_bytes(alpha, ctx)

            def args(m0, mcount):
                return (dp['pay'], dp['offs'], d_scales, dp['signs'], dt['tabs'], dt['bittab'], d_mtype, d_med,
                        np.int32(m0), np.int32(spm), np.int32(bpm), d_out, d_err)

            pads = [0]
            if v == 'A' and nt == 32:
                pads = [0, 8192]   # 8192 B = KAIRA's 32-lane x 64-word cp.async ring (occupancy emulation)
            for pad in pads:
                key = f'{v}/nt{nt}' + (f'/ringpad{pad}' if pad else '')
                smem = sm + pad
                # verify on all matrices
                d_out.fill(0); d_err.fill(0)
                fn((nmat * bpm,), (nt,), args(0, nmat), shared_mem=smem)
                cp.cuda.runtime.deviceSynchronize()
                errs = int(d_err.get()[0])
                bad_streams = int((d_out.reshape(nstreams, 1024) != d_raw.reshape(nstreams, 1024)).any(1).sum())
                r = {'alpha': alpha, 'ctx': ctx, 'ilv': ilv, 'nt': nt, 'smem_bytes': smem,
                     'regs': fn.num_regs, 'local_bytes': fn.local_size_bytes,
                     'verified_streams': nstreams, 'mismatching_streams': bad_streams, 'invariant_errors': errs,
                     'byte_exact': bad_streams == 0 and errs == 0,
                     'compressed_bytes_all': dp['bytes'] + dp['lengths'] + dp['sign_bytes']}
                try:
                    occ = drv.occupancyMaxActiveBlocksPerMultiprocessor(fn.kernel.ptr, nt, smem)
                    r['occ_blocks_per_sm'] = occ
                    r['occ_warps_per_sm'] = occ * nt // 32
                except Exception as e:  # noqa
                    r['occ_err'] = str(e)
                r['blocks_per_matrix'] = bpm
                # timing: warm >= 0.4 s, then per-launch event pairs (median) and one batched pair (mean)
                st = cp.cuda.Stream(non_blocking=True)
                with st:
                    warm(fn, (nmat * bpm,), nt, args(0, nmat), smem)
                for mc in (1, 3, nmat):
                    with st:
                        warm(fn, (mc * bpm,), nt, args(0, mc), smem, 0.15)
                        evs = [(cp.cuda.Event(), cp.cuda.Event()) for _ in range(runs)]
                        for a_, b_ in evs:
                            a_.record(st)
                            fn((mc * bpm,), (nt,), args(0, mc), shared_mem=smem)
                            b_.record(st)
                        st.synchronize()
                        ts = np.array([cp.cuda.get_elapsed_time(a_, b_) * 1000.0 for a_, b_ in evs])
                        nb = 20
                        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
                        e0.record(st)
                        for _ in range(nb):
                            fn((mc * bpm,), (nt,), args(0, mc), shared_mem=smem)
                        e1.record(st)
                        st.synchronize()
                        batched = cp.cuda.get_elapsed_time(e0, e1) * 1000.0 / nb
                    mhz = sm_mhz()
                    med = float(np.median(ts))
                    r[f'm{mc}'] = {'us_median': med, 'us_min': float(ts.min()), 'us_p90': float(np.percentile(ts, 90)),
                                   'us_batched_mean': batched,
                                   'us_per_matrix': med / mc, 'us_per_record_equiv': med / mc * 3,
                                   'decoded_GiB_s': mc * OUT_PER_MATRIX / (med * 1e-6) / 2**30, 'runs': runs,
                                   'sm_mhz_after': mhz}
                # 8 records in flight as 8 launches (one record = 3 matrices each) on 8 streams
                nrec = nmat // 3
                sts = [cp.cuda.Stream(non_blocking=True) for _ in range(nrec)]
                main_s = cp.cuda.Stream(non_blocking=True)
                ts = []
                for it in range(runs + 10):
                    e0 = cp.cuda.Event(); e0.record(main_s)
                    ends = []
                    for ri, s_ in enumerate(sts):
                        s_.wait_event(e0)
                        with s_:
                            fn((3 * bpm,), (nt,), args(3 * ri, 3), shared_mem=smem)
                        ee = cp.cuda.Event(); ee.record(s_); ends.append(ee)
                    for ee in ends:
                        main_s.wait_event(ee)
                    e1 = cp.cuda.Event(); e1.record(main_s)
                    main_s.synchronize()
                    if it >= 10:
                        ts.append(cp.cuda.get_elapsed_time(e0, e1) * 1000.0)
                ts = np.array(ts); med = float(np.median(ts))
                r[f'{nrec}rec_{nrec}streams'] = {'us_median': med, 'us_min': float(ts.min()),
                                                 'us_per_record': med / nrec,
                                                 'decoded_GiB_s': nmat * OUT_PER_MATRIX / (med * 1e-6) / 2**30}
                # re-verify output of the multi-stream run
                cp.cuda.runtime.deviceSynchronize()
                r['multistream_verify_bad_streams'] = int((d_out.reshape(nstreams, 1024) != d_raw.reshape(nstreams, 1024)).any(1).sum())
                res['results'][key] = r
                print(key, json.dumps({k: r[k] for k in ('regs', 'smem_bytes', 'byte_exact', 'mismatching_streams',
                                                         'invariant_errors') if k in r},),
                      'occ', r.get('occ_warps_per_sm', r.get('occ_err')),
                      'm1 %.1f m3 %.1f m%d %.1f us; 8str %.1f us' % (r['m1']['us_median'], r['m3']['us_median'], nmat,
                                                                      r[f'm{nmat}']['us_median'],
                                                                      r[f'{nrec}rec_{nrec}streams']['us_median']),
                      flush=True)
                json.dump(res, open(out_json, 'w'), indent=1)
    json.dump(res, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
