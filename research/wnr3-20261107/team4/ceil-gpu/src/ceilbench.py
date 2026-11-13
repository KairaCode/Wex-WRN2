"""team4/ceil-gpu: byte-exact check + paired timing of the ceil-form inverse inside the WNR2 GPU decoder.

Usage: python -I ceilbench.py ENC_DIR OUT_JSON [rounds] [nts] [opts]

1. Scale streams are decoded on the GPU (sdec) for the original and the ceil-form scale streams and checked
   against the original / CPU-transformed scale slabs.
2. Every weight variant x launch config is run once on cleared outputs and checked on the GPU against the
   ORIGINAL safetensors bytes (weights; for ceil-form also the corrected scale slab).
3. Timing (team-2 bench2.py style, paired): in every round all variants of a config are launched back to back
   on one stream (order rotated per round), each bracketed by CUDA events; per-round ratios -> median.
   Sizes: 1 matrix, 3 matrices = 1 record, 24 matrices = 8 records in one launch, and 8 records as 8 launches
   on 8 streams. Then the same for the pipeline scale decode + weight decode.
"""
import json, os, sys, time, subprocess
import numpy as np
import cupy as cp
from cupy_backends.cuda.api import driver as drv

HERE = os.path.dirname(os.path.abspath(__file__))
NVSMI = r'C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_20f49c6eb5c9b39b\nvidia-smi.exe'
# name: (payload/table file, ALPHA, CTX, ILV, CEIL, scale set)
VARS = {
    'A259': ('A259', 16, 1, 0, 0, 'orig'),
    'D': ('D', 8, 1, 0, 0, 'orig'),
    'CF0': ('CF', 16, 1, 0, 0, 'ceil'),     # ceil-form streams, inverse OFF (negative control / inverse cost)
    'CF': ('CF', 16, 1, 0, 1, 'ceil'),      # inverse + scale patch (byte store for transformed blocks only)
    'CFw': ('CF', 16, 1, 0, 2, 'ceil'),     # inverse + full scale-slab write
    'CF5': ('CF', 16, 1, 0, 5, 'ceil'),     # inverse + deferred end-of-stream scale patch (16-B words)
    'CF6': ('CF', 16, 1, 0, 6, 'ceil'),     # inverse + deferred end-of-stream full scale write
    'CF3': ('CF', 16, 1, 0, 3, 'ceil'),     # diagnostic: weight inverse only (scales NOT corrected)
    'CF4': ('CF', 16, 1, 0, 4, 'ceil'),     # diagnostic: per-block scale patch only (weights NOT corrected)
    'CFD0': ('CFD', 8, 1, 0, 0, 'ceil'),
    'CFD': ('CFD', 8, 1, 0, 1, 'ceil'),
    'CFDw': ('CFD', 8, 1, 0, 2, 'ceil'),
    'CFD5': ('CFD', 8, 1, 0, 5, 'ceil'),
    'CFD6': ('CFD', 8, 1, 0, 6, 'ceil'),
    'DB2': ('DB2', 8, 1, 2, 0, 'orig'),     # team 2 DB2: sign bypass + two chains (even/odd blocks)
    'CFDB20': ('CFDB2', 8, 1, 2, 0, 'ceil'),
    'CFDB2': ('CFDB2', 8, 1, 2, 5, 'ceil'),  # ceil-form + sign bypass + two chains, deferred scale patch
    'CFDB26': ('CFDB2', 8, 1, 2, 6, 'ceil'),
}
PATCH = (1, 4, 5)     # osc pre-holds the decoded transformed slab
FULLW = (2, 6)        # osc pre-filled with garbage, every byte must be written
PIPE_VARS = os.environ.get('CEIL_PIPE_VARS', 'A259,D,CF,CF5,CFD,CFD5').split(',')
ROWS = {0: 567, 1: 259, 2: 1036}
OPTSETS = {'lb': ('-arch=sm_86', '-std=c++17'), 'out4': ('-arch=sm_86', '-std=c++17', '-DOUTMODE=4')}
OUT_PER_MATRIX = 5760 * 1024
SC_PER_MATRIX = 368640
PIPE = os.environ.get('CEIL_PIPE', '1') == '1'


def nvsmi():
    try:
        return subprocess.run([NVSMI, '--query-gpu=clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu',
                               '--format=csv,noheader'], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa
        return str(e)


def kname(alpha, ctx, ilv, nt, ceil):
    return f'wdec<{alpha},{ctx},{ilv},{nt},{ceil}>'


def main():
    enc_dir, out_json = sys.argv[1], sys.argv[2]
    R = int(sys.argv[3]) if len(sys.argv) > 3 else 31
    nts = [int(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 else [32, 64, 128, 256]
    opts = sys.argv[5].split(',') if len(sys.argv) > 5 else ['lb', 'out4']
    vnames = sys.argv[6].split(',') if len(sys.argv) > 6 else list(VARS)
    props = cp.cuda.runtime.getDeviceProperties(0)
    res = {'info': {'gpu': props['name'].decode(), 'sms': props['multiProcessorCount'], 'clock_khz_max': props['clockRate'],
                    'smem_per_sm': props['sharedMemPerMultiprocessor'], 'regs_per_sm': props['regsPerMultiprocessor'],
                    'driver': cp.cuda.runtime.driverGetVersion(), 'runtime': cp.cuda.runtime.runtimeGetVersion(),
                    'cupy': cp.__version__},
           'rounds': R, 'nvsmi_start': nvsmi(), 'scale_decode': {}, 'results': {}}
    print(json.dumps(res['info']), res['nvsmi_start'], flush=True)
    com = np.load(os.path.join(enc_dir, 'common.npz'))
    spm = int(com['spm']); sspm = int(com['sspm']); nmat = com['mtype'].size
    nstreams = nmat * spm
    d_raw = cp.asarray(com['raw'])                       # ORIGINAL weight bytes (safetensors order)
    d_sco_ref = cp.asarray(com['scales'])                # ORIGINAL scale bytes
    d_scc_ref = cp.asarray(com['scales_ceil'])           # CPU-transformed scales (what the ceil scale streams code)
    d_mtype = cp.asarray(com['mtype'])
    d_med = {'orig': cp.asarray(com['med']), 'ceil': cp.asarray(com['med_ceil'])}
    src = open(os.path.join(HERE, 'wnr2dec.cu')).read()
    t0 = time.time()
    mods = {}
    for o in opts:
        names = sorted({kname(VARS[v][1], VARS[v][2], VARS[v][3], nt, VARS[v][4]) for v in vnames for nt in nts}
                       | {f'sdec<{n}>' for n in (32, 64, 128, 256)})
        mods[o] = cp.RawModule(code=src, options=OPTSETS[o], name_expressions=names)
        mods[o].compile()
    res['compile_s'] = time.time() - t0
    print('compile', res['compile_s'], flush=True)
    st = cp.cuda.Stream(non_blocking=True)
    sts = [cp.cuda.Stream(non_blocking=True) for _ in range(nmat // 3)]
    d_err = cp.zeros(1, cp.int32)

    # ---------------------------------------------------------------- scale streams
    sdat = {}
    for key in ('orig', 'ceil'):
        z = np.load(os.path.join(enc_dir, f'scl_{key}.npz'))
        pay = z['payload']
        payp = np.concatenate([pay, np.zeros((-pay.size) % 4 + 8192, np.uint8)])
        offs = np.zeros(z['lengths'].size + 1, np.uint32); offs[1:] = np.cumsum(z['lengths'].astype(np.int64))
        assert offs[-1] == pay.size and z['lengths'].size == nmat * sspm
        sdat[key] = {'pay': cp.asarray(payp.view(np.uint32)), 'offs': cp.asarray(offs),
                     'cum': cp.asarray(z['cum'].reshape(-1)), 'lut': cp.asarray(z['lut'].reshape(-1)),
                     'bytes': int(pay.size + 2 * z['lengths'].size)}
    d_scdec = {k: cp.zeros(nmat * SC_PER_MATRIX, cp.uint8) for k in ('orig', 'ceil')}   # sdec outputs

    def sdec_launcher(key, nt, out_buf, o='lb'):
        fn = mods[o].get_function(f'sdec<{nt}>')
        bpm = -(-sspm // nt)
        sd = sdat[key]

        def launch(mc, m0=0):
            fn((mc * bpm,), (nt,), (sd['pay'], sd['offs'], sd['cum'], sd['lut'], d_mtype, np.int32(m0), np.int32(sspm),
                                    np.int32(bpm), out_buf, d_err))
        return fn, launch

    def timed(launchers, mcs=(1, 3, nmat), multi=True):
        """launchers: {name: launch(mc, m0)}. Returns per-name timings with paired ratios."""
        names = list(launchers)
        out = {}
        for mc in mcs:
            with st:
                tw = time.time()
                while time.time() - tw < 0.25:
                    for n in names:
                        launchers[n](mc)
                    st.synchronize()
                T = np.zeros((R, len(names)))
                evs = []
                for r in range(R):
                    order = [(r + k) % len(names) for k in range(len(names))]
                    row = []
                    for k in order:
                        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
                        e0.record(st); launchers[names[k]](mc); e1.record(st)
                        row.append((k, e0, e1))
                    evs.append(row)
                st.synchronize()
                for r, row in enumerate(evs):
                    for k, e0, e1 in row:
                        T[r, k] = cp.cuda.get_elapsed_time(e0, e1) * 1000.0
            out[f'm{mc}'] = T
        if multi:
            nrec = nmat // 3
            T = np.zeros((R, len(names)))

            def multi_run(lf):
                e0 = cp.cuda.Event(); e0.record(st)
                ends = []
                for ri, s_ in enumerate(sts):
                    s_.wait_event(e0)
                    with s_:
                        lf(3, 3 * ri)
                    ee = cp.cuda.Event(); ee.record(s_); ends.append(ee)
                for ee in ends:
                    st.wait_event(ee)
                e1 = cp.cuda.Event(); e1.record(st)
                return e0, e1
            for r in range(R + 4):
                order = [(r + k) % len(names) for k in range(len(names))]
                row = []
                for k in order:
                    e0, e1 = multi_run(lambda mc, m0, _k=k: launchers[names[_k]](mc, m0))
                    row.append((k, e0, e1))
                st.synchronize()
                if r >= 4:
                    for k, e0, e1 in row:
                        T[r - 4, k] = cp.cuda.get_elapsed_time(e0, e1) * 1000.0
            out[f'{nrec}streams'] = T
        summ = {}
        for key, T in out.items():
            for i, n in enumerate(names):
                d = summ.setdefault(n, {})
                d[key] = {'us_median': float(np.median(T[:, i])), 'us_min': float(T[:, i].min()),
                          'us_p90': float(np.percentile(T[:, i], 90))}
                for j, b in enumerate(names):
                    if b != n:
                        d[key][f'ratio_vs_{b}'] = float(np.median(T[:, i] / T[:, j]))
        return summ

    sdec_cfg = {}
    for key in ('orig', 'ceil'):
        ref = d_sco_ref if key == 'orig' else d_scc_ref
        for nt in (32, 64, 128, 256):
            fn, launch = sdec_launcher(key, nt, d_scdec[key])
            d_scdec[key].fill(0); d_err.fill(0)
            cp.cuda.runtime.deviceSynchronize()
            with st:
                launch(nmat)
            st.synchronize()
            bad = int((d_scdec[key].reshape(-1, 512) != ref.reshape(-1, 512)).any(1).sum())
            errs = int(d_err.get()[0])
            r = {'nt': nt, 'regs': fn.num_regs, 'local_bytes': fn.local_size_bytes, 'static_smem': fn.shared_size_bytes,
                 'occ_blocks_per_sm': drv.occupancyMaxActiveBlocksPerMultiprocessor(fn.kernel.ptr, nt, 0),
                 'bad_streams': bad, 'errors': errs, 'byte_exact': bad == 0 and errs == 0, 'streams': nmat * sspm,
                 'coded_bytes_incl_lengths': sdat[key]['bytes']}
            sdec_cfg[(key, nt)] = launch
            res['scale_decode'][f'{key}/nt{nt}'] = r
            print('sdec', key, nt, r, flush=True)
    # final scale slabs used by the weight kernels = GPU-decoded scale streams
    assert all(res['scale_decode'][k]['byte_exact'] for k in res['scale_decode'])
    sl = {f'{k}/nt{nt}': sdec_cfg[(k, nt)] for k in ('orig', 'ceil') for nt in (32, 64, 128, 256)}
    res['scale_decode_timing'] = timed(sl)
    for k, v in res['scale_decode_timing'].items():
        print('sdec timing', k, {m: round(v[m]['us_median'], 1) for m in v}, flush=True)
    json.dump(res, open(out_json, 'w'), indent=1)

    # ---------------------------------------------------------------- weight variants
    loaded = {}

    def load(vf):
        if vf in loaded:
            return loaded[vf]
        z = np.load(os.path.join(enc_dir, f'var_{vf}.npz'))
        pay = z['payload']
        payp = np.concatenate([pay, np.zeros((-pay.size) % 4 + 8192, np.uint8)])
        offs = np.zeros(z['lengths'].size + 1, np.uint32); offs[1:] = np.cumsum(z['lengths'].astype(np.int64))
        assert offs[-1] == pay.size
        d = {'pay': cp.asarray(payp.view(np.uint32)), 'offs': cp.asarray(offs), 'tabs': cp.asarray(z['tables'].reshape(-1)),
             'signs': cp.asarray(z['signs']) if 'signs' in z.files else cp.zeros(1, cp.uint32),
             'bytes': int(pay.size + 2 * z['lengths'].size + (z['signs'].size * 4 if 'signs' in z.files else 0))}
        loaded[vf] = d
        return d

    d_out = cp.zeros(nstreams * 1024, cp.uint8)
    d_osc = cp.zeros(nmat * SC_PER_MATRIX, cp.uint8)
    d_pipe = {k: cp.zeros(nmat * SC_PER_MATRIX, cp.uint8) for k in ('orig', 'ceil')}
    d_bt = cp.zeros(24, cp.uint16)

    def wlauncher(v, nt, o, sc_in=None, osc=None):
        vf, alpha, ctx, ilv, ceil, sset = VARS[v]
        dp = load(vf)
        fn = mods[o].get_function(kname(alpha, ctx, ilv, nt, ceil))
        bpm = -(-spm // nt)
        smem = 16 + ROWS[ctx] * alpha * 2 + (nt * 144 if o == 'out4' else 0)
        if smem > 48 * 1024:
            fn.max_dynamic_shared_size_bytes = smem
        sc_in = d_scdec[sset] if sc_in is None else sc_in
        osc = d_osc if osc is None else osc
        med = d_med[sset]

        def launch(mc, m0=0):
            fn((mc * bpm,), (nt,), (dp['pay'], dp['offs'], sc_in, dp['signs'], dp['tabs'], d_bt, d_mtype, med,
                                    np.int32(m0), np.int32(spm), np.int32(bpm), d_out, d_err, osc), shared_mem=smem)
        return fn, smem, launch

    def prep_out(v):
        d_out.fill(0); d_err.fill(0)
        ceil = VARS[v][4]
        if ceil in PATCH:
            d_osc[:] = d_scdec['ceil']          # patch mode: osc holds the decoded (transformed) scale slab
        elif ceil in FULLW:
            d_osc.fill(0xAA)                    # full-write mode: every byte must be written
        cp.cuda.runtime.deviceSynchronize()     # fills run on the legacy stream; kernels on non-blocking streams

    def check(v):
        ceil, sset = VARS[v][4], VARS[v][5]
        badw = int((d_out.reshape(nstreams, 1024) != d_raw.reshape(nstreams, 1024)).any(1).sum())
        sc_slab = d_osc if ceil in PATCH + FULLW else d_scdec[sset]
        bads = int((sc_slab.reshape(nstreams, 64) != d_sco_ref.reshape(nstreams, 64)).any(1).sum())
        errs = int(d_err.get()[0])
        return badw, bads, errs

    for o in opts:
        for nt in nts:
            cfg = f'nt{nt}/{o}'
            L = {}
            info = {}
            for v in vnames:
                fn, smem, launch = wlauncher(v, nt, o)
                prep_out(v)
                with st:
                    launch(nmat)
                st.synchronize()
                badw, bads, errs = check(v)
                occ = drv.occupancyMaxActiveBlocksPerMultiprocessor(fn.kernel.ptr, nt, smem)
                info[v] = {'kernel': kname(*VARS[v][1:4], nt, VARS[v][4]), 'regs': fn.num_regs,
                           'local_bytes': fn.local_size_bytes, 'smem_per_block': smem, 'occ_blocks_per_sm': occ,
                           'occ_warps_per_sm': occ * nt // 32, 'weight_bad_streams': badw, 'scale_bad_streams': bads,
                           'errors': errs, 'byte_exact_vs_original': badw == 0 and bads == 0 and errs == 0,
                           'coded_weight_bytes': load(VARS[v][0])['bytes'],
                           'coded_scale_bytes': sdat[VARS[v][5]]['bytes']}
                L[v] = launch
            tm = timed(L)
            # multi-stream / single-launch re-verification on cleared outputs (8 launches on 8 streams)
            for v in vnames:
                prep_out(v)
                e0 = cp.cuda.Event(); e0.record(st)
                for ri, s_ in enumerate(sts):
                    s_.wait_event(e0)
                    with s_:
                        L[v](3, 3 * ri)
                cp.cuda.runtime.deviceSynchronize()
                badw, bads, errs = check(v)
                info[v]['multistream_recheck'] = {'weight_bad_streams': badw, 'scale_bad_streams': bads, 'errors': errs}
                for mc in (1, 3):   # partial launches: check exactly the decoded part
                    prep_out(v)
                    with st:
                        L[v](mc)
                    st.synchronize()
                    n = mc * spm
                    bw = int((d_out[:n * 1024].reshape(n, 1024) != d_raw[:n * 1024].reshape(n, 1024)).any(1).sum())
                    sc_slab = d_osc if VARS[v][4] in PATCH + FULLW else d_scdec[VARS[v][5]]
                    bs = int((sc_slab[:n * 64].reshape(n, 64) != d_sco_ref[:n * 64].reshape(n, 64)).any(1).sum())
                    info[v][f'm{mc}_recheck'] = {'weight_bad_streams': bw, 'scale_bad_streams': bs, 'errors': int(d_err.get()[0])}
                info[v]['timing'] = tm[v]
            # ---- pipeline: scale-stream decode (sdec<32>) then weight decode, same stream. Ceil-form variants
            # patch the decoded scale slab IN PLACE (scales input == osc, each lane reads scale b+1 before it
            # writes scale b and never touches another lane's bytes).
            pipe = {}
            if PIPE:
                for v in [x for x in PIPE_VARS if x in vnames]:
                    sset = VARS[v][5]
                    buf = d_pipe[sset]
                    _, sl_ = sdec_launcher(sset, 32, buf)
                    _, _, wl_ = wlauncher(v, nt, o, sc_in=buf, osc=buf)

                    def pl(mc, m0=0, _s=sl_, _w=wl_):
                        _s(mc, m0); _w(mc, m0)
                    pipe[v] = pl
                    # verify on cleared buffers (single launch over all matrices and 8 launches on 8 streams)
                    for mode in ('single', 'multi'):
                        d_out.fill(0); buf.fill(0); d_err.fill(0)
                        cp.cuda.runtime.deviceSynchronize()
                        if mode == 'single':
                            with st:
                                pl(nmat)
                        else:
                            for ri, s_ in enumerate(sts):
                                with s_:
                                    pl(3, 3 * ri)
                        cp.cuda.runtime.deviceSynchronize()
                        badw = int((d_out.reshape(nstreams, 1024) != d_raw.reshape(nstreams, 1024)).any(1).sum())
                        bads = int((buf.reshape(nstreams, 64) != d_sco_ref.reshape(nstreams, 64)).any(1).sum())
                        info[v][f'pipeline_check_{mode}'] = {'weight_bad_streams': badw, 'scale_bad_streams': bads,
                                                             'errors': int(d_err.get()[0])}
                tp = timed(pipe)
                for v in pipe:
                    info[v]['pipeline_timing'] = tp[v]
            res['results'][cfg] = info
            for v in vnames:
                t = info[v]['timing']
                print(f"{cfg:10s} {v:5s} regs {info[v]['regs']:3d} lmem {info[v]['local_bytes']} smem {info[v]['smem_per_block']:6d} "
                      f"occW {info[v]['occ_warps_per_sm']:2d} exact {info[v]['byte_exact_vs_original']} "
                      f"(w {info[v]['weight_bad_streams']} s {info[v]['scale_bad_streams']} e {info[v]['errors']}) | "
                      + ' '.join(f"{k} {t[k]['us_median']:8.1f}" + (f" /A {t[k].get('ratio_vs_A259', 1):.3f}" if 'A259' in vnames else '')
                                 + (f" /D {t[k].get('ratio_vs_D', 1):.3f}" if 'D' in vnames else '') for k in t), flush=True)
                if 'pipeline_timing' in info[v]:
                    t = info[v]['pipeline_timing']
                    print(f"{'':10s} {v:5s} PIPE check {info[v]['pipeline_check_single']} {info[v]['pipeline_check_multi']} | "
                          + ' '.join(f"{k} {t[k]['us_median']:8.1f}" + (f" /A {t[k].get('ratio_vs_A259', 1):.3f}" if 'A259' in pipe else '')
                                     + (f" /D {t[k].get('ratio_vs_D', 1):.3f}" if 'D' in pipe else '') for k in t), flush=True)
            json.dump(res, open(out_json, 'w'), indent=1)
    res['nvsmi_end'] = nvsmi()
    json.dump(res, open(out_json, 'w'), indent=1)
    print('nvsmi end', res['nvsmi_end'])


if __name__ == '__main__':
    main()
