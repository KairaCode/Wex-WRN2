"""gpu_bench: WNR2 / ceil-form (WNR3 CEIL) GPU decoder, byte-exact check against the ORIGINAL safetensors bytes and
paired timing (team 2 gpu-proto + team 4 ceil-gpu, ported to run on any CUDA GPU; arch detected from the device).

  python -I tests/gpu/run_gpu_bench.py --root <kit> [--data <dir>] [--out <dir>] [--quick] [options]

Steps
 1. Locate the 8 test experts of team 2/4 (sample-manifest indices 17,0,40,80,120,159,60,100; sha256-checked).
 2. Encode them on the CPU (cached in <out>/enc): A259, D, CF, CFD, DB2, CFDB2 weight streams + orig/ceil scale
    streams; every array is compared with the sha256 of team 4's encoder output (must be identical).
 3. NVRTC-compile the decoder (src_bench/wnr2dec.cu) for -arch=sm_<cc of the device> (or compute_<cc>).
 4. Scale-stream decode (sdec) byte-exact vs original / ceil-form scales.
 5. For every launch config (threads per block x store mode) and weight variant: decode all 24 matrices once on
    cleared outputs, compare on the GPU with the ORIGINAL bytes (weights + final scale slab); re-check 8 launches on
    8 streams and partial launches (1 matrix, 1 record); registers / local / shared memory / occupancy per kernel.
 6. Paired timing (>= 21 rounds; every round launches all variants back to back on one stream, order rotated,
    CUDA events): 1 matrix (m1), 1 record = 3 matrices (m3), 8 records in one launch (m24), 8 records on 8 streams
    (8streams); then the pipeline scale decode + weight decode.
 7. Tables: bench_table.txt (all configs), compare_3070.txt (best config per variant/size and same-config paired
    ratios next to team 4's RTX 3070 reference: ref/bench_3070/best_final.txt, ratios_final.json).
Variants (team 4 names): A259 = WNR2 today (16-symbol, 259 rows); D = sign bypass; CF5 / CF6 = ceil-form + inverse,
deferred scale patch / full write; CFD5 = ceil-form + sign bypass; DB2 = sign bypass + two chains; CFDB2 = ceil-form +
DB2. Negative controls (must NOT be byte-exact): CF0, CFD0, CFDB20 = ceil-form streams with the inverse switched off.
PASS = input sha256 ok, encoder output identical to team 4's, every non-control variant byte-exact in every check,
every negative control fails, all timings finite.
"""
# read by setup/kitenv.py with ast (runner discovery for run_all.ps1); never executed there
KIT_META = {'name': 'gpu_bench', 'order': 65, 'needs': ['gpu', 'data'], 'quick_minutes': 5, 'full_minutes': 20,
            'description': 'WNR2 / ceil-form GPU decoder: byte-exact vs original bytes + paired timing'}
import json, math, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'src_bench'))
import gpukit_common as K  # noqa: E402

TEST = 'gpu_bench'
# name: (payload/table file, ALPHA, CTX, ILV, CEIL, scale set)   [team 4 ceilbench.py VARS]
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
NEGATIVE = {'CF0', 'CFD0', 'CFDB20', 'CF3', 'CF4'}          # must NOT be byte-exact vs the original
DEFAULT_VARIANTS = 'A259,D,CF0,CF5,CF6,CFD0,CFD5,DB2,CFDB20,CFDB2'
DEFAULT_PIPE = 'A259,D,CF5,CF6,CFD5,DB2,CFDB2'
PATCH = (1, 4, 5)     # osc pre-holds the decoded transformed slab
FULLW = (2, 6)        # osc pre-filled with garbage, every byte must be written
ROWS = {0: 567, 1: 259, 2: 1036}
SC_PER_MATRIX = 368640
SIZES = ('m1', 'm3', 'm24', '8streams')
# same pairs as team 4 ceilratios.py
PAIRS = [('CF3', 'CF0'), ('CF5', 'CF0'), ('CF6', 'CF0'), ('CF', 'CF0'), ('CFD5', 'CFD0'), ('CFDB2', 'CFDB20'),
         ('CF5', 'A259'), ('CF', 'A259'), ('CF0', 'A259'), ('CF5', 'D'), ('CFD5', 'D'), ('CFD5', 'A259'),
         ('CFDB2', 'DB2'), ('CFDB2', 'A259'), ('CFDB2', 'D'), ('DB2', 'A259'), ('D', 'A259'), ('CF6', 'A259')]


def kname(alpha, ctx, ilv, nt, ceil):
    return f'wdec<{alpha},{ctx},{ilv},{nt},{ceil}>'


def fn_attrs(fn):
    d = {}
    for a in ('num_regs', 'local_size_bytes', 'shared_size_bytes', 'const_size_bytes', 'max_threads_per_block',
              'ptx_version', 'binary_version'):
        try:
            d[a] = int(getattr(fn, a))
        except Exception:  # noqa
            d[a] = None
    return d


def parse_args():
    p = K.base_parser(TEST, __doc__)
    p.add_argument('--rounds', type=int, default=None, help='paired timing rounds (>= 21; default 41, quick 21)')
    p.add_argument('--nts', default=None, help='threads per block list (default 32,64,128,256; quick 32,64)')
    p.add_argument('--opts', default=None, help='store modes: lb (plain stores), out4 (128-B staged stores); default lb,out4')
    p.add_argument('--variants', default=DEFAULT_VARIANTS, help='weight variants (default %s)' % DEFAULT_VARIANTS)
    p.add_argument('--pipe-variants', default=DEFAULT_PIPE, help='variants timed as scale+weight pipeline ("" = none)')
    p.add_argument('--arch-mode', choices=('sm', 'compute'), default='sm',
                   help='sm = NVRTC builds a cubin for sm_<cc> (default); compute = PTX for compute_<cc>, driver JIT')
    p.add_argument('--nvrtc-extra', default='', help='extra NVRTC options, comma separated (e.g. -DLDM=1)')
    p.add_argument('--enc', default=None, help='encoded-stream cache dir (default <out>/enc)')
    p.add_argument('--reencode', action='store_true', help='ignore the cache and encode again')
    p.add_argument('--encode-only', action='store_true', help='stop after the CPU encode (no GPU needed)')
    return p.parse_args()


def get_encoded(args, log):
    enc = os.path.abspath(args.enc) if args.enc else os.path.join(args.out, 'enc')
    with open(os.path.join(K.REF, 't4enc_expected.json')) as f:
        expected = json.load(f)
    repp = os.path.join(enc, 'encode_report.json')
    if not args.reencode and os.path.exists(repp):
        with open(repp) as f:
            rep = json.load(f)
        files_ok = all(os.path.exists(os.path.join(enc, fn)) for fn in expected['files'])
        if rep.get('complete') and files_ok and rep.get('eq_team4_encoder_output_all'):
            log('using cached encode in', enc, '(identical to team 4 encoder output)')
            return enc, rep, expected
        log('cache incomplete or not verified, re-encoding')
    man, paths, _ = K.resolve_experts(args, K.PICK8, log)
    import t4enc
    log('encoding 8 experts x 6 variants on the CPU (team 2/4 encoder, a few minutes) ->', enc)
    rep = t4enc.encode_all(man, paths, os.path.join(K.REF, 'model_counts.npz'), enc, log, expected=expected)
    log('encode done in %.1f s; identical to team 4 output: %s; sizes equal: %s' % (
        rep['total_s'], rep.get('eq_team4_encoder_output_all'), rep.get('eq_team4_sizes')))
    return enc, rep, expected


def main():
    args = parse_args()
    K.resolve_paths(args, TEST)
    log = K.Log(os.path.join(args.out, TEST + '.log'))
    return K.run_main(TEST, lambda: body(args, log), args, args.out)


def body(args, log):
    quick = args.quick
    R = args.rounds if args.rounds is not None else (21 if quick else 41)
    if R < 21:
        raise ValueError('--rounds must be >= 21 (paired method)')
    nts = [int(v) for v in (args.nts or ('32,64' if quick else '32,64,128,256')).split(',')]
    opts = (args.opts or 'lb,out4').split(',')
    vnames = [v for v in args.variants.split(',') if v]
    for v in vnames:
        if v not in VARS:
            raise ValueError('unknown variant %s (known: %s)' % (v, ','.join(VARS)))
    pipe_vars = [v for v in args.pipe_variants.split(',') if v and v in vnames]
    summary = {'args': {k: getattr(args, k) for k in ('root', 'data', 'out', 'quick', 'rounds', 'nts', 'opts', 'variants',
                                                      'pipe_variants', 'arch_mode', 'nvrtc_extra')},
               'rounds': R, 'nts': nts, 'opts': opts, 'variants': vnames}
    log('gpu_bench start; quick=%s rounds=%d nts=%s opts=%s' % (quick, R, nts, opts))

    enc_dir, rep, expected = get_encoded(args, log)
    summary['encode'] = {k: rep.get(k) for k in ('transform', 'eq_team4_encoder_output_all', 'eq_team4_sizes', 'total_s')}
    summary['encode']['sizes'] = {v: {k: rep['sizes'][v][k] for k in ('record_total_bytes_all8', 'ratio_of_raw_record_18800640')}
                                  for v in rep['sizes'] if 'record_total_bytes_all8' in rep['sizes'][v]}
    if args.encode_only:
        summary['env'] = K.env_info(args.device)
        ok = bool(rep.get('eq_team4_encoder_output_all')) and rep['transform']['cpu_roundtrip_fail_matrices'] == 0
        return K.finish(TEST, args.out, 'PASS' if ok else 'FAIL', 'encode-only; identical to team 4: %s' %
                        rep.get('eq_team4_encoder_output_all'), summary, log)

    import cupy as cp
    from cupy_backends.cuda.api import driver as drv
    cp.cuda.Device(args.device).use()
    summary['env'] = K.env_info(args.device)
    log('env', json.dumps(summary['env']))
    arch, cc = K.device_arch(args.device, args.arch_mode)
    extra = tuple(o for o in args.nvrtc_extra.split(',') if o)
    OPTSETS = {'lb': (arch, '-std=c++17') + extra, 'out4': (arch, '-std=c++17', '-DOUTMODE=4') + extra}
    summary['nvrtc_arch'] = arch
    log('NVRTC arch', arch, 'options', OPTSETS)
    res = {'info': summary['env'], 'rounds': R, 'nvsmi_start': K.nvsmi_clocks(), 'scale_decode': {}, 'results': {},
           'nvrtc_options': OPTSETS}

    com = np.load(os.path.join(enc_dir, 'common.npz'))
    spm = int(com['spm']); sspm = int(com['sspm']); nmat = com['mtype'].size
    nstreams = nmat * spm
    # ORIGINAL bytes straight from the expert files (= the safetensors tensor bytes), compared with the encoder copy
    man, paths, _ = K.resolve_experts(args, K.PICK8, log)
    raw_l, sco_l = [], []
    for ei in K.PICK8:
        e = man['experts'][ei]
        t = K.load_record(paths[e['file']], e)
        for w in ('w1', 'w2', 'w3'):
            raw_l.append(t[w + '.weight'][0]); sco_l.append(t[w + '.scale'][0])
    raw = np.concatenate(raw_l); sco = np.concatenate(sco_l)
    eq_raw = bool(raw.size == com['raw'].size and (raw == com['raw']).all() and (sco == com['scales']).all())
    summary['original_bytes_eq_encoder_input'] = eq_raw
    if not eq_raw:
        raise RuntimeError('encoder cache does not hold the original bytes of the 8 experts (delete <out>/enc)')
    d_raw = cp.asarray(raw)                              # ORIGINAL weight bytes (safetensors order)
    d_sco_ref = cp.asarray(sco)                          # ORIGINAL scale bytes
    d_scc_ref = cp.asarray(com['scales_ceil'])           # CPU-transformed scales (what the ceil scale streams code)
    d_mtype = cp.asarray(com['mtype'])
    d_med = {'orig': cp.asarray(com['med']), 'ceil': cp.asarray(com['med_ceil'])}
    del raw_l, sco_l, raw, sco
    with open(os.path.join(HERE, 'src_bench', 'wnr2dec.cu')) as f:
        src = f.read()
    t0 = time.time()
    mods = {}
    for o in opts:
        names = sorted({kname(VARS[v][1], VARS[v][2], VARS[v][3], nt, VARS[v][4]) for v in vnames for nt in nts}
                       | {f'sdec<{n}>' for n in (32, 64, 128, 256)})
        mods[o] = cp.RawModule(code=src, options=OPTSETS[o], name_expressions=names)
        mods[o].compile()
    res['compile_s'] = time.time() - t0
    log('compile %.1f s' % res['compile_s'])
    st = cp.cuda.Stream(non_blocking=True)
    nrec = nmat // 3
    sts = [cp.cuda.Stream(non_blocking=True) for _ in range(nrec)]
    d_err = cp.zeros(1, cp.int32)
    out_json = os.path.join(args.out, 'bench.json')

    def dump():
        with open(out_json, 'w') as f:
            json.dump(res, f, indent=1, default=str)

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

    def sdec_launcher(key, nt, out_buf, o=None):
        fn = mods[o or opts[0]].get_function(f'sdec<{nt}>')
        bpm = -(-sspm // nt)
        sd = sdat[key]

        def launch(mc, m0=0):
            fn((mc * bpm,), (nt,), (sd['pay'], sd['offs'], sd['cum'], sd['lut'], d_mtype, np.int32(m0), np.int32(sspm),
                                    np.int32(bpm), out_buf, d_err))
        return fn, launch

    def timed(launchers, mcs=(1, 3, nmat), multi=True):
        """launchers: {name: launch(mc, m0)}. Returns per-name timings with paired ratios (team 4 method)."""
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
            key2 = 'm24' if key == f'm{nmat}' else ('8streams' if key == f'{nrec}streams' else key)
            for i, n in enumerate(names):
                d = summ.setdefault(n, {})
                d[key2] = {'us_median': float(np.median(T[:, i])), 'us_min': float(T[:, i].min()),
                           'us_p90': float(np.percentile(T[:, i], 90))}
                for j, b in enumerate(names):
                    if b != n:
                        d[key2][f'ratio_vs_{b}'] = float(np.median(T[:, i] / T[:, j]))
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
                 'attrs': fn_attrs(fn),
                 'occ_blocks_per_sm': drv.occupancyMaxActiveBlocksPerMultiprocessor(fn.kernel.ptr, nt, 0),
                 'bad_streams': bad, 'errors': errs, 'byte_exact': bad == 0 and errs == 0, 'streams': nmat * sspm,
                 'coded_bytes_incl_lengths': sdat[key]['bytes']}
            sdec_cfg[(key, nt)] = launch
            res['scale_decode'][f'{key}/nt{nt}'] = r
            log('sdec', key, nt, {k: r[k] for k in ('regs', 'static_smem', 'occ_blocks_per_sm', 'bad_streams', 'errors')})
    sdec_ok = all(res['scale_decode'][k]['byte_exact'] for k in res['scale_decode'])
    if not sdec_ok:
        dump()
        raise RuntimeError('scale-stream decode is not byte-exact: %s' % {k: v['bad_streams'] for k, v in res['scale_decode'].items()})
    sl = {f'{k}/nt{nt}': sdec_cfg[(k, nt)] for k in ('orig', 'ceil') for nt in (32, 64, 128, 256)}
    res['scale_decode_timing'] = timed(sl)
    for k, v in res['scale_decode_timing'].items():
        log('sdec timing', k, {m: round(v[m]['us_median'], 1) for m in v})
    dump()

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
                           'local_bytes': fn.local_size_bytes, 'smem_per_block': smem, 'attrs': fn_attrs(fn),
                           'occ_blocks_per_sm': occ, 'occ_warps_per_sm': occ * nt // 32, 'weight_bad_streams': badw,
                           'scale_bad_streams': bads, 'errors': errs,
                           'byte_exact_vs_original': badw == 0 and bads == 0 and errs == 0,
                           'coded_weight_bytes': load(VARS[v][0])['bytes'],
                           'coded_scale_bytes': sdat[VARS[v][5]]['bytes'], 'negative_control': v in NEGATIVE}
                L[v] = launch
            tm = timed(L)
            # multi-stream / partial-launch re-verification on cleared outputs (8 launches on 8 streams)
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
            for v in pipe_vars:
                sset = VARS[v][5]
                buf = d_pipe[sset]
                _, sl_ = sdec_launcher(sset, 32, buf, o)
                _, _, wl_ = wlauncher(v, nt, o, sc_in=buf, osc=buf)

                def pl(mc, m0=0, _s=sl_, _w=wl_):
                    _s(mc, m0); _w(mc, m0)
                pipe[v] = pl
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
            if pipe:
                tp = timed(pipe)
                for v in pipe:
                    info[v]['pipeline_timing'] = tp[v]
            res['results'][cfg] = info
            for v in vnames:
                t = info[v]['timing']
                log(f"{cfg:10s} {v:6s} regs {info[v]['regs']:3d} lmem {info[v]['local_bytes']} smem {info[v]['smem_per_block']:6d} "
                    f"occW {info[v]['occ_warps_per_sm']:2d} exact {info[v]['byte_exact_vs_original']} "
                    f"(w {info[v]['weight_bad_streams']} s {info[v]['scale_bad_streams']} e {info[v]['errors']}) | "
                    + ' '.join(f"{k} {t[k]['us_median']:8.1f}" for k in SIZES))
            dump()
    res['nvsmi_end'] = K.nvsmi_clocks()
    dump()

    # ---------------------------------------------------------------- verdict + tables
    checks = verdict(res, vnames, pipe_vars)
    checks['input_sha256_and_original_bytes'] = eq_raw
    checks['encoder_identical_to_team4'] = bool(rep.get('eq_team4_encoder_output_all'))
    checks['cpu_ceil_roundtrip'] = rep['transform']['cpu_roundtrip_fail_matrices'] == 0
    checks['scale_decode_byte_exact'] = sdec_ok
    table = tables(res, vnames)
    with open(os.path.join(args.out, 'bench_table.txt'), 'w') as f:
        f.write(table)
    cmp_txt, cmp_json = compare_3070(res, vnames)
    with open(os.path.join(args.out, 'compare_3070.txt'), 'w') as f:
        f.write(cmp_txt)
    log('\n' + cmp_txt)
    summary['checks'] = checks
    summary['compare_3070'] = cmp_json
    summary['kernels'] = {cfg: {v: {k: d[v][k] for k in ('kernel', 'regs', 'local_bytes', 'smem_per_block', 'occ_warps_per_sm')}
                                for v in d} for cfg, d in res['results'].items()}
    summary['best'] = cmp_json['best_here']
    summary['files'] = ['bench.json', 'bench_table.txt', 'compare_3070.txt', TEST + '.log']
    ok = all(v is True for v in checks.values() if isinstance(v, bool))
    b = cmp_json['best_here']
    short = ('byte-exact %s, neg.controls fire %s; best 8streams us: A259 %s CF5 %s CFD5 %s CFDB2 %s (%s)' % (
        checks['all_byte_exact'], checks['negative_controls_fire'],
        *(round(b.get(v, {}).get('8streams', {}).get('us', float('nan')), 1) for v in ('A259', 'CF5', 'CFD5', 'CFDB2')),
        summary['env'].get('gpu', {}).get('name', '?')))
    return K.finish(TEST, args.out, 'PASS' if ok else 'FAIL', short, summary, log)


def verdict(res, vnames, pipe_vars):
    bad, negok, fin = [], [], True
    for cfg, d in res['results'].items():
        for v, r in d.items():
            recs = [r] + [r[k] for k in ('multistream_recheck', 'm1_recheck', 'm3_recheck',
                                         'pipeline_check_single', 'pipeline_check_multi') if k in r]
            clean = all(x['weight_bad_streams'] == 0 and x['scale_bad_streams'] == 0 and x['errors'] == 0 for x in recs)
            if v in NEGATIVE:
                negok.append(r['weight_bad_streams'] > 0 or r['scale_bad_streams'] > 0)
                if not negok[-1]:
                    bad.append(f'{cfg}:{v} negative control did NOT fail')
            elif not clean:
                bad.append(f'{cfg}:{v}')
            for t in [r['timing']] + ([r['pipeline_timing']] if 'pipeline_timing' in r else []):
                for s in t.values():
                    fin &= math.isfinite(s['us_median']) and s['us_median'] > 0
    return {'all_byte_exact': not [b for b in bad if 'negative' not in b], 'negative_controls_fire': all(negok) if negok else None,
            'timings_finite': fin, 'failures': bad}


def tables(res, vnames):
    out = []
    P = out.append
    P(f"GPU {res['info'].get('gpu', {}).get('name')}  nvsmi start/end: {res.get('nvsmi_start')} | {res.get('nvsmi_end')}  "
      f"rounds {res['rounds']}  compile {res.get('compile_s', 0):.1f} s")
    OWN = {'CF5': 'CF0', 'CF6': 'CF0', 'CF': 'CF0', 'CF3': 'CF0', 'CFD5': 'CFD0', 'CFDB2': 'CFDB20'}
    for c, d in res['results'].items():
        P(f'== {c}')
        for v, r in d.items():
            t = r['timing']
            s = f"{v:6s} regs {r['regs']:3d} lmem {r['local_bytes']} smem {r['smem_per_block']:6d} occW {r['occ_warps_per_sm']:2d} exact {str(r['byte_exact_vs_original'])[0]} |"
            for k in SIZES:
                x = t[k]
                s += f" {k} {x['us_median']:7.1f}"
                if 'ratio_vs_A259' in x:
                    s += f" A{x['ratio_vs_A259']:.3f}"
                if 'ratio_vs_D' in x:
                    s += f" D{x['ratio_vs_D']:.3f}"
                if v in OWN and f'ratio_vs_{OWN[v]}' in x:
                    s += f" own{x['ratio_vs_' + OWN[v]]:.3f}"
                s += ' |'
            P(s)
            if 'pipeline_timing' in r:
                t = r['pipeline_timing']
                s = f"{'':6s} PIPE (sdec<32> + wdec) |"
                for k in SIZES:
                    x = t[k]
                    s += f" {k} {x['us_median']:7.1f}"
                    if 'ratio_vs_A259' in x:
                        s += f" A{x['ratio_vs_A259']:.3f}"
                    s += ' |'
                P(s)
    return '\n'.join(out) + '\n'


RX = re.compile(r'^(\S+)\s+(\S+)\s+best\s+([\d.]+) us @ \S+?:(\S+)\s+regs\s+(\d+) smem\s+(\d+) occW\s+(\d+)')


def compare_3070(res, vnames):
    ref = {}
    with open(os.path.join(K.REF, 'bench_3070', 'best_final.txt')) as f:
        for line in f:
            m = RX.match(line)
            if m:
                ref[(m.group(1), m.group(2))] = {'us': float(m.group(3)), 'cfg': m.group(4), 'regs': int(m.group(5)),
                                                 'smem': int(m.group(6)), 'occW': int(m.group(7))}
    with open(os.path.join(K.REF, 'bench_3070', 'ratios_final.json')) as f:
        rref = json.load(f)['summary_min_median_max']
    best = {}
    for cfg, d in res['results'].items():
        for v, r in d.items():
            for k, x in r['timing'].items():
                b = best.setdefault(v, {}).get(k)
                if b is None or x['us_median'] < b['us']:
                    best[v][k] = {'us': x['us_median'], 'cfg': cfg, 'regs': r['regs'], 'smem': r['smem_per_block'],
                                  'occW': r['occ_warps_per_sm']}
    lines = [f"Comparison with team 4's RTX 3070 reference (best config per variant and size = min over launch configs of "
             f"the median us; 3070 numbers from ref/bench_3070/best_final.txt, 41 rounds)",
             f"GPU here: {res['info'].get('gpu', {}).get('name')}  ({res['info'].get('gpu', {}).get('sms')} SMs)", '',
             f"{'variant':7s} {'size':9s} {'here us':>9s} {'@cfg':12s} {'regs':>4s} {'smem':>6s} {'occW':>4s} | "
             f"{'3070 us':>9s} {'@cfg':12s} {'regs':>4s} | {'3070/here':>9s}"]
    for v in vnames:
        for k in SIZES:
            h = best.get(v, {}).get(k)
            r = ref.get((v, k))
            if h is None:
                continue
            sp = (r['us'] / h['us']) if r else float('nan')
            lines.append(f"{v:7s} {k:9s} {h['us']:9.1f} {h['cfg']:12s} {h['regs']:4d} {h['smem']:6d} {h['occW']:4d} | "
                         + (f"{r['us']:9.1f} {r['cfg']:12s} {r['regs']:4d} | {sp:9.3f}" if r else f"{'-':>9s} {'':12s} {'':4s} | {'-':>9s}"))
    lines += ['', 'Same-config paired ratios (per-round t_X / t_Y, median over rounds; then min / median / max over launch '
                  'configs) here vs 3070 (ref/bench_3070/ratios_final.json)',
              f"{'pair':14s} {'size':9s} {'here min/med/max':26s} | {'3070 min/med/max':26s}"]
    here_r = {}
    for x, y in PAIRS:
        if x not in vnames or y not in vnames:
            continue
        for k in SIZES:
            rs = sorted(d[x]['timing'][k][f'ratio_vs_{y}'] for d in res['results'].values()
                        if x in d and y in d and f'ratio_vs_{y}' in d[x]['timing'][k])
            if not rs:
                continue
            here_r.setdefault(f'{x}/{y}', {})[k] = {'n': len(rs), 'min': rs[0], 'median': rs[len(rs) // 2], 'max': rs[-1]}
            q = rref.get(f'{x}/{y}', {}).get(k)
            hq = here_r[f'{x}/{y}'][k]
            lines.append(f"{x + '/' + y:14s} {k:9s} {hq['min']:.3f}/{hq['median']:.3f}/{hq['max']:.3f}{'':10s} | "
                         + (f"{q['min']:.3f}/{q['median']:.3f}/{q['max']:.3f}" if q else '-'))
    lines += ['', 'Spec view (WNR3 draft 11 T9; informative here, the gate is on KAIRA\'s kernel): inverse own-cost '
                  'CF5/CF0 <= 1.02, WNR3 CEIL vs WNR2 CF5/A259 <= 1.05 (median over configs):']
    for p_ in ('CF5/CF0', 'CF5/A259', 'CFD5/CFD0', 'CFDB2/CFDB20'):
        if p_ in here_r:
            lines.append('  %-13s ' % p_ + '  '.join('%s %.3f' % (k, here_r[p_][k]['median']) for k in SIZES if k in here_r[p_]))
    return '\n'.join(lines) + '\n', {'best_here': best, 'best_3070': {f'{a}|{b}': v for (a, b), v in ref.items()},
                                     'ratios_here': here_r}


if __name__ == '__main__':
    sys.exit(main())
