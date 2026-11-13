"""team5/gpu-v1: WNR3 v1 profile (CEIL | LENC | HASH) on the GPU: byte-exact checks, corruption checks, paired timing.

Usage: python -I v1bench.py V1_ENC_DIR T4_ENC_DIR OUT_JSON [rounds] [nts] [opts] [stages]
  stages: comma list of h32,scale,weight,neg,corrupt,lenc,timing,pipe (default all)

Baselines (team 4 ceil-gpu, unmodified kernels compiled from the copied source):
  A259 : team-2/4 wdec<16,1,0,NT,0> on team 4's original-form streams (var_A259.npz)
  CF5  : team-4 wdec<16,1,0,NT,5> on team 4's ceil-form streams (var_CF.npz, median-anchored in-sample model)
v1 family (this member; ceil-ref's held-out g0 model, anchor 120, 259-row index), kernel wdec3<NT,HASH,LENC,CEIL,DIAG>:
  G5   : CEIL only (x0 = L, u16 lengths -> host prefix sum), deferred 16-B scale patch, cs == 0 check
  G5H  : CEIL | HASH (flags 0x18 streams)
  G5L  : CEIL | LENC (same streams as G5; lengths from the bit-packed LENC section, in-kernel warp scan)
  G5HL : CEIL | LENC | HASH = v1 default (flags 0x1C streams)
Scale kernels: team-4 sdec<NT> (orig / ceil streams) and sdec3<NT,HASH,LENC,DUAL,DIAG>.
Original bytes: the sample .bin files (sha256 vs manifest), i.e. the safetensors tensor bytes.
"""
import hashlib, json, os, sys, time, subprocess
import numpy as np
import cupy as cp
from cupy_backends.cuda.api import driver as drv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SAMPLE = os.environ.get('WNR_SAMPLE', 'v41-sample')  # kit: dir with the 160-expert sample (manifest.json + L*_E*.bin)
NVSMI = os.environ.get('WNR_NVSMI', 'nvidia-smi')  # kit: was a DriverStore path of the original PC
_ARCH = os.environ.get('WNR_NVRTC_ARCH') or ('sm_' + str(cp.cuda.Device(0).compute_capability))  # kit: was hard-coded sm_86 (RTX 3070); sm_120 on the RTX 5090 needs NVRTC >= 12.8
OPTSETS = {'lb': ('-arch=' + _ARCH, '-std=c++17'), 'out4': ('-arch=' + _ARCH, '-std=c++17', '-DOUTMODE=4')}
OUT_PER_MATRIX = 5760 * 1024
SC_PER_MATRIX = 368640
SPM, SSPM, NGW, NGS = 5760, 720, 180, 23
FLAGS = {'G': 0x10, 'G14': 0x14, 'H18': 0x18, 'H1C': 0x1C}
# weight variants: name -> (kind, streams, HASH, LENC, CEIL, lenc section, ident flags)
WV = {'A259': ('t4', 'A259', 0, 0, 0, None, None),
      'CF5': ('t4', 'CF', 0, 0, 5, None, None),
      'CF5g': ('t4k', 'G', 0, 0, 5, None, None),        # team-4 CF5 kernel on this member's G streams (same bytes as G5)
      'G5': ('v1', 'G', 0, 0, 5, None, 'G'),
      'G5b': ('v1', 'G', 0, 0, 5, None, 'G'),           # A/A control: identical to G5
      'G5H': ('v1', 'H18', 1, 0, 5, None, 'H18'),
      'G5L': ('v1', 'G', 0, 1, 5, 'G14', 'G14'),
      'G5HL': ('v1', 'H1C', 1, 1, 5, 'H1C', 'H1C')}
# negative controls (correctness only): expected outcome in the comment
NEG = {'NC_inverse_off': ('v1', 'H1C', 1, 1, 0, 'H1C', 'H1C', 0),      # weights/scales wrong, hash passes (coded domain)
       'NC_hash_off': ('v1', 'H1C', 0, 1, 5, 'H1C', 'H1C', 0),         # every stream fails x == L
       'NC_hash_on_plain': ('v1', 'G', 1, 0, 5, None, 'H18', 0),       # x0 = L streams under the hash rule: all fail
       'NC_wrong_model_id': ('v1', 'H1C', 1, 1, 5, 'H1C', 'H1C', 1),   # ident model_id ^ 1: all fail
       'NC_wrong_flags_ident': ('v1', 'H1C', 1, 1, 5, 'H1C', 'H18', 0)}  # ident flags 0x18 on 0x1C streams: all fail
SV = {'S_t4o': ('t4', 'orig', 0, 0, 0), 'S_t4c': ('t4', 'ceil', 0, 0, 0),
      'SG': ('v1', 'G', 0, 0, 0), 'SH': ('v1', 'H18', 1, 0, 0), 'SL': ('v1', 'G', 0, 1, 0),
      'SHL': ('v1', 'H1C', 1, 1, 0), 'SHLd': ('v1', 'H1C', 1, 1, 1)}
SV['SGd'] = ('v1', 'G', 0, 0, 1)
SV['SLd'] = ('v1', 'G', 0, 1, 1)
# DUAL implementations: 2 = end-of-stream 16-B copy, 3 = 16-B stores (single output), 4 = 16-B stores to both
SV['SG2'] = ('v1', 'G', 0, 0, 2); SV['SG3'] = ('v1', 'G', 0, 0, 3); SV['SG4'] = ('v1', 'G', 0, 0, 4)
SV['SHL2'] = ('v1', 'H1C', 1, 1, 2); SV['SHL3'] = ('v1', 'H1C', 1, 1, 3); SV['SHL4'] = ('v1', 'H1C', 1, 1, 4)
SLENC = {'G': 'G14', 'H18': None, 'H1C': 'H1C'}
NULLP = np.uint64(0)   # null device pointer (8-byte kernel argument)


def nvsmi():
    try:
        return subprocess.run([NVSMI, '--query-gpu=clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu',
                               '--format=csv,noheader'], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa
        return str(e)


def wname(v, nt, diag=0, neg=None):
    kind, _, H, Lc, C = (NEG[neg][:5] if neg else WV[v][:5])
    if kind in ('t4', 't4k'):
        return f'wdec<16,1,0,{nt},{C}>'
    return f'wdec3<{nt},{H},{Lc},{C},{diag}>'


def sname(v, nt, diag=0):
    kind, _, H, Lc, D = SV[v]
    if kind == 't4':
        return f'sdec<{nt}>'
    return f'sdec3<{nt},{H},{Lc},{D},{diag}>'


def main():
    v1_dir, t4_dir, out_json = sys.argv[1:4]
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 31
    nts = [int(v) for v in sys.argv[5].split(',')] if len(sys.argv) > 5 else [32, 64, 128, 256]
    opts = sys.argv[6].split(',') if len(sys.argv) > 6 else ['lb', 'out4']
    stages = set(sys.argv[7].split(',')) if len(sys.argv) > 7 else {'h32', 'scale', 'weight', 'neg', 'corrupt', 'lenc', 'timing', 'pipe'}
    props = cp.cuda.runtime.getDeviceProperties(0)
    res = {'info': {'gpu': props['name'].decode(), 'sms': props['multiProcessorCount'], 'clock_khz_max': props['clockRate'],
                    'smem_per_sm': props['sharedMemPerMultiprocessor'], 'regs_per_sm': props['regsPerMultiprocessor'],
                    'driver': cp.cuda.runtime.driverGetVersion(), 'runtime': cp.cuda.runtime.runtimeGetVersion(),
                    'cupy': cp.__version__, 'rounds': R, 'nts': nts, 'opts': opts},
           'nvsmi_start': nvsmi()}
    print(json.dumps(res['info']), res['nvsmi_start'], flush=True)

    def dump():
        json.dump(res, open(out_json, 'w'), indent=1)

    # ------------------------------------------------------------------ data
    com = np.load(os.path.join(v1_dir, 'common_v1.npz'))
    nmat = com['mtype'].size
    nrec = nmat // 3
    nstreams = nmat * SPM
    names = [str(x) for x in com['names']]
    man = json.load(open(os.path.join(SAMPLE, 'manifest.json')))
    byfile = {e['file']: e for e in man['experts']}
    raw_l, sco_l, shaok = [], [], 0
    for fn_ in names:
        e = byfile[fn_]
        b = open(os.path.join(SAMPLE, fn_), 'rb').read()
        shaok += hashlib.sha256(b).hexdigest() == e['sha256']
        a = np.frombuffer(b, np.uint8)
        t, o = {}, 0
        for d in e['tensors']:
            t[d['name']] = a[o:o + d['bytes']]; o += d['bytes']
        for w in ('w1', 'w2', 'w3'):
            raw_l.append(t[w + '.weight']); sco_l.append(t[w + '.scale'])
    raw = np.concatenate(raw_l); sco = np.concatenate(sco_l)
    res['original_bytes'] = {'files': names, 'sha256_ok': int(shaok), 'eq_encoder_raw': bool((raw == com['raw']).all()),
                             'eq_encoder_scales': bool((sco == com['scales']).all())}
    t4c = np.load(os.path.join(t4_dir, 'common.npz'))
    res['original_bytes']['eq_team4_common_raw'] = bool(t4c['raw'].size == raw.size and (t4c['raw'] == raw).all())
    res['original_bytes']['eq_team4_common_scales'] = bool((t4c['scales'] == sco).all())
    print(res['original_bytes'], flush=True)
    assert shaok == nrec and res['original_bytes']['eq_encoder_raw'] and res['original_bytes']['eq_team4_common_raw']
    d_raw = cp.asarray(raw); d_sco = cp.asarray(sco)
    d_scc_v1 = cp.asarray(com['scales_coded']); d_scc_t4 = cp.asarray(t4c['scales_ceil'])
    d_mtype = cp.asarray(com['mtype'])
    d_med_t4 = {'orig': cp.asarray(t4c['med']), 'ceil': cp.asarray(t4c['med_ceil'])}
    model_id = int(com['model_id'])
    layer, expert = com['layer'], com['expert']
    d_tab_v1 = cp.asarray(com['tab259'].reshape(-1))
    d_scum_v1 = cp.asarray(com['scum'].reshape(-1)); d_slut_v1 = cp.asarray(com['slut'].reshape(-1))
    d_etab = cp.asarray(com['etab'].reshape(-1).astype(np.uint32))
    d_anchor120 = cp.asarray(com['anchor'])
    assert (com['anchor'] == 120).all()

    def padded(pay):
        return np.concatenate([pay, np.zeros((-pay.size) % 4 + 8192, np.uint8)])

    def prefix(lens):
        o = np.zeros(lens.size + 1, np.uint32); o[1:] = np.cumsum(lens.astype(np.int64)); return o
    # v1 streams
    ST = {}
    for v in ('G', 'H18', 'H1C'):
        z = np.load(os.path.join(v1_dir, f'str_{v}.npz'))
        wpay, spay = z['wpay'], z['spay']
        ST[v] = {'wpay_h': wpay, 'wlen': z['wlen'].astype(np.int64), 'slen': z['slen'].astype(np.int64),
                 'd_wpay': cp.asarray(padded(wpay).view(np.uint32)), 'd_spay': cp.asarray(padded(spay).view(np.uint32)),
                 'woffs': prefix(z['wlen']), 'soffs': prefix(z['slen']),
                 'wp_rec': z['wp_rec'], 'sp_rec': z['sp_rec'], 'wbytes': int(wpay.size), 'sbytes': int(spay.size)}
        ST[v]['d_woffs'] = cp.asarray(ST[v]['woffs']); ST[v]['d_soffs'] = cp.asarray(ST[v]['soffs'])
        assert ST[v]['woffs'][-1] == wpay.size and ST[v]['soffs'][-1] == spay.size
    LZ = {}
    for lv in ('G14', 'H1C'):
        z = np.load(os.path.join(v1_dir, f'lenc_{lv}.npz'))
        sec = z['sec']; sb = z['sec_bytes']
        assert (sb % 4 == 0).all()
        LZ[lv] = {'sec_h': sec.copy(), 'd_sec': cp.asarray(sec.view(np.uint32)), 'sec_bytes': sb,
                  'sec_word_off': np.concatenate([[0], np.cumsum(sb // 4)])[:-1]}

    def minfo(stv, lv, fl_key, mid_xor=0):
        s = ST[stv]
        wpo = np.concatenate([[0], np.cumsum(s['wp_rec'])])[:-1]; spo = np.concatenate([[0], np.cumsum(s['sp_rec'])])[:-1]
        mi = np.zeros((nmat, 16), np.uint32)
        for m in range(nmat):
            r, mm = divmod(m, 3)
            mi[m, 0] = (int(layer[r]) << 16) | int(expert[r])
            mi[m, 1] = FLAGS[fl_key] | (mm << 8) | (0 << 16) | (120 << 24)
            mi[m, 2] = (3 * SSPM + mm * SPM) | (1 << 31)
            mi[m, 3] = (mm * SSPM) | (1 << 31)
            mi[m, 4] = int(LZ[lv]['sec_word_off'][r]) if lv else 0
            mi[m, 5] = wpo[r]; mi[m, 6] = spo[r]; mi[m, 7] = s['wp_rec'][r]; mi[m, 8] = s['sp_rec'][r]
            mi[m, 9] = mm; mi[m, 10] = 120
            mi[m, 11] = int(LZ[lv]['sec_bytes'][r]) if lv else 0
        return cp.asarray(mi.reshape(-1))
    d_dummy_sec = cp.zeros(4, cp.uint32)
    # team 4 streams
    T4 = {}
    for vf in ('A259', 'CF'):
        z = np.load(os.path.join(t4_dir, f'var_{vf}.npz'))
        pay = z['payload']
        T4[vf] = {'d_pay': cp.asarray(padded(pay).view(np.uint32)), 'd_offs': cp.asarray(prefix(z['lengths'])),
                  'd_tabs': cp.asarray(z['tables'].reshape(-1))}
    T4S = {}
    for key in ('orig', 'ceil'):
        z = np.load(os.path.join(t4_dir, f'scl_{key}.npz'))
        T4S[key] = {'d_pay': cp.asarray(padded(z['payload']).view(np.uint32)), 'd_offs': cp.asarray(prefix(z['lengths'])),
                    'd_cum': cp.asarray(z['cum'].reshape(-1)), 'd_lut': cp.asarray(z['lut'].reshape(-1))}
    d_bt = cp.zeros(24, cp.uint16)
    d_nsig = cp.zeros(1, cp.uint32)

    # ------------------------------------------------------------------ compile
    src = open(os.path.join(HERE, 'wnr3dec.cu')).read()
    t0 = time.time()
    mods = {}
    for o in opts:
        nm = set()
        for nt in nts:
            for v in WV:
                nm.add(wname(v, nt))
            for n in NEG:
                nm.add(wname(None, nt, neg=n))
            for v in SV:
                nm.add(sname(v, nt))
        if o == 'lb':
            nm |= {wname('G5', 32, 1), wname('G5H', 32, 1), wname('G5HL', 32, 1), sname('SG', 32, 1), sname('SH', 32, 1),
                   sname('SHL', 32, 1), 'h32test'}
            nm |= {sname(v, 32) for v in SV}
        mods[o] = cp.RawModule(code=src, options=OPTSETS[o], name_expressions=sorted(nm))
        mods[o].compile()
    res['compile_s'] = time.time() - t0
    print('compile', res['compile_s'], flush=True)
    st = cp.cuda.Stream(non_blocking=True)
    sts = [cp.cuda.Stream(non_blocking=True) for _ in range(nrec)]
    d_err = cp.zeros(1, cp.int32)
    d_null = cp.zeros(1, cp.int32)

    # ------------------------------------------------------------------ H32 on the GPU vs spec vectors
    if 'h32' in stages:
        import h32 as H
        fn = mods['lb'].get_function('h32test')
        r_ = {}
        def gpu_h(words, seed):
            words = np.ascontiguousarray(words, np.uint32)
            n, nw = words.shape
            dw = cp.asarray(words.reshape(-1) if nw else np.zeros(1, np.uint32)); do = cp.zeros(n, cp.uint32)
            fn(((n + 127) // 128,), (128,), (dw, np.int32(n), np.int32(nw), np.uint32(seed), do))
            return do.get()
        r_['empty_seed0'] = f'0x{int(gpu_h(np.zeros((1, 0)), 0)[0]):08X}'
        r_['empty_seedWNR3'] = f'0x{int(gpu_h(np.zeros((1, 0)), H.SEED)[0]):08X}'
        r_['bytes00_0F'] = f'0x{int(gpu_h(np.frombuffer(bytes(range(16)), "<u4")[None], H.SEED)[0]):08X}'
        dec_w = bytes(((37 * i + 11) & 0x77) for i in range(1024))
        msg = H.ident_bytes(0xAB4261BD, 7, 191, 2165, 0, 0x0F, 1, 3, 127) + dec_w
        hw = int(gpu_h(np.frombuffer(msg, '<u4')[None], H.SEED)[0]) & H.MASK23
        r_['weight_vector_h23'] = f'0x{hw:06X}'
        rng = np.random.default_rng(7)
        Wm = rng.integers(0, 2 ** 32, (4096, 132), dtype=np.uint64).astype(np.uint32)
        r_['random_4096x528B_gpu_eq_numpy'] = int((gpu_h(Wm, H.SEED) == H.h32_words_vec(Wm, H.SEED)).sum())
        r_['pass'] = (r_['empty_seed0'] == '0x00000000' and r_['empty_seedWNR3'] == '0xE7F741F1' and
                      r_['bytes00_0F'] == '0x31CE0919' and hw == 0x0C48D3 and r_['random_4096x528B_gpu_eq_numpy'] == 4096)
        res['h32_gpu'] = r_
        print('h32 gpu', r_, flush=True)
        assert r_['pass']

    # ------------------------------------------------------------------ scale kernels
    d_cs = {k: cp.zeros(nmat * SC_PER_MATRIX, cp.uint8) for k in SV}
    d_cs2 = cp.zeros(nmat * SC_PER_MATRIX, cp.uint8)

    def slauncher(v, nt, out_buf, o='lb', diag=0, sterr=None, dg=None, out2=None, pay=None, sec=None):
        kind, sv, H, Lc, D = SV[v]
        fn = mods[o].get_function(sname(v, nt, diag))
        bpm = -(-SSPM // nt)
        if kind == 't4':
            s = T4S[sv]
            def launch(mc, m0=0):
                fn((mc * bpm,), (nt,), (s['d_pay'], s['d_offs'], s['d_cum'], s['d_lut'], d_mtype, np.int32(m0),
                                        np.int32(SSPM), np.int32(bpm), out_buf, d_err))
            return fn, launch
        s = ST[sv]
        lv = SLENC[sv] if Lc else None
        dsec = (LZ[lv]['d_sec'] if sec is None else sec) if lv else d_dummy_sec
        mi = minfo(sv, lv, sv if sv != 'G' else ('G14' if Lc else 'G'))
        dpay = s['d_spay'] if pay is None else pay
        o2 = out2 if out2 is not None else out_buf
        def launch(mc, m0=0):
            fn((mc * bpm,), (nt,), (dpay, s['d_soffs'], d_scum_v1, d_slut_v1, d_mtype, mi, dsec, np.uint32(model_id),
                                    np.int32(m0), np.int32(SSPM), np.int32(bpm), np.int32(NGS), np.int32(NGW),
                                    np.uint32(s['sbytes']),
                                    out_buf, o2, d_err, sterr if sterr is not None else NULLP,
                                    dg if dg is not None else NULLP))
        return fn, launch

    def timed(launchers, mcs=(1, 3, 24), multi=True):
        names_ = list(launchers)
        out = {}
        for mc in mcs:
            with st:
                tw = time.time()
                while time.time() - tw < 0.25:
                    for n in names_:
                        launchers[n](mc)
                    st.synchronize()
                T = np.zeros((R, len(names_)))
                evs = []
                for r in range(R):
                    order = [(r + k) % len(names_) for k in range(len(names_))]
                    row = []
                    for k in order:
                        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
                        e0.record(st); launchers[names_[k]](mc); e1.record(st)
                        row.append((k, e0, e1))
                    evs.append(row)
                st.synchronize()
                for r, row in enumerate(evs):
                    for k, e0, e1 in row:
                        T[r, k] = cp.cuda.get_elapsed_time(e0, e1) * 1000.0
            out[f'm{mc}'] = T
        if multi:
            T = np.zeros((R, len(names_)))

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
                order = [(r + k) % len(names_) for k in range(len(names_))]
                row = []
                for k in order:
                    e0, e1 = multi_run(lambda mc, m0, _k=k: launchers[names_[_k]](mc, m0))
                    row.append((k, e0, e1))
                st.synchronize()
                if r >= 4:
                    for k, e0, e1 in row:
                        T[r - 4, k] = cp.cuda.get_elapsed_time(e0, e1) * 1000.0
            out[f'{nrec}streams'] = T
        summ = {}
        for key, T in out.items():
            for i, n in enumerate(names_):
                d = summ.setdefault(n, {})
                d[key] = {'us_median': float(np.median(T[:, i])), 'us_min': float(T[:, i].min()),
                          'us_p90': float(np.percentile(T[:, i], 90))}
                for j, b in enumerate(names_):
                    if b != n:
                        rr = T[:, i] / T[:, j]
                        d[key][f'ratio_vs_{b}'] = float(np.median(rr))
                        d[key][f'ratio_vs_{b}_q25_q75'] = [float(np.percentile(rr, 25)), float(np.percentile(rr, 75))]
        return summ

    if 'scale' in stages:
        res['scale_decode'] = {}
        sref = {'S_t4o': d_sco, 'S_t4c': d_scc_t4}
        for nt in nts:
            L_ = {}
            for v in SV:
                fn, launch = slauncher(v, nt, d_cs[v], out2=d_cs2 if SV[v][4] else None)
                d_cs[v].fill(0); d_cs2.fill(0); d_err.fill(0)
                cp.cuda.runtime.deviceSynchronize()
                with st:
                    launch(nmat)
                st.synchronize()
                ref = sref.get(v, d_scc_v1)
                bad = int((d_cs[v].reshape(-1, 512) != ref.reshape(-1, 512)).any(1).sum())
                bad2 = int((d_cs2.reshape(-1, 512) != ref.reshape(-1, 512)).any(1).sum()) if SV[v][4] in (1, 2, 4) else 0
                errs = int(d_err.get()[0])
                # partial launches (record 1 only) and 8 launches on 8 streams
                d_cs[v].fill(0); d_err.fill(0); cp.cuda.runtime.deviceSynchronize()
                with st:
                    launch(3, 3)
                st.synchronize()
                badp = int((d_cs[v][3 * SC_PER_MATRIX:6 * SC_PER_MATRIX].reshape(-1, 512) != ref[3 * SC_PER_MATRIX:6 * SC_PER_MATRIX].reshape(-1, 512)).any(1).sum())
                errp = int(d_err.get()[0])
                d_cs[v].fill(0); d_err.fill(0); cp.cuda.runtime.deviceSynchronize()
                for ri, s_ in enumerate(sts):
                    with s_:
                        launch(3, 3 * ri)
                cp.cuda.runtime.deviceSynchronize()
                badm = int((d_cs[v].reshape(-1, 512) != ref.reshape(-1, 512)).any(1).sum())
                errm = int(d_err.get()[0])
                r = {'kernel': sname(v, nt), 'nt': nt, 'regs': fn.num_regs, 'local_bytes': fn.local_size_bytes,
                     'static_smem': fn.shared_size_bytes,
                     'occ_blocks_per_sm': drv.occupancyMaxActiveBlocksPerMultiprocessor(fn.kernel.ptr, nt, 0),
                     'bad_streams': bad, 'bad_streams_out2': bad2, 'errors': errs, 'rec1_bad': badp, 'rec1_err': errp,
                     'multi_bad': badm, 'multi_err': errm,
                     'byte_exact': bad == 0 and bad2 == 0 and errs == 0 and badp == 0 and errp == 0 and badm == 0 and errm == 0,
                     'streams': nmat * SSPM}
                res['scale_decode'][f'{v}/nt{nt}'] = r
                print('sdec', v, nt, r, flush=True)
                L_[v] = launch
            if 'timing' in stages:
                tm = timed(L_)
                res.setdefault('scale_timing', {})[f'nt{nt}'] = tm
                for v in tm:
                    print('sdec timing', nt, v, {k: round(tm[v][k]['us_median'], 1) for k in tm[v]},
                          {k: round(tm[v][k].get('ratio_vs_SG', 1), 4) for k in tm[v]}, flush=True)
        assert all(res['scale_decode'][k]['byte_exact'] for k in res['scale_decode'])
        dump()
    # coded scales used by the weight kernels: GPU-decoded (verified above, or decoded now)
    for v, key in (('S_t4o', 'orig'), ('S_t4c', 'ceil'), ('SHL', 'v1')):
        _, launch = slauncher(v, 32, d_cs[v])
        d_err.fill(0); cp.cuda.runtime.deviceSynchronize()
        with st:
            launch(nmat)
        st.synchronize()
        assert int(d_err.get()[0]) == 0
    assert bool((d_cs['S_t4o'] == d_sco).all()) and bool((d_cs['S_t4c'] == d_scc_t4).all()) and bool((d_cs['SHL'] == d_scc_v1).all())
    d_scin = {'t4o': d_cs['S_t4o'], 't4c': d_cs['S_t4c'], 'v1': d_cs['SHL']}

    # ------------------------------------------------------------------ weight kernels
    d_out = cp.zeros(nstreams * 1024, cp.uint8)
    d_osc = cp.zeros(nmat * SC_PER_MATRIX, cp.uint8)
    d_sterr = cp.zeros(nstreams, cp.int32)
    d_diag = cp.zeros(2 * nstreams, cp.uint32)

    def wlauncher(spec, nt, o, diag=0, sterr=None, dg=None, pay=None, sec=None, sc_in=None, osc=None, name=None):
        kind, sv, H, Lc, C, lv, flk = spec[:7]
        mx = spec[7] if len(spec) > 7 else 0
        fn = mods[o].get_function(name)
        bpm = -(-SPM // nt)
        assert SPM % 32 == 0
        osc_ = d_osc if osc is None else osc
        if kind == 't4':
            dp = T4[sv]
            smem = 16 + 259 * 16 * 2 + (nt * 144 if o == 'out4' else 0)
            sc_ = d_scin['t4o' if C == 0 else 't4c'] if sc_in is None else sc_in
            med = d_med_t4['orig' if C == 0 else 'ceil']
            def launch(mc, m0=0):
                fn((mc * bpm,), (nt,), (dp['d_pay'], dp['d_offs'], sc_, d_bt, dp['d_tabs'], d_bt, d_mtype, med,
                                        np.int32(m0), np.int32(SPM), np.int32(bpm), d_out, d_err, osc_), shared_mem=smem)
            return fn, smem, launch
        if kind == 't4k':
            # team-4 kernel (unmodified code) on this member's streams, tables and constant anchor 120
            s = ST[sv]
            smem = 16 + 259 * 16 * 2 + (nt * 144 if o == 'out4' else 0)
            sc_ = d_scin['v1'] if sc_in is None else sc_in
            dpay = s['d_wpay'] if pay is None else pay

            def launch(mc, m0=0):
                fn((mc * bpm,), (nt,), (dpay, s['d_woffs'], sc_, d_bt, d_tab_v1, d_bt, d_mtype, d_anchor120,
                                        np.int32(m0), np.int32(SPM), np.int32(bpm), d_out, d_err, osc_), shared_mem=smem)
            return fn, smem, launch
        s = ST[sv]
        smem = 16 + 259 * 16 * 2 + (112 if Lc else 0) + (nt * 144 if o == 'out4' else 0)
        dsec = (LZ[lv]['d_sec'] if sec is None else sec) if Lc else d_dummy_sec
        mi = minfo(sv, lv if Lc else None, flk)
        dpay = s['d_wpay'] if pay is None else pay
        sc_ = d_scin['v1'] if sc_in is None else sc_in
        mid = model_id ^ mx

        def launch(mc, m0=0):
            fn((mc * bpm,), (nt,), (dpay, s['d_woffs'], sc_, d_tab_v1, d_mtype, mi, dsec, d_etab, np.uint32(mid),
                                    np.int32(m0), np.int32(SPM), np.int32(bpm), np.int32(NGS), np.int32(NGW),
                                    np.uint32(s['wbytes']), d_out, d_err, osc_,
                                    sterr if sterr is not None else NULLP, dg if dg is not None else NULLP), shared_mem=smem)
        return fn, smem, launch

    def prep_out(C, sc_src):
        d_out.fill(0); d_err.fill(0)
        if C == 5:
            d_osc[:] = sc_src            # mask form: output scales pre-filled with the coded scales
        else:
            d_osc.fill(0xAA)
        cp.cuda.runtime.deviceSynchronize()

    def check(C, sc_src, n=None):
        n = nstreams if n is None else n
        badw = int((d_out[:n * 1024].reshape(n, 1024) != d_raw[:n * 1024].reshape(n, 1024)).any(1).sum())
        slab = d_osc if C == 5 else sc_src
        bads = int((slab[:n * 64].reshape(n, 64) != d_sco[:n * 64].reshape(n, 64)).any(1).sum())
        return badw, bads, int(d_err.get()[0])

    def scsrc(spec):
        return d_scin['v1'] if spec[0] in ('v1', 't4k') else d_scin['t4o' if spec[4] == 0 else 't4c']

    if 'weight' in stages or 'timing' in stages or 'neg' in stages:
        res['weight'] = {}
        for o in opts:
            for nt in nts:
                cfg = f'nt{nt}/{o}'
                info, L_ = {}, {}
                for v in WV:
                    spec = WV[v]
                    fn, smem, launch = wlauncher(spec, nt, o, name=wname(v, nt))
                    prep_out(spec[4], scsrc(spec))
                    with st:
                        launch(nmat)
                    st.synchronize()
                    badw, bads, errs = check(spec[4], scsrc(spec))
                    occ = drv.occupancyMaxActiveBlocksPerMultiprocessor(fn.kernel.ptr, nt, smem)
                    info[v] = {'kernel': wname(v, nt), 'regs': fn.num_regs, 'local_bytes': fn.local_size_bytes,
                               'smem_per_block': smem, 'occ_blocks_per_sm': occ, 'occ_warps_per_sm': occ * nt // 32,
                               'weight_bad_streams': badw, 'scale_bad_streams': bads, 'errors': errs,
                               'byte_exact_vs_original': badw == 0 and bads == 0 and errs == 0}
                    # 8 launches on 8 streams, and partial launches (1 matrix, 1 record)
                    prep_out(spec[4], scsrc(spec))
                    e0 = cp.cuda.Event(); e0.record(st)
                    for ri, s_ in enumerate(sts):
                        s_.wait_event(e0)
                        with s_:
                            launch(3, 3 * ri)
                    cp.cuda.runtime.deviceSynchronize()
                    bw, bs, er = check(spec[4], scsrc(spec))
                    info[v]['multistream_recheck'] = {'weight_bad_streams': bw, 'scale_bad_streams': bs, 'errors': er}
                    for mc in (1, 3):
                        prep_out(spec[4], scsrc(spec))
                        with st:
                            launch(mc)
                        st.synchronize()
                        bw, bs, er = check(spec[4], scsrc(spec), mc * SPM)
                        info[v][f'm{mc}_recheck'] = {'weight_bad_streams': bw, 'scale_bad_streams': bs, 'errors': er}
                    info[v]['all_exact'] = (info[v]['byte_exact_vs_original'] and
                                            all(info[v][k]['weight_bad_streams'] == 0 and info[v][k]['scale_bad_streams'] == 0 and info[v][k]['errors'] == 0
                                                for k in ('multistream_recheck', 'm1_recheck', 'm3_recheck')))
                    L_[v] = launch
                    print(cfg, v, {k: info[v][k] for k in ('regs', 'local_bytes', 'smem_per_block', 'occ_warps_per_sm', 'all_exact')}, flush=True)
                if 'neg' in stages:
                    for n in NEG:
                        spec = NEG[n]
                        fn, smem, launch = wlauncher(spec, nt, o, name=wname(None, nt, neg=n))
                        prep_out(spec[4], scsrc(spec))
                        with st:
                            launch(nmat)
                        st.synchronize()
                        badw, bads, errs = check(spec[4], scsrc(spec))
                        info[n] = {'kernel': wname(None, nt, neg=n), 'weight_bad_streams': badw, 'scale_bad_streams': bads,
                                   'errors': errs, 'streams': nstreams}
                        print(cfg, n, info[n], flush=True)
                if 'timing' in stages:
                    tm = timed(L_)
                    for v in WV:
                        info[v]['timing'] = tm[v]
                    for v in WV:
                        t = tm[v]
                        print(f"{cfg:10s} {v:5s} " + ' '.join(
                            f"{k} {t[k]['us_median']:8.1f} /G5 {t[k].get('ratio_vs_G5', 1):.4f} /CF5 {t[k].get('ratio_vs_CF5', 1):.4f} /A {t[k].get('ratio_vs_A259', 1):.4f}"
                            for k in t), flush=True)
                res['weight'][cfg] = info
                dump()
        ok = all(res['weight'][c][v]['all_exact'] for c in res['weight'] for v in WV)
        res['weight_all_exact'] = ok
        # the coded-scale inputs are never written by the weight kernels (idempotent re-decode, spec 7.2 item 4)
        res['coded_scale_inputs_unchanged_after_all_runs'] = {
            'v1': bool((d_scin['v1'] == d_scc_v1).all()), 't4c': bool((d_scin['t4c'] == d_scc_t4).all()),
            't4o': bool((d_scin['t4o'] == d_sco).all())}
        # retry idempotence: decode the full set twice into the same outputs without clearing; compare with the original
        for v in ('G5HL',):
            spec = WV[v]
            _, smem, launch = wlauncher(spec, 32, 'lb', name=wname(v, 32))
            prep_out(spec[4], scsrc(spec))
            with st:
                launch(nmat); launch(nmat)
            st.synchronize()
            bw, bs, er = check(spec[4], scsrc(spec))
            res['retry_idempotence_G5HL_nt32'] = {'weight_bad_streams': bw, 'scale_bad_streams': bs, 'errors': er}
        print('weight all exact', ok, res['coded_scale_inputs_unchanged_after_all_runs'], res['retry_idempotence_G5HL_nt32'], flush=True)

    # ------------------------------------------------------------------ corruption (per-stream GPU status)
    if 'corrupt' in stages:
        rng = np.random.default_rng(20261107)
        cres = {}
        NSEED = 10
        for v in ('G5', 'G5H', 'G5HL'):
            spec = WV[v]
            s = ST[spec[1]]
            fn, smem, launch = wlauncher(spec, 32, 'lb', diag=1, sterr=d_sterr, dg=d_diag, name=wname(v, 32, 1))
            # clean run: diag gives the encoder's start state (final state of a correct decode)
            prep_out(5, d_scin['v1']); d_sterr.fill(0)
            with st:
                launch(nmat)
            st.synchronize()
            dg = d_diag.get().reshape(-1, 2)
            assert int(d_err.get()[0]) == 0 and (dg[:, 0] == dg[:, 1]).all()
            x_clean = cp.asarray(dg[:, 0])
            offs = s['woffs'][:-1].astype(np.int64); lens = s['wlen']
            d_pc = cp.empty_like(s['d_wpay'])
            out_v = {}
            for case in ('A_random_bit', 'B_last_byte'):
                agg = {'trials': 0, 'flagged': 0, 'accepted_exact': 0, 'accepted_wrong': 0,
                       'structural_pass': 0, 'structural_pass_but_hash_reject': 0, 'status_bits_hist': {},
                       'accepted_wrong_sign_only_one_nibble': 0, 'accepted_wrong_other': 0,
                       'structural_pass_output_wrong': 0, 'structural_pass_one_sign_nibble': 0}
                for sd in range(NSEED):
                    if case == 'A_random_bit':
                        pos = offs + (rng.random(nstreams) * lens).astype(np.int64)
                    else:
                        pos = offs + lens - 1
                    bit = rng.integers(0, 8, nstreams)
                    d_pc[:] = s['d_wpay']
                    p8 = d_pc.view(cp.uint8)
                    dpos = cp.asarray(pos); p8[dpos] ^= cp.asarray((1 << bit).astype(np.uint8))
                    prep_out(5, d_scin['v1']); d_sterr.fill(0)
                    _, _, lc = wlauncher(spec, 32, 'lb', diag=1, sterr=d_sterr, dg=d_diag, pay=d_pc, name=wname(v, 32, 1))
                    with st:
                        lc(nmat)
                    st.synchronize()
                    se = d_sterr
                    flagged = se != 0
                    outw = (d_out.reshape(nstreams, 1024) != d_raw.reshape(nstreams, 1024)).any(1)
                    outs = (d_osc.reshape(nstreams, 64) != d_sco.reshape(nstreams, 64)).any(1)
                    wrong = outw | outs
                    dgc = d_diag.reshape(-1, 2)
                    struct_ok = (dgc[:, 0] == x_clean) & ((se & 3) == 0)
                    agg['trials'] += nstreams
                    agg['flagged'] += int(flagged.sum())
                    agg['accepted_exact'] += int((~flagged & ~wrong).sum())
                    agg['accepted_wrong'] += int((~flagged & wrong).sum())
                    agg['structural_pass'] += int(struct_ok.sum())
                    mtyp = (cp.arange(nstreams) // SPM) % 3
                    for mm_, mn_ in enumerate(('w1', 'w2', 'w3')):
                        sel = mtyp == mm_
                        bm = agg.setdefault('by_matrix', {}).setdefault(mn_, {'trials': 0, 'accepted_wrong': 0, 'structural_pass': 0,
                                                                               'structural_pass_but_hash_reject': 0})
                        bm['trials'] += int(sel.sum())
                        bm['accepted_wrong'] += int((sel & ~flagged & wrong).sum())
                        bm['structural_pass'] += int((sel & struct_ok).sum())
                        bm['structural_pass_but_hash_reject'] += int((sel & struct_ok & flagged).sum())
                    agg['structural_pass_but_hash_reject'] += int((struct_ok & flagged).sum())
                    agg['structural_pass_output_wrong'] += int((struct_ok & wrong).sum())
                    hb = np.bincount(se.get(), minlength=32)
                    for b_, c_ in enumerate(hb):
                        if c_:
                            agg['status_bits_hist'][str(b_)] = agg['status_bits_hist'].get(str(b_), 0) + int(c_)
                    # nibble analysis of streams whose structure passed with wrong output (the sign-flip class)
                    idx = cp.flatnonzero(struct_ok & wrong)
                    if idx.size:
                        dx = d_out.reshape(nstreams, 1024)[idx] ^ d_raw.reshape(nstreams, 1024)[idx]
                        lo, hi = dx & 15, dx >> 4
                        nnz = (lo != 0).sum(1) + (hi != 0).sum(1)
                        sgn = (((lo == 0) | (lo == 8)).all(1)) & (((hi == 0) | (hi == 8)).all(1))
                        one_sign = (nnz == 1) & sgn & ~outs[idx]
                        fl = flagged[idx]
                        agg['structural_pass_one_sign_nibble'] += int(one_sign.sum())
                        agg['accepted_wrong_sign_only_one_nibble'] += int((one_sign & ~fl).sum())
                        agg['accepted_wrong_other'] += int((~one_sign & ~fl).sum())
                        agg.setdefault('structural_pass_wrong_nibbles_hist', {})
                        hh = np.bincount(np.minimum(nnz.get(), 64), minlength=65)
                        for b_, c_ in enumerate(hh):
                            if c_:
                                k_ = str(b_)
                                agg['structural_pass_wrong_nibbles_hist'][k_] = agg['structural_pass_wrong_nibbles_hist'].get(k_, 0) + int(c_)
                agg['accepted_wrong_rate'] = agg['accepted_wrong'] / agg['trials']
                agg['structural_pass_rate'] = agg['structural_pass'] / agg['trials']
                out_v[case] = agg
                print('corrupt', v, case, agg, flush=True)
            cres[v] = out_v
            dump()
        # scale streams
        for v in ('SG', 'SH', 'SHL'):
            sv = SV[v][1]
            s = ST[sv]
            d_sst = cp.zeros(nmat * SSPM, cp.int32); d_sdg = cp.zeros(2 * nmat * SSPM, cp.uint32)
            _, launch = slauncher(v, 32, d_cs2, diag=1, sterr=d_sst, dg=d_sdg)
            d_err.fill(0); cp.cuda.runtime.deviceSynchronize()
            with st:
                launch(nmat)
            st.synchronize()
            dg = d_sdg.get().reshape(-1, 2)
            assert int(d_err.get()[0]) == 0 and (dg[:, 0] == dg[:, 1]).all()
            x_clean = cp.asarray(dg[:, 0])
            offs = s['soffs'][:-1].astype(np.int64); lens = s['slen']
            ns = nmat * SSPM
            d_pc = cp.empty_like(s['d_spay'])
            out_v = {}
            for case in ('A_random_bit', 'B_last_byte'):
                agg = {'trials': 0, 'flagged': 0, 'accepted_exact': 0, 'accepted_wrong': 0, 'structural_pass': 0,
                       'structural_pass_but_hash_reject': 0, 'structural_pass_output_wrong': 0}
                for sd in range(NSEED):
                    pos = offs + ((rng.random(ns) * lens).astype(np.int64) if case == 'A_random_bit' else lens - 1)
                    bit = rng.integers(0, 8, ns)
                    d_pc[:] = s['d_spay']
                    p8 = d_pc.view(cp.uint8); p8[cp.asarray(pos)] ^= cp.asarray((1 << bit).astype(np.uint8))
                    d_sst.fill(0); d_err.fill(0); cp.cuda.runtime.deviceSynchronize()
                    _, lc = slauncher(v, 32, d_cs2, diag=1, sterr=d_sst, dg=d_sdg, pay=d_pc)
                    with st:
                        lc(nmat)
                    st.synchronize()
                    flagged = d_sst != 0
                    wrong = (d_cs2.reshape(ns, 512) != d_scc_v1.reshape(ns, 512)).any(1)
                    dgc = d_sdg.reshape(-1, 2)
                    struct_ok = (dgc[:, 0] == x_clean) & ((d_sst & 3) == 0)
                    agg['trials'] += ns
                    agg['flagged'] += int(flagged.sum())
                    agg['accepted_exact'] += int((~flagged & ~wrong).sum())
                    agg['accepted_wrong'] += int((~flagged & wrong).sum())
                    agg['structural_pass'] += int(struct_ok.sum())
                    agg['structural_pass_but_hash_reject'] += int((struct_ok & flagged).sum())
                    agg['structural_pass_output_wrong'] += int((struct_ok & wrong).sum())
                agg['accepted_wrong_rate'] = agg['accepted_wrong'] / agg['trials']
                out_v[case] = agg
                print('corrupt', v, case, agg, flush=True)
            cres[v] = out_v
        res['corruption'] = cres
        dump()

    # ------------------------------------------------------------------ LENC section corruption (one record per trial)
    if 'lenc' in stages:
      lres_all = {}
      for lv, wv_, sv_ in (('H1C', 'G5HL', 'SHLd'), ('G14', 'G5L', 'SLd')):
        rng = np.random.default_rng(7)
        lres = {}
        NT_ = 100      # trials per (record, field)
        secb = LZ[lv]['sec_bytes']; swo = LZ[lv]['sec_word_off']
        d_sec_c = cp.empty_like(LZ[lv]['d_sec'])
        spec = WV[wv_]
        d_sst = cp.zeros(nmat * SSPM, cp.int32)
        fields = {}
        for r in range(nrec):
            base = int(swo[r]) * 4
            sec = LZ[lv]['sec_h'][base:base + int(secb[r])]
            nsg, nwg = 69, 540
            sp0 = 12
            sws = int(sec[sp0 + 4 * nsg: sp0 + 4 * nsg + nsg].astype(np.int64).sum())
            spl = 4 * nsg + nsg + (-nsg % 4) + 4 * sws
            wp0 = sp0 + spl
            bnd = [('header_sbase_alpha', 0, 12), ('s_base_off', sp0, sp0 + 4 * nsg), ('s_width', sp0 + 4 * nsg, sp0 + 5 * nsg),
                   ('s_pad', sp0 + 5 * nsg, sp0 + 5 * nsg + (-nsg % 4)), ('s_words', sp0 + 5 * nsg + (-nsg % 4), wp0),
                   ('w_base_off', wp0, wp0 + 4 * nwg), ('w_width', wp0 + 4 * nwg, wp0 + 5 * nwg),
                   ('w_words', wp0 + 5 * nwg + (-nwg % 4), int(secb[r]))]
            fields[r] = [f for f in bnd if f[2] > f[1]]
        stat = {}
        _, slaunch = slauncher(sv_, 32, d_cs2, out2=d_osc, sec=d_sec_c, sterr=d_sst)
        fn, smem, wl = wlauncher(spec, 32, 'lb', sec=d_sec_c, sc_in=d_cs2, sterr=d_sterr, name=wname(wv_, 32))
        t0 = time.time()
        # clean run of every record first (must be exact, 0 errors)
        clean_ok = 0
        for r in range(nrec):
            d_sec_c[:] = LZ[lv]['d_sec']
            d_cs2.fill(0); d_out.fill(0); d_err.fill(0); d_osc.fill(0xAA); cp.cuda.runtime.deviceSynchronize()
            with st:
                slaunch(3, 3 * r); wl(3, 3 * r)
            st.synchronize()
            a0, a1 = 3 * r * SPM, 3 * (r + 1) * SPM
            clean_ok += int(int(d_err.get()[0]) == 0 and bool((d_out[a0 * 1024:a1 * 1024] == d_raw[a0 * 1024:a1 * 1024]).all())
                            and bool((d_osc[a0 * 64:a1 * 64] == d_sco[a0 * 64:a1 * 64]).all()))
        lres['clean_records_exact'] = clean_ok
        for r in range(nrec):
            base = int(swo[r]) * 4
            for fld, fa, fb in fields[r]:
                for tr in range(NT_):
                    pos = base + int(rng.integers(fa, fb))
                    bit = int(rng.integers(0, 8))
                    d_sec_c[:] = LZ[lv]['d_sec']
                    c8 = d_sec_c.view(cp.uint8)
                    c8[pos:pos + 1] ^= np.uint8(1 << bit)
                    d_cs2.fill(0); d_out.fill(0); d_err.fill(0); d_osc.fill(0xAA); d_sterr.fill(0); d_sst.fill(0)
                    cp.cuda.runtime.deviceSynchronize()
                    with st:
                        slaunch(3, 3 * r)
                        wl(3, 3 * r)
                    st.synchronize()
                    e = int(d_err.get()[0])
                    a0, a1 = 3 * r * SPM, 3 * (r + 1) * SPM
                    exact = bool((d_out[a0 * 1024:a1 * 1024] == d_raw[a0 * 1024:a1 * 1024]).all()) and \
                        bool((d_osc[a0 * 64:a1 * 64] == d_sco[a0 * 64:a1 * 64]).all()) and \
                        bool((d_cs2[3 * r * SC_PER_MATRIX:3 * (r + 1) * SC_PER_MATRIX] == d_scc_v1[3 * r * SC_PER_MATRIX:3 * (r + 1) * SC_PER_MATRIX]).all())
                    d = stat.setdefault(fld, {'trials': 0, 'rejected': 0, 'accepted_exact': 0, 'accepted_wrong': 0,
                                              'weight_status_or_hist': {}, 'scale_status_or_hist': {},
                                              'rejected_by_lenc_bit_any': 0})
                    d['trials'] += 1
                    def bor(a):
                        return int(sum(1 << b for b in range(5) if bool(((a >> b) & 1).any())))
                    wor = bor(d_sterr[a0:a1]) if e else 0
                    sor = bor(d_sst[3 * r * SSPM:3 * (r + 1) * SSPM]) if e else 0
                    if e:
                        d['rejected'] += 1
                        d['weight_status_or_hist'][str(wor)] = d['weight_status_or_hist'].get(str(wor), 0) + 1
                        d['scale_status_or_hist'][str(sor)] = d['scale_status_or_hist'].get(str(sor), 0) + 1
                        d['rejected_by_lenc_bit_any'] += int(((wor | sor) & 8) != 0 or (wor | sor) == 0)
                    elif exact:
                        d['accepted_exact'] += 1
                    else:
                        d['accepted_wrong'] += 1
        lres['by_field'] = stat
        lres['trials'] = int(sum(v['trials'] for v in stat.values()))
        lres['seconds'] = time.time() - t0
        lres['variant'] = f'{wv_} + {sv_}, section {lv}'
        lres_all[lv] = lres
        print('lenc corruption', json.dumps(lres), flush=True)
        res['lenc_corruption'] = lres_all
        dump()

    # ------------------------------------------------------------------ pipelines (separate output region)
    if 'pipe' in stages:
        pres = {}
        d_pcs = cp.zeros(nmat * SC_PER_MATRIX, cp.uint8)
        d_posc = cp.zeros(nmat * SC_PER_MATRIX, cp.uint8)
        for o in opts:
            for nt in nts:
                cfg = f'nt{nt}/{o}'
                P_ = {}
                # A259: scales decoded straight into the output (no ceil), weights read them
                _, sl_o = slauncher('S_t4o', 32, d_pcs)
                _, _, wl_a = wlauncher(WV['A259'], nt, o, sc_in=d_pcs, osc=d_posc, name=wname('A259', nt))
                P_['P_A259'] = lambda mc, m0=0, _s=sl_o, _w=wl_a: (_s(mc, m0), _w(mc, m0))
                # CF5: coded scales -> d_pcs, async copy to the output region, CF5 patches the output region
                _, sl_c = slauncher('S_t4c', 32, d_pcs)
                _, _, wl_c = wlauncher(WV['CF5'], nt, o, sc_in=d_pcs, osc=d_posc, name=wname('CF5', nt))

                def p_cf5(mc, m0=0, _s=sl_c, _w=wl_c):
                    _s(mc, m0)
                    cs_ = cp.cuda.get_current_stream()
                    cp.cuda.runtime.memcpyAsync(d_posc.data.ptr + m0 * SC_PER_MATRIX, d_pcs.data.ptr + m0 * SC_PER_MATRIX,
                                                mc * SC_PER_MATRIX, 3, cs_.ptr)
                    _w(mc, m0)
                P_['P_CF5'] = p_cf5
                # v1 family, form (b) two ways: (c) scale kernel -> async copy -> weight kernel, as P_CF5;
                # (d) the scale kernel writes the coded-scale buffer AND the output region (DUAL 4, 16-B stores)
                for v, sv, pn in (('G5', 'SG', 'P_G5c'), ('G5HL', 'SHL', 'P_G5HLc')):
                    _, slx = slauncher(sv, 32, d_pcs)
                    _, _, wlx = wlauncher(WV[v], nt, o, sc_in=d_pcs, osc=d_posc, name=wname(v, nt))

                    def p_c(mc, m0=0, _s=slx, _w=wlx):
                        _s(mc, m0)
                        cs_ = cp.cuda.get_current_stream()
                        cp.cuda.runtime.memcpyAsync(d_posc.data.ptr + m0 * SC_PER_MATRIX, d_pcs.data.ptr + m0 * SC_PER_MATRIX,
                                                    mc * SC_PER_MATRIX, 3, cs_.ptr)
                        _w(mc, m0)
                    P_[pn] = p_c
                for v, sv, pn in (('G5', 'SG4', 'P_G5d'), ('G5HL', 'SHL4', 'P_G5HLd')):
                    _, slx = slauncher(sv, 32, d_pcs, out2=d_posc)
                    _, _, wlx = wlauncher(WV[v], nt, o, sc_in=d_pcs, osc=d_posc, name=wname(v, nt))
                    P_[pn] = lambda mc, m0=0, _s=slx, _w=wlx: (_s(mc, m0), _w(mc, m0))
                chk = {}
                for pn, pl in P_.items():
                    for mode in ('single', 'multi'):
                        d_out.fill(0); d_pcs.fill(0); d_posc.fill(0xAA); d_err.fill(0)
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
                        slab = d_pcs if pn == 'P_A259' else d_posc
                        bads = int((slab.reshape(nstreams, 64) != d_sco.reshape(nstreams, 64)).any(1).sum())
                        chk[f'{pn}/{mode}'] = {'weight_bad_streams': badw, 'scale_bad_streams': bads, 'errors': int(d_err.get()[0])}
                print(cfg, 'pipe checks', chk, flush=True)
                tp = timed(P_) if 'timing' in stages else None
                pres[cfg] = {'checks': chk, 'timing': tp}
                if tp:
                    for pn in tp:
                        print(f"{cfg:10s} {pn:12s} " + ' '.join(f"{k} {tp[pn][k]['us_median']:8.1f} /P_G5 {tp[pn][k].get('ratio_vs_P_G5', 1):.4f} /P_CF5 {tp[pn][k].get('ratio_vs_P_CF5', 1):.4f}" for k in tp[pn]), flush=True)
                res['pipeline'] = pres
                dump()
    res['nvsmi_end'] = nvsmi()
    dump()
    print('done', res['nvsmi_end'])


if __name__ == '__main__':
    main()
