"""team6/signplane-z: GPU check + paired timing of the compacted-sign-plane kernel (wdecz) against team 4's D, CFD5, CF5,
A259 kernels (copied verbatim in wnr2dec_z.cu), on team 4's ceil-gpu encoded data (8 experts, in-sample models, read-only).

Z streams are built here from team 4's CFD streams (ceil-form magnitudes, x0 = L, no hash) + the compacted plane of the
nonzero signs appended in descending address order (signplane.assemble_streams). Expected Z output = canonical raw
(-0 -> +0); scales through the deferred patch (CEIL 5) must equal the ORIGINAL scale bytes.

Usage: python -I zbench.py ENC_DIR OUT_JSON [rounds] [nts] [opts] [variants]
GPU guard: before every launch config the script lists compute apps; any python/other compute process that is not this
process (and not one of the desktop C+G apps present at start) marks the config as contaminated; utilization.gpu > 5 %
before start aborts.
"""
import json, os, subprocess, sys, time
import numpy as np
import cupy as cp
from cupy_backends.cuda.api import driver as drv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import signplane as Z  # noqa: E402

NVSMI = os.environ.get('WNR_NVSMI', 'nvidia-smi')  # kit: was a DriverStore path of the original PC
# name: (data, ALPHA, CEIL, scale set, kernel family)
VARS = {'A259': ('A259', 16, 0, 'orig', 'wdec'), 'D': ('D', 8, 0, 'orig', 'wdec'),
        'CF5': ('CF', 16, 5, 'ceil', 'wdec'), 'CFD5': ('CFD', 8, 5, 'ceil', 'wdec'),
        'CFZ5': ('CFZ', 8, 5, 'ceil', 'wdecz'), 'CFZ0': ('CFZ', 8, 0, 'ceil', 'wdecz'), 'CFD0': ('CFD', 8, 0, 'ceil', 'wdec')}
_ARCH = os.environ.get('WNR_NVRTC_ARCH') or ('sm_' + str(cp.cuda.Device(0).compute_capability))  # kit: was hard-coded sm_86 (RTX 3070); sm_120 on the RTX 5090 needs NVRTC >= 12.8
OPTSETS = {'lb': ('-arch=' + _ARCH, '-std=c++17'), 'out4': ('-arch=' + _ARCH, '-std=c++17', '-DOUTMODE=4')}
SC_PER_MATRIX = 368640


def nvsmi_apps():
    try:
        o = subprocess.run([NVSMI, '--query-compute-apps=pid,process_name', '--format=csv,noheader'], capture_output=True,
                           text=True, timeout=10).stdout
        return [(int(l.split(',')[0]), l.split(',', 1)[1].strip()) for l in o.strip().splitlines() if l.strip()]
    except Exception as e:  # noqa
        return [(-1, str(e))]


def nvsmi_gpu():
    try:
        return subprocess.run([NVSMI, '--query-gpu=clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu,memory.used',
                               '--format=csv,noheader'], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa
        return str(e)


def build_z(enc_dir, raw):
    z = np.load(os.path.join(enc_dir, 'var_CFD.npz'))
    pay, ln = z['payload'], z['lengths'].astype(np.int64)
    S = ln.size
    offs = np.zeros(S + 1, np.int64); offs[1:] = np.cumsum(ln)
    outs, lens = [], []
    CH = 17280
    nsg_all = []
    for a in range(0, S, CH):
        b = min(S, a + CH)
        rb = raw[a * 1024:b * 1024].reshape(b - a, 1024)
        sym = np.empty((b - a, 2048), np.uint8)
        sym[:, 0::2] = rb & 15
        sym[:, 1::2] = rb >> 4
        sym = np.where(sym == 8, np.uint8(0), sym)          # canonical zero: a zero has no sign
        planes, nP, t, r, nsg = Z.sign_planes(np.ascontiguousarray(sym.T), False)
        rp = pay[offs[a]:offs[b]]
        o, l, nsb = Z.assemble_streams(rp, ln[a:b], planes, nP)
        outs.append(o); lens.append(l); nsg_all.append(nsg)
    return {'payload': np.concatenate(outs), 'lengths': np.concatenate(lens), 'tables': z['tables'],
            'nsg': np.concatenate(nsg_all), 'cfd_bytes': int(pay.size)}


def main():
    enc_dir, out_json = sys.argv[1], sys.argv[2]
    R = int(sys.argv[3]) if len(sys.argv) > 3 else 41
    nts = [int(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 else [32, 64, 128, 256]
    opts = sys.argv[5].split(',') if len(sys.argv) > 5 else ['lb', 'out4']
    vnames = sys.argv[6].split(',') if len(sys.argv) > 6 else ['A259', 'D', 'CF5', 'CFD5', 'CFZ5']
    me = os.getpid()
    apps0 = nvsmi_apps()
    g0 = nvsmi_gpu()
    desktop = {n for p, n in apps0 if 'python' not in n.lower()}
    others = [(p, n) for p, n in apps0 if p != me and n not in desktop]
    res = {'nvsmi_apps_start': apps0, 'nvsmi_gpu_start': g0, 'other_compute_at_start': others, 'rounds': R, 'results': {}}
    util = int(g0.split(',')[2].strip().split()[0]) if g0 and ',' in g0 else -1
    if others or util > 5:
        res['aborted'] = f'GPU not idle at start: others={others} util={util}'
        json.dump(res, open(out_json, 'w'), indent=1)
        print(res['aborted'])
        return
    props = cp.cuda.runtime.getDeviceProperties(0)
    res['info'] = {'gpu': props['name'].decode(), 'sms': props['multiProcessorCount'], 'clock_khz_max': props['clockRate']}
    print(json.dumps(res['info']), g0, flush=True)
    com = np.load(os.path.join(enc_dir, 'common.npz'))
    spm = int(com['spm']); nmat = com['mtype'].size
    nstreams = nmat * spm
    raw = com['raw']
    lo, hi = raw & 15, raw >> 4
    rawc = (np.where(lo == 8, 0, lo) | (np.where(hi == 8, 0, hi) << 4)).astype(np.uint8)
    t0 = time.time()
    zd = build_z(enc_dir, raw)
    res['z_build'] = {'seconds': time.time() - t0, 'cfd_payload_bytes': zd['cfd_bytes'], 'z_payload_bytes': int(zd['payload'].size),
                      'plane_bytes': int(zd['payload'].size - zd['cfd_bytes']), 'mean_nsg': float(zd['nsg'].mean()),
                      'streams': int(zd['lengths'].size), 'neg0_bytes_changed_in_canonical_raw': int((rawc != raw).sum())}
    print('z build', res['z_build'], flush=True)
    d_raw, d_rawc = cp.asarray(raw), cp.asarray(rawc)
    d_sco = cp.asarray(com['scales'])
    d_scc = cp.asarray(com['scales_ceil'])
    d_mtype = cp.asarray(com['mtype'])
    d_med = {'orig': cp.asarray(com['med']), 'ceil': cp.asarray(com['med_ceil'])}
    d_sc = {'orig': d_sco, 'ceil': d_scc}
    src = open(os.path.join(HERE, 'wnr2dec_z.cu')).read()

    def kname(v, nt):
        vf, alpha, ceil, sset, fam = VARS[v]
        return f'wdec<{alpha},1,0,{nt},{ceil}>' if fam == 'wdec' else f'wdecz<{nt},{ceil}>'
    mods = {}
    t0 = time.time()
    for o in opts:
        names = sorted({kname(v, nt) for v in vnames for nt in nts})
        mods[o] = cp.RawModule(code=src, options=OPTSETS[o], name_expressions=names)
        mods[o].compile()
    res['compile_s'] = time.time() - t0
    loaded = {}

    def load(vf):
        if vf in loaded:
            return loaded[vf]
        z = zd if vf == 'CFZ' else np.load(os.path.join(enc_dir, f'var_{vf}.npz'))
        pay = z['payload']
        payp = np.concatenate([pay, np.zeros((-pay.size) % 4 + 8192, np.uint8)])
        lens = z['lengths'].astype(np.int64)
        offs = np.zeros(lens.size + 1, np.uint32); offs[1:] = np.cumsum(lens)
        assert offs[-1] == pay.size
        signs = z['signs'] if (vf != 'CFZ' and 'signs' in z.files) else None
        d = {'pay': cp.asarray(payp.view(np.uint32)), 'offs': cp.asarray(offs), 'tabs': cp.asarray(z['tables'].reshape(-1)),
             'signs': cp.asarray(signs) if signs is not None else cp.zeros(1, cp.uint32),
             'bytes': int(pay.size + 2 * lens.size + (signs.size * 4 if signs is not None else 0))}
        loaded[vf] = d
        return d
    st = cp.cuda.Stream(non_blocking=True)
    sts = [cp.cuda.Stream(non_blocking=True) for _ in range(nmat // 3)]
    d_err = cp.zeros(1, cp.int32)
    d_out = cp.zeros(nstreams * 1024, cp.uint8)
    d_osc = cp.zeros(nmat * SC_PER_MATRIX, cp.uint8)
    d_bt = cp.zeros(24, cp.uint16)

    def wlauncher(v, nt, o):
        vf, alpha, ceil, sset, fam = VARS[v]
        dp = load(vf)
        fn = mods[o].get_function(kname(v, nt))
        bpm = -(-spm // nt)
        smem = 16 + 259 * alpha * 2 + (nt * 144 if o == 'out4' else 0)
        if smem > 48 * 1024:
            fn.max_dynamic_shared_size_bytes = smem
        sc_in, med = d_sc[sset], d_med[sset]
        if fam == 'wdec':
            def launch(mc, m0=0):
                fn((mc * bpm,), (nt,), (dp['pay'], dp['offs'], sc_in, dp['signs'], dp['tabs'], d_bt, d_mtype, med,
                                        np.int32(m0), np.int32(spm), np.int32(bpm), d_out, d_err, d_osc), shared_mem=smem)
        else:
            def launch(mc, m0=0):
                fn((mc * bpm,), (nt,), (dp['pay'], dp['offs'], sc_in, dp['tabs'], d_mtype, med,
                                        np.int32(m0), np.int32(spm), np.int32(bpm), d_out, d_err, d_osc), shared_mem=smem)
        return fn, smem, launch

    def prep(v):
        d_out.fill(0); d_err.fill(0)
        if VARS[v][2] == 5:
            d_osc[:] = d_scc
        cp.cuda.runtime.deviceSynchronize()

    def check(v, n=None):
        vf, alpha, ceil, sset, fam = VARS[v]
        n = nstreams if n is None else n
        ref = d_rawc if vf == 'CFZ' else d_raw
        if ceil == 0 and sset == 'ceil':
            # no inverse: output is in coded (ceil) form -> only error flag + Z vs CFD equivalence is checked elsewhere
            badw = -1
        else:
            badw = int((d_out[:n * 1024].reshape(n, 1024) != ref[:n * 1024].reshape(n, 1024)).any(1).sum())
        if ceil == 5:
            bads = int((d_osc[:n * 64].reshape(n, 64) != d_sco[:n * 64].reshape(n, 64)).any(1).sum())
        else:
            bads = 0
        return badw, bads, int(d_err.get()[0])

    def timed(launchers):
        names = list(launchers)
        out = {}
        for mc in (1, 3, nmat):
            with st:
                tw = time.time()
                while time.time() - tw < 0.25:
                    for n_ in names:
                        launchers[n_](mc)
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
        T = np.zeros((R, len(names)))
        for r in range(R + 4):
            order = [(r + k) % len(names) for k in range(len(names))]
            row = []
            for k in order:
                e0 = cp.cuda.Event(); e0.record(st)
                ends = []
                for ri, s_ in enumerate(sts):
                    s_.wait_event(e0)
                    with s_:
                        launchers[names[k]](3, 3 * ri)
                    ee = cp.cuda.Event(); ee.record(s_); ends.append(ee)
                for ee in ends:
                    st.wait_event(ee)
                e1 = cp.cuda.Event(); e1.record(st)
                row.append((k, e0, e1))
            st.synchronize()
            if r >= 4:
                for k, e0, e1 in row:
                    T[r - 4, k] = cp.cuda.get_elapsed_time(e0, e1) * 1000.0
        out[f'{nmat // 3}streams'] = T
        summ = {}
        for key, T in out.items():
            for i, n_ in enumerate(names):
                d = summ.setdefault(n_, {})
                d[key] = {'us_median': float(np.median(T[:, i])), 'us_min': float(T[:, i].min()),
                          'us_p90': float(np.percentile(T[:, i], 90))}
                for j, b in enumerate(names):
                    if b != n_:
                        d[key][f'ratio_vs_{b}'] = float(np.median(T[:, i] / T[:, j]))
        return summ

    for o in opts:
        for nt in nts:
            cfg = f'nt{nt}/{o}'
            apps = nvsmi_apps()
            oth = [(p, n_) for p, n_ in apps if p != me and n_ not in desktop]
            Lc, info = {}, {}
            for v in vnames:
                fn, smem, launch = wlauncher(v, nt, o)
                prep(v)
                with st:
                    launch(nmat)
                st.synchronize()
                badw, bads, errs = check(v)
                occ = drv.occupancyMaxActiveBlocksPerMultiprocessor(fn.kernel.ptr, nt, smem)
                info[v] = {'kernel': kname(v, nt), 'regs': fn.num_regs, 'local_bytes': fn.local_size_bytes, 'smem': smem,
                           'occ_warps_per_sm': occ * nt // 32, 'weight_bad_streams': badw, 'scale_bad_streams': bads,
                           'errors': errs, 'coded_weight_bytes': load(VARS[v][0])['bytes']}
                Lc[v] = launch
            tm = timed(Lc)
            for v in vnames:
                for mc in (1, 3):
                    prep(v)
                    with st:
                        Lc[v](mc)
                    st.synchronize()
                    info[v][f'm{mc}_recheck'] = check(v, mc * spm)
                info[v]['timing'] = tm[v]
            apps2 = nvsmi_apps()
            oth2 = [(p, n_) for p, n_ in apps2 if p != me and n_ not in desktop]
            res['results'][cfg] = {'variants': info, 'other_compute_before': oth, 'other_compute_after': oth2,
                                   'contaminated': bool(oth or oth2), 'nvsmi_gpu_after': nvsmi_gpu()}
            for v in vnames:
                t = info[v]['timing']
                print(f"{cfg:10s} {v:5s} regs {info[v]['regs']:3d} lmem {info[v]['local_bytes']} occW {info[v]['occ_warps_per_sm']:2d} "
                      f"w {info[v]['weight_bad_streams']} s {info[v]['scale_bad_streams']} e {info[v]['errors']} | "
                      + ' '.join(f"{k} {t[k]['us_median']:8.1f}" + (f" /CFD5 {t[k].get('ratio_vs_CFD5', 1):.3f}" if 'CFD5' in vnames else '')
                                 for k in t), 'CONTAMINATED' if (oth or oth2) else '', flush=True)
            json.dump(res, open(out_json, 'w'), indent=1)
    res['nvsmi_gpu_end'] = nvsmi_gpu()
    res['nvsmi_apps_end'] = nvsmi_apps()
    json.dump(res, open(out_json, 'w'), indent=1)
    print('end', res['nvsmi_gpu_end'])


if __name__ == '__main__':
    main()
