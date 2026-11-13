"""Paired benchmark: every configuration X is timed interleaved launch-by-launch with the baseline
kernel A (wdec<16,0,0,32>, 567 rows, 32 threads/block, default launch bounds), so that ratios X/A stay
meaningful while another process shares the GPU. Every configuration is verified byte-exact first.

Usage: python -I bench2.py ENC_DIR OUT_JSON SET [pairs]
"""
import json, os, sys, time, subprocess
import numpy as np
import cupy as cp
from cupy_backends.cuda.api import driver as drv

HERE = os.path.dirname(os.path.abspath(__file__))
NVSMI = 'nvidia-smi'  # wnr3-5090: PATH lookup instead of the research PC driver-store path
ARCH_OPT = '-arch=' + os.environ.get('CEIL_ARCH', 'sm_120')  # wnr3-5090: was -arch=sm_86
VARIANTS = {  # payload file, table file, ALPHA, CTX, ILV
    'A': ('A', 'A', 16, 0, 0), 'A259': ('A', 'A259', 16, 1, 0),
    'B1': ('B1', 'B1', 16, 0, 1), 'B1c': ('B1', 'A259', 16, 1, 1),
    'B2': ('B2', 'B2', 16, 0, 2), 'B2c': ('B2', 'A259', 16, 1, 2),
    'C': ('C', 'C', 16, 2, 0), 'CB2': ('CB2', 'CB2', 16, 2, 2),
    'D': ('D', 'D', 8, 1, 0), 'DB2': ('DB2', 'DB2', 8, 1, 2),
    'B2s': ('B2s', 'A259', 16, 1, 3), 'DB2s': ('DB2s', 'DB2s', 8, 1, 3),
}
ROWS = {0: 567, 1: 259, 2: 1036}
OUT_PER_MATRIX = 5760 * 1024
OPTSETS = {
    'lb': (ARCH_OPT, '-std=c++17'),
    'nolb': (ARCH_OPT, '-std=c++17', '-DNOLB'),
    'r64': (ARCH_OPT, '-std=c++17', '-DNOLB', '-maxrregcount=64'),
    'r48': (ARCH_OPT, '-std=c++17', '-DNOLB', '-maxrregcount=48'),
    'r40': (ARCH_OPT, '-std=c++17', '-DNOLB', '-maxrregcount=40'),
    'nolb_cg': (ARCH_OPT, '-std=c++17', '-DNOLB', '-DLDM=1'),
    'cg': (ARCH_OPT, '-std=c++17', '-DLDM=1'),
    'out1': (ARCH_OPT, '-std=c++17', '-DOUTMODE=1'),
    'out2': (ARCH_OPT, '-std=c++17', '-DOUTMODE=2'),
    'out3': (ARCH_OPT, '-std=c++17', '-DOUTMODE=3'),
    'cg_out1': (ARCH_OPT, '-std=c++17', '-DLDM=1', '-DOUTMODE=1'),
    'out4': (ARCH_OPT, '-std=c++17', '-DOUTMODE=4'),
}
BASE = ('A', 'wdec', 32, 'lb', 0)


def kname(kind, alpha, ctx, ilv, nt):
    return {'wdec': f'wdec<{alpha},{ctx},{ilv},{nt}>', 'wdec2r': f'wdec2<{alpha},{ctx},{ilv},{nt},1>',
            'wdec2u': f'wdec2<{alpha},{ctx},{ilv},{nt},4>'}[kind]


def configs(which):
    cf = []
    if which == 'regs':
        for kind in ('wdec', 'wdec2r'):
            for nt in (32, 64, 128, 256):
                for o in ('lb', 'nolb', 'r64', 'r48'):
                    cf.append(('A', kind, nt, o, 0))
    elif which == 'occ':   # occupancy sweep at fixed registers via smem padding (blocks/SM target)
        for kind, o in (('wdec', 'nolb'), ('wdec2r', 'nolb')):
            for nt in (32, 64, 128):
                for blk in (1, 2, 3, 4, 6, 8, 12, 16):
                    if blk * nt > 1536:
                        continue
                    cf.append(('A', kind, nt, o, blk))
    elif which == 'mem':   # where does the high-occupancy slowdown come from? (out1/out2 are NOT byte-exact by design)
        for nt in (32, 64, 128, 256):
            for o in ('lb', 'cg', 'out3', 'out2', 'out1', 'cg_out1'):
                cf.append(('A', 'wdec', nt, o, 0))
        for nt in (64, 128, 256):
            for o in ('lb', 'out1'):
                cf.append(('B2c', 'wdec', nt, o, 0))
                cf.append(('D', 'wdec', nt, o, 0))
    elif which == 'final':
        for v in VARIANTS:
            for o in ('lb', 'out4'):
                for nt in (32, 64, 128, 256):
                    cf.append((v, 'wdec', nt, o, 0))
    elif which == 'split':
        for v in ('B2s', 'DB2s', 'B2c', 'DB2', 'A259', 'D'):
            for o in ('lb', 'out4'):
                for nt in (32, 64, 128):
                    cf.append((v, 'wdec', nt, o, 0))
    elif which == 'stage':
        for v in ('A', 'A259', 'B2', 'B2c', 'C', 'CB2', 'D', 'DB2'):
            for nt in (32, 64, 128):
                cf.append((v, 'wdec', nt, 'out4', 0))
    elif which.startswith('vars:'):
        _, kinds, opts, nts = which.split(':')
        for v in VARIANTS:
            for kind in kinds.split(','):
                for o in opts.split(','):
                    for nt in nts.split(','):
                        cf.append((v, kind, int(nt), o, 0))
    return cf


def nvsmi():
    try:
        return subprocess.run([NVSMI, '--query-gpu=clocks.sm,utilization.gpu,power.draw', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa
        return str(e)


def main():
    enc_dir, out_json, which = sys.argv[1], sys.argv[2], sys.argv[3]
    pairs = int(sys.argv[4]) if len(sys.argv) > 4 else 21
    cf = configs(which)
    com = np.load(os.path.join(enc_dir, 'common.npz'))
    spm = int(com['spm']); nmat = com['mtype'].size
    nstreams = nmat * spm
    d_raw = cp.asarray(com['raw']); d_scales = cp.asarray(com['scales'])
    d_mtype = cp.asarray(com['mtype']); d_med = cp.asarray(com['med'])
    src = open(os.path.join(HERE, 'wnr2dec.cu')).read()
    t0 = time.time()
    mods = {}
    for o in sorted({c[3] for c in cf} | {BASE[3]}):
        names = {kname(c[1], *VARIANTS[c[0]][2:], c[2]) for c in cf if c[3] == o}
        if o == BASE[3]:
            names.add(kname(BASE[1], *VARIANTS[BASE[0]][2:], BASE[2]))
        mods[o] = cp.RawModule(code=src, options=OPTSETS[o], name_expressions=sorted(names))
        mods[o].compile()
    res = {'compile_s': time.time() - t0, 'nvsmi_start': nvsmi(), 'pairs': pairs, 'baseline': list(BASE), 'results': {}}
    print(json.dumps({k: res[k] for k in ('compile_s', 'nvsmi_start')}), flush=True)
    d_out = cp.zeros(nstreams * 1024, cp.uint8)
    d_err = cp.zeros(1, cp.int32)
    loaded = {}

    def load(vname):
        if vname in loaded:
            return loaded[vname]
        z = np.load(os.path.join(enc_dir, f'var_{vname}.npz'))
        pay = z['payload']
        payp = np.concatenate([pay, np.zeros((-pay.size) % 4 + 8192, np.uint8)])
        offs = np.zeros(z['lengths'].size + 1, np.uint32); offs[1:] = np.cumsum(z['lengths'].astype(np.int64))
        assert offs[-1] == pay.size
        d = {'pay': cp.asarray(payp.view(np.uint32)), 'offs': cp.asarray(offs), 'tabs': cp.asarray(z['tables'].reshape(-1))}
        bt = np.zeros((3, 8), np.uint16)
        if 'bittab' in z.files:
            bt[:, :7] = z['bittab']
        d['bittab'] = cp.asarray(bt.reshape(-1))
        d['signs'] = cp.asarray(z['signs']) if 'signs' in z.files else cp.zeros(1, cp.uint32)
        loaded[vname] = d
        return d

    def prep(v, kind, nt, o, blk):
        pv, tv, alpha, ctx, ilv = VARIANTS[v]
        dp, dt = load(pv), load(tv)
        fn = mods[o].get_function(kname(kind, alpha, ctx, ilv, nt))
        bpm = -(-spm // nt)
        smem = 16 + ROWS[ctx] * alpha * 2 + (16 if ctx == 2 else 0)
        if o == 'out4':
            smem += nt * 144   # per-lane 128-B output line staging (stride 144 B)
        if blk:
            smem = max(smem, (102400 // blk) - 1024 - 16 - 64)
        if smem > 48 * 1024:
            fn.max_dynamic_shared_size_bytes = smem

        def launch(mc, stream=None, m0=0):
            fn((mc * bpm,), (nt,), (dp['pay'], dp['offs'], d_scales, dp['signs'], dt['tabs'], dt['bittab'], d_mtype,
                                     d_med, np.int32(m0), np.int32(spm), np.int32(bpm), d_out, d_err), shared_mem=smem)
        return fn, smem, launch

    _, _, base_launch = prep(*BASE)
    st = cp.cuda.Stream(non_blocking=True)
    for c in cf:
        v, kind, nt, o, blk = c
        fn, smem, launch = prep(*c)
        key = f'{v}/{kind}/nt{nt}/{o}' + (f'/blk{blk}' if blk else '')
        d_out.fill(0); d_err.fill(0)
        with st:
            launch(nmat)
        st.synchronize()
        errs = int(d_err.get()[0])
        bad = int((d_out.reshape(nstreams, 1024) != d_raw.reshape(nstreams, 1024)).any(1).sum())
        occ = drv.occupancyMaxActiveBlocksPerMultiprocessor(fn.kernel.ptr, nt, smem)
        r = {'variant': v, 'kernel': kname(kind, *VARIANTS[v][2:], nt), 'opts': o, 'nt': nt, 'smem': smem,
             'regs': fn.num_regs, 'local_bytes': fn.local_size_bytes, 'occ_blocks_per_sm': occ,
             'occ_warps_per_sm': occ * nt // 32, 'byte_exact': bad == 0 and errs == 0, 'bad_streams': bad, 'errors': errs}
        for mc in (1, 3, nmat):
            with st:
                tw = time.time()
                while time.time() - tw < 0.25:
                    for _ in range(4):
                        base_launch(mc); launch(mc)
                    st.synchronize()
                ev = [[cp.cuda.Event() for _ in range(4)] for _ in range(pairs)]
                for e in ev:
                    e[0].record(st); base_launch(mc); e[1].record(st)
                    e[2].record(st); launch(mc); e[3].record(st)
                st.synchronize()
            ta = np.array([cp.cuda.get_elapsed_time(e[0], e[1]) for e in ev]) * 1000
            tx = np.array([cp.cuda.get_elapsed_time(e[2], e[3]) for e in ev]) * 1000
            r[f'm{mc}'] = {'x_us_median': float(np.median(tx)), 'x_us_min': float(tx.min()),
                           'a_us_median': float(np.median(ta)), 'a_us_min': float(ta.min()),
                           'ratio_median_of_pairs': float(np.median(tx / ta)),
                           'ratio_of_minima': float(tx.min() / ta.min()),
                           'x_us_per_matrix': float(np.median(tx)) / mc,
                           'x_us_per_record_equiv': float(np.median(tx)) / mc * 3,
                           'x_decoded_GiB_s': mc * OUT_PER_MATRIX / (np.median(tx) * 1e-6) / 2**30}
        # 8 records in flight as 8 separate 3-matrix launches on 8 streams (KAIRA batch-lane style), paired with A
        nrec = nmat // 3
        sts = [cp.cuda.Stream(non_blocking=True) for _ in range(nrec)]

        def multi(lf):
            e0 = cp.cuda.Event(); e0.record(st)
            ends = []
            for ri, s_ in enumerate(sts):
                s_.wait_event(e0)
                with s_:
                    lf(3, m0=3 * ri)
                ee = cp.cuda.Event(); ee.record(s_); ends.append(ee)
            for ee in ends:
                st.wait_event(ee)
            e1 = cp.cuda.Event(); e1.record(st)
            return e0, e1
        pa, px = [], []
        for it in range(pairs + 4):
            a0, a1 = multi(base_launch)
            x0, x1 = multi(launch)
            st.synchronize()
            if it >= 4:
                pa.append(cp.cuda.get_elapsed_time(a0, a1) * 1000); px.append(cp.cuda.get_elapsed_time(x0, x1) * 1000)
        pa, px = np.array(pa), np.array(px)
        r[f'{nrec}streams'] = {'x_us_median': float(np.median(px)), 'a_us_median': float(np.median(pa)),
                               'ratio_median_of_pairs': float(np.median(px / pa)),
                               'x_us_per_record': float(np.median(px)) / nrec,
                               'x_decoded_GiB_s': nmat * OUT_PER_MATRIX / (np.median(px) * 1e-6) / 2**30}
        cp.cuda.runtime.deviceSynchronize()
        if r['byte_exact']:
            r['multistream_bad_streams'] = int((d_out.reshape(nstreams, 1024) != d_raw.reshape(nstreams, 1024)).any(1).sum())
        res['results'][key] = r
        m = r[f'm{nmat}']; m3 = r['m3']; m1 = r['m1']
        print(f"{key:30s} regs {r['regs']:3d} occW {r['occ_warps_per_sm']:2d} exact {r['byte_exact']} | "
              f"m1 {m1['x_us_median']:7.1f} (A {m1['a_us_median']:7.1f} r {m1['ratio_median_of_pairs']:.3f}) "
              f"m3 {m3['x_us_median']:7.1f} (A {m3['a_us_median']:7.1f} r {m3['ratio_median_of_pairs']:.3f}) "
              f"m{nmat} {m['x_us_median']:8.1f} (A {m['a_us_median']:8.1f} r {m['ratio_median_of_pairs']:.3f}) "
              f"{m['x_decoded_GiB_s']:5.1f} GiB/s | 8str {r[f'{nrec}streams']['x_us_median']:8.1f} "
              f"r {r[f'{nrec}streams']['ratio_median_of_pairs']:.3f}", flush=True)
        json.dump(res, open(out_json, 'w'), indent=1)
    res['nvsmi_end'] = nvsmi()
    json.dump(res, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
