"""nvrtc_check: compile-only smoke check of every GPU kernel of tests/gpu (no module load, no kernel launch; a GPU is
not needed). For each arch: NVRTC front end to PTX (compute_XY) and NVRTC + ptxas to a cubin (sm_XY, ptxas -v), with
registers / spills / static shared memory per kernel parsed from the ptxas log.

  python -I tests/gpu/run_nvrtc_check.py --root <kit> [--archs 86,120] [--all-variants] [--quick]

Default archs: 86 and 120 (RTX 3070 and RTX 5090) plus the arch of device 0 if a GPU is visible.
Kernel sets: bench (src_bench/wnr2dec.cu, plain and OUTMODE=4, all gpu_bench variants x nt 32..256 + sdec),
zero (src_zero/zs_gpu_kernels.cu + zs_extra_kernels.cu; zs_sm120_mxf4.cu for sm_12x only, as sm_12xa),
v1 (src_v1/wnr3dec.cu with the name list of v1bench.py; EXPERIMENTAL). --quick: nt 64 only, sm_ only.
"""
# read by setup/kitenv.py with ast (runner discovery for run_all.ps1); never executed there.
# needs 'nvrtc' (CuPy + NVRTC wheels), NOT 'gpu' and NOT 'data': it compiles only, so -NoGpu keeps it.
KIT_META = {'name': 'nvrtc_check', 'order': 55, 'needs': ['nvrtc'], 'quick_minutes': 1, 'full_minutes': 3,
            'description': 'compile-only NVRTC/ptxas check of every GPU kernel for sm_86 and sm_120 (no GPU needed)'}
import json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpukit_common as K  # noqa: E402

TEST = 'nvrtc_check'


def parse_args():
    p = K.base_parser(TEST, __doc__)
    p.add_argument('--archs', default=None, help='comma list of compute capabilities, e.g. 86,120 (default 86,120 + device)')
    p.add_argument('--all-variants', action='store_true', help='bench: all 18 team-4 variants (default: the gpu_bench set)')
    p.add_argument('--sets', default='bench,zero,v1', help='kernel sets to compile (default bench,zero,v1)')
    return p.parse_args()


def kernel_sets(args, nts):
    sys.path.insert(0, os.path.join(HERE, 'src_bench'))
    import run_gpu_bench as B
    sets = {}
    rd = lambda *p: open(os.path.join(HERE, *p)).read()  # noqa: E731
    vn = list(B.VARS) if args.all_variants else B.DEFAULT_VARIANTS.split(',')
    bn = sorted({B.kname(B.VARS[v][1], B.VARS[v][2], B.VARS[v][3], nt, B.VARS[v][4]) for v in vn for nt in nts}
                | {f'sdec<{n}>' for n in nts})
    src = rd('src_bench', 'wnr2dec.cu')
    sets['bench/lb'] = (src, bn, ('-std=c++17',), False)
    sets['bench/out4'] = (src, bn, ('-std=c++17', '-DOUTMODE=4'), False)
    import run_zero_bittest as Z
    sets['zero/colibri'] = (rd('src_zero', 'zs_gpu_kernels.cu'), Z.TPL + ['v4f_rows'], ('-std=c++17', '--ftz=false', '--fmad=true'), False)
    sets['zero/extra'] = (rd('src_zero', 'zs_extra_kernels.cu'), ['tc_gemm', 'mma_f16', 'mma_bf16', 'negctl_seeded'],
                          ('-std=c++17', '--ftz=false', '--fmad=true'), False)
    sets['zero/sm120_mxf4(EXPERIMENTAL)'] = (rd('src_zero', 'zs_sm120_mxf4.cu'), ['mxf4_gemm'], ('-std=c++17',), True)
    try:
        sys.path.insert(0, os.path.join(HERE, 'src_v1'))
        import v1bench as V
        v1src = rd('src_v1', 'wnr3dec.cu')
        for o in ('lb', 'out4'):
            nm = set()
            for nt in nts:
                for v in V.WV:
                    nm.add(V.wname(v, nt))
                for n in V.NEG:
                    nm.add(V.wname(None, nt, neg=n))
                for v in V.SV:
                    nm.add(V.sname(v, nt))
            if o == 'lb':
                nm |= {V.wname('G5', 32, 1), V.wname('G5H', 32, 1), V.wname('G5HL', 32, 1), V.sname('SG', 32, 1),
                       V.sname('SH', 32, 1), V.sname('SHL', 32, 1), 'h32test'}
                nm |= {V.sname(v, 32) for v in V.SV}
            sets[f'v1/{o}(EXPERIMENTAL)'] = (v1src, sorted(nm), ('-std=c++17',) + (('-DOUTMODE=4',) if o == 'out4' else ()), False)
    except Exception as ex:  # noqa
        sets['v1(EXPERIMENTAL)'] = ex
    want = set(args.sets.split(','))
    return {k: v for k, v in sets.items() if k.split('/')[0].split('(')[0] in want}


RX_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
RX_USED = re.compile(r'Used (\d+) registers(?:, used (\d+) barriers)?(?:, (\d+) bytes smem)?')
RX_SPILL = re.compile(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads')


def compile_one(nvrtc, src, names, opts, arch_opt):
    prog = nvrtc.createProgram(src, 'k.cu', [], [])
    try:
        for n in names:
            nvrtc.addNameExpression(prog, n)
        o = [arch_opt] + list(opts) + (['--ptxas-options=-v'] if arch_opt.startswith('-arch=sm_') else [])
        t0 = time.time()
        try:
            nvrtc.compileProgram(prog, o)
        except Exception as ex:  # noqa
            return {'ok': False, 'error': str(ex), 'log': nvrtc.getProgramLog(prog)[-4000:]}
        log = nvrtc.getProgramLog(prog)
        low = {}
        for n in names:
            try:
                low[nvrtc.getLoweredName(prog, n)] = n
            except Exception:  # noqa
                low['?' + n] = n
        r = {'ok': True, 'seconds': round(time.time() - t0, 1), 'kernels': len(names),
             'missing_name_expressions': [n for k, n in low.items() if k.startswith('?')]}
        if arch_opt.startswith('-arch=sm_'):
            r['cubin_bytes'] = len(nvrtc.getCUBIN(prog))
            per, cur = {}, None
            for line in log.splitlines():
                m = RX_ENTRY.search(line)
                if m:
                    cur = low.get(m.group(1), m.group(1)); per[cur] = {}
                    continue
                m = RX_SPILL.search(line)
                if m and cur:
                    per[cur].update(stack=int(m.group(1)), spill_st=int(m.group(2)), spill_ld=int(m.group(3)))
                m = RX_USED.search(line)
                if m and cur:
                    per[cur].update(regs=int(m.group(1)), barriers=int(m.group(2) or 0), static_smem=int(m.group(3) or 0))
            r['ptxas'] = per
            r['max_regs'] = max((v.get('regs', 0) for v in per.values()), default=None)
            r['any_spill'] = any(v.get('spill_st', 0) or v.get('spill_ld', 0) for v in per.values())
        else:
            r['ptx_bytes'] = len(nvrtc.getPTX(prog))
        return r
    finally:
        try:
            nvrtc.destroyProgram(prog)
        except Exception:  # noqa
            pass


def main():
    args = parse_args()
    K.resolve_paths(args, TEST)
    log = K.Log(os.path.join(args.out, TEST + '.log'))
    return K.run_main(TEST, lambda: body(args, log), args, args.out)


def body(args, log):
    from cupy_backends.cuda.libs import nvrtc
    env = K.env_info(args.device)
    archs = [a.strip() for a in (args.archs or '86,120').split(',') if a.strip()]
    if not args.archs and 'gpu' in env:
        dev = env['gpu']['cc'].replace('.', '')
        if dev not in archs:
            archs.append(dev)
    supported = [str(a) for a in nvrtc.getSupportedArchs()]
    nts = [64] if args.quick else [32, 64, 128, 256]
    sets = kernel_sets(args, nts)
    log('NVRTC %d.%d, supported archs %s; testing %s; sets %s' % (*nvrtc.getVersion(), ','.join(supported), archs, list(sets)))
    results, fails = {}, []
    for a in archs:
        if a not in supported:
            results[a] = 'NVRTC %d.%d does not support arch %s' % (*nvrtc.getVersion(), a)
            fails.append(f'{a}: unsupported by this NVRTC')
            log(results[a])
            continue
        for sname, spec in sets.items():
            if isinstance(spec, Exception):
                fails.append(f'{sname}: {spec}')
                continue
            src, names, opts, arch_specific = spec
            if arch_specific and not a.startswith('12'):
                continue
            modes = ('sm',) if args.quick else ('compute', 'sm')
            for mode in modes:
                ao = '-arch=%s_%s%s' % (mode, a, 'a' if arch_specific else '')
                r = compile_one(nvrtc, src, names, opts, ao)
                results[f'{ao[6:]}/{sname}'] = r
                if not r['ok'] or r.get('missing_name_expressions'):
                    fails.append(f'{ao[6:]}/{sname}')
                log('%-14s %-34s %s %s' % (ao[6:], sname, 'OK' if r['ok'] else 'FAIL',
                                          ('%d kernels %.1f s max_regs %s spill %s' % (r['kernels'], r['seconds'], r.get('max_regs'), r.get('any_spill')))
                                          if r['ok'] else r['error']))
                if not r['ok']:
                    log(r['log'])
    with open(os.path.join(args.out, 'nvrtc_check.json'), 'w') as f:
        json.dump(results, f, indent=1)
    regs = {k: {n: v.get('regs') for n, v in r['ptxas'].items()} for k, r in results.items()
            if isinstance(r, dict) and r.get('ptxas')}
    summary = {'env': env, 'archs': archs, 'nvrtc_supported': supported, 'failures': fails, 'registers': regs,
               'files': ['nvrtc_check.json', TEST + '.log']}
    status = 'PASS' if not fails else 'FAIL'
    return K.finish(TEST, args.out, status, 'archs %s: %d compiles, %d failures' % (
        ','.join(archs), sum(1 for r in results.values() if isinstance(r, dict)), len(fails)), summary, log)


if __name__ == '__main__':
    sys.exit(main())
