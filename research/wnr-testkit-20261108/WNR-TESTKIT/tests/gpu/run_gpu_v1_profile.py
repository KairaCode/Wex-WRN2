"""gpu_v1_profile  ***EXPERIMENTAL - UNTESTED IN THIS KIT***

WNR3 v1 profile (CEIL | LENC | HASH) decoded in-kernel on the GPU: team 5 gpu-v1 partial work, wrapped for the kit.
Team 5's sources (src_v1/: v1enc.py, v1bench.py, wnr3dec.cu, h32.py, wnr2fmt.py, v1summ.py) are copied unchanged except
for the path / arch constants (now from environment variables set here). Their first full run (RTX 3070, 41 rounds)
is in ref/v1_3070/; the copied v1bench.py is the LATER version (extra DUAL scale-store variants and pipeline forms)
whose second run was stopped before it produced results, so this exact code has never completed a run.

  python -I tests/gpu/run_gpu_v1_profile.py --root <kit> [--data <dir>] [--out <dir>] [--quick] [--stages ...]

Steps: (1) team-4 baseline streams = the gpu_bench encode cache (<parent of --out>/gpu_bench/enc, else
<root>/results/gpu_bench/enc, or --t4enc; created with the gpu_bench encoder in the first if missing); (2) v1enc.py encodes the 8 experts as v1-profile streams (held-out g0 = 120 ceil
model of team 4 ceil-ref, src_v1/ceilref_even_model.npz) -> <out>/v1enc; (3) v1bench.py: H32 GPU vectors, scale
kernels, weight kernels byte-exact vs the original bytes, negative controls, corruption and LENC corruption trials,
paired timing (>= 21 rounds), pipelines; (4) v1summ.py tables.
Result line: DONE (never PASS: experimental) when v1bench finished, with the byte-exact flags in the summary;
FAIL if a step crashed.
"""
# read by setup/kitenv.py with ast (runner discovery for run_all.ps1); never executed there.
# optional: run_all.ps1 runs it only with -All (or when named in -Only); order 70 = after gpu_bench (reuses its cache)
KIT_META = {'name': 'gpu_v1_profile', 'order': 70, 'needs': ['gpu', 'data'], 'optional': True, 'quick_minutes': 10,
            'full_minutes': 40, 'description': 'EXPERIMENTAL: WNR3 v1 profile CEIL|LENC|HASH on the GPU (never completed a run)'}
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'src_bench'))
import gpukit_common as K  # noqa: E402

TEST = 'gpu_v1_profile'
SRC = os.path.join(HERE, 'src_v1')


def parse_args():
    p = K.base_parser(TEST, __doc__)
    p.add_argument('--rounds', type=int, default=None, help='paired rounds (>= 21; default 41, quick 21)')
    p.add_argument('--nts', default=None, help='threads per block (default 32,64,128,256; quick 32,64)')
    p.add_argument('--opts', default='lb,out4')
    p.add_argument('--stages', default=None, help='v1bench stages (default all: h32,scale,weight,neg,corrupt,lenc,timing,pipe)')
    p.add_argument('--t4enc', default=None, help='team-4 encode dir (default: <parent of --out>/gpu_bench/enc, then '
                                                 '<root>/results/gpu_bench/enc; encoded into the first if neither is valid)')
    p.add_argument('--arch-mode', choices=('sm', 'compute'), default='sm')
    p.add_argument('--reencode', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    K.resolve_paths(args, TEST)
    log = K.Log(os.path.join(args.out, TEST + '.log'))
    return K.run_main(TEST, lambda: body(args, log), args, args.out)


def run_logged(cmd, env, logpath, log):
    log('run', ' '.join('"%s"' % c if ' ' in c else c for c in cmd))
    with open(logpath, 'w', encoding='utf-8') as f:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             encoding='utf-8', errors='replace')
        for line in p.stdout:
            f.write(line); f.flush()
            print(line, end='', flush=True)
        return p.wait()


def body(args, log):
    log('*** EXPERIMENTAL gpu_v1_profile (team 5 gpu-v1 partial work; this code version never completed a run) ***')
    R = args.rounds if args.rounds is not None else (21 if args.quick else 41)
    if R < 21:
        raise ValueError('--rounds must be >= 21')
    nts = args.nts or ('32,64' if args.quick else '32,64,128,256')
    summary = {'EXPERIMENTAL': True, 'rounds': R, 'nts': nts, 'opts': args.opts}
    # (1) team-4 baseline encode
    # gpu_bench's cache next to this runner's --out (run_all -Out D:\res -> D:\res\gpu_bench\enc), then the default
    # <root>/results/gpu_bench/enc; if neither holds a verified encode, encode into the first (inside the results
    # folder that was asked for, never outside it)
    if args.t4enc:
        t4_cands = [os.path.abspath(args.t4enc)]
    else:
        t4_cands = [os.path.join(os.path.dirname(os.path.abspath(args.out)), 'gpu_bench', 'enc'),
                    os.path.join(args.root, 'results', 'gpu_bench', 'enc')]
    t4, ok_t4 = t4_cands[0], False
    for c in dict.fromkeys(os.path.normcase(os.path.abspath(x)) for x in t4_cands):
        rp = os.path.join(c, 'encode_report.json')
        if os.path.exists(rp):
            with open(rp) as f:
                if json.load(f).get('eq_team4_encoder_output_all'):
                    t4, ok_t4 = c, True
                    break
    log('team-4 baseline encode dir', t4, '(verified cache)' if ok_t4 else '(will encode)')
    summary['t4enc'] = t4
    if not ok_t4:
        import t4enc
        man, paths, _ = K.resolve_experts(args, K.PICK8, log)
        with open(os.path.join(K.REF, 't4enc_expected.json')) as f:
            expected = json.load(f)
        log('team-4 baseline encode missing -> encoding into', t4)
        rep = t4enc.encode_all(man, paths, os.path.join(K.REF, 'model_counts.npz'), t4, log, expected=expected)
        if not rep.get('eq_team4_encoder_output_all'):
            raise RuntimeError('team-4 baseline encode differs from team 4 output')
    # (2) sample dir in one place (team-5 scripts take a directory)
    man, paths, sample_dir = K.resolve_experts(args, K.PICK8, log, need_dir=True)
    import cupy as cp
    cp.cuda.Device(args.device).use()
    env_info = K.env_info(args.device)
    summary['env'] = env_info
    arch, cc = K.device_arch(args.device, args.arch_mode)
    env = dict(os.environ)
    env.update({'WNRKIT_SAMPLE': sample_dir, 'WNRKIT_ARCH': arch, 'WNRKIT_NVSMI': K.find_nvsmi() or 'nvidia-smi',
                'WNRKIT_V1_MODEL': os.path.join(SRC, 'ceilref_even_model.npz'), 'CUDA_VISIBLE_DEVICES': str(args.device), 'PYTHONIOENCODING': 'utf-8'})
    summary['nvrtc_arch'] = arch
    v1dir = os.path.join(args.out, 'v1enc')
    py = sys.executable
    if args.reencode or not os.path.exists(os.path.join(v1dir, 'encode_report_v1.json')):
        rc = run_logged([py, '-I', '-X', 'utf8', os.path.join(SRC, 'v1enc.py'), v1dir], env, os.path.join(args.out, 'v1enc.log'), log)
        if rc != 0:
            return K.finish(TEST, args.out, 'FAIL', 'EXPERIMENTAL: v1enc.py exit %d (see v1enc.log)' % rc, summary, log)
    else:
        log('using cached v1 encode', v1dir)
    # (3) bench
    bj = os.path.join(args.out, 'bench_v1.json')
    cmd = [py, '-I', '-X', 'utf8', os.path.join(SRC, 'v1bench.py'), v1dir, t4, bj, str(R), nts, args.opts]
    if args.stages:
        cmd.append(args.stages)
    t0 = time.time()
    rc = run_logged(cmd, env, os.path.join(args.out, 'v1bench.log'), log)
    summary['bench_s'] = round(time.time() - t0, 1)
    if rc != 0 or not os.path.exists(bj):
        return K.finish(TEST, args.out, 'FAIL', 'EXPERIMENTAL: v1bench.py exit %d (see v1bench.log)' % rc, summary, log)
    # (4) summary tables
    with open(os.path.join(args.out, 'bench_v1_table.txt'), 'w', encoding='utf-8') as f:
        rc2 = subprocess.run([py, '-I', '-X', 'utf8', os.path.join(SRC, 'v1summ.py'), bj], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    with open(bj) as f:
        res = json.load(f)
    flags = {'h32_gpu_pass': res.get('h32_gpu', {}).get('pass'),
             'scale_all_byte_exact': all(v.get('byte_exact') for v in res.get('scale_decode', {}).values()) if 'scale_decode' in res else None,
             'weight_all_exact': res.get('weight_all_exact'),
             'original_bytes': res.get('original_bytes'),
             'coded_scale_inputs_unchanged': res.get('coded_scale_inputs_unchanged_after_all_runs')}
    summary['flags'] = flags
    summary['v1summ_exit'] = rc2
    summary['files'] = ['bench_v1.json', 'bench_v1_table.txt', 'v1enc.log', 'v1bench.log', TEST + '.log']
    summary['reference_3070'] = 'tests/gpu/ref/v1_3070/bench_v1_table.txt (team 5, earlier code version)'
    short = 'EXPERIMENTAL untested: weight_all_exact %s, scales exact %s, h32 %s (%s)' % (
        flags['weight_all_exact'], flags['scale_all_byte_exact'], flags['h32_gpu_pass'], env_info.get('gpu', {}).get('name'))
    return K.finish(TEST, args.out, 'DONE', short, summary, log)


if __name__ == '__main__':
    sys.exit(main())
