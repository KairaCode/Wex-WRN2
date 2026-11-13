"""Test kit / audit: WNR3 spec v2.1 test T3 (corruption audit, silent acceptance) on REAL ceil-form records.

Records: the 8 held-out (odd-index) V4.1 experts of the 160-sample manifest that team 3 / ceil-ref used for testing
(L00_E113 ... L39_E156), encoded with a byte copy of team 4 ceil-ref's coder (wnr_common_cr.py; features f ceil form,
g0 constant anchor 120, d coded lengths, and e hash / t tail), model fg_a0_K0_c0 of ceil-ref models/even.npz (trained on
the 80 even-index experts: held-out protocol). Record kinds / integrity rules:
  fg0d   CEIL|LENC              rule R0     x0 = L, final state == L, exact length (today's rule)
  fg0de  CEIL|LENC|HASH         rule HASH   ceil-ref feature e (FNV-1a + lowbias32, 23 bits in x0)
  fg0dh  CEIL|LENC|HASH         rule HASH32 the normative spec 5.1 H32 (MurmurHash3 over ident16 || coded bytes)
  fg0dt  CEIL|LENC|TAIL         rule TAIL   feature t alone (5 tail nibbles / 2 tail bytes in x0, window check)
  fg0dtD CEIL|LENC|TAIL|DIGEST  rule TAILD  TAIL + u32 record digest (spec 5.8)
Cases: team 2's A-J (audit_corrupt.py), + W0/S0 controls, T (tail magnitude change), M1-M3 (length table), N1-N5
(swapped streams / records), K, L (scale streams), O (record in another slot), P1-P3 (truncation / torn writes).
For every accepted corruption: damage after the spec ceil inverse (weights and scales differing, whole-block M' == 5
decision flips, max error in E2M1 units of the source block scale and absolute).

Usage (from the kit root):
  python -I tests/audit/run_audit.py [--root KIT] [--data DIR] [--out DIR] [--quick] [--jobs N] [--shards DIR] ...
CPU only (NumPy). No GPU, no network.
"""
# read by setup/kitenv.py with ast (runner discovery for run_all.ps1); never executed there
KIT_META = {'name': 'audit', 'order': 40, 'needs': ['data'], 'quick_minutes': 10, 'full_minutes': 120,
            'description': 'T3 corruption audit on real ceil-form records (CPU; --jobs auto, ~3 GB RAM per job)'}
import argparse, hashlib, json, math, os, platform, shutil, subprocess, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

TEST = 'audit'
RULES_ALL = ('R0', 'HASH', 'HASH32', 'TAIL', 'TAILD')
STRONG = ('HASH', 'HASH32', 'TAILD')          # spec T3 / 5.8: 0 silent acceptances required in every case
DK_OF_RULE = {'R0': 'fg0d', 'HASH': 'fg0de', 'HASH32': 'fg0dh', 'TAIL': 'fg0dt', 'TAILD': 'fg0dt'}
TARGET_RANDOM_FLIPS = 24000                    # per rule and matrix type (cases A and K)


# ------------------------------------------------------------------------------------------------ logging / env
class Log:
    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8') if path else None

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        if self.f:
            self.f.write(s + '\n')
            self.f.flush()


def _cpu_name():
    try:
        if sys.platform.startswith('win'):
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
        if os.path.exists('/proc/cpuinfo'):
            for line in open('/proc/cpuinfo', encoding='utf-8', errors='replace'):
                if line.lower().startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def _ram_gb():
    try:
        if sys.platform.startswith('win'):
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 2 ** 30, 1)
        if os.path.exists('/proc/meminfo'):
            for line in open('/proc/meminfo'):
                if line.startswith('MemTotal'):
                    return round(int(line.split()[1]) / 2 ** 20, 1)
    except Exception:
        pass
    return None


def environment():
    env = {'os': platform.platform(), 'machine': platform.machine(), 'cpu': _cpu_name(), 'cpu_count': os.cpu_count(),
           'ram_gb': _ram_gb(), 'python': sys.version.split()[0], 'numpy': np.__version__, 'cupy': None, 'gpu': None}
    try:
        from importlib import metadata
        cp = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                    if (d.metadata['Name'] or '').lower().startswith('cupy'))
        env['cupy'] = cp or None                       # not imported: this test does not use the GPU
    except Exception:
        pass
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            env['gpu'] = [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
    except Exception:
        pass
    return env


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ data
def find_sample_dir(args):
    cands = [args.sample] if args.sample else [os.path.join(args.data, 'v41-sample'), args.data]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return cands[0]


def ensure_experts(sample_dir, man, idxs, shards, log):
    """Check (and, with --shards, extract from local safetensors shards) the expert .bin files; sha256 verified."""
    shard_map = None
    status = []
    for i in idxs:
        e = man['experts'][i]
        p = os.path.join(sample_dir, e['file'])
        st = {'idx': i, 'file': e['file'], 'present': os.path.exists(p), 'extracted': False}
        if st['present']:
            st['sha256_ok'] = sha256_file(p) == e['sha256']
        if (not st['present'] or not st['sha256_ok']) and shards:
            if shard_map is None:
                shard_map = {}
                for dp, _, fs in os.walk(shards):
                    for fn in fs:
                        if fn.endswith('.safetensors'):
                            shard_map.setdefault(fn, os.path.join(dp, fn))
            blobs = []
            for t in e['tensors']:
                sp = shard_map.get(t['shard'])
                if sp is None:
                    raise FileNotFoundError(f"shard {t['shard']} not found under --shards")
                with open(sp, 'rb') as f:
                    f.seek(t['offset'])
                    b = f.read(t['bytes'])
                if len(b) != t['bytes']:
                    raise IOError(f"short read in {sp}")
                blobs.append(b)
            data = b''.join(blobs)
            if hashlib.sha256(data).hexdigest() != e['sha256']:
                raise RuntimeError(f"{e['file']}: bytes from the local shards do not match the manifest sha256 "
                                   f"(different checkpoint revision?)")
            os.makedirs(sample_dir, exist_ok=True)
            with open(p + '.part', 'wb') as f:
                f.write(data)
            os.replace(p + '.part', p)
            st.update(present=True, sha256_ok=True, extracted=True)
            log(f"extracted {e['file']} from local shards")
        status.append(st)
    return status


# ------------------------------------------------------------------------------------------------ workers
def _configure(cfg):
    import audit_lib as A
    A.configure(sample_dir=cfg['sample'], manifest=cfg['manifest'], models=cfg['models'], rec_dir=cfg['rec_dir'])
    return A


def encode_one(cfg, idx, kinds, verify, ref):
    """Encode one expert into the record kinds, write them, verify (own full decoder + original ceil-ref decoder)."""
    A = _configure(cfg)
    import audit_rec as R
    W = A.W
    tab = A.load_tables()
    t0 = time.time()
    src = A.load_source(idx)
    stem = src['file'][:-4]
    index_p = os.path.join(cfg['rec_dir'], f'{stem}.json')
    if os.path.exists(index_p):                      # cached from an earlier run in the same --out
        try:
            row = json.load(open(index_p, encoding='utf-8'))
            ok = all(k in row['kinds'] and os.path.exists(os.path.join(cfg['rec_dir'], f'{stem}_{k}.rec'))
                     and sha256_file(os.path.join(cfg['rec_dir'], f'{stem}_{k}.rec')) == row['kinds'][k]['sha256']
                     and (row['kinds'][k].get('verified') or not verify) for k in kinds)
            if ok and row.get('model_sha256') == cfg['model_sha256']:
                row['reused'] = True
                return row
        except Exception:
            pass
    enc = A.encode_kinds(src, tab, kinds)
    tenc = time.time() - t0
    for k, (rec, info) in enc.items():
        with open(os.path.join(cfg['rec_dir'], f'{stem}_{k}.rec'), 'wb') as f:
            f.write(rec)
    row = {'idx': idx, 'file': src['file'], 'layer': src['layer'], 'expert': src['expert'], 'model_id': int(tab.model_id),
           'model_sha256': cfg['model_sha256'], 'ceil': {k: src['finfo'][k] for k in ('blocks', 'even', 'below6', 'overflow')},
           'encode_s': round(tenc, 1), 'kinds': {}, 'reused': False}
    E = R.Expert(idx, tab, kinds=kinds) if verify else None
    for k, (rec, info) in enc.items():
        sha = hashlib.sha256(rec).hexdigest()
        rf = (ref.get(src['file']) or {}).get(k)
        kr = {'flags': int(rec[27]), 'total': len(rec), 'used': int(info['used']) + (4 if k == 'fg0dtD' else 0),
              'length_table': int(info['length_table']), 'scale_payload': int(info['scale_payload']),
              'weight_payload': int(info['weight_payload']), 'sha256': sha,
              'sha256_equal_to_reference': (rf['sha256'] == sha) if rf else None, 'verified': False}
        if verify:
            t1 = time.time()
            d = R.decode_r(E, k, np.frombuffer(rec, np.uint8), tab, use_clean=False, build_slab=True)
            ok_mine = d['status'] == 'accepted' and d['slab'].size == E.raw.size and bool((d['slab'] == E.raw).all())
            kr.update(mine_byte_exact=ok_mine, mine_status=d['status'], mine_stage=d['stage'],
                      mine_ceilref_reject=d['ceilref_reject'], mine_dec_s=round(time.time() - t1, 1))
            ok_cr = True
            if k in ('fg0d', 'fg0de', 'fg0dt'):
                t2 = time.time()
                try:
                    slab, _ = W.decode_record(rec, tab, src['layer'], src['expert'], src['shapes'])
                    kr['ceilref_decoder_byte_exact'] = ok_cr = bool(slab.size == E.raw.size and (slab == E.raw).all())
                except W.DecodeError as ex:
                    kr['ceilref_decoder_byte_exact'] = ok_cr = False
                    kr['ceilref_decoder_error'] = str(ex)
                kr['ceilref_dec_s'] = round(time.time() - t2, 1)
            kr['verified'] = bool(ok_mine and ok_cr)
        if k == 'fg0dtD':
            kr['digest'] = int(info['digest'])
        row['kinds'][k] = kr
    with open(index_p, 'w', encoding='utf-8') as f:
        json.dump(row, f, indent=1)
    return row


def run_cases_worker(cfg, idxs, dks, kinds, cases, pdict, log_path, part_dir):
    """Load the experts for record kinds `kinds` and run `cases` for decoder kinds `dks`. Returns {case: result}."""
    A = _configure(cfg)
    import audit_rec as R
    import cases as CS
    log = Log(log_path)
    tab = A.load_tables()
    P = CS.Params(pdict['per_w'], pdict['per_a'], pdict['per_s'], pdict['rper'])
    t0 = time.time()
    Es = [R.Expert(i, tab, kinds=kinds) for i in idxs]
    log(f"[{','.join(dks)}] loaded {[E.file for E in Es]} kinds {kinds} {time.time() - t0:.0f}s")
    out = {}
    for case in cases:
        t1 = time.time()
        try:
            res = CS.run_case(case, Es, dks, tab, P, log=lambda s: log(f"[{','.join(dks)}] {s}"))
        except Exception as ex:
            res = {'case': case, 'error': f'{type(ex).__name__}: {ex}', 'traceback': traceback.format_exc(), 'by_kind': {}}
            log(f"[{','.join(dks)}] case {case} ERROR {res['error']}")
        res['seconds'] = round(time.time() - t1, 1)
        out[case] = res
        if part_dir:
            A.jdump(res, os.path.join(part_dir, f"{case}__{'_'.join(dks)}.json"))
    return out


# ------------------------------------------------------------------------------------------------ gates
def evaluate(results, rules, enc_rows, verify, val, h32, idxs, P, quick, log):
    import audit_lib as A
    g = {'h32_selftest': h32['ok']}
    if verify:
        g['records_byte_exact'] = all(kr.get('verified') for r in enc_rows for kr in r['kinds'].values())
    else:
        g['records_byte_exact'] = None
    errs = [c for c, v in results.items() if v.get('error')]
    g['cases_without_error'] = not errs
    ctrl = True
    for c in ('W0', 'S0'):
        for rule, v in results.get(c, {}).get('by_kind', {}).items():
            a = v['all']
            ctrl &= a['silent'] == 0 and a['accepted_identical_output'] == a['trials']
    g['controls_clean_all_accepted'] = bool(ctrl)
    strong = {}
    for rule in STRONG:
        if rule not in rules:
            continue
        bad = {}
        for c, v in results.items():
            bk = v.get('by_kind', {}).get(rule)
            if bk is None:
                continue
            s = bk['all']['silent'] if 'all' in bk else bk['silent']
            if s:
                bad[c] = s
        strong[rule] = bad
    g['strong_rules_zero_silent'] = all(not b for b in strong.values()) if strong else None
    g['validate_paths_agree'] = (val['all_agree'] and val['full_consistent']) if val else None
    # trial counts of random single-bit flips per rule and matrix
    flips = {}
    for c in ('A', 'K'):
        for rule, v in results.get(c, {}).get('by_kind', {}).items():
            flips.setdefault(rule, {})[c] = {mk: v[mk]['trials'] for mk in ('w1', 'w2', 'w3')}
    size_ok = all(min(d.values()) >= TARGET_RANDOM_FLIPS for r in flips.values() for d in r.values()) if flips else False
    g['random_flip_target_met'] = bool(size_ok)
    hard = [k for k in ('h32_selftest', 'records_byte_exact', 'cases_without_error', 'controls_clean_all_accepted',
                        'strong_rules_zero_silent', 'validate_paths_agree') if g.get(k) is False]
    if hard:
        status = 'FAIL'
    elif g['strong_rules_zero_silent'] is None or not verify or g['validate_paths_agree'] is None:
        status = 'DONE'
    else:
        status = 'PASS'
    # expected chance acceptances for the hash-based rules (all corruption trials, upper bound)
    exp = {}
    for rule, bits in (('HASH', 23), ('HASH32', 23), ('TAILD', 32)):
        if rule in rules:
            n = 0
            for c, v in results.items():
                bk = v.get('by_kind', {}).get(rule)
                if bk is not None and c not in ('W0', 'S0'):
                    n += bk['all']['trials'] if 'all' in bk else bk['trials']
            exp[rule] = {'corruption_trials': n, 'expected_chance_acceptances': n * 2.0 ** -bits}
    return status, g, strong, flips, hard, exp


def headline(results, rules):
    """Per rule: random-flip silent rate (A, K), worst case, block flips, max error."""
    H = {}
    for rule in rules:
        tot_t = tot_s = bf = 0
        mx_u = mx_a = 0.0
        per_case = {}
        for c, v in results.items():
            bk = v.get('by_kind', {}).get(rule)
            if bk is None:
                continue
            if 'all' in bk:
                a = bk['all']
                per_case[c] = [a['trials'], a['silent'], a['silent_with_block_flip']]
                sd = bk['silent_damage']
                if sd['trial']:
                    mx_u = max(mx_u, max(sd['err_units']))
                    mx_a = max(mx_a, max(sd['err_abs']))
            else:
                per_case[c] = [bk['trials'], bk['silent'], bk['silent_with_block_flip']]
                for x in bk['rows']:
                    if x['status'] == 'accepted' and 'damage' in x:
                        mx_u = max(mx_u, x['damage']['err_units'])
                        mx_a = max(mx_a, x['damage']['err_abs'])
            if c not in ('W0', 'S0'):
                tot_t += per_case[c][0]
                tot_s += per_case[c][1]
                bf += per_case[c][2]
        rf = {}
        for c in ('A', 'K'):
            bk = results.get(c, {}).get('by_kind', {}).get(rule)
            if bk is not None:
                rf[c] = {mk: {'trials': bk[mk]['trials'], 'silent': bk[mk]['silent'], 'rate': bk[mk]['silent_rate'],
                              'ci95': bk[mk]['silent_ci95']} for mk in ('w1', 'w2', 'w3', 'all')}
        H[rule] = {'random_flip': rf, 'per_case_trials_silent_blockflip': per_case,
                   'corruption_trials_total': tot_t, 'silent_total': tot_s, 'silent_with_block_flip_total': bf,
                   'max_err_units_silent': mx_u, 'max_err_abs_silent': mx_a}
    return H


# ------------------------------------------------------------------------------------------------ main
def parse_args():
    ap = argparse.ArgumentParser(description='WNR3 T3 corruption audit on real ceil-form records (CPU).')
    ap.add_argument('--root', default=os.path.dirname(os.path.dirname(HERE)), help='kit root (default: two levels up)')
    ap.add_argument('--data', default=None, help='data dir (default <root>/data); experts in <data>/v41-sample or <data>')
    ap.add_argument('--out', default=None, help='results dir (default <root>/results/audit)')
    ap.add_argument('--quick', action='store_true', help='reduced run (2 experts, sampled trials; minutes)')
    ap.add_argument('--sample', default=None, help='dir with the expert .bin files (overrides --data)')
    ap.add_argument('--shards', default=None, help='dir with the V4.1 safetensors shards: extract missing experts locally')
    ap.add_argument('--rules', default=','.join(RULES_ALL), help='comma list of R0,HASH,HASH32,TAIL,TAILD')
    ap.add_argument('--cases', default='all', help='comma list of case ids (default all)')
    ap.add_argument('--experts', type=int, default=None, help='number of the 8 held-out experts (default 8, quick 2)')
    ap.add_argument('--per', type=int, default=None, help='streams per (expert, matrix) for structured weight cases')
    ap.add_argument('--per-a', type=int, default=None, help='streams per (expert, matrix) for W0 / A (default all)')
    ap.add_argument('--per-s', type=int, default=None, help='scale streams per (expert, matrix, pass) (default all)')
    ap.add_argument('--rper', type=int, default=None, help='record-level trials per expert for M1 / P2')
    ap.add_argument('--jobs', type=int, default=0,
                    help='parallel processes (one per decoder kind, at most 4 useful; ~3 GB RAM each); '
                         '0 = auto: min(4, logical CPUs / 2, 0.7 x RAM / 3 GB), at least 1')
    ap.add_argument('--skip-verify', action='store_true', help='skip the byte-exact decode of the clean records')
    ap.add_argument('--validate-n', type=int, default=None, help='trials per (case, rule, silent/non-silent) re-decoded at record level')
    ap.add_argument('--validate-full', type=int, default=None, help='of those, records run through the full decoders')
    ap.add_argument('--keep-records', action='store_true', help='keep the encoded records (~16.5 MB each) in <out>/records')
    return ap.parse_args()


def _total_ram_bytes():
    try:
        if os.name == 'nt':
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.ullTotalPhys)
        return int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    except Exception:
        return 0


def auto_jobs():
    """min(4 decoder kinds, logical CPUs / 2, 0.7 x RAM / 3 GB), at least 1 (RAM per job ~3 GB is an estimate)."""
    n = os.cpu_count() or 1
    ram = _total_ram_bytes()
    by_ram = int(0.7 * ram // (3 * 2 ** 30)) if ram else 1
    return max(1, min(4, n // 2, by_ram))


def main():
    args = parse_args()
    if args.jobs <= 0:
        args.jobs = auto_jobs()
    args.root = os.path.abspath(args.root)
    args.data = os.path.abspath(args.data or os.path.join(args.root, 'data'))
    out = os.path.abspath(args.out or os.path.join(args.root, 'results', TEST))
    os.makedirs(out, exist_ok=True)
    log = Log(os.path.join(out, f'{TEST}.log'))
    t_start = time.time()
    summary = {'test': TEST, 'spec': 'WNR3-DRAFT-v2.1 section 11 T3 (+ 5.8 TAIL profile gate)', 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
               'argv': sys.argv[1:], 'quick': bool(args.quick)}
    status, short = 'FAIL', ''
    try:
        import audit_lib as A
        import cases as CS
        summary['environment'] = env = environment()
        log(f"environment: {json.dumps(env)}")
        rules = [r.strip() for r in args.rules.split(',') if r.strip()]
        for r in rules:
            if r not in RULES_ALL:
                raise ValueError(f'unknown rule {r}')
        if 'TAIL' in rules and 'TAILD' not in rules:
            rules.append('TAILD')          # same decoder kind; costs nothing extra
        if 'TAILD' in rules and 'TAIL' not in rules:
            rules.append('TAIL')
        rules = [r for r in RULES_ALL if r in rules]
        dks = sorted({DK_OF_RULE[r] for r in rules}, key=lambda k: ('fg0d', 'fg0de', 'fg0dh', 'fg0dt').index(k))
        kinds = list(dks) + (['fg0dtD'] if 'fg0dt' in dks else [])
        cases = list(CS.ALL_CASES) if args.cases == 'all' else [c.strip() for c in args.cases.split(',') if c.strip()]
        for c in cases:
            if c not in CS.ALL_CASES:
                raise ValueError(f'unknown case {c}')
        nexp = args.experts or (2 if args.quick else 8)
        nexp = max(1, min(8, nexp))
        sel = np.unique(np.round(np.linspace(0, 7, nexp)).astype(int)) if nexp > 1 else np.array([0])
        idxs = [A.EXPERT_IDX[i] for i in sel]
        if args.quick:
            P = CS.Params(per_w=args.per or 300, per_a=args.per_a or 1000, per_s=args.per_s or 60,
                          rper={'M1': args.rper or 50, 'P2': args.rper or 20})
            vn = 5 if args.validate_n is None else args.validate_n
            vf = 1 if args.validate_full is None else args.validate_full
        else:
            P = CS.Params(per_w=args.per or 3000, per_a=args.per_a, per_s=args.per_s,
                          rper={'M1': args.rper or 500, 'P2': args.rper or 100} if args.rper else None)
            vn = 25 if args.validate_n is None else args.validate_n
            vf = 3 if args.validate_full is None else args.validate_full
        verify = not args.skip_verify
        summary['config'] = {'rules': rules, 'decoder_kinds': dks, 'record_kinds': kinds, 'cases': cases,
                             'experts_idx': idxs, 'params': P.as_dict(), 'verify_records': verify,
                             'validate_n': vn, 'validate_full': vf, 'jobs': args.jobs,
                             'model': f'models/even.npz:{A.MODEL_KEY} (ceil-ref, trained on the 80 even-index experts)',
                             'anchor': A.ANCHOR0, 'case_doc': {c: CS.CASE_DOC[c] for c in cases}}
        log(f"config: rules {rules} cases {cases} experts {idxs} params {P.as_dict()} jobs {args.jobs}")

        # ---- self tests
        v = A.h32_verification()
        vec_ok = True
        for s, seed, want in ((b'', 0, 0), (b'', 1, 0x514E28B7), (b'hello', 0, 0x248BFA47), (b'Hello, world!', 1234, 0xFAF6CDB3)):
            h = int(A.h32_lanes(np.frombuffer(s, np.uint8)[None, :] if s else np.zeros((1, 0), np.uint8), seed)[0])
            vec_ok &= h == want
        h32 = {'verification': hex(v), 'expected': '0xb0f57ee3', 'vectors_ok': bool(vec_ok), 'ok': bool(v == 0xB0F57EE3 and vec_ok)}
        summary['h32_selftest'] = h32
        log(f"H32 self test: {h32}")

        # ---- data
        kit_man = os.path.join(HERE, 'manifest_v41_sample.json')
        sample = find_sample_dir(args)
        A.configure(sample_dir=sample, manifest=kit_man)
        man = A.manifest()
        log(f"sample dir: {sample}")
        dstat = ensure_experts(sample, man, idxs, args.shards, log)
        summary['data'] = {'sample_dir': sample, 'experts': dstat, 'manifest': 'tests/audit/manifest_v41_sample.json',
                           'repo': man['repo'], 'revision': man['revision']}
        miss = [d['file'] for d in dstat if not d['present'] or not d.get('sha256_ok')]
        if miss:
            raise RuntimeError(f"expert files missing or wrong sha256 in {sample}: {miss}. Copy them from the research PC "
                               f"(v41-sample) or pass --shards <dir with the V4.1 safetensors shards>")
        models = os.path.join(HERE, 'models', 'even.npz')
        rec_dir = os.path.join(out, 'records')
        os.makedirs(rec_dir, exist_ok=True)
        cfg = {'sample': sample, 'manifest': kit_man, 'models': models, 'rec_dir': rec_dir, 'model_sha256': sha256_file(models)}
        A.configure(models=models, rec_dir=rec_dir)
        ref = json.load(open(os.path.join(HERE, 'reference', 'records_sha256.json'), encoding='utf-8'))['records']

        # ---- records
        t0 = time.time()
        jobs = max(1, args.jobs)
        if jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(min(jobs, len(idxs))) as ex:
                enc_rows = list(ex.map(encode_one, [cfg] * len(idxs), idxs, [kinds] * len(idxs), [verify] * len(idxs),
                                       [ref] * len(idxs)))
        else:
            enc_rows = [encode_one(cfg, i, kinds, verify, ref) for i in idxs]
        for r in enc_rows:
            log(f"record {r['file']} reused={r['reused']} encode {r['encode_s']}s: " +
                ', '.join(f"{k} total {kr['total']} verified {kr['verified']} sha=ref {kr['sha256_equal_to_reference']}"
                          for k, kr in r['kinds'].items()))
        summary['records'] = enc_rows
        summary['records_seconds'] = round(time.time() - t0, 1)
        shas = [kr['sha256_equal_to_reference'] for r in enc_rows for kr in r['kinds'].values()]
        summary['records_sha256_equal_to_reference'] = {'equal': shas.count(True), 'different': shas.count(False),
                                                        'no_reference': shas.count(None)}

        # ---- cases
        t0 = time.time()
        part_dir = os.path.join(out, 'parts')
        os.makedirs(part_dir, exist_ok=True)
        pdict = P.as_dict()
        groups = [dks[i::jobs] for i in range(min(jobs, len(dks)))]
        gkinds = [[k for k in kinds if k in g or (k == 'fg0dtD' and 'fg0dt' in g)] for g in groups]
        if len(groups) > 1:
            from concurrent.futures import ProcessPoolExecutor
            logs = [os.path.join(out, f"worker_{'_'.join(g)}.log") for g in groups]
            with ProcessPoolExecutor(len(groups)) as ex:
                futs = [ex.submit(run_cases_worker, cfg, idxs, g, gk, cases, pdict, lp, part_dir)
                        for g, gk, lp in zip(groups, gkinds, logs)]
                parts = [f.result() for f in futs]
            for lp in logs:                       # append worker logs to the main log file
                with open(lp, encoding='utf-8') as f:
                    log.f.write(f.read())
                os.remove(lp)
        else:
            log.f.flush()
            parts = [run_cases_worker(cfg, idxs, dks, kinds, cases, pdict, os.path.join(out, f'{TEST}.log'), part_dir)]
        results = {}
        for c in cases:
            M = None
            for p in parts:
                r = p.get(c)
                if r is None:
                    continue
                if M is None:
                    M = {k: v for k, v in r.items()}
                    M['by_kind'] = dict(r.get('by_kind', {}))
                else:
                    M['by_kind'].update(r.get('by_kind', {}))
                    if r.get('error'):
                        M['error'] = (M.get('error') or '') + ' | ' + r['error']
            results[c] = M
            A.jdump(M, os.path.join(out, f'case_{c}.json'))
        shutil.rmtree(part_dir, ignore_errors=True)
        summary['cases_seconds'] = round(time.time() - t0, 1)

        # ---- validate (path W/S vs record-level decode vs original ceil-ref decoder)
        val = None
        if vn > 0 and any(c in results and results[c] and results[c].get('by_kind') for c in ('A', 'K')):
            import audit_rec as R
            import validate as V
            t0 = time.time()
            tab = A.load_tables()
            Es = [R.Expert(i, tab, kinds=kinds) for i in idxs]
            val = V.validate(Es, tab, P, results, n_each=vn, n_full=vf, log=log)
            A.jdump(val, os.path.join(out, 'validate.json'))
            val = {k: v for k, v in val.items() if k != 'cases'} | {
                'per_case_rule': {c: {r: {k: x[k] for k in ('checked', 'agree', 'silent_checked', 'full_decoder_checked',
                                                            'full_decoder_consistent')} for r, x in d.items()}
                                  for c, d in val['cases'].items()}}
            summary['validate_seconds'] = round(time.time() - t0, 1)
            del Es
        summary['validate'] = val

        # ---- summary
        import summarize as SM
        S, lines = SM.summarize({c: v for c, v in results.items() if v is not None})
        with open(os.path.join(out, 'tables.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        A.jdump(S, os.path.join(out, 'damage_tables.json'))
        for ln in lines:
            log(ln)
        status, gates, strong, flips, hard, exp = evaluate(results, rules, enc_rows, verify,
                                                            summary['validate'] and {'all_agree': val['all_agree'],
                                                                                     'full_consistent': val['full_consistent']},
                                                            h32, idxs, P, args.quick, log)
        summary['gates'] = gates
        summary['strong_rule_silent_by_case'] = strong
        summary['random_flip_trials_per_matrix'] = flips
        summary['expected_chance_acceptances'] = exp
        summary['headline'] = H = headline(results, rules)
        summary['pass_rule'] = ('PASS = H32 self test, clean records byte-exact (own + ceil-ref decoder), W0/S0 controls all '
                                'accepted identical, no case errors, 0 silent acceptances for every strong rule '
                                '(HASH, HASH32, TAILD) in every case, record-level cross-check agrees. R0 and TAIL-alone '
                                'acceptance and damage are reported, not gated. DONE = nothing failed but a gate was not run.')
        parts_s = []
        for r in rules:
            rf = H[r]['random_flip']
            a = ' '.join(f"{c} {rf[c]['all']['silent']}/{rf[c]['all']['trials']}" for c in ('A', 'K') if c in rf)
            parts_s.append(f"{r} {a} all-cases {H[r]['silent_total']}/{H[r]['corruption_trials_total']}"
                           + (f" blockflip {H[r]['silent_with_block_flip_total']}" if H[r]['silent_total'] else ''))
        short = ('quick; ' if args.quick else '') + '; '.join(parts_s)
        if hard:
            short = 'failed: ' + ','.join(hard) + '; ' + short
        if not gates['random_flip_target_met'] and not args.quick:
            short += f'; random-flip target {TARGET_RANDOM_FLIPS}/matrix not met'
    except Exception as ex:
        summary['error'] = f'{type(ex).__name__}: {ex}'
        summary['traceback'] = traceback.format_exc()
        log(summary['traceback'])
        status, short = 'FAIL', summary['error'][:200]
    finally:
        rd = os.path.join(out, 'records')
        if not args.keep_records and os.path.isdir(rd):
            shutil.rmtree(rd, ignore_errors=True)
        summary['status'] = status
        summary['seconds_total'] = round(time.time() - t_start, 1)
        try:
            import audit_lib as A
            A.jdump(summary, os.path.join(out, 'summary.json'))
        except Exception:
            with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=1, default=str)
    log(f'RESULT {TEST} {status} {short}')
    return 0 if status != 'FAIL' else 1


if __name__ == '__main__':
    sys.exit(main())
