"""Shared helpers of the WNR2/WNR3 CPU test kit (tests/cpu): arguments, data discovery, environment, logging, result
files, the per-expert anchor-free count cache, model training, record encode/decode jobs, bootstrap statistics.

Coder: lib/wnr_common.py = byte copy of team 4 ceil-ref wnr_common.py plus team 6's set_chunks() (chunk switch).
Everything here is plain Python 3.10+ / numpy; no GPU, no network.

Counts (cache, shared by every CPU test): per expert and per form (o = stored/plain MXFP4, c = ceil form), weight
counts by ABSOLUTE scale value s (window s0 .. s0+ns-1 per expert, exact: the window covers every scale of the
expert), 81-way context pm*9+mx (pm = mx = 8 at block start) and 4-bit symbol; plus order-0 scale histograms. Any
srel anchor a folds them into the 567-context WNR2 table ctx = (clamp(s-a,-3,3)+3)*81 + pm*9 + mx, and into the
25-bin dbin table of the length predictor (D, etab). Canonical zero (C) = merge symbol 8 into 0 (contexts use
magnitudes only). This is team 5's count_abs.py design with an exact (expert-specific) scale window.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
import traceback

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

LIB = os.path.dirname(os.path.abspath(__file__))
CPU_DIR = os.path.dirname(LIB)
REF_DIR = os.path.join(CPU_DIR, 'ref')
if LIB not in sys.path:
    sys.path.insert(0, LIB)
import wnr_common as W  # noqa: E402

RAW = 18_800_640
W_PER_REC = 3 * 2304 * 5120
S_PER_REC = W_PER_REC // 32
BOOT_B, BOOT_SEED = 10000, 20261107
MAT = ('w1', 'w2', 'w3')


# ------------------------------------------------------------------------------------------------ protocol
def test_indices():
    """The 40 held-out sample experts used by team 3 / team 4 (one per layer, odd manifest indices)."""
    return sorted(4 * l + (1 if l % 2 == 0 else 3) for l in range(40))


def train_indices():
    """The 80 even-indexed sample experts (training set of team 3 / team 4)."""
    return list(range(0, 160, 2))


def spread(seq, n):
    """n items of seq spread evenly (first and last included)."""
    seq = list(seq)
    if n >= len(seq):
        return seq
    if n <= 1:
        return seq[:1]
    return [seq[round(i * (len(seq) - 1) / (n - 1))] for i in range(n)]


# ------------------------------------------------------------------------------------------------ arguments
def base_parser(test_name, script_file, description):
    here = os.path.dirname(os.path.abspath(script_file))
    p = argparse.ArgumentParser(prog=test_name, description=description)
    p.add_argument('--root', default=os.path.abspath(os.path.join(here, '..', '..')),
                   help='kit root (default: two levels above this script)')
    p.add_argument('--data', default=None, help='data dir (default: <root>/data)')
    p.add_argument('--out', default=None, help=f'results dir (default: <root>/results/{test_name})')
    p.add_argument('--quick', action='store_true', help='reduced run (minutes) for a first check')
    p.add_argument('--sample-dir', default=None,
                   help='dir with the 160 sample .bin files (listed in <root>/setup/manifests/v41-sample.json; a '
                        'manifest.json next to them is optional); default: first of <data>/v41-sample, '
                        '<data>/v41_sample, <data>/sample, <data> that holds them')
    p.add_argument('--extra-dir', default=None,
                   help='extra experts dir (70 routed + 6 MTP, setup/manifests/extra.json); default: first of '
                        '<data>/extra, <data>/v41-extra, <data>/team4-data/extra')
    p.add_argument('--cache', default=None, help='count cache dir (default: <root>/results/_cache/cpu_counts)')
    p.add_argument('--jobs', type=int, default=0, help='worker processes (0 = auto from CPU count and RAM)')
    p.add_argument('--limit-test', type=int, default=0, help='(debug/smoke) use only the first N test experts')
    p.add_argument('--limit-train', type=int, default=0, help='(debug/smoke) use only N training experts')
    return p


def finish_args(a, test_name):
    a.root = os.path.abspath(a.root)
    a.data = os.path.abspath(a.data or os.path.join(a.root, 'data'))
    a.out = os.path.abspath(a.out or os.path.join(a.root, 'results', test_name))
    a.cache = os.path.abspath(a.cache or os.path.join(a.root, 'results', '_cache', 'cpu_counts'))
    os.makedirs(a.out, exist_ok=True)
    if a.jobs <= 0:
        a.jobs = auto_jobs()
    return a


_MAN_BY_DIR = {}   # data dir -> manifest chosen by find_sample / find_extra (load_manifest returns it)


def _read_json(p):
    try:
        with open(p, encoding='utf-8-sig') as fh:
            return json.load(fh)
    except Exception:
        return None


def kit_manifest(root, which):
    """The kit's bundled manifest (<root>/setup/manifests/v41-sample.json or extra.json) or None if absent."""
    return _read_json(os.path.join(root, 'setup', 'manifests', 'v41-sample.json' if which == 'sample' else 'extra.json'))


def _sha_list(man):
    return [(e.get('file'), e.get('sha256')) for e in man.get('experts', [])] if isinstance(man, dict) else None


def _present(d, man):
    """number of manifest experts present in d with the right size (the sha256 is checked later, per file, by load_checked)"""
    n = 0
    for e in man['experts']:
        p = os.path.join(d, e['file'])
        if os.path.isfile(p) and os.path.getsize(p) == sum(t['bytes'] for t in e['tensors']):
            n += 1
    return n


def _find_set(a, which, n_expected, cands, R=None):
    """Data dir of one expert set. The reference list of experts is the kit's own manifest (setup/manifests), so the
    .bin files alone are enough (no manifest.json needed next to them). A manifest.json in the data dir is used only
    when the kit manifest is not available, or when its (file, sha256) list equals the kit's; a foreign one is ignored.
    Picks the first candidate holding every expert, else the one holding most (missing files then fail in
    load_checked, which also verifies every sha256)."""
    say = R.log if R is not None else (lambda *x: print(*x, flush=True))
    kit = kit_manifest(a.root, which)
    if kit is not None and len(kit.get('experts', [])) != n_expected:
        say(f'WARNING kit manifest for {which} has {len(kit.get("experts", []))} experts, expected {n_expected}')
    best = None
    for d in [c for c in cands if c and os.path.isdir(c)]:
        local = _read_json(os.path.join(d, 'manifest.json'))
        if kit is not None:
            if local is not None and _sha_list(local) != _sha_list(kit):
                say(f'note: {os.path.join(d, "manifest.json")} differs from the kit manifest; ignored (kit manifest used)')
            man, src = kit, 'kit'
        elif local is not None and len(local.get('experts', [])) == n_expected:
            man, src = local, 'local manifest.json (kit manifest missing)'
        else:
            continue
        n = _present(d, man)
        if n == len(man['experts']):
            best = (d, man, src, n)
            break
        if n > 0 and (best is None or n > best[3]):
            best = (d, man, src, n)
    if best is None:
        return None
    d, man, src, n = best
    d = os.path.abspath(d)
    if n < len(man['experts']):
        say(f'WARNING {which}: only {n} of {len(man["experts"])} experts present in {d}; runs needing the others fail')
    say(f'{which} dir {d}: {n}/{len(man["experts"])} experts present, manifest = {src}')
    _MAN_BY_DIR[d] = man
    return d


def find_sample(a, R=None):
    cands = [a.sample_dir] if a.sample_dir else [os.path.join(a.data, 'v41-sample'), os.path.join(a.data, 'v41_sample'),
                                                 os.path.join(a.data, 'sample'), a.data]
    d = _find_set(a, 'sample', 160, cands, R)
    if d:
        return d
    raise SystemExit('160-expert sample not found; looked in: ' + ', '.join(str(c) for c in cands) +
                     ' for the L??_E???.bin files of setup/manifests/v41-sample.json'
                     ' (run setup/fetch_data.py, copy the .bin files there, or pass --sample-dir)')


def find_extra(a, R=None):
    cands = [a.extra_dir] if a.extra_dir else [os.path.join(a.data, 'extra'), os.path.join(a.data, 'v41-extra'),
                                               os.path.join(a.data, 'team4-data', 'extra')]
    d = _find_set(a, 'extra', 76, cands, R)
    if d:
        return d
    raise SystemExit('extra experts (76 = 70 routed + 6 MTP) not found; looked in: ' + ', '.join(str(c) for c in cands) +
                     ' for the files of setup/manifests/extra.json (run setup/fetch_data.py with --extra, or pass --extra-dir)')


def load_manifest(d):
    m = _MAN_BY_DIR.get(os.path.abspath(d))
    if m is not None:
        return m
    return json.load(open(os.path.join(d, 'manifest.json')))


# ------------------------------------------------------------------------------------------------ environment
def _ram_bytes():
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
    n = os.cpu_count() or 1
    ram = _ram_bytes() / 2 ** 30
    by_ram = int(ram // 2) if ram else 4          # ~1 GB peak per encode/decode worker, keep 2x headroom
    return max(1, min(max(1, n // 2), max(1, by_ram), 16))


def _cpu_name():
    try:
        if os.name == 'nt':
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return str(winreg.QueryValueEx(k, 'ProcessorNameString')[0]).strip()
        if os.path.exists('/proc/cpuinfo'):
            for line in open('/proc/cpuinfo'):
                if line.lower().startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _gpu():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap',
                            '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
        return f'nvidia-smi rc {r.returncode}'
    except Exception as ex:
        return f'none ({type(ex).__name__})'


def _pkg_versions(prefixes):
    out = {}
    try:
        from importlib import metadata
        for d in metadata.distributions():
            n = (d.metadata.get('Name') or '').lower()
            if any(n.startswith(p) for p in prefixes):
                out[n] = d.version
    except Exception as ex:
        out['error'] = str(ex)
    return out


def environment():
    return {'os': platform.platform(), 'system': platform.system(), 'machine': platform.machine(),
            'cpu': _cpu_name(), 'logical_cpus': os.cpu_count(), 'ram_gb': round(_ram_bytes() / 2 ** 30, 1),
            'gpu': _gpu(), 'python': sys.version.split()[0], 'python_impl': platform.python_implementation(),
            'numpy': np.__version__, 'cupy': _pkg_versions(('cupy',)) or 'not installed',
            'nvrtc': _pkg_versions(('nvidia-cuda-nvrtc',)) or 'not installed'}


# ------------------------------------------------------------------------------------------------ run context
class Run:
    """Log file + summary.json + the final RESULT line. Use as: with Run(name, args) as R: ...; R.result(...)"""

    def __init__(self, name, args):
        self.name, self.args = name, args
        self.t0 = time.time()
        self.logf = open(os.path.join(args.out, f'{name}.log'), 'w', encoding='utf-8')
        self.summary = {'test': name, 'mode': 'quick' if args.quick else 'full',
                        'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'args': {k: v for k, v in vars(args).items()},
                        'environment': environment(), 'checks': {}}
        self.status = None
        self.short = ''
        self.log(f'== {name} ({self.summary["mode"]}) root={args.root} out={args.out} jobs={args.jobs}')
        self.log('environment ' + json.dumps(self.summary['environment']))

    def log(self, *a):
        s = ' '.join(str(x) for x in a)
        line = f'[{time.time() - self.t0:8.1f}s] {s}'
        print(line, flush=True)
        self.logf.write(line + '\n')
        self.logf.flush()

    def check(self, name, ok, detail=None, hard=True):
        """hard checks decide PASS/FAIL; soft ones (hard=False) are reported only. ok None = skipped."""
        self.summary['checks'][name] = {'ok': ok, 'hard': hard, 'detail': detail}
        self.log(f'check {name}: {"SKIP" if ok is None else ("ok" if ok else ("FAIL" if hard else "differs (soft)"))}'
                 + (f' {detail}' if detail is not None else ''))
        return ok

    def write(self):
        self.summary['elapsed_s'] = round(time.time() - self.t0, 1)
        self.summary['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
        p = os.path.join(self.args.out, 'summary.json')
        tmp = p + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, indent=1, default=_json_default)
        os.replace(tmp, p)

    def result(self, short, status=None):
        hard = [v['ok'] for v in self.summary['checks'].values() if v['hard'] and v['ok'] is not None]
        if status is None:
            status = 'PASS' if hard and all(hard) else ('FAIL' if hard else 'DONE')
        self.status, self.short = status, short
        self.summary['result'] = status
        self.summary['result_line'] = f'RESULT {self.name} {status} {short}'

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is not None:
            if issubclass(et, KeyboardInterrupt):
                self.result('interrupted', 'FAIL')
            elif issubclass(et, SystemExit):
                self.log(f'stopped: {ev}')
                self.result(f'stopped: {ev}', 'FAIL')
            else:
                self.summary['exception'] = ''.join(traceback.format_exception(et, ev, tb))
                self.log('EXCEPTION', self.summary['exception'])
                self.result(f'exception {et.__name__}: {ev}', 'FAIL')
        if self.status is None:
            self.result('no result', 'FAIL')
        self.write()
        self.log(f'summary {os.path.join(self.args.out, "summary.json")}')
        print(self.summary['result_line'], flush=True)
        self.logf.write(self.summary['result_line'] + '\n')
        self.logf.close()
        return et is not None and not issubclass(et, KeyboardInterrupt)

    def exit_code(self):
        return 0 if self.status in ('PASS', 'DONE') else 1


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def jdump(obj):
    return json.dumps(obj, default=_json_default)


# ------------------------------------------------------------------------------------------------ parallel jobs
def run_jobs(fn, jobs, n_workers, R=None, label='job', initializer=None, initargs=()):
    """Run fn(job) for every job; yields (job, result) as they complete. n_workers == 1 runs inline."""
    t0 = time.time()
    n = len(jobs)
    if n_workers <= 1 or n <= 1:
        if initializer is not None:
            initializer(*initargs)
        for k, j in enumerate(jobs):
            r = fn(j)
            if R is not None:
                R.log(f'{label} {k + 1}/{n} done ({time.time() - t0:.0f}s)')
            yield j, r
        return
    import concurrent.futures as cf
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with cf.ProcessPoolExecutor(max_workers=min(n_workers, n), mp_context=ctx, initializer=initializer,
                                initargs=initargs) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        done = 0
        for f in cf.as_completed(futs):
            done += 1
            r = f.result()
            if R is not None:
                R.log(f'{label} {done}/{n} done ({time.time() - t0:.0f}s)')
            yield futs[f], r


# ------------------------------------------------------------------------------------------------ data
def sha256_file_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_checked(data_dir, e):
    """raw bytes + matrices of one expert; checks the whole-file sha256 and every tensor sha256 of the manifest."""
    raw, mats = W.load_expert(data_dir, e)
    ok = True
    if 'sha256' in e:
        ok &= sha256_file_bytes(raw.tobytes()) == e['sha256']
    off = 0
    for d in e['tensors']:
        if 'sha256' in d:
            ok &= sha256_file_bytes(raw[off:off + d['bytes']].tobytes()) == d['sha256']
        off += d['bytes']
    return raw, mats, bool(ok)


def canon_bytes(raw, nscale_bytes):
    """Value-exact reference for canonical-zero records: every FP4 nibble 8 (-0) -> 0 (ceil-ref run_ceil.py)."""
    out = raw.copy()
    w = out[nscale_bytes:]
    lo = w & 15
    hi = w >> 4
    lo = np.where(lo == 8, 0, lo)
    hi = np.where(hi == 8, 0, hi)
    out[nscale_bytes:] = (lo | (hi << 4)).astype(np.uint8)
    return out


# ------------------------------------------------------------------------------------------------ counts
def _ctx81(sym):
    blk = sym.reshape(-1, 32)
    mag = (blk & 7).astype(np.int32)
    pm = np.empty_like(mag)
    pm[:, 0] = 8
    pm[:, 1:] = mag[:, :-1]
    cm = np.maximum.accumulate(mag, axis=1)
    mx = np.empty_like(mag)
    mx[:, 0] = 8
    mx[:, 1:] = cm[:, :-1]
    return blk, pm * 9 + mx


def _abs_counts(sym, sc, s0, ns):
    blk, c81 = _ctx81(sym)
    idx = ((sc.astype(np.int32) - s0)[:, None] * 81 + c81) * 16 + blk.astype(np.int32)
    return np.bincount(idx.ravel(), minlength=ns * 81 * 16).reshape(ns, 81, 16).astype(np.int32)


COUNT_VERSION = 2


def count_expert(job):
    """Worker: compute (or reuse) the cached counts of one expert. job = (set, data_dir, index, entry, cache_dir)."""
    setname, data_dir, i, e, cache = job
    base = os.path.join(cache, f"{setname}_{e['file'].replace('.bin', '')}_{e.get('sha256', 'nosha')[:12]}")
    if os.path.exists(base + '.json') and os.path.exists(base + '.npz'):
        try:
            info = json.load(open(base + '.json'))
            if info.get('version') == COUNT_VERSION:
                return base
        except Exception:
            pass
    t0 = time.time()
    raw, mats, sha_ok = load_checked(data_dir, e)
    info = {'version': COUNT_VERSION, 'set': setname, 'index': i, 'file': e['file'], 'layer': e['layer'],
            'expert': e['expert'], 'kind': e.get('kind', 'layers'), 'sha_ok': sha_ok, 'm': []}
    forms = []
    for m, mt in enumerate(mats):
        tsym, tsc, ci = W.ceil_forward(mt['sym'], mt['sc'])
        ceil_ok = ci['below6'] == 0 and ci['overflow'] == 0
        rt_ok = None
        if ceil_ok:
            isym, isc = W.ceil_inverse(tsym, tsc)
            rt_ok = bool((isym == mt['sym']).all() and (isc == mt['sc']).all())
            del isym, isc
        else:
            tsym, tsc = mt['sym'], mt['sc']          # fallback: a ceil record of this expert is coded plain
        forms.append((mt['sym'], mt['sc'], tsym, tsc))
        sh_o = np.bincount(mt['sc'], minlength=256)[:256]
        sh_c = np.bincount(tsc, minlength=256)[:256]
        info['m'].append({'blocks': ci['blocks'], 'even': ci['even'], 'below6': ci['below6'], 'overflow': ci['overflow'],
                          'ceil_ok': bool(ceil_ok), 'roundtrip_ok': rt_ok,
                          'med_o': W.lower_median(mt['sc']), 'pair_o': W.anchor_pair(mt['sc']), 'mode_o': int(np.argmax(sh_o)),
                          'med_c': W.lower_median(tsc), 'pair_c': W.anchor_pair(tsc), 'mode_c': int(np.argmax(sh_c)),
                          'smin_o': int(mt['sc'].min()), 'smax_o': int(mt['sc'].max()),
                          'smin_c': int(tsc.min()), 'smax_c': int(tsc.max()), 'neg0': int((mt['sym'] == 8).sum())})
    s0 = min(min(x['smin_o'], x['smin_c']) for x in info['m'])
    s1 = max(max(x['smax_o'], x['smax_c']) for x in info['m'])
    ns = s1 - s0 + 1
    Wo = np.stack([_abs_counts(f[0], f[1], s0, ns) for f in forms])
    Wc = np.stack([_abs_counts(f[2], f[3], s0, ns) for f in forms])
    So = np.stack([np.bincount(f[1], minlength=256)[:256] for f in forms]).astype(np.int64)
    Sc = np.stack([np.bincount(f[3], minlength=256)[:256] for f in forms]).astype(np.int64)
    info['s0'], info['ns'] = int(s0), int(ns)
    info['ceil_ok'] = all(x['ceil_ok'] for x in info['m'])
    info['count_s'] = round(time.time() - t0, 2)
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(base + '.npz', Wo=Wo, Wc=Wc, So=So, Sc=Sc)
    tmp = base + '.json.tmp'
    json.dump(info, open(tmp, 'w'))
    os.replace(tmp, base + '.json')
    return base


def load_counts(base):
    info = json.load(open(base + '.json'))
    z = np.load(base + '.npz')
    for k in ('Wo', 'Wc', 'So', 'Sc'):
        info[k] = z[k]
    info['key'] = f"{info['set']}_{info['file'].replace('.bin', '')}"
    return info


def ensure_counts(R, items, cache, jobs):
    """items: list of (set, data_dir, index, entry). Returns {key: counts dict} (key = '<set>_<file stem>')."""
    jl = [(s, d, i, e, cache) for s, d, i, e in items]
    R.log(f'counts: {len(jl)} experts (cache {cache})')
    out = {}
    for j, base in run_jobs(count_expert, jl, jobs, R, 'count'):
        c = load_counts(base)
        out[c['key']] = c
    return out


def fold(Wm, s0, a):
    """[ns, 81, 16] absolute-scale counts -> [567, 16] WNR2 context counts for srel anchor a."""
    ns = Wm.shape[0]
    sr = np.clip(s0 + np.arange(ns) - int(a), -3, 3) + 3
    out = np.zeros((7, 81, 16), np.int64)
    for k in range(7):
        sel = sr == k
        if sel.any():
            out[k] = Wm[sel].sum(0, dtype=np.int64)
    return out.reshape(W.NCTX, 16)


def fold_db(Wm, s0, a):
    """[ns, 81, 16] -> [25, 567, 16] counts by (dbin = clamp(s-a,-12,12)+12, ctx, symbol) (length predictor D)."""
    ns = Wm.shape[0]
    s = s0 + np.arange(ns)
    db = np.clip(s - int(a), -W.DBC, W.DBC) + W.DBC
    sr = np.clip(s - int(a), -3, 3) + 3
    out = np.zeros((W.NDB, W.NCTX, 16), np.int64)
    for j in range(ns):
        out[db[j], sr[j] * 81:(sr[j] + 1) * 81] += Wm[j]
    return out


def merge_c(x):
    x = x.copy()
    x[..., 0] += x[..., 8]
    x[..., 8] = 0
    return x


def anchors_of(cnt, form, rule):
    """rule: 'med' | 'pair' | 'c<N>' (constant) | 'pair+1' | tuple/list of 3 ints (explicit)."""
    if isinstance(rule, (tuple, list)):
        return [int(x) for x in rule]
    out = []
    for m in range(3):
        mi = cnt['m'][m]
        if rule == 'med':
            out.append(int(mi['med_' + form]))
        elif rule == 'pair':
            out.append(int(mi['pair_' + form]))
        elif rule == 'pair+1':
            out.append(int(mi['pair_o']) + 1)
        elif rule.startswith('c') and rule[1:].isdigit():
            out.append(int(rule[1:]))
        else:
            raise ValueError(rule)
    return out


def expert_ctx(cnt, form, rule, c=False):
    """-> (C [3,567,16] int64, S [3,256] int64, anchors)."""
    a = anchors_of(cnt, form, rule)
    Wx = cnt['Wc'] if form == 'c' else cnt['Wo']
    C = np.stack([fold(Wx[m], cnt['s0'], a[m]) for m in range(3)])
    if c:
        C = merge_c(C)
    S = (cnt['Sc'] if form == 'c' else cnt['So']).astype(np.int64)
    return C, S, a


def train_model(cnts, form, rule, c=False, etab=False, anchors_by_key=None):
    """Train one WNR2-style 567-context model set (+ order-0 scales, + etab for D) from counts.
    Same arithmetic as team 4 ceil-ref train_ceil.py (a = 0, K = 0) / team 3 train_models.py.
    rule applies to every expert unless anchors_by_key[key] gives explicit anchors."""
    C = np.zeros((3, W.NCTX, 16), np.int64)
    S = np.zeros((3, 256), np.int64)
    A = np.zeros((3, W.NDB, W.NCTX, 16), np.int64) if etab else None
    NB = np.zeros((3, W.NDB), np.int64) if etab else None
    census = {}
    for cn in cnts:
        r = anchors_by_key[cn['key']] if anchors_by_key and cn['key'] in anchors_by_key else rule
        a = anchors_of(cn, form, r)
        Wx = cn['Wc'] if form == 'c' else cn['Wo']
        Sx = cn['Sc'] if form == 'c' else cn['So']
        for m in range(3):
            C[m] += fold(Wx[m], cn['s0'], a[m])
            census[f'{MAT[m]}={a[m]}'] = census.get(f'{MAT[m]}={a[m]}', 0) + 1
            if etab:
                A[m] += fold_db(Wx[m], cn['s0'], a[m])
                db = np.clip(np.arange(256) - a[m], -W.DBC, W.DBC) + W.DBC
                NB[m] += np.bincount(db, weights=Sx[m], minlength=W.NDB).astype(np.int64)
        S += Sx.astype(np.int64)
    if c:
        C = merge_c(C)
        if etab:
            A = merge_c(A)
    wf = W.quantize(C)
    scf = W.quantize(S)
    et = None
    if etab:
        et = np.zeros((3, W.NDB))
        for m in range(3):
            bits = (A[m].astype(np.float64) * -np.log2(wf[m] / W.P)[None]).sum((1, 2))
            with np.errstate(invalid='ignore', divide='ignore'):
                e_m = bits / NB[m]
            good = NB[m] > 0
            idx = np.arange(W.NDB)
            gi = idx[good]
            near = gi[np.abs(idx[:, None] - gi[None, :]).argmin(1)]
            et[m] = e_m[near]
        et = np.round(et * 256).astype(np.int64)
    return {'wf': wf.reshape(-1, 16), 'scf': scf, 'etab': et, 'C': C, 'S': S, 'census': census, 'n_train': len(cnts),
            'form': form, 'rule': str(rule), 'c': bool(c)}


def est_bits(model, C, S):
    """Code length (bits) of count arrays under a model: (weights [3], scales [3])."""
    lw = -np.log2(model['wf'].reshape(3, W.NCTX, 16) / W.P)
    ls = -np.log2(model['scf'] / W.P)
    return (C * lw).sum((1, 2)), (S * ls).sum(1)


def compare_model(mine, ref_npz, key):
    """Model equality vs a reference npz entry (key + _wf/_scf/_etab)."""
    out = {}
    for s, arr in (('_wf', mine['wf']), ('_scf', mine['scf']), ('_etab', mine['etab'])):
        if arr is None or key + s not in ref_npz.files:
            continue
        ref = ref_npz[key + s]
        eq = bool(ref.shape == np.asarray(arr).shape and (ref == arr).all())
        out[s[1:]] = eq if eq else {'equal': False, 'n_diff': int((ref != arr).sum()) if ref.shape == np.asarray(arr).shape else 'shape'}
    return out


def model_from_ref(ref_npz, key):
    return {'wf': ref_npz[key + '_wf'], 'scf': ref_npz[key + '_scf'],
            'etab': ref_npz[key + '_etab'] if key + '_etab' in ref_npz.files else None, 'ref_key': key}


def model_arrays(model, wnr2=False):
    """Picklable description of the tables of a model (built into wnr_common.Tables inside the workers)."""
    return {'wf': np.asarray(model['wf']), 'scf': np.asarray(model['scf']),
            'etab': None if (wnr2 or model.get('etab') is None) else np.asarray(model['etab']), 'wnr2': bool(wnr2)}


_TABLES = {}
_MODELS = {}


def init_models(models):
    _MODELS.clear()
    _MODELS.update(models)
    _TABLES.clear()


def tables(name):
    if name not in _TABLES:
        m = _MODELS[name]
        if m['wnr2']:
            _TABLES[name] = W.Tables(m['wf'], m['scf'])
        else:
            _TABLES[name] = W.Tables(m['wf'], m['scf'], None, m['etab'], K=0, nctx=W.NCTX)
    return _TABLES[name]


def model_id_of(m):
    t = W.Tables(m['wf'], m['scf']) if m['wnr2'] else W.Tables(m['wf'], m['scf'], None, m['etab'], K=0, nctx=W.NCTX)
    return t.model_id, len(t.blob)


# ------------------------------------------------------------------------------------------------ encode / decode
def anchor_value(rule, sc):
    """Anchor of one matrix from its coding-domain scales: 'med' lower median, 'pair' densest adjacent pair,
    'c<N>' constant, int explicit."""
    if isinstance(rule, (int, np.integer)):
        return int(rule)
    if rule == 'med':
        return W.lower_median(sc)
    if rule == 'pair':
        return W.anchor_pair(sc)
    if isinstance(rule, str) and rule.startswith('c') and rule[1:].isdigit():
        return int(rule[1:])
    raise ValueError(rule)


def encode_with_anchors(mats, tab, cfg, layer, expert, rule, timing=None):
    """wnr_common.encode_record with the srel anchor of every matrix forced by rule (None = coder default:
    lower median, or densest pair when cfg.g). rule may be a list of 3 per-matrix rules/ints. The anchor is
    written in the record's median[m] byte; decoders read only that byte."""
    if rule is None:
        return W.encode_record(mats, tab, cfg, layer, expert, timing)
    orig = W.prepare
    rules = list(rule) if isinstance(rule, (list, tuple)) else [rule] * 3

    def prep(mats_, f, c, g):
        syms, scs, anchors, finfo = orig(mats_, f, c, g)
        return syms, scs, [anchor_value(rules[m], scs[m]) for m in range(3)], finfo
    W.prepare = prep
    try:
        return W.encode_record(mats, tab, cfg, layer, expert, timing)
    finally:
        W.prepare = orig


FG = dict(f=True, g=True)
CONFIGS = {
    'wnr2': (dict(wnr2=True), None),
    'f': (dict(f=True), None),
    'fg0': (dict(**FG), 'c120'), 'fg0d': (dict(d=True, **FG), 'c120'),
    'fg0e': (dict(e=True, **FG), 'c120'), 'fg0de': (dict(d=True, e=True, **FG), 'c120'),
    'fg0t': (dict(t=True, **FG), 'c120'), 'fg0dt': (dict(d=True, t=True, **FG), 'c120'),
    'fg0c': (dict(c=True, **FG), 'c120'), 'fg0cd': (dict(c=True, d=True, **FG), 'c120'),
    'fg0ce': (dict(c=True, e=True, **FG), 'c120'), 'fg0cde': (dict(c=True, d=True, e=True, **FG), 'c120'),
    'fg0ct': (dict(c=True, t=True, **FG), 'c120'), 'fg0cdt': (dict(c=True, d=True, t=True, **FG), 'c120'),
    # plain ceil-form with an explicit rule (anchor given per job): 'fgx' / 'fgxd' / 'fgxde'
    'fgx': (dict(**FG), None), 'fgxd': (dict(d=True, **FG), None), 'fgxde': (dict(d=True, e=True, **FG), None),
}
D_SIBLING = {'fg0': 'fg0d', 'fg0e': 'fg0de', 'fg0t': 'fg0dt', 'fg0c': 'fg0cd', 'fg0ce': 'fg0cde', 'fg0ct': 'fg0cdt',
             'fgx': 'fgxd'}


def make_config(name):
    kw, _ = CONFIGS[name]
    return W.Config(name, **kw)


def record_row(e, idx, cname, cfg, tab, info, rec, slab, derr, raw, craw, tm, chunk, parent=None):
    ref = craw if cfg.c else raw
    ci = info.get('ceil')
    st_m = None
    return {'idx': idx, 'file': e['file'], 'layer': e['layer'], 'expert': e['expert'], 'kind': e.get('kind', 'layers'),
            'config': cname, 'chunk': list(chunk), 'flags': cfg.flags, 'rec_flags': info['rec_flags'], 'anchors': info['meds'],
            'model_id': tab.model_id, 'model_blob_bytes': len(tab.blob),
            'ceil': None if ci is None else {'applied': ci['applied'], 'even_blocks': ci['even'], 'blocks': ci['blocks'],
                                             'below6': ci['below6'], 'overflow': ci['overflow']},
            'sizes': {'header': info['header'], 'length_table': info['length_table'], 'scale_payload': info['scale_payload'],
                      'weight_payload': info['weight_payload'], 'used': info['used'], 'padding': info['padding'],
                      'total': info['total'], 'record_bytes_check': len(rec)},
            'record_sha256': hashlib.sha256(rec).hexdigest(),
            'est': {'weight_bits': info['est_weight_bits'], 'scale_bits': info['est_scale_bits']},
            'verify': {'decoded': slab is not None or derr is not None, 'decode_error': derr,
                       'equal_to_reference': None if slab is None and derr is None else bool(slab is not None and slab.size == ref.size and (slab == ref).all()),
                       'byte_exact_vs_source': None if slab is None and derr is None else bool(slab is not None and slab.size == raw.size and (slab == raw).all()),
                       'mode': 'value-exact (8->0)' if cfg.c else 'byte-exact'},
            'timing': {k: (round(v, 3) if isinstance(v, float) else v) for k, v in tm.items()},
            'payload_parent': parent, 'len_bits': info.get('len_bits'),
            'len_resid_std': [info.get('len_resid_std_s'), info.get('len_resid_std_w')],
            'lt_parts': {k: info.get(k) for k in ('lt_scale_part_bytes', 'lt_weight_part_bytes', 'lt_scale_bits', 'lt_weight_bits')},
            'n_weight_streams': info['n_weight_streams'], 'n_scale_streams': info['n_scale_streams'],
            'wlen': {'mean': float(info['wlens'].mean()), 'std': float(info['wlens'].std()), 'max': int(info['wlens'].max())},
            'slen': {'mean': float(info['slens'].mean()), 'std': float(info['slens'].std()), 'max': int(info['slens'].max())},
            'per_matrix_bytes': st_m}


def encode_job(job):
    """Worker: encode one expert under several configs, build D siblings from the same payload, decode, verify.
    job: {data_dir, idx, entry, items: [{config, model, anchor (rule or None), decode: bool, sib_decode: bool}],
          chunk: (cw, cs)}. Returns {'rows': [...], 'sha_ok': bool}."""
    e = job['entry']
    cw, cs = job.get('chunk', (2048, 512))
    W.set_chunks(cw, cs)
    t0 = time.perf_counter()
    raw, mats, sha_ok = load_checked(job['data_dir'], e)
    tload = time.perf_counter() - t0
    shapes = [(mt['R'], mt['C']) for mt in mats]
    nsb = sum(mt['sc'].size for mt in mats)
    craw = canon_bytes(raw, nsb)
    rows = []
    for it in job['items']:
        cname = it['config']
        cfg = make_config(cname)
        tab = tables(it['model'])
        rule = it.get('anchor', CONFIGS[cname][1])
        tm = {}
        rec, info, st, base_info = encode_with_anchors(mats, tab, cfg, e['layer'], e['expert'], rule, tm)
        enc_tm = dict(tm)
        pm = per_matrix_bytes(st, info)
        todo = [(cname, cfg, rec, info, None, it.get('decode', True))]
        sib = it.get('sibling', D_SIBLING.get(cname))
        if sib and it.get('build_sibling', True):
            scfg = make_config(sib)
            srec, sinfo = W.build_record(st, tab, scfg, base_info)
            todo.append((sib, scfg, srec, sinfo, cname, it.get('sib_decode', True)))
        for nm, c2, r2, inf2, parent, dec in todo:
            tm2 = dict(enc_tm)
            if parent is not None:
                tm2['enc_total_s'] = enc_tm['enc_total_s'] + inf2['build_s']
            slab = derr = None
            if dec:
                try:
                    slab, _ = W.decode_record(r2, tab, e['layer'], e['expert'], shapes, tm2)
                except W.DecodeError as ex:
                    derr = str(ex)
            row = record_row(e, job['idx'], nm, c2, tab, inf2, r2, slab, derr, raw, craw, tm2, (cw, cs), parent)
            row['model'] = it['model']
            row['label'] = it.get('label')
            row['per_matrix_bytes'] = pm
            row['source_sha256_ok'] = sha_ok
            row['timing']['load_s'] = round(tload, 3)
            rows.append(row)
            del slab
        del rec, st, base_info
    W.set_chunks(2048, 512)
    return {'rows': rows, 'sha_ok': sha_ok}


def per_matrix_bytes(st, info):
    """Coded payload bytes (scale + weight streams) of each matrix of a record."""
    out = []
    for m in range(3):
        out.append(int(info['slens'][st['smat'] == m].sum() + info['wlens'][st['wmat'] == m].sum()))
    return out


# ------------------------------------------------------------------------------------------------ statistics
def boot_ci(x, rng, B=BOOT_B):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return [None, None]
    if x.size == 1:
        return [float(x[0]), float(x[0])]
    idx = rng.integers(0, x.size, (B, x.size))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def describe_delta(d, rng, raw=RAW):
    d = np.asarray(d, np.float64)
    ci = boot_ci(d, rng)
    return {'mean_bytes': float(d.mean()), 'CI95_bytes': ci, 'mean_pct_of_raw': float(100 * d.mean() / raw),
            'CI95_pct_of_raw': [None if v is None else 100 * v / raw for v in ci],
            'median_bytes': float(np.median(d)), 'min_bytes': float(d.min()), 'max_bytes': float(d.max()),
            'n_smaller': int((d < 0).sum()), 'n_equal': int((d == 0).sum()), 'n_larger': int((d > 0).sum()), 'n': int(d.size),
            'bootstrap': {'resamples': BOOT_B, 'seed': BOOT_SEED}}


def config_table(rows):
    """rows of one config -> mean sizes and rates."""
    tot = np.array([r['sizes']['total'] for r in rows], np.float64)
    used = np.array([r['sizes']['used'] for r in rows], np.float64)
    wp = np.array([r['sizes']['weight_payload'] for r in rows], np.float64)
    sp = np.array([r['sizes']['scale_payload'] for r in rows], np.float64)
    lt = np.array([r['sizes']['length_table'] for r in rows], np.float64)
    pad = np.array([r['sizes']['padding'] for r in rows], np.float64)
    dec = [r for r in rows if r['verify']['decoded']]
    ok = sum(bool(r['verify']['equal_to_reference']) for r in dec)
    enc_t = [r['timing'].get('enc_total_s') for r in rows if r['timing'].get('enc_total_s') is not None]
    dec_t = [r['timing'].get('dec_total_s') for r in rows if r['timing'].get('dec_total_s') is not None]
    return {'n_experts': len(rows), 'decoded': len(dec), 'decode_ok': ok,
            'decode_check': f"{ok}/{len(dec)} equal to reference ({rows[0]['verify']['mode']})" if dec else 'not decoded',
            'decode_errors': [r['verify']['decode_error'] for r in dec if r['verify']['decode_error']],
            'source_sha256_ok': f"{sum(bool(r.get('source_sha256_ok')) for r in rows)}/{len(rows)}",
            'ceil_fallback_records': [r['file'] for r in rows if r['ceil'] is not None and not r['ceil']['applied']],
            'record_flags_seen': sorted({r['rec_flags'] for r in rows}),
            'mean_total_bytes': float(tot.mean()), 'ratio_of_raw': float(tot.mean() / RAW),
            'mean_used_bytes_unpadded': float(used.mean()), 'ratio_used_of_raw': float(used.mean() / RAW),
            'mean_parts': {'header': rows[0]['sizes']['header'], 'length_table': float(lt.mean()),
                           'scale_payload': float(sp.mean()), 'weight_payload': float(wp.mean()), 'padding': float(pad.mean())},
            'bit_per_w_weight_payload': float(wp.mean() * 8 / W_PER_REC),
            'bit_per_scale_scale_payload': float(sp.mean() * 8 / S_PER_REC),
            'all_in_bit_per_w_used': float(used.mean() * 8 / W_PER_REC),
            'mean_weight_stream_bytes': float(np.mean([r['wlen']['mean'] for r in rows])),
            'max_weight_stream_bytes': int(max(r['wlen']['max'] for r in rows)),
            'timing_s_median': {'encode': float(np.median(enc_t)) if enc_t else None,
                                'decode': float(np.median(dec_t)) if dec_t else None},
            'model_id': rows[0]['model_id'], 'model_blob_bytes': rows[0]['model_blob_bytes']}


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(jdump(r) + '\n')


def load_ref(name):
    p = os.path.join(REF_DIR, name)
    if name.endswith('.npz'):
        return np.load(p)
    return json.load(open(p, encoding='utf-8'))
