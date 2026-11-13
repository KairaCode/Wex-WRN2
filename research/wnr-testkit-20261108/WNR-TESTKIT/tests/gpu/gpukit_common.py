"""Shared helpers for the portable GPU tests of the WNR2/WNR3 test kit (tests/gpu/run_*.py).

Conventions (kit-wide):
  python -I tests/gpu/run_<test>.py --root <kit root> [--data <data dir>] [--out <results dir>] [--quick]
  defaults: root = two levels above the script, data = <root>/data, out = <root>/results/<test-name>/
  every runner writes <out>/summary.json and <out>/<test-name>.log and prints one final line
  'RESULT <test-name> PASS|FAIL|DONE <short summary>'.

No paths of the machine the kit was built on are used anywhere; everything is relative to --root / --data / --out.
"""
import argparse, datetime, glob, hashlib, json, os, platform, shutil, subprocess, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'ref')
REF_MANIFEST = os.path.join(REF, 'manifest_v41_sample160.json')
# the 8 experts of team 2 gpu-proto / team 4 ceil-gpu / team 5 gpu-v1 (indices into the 160-sample manifest)
PICK8 = [17, 0, 40, 80, 120, 159, 60, 100]
HF_REPO = 'patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
HF_REV = '4ab649b2254afdcf6c2f7c1fbea1c05eda11828c'


# ------------------------------------------------------------------------------------------------ arguments / logging
def base_parser(test_name, doc):
    p = argparse.ArgumentParser(prog=f'run_{test_name}.py', description=doc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=None, help='kit root (default: two levels above this script)')
    p.add_argument('--data', default=None, help='data dir (default: <root>/data); see README for the layout')
    p.add_argument('--out', default=None, help='results dir (default: <root>/results/%s)' % test_name)
    p.add_argument('--quick', action='store_true', help='reduced run (minutes) for a first check')
    p.add_argument('--shards', default=None,
                   help='optional dir with the V4.1 safetensors shards (model-000NN-of-00048.safetensors); missing '
                        'expert .bin files are cut out of them by manifest offset and sha256-checked')
    p.add_argument('--fetch', action='store_true',
                   help='allow HTTP Range download of missing expert tensors from Hugging Face (pinned revision, '
                        'sha256-checked). Off by default.')
    p.add_argument('--device', type=int, default=0, help='CUDA device ordinal (default 0)')
    return p


def resolve_paths(args, test_name):
    root = os.path.abspath(args.root) if args.root else os.path.abspath(os.path.join(HERE, '..', '..'))
    data = os.path.abspath(args.data) if args.data else os.path.join(root, 'data')
    out = os.path.abspath(args.out) if args.out else os.path.join(root, 'results', test_name)
    os.makedirs(out, exist_ok=True)
    args.root, args.data, args.out = root, data, out
    return root, data, out


class Log:
    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8')
        self.t0 = time.time()
        self.f.write('\n===== %s =====\n' % datetime.datetime.now().isoformat(timespec='seconds'))

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        line = '[%7.1f s] %s' % (time.time() - self.t0, s)
        print(line, flush=True)
        self.f.write(line + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


# ------------------------------------------------------------------------------------------------ environment
def _run(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:  # noqa
        return 'n/a (%s)' % type(e).__name__


def find_nvsmi():
    p = shutil.which('nvidia-smi')
    if p:
        return p
    if os.name == 'nt':
        sr = os.environ.get('SystemRoot', 'C:/Windows')
        for c in [os.path.join(sr, 'System32', 'nvidia-smi.exe')] + \
                sorted(glob.glob(os.path.join(sr, 'System32', 'DriverStore', 'FileRepository', 'nv*', 'nvidia-smi.exe'))):
            if os.path.exists(c):
                return c
    return None


def nvsmi_clocks():
    p = find_nvsmi()
    if not p:
        return 'nvidia-smi not found'
    return _run([p, '--query-gpu=clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu',
                 '--format=csv,noheader'])


def cpu_name():
    if os.name == 'nt':
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
        except Exception:  # noqa
            return os.environ.get('PROCESSOR_IDENTIFIER', platform.processor())
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except Exception:  # noqa
        pass
    return platform.processor()


def env_info(device=0):
    import numpy as np
    e = {'os': platform.platform(), 'machine': platform.machine(), 'cpu': cpu_name(), 'cpu_count': os.cpu_count(),
         'python': sys.version.split()[0], 'numpy': np.__version__, 'time': datetime.datetime.now().isoformat(timespec='seconds')}
    try:
        import cupy as cp
        e['cupy'] = cp.__version__
        rt = cp.cuda.runtime
        props = rt.getDeviceProperties(device)
        e['gpu'] = {'name': props['name'].decode(errors='replace'), 'cc': '%d.%d' % (props['major'], props['minor']),
                    'sms': props['multiProcessorCount'], 'clock_khz_max': props.get('clockRate'),
                    'mem_bytes': props['totalGlobalMem'], 'smem_per_sm': props['sharedMemPerMultiprocessor'],
                    'smem_per_block_optin': props.get('sharedMemPerBlockOptin'), 'regs_per_sm': props['regsPerMultiprocessor'],
                    'l2_bytes': props.get('l2CacheSize')}
        e['cuda_driver'] = rt.driverGetVersion()
        e['cuda_runtime'] = rt.runtimeGetVersion()
        try:
            e['nvrtc'] = '%d.%d' % cp.cuda.nvrtc.getVersion()
        except Exception as ex:  # noqa
            e['nvrtc'] = 'n/a (%s)' % ex
    except Exception as ex:  # noqa
        e['cupy_error'] = '%s: %s' % (type(ex).__name__, ex)
    p = find_nvsmi()
    if p:
        e['nvidia_smi'] = _run([p, '--query-gpu=name,driver_version,pci.bus_id,power.limit,clocks.max.sm,clocks.max.mem',
                                '--format=csv,noheader'])
    return e


def device_arch(device=0, mode='sm'):
    """NVRTC -arch option from the device compute capability: sm_XY (cubin) or compute_XY (PTX, driver JIT)."""
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(device)
    cc = '%d%d' % (props['major'], props['minor'])
    return ('-arch=%s_%s' % ('sm' if mode == 'sm' else 'compute', cc)), cc


# ------------------------------------------------------------------------------------------------ result
def finish(test_name, out, status, short, summary, log=None):
    summary = dict(summary)
    summary['test'] = test_name
    summary['status'] = status
    summary['short'] = short
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, default=str)
    line = 'RESULT %s %s %s' % (test_name, status, short)
    if log:
        log(line)
        log.close()
    else:
        print(line, flush=True)
    return 0 if status in ('PASS', 'DONE') else 1


def run_main(test_name, fn, args, out):
    """Run fn(); any exception -> FAIL with the traceback in summary.json."""
    try:
        return fn()
    except SystemExit:
        raise
    except BaseException as ex:  # noqa
        tb = traceback.format_exc()
        print(tb, flush=True)
        try:
            with open(os.path.join(out, test_name + '.log'), 'a', encoding='utf-8') as f:
                f.write(tb)
        except Exception:  # noqa
            pass
        return finish(test_name, out, 'FAIL', 'exception %s: %s' % (type(ex).__name__, str(ex)[:160]),
                      {'exception': tb})


# ------------------------------------------------------------------------------------------------ data
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def ref_manifest():
    with open(REF_MANIFEST) as f:
        return json.load(f)


def _candidates(data):
    c = [data, os.path.join(data, 'v41-sample'), os.path.join(data, 'v41_sample'), os.path.join(data, 'sample'),
         os.path.join(data, 'v41-sample160')]
    return [d for d in c if os.path.isdir(d)]


def _extract_from_shards(shards, e, dst):
    """Cut one expert record out of local safetensors shards (offsets of the manifest are absolute file offsets)."""
    blobs = []
    for t in e['tensors']:
        hits = glob.glob(os.path.join(shards, '**', t['shard']), recursive=True)
        if not hits:
            raise FileNotFoundError('shard %s not under %s' % (t['shard'], shards))
        with open(hits[0], 'rb') as f:
            f.seek(t['offset'])
            b = f.read(t['bytes'])
        if len(b) != t['bytes']:
            raise IOError('short read from %s' % hits[0])
        blobs.append(b)
    with open(dst + '.part', 'wb') as f:
        f.write(b''.join(blobs))
    os.replace(dst + '.part', dst)


class RangeRefused(IOError):
    """The server did not honour the Range request (no 206 / wrong Content-Range): never retried."""


def _read_range(url, a, b, timeout=300):
    """Bytes a..b (inclusive) of url. Requires HTTP 206 with a matching Content-Range and streams the body in 1 MiB
    chunks with a hard cap of b-a+1 bytes, so a server or proxy that ignores Range can never pull a whole shard
    into memory (setup/fetch_data.py applies the same rule)."""
    import urllib.request
    n = b - a + 1
    req = urllib.request.Request(url, headers={'User-Agent': 'wnr-testkit/1.0', 'Range': 'bytes=%d-%d' % (a, b),
                                               'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        st = getattr(r, 'status', None) or r.getcode()
        if st != 206:
            raise RangeRefused('HTTP %s instead of 206 Partial Content for a Range request (server/proxy ignores '
                               'Range); refusing to download the whole shard' % st)
        cr = r.headers.get('Content-Range', '')
        if not cr.replace(' ', '').lower().startswith('bytes%d-%d/' % (a, b)):
            raise RangeRefused('unexpected Content-Range %r for bytes=%d-%d' % (cr, a, b))
        cl = r.headers.get('Content-Length')
        if cl is not None and cl.strip().isdigit() and int(cl) != n:
            raise RangeRefused('Content-Length %s != %d requested bytes' % (cl, n))
        parts, got = [], 0
        while got < n:
            c = r.read(min(1 << 20, n - got))
            if not c:
                break
            parts.append(c)
            got += len(c)
        if got != n:
            raise IOError('short read %d of %d bytes' % (got, n))
        return b''.join(parts)


def _fetch_hf(e, dst, log):
    base = 'https://huggingface.co/%s/resolve/%s/' % (HF_REPO, HF_REV)
    blobs = []
    for t in e['tensors']:
        a, b = t['offset'], t['offset'] + t['bytes'] - 1
        for k in range(5):
            try:
                blobs.append(_read_range(base + t['shard'], a, b))
                break
            except RangeRefused:
                raise
            except Exception as ex:  # noqa
                if k == 4:
                    raise
                log('fetch retry', e['file'], t['name'], ex)
                time.sleep(2 + 3 * k)
    with open(dst + '.part', 'wb') as f:
        f.write(b''.join(blobs))
    os.replace(dst + '.part', dst)


def resolve_experts(args, indices, log, need_dir=False):
    """Locate (and verify) the expert .bin records with the given manifest indices.

    Search order: <data>, <data>/v41-sample, <data>/v41_sample, <data>/sample (files named L??_E???.bin as written by
    fetch_experts.py), then the kit cache <root>/results/_cache/v41-sample, then --shards (cut out of local
    safetensors), then --fetch (HTTP Range from the pinned HF revision). Every file is sha256-checked against the
    kit's copy of the 160-expert manifest. Returns (manifest, {file: path}, sample_dir or None).
    need_dir=True: all files are made available in ONE directory with a manifest.json (the cache dir is used and
    files found elsewhere are copied there) - needed by the team-5 scripts that take a sample directory."""
    man = ref_manifest()
    cache = os.path.join(args.root, 'results', '_cache', 'v41-sample')
    dirs = _candidates(args.data) + ([cache] if os.path.isdir(cache) else [])
    paths, missing, status = {}, [], {}
    for i in indices:
        e = man['experts'][i]
        fn = e['file']
        found = None
        for d in dirs:
            p = os.path.join(d, fn)
            if os.path.exists(p) and os.path.getsize(p) == sum(t['bytes'] for t in e['tensors']):
                if sha256_file(p) == e['sha256']:
                    found = p
                    break
                log('WARNING sha256 mismatch, ignoring', p)
        if found is None and (args.shards or args.fetch):
            os.makedirs(cache, exist_ok=True)
            dst = os.path.join(cache, fn)
            try:
                if args.shards:
                    log('extracting', fn, 'from shards in', args.shards)
                    _extract_from_shards(args.shards, e, dst)
                else:
                    log('fetching', fn, 'from HF', HF_REPO, HF_REV[:12])
                    _fetch_hf(e, dst, log)
                if sha256_file(dst) == e['sha256']:
                    found = dst
                else:
                    log('ERROR sha256 mismatch after extraction/fetch', dst)
            except Exception as ex:  # noqa
                log('ERROR could not obtain', fn, ex)
        if found is None:
            missing.append(fn)
        else:
            paths[fn] = found
    if missing:
        raise FileNotFoundError(
            '%d expert record(s) missing or with wrong sha256: %s ... Put the V4.1 sample .bin files (fetch_experts.py '
            'layout, see tests/gpu/README.md) in %s or %s/v41-sample, or pass --shards <dir with the safetensors '
            'shards>, or --fetch.' % (len(missing), ', '.join(missing[:6]), args.data, args.data))
    sample_dir = None
    if need_dir:
        dset = {os.path.dirname(p) for p in paths.values()}
        d0 = next(iter(dset))
        if len(dset) == 1 and _manifest_compatible(os.path.join(d0, 'manifest.json'), man):
            sample_dir = d0
        else:
            os.makedirs(cache, exist_ok=True)
            for fn, p in paths.items():
                dst = os.path.join(cache, fn)
                if os.path.abspath(p) != os.path.abspath(dst):
                    shutil.copyfile(p, dst)
                paths[fn] = dst
            with open(os.path.join(cache, 'manifest.json'), 'w') as f:
                json.dump(man, f, indent=1)
            sample_dir = cache
    log('experts ok (sha256 verified):', len(paths), 'sample_dir' if need_dir else '', sample_dir or '')
    return man, paths, sample_dir


def _manifest_compatible(path, man):
    """True if the manifest at path lists the same 160 experts in the same order (PICK indices are positional)."""
    try:
        with open(path) as f:
            m = json.load(f)
        return [e['file'] for e in m['experts']] == [e['file'] for e in man['experts']] and \
            all(a.get('sha256') == b.get('sha256') for a, b in zip(m['experts'], man['experts']))
    except Exception:  # noqa
        return False


def load_record(path, e):
    """-> {tensor name: (uint8 array, shape, dtype)} for one expert record (fetch_experts.py order)."""
    import numpy as np
    raw = np.fromfile(path, dtype=np.uint8)
    out, off = {}, 0
    for t in e['tensors']:
        n = t['bytes']
        out[t['name']] = (raw[off:off + n], t['shape'], t['dtype'])
        off += n
    assert off == raw.size, (path, off, raw.size)
    return out
