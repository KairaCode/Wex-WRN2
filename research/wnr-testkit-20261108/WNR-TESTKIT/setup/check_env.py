"""Environment check for the WNR test kit. Read-only: launches no GPU kernels, downloads nothing.

Usage (PowerShell, from the kit root):
    python -I setup\\check_env.py [--root <kit root>] [--data <data dir>] [--out <results dir>] [--quick]

Reports OS, CPU (name, cores, AVX2 / AVX-512 support; on Windows from IsProcessorFeaturePresent, or with --cpuid
from the cpuid instruction run in a child process), RAM, GPUs (nvidia-smi: name, compute capability, driver,
memory), Python / numpy / CuPy / CUDA runtime / NVRTC versions, an NVRTC compile-only probe (PTX for the local GPU
arch and for sm_120, nothing is loaded or launched), C compilers (cl, gcc, clang, nvcc; MSVC via vswhere), free
disk at --data, which data sets are present, and which kit runners can run here.
Writes <out>/summary.json and <out>/check_env.log; last line: RESULT check_env PASS|FAIL <summary>.
--quick only skips the physical core count query (the CuPy / NVRTC probe takes seconds and always runs).
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitenv  # noqa: E402

TEST = 'check_env'
SAMPLE_BYTES = 160 * 18800640       # 3,008,102,400
EXTRA_BYTES = 76 * 18800640         # 1,428,848,640
TINY_KERNEL = r'''
extern "C" __global__ void kit_probe(const unsigned int* __restrict__ a, unsigned int* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __funnelshift_l(a[i], a[i] ^ 0x9e3779b9u, i & 31) + __popc(a[i]);
}
'''

# --------------------------------------------------------------- probes run in a child process

# x86-64 machine code. Win64 ABI: void cpuid(uint32 leaf=ecx, uint32 sub=edx, uint32 out[4]=r8)
CPUID_WIN64 = bytes([0x53, 0x89, 0xC8, 0x89, 0xD1, 0x0F, 0xA2, 0x41, 0x89, 0x00, 0x41, 0x89, 0x58, 0x04,
                     0x41, 0x89, 0x48, 0x08, 0x41, 0x89, 0x50, 0x0C, 0x5B, 0xC3])
# uint64 xgetbv(uint32 idx=ecx)
XGETBV_WIN64 = bytes([0x0F, 0x01, 0xD0, 0x48, 0xC1, 0xE2, 0x20, 0x48, 0x09, 0xD0, 0xC3])


def probe_cpuid_win():
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
    k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
    k32.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
    # W^X: allocate read/write (PAGE_READWRITE 0x04), copy the code, then switch to PAGE_EXECUTE_READ (0x20);
    # the page is never writable and executable at the same time
    mem = k32.VirtualAlloc(None, 4096, 0x3000, 0x04)
    if not mem:
        raise OSError('VirtualAlloc failed')
    ctypes.memmove(mem, CPUID_WIN64, len(CPUID_WIN64))
    ctypes.memmove(mem + 64, XGETBV_WIN64, len(XGETBV_WIN64))
    old = ctypes.c_ulong(0)
    if not k32.VirtualProtect(mem, 4096, 0x20, ctypes.byref(old)):
        k32.VirtualFree(mem, 0, 0x8000)
        raise OSError('VirtualProtect(PAGE_EXECUTE_READ) failed')
    arr = ctypes.c_uint32 * 4
    f_cpuid = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(arr))(mem)
    f_xgetbv = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint32)(mem + 64)

    def cpuid(leaf, sub=0):
        o = arr()
        f_cpuid(leaf, sub, ctypes.byref(o))
        return list(o)  # eax ebx ecx edx

    r0 = cpuid(0)
    vendor = b''.join(x.to_bytes(4, 'little') for x in (r0[1], r0[3], r0[2])).decode('ascii', 'replace')
    maxleaf = r0[0]
    _, _, c1, d1 = cpuid(1)
    osxsave = bool(c1 >> 27 & 1)
    xcr0 = f_xgetbv(0) if osxsave else 0
    os_ymm = (xcr0 & 0x6) == 0x6
    os_zmm = (xcr0 & 0xE6) == 0xE6
    b7 = c7 = d7 = 0
    if maxleaf >= 7:
        _, b7, c7, d7 = cpuid(7, 0)
    hw = {
        'sse4_2': bool(c1 >> 20 & 1), 'popcnt': bool(c1 >> 23 & 1), 'avx': bool(c1 >> 28 & 1),
        'fma': bool(c1 >> 12 & 1), 'bmi1': bool(b7 >> 3 & 1), 'avx2': bool(b7 >> 5 & 1), 'bmi2': bool(b7 >> 8 & 1),
        'avx512f': bool(b7 >> 16 & 1), 'avx512dq': bool(b7 >> 17 & 1), 'avx512cd': bool(b7 >> 28 & 1),
        'avx512bw': bool(b7 >> 30 & 1), 'avx512vl': bool(b7 >> 31 & 1), 'avx512vbmi': bool(c7 >> 1 & 1),
        'avx512vbmi2': bool(c7 >> 6 & 1), 'gfni': bool(c7 >> 8 & 1), 'avx512vpopcntdq': bool(c7 >> 14 & 1),
        'avx512bitalg': bool(c7 >> 12 & 1),
    }
    usable = dict(hw)
    for k in hw:
        if k in ('avx', 'fma', 'avx2') and not os_ymm:
            usable[k] = False
        if k.startswith('avx512') and not os_zmm:
            usable[k] = False
    k32.VirtualFree(mem, 0, 0x8000)
    return {'source': 'cpuid', 'vendor': vendor, 'xcr0': hex(xcr0), 'os_saves_ymm': os_ymm, 'os_saves_zmm': os_zmm,
            'flags': usable}


def probe_cupy(archs):
    """Import CuPy, query devices (creates a CUDA context, launches nothing) and NVRTC-compile a tiny kernel to PTX
    for each arch (compile only: nothing is loaded into the GPU)."""
    out = {}
    t0 = time.time()
    import cupy
    out['cupy_version'] = cupy.__version__
    out['import_s'] = round(time.time() - t0, 2)
    rt = cupy.cuda.runtime
    try:
        out['cuda_runtime_version'] = rt.runtimeGetVersion()
    except Exception as e:  # noqa: BLE001
        out['cuda_runtime_version'] = f'error: {e}'
    try:
        out['driver_cuda_version'] = rt.driverGetVersion()
    except Exception as e:  # noqa: BLE001
        out['driver_cuda_version'] = f'error: {e}'
    devs = []
    try:
        n = rt.getDeviceCount()
        for i in range(n):
            p = rt.getDeviceProperties(i)
            d = {'index': i}
            for k in ('name', 'major', 'minor', 'totalGlobalMem', 'multiProcessorCount', 'clockRate',
                      'memoryClockRate', 'memoryBusWidth', 'l2CacheSize', 'sharedMemPerBlockOptin',
                      'regsPerMultiprocessor'):
                if k in p:
                    v = p[k]
                    d[k] = v.decode(errors='replace') if isinstance(v, bytes) else v
            devs.append(d)
    except Exception as e:  # noqa: BLE001
        out['device_error'] = f'{type(e).__name__}: {e}'
    out['devices'] = devs
    archs = sorted(set(archs) | {f"{d['major']}{d['minor']}" for d in devs if 'major' in d and 'minor' in d})
    from cupy_backends.cuda.libs import nvrtc
    try:
        out['nvrtc_version'] = list(nvrtc.getVersion())
    except Exception as e:  # noqa: BLE001
        out['nvrtc_version'] = f'error: {e}'
    try:
        out['nvrtc_supported_archs'] = list(nvrtc.getSupportedArchs())
    except Exception as e:  # noqa: BLE001
        out['nvrtc_supported_archs'] = f'n/a: {type(e).__name__}'
    comp = {}
    for arch in archs:
        t1 = time.time()
        try:
            prog = nvrtc.createProgram(TINY_KERNEL, 'kit_probe.cu', [], [])
            try:
                nvrtc.compileProgram(prog, [f'--gpu-architecture=compute_{arch}', '-std=c++17'])
                ptx = nvrtc.getPTX(prog)
                comp[f'sm_{arch}'] = {'ok': True, 'ptx_bytes': len(ptx), 's': round(time.time() - t1, 3)}
            except Exception as e:  # noqa: BLE001
                try:
                    lg = nvrtc.getProgramLog(prog)
                except Exception:  # noqa: BLE001
                    lg = ''
                comp[f'sm_{arch}'] = {'ok': False, 'error': f'{type(e).__name__}: {e}', 'log': str(lg)[:400]}
            finally:
                nvrtc.destroyProgram(prog)
        except Exception as e:  # noqa: BLE001
            comp[f'sm_{arch}'] = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
    out['nvrtc_compile'] = comp
    return out


def child_main(kind, arg):
    try:
        if kind == 'cpuid':
            r = probe_cpuid_win()
        else:
            r = probe_cupy([a for a in arg.split(',') if a])
        print('PROBE_JSON ' + json.dumps(r, default=str))
    except Exception as e:  # noqa: BLE001
        print('PROBE_JSON ' + json.dumps({'error': f'{type(e).__name__}: {e}'}))


def run_probe(kind, arg='', timeout=180):
    args = [sys.executable, '-I', os.path.abspath(__file__), '--_probe', kind, '--_probe-arg', arg]
    rc, out = kitenv.run_cmd(args, timeout=timeout)
    for line in (out or '').splitlines():
        if line.startswith('PROBE_JSON '):
            return json.loads(line[len('PROBE_JSON '):])
    return {'error': f'probe {kind} failed (rc={rc}): {(out or "")[-400:]}'}


# --------------------------------------------------------------- other checks

def cpu_features_win_api():
    """CPU features from the Windows API only (IsProcessorFeaturePresent; no machine code is run).
    PF_SSE4_1=37, PF_SSE4_2=38, PF_AVX=39, PF_AVX2=40, PF_AVX512F=41. These report OS-usable support; Windows builds
    older than about 10 2004 do not know 39-41 and return 0 (false negative: use --cpuid there)."""
    import ctypes
    f = ctypes.windll.kernel32.IsProcessorFeaturePresent
    fl = {'sse4_1': bool(f(37)), 'sse4_2': bool(f(38)), 'avx': bool(f(39)), 'avx2': bool(f(40)),
          'avx512f': bool(f(41))}
    return {'source': 'IsProcessorFeaturePresent (default; --cpuid for the full cpuid probe)', 'flags': fl,
            'note': 'avx512 sub-features (bw, vl, vbmi2, ...) are only known with --cpuid'}


def cpu_features(use_cpuid=False):
    if sys.platform.startswith('linux'):
        fl = kitenv.cpu_flags_linux()
        keys = ['sse4_2', 'popcnt', 'avx', 'fma', 'bmi1', 'avx2', 'bmi2', 'avx512f', 'avx512dq', 'avx512cd',
                'avx512bw', 'avx512vl', 'avx512vbmi', 'avx512_vbmi2', 'gfni', 'avx512_vpopcntdq', 'avx512_bitalg']
        return {'source': '/proc/cpuinfo', 'flags': {k.replace('_', '') if k.startswith('avx512_') else k: k in fl
                                                     for k in keys}}
    if sys.platform == 'win32' and not use_cpuid:
        try:
            return cpu_features_win_api()
        except Exception as e:  # noqa: BLE001
            return {'source': 'IsProcessorFeaturePresent', 'flags': {}, 'error': f'{type(e).__name__}: {e}'}
    if sys.platform == 'win32' and platform.machine().lower() in ('amd64', 'x86_64'):
        r = run_probe('cpuid', timeout=30)
        try:  # cross-check with the OS API (PF_AVX2=40, PF_AVX512F=41; older Windows builds return 0)
            import ctypes
            r['IsProcessorFeaturePresent'] = {
                'avx2': bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40)),
                'avx512f': bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(41))}
        except Exception:  # noqa: BLE001
            pass
        return r
    return {'source': 'unknown platform', 'flags': {}}


def find_compilers():
    res = {}
    for name, vargs in (('cl', []), ('clang-cl', ['--version']), ('gcc', ['--version']), ('clang', ['--version']),
                        ('cc', ['--version']), ('nvcc', ['--version'])):
        p = shutil.which(name)
        if not p:
            continue
        rc, out = kitenv.run_cmd([p] + vargs, timeout=30)
        lines = [ln.strip() for ln in (out or '').splitlines() if ln.strip()]
        ver = next((ln for ln in lines if 'version' in ln.lower() or 'compiler' in ln.lower()), lines[0] if lines else '')
        res[name] = {'path': p, 'version': ver[:200]}
    if sys.platform == 'win32':
        vsw = os.path.join(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
                           'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
        if os.path.exists(vsw):
            rc, out = kitenv.run_cmd([vsw, '-latest', '-products', '*', '-requires',
                                      'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property',
                                      'installationPath'], timeout=30)
            inst = (out or '').strip().splitlines()
            if rc == 0 and inst:
                ip = inst[0].strip()
                msvc = os.path.join(ip, 'VC', 'Tools', 'MSVC')
                cls = []
                if os.path.isdir(msvc):
                    for v in sorted(os.listdir(msvc)):
                        c = os.path.join(msvc, v, 'bin', 'Hostx64', 'x64', 'cl.exe')
                        if os.path.exists(c):
                            cls.append(c)
                res['msvc_installed'] = {'installation': ip, 'cl_exe': cls[-1] if cls else None,
                                         'vcvars64': os.path.join(ip, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat'),
                                         'in_path': 'cl' in res}
    return res


def data_status(data):
    st = {}
    for sub, nexp, nbytes in ((kitenv.SAMPLE_SET, 160, SAMPLE_BYTES), (kitenv.EXTRA_SET, 76, EXTRA_BYTES)):
        d = os.path.join(data, sub)
        n = sz = 0
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith('.bin'):
                    s = os.path.getsize(os.path.join(d, fn))
                    if s == 18800640:
                        n += 1
                        sz += s
        st[sub] = {'dir_exists': os.path.isdir(d), 'experts_present': n, 'experts_expected': nexp,
                   'bytes_present': sz, 'bytes_expected': nbytes, 'complete_by_size': n == nexp,
                   'manifest_present': os.path.exists(os.path.join(d, 'manifest.json'))}
    p = os.path.abspath(data)
    while not os.path.exists(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    try:
        du = shutil.disk_usage(p)
        st['disk'] = {'checked_path': p, 'free': du.free, 'total': du.total}
    except Exception as e:  # noqa: BLE001
        st['disk'] = {'error': str(e)}
    return st


def gb(x):
    return f'{x / 1e9:.1f} GB' if isinstance(x, (int, float)) else str(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=kitenv.default_root(__file__))
    ap.add_argument('--data', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--target-arch', default='120', help='extra NVRTC compile target (default 120 = RTX 50xx)')
    ap.add_argument('--cpuid', action='store_true',
                    help='Windows x64: read CPU features with the cpuid instruction (a few bytes of machine code run '
                         'in a child process from a W^X page). Default: IsProcessorFeaturePresent only, which needs '
                         'no machine code (security software may flag runtime code generation).')
    ap.add_argument('--_probe', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--_probe-arg', default='', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._probe:
        child_main(a._probe, a._probe_arg)
        return 0
    root = os.path.abspath(a.root)
    data = os.path.abspath(a.data or os.path.join(root, 'data'))
    out = os.path.abspath(a.out or os.path.join(root, 'results', TEST))
    os.makedirs(out, exist_ok=True)
    log = kitenv.Log(os.path.join(out, TEST + '.log'))
    t0 = time.time()
    problems, warnings = [], []

    env = kitenv.env_info(with_gpu=True)
    if not a.quick:
        env['physical_cores'] = kitenv.physical_cores()
    env['cpu_features'] = cpu_features(a.cpuid)
    smi = env['nvidia_smi']
    archs = []
    for g in smi.get('gpus', []):
        cc = g.get('compute_cap', '')
        if cc and '.' in cc:
            archs.append(cc.replace('.', ''))
    archs = sorted(set(archs + [a.target_arch]))
    if env['cupy']:
        log('probing CuPy / NVRTC in a child process (compile only, no kernel launch) ...')
        cupy_info = run_probe('cupy', ','.join(archs), timeout=300)
    else:
        cupy_info = {'error': 'CuPy not installed (pip install -r setup/requirements-gpu.txt)'}
    env['cupy_probe'] = cupy_info
    comps = find_compilers()
    dst = data_status(data)
    runners = kitenv.discover_runners(root)

    # ---------------- verdicts
    pyv = sys.version_info
    if pyv < (3, 10):
        problems.append(f'Python {pyv.major}.{pyv.minor} < 3.10')
    if not env['numpy']:
        problems.append('numpy not importable (pip install -r setup/requirements-cpu.txt)')
    gpu_ok = False
    if cupy_info and not cupy_info.get('error') and cupy_info.get('devices'):
        gpu_ok = True
        for d in cupy_info['devices']:
            arch = f"{d.get('major')}{d.get('minor')}"
            c = cupy_info.get('nvrtc_compile', {}).get(f'sm_{arch}', {})
            if not c.get('ok'):
                gpu_ok = False
                warnings.append(f"NVRTC cannot compile for sm_{arch} ({d.get('name')}): needs CUDA/NVRTC >= 12.8 "
                                f"for sm_120; {c.get('error', '')}")
            if d.get('major', 0) >= 12:
                rv = cupy_info.get('cuda_runtime_version')
                if isinstance(rv, int) and rv < 12080:
                    warnings.append(f'CUDA runtime {rv} < 12080 cannot target sm_120 (RTX 50xx)')
                dv = cupy_info.get('driver_cuda_version')
                if isinstance(dv, int) and dv < 12080:
                    warnings.append(f'driver supports CUDA {dv} < 12080: update NVIDIA driver to R570 or newer')
    elif smi.get('gpus'):
        warnings.append(f"NVIDIA GPU present but CuPy probe failed: {(cupy_info or {}).get('error', '')[:300]}")
    cc_ok = any(k in comps for k in ('cl', 'gcc', 'clang', 'clang-cl', 'cc'))
    if not cc_ok and 'msvc_installed' in comps:
        warnings.append('MSVC is installed but cl is not in PATH: run from an "x64 Native Tools" prompt or call '
                        'vcvars64.bat first if a test needs a C compiler')
    data_ok = dst[kitenv.SAMPLE_SET]['complete_by_size']
    need_bytes = (0 if data_ok else SAMPLE_BYTES - dst[kitenv.SAMPLE_SET]['bytes_present'])
    if not dst[kitenv.EXTRA_SET]['complete_by_size']:
        need_bytes_extra = EXTRA_BYTES - dst[kitenv.EXTRA_SET]['bytes_present']
    else:
        need_bytes_extra = 0
    free = dst.get('disk', {}).get('free')
    if isinstance(free, int) and free < need_bytes + need_bytes_extra + 2_000_000_000:
        warnings.append(f'free disk {gb(free)} at {dst["disk"]["checked_path"]} may be too small '
                        f'(missing data {gb(need_bytes + need_bytes_extra)} + ~2 GB results)')
    if not data_ok:
        warnings.append(f'data set {kitenv.SAMPLE_SET} incomplete at {data}: run '
                        f'python -I setup\\fetch_data.py (add --extra for the 70+6 extra experts)')
    nvrtc_ok = bool(cupy_info and not cupy_info.get('error') and
                    any(v.get('ok') for v in (cupy_info.get('nvrtc_compile') or {}).values()))
    avail = {'gpu': gpu_ok, 'nvrtc': nvrtc_ok, 'cc': cc_ok, 'data': data_ok,
             'extra': dst[kitenv.EXTRA_SET]['complete_by_size'],
             'avx2': bool(env['cpu_features'].get('flags', {}).get('avx2')),
             'avx512': bool(env['cpu_features'].get('flags', {}).get('avx512f'))}
    can = []
    for r in runners:
        miss = [n for n in r.get('needs', []) if not avail.get(n, True)]
        can.append({'path': r['path'], 'name': r['name'], 'needs': r.get('needs', []), 'missing': miss,
                    'can_run': not miss, 'meta_source': r.get('meta_source')})

    # ---------------- report
    log(f"OS        : {env['os']} ({env['machine']})")
    log(f"CPU       : {env['cpu']}  logical={env['logical_cpus']} physical={env.get('physical_cores')}")
    fl = env['cpu_features'].get('flags', {})
    log(f"CPU ISA   : avx2={fl.get('avx2')} bmi2={fl.get('bmi2')} avx512f={fl.get('avx512f')} "
        f"avx512bw={fl.get('avx512bw')} avx512vbmi2={fl.get('avx512vbmi2')} "
        f"(source {env['cpu_features'].get('source')}{', ' + env['cpu_features']['error'] if 'error' in env['cpu_features'] else ''})")
    ram = env['ram'] or {}
    log(f"RAM       : total {gb(ram.get('total'))}, available {gb(ram.get('available'))}")
    log(f"Python    : {env['python']}  numpy {env['numpy']}  cupy {env['cupy']}  nvrtc-wheel {env['nvrtc_wheel']}")
    if smi.get('gpus'):
        for g in smi['gpus']:
            log(f"GPU {g.get('index')}     : {g.get('name')} cc={g.get('compute_cap')} mem={g.get('memory.total')} MiB "
                f"driver={g.get('driver_version')} (driver CUDA {smi.get('driver_cuda_version')})")
    else:
        log(f"GPU       : none via nvidia-smi ({smi.get('error')})")
    if cupy_info:
        if cupy_info.get('error'):
            log(f"CuPy      : {cupy_info['error'][:300]}")
        else:
            log(f"CuPy      : {cupy_info.get('cupy_version')} runtime={cupy_info.get('cuda_runtime_version')} "
                f"driver={cupy_info.get('driver_cuda_version')} nvrtc={cupy_info.get('nvrtc_version')}")
            for k, v in (cupy_info.get('nvrtc_compile') or {}).items():
                log(f"NVRTC {k:7s}: {'ok' if v.get('ok') else 'FAILED ' + v.get('error', '')[:200]}")
    comp_txt = ', '.join('%s=%s' % (k, v.get('path', v.get('cl_exe'))) for k, v in comps.items())
    log(f"Compilers : {comp_txt or 'none found'}")
    for sub in (kitenv.SAMPLE_SET, kitenv.EXTRA_SET):
        s = dst[sub]
        log(f"Data {sub:10s}: {s['experts_present']}/{s['experts_expected']} experts ({gb(s['bytes_present'])})")
    if 'free' in dst.get('disk', {}):
        log(f"Disk      : {gb(dst['disk']['free'])} free at {dst['disk']['checked_path']}")
    log(f"Runners   : {len(runners)} discovered")
    for c in can:
        log(f"  {'ok  ' if c['can_run'] else 'SKIP'} {c['path']}  needs={c['needs']} missing={c['missing']}")
    for w in warnings:
        log('WARNING: ' + w)
    for p in problems:
        log('PROBLEM: ' + p)

    status = 'FAIL' if problems else 'PASS'
    summary = {'test': TEST, 'status': status, 'quick': a.quick, 'seconds': round(time.time() - t0, 2),
               'root': root, 'data': data, 'environment': env, 'compilers': comps, 'data_status': dst,
               'capabilities': avail, 'runners': can, 'warnings': warnings, 'problems': problems}
    kitenv.write_json(os.path.join(out, 'summary.json'), summary)
    short = (f"py{env['python']} np{env['numpy']} gpu={'yes' if gpu_ok else 'no'} cc={'yes' if cc_ok else 'no'} "
             f"data={'yes' if data_ok else 'no'} runnable={sum(c['can_run'] for c in can)}/{len(can)} "
             f"warnings={len(warnings)}")
    log(f'RESULT {TEST} {status} {short}')
    log.close()
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
