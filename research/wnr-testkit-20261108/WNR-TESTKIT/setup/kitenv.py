"""Shared helpers for the WNR test kit (standard library only; numpy is optional).

Importable by any runner:
    import os, sys
    sys.path.insert(0, os.path.join(KIT_ROOT, 'setup'))   # needed because python -I drops the script dir
    import kitenv
    env = kitenv.env_info()          # dict: os, cpu, ram, gpu (nvidia-smi), python/numpy/cupy/nvrtc versions
    log = kitenv.Log(path_to_log)    # tee to stdout + file

Also a small CLI used by run_all.ps1 / check_env.py:
    python -I setup/kitenv.py --root <kit root> --list-runners     # JSON list of discovered runners, in run order
    python -I setup/kitenv.py --root <kit root> --env              # JSON env_info()

Runner discovery rule (documented in setup/README.md):
  * if <root>/runners.txt exists: one relative path per line (# comments), in that order;
  * otherwise every run_*.py with a __main__ block under <root> (outside setup, results, data, docs, partial,
    workflows, venv, __pycache__) is a runner, plus any .py that defines a module-level literal
        KIT_META = {'name': 'ceil-heldout', 'order': 20, 'needs': ['data', 'gpu', 'cc'], 'quick_minutes': 5}
    which is read with ast.literal_eval (never executed). Known needs: data (160-expert sample), extra (the 70+6
    extra experts), gpu (usable CUDA GPU + CuPy), nvrtc (CuPy + NVRTC compile only, no GPU), cc (C compiler).
    'optional': True marks an experimental runner that run_all.ps1 runs only with -All or when named in -Only.
    All kit runners carry KIT_META. Without it, name = file stem minus 'run_', needs =
    data (+ gpu if the path has a 'gpu' folder or the source imports cupy, + cc if it calls cl/gcc/clang), order
    from path keywords (heldout/ref 20, ceil 22, capacity 25, zero/safety 30, audit/corrupt 40, cpu/simd 45,
    other 50, gpu 60, bench 65).
"""
import ast
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

SAMPLE_SET = 'v41-sample'
EXTRA_SET = 'extra'
SKIP_DIRS = {'setup', 'results', 'data', 'docs', 'venv', 'venv-gpu', '.venv', 'env', '__pycache__', '.git',
             'site-packages', 'node_modules', 'partial', 'workflows'}


def default_root(script_file, levels=2):
    p = os.path.abspath(script_file)
    for _ in range(levels):
        p = os.path.dirname(p)
    return p


def run_cmd(args, timeout=20):
    """Run a local command, return (rc, stdout+stderr text) or (None, error string). Never raises."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, errors='replace')
        return r.returncode, (r.stdout or '') + (r.stderr or '')
    except FileNotFoundError:
        return None, 'not found'
    except subprocess.TimeoutExpired:
        return None, 'timeout'
    except Exception as e:  # noqa: BLE001
        return None, f'{type(e).__name__}: {e}'


class Log:
    """Tee: print to stdout and append to a log file (utf-8)."""

    def __init__(self, path):
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self.fh = open(path, 'a', encoding='utf-8')
        self.fh.write(f'---- {time.strftime("%Y-%m-%d %H:%M:%S")} {" ".join(sys.argv)}\n')

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        self.fh.write(s + '\n')
        self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------ environment

def pkg_version(name):
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:  # noqa: BLE001
        return None


def cpu_name():
    if sys.platform == 'win32':
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
        except Exception:  # noqa: BLE001
            return platform.processor()
    if os.path.exists('/proc/cpuinfo'):
        try:
            for line in open('/proc/cpuinfo', encoding='utf-8', errors='replace'):
                if line.lower().startswith('model name'):
                    return line.split(':', 1)[1].strip()
        except Exception:  # noqa: BLE001
            pass
    if sys.platform == 'darwin':
        rc, out = run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string'])
        if rc == 0:
            return out.strip()
    return platform.processor() or platform.machine()


def physical_cores():
    if sys.platform.startswith('linux') and os.path.exists('/proc/cpuinfo'):
        try:
            pairs = set()
            phys = core = None
            for line in open('/proc/cpuinfo', encoding='utf-8', errors='replace'):
                if line.startswith('physical id'):
                    phys = line.split(':')[1].strip()
                elif line.startswith('core id'):
                    core = line.split(':')[1].strip()
                elif not line.strip():
                    if core is not None:
                        pairs.add((phys, core))
                    phys = core = None
            return len(pairs) or None
        except Exception:  # noqa: BLE001
            return None
    if sys.platform == 'win32':
        ps = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'WindowsPowerShell', 'v1.0',
                          'powershell.exe')
        if not os.path.exists(ps):
            ps = shutil.which('powershell') or shutil.which('pwsh') or 'powershell'
        rc, out = run_cmd([ps, '-NoProfile', '-NonInteractive', '-Command',
                           '(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum'], timeout=30)
        if rc == 0:
            try:
                return int(out.strip().splitlines()[-1])
            except Exception:  # noqa: BLE001
                return None
    return None


def ram_bytes():
    if sys.platform == 'win32':
        try:
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            m = MEMSTAT()
            m.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return {'total': int(m.ullTotalPhys), 'available': int(m.ullAvailPhys)}
        except Exception:  # noqa: BLE001
            return None
    if os.path.exists('/proc/meminfo'):
        try:
            d = {}
            for line in open('/proc/meminfo', encoding='utf-8'):
                k, v = line.split(':', 1)
                d[k] = int(v.split()[0]) * 1024
            return {'total': d.get('MemTotal'), 'available': d.get('MemAvailable')}
        except Exception:  # noqa: BLE001
            return None
    return None


def cpu_flags_linux():
    try:
        for line in open('/proc/cpuinfo', encoding='utf-8', errors='replace'):
            if line.startswith('flags'):
                return set(line.split(':', 1)[1].split())
    except Exception:  # noqa: BLE001
        pass
    return set()


def nvidia_smi():
    """GPU list from nvidia-smi (no CUDA context, no kernels). Returns dict."""
    exe = shutil.which('nvidia-smi')
    if exe is None and sys.platform == 'win32':
        cand = os.path.join(os.environ.get('SystemRoot', 'C:/Windows'), 'System32', 'nvidia-smi.exe')
        if os.path.exists(cand):
            exe = cand
    if exe is None:
        return {'available': False, 'error': 'nvidia-smi not found', 'gpus': []}
    fields = ['index', 'name', 'compute_cap', 'memory.total', 'driver_version', 'pci.bus_id', 'clocks.max.sm',
              'power.limit']
    rc, out = run_cmd([exe, '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'])
    if rc != 0:  # very old drivers do not know compute_cap
        fields = [f for f in fields if f != 'compute_cap']
        rc, out = run_cmd([exe, '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'])
    gpus = []
    if rc == 0:
        for line in out.strip().splitlines():
            vals = [v.strip() for v in line.split(',')]
            if len(vals) == len(fields):
                gpus.append(dict(zip(fields, vals)))
    rc2, out2 = run_cmd([exe])
    m = re.search(r'CUDA (?:UMD |Driver )?Version:\s*([0-9.]+)', out2 or '')
    return {'available': rc == 0, 'driver_cuda_version': m.group(1) if m else None, 'gpus': gpus,
            'error': None if rc == 0 else out.strip()[:300]}


def env_info(with_gpu=True):
    """Light environment record for summary.json (does not import CuPy, does not touch the GPU)."""
    try:
        import numpy
        npv = numpy.__version__
    except Exception:  # noqa: BLE001
        npv = None
    info = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'os': platform.platform(),
        'os_system': platform.system(),
        'machine': platform.machine(),
        'hostname_hash': None,
        'cpu': cpu_name(),
        'logical_cpus': os.cpu_count(),
        'ram': ram_bytes(),
        'python': sys.version.split()[0],
        'python_impl': platform.python_implementation(),
        'numpy': npv,
        'cupy': pkg_version('cupy-cuda12x') or pkg_version('cupy-cuda13x') or pkg_version('cupy'),
        'nvrtc_wheel': pkg_version('nvidia-cuda-nvrtc-cu12') or pkg_version('nvidia-cuda-nvrtc'),
        'cuda_runtime_wheel': pkg_version('nvidia-cuda-runtime-cu12') or pkg_version('nvidia-cuda-runtime'),
    }
    try:
        import hashlib
        info['hostname_hash'] = hashlib.sha256(platform.node().encode()).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        pass
    if with_gpu:
        info['nvidia_smi'] = nvidia_smi()
    return info


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


# ------------------------------------------------------------------ runner discovery

def _kit_meta(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'KIT_META' for t in node.targets):
            try:
                v = ast.literal_eval(node.value)
                return v if isinstance(v, dict) else None
            except Exception:  # noqa: BLE001
                return None
    return None


def _infer(path, src, root):
    rel = os.path.relpath(path, root).replace('\\', '/')
    parts = rel.lower().split('/')
    needs = []
    if 'import cupy' in src or 'from cupy' in src or 'gpu' in parts or 'gpukit_common' in src:
        needs.append('gpu')
    needs.append('data')  # every kit test reads the expert sample unless KIT_META says otherwise
    if re.search(r"['\"](cl|cl\.exe|gcc|clang|clang-cl)['\"]", src):
        needs.append('cc')
    stem = os.path.splitext(os.path.basename(path))[0]
    name = stem[4:] if stem.startswith('run_') else stem
    order = 50
    for kw, o in (('ref', 20), ('heldout', 20), ('ceil', 22), ('capacity', 25), ('zero', 30), ('safety', 30),
                  ('audit', 40), ('corrupt', 40), ('simd', 45), ('cpu', 45), ('gpu', 60), ('bench', 65)):
        if kw in rel.lower():
            order = o
            break
    if 'gpu' in needs and order < 60:
        order = 60
    return {'name': name, 'order': order, 'needs': needs, 'meta_source': 'inferred'}


def discover_runners(root):
    root = os.path.abspath(root)
    out = []
    lst = os.path.join(root, 'runners.txt')
    paths = []
    explicit = os.path.exists(lst)
    if explicit:
        for line in open(lst, encoding='utf-8-sig'):
            line = line.strip()
            if line and not line.startswith('#'):
                paths.append(os.path.normpath(os.path.join(root, line)))
    else:
        for dp, dns, fns in os.walk(root):
            rel_parts = os.path.relpath(dp, root).replace('\\', '/').split('/')
            dns[:] = [d for d in dns if d.lower() not in SKIP_DIRS and not d.startswith('.')
                      and not os.path.exists(os.path.join(dp, d, 'pyvenv.cfg'))]   # any virtualenv, whatever its name
            if rel_parts[0].lower() in SKIP_DIRS:
                continue
            for fn in fns:
                if fn.endswith('.py'):
                    paths.append(os.path.join(dp, fn))
    for i, p in enumerate(sorted(paths) if not explicit else paths):
        try:
            src = open(p, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        meta = _kit_meta(src) if 'KIT_META' in src else None
        if not explicit and not (meta or (os.path.basename(p).startswith('run_') and '__main__' in src)):
            continue
        r = _infer(p, src, root)
        if meta:
            r.update({k: v for k, v in meta.items() if k in ('name', 'order', 'needs', 'quick_minutes',
                                                              'full_minutes', 'description', 'optional')})
            r['meta_source'] = 'KIT_META'
        if explicit:
            r['order'] = i
        r['path'] = os.path.relpath(p, root).replace('\\', '/')
        out.append(r)
    out.sort(key=lambda r: (r['order'], r['path']))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=default_root(__file__))
    ap.add_argument('--list-runners', action='store_true')
    ap.add_argument('--env', action='store_true')
    a = ap.parse_args()
    if a.list_runners:
        print(json.dumps(discover_runners(a.root)))
    if a.env:
        print(json.dumps(env_info(), indent=1, default=str))


if __name__ == '__main__':
    main()
