# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Hardware benchmark for WEX testers (Linux and Windows; Python 3 stdlib only).

Collects what decides how a streamed MoE model runs on a machine and writes ONE results.json:
  sysinfo    CPU, cores, RAM, GPUs, kernel, the filesystem of the test directory (no host or user names)
  ram_bw     RAM read / write / copy bandwidth, 1 thread and all threads (bench/ram_bw.c, needs gcc)
  h2d        host <-> device copies, pinned and pageable (bench/h2d.cu, needs nvcc; skipped otherwise)
  disk       4 KiB-aligned reads of a large file: sequential 8 MiB blocks and random 1 / 18 MiB records
             (expert-sized), past the OS cache when the filesystem accepts it (O_DIRECT on Linux,
             FILE_FLAG_NO_BUFFERING on Windows), first pass vs second pass
  wex_check  the repository's CPU checks (scripts/check_all.sh): pass/fail and duration

  python3 bench/bench.py --dir /path/on/the/model/volume [--file BIG_EXISTING_FILE] [--size-gb 16] [--quick]

--dir is where a test file is created (and deleted at the end). --file reads an existing large file
instead (for example a model file not read recently: the closest to a cold read on ZFS, whose ARC
cache cannot be dropped without root). Nothing is sent anywhere: attach results.json to a GitHub issue.
A message kept in the file (an error, the tail of a failed check) has the paths of the run, the user
name and the host name replaced by placeholders.
"""
import argparse
import getpass
import json
import mmap
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'out')


def sh(cmd, timeout=60, env=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, '', str(e)


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ''


def sysinfo(test_dir):
    info = {'os': platform.platform(), 'kernel': platform.release(), 'machine': platform.machine(),
            'python': platform.python_version(), 'logical_cpus': os.cpu_count()}
    for line in read('/proc/cpuinfo').splitlines():
        if line.startswith('model name') or line.startswith('Model'):
            info['cpu'] = line.split(':', 1)[1].strip()
            break
    flags = next((ln for ln in read('/proc/cpuinfo').splitlines() if ln.startswith(('flags', 'Features'))), '')
    info['cpu_avx2'] = ' avx2' in flags
    info['cpu_avx512f'] = ' avx512f' in flags
    for line in read('/proc/meminfo').splitlines():
        k, _, v = line.partition(':')
        if k in ('MemTotal', 'MemAvailable', 'HugePages_Total'):
            info[k] = v.strip()
    rc, out, _ = sh(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max',
                     '--format=csv,noheader'])
    info['nvidia'] = out.splitlines() if rc == 0 else []
    rc, out, _ = sh(['lspci'])
    info['display_devices'] = [ln.split(': ', 1)[-1] for ln in out.splitlines()
                               if rc == 0 and any(w in ln for w in ('VGA', 'Display', '3D controller'))]
    rc, out, _ = sh(['df', '-T', test_dir])
    if rc == 0 and len(out.splitlines()) > 1:
        f = out.splitlines()[1].split()
        info['test_fs'] = {'type': f[1], 'size_kb': f[2], 'avail_kb': f[4]}
    if shutil.which('zfs'):
        rc, out, _ = sh(['zfs', 'get', '-H', '-o', 'property,value', 'recordsize,compression,primarycache,direct',
                         test_dir])
        info['zfs'] = dict(ln.split('\t', 1) for ln in out.splitlines() if '\t' in ln) if rc == 0 else {}
        info['zfs_arc_max'] = read('/sys/module/zfs/parameters/zfs_arc_max').strip()
    if os.name == 'nt':
        win_sysinfo(info, test_dir)
    return info


def win_sysinfo(info, test_dir):
    """On Windows, what /proc and df give on Linux: CPU name (registry), AVX2 and AVX-512F
    (IsProcessorFeaturePresent 40 and 41), RAM (GlobalMemoryStatusEx), the filesystem of test_dir. lspci
    does not exist there, so display_devices stays empty (nvidia-smi still lists NVIDIA GPUs)."""
    import ctypes
    import winreg
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    info['kernel'] = platform.version()
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
            info['cpu'] = str(winreg.QueryValueEx(k, 'ProcessorNameString')[0]).strip()
    except OSError:
        pass
    info['cpu_avx2'] = bool(k32.IsProcessorFeaturePresent(40))
    info['cpu_avx512f'] = bool(k32.IsProcessorFeaturePresent(41))

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong)] + \
                   [(n, ctypes.c_ulonglong) for n in ('ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile', 'ullAvailPageFile',
                                                      'ullTotalVirtual', 'ullAvailVirtual', 'ullAvailExtendedVirtual')]
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    if k32.GlobalMemoryStatusEx(ctypes.byref(m)):
        info['MemTotal'], info['MemAvailable'] = f'{m.ullTotalPhys >> 10} kB', f'{m.ullAvailPhys >> 10} kB'
    root, fs = ctypes.create_unicode_buffer(1024), ctypes.create_unicode_buffer(64)
    if k32.GetVolumePathNameW(os.path.abspath(test_dir), root, 1024) and \
            k32.GetVolumeInformationW(root.value, None, 0, None, None, None, fs, 64):
        du = shutil.disk_usage(test_dir)
        info['test_fs'] = {'type': fs.value.lower(), 'size_kb': str(du.total >> 10), 'avail_kb': str(du.free >> 10)}


class PosixFile:
    """Positioned reads with os.preadv, through O_DIRECT when direct is true."""

    def __init__(self, path, direct):
        self.fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if direct else 0))

    def read(self, buf, off):
        return os.preadv(self.fd, [buf], off)

    def close(self):
        os.close(self.fd)


class WinFile:
    """Positioned reads on Windows: one blocking ReadFile per read with the offset in an OVERLAPPED, on a handle
    opened with FILE_FLAG_NO_BUFFERING when direct is true (the counterpart of O_DIRECT: the file cache is not
    used; offsets, sizes and the buffer must be multiples of the sector size)."""

    def __init__(self, path, direct):
        import ctypes
        from ctypes import wintypes

        class OVERLAPPED(ctypes.Structure):
            _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t), ('Offset', wintypes.DWORD),
                        ('OffsetHigh', wintypes.DWORD), ('hEvent', wintypes.HANDLE)]
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                    wintypes.DWORD, wintypes.HANDLE]
        k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                 ctypes.POINTER(OVERLAPPED)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.ct, self.k32, self.ov, self.got = ctypes, k32, OVERLAPPED(), wintypes.DWORD()
        self.buf = self.addr = None
        # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING
        self.h = k32.CreateFileW(path, 0x80000000, 1, None, 3, 0x20000000 if direct else 0, None)
        if self.h is None or self.h == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())

    def read(self, buf, off):
        ct = self.ct
        if buf is not self.buf:
            self.buf, self.view = buf, (ct.c_char * len(buf)).from_buffer(buf)
            self.addr = ct.addressof(self.view)
        ct.memset(ct.byref(self.ov), 0, ct.sizeof(self.ov))
        self.ov.Offset, self.ov.OffsetHigh = off & 0xFFFFFFFF, off >> 32
        if not self.k32.ReadFile(self.h, self.addr, len(buf), ct.byref(self.got), ct.byref(self.ov)):
            raise ct.WinError(ct.get_last_error())
        return self.got.value

    def close(self):
        self.k32.CloseHandle(self.h)
        self.buf = self.view = None


def build_run(src, exe, compiler, args):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, exe)
    if not shutil.which(compiler[0]):
        return {'skipped': f'{compiler[0]} not found'}
    rc, _, err = sh(compiler + [os.path.join(HERE, src), '-o', path], timeout=300)
    if rc != 0:
        return {'skipped': 'build failed', 'error': err[-400:]}
    rc, out, err = sh([path] + args, timeout=600)
    if rc != 0:
        return {'failed': err[-400:] or f'exit {rc}'}
    try:
        return json.loads(out)
    except ValueError:
        return {'failed': 'bad output', 'output': out[-400:]}


def make_file(path, size):
    block = os.urandom(8 << 20)   # incompressible, so ZFS compression does not flatter the numbers
    with open(path, 'wb') as f:
        left = size
        while left > 0:
            n = min(left, len(block))
            f.write(block[:n])
            left -= n
        f.flush()
        os.fsync(f.fileno())


Reader = WinFile if os.name == 'nt' else PosixFile


def open_reader(path):
    """(file, True) when the reads can go past the OS cache (O_DIRECT, FILE_FLAG_NO_BUFFERING), else (file, False)."""
    if os.name == 'nt' or hasattr(os, 'O_DIRECT'):
        try:
            return Reader(path, True), True
        except OSError:
            pass
    return Reader(path, False), False


def read_pass(path, offsets, rec, budget_s):
    f, direct = open_reader(path)
    buf = mmap.mmap(-1, rec)   # page-aligned, as O_DIRECT and FILE_FLAG_NO_BUFFERING need
    done = 0
    t0 = time.perf_counter()
    lat = []
    try:
        for off in offsets:
            t = time.perf_counter()
            try:
                n = f.read(buf, off)
            except OSError:
                if not direct:
                    raise
                f.close()
                f, direct = Reader(path, False), False
                n = f.read(buf, off)
            lat.append(time.perf_counter() - t)
            done += n
            if time.perf_counter() - t0 > budget_s:
                break
    finally:
        f.close()
    secs = time.perf_counter() - t0
    lat.sort()
    return {'o_direct': direct, 'reads': len(lat), 'gbs': round(done / secs / 1e9, 3),
            'lat_ms_p50': round(lat[len(lat) // 2] * 1e3, 3), 'lat_ms_p99': round(lat[int(len(lat) * 0.99)] * 1e3, 3)}


def disk(test_dir, existing, size_gb, quick):
    made = None
    if existing:
        path = existing
    else:
        path = os.path.join(test_dir, 'wex-bench.tmp')
        made = path
        t = time.perf_counter()
        try:
            make_file(path, size_gb << 30)
        except BaseException:   # a full disk or Ctrl-C while writing: do not leave gigabytes behind
            try:
                os.remove(path)
            except OSError:
                pass
            raise
        write_s = time.perf_counter() - t
    size = os.path.getsize(path)
    res = {'file_gib': round(size / (1 << 30), 2), 'existing_file': bool(existing)}
    if made:
        res['write_gbs_with_fsync'] = round(size / write_s / 1e9, 3)
    budget = 8 if quick else 30
    rng = random.Random(1)
    try:
        for name, rec in (('seq_8mib', 8 << 20), ('rand_1mib', 1 << 20), ('rand_18mib', 18 << 20)):
            count = max(1, (size - rec) // 4096)
            if name.startswith('seq'):
                offs = list(range(0, size - rec, rec))
            else:
                offs = [rng.randrange(count) * 4096 for _ in range(100000)]
            res[name] = {'first': read_pass(path, offs, rec, budget), 'second': read_pass(path, offs, rec, budget)}
    finally:
        if made:
            os.remove(made)
    return res


def find_bash():
    """The first bash on PATH. On Windows, folders under the Windows directory are left out: its bash.exe is the
    WSL launcher, which would run the checks in another system (and CreateProcess would find it before PATH)."""
    path = os.environ.get('PATH', '')
    if os.name == 'nt':
        windir = os.path.normcase(os.environ.get('SystemRoot', r'C:\Windows'))
        path = os.pathsep.join(p for p in path.split(os.pathsep) if not os.path.normcase(p).startswith(windir))
    return shutil.which('bash', path=path)


def wex_check():
    bash = find_bash()
    if not bash:
        return {'skipped': 'no bash on PATH' + (' (the bash.exe of WSL is not used)' if os.name == 'nt' else '')}
    t = time.perf_counter()
    py, script = sys.executable, os.path.join(ROOT, 'scripts', 'check_all.sh')
    if os.name == 'nt':   # the bash of MSYS2 or Git takes C:/... paths, not C:\...
        py, script = py.replace('\\', '/'), script.replace('\\', '/')
    env = dict(os.environ, PYTHON=py, USER='', USERNAME='')
    rc, out, err = sh([bash, script], timeout=3600, env=env)
    last = (out.splitlines() or [''])[-1]
    return {'passed': rc == 0 and last == 'all checks passed', 'seconds': round(time.perf_counter() - t, 1),
            'tail': (out + '\n' + err).strip().splitlines()[-5:] if rc != 0 else [last]}


def placeholders(a):
    """(pattern, placeholder) pairs for scrub(): the paths of this run in the spellings a message can have
    them in (as given, absolute, with / or \\, as /c/... and with the doubled backslashes of a Python error),
    longest first, then the folder of any other user profile (under "home" or "Users"), then the user's
    and the machine's names as whole words."""
    found = []
    for path, tag in ((a.file, '<FILE>'), (ROOT, '<Wex-WRN2>'), (a.dir, '<DIR>'), (os.path.expanduser('~'), '<HOME>')):
        if not path:
            continue
        for p in {os.path.abspath(path), os.path.realpath(path)} | ({path} if os.path.isabs(path) else set()):
            if os.path.dirname(os.path.abspath(p)) == os.path.abspath(p):   # a filesystem root: in every path
                continue
            for q in {p, p.replace('/', '\\'), p.replace('\\', '/')}:
                found += [(q, tag), (q.replace('\\', '\\\\'), tag)]
                if re.match(r'[A-Za-z]:[\\/]', q):
                    found.append(('/' + q[0].lower() + q[2:].replace('\\', '/'), tag))
    paths = [(re.escape(f), tag) for f, tag in sorted(set(found), key=lambda x: -len(x[0]))]
    paths.append((r'(?i)(?:\b[a-z]:)?[\\/]+(?:users|home)[\\/]+[^\\/\s\'"]+', '<HOME>'))
    users = {os.environ.get(k, '') for k in ('USER', 'USERNAME', 'LOGNAME')} | {os.path.basename(os.path.expanduser('~'))}
    try:
        users.add(getpass.getuser())
    except Exception:   # OSError on Windows when USERNAME is empty, KeyError without a passwd entry
        pass
    host = socket.gethostname()
    hosts = {host, host.split('.')[0], platform.node()}
    words = [(r'\b' + re.escape(n) + r'\b', tag) for names, tag in ((hosts, '<HOST>'), (users, '<USER>'))
             for n in sorted(names, key=len, reverse=True) if len(n) >= 3 and n != 'root']
    return paths, words


def scrub(obj, subs):
    """obj with every pattern of subs replaced in every string (case-insensitive on Windows)."""
    if isinstance(obj, str):
        for pattern, tag in subs:
            obj = re.sub(pattern, lambda _: tag, obj, flags=re.IGNORECASE if os.name == 'nt' else 0)
        return obj
    if isinstance(obj, list):
        return [scrub(x, subs) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub(v, subs) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dir', required=True, help='a directory on the volume the models live on')
    ap.add_argument('--file', help='an existing large file to read instead of creating one')
    ap.add_argument('--size-gb', type=int, default=None, help='size of the test file to create (default 16, 4 with --quick)')
    ap.add_argument('--quick', action='store_true', help='shorter runs (smaller buffers, 8 s per read test)')
    ap.add_argument('--skip', default='', help='comma list of sections to skip: ram_bw,h2d,disk,wex_check')
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(',')))
    paths, words = placeholders(a)
    if not os.path.isdir(a.dir):
        sys.exit(scrub(f'--dir {a.dir}: not a directory', paths + words))
    res = {'bench_version': 2, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'sysinfo': scrub(sysinfo(a.dir), paths)}
    steps = [
        ('ram_bw', lambda: build_run('ram_bw.c', 'ram_bw', ['gcc', '-O2', '-fopenmp'], ['256' if a.quick else '2048'])),
        ('h2d', lambda: build_run('h2d.cu', 'h2d', ['nvcc', '-O2'], [])),
        ('disk', lambda: disk(a.dir, a.file, a.size_gb if a.size_gb is not None else (4 if a.quick else 16), a.quick)),
        ('wex_check', wex_check),
    ]
    for name, fn in steps:
        if name in skip:
            res[name] = {'skipped': 'by --skip'}
            continue
        print(f'== {name} ...', flush=True)
        try:
            res[name] = fn()
        except Exception as e:   # one failing section must not lose the others
            res[name] = {'failed': f'{type(e).__name__}: {e}'}
        res[name] = scrub(res[name], paths + words)   # an error or a tail can hold a path or a name
        print(json.dumps(res[name])[:300], flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'results.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'\nwrote {os.path.relpath(path, ROOT)}\nRead it, then attach it to a "Benchmark results" issue on GitHub.')


if __name__ == '__main__':
    main()
