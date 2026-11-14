# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Hardware benchmark for WEX testers (Linux; Python 3 stdlib only).

Collects what decides how a streamed MoE model runs on a machine and writes ONE results.json:
  sysinfo    CPU, cores, RAM, GPUs, kernel, the filesystem of the test directory (no host or user names)
  ram_bw     RAM read / write / copy bandwidth, 1 thread and all threads (bench/ram_bw.c, needs gcc)
  h2d        host <-> device copies, pinned and pageable (bench/h2d.cu, needs nvcc; skipped otherwise)
  disk       4 KiB-aligned reads of a large file: sequential 8 MiB blocks and random 1 / 18 MiB records
             (expert-sized), with O_DIRECT when the filesystem accepts it, first pass vs second pass
  wex_check  the repository's CPU checks (scripts/check_all.sh): pass/fail and duration

  python3 bench/bench.py --dir /path/on/the/model/volume [--file BIG_EXISTING_FILE] [--size-gb 16] [--quick]

--dir is where a test file is created (and deleted at the end). --file reads an existing large file
instead (for example a model file not read recently: the closest to a cold read on ZFS, whose ARC
cache cannot be dropped without root). Nothing is sent anywhere: attach results.json to a GitHub issue.
"""
import argparse
import json
import mmap
import os
import platform
import random
import shutil
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
    return info


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


def open_reader(path):
    flags = os.O_RDONLY
    direct = hasattr(os, 'O_DIRECT')
    if direct:
        try:
            return os.open(path, flags | os.O_DIRECT), True
        except OSError:
            pass
    return os.open(path, flags), False


def read_pass(path, offsets, rec, budget_s):
    fd, direct = open_reader(path)
    buf = mmap.mmap(-1, rec)   # page-aligned, as O_DIRECT needs
    done = 0
    t0 = time.perf_counter()
    lat = []
    try:
        for off in offsets:
            t = time.perf_counter()
            try:
                n = os.preadv(fd, [buf], off)
            except OSError:
                if not direct:
                    raise
                os.close(fd)
                fd, direct = os.open(path, os.O_RDONLY), False
                n = os.preadv(fd, [buf], off)
            lat.append(time.perf_counter() - t)
            done += n
            if time.perf_counter() - t0 > budget_s:
                break
    finally:
        os.close(fd)
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
        make_file(path, size_gb << 30)
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


def wex_check():
    t = time.perf_counter()
    env = dict(os.environ, PYTHON=sys.executable, USER='', USERNAME='')
    rc, out, err = sh(['bash', os.path.join(ROOT, 'scripts', 'check_all.sh')], timeout=3600, env=env)
    last = (out.splitlines() or [''])[-1]
    return {'passed': rc == 0 and last == 'all checks passed', 'seconds': round(time.perf_counter() - t, 1),
            'tail': (out + '\n' + err).strip().splitlines()[-5:] if rc != 0 else [last]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dir', required=True, help='a directory on the volume the models live on')
    ap.add_argument('--file', help='an existing large file to read instead of creating one')
    ap.add_argument('--size-gb', type=int, default=16, help='size of the test file to create (default 16)')
    ap.add_argument('--quick', action='store_true', help='shorter runs (smaller buffers, 8 s per read test)')
    ap.add_argument('--skip', default='', help='comma list of sections to skip: ram_bw,h2d,disk,wex_check')
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(',')))
    if not os.path.isdir(a.dir):
        sys.exit(f'--dir {a.dir}: not a directory')
    res = {'bench_version': 1, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'sysinfo': sysinfo(a.dir)}
    steps = [
        ('ram_bw', lambda: build_run('ram_bw.c', 'ram_bw', ['gcc', '-O2', '-fopenmp'], ['256' if a.quick else '2048'])),
        ('h2d', lambda: build_run('h2d.cu', 'h2d', ['nvcc', '-O2'], [])),
        ('disk', lambda: disk(a.dir, a.file, 4 if a.quick else a.size_gb, a.quick)),
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
        print(json.dumps(res[name])[:300], flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'results.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'\nwrote {path}\nRead it, then attach it to a "Benchmark results" issue on GitHub.')


if __name__ == '__main__':
    main()
