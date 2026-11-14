# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Linux benchmark of the read paths for expert records (Python 3 stdlib only, gcc; Linux only).

Writes a synthetic WEX4 container (random bytes, no model weights), builds bench/wex_io.c and reads the
container's expert records through each read path: pread through the page cache (with and without
posix_fadvise hints), pread with O_DIRECT, both on 1 to 8 threads, and io_uring with several records or
several sub-reads of one record in flight. Every path is first checked (SHA-256 of the records it reads
against the index), then timed `--runs` times; the runs of all paths are interleaved, every pass starts
from a page cache that does not hold the file (POSIX_FADV_DONTNEED, checked with mincore, plus
/proc/sys/vm/drop_caches with --drop-caches when the user may write it), and a run reads the same
records in the same order on every path.

  python3 -I bench/wex_io.py --dir /path/on/the/model/volume [--runs 5] [--records 48] [--drop-caches]

Prints a Markdown table (median, minimum and maximum MB/s over the runs) and writes bench/out/wex_io.json,
which holds every pass, the CPU, the kernel, the filesystem and the block device, but no host name, user
name or path. --file reads an existing WEX4 container instead of writing one. --smoke is a seconds-long
run on a tiny container that only shows that every path works (CI).
"""
import argparse
import json
import os
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'out')
sys.path[:0] = [HERE, os.path.join(ROOT, 'wex4')]   # python -I leaves the script's folder out
import bench   # noqa: E402  (sysinfo)
import wex4    # noqa: E402  (the reference writer)

# (label, options of wex_io, what it stands for)
CONFIGS = [
    ('page cache, pread, 1 thread', 'path=sync direct=0', 'c/expert_store_wex.c and c/tools/wex4_verify.c on Linux today'),
    ('page cache, pread, FADV_RANDOM', 'path=sync direct=0 advise=random', 'kernel readahead off'),
    ('page cache, pread, FADV_SEQUENTIAL', 'path=sync direct=0 advise=sequential', 'kernel readahead doubled'),
    ('page cache, pread, WILLNEED 1 record ahead', 'path=sync direct=0 ahead=1', 'the next record is announced'),
    ('page cache, pread, WILLNEED 4 records ahead', 'path=sync direct=0 ahead=4', 'the next 4 records are announced'),
    ('page cache, pread, 4 threads', 'path=sync direct=0 threads=4', ''),
    ('page cache, pread, 8 threads', 'path=sync direct=0 threads=8', ''),
    ('page cache, io_uring, 8 records in flight', 'path=uring direct=0 depth=8', ''),
    ('O_DIRECT, pread, 1 thread', 'path=sync direct=1', 'c/qwen38_wex.h on Linux today, one read'),
    ('O_DIRECT, pread, 2 threads', 'path=sync direct=1 threads=2', ''),
    ('O_DIRECT, pread, 4 threads', 'path=sync direct=1 threads=4', ''),
    ('O_DIRECT, pread, 8 threads', 'path=sync direct=1 threads=8', ''),
    ('O_DIRECT, io_uring, 1 record in flight', 'path=uring direct=1 depth=1', 'io_uring against pread, same work'),
    ('O_DIRECT, io_uring, 1 record as 4 sub-reads', 'path=uring direct=1 depth=1 split=4', 'a lone miss, split'),
    ('O_DIRECT, io_uring, 1 record as 16 sub-reads', 'path=uring direct=1 depth=1 split=16', 'a lone miss, split'),
    ('O_DIRECT, io_uring, 4 records in flight', 'path=uring direct=1 depth=4', ''),
    ('O_DIRECT, io_uring, 8 records in flight', 'path=uring direct=1 depth=8', ''),
    ('O_DIRECT, io_uring, 16 records in flight', 'path=uring direct=1 depth=16', ''),
    ('O_DIRECT, io_uring, 8 records x 4 sub-reads', 'path=uring direct=1 depth=8 split=4', ''),
    ('page cache already warm, pread, 1 thread', 'path=sync direct=0 warm=1', 'the records are in RAM: no drive in it'),
]


def block_device(path):
    """Queue settings of the block device under `path` (empty when the filesystem has no single device)."""
    try:
        dev = os.stat(path).st_dev
        real = os.path.realpath(f'/sys/dev/block/{os.major(dev)}:{os.minor(dev)}')
    except OSError:
        return {}
    if not os.path.isdir(os.path.join(real, 'queue')):
        real = os.path.dirname(real)   # a partition: the settings are on its disk
    info = {k: bench.read(os.path.join(real, 'queue', k)).strip()
            for k in ('read_ahead_kb', 'rotational', 'scheduler', 'max_sectors_kb', 'nr_requests', 'logical_block_size')}
    info['model'] = bench.read(os.path.join(real, 'device', 'model')).strip()
    return {k: v for k, v in info.items() if v}


def make_container(path, layers, experts, rows, cols, seed=1):
    """A WEX4 file with layers x experts RAW expert records (3 scale and 3 weight matrices each) of seeded
    random bytes: incompressible, so a compressing filesystem or a thin virtual disk cannot flatter the reads."""
    if cols % 32 or not 0 < rows <= 65535 or not 0 < cols // 2 <= 65535:
        sys.exit('--rows and --cols: 1 to 65535 rows, cols a multiple of 32 up to 131070')
    scale, weight = (rows, cols // 32), (rows, cols // 2)
    decoded = 3 * (scale[0] * scale[1] + weight[0] * weight[1])
    rng = random.Random(seed)
    records = []
    for layer in range(layers):
        for expert in range(experts):
            names = [f'L{layer}.E{expert}.w{m}.{kind}' for kind in ('scale', 'weight') for m in (1, 2, 3)]
            tensors = [wex4.Tensor(name, 'U8', shape, shape[0] * shape[1]) for name, shape in zip(names, [scale] * 3 + [weight] * 3)]
            records.append(wex4.Record(tensors, decoded, lambda: rng.getrandbits(decoded * 8).to_bytes(decoded, 'little'),
                                       'raw', wex4.REC_EXPERT, layer << 16 | expert))
    return wex4.write(path, records, layers=layers, experts=experts)


def drop_caches():
    """sync, then drop the page cache of the whole machine; False when this user may not."""
    script = 'sync; echo 3 > /proc/sys/vm/drop_caches'
    for cmd in (['sh', '-c', script], ['sudo', '-n', 'sh', '-c', script]):
        try:
            if subprocess.run(cmd, capture_output=True, timeout=120).returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
    return False


def one_pass(exe, container, options, **more):
    args = [exe, container] + options.split() + [f'{k}={v}' for k, v in more.items()]
    r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        return {'failed': (r.stderr.strip() or f'exit {r.returncode}')[-300:]}
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {'failed': 'bad output: ' + r.stdout[-200:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dir', required=True, help='a directory on the volume to measure; the container is written there')
    ap.add_argument('--file', help='an existing WEX4 container with expert records to read instead')
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--experts', type=int, default=48, help='experts per layer (default 8 x 48 = 384 records)')
    ap.add_argument('--rows', type=int, default=2048)
    ap.add_argument('--cols', type=int, default=4096, help='default 2048 x 4096: records of 13,369,344 bytes')
    ap.add_argument('--runs', type=int, default=5, help='timed passes per path (default 5)')
    ap.add_argument('--records', type=int, default=48, help='records read per pass (default 48)')
    ap.add_argument('--drop-caches', action='store_true', help='also drop the page cache of the machine before every pass')
    ap.add_argument('--keep', action='store_true', help='leave the container in --dir')
    ap.add_argument('--smoke', action='store_true', help='tiny container, 1 run: only shows that every path works')
    a = ap.parse_args()
    if not sys.platform.startswith('linux'):
        sys.exit('bench/wex_io.py measures the Linux read paths; it does nothing on this system')
    if not os.path.isdir(a.dir):
        sys.exit(f'--dir {a.dir}: not a directory')
    if a.smoke:
        a.layers, a.experts, a.rows, a.cols, a.runs, a.records = 2, 12, 256, 1024, 1, 8

    os.makedirs(OUT, exist_ok=True)
    exe = os.path.join(OUT, 'wex_io')
    cc = ['gcc', '-O2', '-Wall', '-Wextra', '-Wno-unused-function', '-Wno-unused-parameter', '-pthread'] + \
        shlex.split(os.environ.get('WEX_CFLAGS', '')) + [os.path.join(HERE, 'wex_io.c'), '-o', exe]
    r = subprocess.run(cc, capture_output=True, text=True)
    if r.returncode != 0 or r.stderr.strip():
        sys.exit('bench/wex_io.c does not build without warnings:\n' + r.stderr)

    res = {'bench': 'wex_io', 'version': 1, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'sysinfo': bench.sysinfo(a.dir), 'block_device': block_device(a.dir),
           'runs': a.runs, 'records_per_pass': a.records, 'smoke': a.smoke}
    container, made = a.file, None
    if not container:
        container = made = os.path.join(a.dir, 'wex-io-bench.wex4')
        size = 3 * a.rows * a.cols * 17 // 32 * a.layers * a.experts
        if shutil.disk_usage(a.dir).free < size * 1.1:
            sys.exit(f'--dir needs about {size / 1e9:.1f} GB free (smaller: --layers, --experts, --rows)')
        print(f'writing {a.layers} x {a.experts} expert records, about {size / 1e9:.2f} GB ...', flush=True)
        t = time.perf_counter()
        root = make_container(container, a.layers, a.experts, a.rows, a.cols)
        res['container'] = {'synthetic': True, 'layers': a.layers, 'experts': a.experts, 'rows': a.rows, 'cols': a.cols,
                            'root': root.hex(), 'write_seconds': round(time.perf_counter() - t, 1)}
    else:
        res['container'] = {'synthetic': False}
    res['container']['bytes'] = os.path.getsize(container)
    failed = []
    try:
        # 1. identity: every path returns the bytes the index names
        res['check'] = {}
        for label, options, _ in CONFIGS:
            got = one_pass(exe, container, options, records=min(8, a.records), seed=99, verify=1)
            res['check'][options] = got.get('skipped') or got.get('failed') or f"{got['records']} records match their SHA-256"
            if 'failed' in got:
                failed.append(f'{label}: {got["failed"]}')
            print(f'check  {label}: {res["check"][options]}', flush=True)
        # 2. speed: the runs of all paths interleaved, the order rotated from run to run
        res['drop_caches'] = bool(a.drop_caches and drop_caches())
        res['passes'] = {options: [] for _, options, _ in CONFIGS}
        todo = [c for c in CONFIGS if 'records' in res['check'][c[1]]]
        for run in range(a.runs):
            for label, options, _ in todo[run % len(todo):] + todo[:run % len(todo)] if todo else []:
                if res['drop_caches']:
                    drop_caches()
                got = one_pass(exe, container, options, records=a.records, seed=run + 1)
                if 'mb_s' not in got:
                    failed.append(f'{label}, run {run + 1}: {got}')
                    continue
                res['passes'][options].append(got)
                print(f'run {run + 1}  {label}: {got["mb_s"]} MB/s, cached before {got["cached_before"]}', flush=True)
    finally:
        if made and not a.keep and os.path.exists(made):
            os.remove(made)

    # 3. the table
    base = None
    rows = ['| Read path | MB/s, median | min to max | against row 1 | ms per record, median | in page cache before | bytes checked |',
            '|---|---|---|---|---|---|---|']
    res['summary'] = {}
    for label, options, note in CONFIGS:
        passes = res['passes'][options]
        if not passes:
            rows.append(f'| {label} | not run: {res["check"][options]} | | | | | |')
            continue
        speeds = [p['mb_s'] for p in passes]
        s = {'label': label, 'mb_s_median': round(statistics.median(speeds), 1), 'mb_s_min': min(speeds), 'mb_s_max': max(speeds),
             'record_ms_p50_median': round(statistics.median(p['record_ms_p50'] for p in passes), 2),
             'cached_before_max': max(p['cached_before'] for p in passes), 'runs': len(passes), 'note': note}
        res['summary'][options] = s
        base = base or s['mb_s_median']
        rows.append(f'| {label} | {s["mb_s_median"]:.0f} | {s["mb_s_min"]:.0f} to {s["mb_s_max"]:.0f} | '
                    f'{s["mb_s_median"] / base:.2f}x | {s["record_ms_p50_median"]:.1f} | {s["cached_before_max"] * 100:.1f}% | '
                    f'{res["check"][options]} |')
    table = '\n'.join(rows)
    res['table_markdown'] = table
    path = os.path.join(OUT, 'wex_io.json')
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'\n{a.runs} runs per path, {a.records} records per pass, MB = 1e6 bytes; '
          f'drop_caches {"used" if res["drop_caches"] else "not used"}\n\n{table}\n\nwrote {path}')
    if failed:
        sys.exit('FAILED:\n' + '\n'.join(failed))


if __name__ == '__main__':
    main()
