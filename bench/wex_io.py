# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Linux and macOS benchmark of the read paths for expert records (Python 3 stdlib only, gcc or the compiler named in CC).

Writes a synthetic WEX4 container (random bytes, no model weights), builds bench/wex_io.c and reads the
container's expert records through each read path: pread through the page cache (with and without
posix_fadvise hints), pread through the page cache with the next record prefetched (POSIX_FADV_WILLNEED on
the whole record or in chunks of the device readahead, readahead(2), or a helper thread that preads it;
some also with 20 ms of work after each record, a spin that stands for a decode),
pread with O_DIRECT, both on 1 to 8 threads, and io_uring with several records or
several sub-reads of one record in flight. Every path is first checked (SHA-256 of the records it reads
against the index), then timed `--runs` times; the runs of all paths are interleaved, every pass starts
from a page cache that does not hold the file (POSIX_FADV_DONTNEED, checked with mincore, plus
/proc/sys/vm/drop_caches with --drop-caches when the user may write it), and a run reads the same
records in the same order on every path.

  python3 -I bench/wex_io.py --dir /path/on/the/model/volume [--runs 5] [--records 48] [--drop-caches]

Prints a Markdown table (median, minimum and maximum MB/s over the runs) and writes bench/out/wex_io.json,
which holds every pass, the CPU, the kernel, the filesystem and the block device, but no host name, user
name or path. --file reads an existing WEX4 container instead of writing one (no --dir needed: the
filesystem and the block device recorded are those of the file). --smoke is a seconds-long run on a tiny
container that only shows that every path works; --strict makes a path that was skipped a failure (CI
uses both).

On macOS the same rows run through the macOS calls (macos_row below, and the blocks __APPLE__ of
wex_io.c): F_NOCACHE where Linux has O_DIRECT, fcntl F_RDAHEAD 0 and F_RDADVISE for the posix_fadvise hints,
msync MS_INVALIDATE to drop the file's pages before a pass, and --drop-caches runs purge (as root, through
sudo without a password, when the user may). The io_uring rows, the POSIX_FADV_SEQUENTIAL row and the
readahead(2) row do not exist there: wex_io prints them as skipped, and --strict does not count those skips.
The other prefetch rows (WILLNEED in chunks through F_RDADVISE, the helper thread) and the rows with work
between records run as on Linux; the chunk is 128 KiB there, as macOS gives no readahead size. The JSON then holds
the CPU, the RAM, the page size and the disk from sysctl and diskutil, again without host name or path.
Neither the msync nor purge reaches a cache below the kernel, such as the host of a virtual machine.
"""
import argparse
import json
import os
import plistlib
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
GAP_BASE = 'path=sync direct=0 gap_ms=20'   # the rows with work between records are compared with this one
CONFIGS = [
    ('page cache, pread, 1 thread', 'path=sync direct=0', 'c/expert_store_wex.c and c/tools/wex4_verify.c on Linux today'),
    ('page cache, pread, FADV_RANDOM', 'path=sync direct=0 advise=random', 'kernel readahead off'),
    ('page cache, pread, FADV_SEQUENTIAL', 'path=sync direct=0 advise=sequential', 'kernel readahead doubled'),
    ('page cache, pread, WILLNEED 1 record ahead', 'path=sync direct=0 ahead=1',
     'the next record is announced; how much of it the kernel then holds is measured (ahead_cached)'),
    ('page cache, pread, WILLNEED 4 records ahead', 'path=sync direct=0 ahead=4',
     'the next 4 records are announced; how much of them the kernel then holds is measured (ahead_cached)'),
    ('page cache, pread, readahead(2) 1 record ahead', 'path=sync direct=0 ahead=1 prefetch=readahead',
     'one readahead(2) call on the whole next record (ahead_cached measured)'),
    ('page cache, pread, WILLNEED in chunks 1 record ahead', 'path=sync direct=0 ahead=1 prefetch=chunks',
     'the next record announced as chunks of the device readahead, one POSIX_FADV_WILLNEED each (ahead_cached measured)'),
    ('page cache, pread, helper thread 1 record ahead', 'path=sync direct=0 ahead=1 prefetch=thread',
     'a second thread preads the next record into the page cache while the first reads (ahead_cached measured)'),
    # the same with 20 ms of work after each record (a spin on the clock, no decode): time for a prefetch to land
    ('page cache, pread, 1 thread, 20 ms of work per record', GAP_BASE,
     'after each record the thread spins 20 ms, standing for a decode; the baseline of the next three rows'),
    ('page cache, pread, WILLNEED 1 record ahead, 20 ms of work', 'path=sync direct=0 ahead=1 gap_ms=20',
     'the hint of row 4 with time to land'),
    ('page cache, pread, WILLNEED in chunks 1 record ahead, 20 ms of work', 'path=sync direct=0 ahead=1 prefetch=chunks gap_ms=20',
     'the hint of row 7 with time to land'),
    ('page cache, pread, helper thread 1 record ahead, 20 ms of work', 'path=sync direct=0 ahead=1 prefetch=thread gap_ms=20',
     'the helper thread of row 8 with time to land'),
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
    scale, weight = (rows, cols // 32), (rows, cols // 2)
    decoded = 3 * (scale[0] * scale[1] + weight[0] * weight[1])
    if cols % 32 or not 0 < rows <= 65535 or not 0 < cols // 2 <= 65535 or decoded > wex4.MAX_DECODED:
        sys.exit('--rows and --cols: 1 to 65535 rows, cols a multiple of 32 up to 131040, '
                 'and 3 x rows x cols x 17/32 at most 2^30 bytes per record')
    rng = random.Random(seed)
    records = []
    for layer in range(layers):
        for expert in range(experts):
            names = [f'L{layer}.E{expert}.w{m}.{kind}' for kind in ('scale', 'weight') for m in (1, 2, 3)]
            tensors = [wex4.Tensor(name, 'U8', shape, shape[0] * shape[1]) for name, shape in zip(names, [scale] * 3 + [weight] * 3)]
            records.append(wex4.Record(tensors, decoded, lambda: b''.join(rng.getrandbits(8 * min(1 << 26, decoded - i)).to_bytes(min(1 << 26, decoded - i), 'little')
                                                                       for i in range(0, decoded, 1 << 26)),
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


def one_pass(exe, container, options, chunk_kb, **more):
    if 'prefetch=chunks' in options.split():
        more['chunk_kb'] = chunk_kb
    args = [exe, container] + options.split() + [f'{k}={v}' for k, v in more.items()]
    r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        return {'failed': (r.stderr.strip() or f'exit {r.returncode}')[-300:]}
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {'failed': 'bad output: ' + r.stdout[-200:]}


# macOS. The Linux code above and below is what runs on Linux; on macOS the names of this block replace
# CONFIGS, drop_caches and block_device (at its end). macos_row turns each row of CONFIGS into its macOS row:
# the same options, so the JSON of the two systems has the same keys, with a label and a note that name the
# macOS call (F_NOCACHE where Linux has O_DIRECT, F_RDAHEAD 0 for POSIX_FADV_RANDOM, F_RDADVISE for WILLNEED)
MACOS_LABELS = (('O_DIRECT, pread', 'F_NOCACHE, pread'), ('FADV_RANDOM', 'F_RDAHEAD off'), ('WILLNEED', 'F_RDADVISE'))
MACOS_NOTES = {
    'path=sync direct=0': 'c/expert_store_wex.c on macOS today (open and pread, no hint)',
    'path=sync direct=0 advise=random': 'fcntl F_RDAHEAD 0: readahead off on the descriptor',
    'path=sync direct=1': 'fcntl F_NOCACHE; on macOS c/qwen38_wex.h opens through compat_open_direct of the engine, '
                          'which is not in this repository',
    'path=sync direct=0 ahead=1 prefetch=chunks': 'the next record announced as chunks of 128 KiB (macOS gives no '
                                                  'readahead size), one fcntl F_RDADVISE each (ahead_cached measured)',
}
# options of the rows macOS cannot have (wex_io prints them skipped): io_uring, POSIX_FADV_SEQUENTIAL and readahead(2).
# Every other row of CONFIGS runs there through the stand-ins of wex_io.c: the prefetch rows with F_RDADVISE or the
# helper thread, the rows with work between records as on Linux
MACOS_ABSENT = ('path=uring', 'advise=sequential', 'prefetch=readahead')
NOT_ON_THIS_SYSTEM = set()   # options of the rows this system cannot have: --strict does not count their skip


def macos_row(label, options, note):
    """The macOS form of one row of CONFIGS."""
    if any(word in options.split() for word in MACOS_ABSENT):
        return label, options, 'not on macOS: skipped'
    for linux, mac in MACOS_LABELS:
        label = label.replace(linux, mac)
    return label, options, MACOS_NOTES.get(options, note)


def drop_caches_macos():
    """sync, then purge, which empties the file cache of the whole machine; False when this user may not."""
    for cmd in (['sh', '-c', 'sync && purge'], ['sudo', '-n', 'sh', '-c', 'sync && purge']):
        try:
            if subprocess.run(cmd, capture_output=True, timeout=300).returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
    return False


def block_device_macos(path):
    """The filesystem under `path` and the whole disk it is on, from diskutil: a few fields, no volume name, no UUID."""
    def plist(target):
        rc, out, _ = bench.sh(['diskutil', 'info', '-plist', target])
        try:
            return plistlib.loads(out.encode()) if rc == 0 and out else {}
        except Exception:   # plistlib raises several kinds; the fields then stay empty
            return {}
    rc, out, _ = bench.sh(['/bin/df', path])
    lines = out.splitlines() if rc == 0 else []
    volume = plist(lines[-1].split()[0]) if len(lines) > 1 and lines[-1].startswith('/dev/') else {}
    stores = volume.get('APFSPhysicalStores') or [{}]
    store = plist(stores[0].get('APFSPhysicalStore', '')) if stores[0] else volume
    disk = plist(store.get('ParentWholeDisk', '')) if store.get('ParentWholeDisk') else store
    info = {'filesystem': volume.get('FilesystemType'), 'model': disk.get('MediaName'), 'bus_protocol': disk.get('BusProtocol'),
            'solid_state': disk.get('SolidState'), 'device_block_size': disk.get('DeviceBlockSize'), 'internal': disk.get('Internal')}
    return {k: v for k, v in info.items() if v not in (None, '')}


def sysinfo_macos():
    """What sysctl says of the machine (no host name, no serial number). in_vm is kern.hv_vmm_present: 1 inside a
    virtual machine."""
    keys = (('cpu', 'machdep.cpu.brand_string'), ('hw_model', 'hw.model'), ('in_vm', 'kern.hv_vmm_present'),
            ('physical_cpus', 'hw.physicalcpu'), ('logical_cpus', 'hw.logicalcpu'), ('mem_bytes', 'hw.memsize'),
            ('page_bytes', 'hw.pagesize'), ('macos', 'kern.osproductversion'), ('macos_build', 'kern.osversion'))
    info = {}
    for name, key in keys:
        rc, out, _ = bench.sh(['sysctl', '-n', key])
        if rc == 0 and out:
            info[name] = int(out) if out.isdigit() else out
    return info


if sys.platform == 'darwin':
    CONFIGS = [macos_row(*config) for config in CONFIGS]
    drop_caches, block_device = drop_caches_macos, block_device_macos
    NOT_ON_THIS_SYSTEM = {options for _, options, _ in CONFIGS if any(word in options.split() for word in MACOS_ABSENT)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dir', help='a directory on the volume to measure; the container is written there')
    ap.add_argument('--file', help='an existing WEX4 container with expert records to read instead (then no --dir)')
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--experts', type=int, default=48, help='experts per layer (default 8 x 48 = 384 records)')
    ap.add_argument('--rows', type=int, default=2048)
    ap.add_argument('--cols', type=int, default=4096, help='default 2048 x 4096: records of 13,369,344 bytes')
    ap.add_argument('--runs', type=int, default=5, help='timed passes per path (default 5)')
    ap.add_argument('--records', type=int, default=48, help='records read per pass (default 48)')
    ap.add_argument('--drop-caches', action='store_true', help='also drop the page cache of the machine before every pass')
    ap.add_argument('--keep', action='store_true', help='leave the container in --dir')
    ap.add_argument('--smoke', action='store_true', help='tiny container, 1 run: only shows that every path works')
    ap.add_argument('--strict', action='store_true', help='a read path that is skipped counts as a failure')
    a = ap.parse_args()
    if not sys.platform.startswith(('linux', 'darwin')):
        sys.exit('bench/wex_io.py measures the Linux and macOS read paths; it does nothing on this system')
    if bool(a.dir) == bool(a.file):
        sys.exit('give --dir (a container is written there) or --file (an existing container), not both')
    if a.file and not os.path.isfile(a.file):
        sys.exit(f'--file {a.file}: not a file')
    if a.dir and not os.path.isdir(a.dir):
        sys.exit(f'--dir {a.dir}: not a directory')
    where = a.dir or os.path.dirname(os.path.abspath(a.file))   # the volume that is read
    if a.smoke:
        a.layers, a.experts, a.rows, a.cols, a.runs, a.records = 2, 12, 256, 1024, 1, 8

    os.makedirs(OUT, exist_ok=True)
    exe = os.path.join(OUT, 'wex_io')
    # CC names the compiler as in the build scripts (default gcc; CI also builds with clang, under ThreadSanitizer), WEX_CFLAGS adds flags
    cc = [os.environ.get('CC') or 'gcc', '-O2', '-Wall', '-Wextra', '-Wno-unused-function', '-Wno-unused-parameter', '-pthread'] + \
        shlex.split(os.environ.get('WEX_CFLAGS', '')) + [os.path.join(HERE, 'wex_io.c'), '-o', exe]
    r = subprocess.run(cc, capture_output=True, text=True)
    if r.returncode != 0 or r.stderr.strip():
        sys.exit('bench/wex_io.c does not build without warnings:\n' + r.stderr)

    res = {'bench': 'wex_io', 'version': 3, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'sysinfo': bench.sysinfo(where), 'block_device': block_device(where),
           'runs': a.runs, 'records_per_pass': a.records, 'smoke': a.smoke, 'strict': a.strict}
    # the chunk of prefetch=chunks: the device's readahead, so that no POSIX_FADV_WILLNEED call asks for more
    # than the kernel reads ahead on its own; 128 KiB when the device does not say
    ra = res['block_device'].get('read_ahead_kb', '')
    chunk = int(ra) // 4 * 4 if ra.isdigit() and int(ra) >= 4 else 0
    res['prefetch_chunk_kb'] = {'kb': chunk or 128,
                                'from': 'read_ahead_kb of the block device' if chunk else 'default, read_ahead_kb not known'}
    if sys.platform == 'darwin':
        res['sysinfo'].update(sysinfo_macos())
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
        todo = []   # the paths that passed the check: only these are timed
        for config in CONFIGS:
            label, options, _ = config
            got = one_pass(exe, container, options, res['prefetch_chunk_kb']['kb'], records=min(8, a.records), seed=99, verify=1)
            if 'failed' in got:
                res['check'][options] = 'failed: ' + got['failed']
                failed.append(f'{label}: {got["failed"]}')
            elif 'skipped' in got:
                res['check'][options] = 'skipped: ' + got['skipped']
                if a.strict and options not in NOT_ON_THIS_SYSTEM:
                    failed.append(f'{label}: skipped ({got["skipped"]}), and --strict was given')
            else:
                res['check'][options] = f"{got['records']} records match their SHA-256"
                todo.append(config)
            print(f'check  {label}: {res["check"][options]}', flush=True)
        # 2. speed: the runs of all paths interleaved, the order rotated from run to run
        res['drop_caches'] = bool(a.drop_caches and drop_caches())
        res['passes'] = {options: [] for _, options, _ in CONFIGS}
        for run in range(a.runs):
            for label, options, _ in todo[run % len(todo):] + todo[:run % len(todo)] if todo else []:
                if res['drop_caches']:
                    drop_caches()
                got = one_pass(exe, container, options, res['prefetch_chunk_kb']['kb'], records=a.records, seed=run + 1)
                if 'mb_s' not in got:
                    failed.append(f'{label}, run {run + 1}: {got}')
                    continue
                res['passes'][options].append(got)
                seen = (f', announced records in page cache before their read {got["ahead_cached"]}, '
                        f'announcing calls {got["announce_ms"]} ms') if got.get('ahead') else ''
                print(f'run {run + 1}  {label}: {got["mb_s"]} MB/s, cached before {got["cached_before"]}{seen}', flush=True)
    finally:
        if made and not a.keep and os.path.exists(made):
            os.remove(made)

    # 3. the table
    base, notes = None, []   # base: the median of row 1; the column stays empty when row 1 did not run
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
        seen = [p['ahead_cached'] for p in passes if p.get('ahead') and p.get('ahead_cached', -1) >= 0]
        if any(p.get('ahead') for p in passes):
            s['announce_ms_median'] = round(statistics.median(p['announce_ms'] for p in passes), 1)
        if seen:
            s['ahead_cached_median'], s['ahead_cached_min'], s['ahead_cached_max'] = statistics.median(seen), min(seen), max(seen)
            notes.append(f'{label}: {s["ahead_cached_median"] * 100:.1f}% ({min(seen) * 100:.1f}% to {max(seen) * 100:.1f}%) '
                         f'of the pages of the announced records were in the page cache just before their read; '
                         f'the announcing calls took {s["announce_ms_median"]:.1f} ms of the pass (median)')
        elif any(p.get('ahead') for p in passes):
            notes.append(f'{label}: what the kernel read ahead was not measured (mincore failed or pages are not 4096 bytes)')
        res['summary'][options] = s
        if options == CONFIGS[0][1]:
            base = s['mb_s_median']
        gap = res['summary'].get(GAP_BASE)
        if 'gap_ms=' in options and options != GAP_BASE and gap and gap['mb_s_median'] > 0 and gap['record_ms_p50_median'] > 0:
            notes.append(f'{label}: {s["mb_s_median"] / gap["mb_s_median"]:.2f}x the MB/s and '
                         f'{s["record_ms_p50_median"] / gap["record_ms_p50_median"]:.2f}x the ms per record of "{gap["label"]}"')
        against = f'{s["mb_s_median"] / base:.2f}x' if base else ''
        rows.append(f'| {label} | {s["mb_s_median"]:.0f} | {s["mb_s_min"]:.0f} to {s["mb_s_max"]:.0f} | '
                    f'{against} | {s["record_ms_p50_median"]:.1f} | {s["cached_before_max"] * 100:.1f}% | '
                    f'{res["check"][options]} |')
    table = '\n'.join(rows + [''] + ['- ' + x for x in notes]) if notes else '\n'.join(rows)
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
