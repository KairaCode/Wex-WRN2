# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Ratio and decode speed of the general-purpose codecs of the Python standard library on real expert slabs.

  python -I baselines.py RUNS DIR [--quick]

DIR holds the L<layer>_E<expert>.slab files that records.py extract wrote (one checkpoint). Codecs:
compression.zstd (Python 3.14) at levels 1, 3, 9, 19 and 22, lzma (xz) at presets 0, 6 and 9 | PRESET_EXTREME,
zlib at levels 1, 6 and 9, bz2 at level 9. Each slab is compressed whole, then decompressed and compared with
the slab before any timing. A run is one pass over the slabs: "1 thread" decodes them one after the other;
"T records at once" decodes T at a time on a thread pool (T = os.cpu_count(); with fewer slabs than T the
slabs are taken again until there are T), which is parallel only where the codec releases the GIL. One pass
that is not counted comes first. A row gives the median, the minimum and the maximum over RUNS of the time
per record and MB/s = decoded bytes per record / median, MB = 10^6 bytes; the same table and "json" lines as
bench_real.c (with the local time at which the row ended and how long it ran). --quick leaves out zstd 22 and lzma 9e (the slowest to compress). Standard library only.
"""
import bz2
import concurrent.futures as cf
import json
import lzma
import os
import statistics
import sys
import time
import zlib
from compression import zstd

CODECS = [
    ('zstd level 1 (compression.zstd)', lambda d: zstd.compress(d, 1), zstd.decompress),
    ('zstd level 3 (compression.zstd)', lambda d: zstd.compress(d, 3), zstd.decompress),
    ('zstd level 9 (compression.zstd)', lambda d: zstd.compress(d, 9), zstd.decompress),
    ('zstd level 19 (compression.zstd)', lambda d: zstd.compress(d, 19), zstd.decompress),
    ('zstd level 22 (compression.zstd)', lambda d: zstd.compress(d, 22), zstd.decompress),
    ('lzma preset 0', lambda d: lzma.compress(d, preset=0), lzma.decompress),
    ('lzma preset 6', lambda d: lzma.compress(d, preset=6), lzma.decompress),
    ('lzma preset 9e', lambda d: lzma.compress(d, preset=9 | lzma.PRESET_EXTREME), lzma.decompress),
    ('zlib level 1', lambda d: zlib.compress(d, 1), zlib.decompress),
    ('zlib level 6', lambda d: zlib.compress(d, 6), zlib.decompress),
    ('zlib level 9', lambda d: zlib.compress(d, 9), zlib.decompress),
    ('bz2 level 9', lambda d: bz2.compress(d, 9), bz2.decompress),
]
SLOW = ('zstd level 22 (compression.zstd)', 'lzma preset 9e')


def timed(row, mode, threads, decode, packed, raw, runs, ratio, pool):
    items = list(range(len(packed)))
    if threads > 1 and len(items) < threads:
        items = [i % len(packed) for i in range(threads)]
    times, began = [], time.perf_counter()
    for run in range(-1, runs):
        start = time.perf_counter()
        if threads > 1:
            list(pool.map(lambda i: decode(packed[i]), items))
        else:
            for i in items:
                decode(packed[i])
        if run >= 0:
            times.append((time.perf_counter() - start) * 1000.0 / len(items))
    median = statistics.median(times)
    mb_s = raw / median / 1000.0
    print(f'| {row} | {mode} | {ratio:.4f} | {median:.2f} | {min(times):.2f} to {max(times):.2f} | {mb_s:.1f} |')
    print('json ' + json.dumps({'row': row, 'mode': mode, 'threads': threads, 'records': len(packed), 'runs': runs,
                                'ratio': round(ratio, 6), 'median_ms': round(median, 4), 'min_ms': round(min(times), 4),
                                'max_ms': round(max(times), 4), 'mb_s': round(mb_s, 2),
                                'decoded_bytes_per_record': round(raw), 'ended': time.strftime('%H:%M:%S'),
                                'seconds': round(time.perf_counter() - began, 1)}, separators=(',', ':')), flush=True)


def main(argv):
    runs, folder = int(argv[1]), argv[2]
    names = sorted(f for f in os.listdir(folder) if f.endswith('.slab'))
    slabs = []
    for name in names:
        with open(os.path.join(folder, name), 'rb') as f:
            slabs.append(f.read())
    threads = os.cpu_count() or 1
    raw = sum(map(len, slabs)) / len(slabs)
    print(f'baselines: {len(slabs)} records, Python {sys.version.split()[0]}, zstd {zstd.zstd_version}, '
          f'zlib {zlib.ZLIB_RUNTIME_VERSION}; {threads} logical CPUs, {runs} runs per row, one pass not counted first')
    print('| decoder | threads | stored / raw | median ms/record | min to max ms | MB/s |\n|---|---|---:|---:|---|---:|')
    with cf.ThreadPoolExecutor(threads) as pool:
        for row, compress, decode in CODECS:
            if '--quick' in argv and row in SLOW:
                continue
            start = time.perf_counter()
            packed = list(pool.map(compress, slabs))
            print(f'compressed with {row}: {time.perf_counter() - start:.1f} s for the {len(slabs)} records, '
                  f'{threads} threads (not a speed measurement)', flush=True)
            if any(decode(p) != s for p, s in zip(packed, slabs)):
                raise SystemExit(f'MISMATCH {row}')
            ratio = sum(map(len, packed)) / sum(map(len, slabs))
            timed(row, '1', 1, decode, packed, raw, runs, ratio, pool)
            timed(row, f'{threads} records at once', threads, decode, packed, raw, runs, ratio, pool)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
