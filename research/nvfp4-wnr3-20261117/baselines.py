# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""General-purpose baselines on the slabs of analyze.py: zstd -19, lz4 (default and high compression level 12),
and WBP1 (codec 6 of WEX4, wex4_bp_encode_record of wex4/wex4_tools.c) with 1 plane and with 2 planes.

  python -I baselines.py RUNS FAST_DLL DIR [DIR ...]

Each DIR holds L<layer>_E<expert>.slab files of one checkpoint. Every slab is compressed whole, decompressed and
compared with itself before timing; the decode time is the median of RUNS passes after one pass that is not
counted, one thread, MB/s = decoded MB (10^6 B) per second. WBP1 sizes are the padded record (4096). Needs the
zstd module of Python 3.14 (compression.zstd) and the lz4 package.
"""
import json
import os
import statistics
import sys
import time
from compression import zstd

import lz4.frame

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'wex4'))
import wex4  # noqa: E402


def main(argv):
    runs, dll, dirs = int(argv[1]), argv[2], argv[3:]
    clib = wex4.CLib(dll)
    codecs = [
        ('zstd -19', lambda d: zstd.compress(d, 19), zstd.decompress),
        ('lz4 default', lambda d: lz4.frame.compress(d), lz4.frame.decompress),
        ('lz4 -12 (HC)', lambda d: lz4.frame.compress(d, compression_level=12), lz4.frame.decompress),
        ('WBP1 1 plane', lambda d: clib.bp_encode(d, 1, 15, 1), None),
        ('WBP1 2 planes', lambda d: clib.bp_encode(d, 2, 15, 1) if len(d) % 2 == 0 else None, None),
    ]
    for folder in dirs:
        names = sorted(f for f in os.listdir(folder) if f.endswith('.slab'))
        slabs = [open(os.path.join(folder, n), 'rb').read() for n in names]
        raw = sum(map(len, slabs))
        for row, comp, dec in codecs:
            packed = [comp(s) for s in slabs]
            if any(p is None for p in packed):
                continue
            if dec is None:
                def dec(p, n=None):
                    return clib.bp_decode(p, n, 1)
                pairs = list(zip(packed, [len(s) for s in slabs]))
                run = lambda: [dec(p, n) for p, n in pairs]
                assert all(dec(p, len(s)) == s for p, s in zip(packed, slabs)), row
            else:
                run = lambda: [dec(p) for p in packed]
                assert all(dec(p) == s for p, s in zip(packed, slabs)), row
            times = []
            for i in range(runs + 1):
                t0 = time.perf_counter()
                run()
                if i:
                    times.append(time.perf_counter() - t0)
            med = statistics.median(times)
            r = {'dir': os.path.basename(folder.rstrip('/')), 'codec': row, 'slabs': len(slabs), 'raw': raw,
                 'stored': sum(map(len, packed)), 'ratio': round(sum(map(len, packed)) / raw, 6),
                 'decode_mb_s_1thread': round(raw / med / 1e6, 1), 'runs': runs}
            print('json ' + json.dumps(r), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
