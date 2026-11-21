# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""WNR4-GG (wnr4/wnr4_gg.h through wnr4/tools/wnr4_gg_tools.c) on the sample of sample_pack.py: trains a model on the
training records, encodes and decodes every record at G = 32, checks byte identity and prints ratio and MB/s.

    python -I measure_pack.py SAMPLE_DIR wnr4_gg_tools.dll [DECODE_THREADS]

Encode runs on one thread; decode on DECODE_THREADS (default 1). Ratio = coded record bytes (the WNR4 record as
the codec returns it, which the record format pads to a multiple of 4096 bytes) / raw bytes. "down~" is the down slice's ratio, taking the
gate and up slices as stored verbatim. Standard library + ctypes."""
import ctypes as C
import json
import os
import sys
import time
from collections import defaultdict

G = 32


def main(sample_dir, dll, threads=1):
    lib = C.CDLL(os.path.abspath(dll))
    u32p, u64p, ub = C.POINTER(C.c_uint32), C.POINTER(C.c_uint64), C.c_char_p
    lib.wnr4gg_abi.restype = C.c_uint32
    if lib.wnr4gg_abi() != 1:
        raise SystemExit('wnr4_gg_tools ABI')
    lib.wnr4gg_counts_words.restype = C.c_uint64
    lib.wnr4gg_model_bytes.restype = C.c_uint32
    lib.wnr4gg_count.argtypes = [C.c_uint32, u32p, u32p, ub, u64p]
    lib.wnr4gg_build.restype = C.c_int
    lib.wnr4gg_build.argtypes = [u64p, C.c_void_p]
    lib.wnr4gg_model_open.restype = C.c_void_p
    lib.wnr4gg_model_open.argtypes = [C.c_void_p, C.c_uint64]
    lib.wnr4gg_model_close.argtypes = [C.c_void_p]
    lib.wnr4gg_encode.restype = C.c_uint64
    lib.wnr4gg_encode.argtypes = [C.c_void_p, C.c_uint32, u32p, u32p, ub, C.c_uint32, C.c_uint32, C.c_void_p,
                                  C.c_uint64]
    lib.wnr4gg_decode.restype = C.c_uint32
    lib.wnr4gg_decode.argtypes = [C.c_void_p, C.c_uint32, u32p, u32p, C.c_void_p, C.c_uint64, C.c_uint32,
                                  C.c_void_p, C.c_int]
    with open(os.path.join(sample_dir, 'sample.json')) as f:
        recs = json.load(f)
    with open(os.path.join(sample_dir, 'sample.bin'), 'rb') as f:
        data = f.read()

    def geo(r):
        return (C.c_uint32 * 3)(*r['types']), (C.c_uint32 * 3)(*r['bytes'])

    def slab(r):
        return data[r['at']:r['at'] + sum(r['bytes'])]

    counts = (C.c_uint64 * lib.wnr4gg_counts_words())()
    for r in recs:
        if r['train']:
            t, b = geo(r)
            if lib.wnr4gg_count(3, t, b, slab(r), counts):
                raise SystemExit(f'record {r}: geometry refused')
    blob = C.create_string_buffer(lib.wnr4gg_model_bytes())
    if lib.wnr4gg_build(counts, blob):
        raise SystemExit('model build: out of memory')
    m = lib.wnr4gg_model_open(blob, len(blob.raw))
    if not m:
        raise SystemExit('model refused')

    tot = defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0])  # key -> records, raw, coded, encode s, decode s, down bytes
    bad = 0
    for r in recs:
        t, b = geo(r)
        src = slab(r)
        raw = len(src)
        cap = raw + 65536
        rec, out = C.create_string_buffer(cap), C.create_string_buffer(raw)
        owner = r['L'] * 256 + r['e']
        t0 = time.perf_counter()
        n = lib.wnr4gg_encode(m, 3, t, b, src, owner, G, rec, cap)
        t1 = time.perf_counter()
        if not n:
            raise SystemExit(f'record {r}: encode failed')
        t2 = time.perf_counter()
        err = lib.wnr4gg_decode(m, 3, t, b, rec, n, owner, out, threads)
        t3 = time.perf_counter()
        bad += not (err == 0 and out.raw == src)
        keys = ['train' if r['train'] else 'test']
        if not r['train']:
            keys += ['test d=%d' % r['types'][2], 'test gu=%d' % r['types'][0]]
        for k in keys:
            a = tot[k]
            a[0] += 1
            a[1] += raw
            a[2] += n
            a[3] += t1 - t0
            a[4] += t3 - t2
            a[5] += r['bytes'][2]
    lib.wnr4gg_model_close(m)
    print('G=%d decode threads=%d  byte-identical failures: %d / %d' % (G, threads, bad, len(recs)))
    for k in sorted(tot):
        n, raw, comp, es, ds, dn = tot[k]
        print('%-12s rec %4d raw %11d coded %11d ratio %.4f  down~%.4f  enc %7.1f MB/s  dec %7.1f MB/s' %
              (k, n, raw, comp, comp / raw, (comp - (raw - dn)) / dn, raw / es / 1e6, raw / ds / 1e6))
    return bad


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    sys.exit(1 if main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else 1) else 0)
