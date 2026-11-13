# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""A4: WNR3 decoders on hostile input, in C (harness.c -> out/rv_harness.dll): random valid records (random shapes,
chunk sizes, contents, the real tiny model or random valid models) decoded by the reference decoder and by
wex_fast_decode in 19 kernel/lane/fuse configurations, then byte mutations of every record. Record copy and output
buffers end at guard pages. Also wex4_fuzz_run mode 1 of both product DLLs with fresh seeds.
    python -I -B a4_wnr3.py [scale]"""
import ctypes
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rvlib import *  # noqa: E402,F401,F403

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
H = ctypes.CDLL(os.path.join(OUT, 'rv_harness.dll'))
H.rv_nr3_campaign.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int,
                              ctypes.c_uint32, ctypes.c_void_p]
model = open(os.path.join(NOUT, 'tiny.wnm2'), 'rb').read()
NAMES = ['cases', 'roundtrip_failures', 'fast_ne_ref_on_valid', 'mutations', 'parser_refused', 'both_refused',
         'fast_ref_disagree_on_mutated', 'accepted_wrong_bytes', 'accepted_same_bytes', 'first_case', 'first_mutation',
         'first_option']
out = []
for label, seed, cases, muts, threads, max_scales in (
        ('small shapes, 1 thread', 11, 3000, 300, 1, 1024),
        ('medium shapes, 1 thread', 12, 400, 1500, 1, 8192),
        ('large shapes (many tasks), 8 threads', 13, 40, 1500, 8, 70000),
        ('small shapes, 8 threads', 14, 1500, 300, 8, 2048)):
    counts = (ctypes.c_uint64 * 76)()
    t = time.time()
    try:
        st = H.rv_nr3_campaign(model, seed, int(cases * SCALE), muts, threads, max_scales, counts)
    except OSError as e:
        st = repr(e)
    row = dict(label=label, seed=seed, status=st, seconds=round(time.time() - t, 1), **dict(zip(NAMES, counts)),
               same_bytes_edit_offsets={i: counts[12 + i] for i in range(64) if counts[12 + i]})
    print(row, flush=True)
    out.append(row)

# product DLLs: mode 1 (compressed record bytes, record SHA-256 off) with seeds the test suite does not use
for name in ('tiny2_cz.wex', ):
    pass
import shutil, io
import wex4_convert as conv
work = os.path.join(OUT, 'up_tiny2_cz.wex4')
shutil.copyfile(os.path.join(NOUT, 'tiny2_cz.wex'), work)
conv.upgrade(work)
corpora = [('upgraded tiny2 (2 WNR3-CZ experts, index last)', open(work, 'rb').read()),
           ('t_py (WBP1, generic)', open(os.path.join(NOUT, 't_py.wex4'), 'rb').read())]
for label, base in corpora:
    for mode in (1, 2):
        for seed in (777001, 777002):
            a, b = (ctypes.c_uint64 * 4)(), (ctypes.c_uint64 * 4)()
            n = int(40000 * SCALE)
            t = time.time()
            sa = C.lib.wex4_fuzz_run(base, len(base), seed, n, mode, a)
            sb = CREF.lib.wex4_fuzz_run(base, len(base), seed, n, mode, b)
            row = dict(label=label, mode=mode, seed=seed, n=n, status=[sa, sb], fast=list(a), ref=list(b),
                       agree=list(a) == list(b), seconds=round(time.time() - t, 1))
            print(row, flush=True)
            out.append(row)
json.dump(out, open(os.path.join(OUT, 'a4_wnr3.json'), 'w'), indent=1)
