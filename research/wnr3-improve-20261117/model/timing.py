"""Interleaved A/B timing of the prototype decoder (scalar, binary search, one thread, no SIMD) on model
variants, to see whether table sets / precision / scale alphabet change decode time (per-symbol work is
the same; only table footprint and per-stream set choice differ). 3 TEST records, R rounds, variants
alternated in rotating order each round. Prints median and min..max ms per record. Writes timing.json."""
import time, json, ctypes, numpy as np, sys
from common import *
import real
train, test = split()
VARS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["P14", "P14_C16_K1", "P14_K4", "P16", "P14_sesc"]
R = 7
recs = ["05_080", "20_107", "33_028"]
built = {v: real.build(v)[1] for v in VARS}
enc = {(v, t): built[v].encode(t)[0] for v in VARS for t in recs}
for (v, t), r in enc.items(): assert built[v].decode_ok(t, r)
out = ctypes.create_string_buffer(RAW)
T = {v: [] for v in VARS}
for rd in range(R):
    order = VARS[rd % len(VARS):] + VARS[:rd % len(VARS)]
    for v in order:
        m = built[v]; tt = 0.0
        for t in recs:
            r = enc[(v, t)]; e = entry(t)
            t0 = time.perf_counter(); rc = LIB.m_decode(ctypes.byref(m.m), e, r, ctypes.c_size_t(len(r)), m.model_id, out); tt += time.perf_counter() - t0
            assert rc == 0
        T[v].append(tt / len(recs) * 1000)
res = {}
for v in VARS:
    a = np.array(T[v]); res[v] = dict(median_ms=float(np.median(a)), min_ms=float(a.min()), max_ms=float(a.max()), rounds=T[v])
    print("%-14s median %.0f ms/record (%.0f..%.0f)  ratio to %s %.3f" % (v, np.median(a), a.min(), a.max(), VARS[0], np.median(a) / np.median(T[VARS[0]])))
json.dump(dict(records=recs, res=res, cpu="Xeon 2.1 GHz shared, scalar prototype, -O2"), open(W + "/timing.json", "w"), indent=1)
