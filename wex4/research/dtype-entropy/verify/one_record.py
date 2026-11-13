#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""embed.weight as ONE record with one table per plane (estimate): body histogram from the 32 stratified
500-row slices scaled to rows [0, 126155), tail rows [126155, 129280) read exactly.  python -I one_record.py <model_dir>"""
import sys
from math import ceil
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import verify as V  # noqa: E402

idx = V.index(sys.argv[1])
name, R, T0 = "embed.weight", 129280, 126155
planes = lambda u: {"lo": u & 255, "hi": u >> 8, "exp8": (u >> 7) & 255}
hist = lambda u: {k: np.bincount(p.ravel().astype(np.intp), minlength=256) for k, p in planes(u).items()}
body = None
for i in range(32):
    a = i * (R // 32) + (R // 32 - 500) // 2
    if a + 500 > T0:
        continue
    h = hist(V.read(idx, name, (a, a + 500)))
    body = h if body is None else {k: body[k] + h[k] for k in h}
body_rows = 31 * 500
tail = [hist(V.read(idx, name, (a, min(a + 250, R)))) for a in range(T0, R, 250)]
tail_sum = {k: sum(t[k] for t in tail) for k in body}
C = 5120
xent = lambda c, f: float((c[c > 0] * (14 - np.log2(f[c > 0]))).sum()) / 8
for label, keys, side in (("WBP1 (lo + hi planes)", ("lo", "hi"), 0), ("exp8 coded + 8 raw bits", ("exp8",), 1)):
    whole = own_tail = tail_bodytab = body_cost = 0.0
    for k in keys:
        est = body[k] * (T0 / body_rows) + tail_sum[k]                 # estimated whole-tensor histogram
        f = V.quant(np.round(est).astype(np.int64))
        fb = V.quant(body[k] + 1)
        whole += xent(est, f) + R * C / V.CH * V.OVH + 514
        body_cost += xent(body[k], V.quant(body[k])) * (T0 / body_rows)
        tail_bodytab += xent(tail_sum[k], fb)
        own_tail += sum(xent(t[k], V.quant(t[k])) + 514 for t in tail)
    raw, traw = R * C * 2, (R - T0) * C * 2
    print(f"{label}: whole tensor as one record {(whole + side * R * C) / raw:.5f} | "
          f"tail rows with the body table {(tail_bodytab + side * (R - T0) * C + (R - T0) * C / V.CH * V.OVH * len(keys)) / traw:.5f} "
          f"vs with 250-row tables {(own_tail + side * (R - T0) * C + (R - T0) * C / V.CH * V.OVH * len(keys)) / traw:.5f}")
print(f"read {V.READ:,} B")
