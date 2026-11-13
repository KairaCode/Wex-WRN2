# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Read-only: compare a slab decoded by the fast decoder with the safetensors source (canonical zero applied)."""
import json, struct, sys, glob, os
import numpy as np
D = r'U:\models\DeepSeek-V4.1-Flash-1017-REAP192-MXFP4'
idx = {}
for p in sorted(glob.glob(os.path.join(D, '*.safetensors'))):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n))
    for k, v in h.items():
        if k != '__metadata__' and '.ffn.experts.' in k and k.startswith('layers.'): idx[k] = (p, 8 + n + v['data_offsets'][0], v['data_offsets'][1] - v['data_offsets'][0], v['dtype'], v['shape'])
print('expert tensors indexed:', len(idx))
def rd(name):
    p, o, n, dt, sh = idx[name]
    with open(p, 'rb') as f:
        f.seek(o); return np.frombuffer(f.read(n), dtype=np.uint8)
for arg in sys.argv[1:]:
    L, E, slab = arg.split(',')
    got = np.fromfile(slab, dtype=np.uint8)
    sc = [rd(f'layers.{L}.ffn.experts.{E}.{m}.scale') for m in ('w1', 'w2', 'w3')]
    wt = [rd(f'layers.{L}.ffn.experts.{E}.{m}.weight').copy() for m in ('w1', 'w2', 'w3')]
    neg0 = 0
    for w in wt:
        lo = (w & 15) == 8; hi = (w >> 4) == 8
        neg0 += int(lo.sum()) + int(hi.sum())
        w[lo] &= 0xf0; w[hi] &= 0x0f
    src = np.concatenate(sc + wt)
    print(f'L{L} E{E}: source {src.size} B, decoded {got.size} B, -0 nibbles canonicalised {neg0}, equal: {src.size == got.size and bool((src == got).all())}')
