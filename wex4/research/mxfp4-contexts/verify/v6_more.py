# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Verifier step 6: independent numpy check of four 'no gain' rows that step 2 did not cover:
nibble above (cross-stream), magnitude two back, exact position, two-pass block max.

  python -I -B v6_more.py ..\\out out"""
import concurrent.futures as cf
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v2_counts import ENTRY  # noqa: E402
from v3_eval import bits, W, RAW  # noqa: E402

FEATS = {'above9': 9, 'pm2': 9, 'pos32': 32, 'bmax3': 3}


def walk(slab, entry_hex, anchors):
    e = ENTRY.unpack(bytes.fromhex(entry_hex))
    out = {k: np.zeros((3, 567 * r * 16), np.uint32) for k, r in FEATS.items()}
    side = np.zeros((3, 7, 3), np.int64)
    for m in range(3):
        R, B = e[4 + 2 * m], e[5 + 2 * m]
        S = np.frombuffer(slab, np.uint8, R * B, e[16 + m]).astype(np.int32)
        Wb = np.frombuffer(slab, np.uint8, R * B * 16, e[19 + m])
        x = np.empty(R * B * 32, np.uint8)
        x[0::2] = Wb & 15
        x[1::2] = Wb >> 4
        x = x.reshape(-1, 32).astype(np.int32)
        a = x & 7
        prev = np.full(a.shape, 8, np.int32)
        prev[:, 1:] = a[:, :-1]
        pm2 = np.full(a.shape, 8, np.int32)
        pm2[:, 2:] = a[:, :-2]
        rmax = np.full(a.shape, 8, np.int32)
        rmax[:, 1:] = np.maximum.accumulate(a, axis=1)[:, :-1]
        up = np.full((R, B * 32), 8, np.int32)
        up[1:] = a.reshape(R, B * 32)[:-1]
        srel = np.clip(S - anchors[m], -3, 3) + 3
        bm = a.max(1)
        bmax3 = np.where(bm <= 5, 0, bm - 5)
        base = (srel[:, None] * 9 + prev) * 9 + rmax
        f = {'above9': up.reshape(-1, 32), 'pm2': pm2, 'pos32': np.broadcast_to(np.arange(32, dtype=np.int32), a.shape),
             'bmax3': np.broadcast_to(bmax3[:, None], a.shape)}
        for k, r in FEATS.items():
            out[k][m] = np.bincount(((base * r + f[k]) * 16 + x).ravel(), minlength=567 * r * 16)
        side[m] = np.bincount(srel * 3 + bmax3, minlength=21).reshape(7, 3)
    return out, side


def main():
    theirs, o = sys.argv[1], sys.argv[2]
    rows = json.load(open(os.path.join(theirs, 'manifest.json')))['rows']

    def one(r):
        return walk(open(os.path.join(theirs, 'slabs', r['file']), 'rb').read(), r['entry_hex'], r['anchors'])

    with cf.ThreadPoolExecutor(4) as ex:
        got = list(ex.map(one, rows))
    t24 = list(range(24))
    splits = {'A': ([i for i in t24 if i % 3 != 2], [i for i in t24 if i % 3 == 2]),
              'B': ([i for i in t24 if i % 3 != 0], [i for i in t24 if i % 3 == 0]),
              'C': ([i for i in t24 if i // 3 in (0, 2, 3, 5, 6)], [i for i in t24 if i // 3 in (1, 4, 7)])}
    side = np.stack([g[1] for g in got])
    res = {}
    for k, r in FEATS.items():
        c = np.stack([g[0][k] for g in got]).astype(np.int64).reshape(24, 3, 567, r, 16)
        b = c.sum(3)
        res[k] = {}
        for sp, (tr, te) in splits.items():
            n = len(te) * W
            base = bits(b[te].sum(0), b[tr].sum(0)) / n
            h = bits(c[te].sum(0).reshape(3, -1, 16), c[tr].sum(0).reshape(3, -1, 16)) / n
            extra = bits(side[te].sum(0), side[tr].sum(0)) / n if k == 'bmax3' else 0.0
            res[k][sp] = {'base_bpw': base, 'heldout_bpw': h + extra, 'side_bpw': extra,
                          'delta_pct_raw': (h + extra - base) / RAW * 100}
        print(f'{k:8s}', ' '.join(f"{sp} {res[k][sp]['heldout_bpw']:.5f} ({res[k][sp]['delta_pct_raw']:+.4f})" for sp in splits),
              f"side {res[k]['A']['side_bpw']:.4f}", flush=True)
    json.dump(res, open(os.path.join(o, 'v6.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
