# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Verifier step 2: an independent (numpy only, no ctxlib) context walk over the coded slabs.

  python -I v2_counts.py ..\\out out

Per expert it writes out/counts/NAME.npz with
  fine   uint32 [3][7 srel][9 prev][9 rmax][5 rowlvl][3 posclass][16 symbol]
  shist  uint16 [17280 streams][7 srel][16 symbol]   (for the per-stream table-set check)
All other tested models are marginals of `fine`."""
import concurrent.futures as cf
import json
import os
import struct
import sys

import numpy as np

ENTRY = struct.Struct('<QIII6H6H3I3II')


def rule_g_rows(S):
    """rule g per row of S [R][B]: lower value of the densest adjacent pair, first maximum wins"""
    R = S.shape[0]
    h = np.bincount((np.arange(R, dtype=np.int64)[:, None] * 256 + S).ravel(), minlength=R * 256).reshape(R, 256)
    return np.argmax(h[:, :-1] + h[:, 1:], axis=1)


def walk(slab, entry_hex, anchors):
    e = ENTRY.unpack(bytes.fromhex(entry_hex))
    fine = np.zeros((3, 7 * 9 * 9 * 5 * 3 * 16), np.uint32)
    shist = []
    for m in range(3):
        R, B = e[4 + 2 * m], e[5 + 2 * m]
        so, wo = e[16 + m], e[19 + m]
        S = np.frombuffer(slab, np.uint8, R * B, so).astype(np.int32).reshape(R, B)
        Wb = np.frombuffer(slab, np.uint8, R * B * 16, wo)
        x = np.empty(R * B * 32, np.uint8)
        x[0::2] = Wb & 15
        x[1::2] = Wb >> 4
        a = (x & 7).reshape(-1, 32)
        prev = np.full(a.shape, 8, np.uint8)
        prev[:, 1:] = a[:, :-1]
        rmax = np.full(a.shape, 8, np.uint8)
        rmax[:, 1:] = np.maximum.accumulate(a, axis=1)[:, :-1]
        assert int(np.bincount(S.ravel(), minlength=256)[:-1].__add__(np.bincount(S.ravel(), minlength=256)[1:]).argmax()) == anchors[m], \
            'record anchor is not rule g of the coded scales'
        srel = (np.clip(S - anchors[m], -3, 3) + 3).ravel()                                # per block
        rowlvl = np.repeat(np.clip(rule_g_rows(S) - anchors[m], -2, 2) + 2, B)             # per block
        posc = np.array([0] + [1] * 7 + [2] * 24, np.int32)
        blk = ((srel * 81).astype(np.int32))[:, None]
        ctx = ((blk + prev.astype(np.int32) * 9 + rmax) * 5 + rowlvl[:, None].astype(np.int32)) * 3 + posc[None, :]
        idx = (ctx * 16 + x.reshape(-1, 32)).ravel()
        fine[m] = np.bincount(idx, minlength=fine.shape[1])
        streams = R * B * 32 // 2048
        sid = (np.arange(R * B, dtype=np.int32) // 64)[:, None]                            # 64 blocks per stream
        sidx = ((sid * 7 + srel[:, None].astype(np.int32)) * 16 + x.reshape(-1, 32)).ravel()
        shist.append(np.bincount(sidx, minlength=streams * 112).astype(np.uint16).reshape(streams, 7, 16))
    return fine.reshape(3, 7, 9, 9, 5, 3, 16), np.concatenate(shist)


def main():
    theirs, out = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(out, 'counts'), exist_ok=True)
    jobs = [(os.path.join(theirs, 'slabs', r['file']), r) for r in json.load(open(os.path.join(theirs, 'manifest.json')))['rows']]
    jobs += [(os.path.join(out, 'newslabs', r['file']), r) for r in json.load(open(os.path.join(out, 'v1.json')))['new_rows']]

    def one(job):
        path, r = job
        fine, shist = walk(open(path, 'rb').read(), r['entry_hex'], r['anchors'])
        assert int(fine.sum()) == 35389440 and int(shist.sum()) == 35389440
        np.savez(os.path.join(out, 'counts', r['file'] + '.npz'), fine=fine, shist=shist)
        return r['file']

    with cf.ThreadPoolExecutor(4) as ex:
        for name in ex.map(one, jobs):
            print(name, flush=True)


if __name__ == '__main__':
    main()
