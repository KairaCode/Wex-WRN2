# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Verifier step 4: the K-table-set result (item 8 and the 'combined best') rerun with the experiment's own C
walker (my build of ctxlib.c), on its split A (reproduction) and on experts/layers it never saw.

  python -I v4_kmix.py ..\\out out FEATURES [SPLIT ...]     e.g. SREL+RMAX+ROWLVL+POSC A N ALL32

Splits: A = theirs. N = train on their 24, test on my 8 new experts (8 other layers).
ALL32 = train and test on all 32 (in-sample on purpose: production trains on every record)."""
import ctypes as C
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ctxeval as ce  # noqa: E402  (read-only import of the experiment's module)


def main():
    theirs, out, names = sys.argv[1], sys.argv[2], sys.argv[3].split('+')
    d = ce.Data(theirs)
    lib = C.CDLL(os.path.join(os.path.abspath(out), 'ctxlib.dll'))        # my build of the same source
    lib.ctx_walk.restype = C.c_uint32
    lib.ctx_walk.argtypes = d.lib.ctx_walk.argtypes
    lib.ctx_radix.restype = C.c_uint32
    d.lib = lib
    for r in json.load(open(os.path.join(out, 'v1.json')))['new_rows']:
        d.slabs.append(open(os.path.join(out, 'newslabs', r['file']), 'rb').read())
        d.geom.append(d.geom[0])
        d.anchor.append(np.array(r['anchors'], np.uint8))
    ids, nctx = d.spec(names)
    t24 = list(range(24))
    splits = {'A': ce.SPLITS['A'], 'B': ce.SPLITS['B'], 'N': (t24, list(range(24, 32))),
              'ALL32': (list(range(32)), list(range(32)))}
    # baseline of today's 567 contexts on the same test sets
    c567 = np.zeros((32, 3, 567, 16), np.uint32)
    bids, _ = d.spec(ce.BASE)
    d.pool(lambda i: d._walk(i, bids, 0, counts=c567[i]), range(32))

    def stream_costs(idx, cost, K):
        sc = {i: np.zeros((d.streams, K), np.float64) for i in idx}
        d.pool(lambda i: d._walk(i, ids, 1, cost=cost, K=K, scost=sc[i]), idx)
        return sc

    res = {}
    for sp in sys.argv[4:]:
        tr, te = splits[sp]
        today = ce.bits(c567[te].sum(0, dtype=np.uint64), c567[tr].sum(0, dtype=np.uint64)) / (len(te) * d.weights)
        c = np.zeros((len(tr), 3, nctx, 16), np.uint32)
        d.pool(lambda j: d._walk(tr[j], ids, 0, counts=c[j]), range(len(tr)))
        one = np.ascontiguousarray((14.0 - np.log2(ce.quantize(c.sum(0, dtype=np.uint64))))[None], np.float32)
        base_te = sum(v.sum() for v in stream_costs(te, one, 1).values()) / (len(te) * d.weights)
        s1 = stream_costs(tr, one, 1)
        res[sp] = {'today_567_bpw': today, 'single_set_bpw': base_te,
                   'single_set_delta_vs_today_pct_raw': (base_te - today) / 4.25 * 100}
        print(sp, f'today {today:.5f}  single set {base_te:.5f} ({(base_te - today) / 4.25 * 100:+.4f})', flush=True)
        for K in (2, 4):
            allc = np.concatenate([s1[i][:, 0] for i in tr])
            cuts = np.quantile(allc, [k / K for k in range(1, K)])
            assign = {i: np.searchsorted(cuts, s1[i][:, 0]).astype(np.uint8) for i in tr}
            for it in range(8):
                ck = {i: np.zeros((K, 3, nctx, 16), np.uint32) for i in tr}
                d.pool(lambda i: d._walk(i, ids, 2, counts=ck[i], K=K, assign=assign[i]), tr)
                tot = sum(ck[i].astype(np.uint64) for i in tr)
                cost = np.ascontiguousarray(14.0 - np.log2(ce.quantize(tot)), np.float32)
                sc = stream_costs(tr, cost, K)
                assign = {i: sc[i].argmin(1).astype(np.uint8) for i in tr}
            ste = stream_costs(te, cost, K)
            payload = sum(v.min(1).sum() for v in ste.values()) / (len(te) * d.weights)
            for selbits, tag in ((np.log2(K), 'log2K'),):
                h = payload + selbits * d.streams / d.weights
                res[sp][f'K{K}'] = {'heldout_bpw': h, 'delta_vs_single_pct_raw': (h - base_te) / 4.25 * 100,
                                    'delta_vs_today_pct_raw': (h - today) / 4.25 * 100,
                                    'selector_pct_raw': selbits * d.streams / d.weights / 4.25 * 100}
            print(sp, K, f"{h:.5f}  vs single {(h - base_te) / 4.25 * 100:+.4f}  vs today {(h - today) / 4.25 * 100:+.4f}", flush=True)
    path = os.path.join(out, 'v4.json')
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev['+'.join(names)] = res
    json.dump(prev, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
