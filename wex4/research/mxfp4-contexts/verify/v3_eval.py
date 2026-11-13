# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Verifier step 3: held-out costs from the independent counts (v2_counts.py), own quantiser, own splits.

  python -I v3_eval.py ..\\out out

Splits A, B, C are the experiment's. N = train on all 24 of its experts, test on my 8 new experts from
layers it never read. N16 = train on its split-A training set (16), test on the same 8 new experts."""
import json
import os
import sys

import numpy as np

W, STREAMS, RAW = 35389440, 17280, 4.25


def quant(c, total=16384):
    """own version of the wex_nr2_quantize rule (last axis): 1 + floor share, leftovers by largest remainder,
    ties to the lower symbol"""
    c = np.asarray(c, np.int64)
    s = c.shape[-1]
    spare = total - s
    tot = c.sum(-1, keepdims=True)
    num = np.where(tot > 0, c * spare, spare)
    den = np.where(tot > 0, tot, s)
    f = 1 + num // den
    rem = num % den
    left = total - f.sum(-1)
    flat_f, flat_r, flat_l = f.reshape(-1, s), rem.reshape(-1, s), left.reshape(-1)
    for i in range(flat_f.shape[0]):               # plain loop on purpose (independent of their rank trick)
        order = sorted(range(s), key=lambda k: (-int(flat_r[i, k]), k))
        for k in order[:int(flat_l[i])]:
            flat_f[i, k] += 1
    assert (f.sum(-1) == total).all() and (f >= 1).all()
    return f


def bits(test, train, total=16384):
    return float((test.astype(np.float64) * (np.log2(total) - np.log2(quant(train, total)))).sum())


def main():
    theirs, out = sys.argv[1], sys.argv[2]
    rows = json.load(open(os.path.join(theirs, 'manifest.json')))['rows']
    new = json.load(open(os.path.join(out, 'v1.json')))['new_rows']
    allrows = rows + new
    fine = np.stack([np.load(os.path.join(out, 'counts', r['file'] + '.npz'))['fine'] for r in allrows]).astype(np.int64)
    # axes: expert, matrix, srel, prev, rmax, rowlvl, posc, sym
    AX = {'srel': 2, 'prev': 3, 'rmax': 4, 'rowlvl': 5, 'posc': 6}

    def marg(keep):
        drop = tuple(a for n, a in AX.items() if n not in keep)
        c = fine.sum(drop)
        return c.reshape(c.shape[0], 3, -1, 16)

    t24 = list(range(24))
    splits = {'A': ([i for i in t24 if i % 3 != 2], [i for i in t24 if i % 3 == 2]),
              'B': ([i for i in t24 if i % 3 != 0], [i for i in t24 if i % 3 == 0]),
              'C': ([i for i in t24 if i // 3 in (0, 2, 3, 5, 6)], [i for i in t24 if i // 3 in (1, 4, 7)]),
              'N': (t24, list(range(24, 32))),
              'N16': ([i for i in t24 if i % 3 != 2], list(range(24, 32)))}
    res = {}

    # ---- baseline vs real payloads, production model ----
    base = marg(('srel', 'prev', 'rmax'))
    prod = open(os.path.join(theirs, 'prod_model.wnm2'), 'rb').read()
    pf = np.diff(np.frombuffer(prod, '<u2', 3 * 567 * 17, 32).reshape(3, 567, 17).astype(np.int64), axis=-1)
    assert (pf > 0).all() and (pf.sum(-1) == 16384).all()
    pcost = 14.0 - np.log2(pf)
    ideal = (base * pcost).sum((1, 2, 3)) / 8
    real = np.array([r['weight_payload_bytes'] for r in allrows], np.float64)
    fl = (real - ideal) / STREAMS
    res['validate'] = {'ideal_bpw_prod_model_24': float(ideal[:24].sum() * 8 / (24 * W)),
                       'real_bpw_24': float(real[:24].sum() * 8 / (24 * W)),
                       'flush_bytes_per_stream_24': float(fl[:24].mean()), 'flush_min_max_24': [float(fl[:24].min()), float(fl[:24].max())],
                       'ideal_bpw_prod_model_new8': float(ideal[24:].sum() * 8 / (8 * W)),
                       'real_bpw_new8': float(real[24:].sum() * 8 / (8 * W)),
                       'flush_bytes_per_stream_new8': float(fl[24:].mean()),
                       'max_rel_err_ideal_plus_3p5': float(np.abs(ideal + 3.5 * STREAMS - real).max() / real.min()),
                       'max_rel_err_ideal_plus_3p57': float(np.abs(ideal + 3.57 * STREAMS - real).max() / real.min())}
    tot = base.sum((0, 1, 2)).astype(np.float64)
    p = tot[:16] if False else base[:24].sum((0, 1, 2)).astype(np.float64)
    p = p / p.sum()
    res['validate']['order0_bpw_24'] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    res['validate']['nibble8_count_all32'] = int(base[..., 8].sum())
    print(json.dumps(res['validate'], indent=1), flush=True)

    # ---- variants ----
    variants = {'base(567)': ('srel', 'prev', 'rmax'), 'srel+rmax(63)': ('srel', 'rmax'), 'srel(7)': ('srel',),
                'srel+prev(63)': ('srel', 'prev'), 'prev+rmax(81)': ('prev', 'rmax'),
                'base+posc(1701)': ('srel', 'prev', 'rmax', 'posc'), 'base+rowlvl(2835)': ('srel', 'prev', 'rmax', 'rowlvl'),
                'base+rowlvl+posc(8505)': ('srel', 'prev', 'rmax', 'rowlvl', 'posc'),
                'srel+rmax+rowlvl+posc(945)': ('srel', 'rmax', 'rowlvl', 'posc'), 'srel+rowlvl(35)': ('srel', 'rowlvl')}
    res['variants'] = {}
    bb = {}
    for name, keep in variants.items():
        c = marg(keep)
        r = {}
        for sp, (tr, te) in splits.items():
            h = bits(c[te].sum(0), c[tr].sum(0)) / (len(te) * W)
            ins = bits(c[tr].sum(0), c[tr].sum(0)) / (len(tr) * W)
            if name.startswith('base('):
                bb[sp] = h
            r[sp] = {'heldout_bpw': h, 'insample_bpw': ins, 'delta_pct_raw': (h - bb[sp]) / RAW * 100}
        res['variants'][name] = r
        print(f'{name:28s}', ' '.join(f"{sp} {r[sp]['heldout_bpw']:.5f} ({r[sp]['delta_pct_raw']:+.4f})" for sp in splits), flush=True)

    # ---- sign: coin? and the 8-magnitude forms (F model = srel only, 12 bit, raw sign) ----
    res['sign'] = {}
    for name, keep in (('base(567)', ('srel', 'prev', 'rmax')), ('srel+rmax(63)', ('srel', 'rmax')), ('srel(7)', ('srel',))):
        c = marg(keep)
        mag = c[..., :8] + c[..., 8:]
        nz = mag[..., 1:].sum((1, 2, 3)).astype(np.float64)
        pairs = np.stack([c[..., 1:8], c[..., 9:16]], -1)
        r = {}
        for sp, (tr, te) in splits.items():
            n = len(te) * W
            r[sp] = {'sign_bits_per_nonzero_modelled': bits(pairs[te].sum(0), pairs[tr].sum(0)) / nz[te].sum(),
                     'minus_share_test': float(c[te][..., 9:16].sum() / nz[te].sum())}
            for nb in (14, 12, 10):
                h = (bits(mag[te].sum(0), mag[tr].sum(0), 1 << nb) + nz[te].sum()) / n
                r[sp][f'mag8_raw_sign_{nb}bit_bpw'] = h
                r[sp][f'mag8_raw_sign_{nb}bit_delta_pct_raw'] = (h - bb[sp]) / RAW * 100
            r[sp]['joint16_12bit_delta_pct_raw'] = (bits(c[te].sum(0), c[tr].sum(0), 4096) / n - bb[sp]) / RAW * 100
        res['sign'][name] = r
        print('sign/mag8', name, json.dumps({sp: {k: round(v, 6) for k, v in r[sp].items()} for sp in ('A', 'B', 'C', 'N')}), flush=True)

    # ---- per-layer cost of the global model on the new layers ----
    c = base
    f = c[:24].sum(0)
    res['new_layer_bpw_under_24_model'] = {allrows[i]['file']: bits(c[i], f) / W for i in range(24, 32)}

    # ---- K table sets per stream, srel-only contexts, independent Lloyd on per-stream histograms ----
    H = np.stack([np.load(os.path.join(out, 'counts', r['file'] + '.npz'))['shist'] for r in allrows]).astype(np.float32)
    H = H.reshape(32, STREAMS, 112)
    mat = np.repeat(np.arange(3), STREAMS // 3)
    res['kmix_srel'] = {}

    def tables(Hs, assign, K):
        """Hs [n][STREAMS][112], assign [n][STREAMS] -> cost [K][3][112]"""
        cnt = np.zeros((K, 3, 112), np.int64)
        for k in range(K):
            for m in range(3):
                sel = (assign == k) & (mat[None, :] == m)
                cnt[k, m] = Hs[sel].sum(0).astype(np.int64)
        return (14.0 - np.log2(quant(cnt.reshape(K, 3, 7, 16)))).reshape(K, 3, 112).astype(np.float32)

    def costs(Hs, cost):
        out_ = np.empty(Hs.shape[:2] + (cost.shape[0],), np.float64)
        for m in range(3):
            out_[:, mat == m] = Hs[:, mat == m] @ cost[:, m].T
        return out_

    rng = np.random.default_rng(1)
    for sp in ('A', 'B', 'N'):
        tr, te = splits[sp]
        Htr, Hte = H[tr], H[te]
        one = tables(Htr, np.zeros((len(tr), STREAMS), np.int64), 1)
        base_te = costs(Hte, one).sum() / (len(te) * W)
        s1 = costs(Htr, one)[..., 0]
        r = {'single_model_bpw': base_te, 'single_model_delta_vs_567_pct_raw': (base_te - bb[sp]) / RAW * 100}
        for K in (2, 4):
            best = None
            for start in ('quantile', 'random', 'random2'):
                if start == 'quantile':
                    cuts = np.quantile(s1, [k / K for k in range(1, K)])
                    assign = np.searchsorted(cuts, s1)
                else:
                    assign = rng.integers(0, K, s1.shape)
                for it in range(12):
                    cost = tables(Htr, assign, K)
                    assign = costs(Htr, cost).argmin(-1)
                cost = tables(Htr, assign, K)
                h = costs(Hte, cost).min(-1).sum() / (len(te) * W) + np.log2(K) * STREAMS / W
                r[f'K{K}_{start}'] = {'heldout_bpw': h, 'delta_vs_single_pct_raw': (h - base_te) / RAW * 100,
                                      'delta_vs_567_pct_raw': (h - bb[sp]) / RAW * 100}
                print(sp, K, start, f"{base_te:.5f} -> {h:.5f}  vs single {r[f'K{K}_{start}']['delta_vs_single_pct_raw']:+.4f}"
                      f"  vs today {r[f'K{K}_{start}']['delta_vs_567_pct_raw']:+.4f}", flush=True)
        res['kmix_srel'][sp] = r
    json.dump(res, open(os.path.join(out, 'v3.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
