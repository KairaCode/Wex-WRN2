# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Step 2: held-out cost of candidate context models for the WNR3-CZ weight streams.

  python -I ctxeval.py out STAGE        STAGE = validate | contexts | combo | kmix | layers | precision | decbench

Data: the 24 coded slabs written by extract.py (8 layers x 3 experts). Every number is the cost of the
weight nibbles in bits/weight with 14-bit CDFs quantised by the wex_nr2_quantize rule, trained on one set of
experts and measured on another:
  split A: train = experts 0,1 of each layer (16), test = expert 2 (8)
  split B: train = experts 1,2 (16), test = expert 0 (8)
  split C: train = layers 0,11,16,28,33 (15 experts), test = layers 5,22,39 (9): no layer shared
The cost is the ideal one (sum of -log2(f/16384)); the rANS flush per stream is the same for every variant
and is measured once in `validate` against the real record sizes. Results are merged into results.json."""
import concurrent.futures as cf
import ctypes as C
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = struct.Struct('<QIII6H6H3I3II')
TOTAL, RAW_BPW, THREADS, STREAM = 16384, 4.25, 8, 2048
FEATURES = ['SREL', 'PM1', 'RMAX', 'PM2', 'POS', 'POSC', 'PSIGN', 'ABOVE', 'ABOVE3', 'ASIGN', 'DNEXT', 'DLEFT',
            'DUP', 'DDOWN', 'LOCMAX', 'LOCAVG', 'SROW', 'SCOL', 'ZC', 'ZC4', 'BIG', 'BMAX3', 'ZBLK', 'PBMAX',
            'PM1X', 'PAR', 'REMZ', 'REMP', 'RMAXC', 'PM1C', 'SREL5', 'SUM3', 'ROWLVL', 'COLLVL']
CROSS = {'ABOVE', 'ABOVE3', 'ASIGN'}          # need a nibble of another stream
SIDE = {'BMAX3', 'ZBLK', 'REMZ'}              # need a per-block side symbol
PER_SYMBOL = {'PM2', 'POS', 'POSC', 'PSIGN', 'ZC', 'ZC4', 'BIG', 'PM1X', 'PAR', 'REMP', 'RMAXC', 'PM1C', 'SUM3',
              'PBMAX'}   # in-stream state beyond today's previous magnitude and running max
BASE = ['SREL', 'PM1', 'RMAX']
SPLITS = {'A': ([i for i in range(24) if i % 3 != 2], [i for i in range(24) if i % 3 == 2]),
          'B': ([i for i in range(24) if i % 3 != 0], [i for i in range(24) if i % 3 == 0]),
          'C': ([i for i in range(24) if i // 3 in (0, 2, 3, 5, 6)], [i for i in range(24) if i // 3 in (1, 4, 7)])}


class Data:
    def __init__(self, out):
        self.out = out
        self.lib = C.CDLL(os.path.join(os.path.abspath(out), 'ctxlib.dll'))
        self.lib.ctx_walk.restype = C.c_uint32
        self.lib.ctx_walk.argtypes = [C.c_void_p] * 4 + [C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_int,
                                                         C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
        self.lib.ctx_radix.restype = C.c_uint32
        assert self.lib.ctx_features() == len(FEATURES)
        self.rows = json.load(open(os.path.join(out, 'manifest.json')))['rows']
        self.slabs, self.geom, self.anchor, self.entries = [], [], [], []
        for r in self.rows:
            e = ENTRY.unpack(bytes.fromhex(r['entry_hex']))
            ss, ws = e[4:10], e[10:16]
            g = []
            for m in range(3):
                assert ws[2 * m + 1] == ss[2 * m + 1] * 16
                g += [e[16 + m], e[19 + m], ss[2 * m], ss[2 * m + 1]]
            self.geom.append(np.array(g, np.uint32))
            self.anchor.append(np.array(r['anchors'], np.uint8))
            self.slabs.append(open(os.path.join(out, 'slabs', r['file']), 'rb').read())
            self.entries.append(bytes.fromhex(r['entry_hex']))
        g = self.geom[0]
        self.weights = sum(int(g[4 * m + 2]) * int(g[4 * m + 3]) * 32 for m in range(3))   # per expert
        self.blocks = self.weights // 32
        self.streams = sum((int(g[4 * m + 2]) * int(g[4 * m + 3]) * 32 + STREAM - 1) // STREAM for m in range(3))
        assert all((x == g).all() for x in self.geom), 'all experts must share one geometry'

    def spec(self, names):
        ids = np.array([FEATURES.index(n) for n in names], np.int32)
        nctx = int(np.prod([self.lib.ctx_radix(int(i)) for i in ids]))
        return ids, nctx

    def _walk(self, i, ids, mode, counts=None, cost=None, K=0, assign=None, scost=None, side=None, kind=0):
        p = lambda a: a.ctypes.data if a is not None else None
        n = self.lib.ctx_walk(self.slabs[i], p(self.geom[i]), p(self.anchor[i]), p(ids), len(ids), mode,
                              p(counts), p(cost), K, p(assign), p(scost), p(side), kind)
        assert n, 'bad spec'

    def pool(self, fn, items=range(24)):
        with cf.ThreadPoolExecutor(THREADS) as ex:   # ctypes releases the GIL
            return list(ex.map(fn, items))

    def counts(self, names):
        """-> uint32 [24][3][nctx][16]"""
        ids, nctx = self.spec(names)
        out = np.zeros((24, 3, nctx, 16), np.uint32)
        self.pool(lambda i: self._walk(i, ids, 0, counts=out[i]))
        return out

    def side(self, kind, symbols):
        """per-block side symbol counts -> uint32 [24][3][7][symbols]"""
        ids, _ = self.spec(BASE)
        out = np.zeros((24, 3, 7, 64), np.uint32)
        self.pool(lambda i: self._walk(i, ids, 3, side=out[i], kind=kind))
        assert not out[..., symbols:].any()
        return out[..., :symbols]


def quantize(c, total=TOTAL):
    """wex_nr2_quantize on the last axis: every symbol >= 1, rest in proportion, remainders largest first
    (ties: lower symbol). -> integer frequencies summing to 16384."""
    c = c.astype(np.int64)
    s = c.shape[-1]
    spare = total - s
    tot = c.sum(-1, keepdims=True)
    scaled = np.where(tot > 0, c * spare, spare)
    div = np.where(tot > 0, tot, s)
    f = 1 + scaled // div
    rem = scaled % div
    order = np.argsort(-rem, axis=-1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(s), order.shape), axis=-1)
    f += rank < (total - f.sum(-1, keepdims=True))
    assert (f.sum(-1) == total).all() and (f >= 1).all()
    return f


def bits(test, train, total=TOTAL):
    """ideal bits of the `test` counts under the CDFs (14-bit unless `total` says otherwise) built from `train`"""
    return float((test.astype(np.float64) * (np.log2(total) - np.log2(quantize(train, total)))).sum())


def save(key, value):
    path = os.path.join(HERE, 'results.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    res[key] = value
    json.dump(res, open(path, 'w'), indent=1)


def split_eval(d, c, extra=None):
    """c = per-expert counts [24][...]; -> per split held-out / in-sample bits per weight"""
    r = {}
    for name, (tr, te) in SPLITS.items():
        ctr, cte = c[tr].sum(0, dtype=np.uint64), c[te].sum(0, dtype=np.uint64)
        r[name] = {'heldout_bpw': bits(cte, ctr) / (len(te) * d.weights),
                   'insample_bpw': bits(ctr, ctr) / (len(tr) * d.weights)}
        if extra:
            for k in r[name]:
                r[name][k] += extra[name]
    return r


def stage_validate(d):
    """Harness == codec: counts vs wex_nr3_count_expert, quantiser vs wex_nr3_build_model, ideal cost under the
    production model vs the real weight payload of the production records."""
    tools = C.CDLL(os.path.join(os.path.abspath(d.out), 'wex_nr3_tools.dll'))
    P = C.c_void_p
    tools.wex_nr3_count_expert.argtypes = [P, P, P, P]
    tools.wex_nr3_build_model.argtypes = [P, P, P, P]
    c = d.counts(BASE)
    wc_all, sc_all = np.zeros(3 * 567 * 16, np.uint64), np.zeros(3 * 256, np.uint64)
    for i in range(24):
        wc, sc = np.zeros(3 * 567 * 16, np.uint64), np.zeros(3 * 256, np.uint64)
        assert tools.wex_nr3_count_expert(d.entries[i], d.slabs[i], wc.ctypes.data, sc.ctypes.data) == 0
        assert (wc.reshape(3, 567, 16) == c[i]).all(), 'context walk differs from the codec'
        wc_all += wc
        sc_all += sc
    blob = C.create_string_buffer(59408)
    assert tools.wex_nr3_build_model(wc_all.ctypes.data, sc_all.ctypes.data, blob, None) == 0
    cdf = np.frombuffer(blob.raw, '<u2', 3 * 567 * 17, 32).reshape(3, 567, 17).astype(np.int64)
    assert (np.diff(cdf, axis=-1) == quantize(c.sum(0, dtype=np.uint64))).all(), 'quantiser differs from the codec'
    prod = open(os.path.join(d.out, 'prod_model.wnm2'), 'rb').read()
    pf = np.diff(np.frombuffer(prod, '<u2', 3 * 567 * 17, 32).reshape(3, 567, 17).astype(np.int64), axis=-1)
    pcost = 14.0 - np.log2(pf)
    rows, ideal_sum, real_sum = [], 0.0, 0
    for i, r in enumerate(d.rows):
        ideal = float((c[i] * pcost).sum()) / 8
        real = r['weight_payload_bytes']
        ideal_sum += ideal
        real_sum += real
        rows.append({'layer': r['layer'], 'expert': r['expert'], 'real_weight_payload_bytes': real,
                     'ideal_bytes_prod_model': ideal, 'flush_bytes_per_stream': (real - ideal) / d.streams})
    flush = (real_sum - ideal_sum) / (24 * d.streams)
    est = ideal_sum + 24 * d.streams * flush
    res = {'weights_per_expert': d.weights, 'weight_streams_per_expert': d.streams,
           'raw_slab_bytes': d.rows[0]['raw_bytes'],
           'mean_record_total_bytes': sum(r['record_total_bytes'] for r in d.rows) / 24,
           'mean_record_ratio_of_raw': sum(r['record_total_bytes'] for r in d.rows) / 24 / d.rows[0]['raw_bytes'],
           'real_bpw_weights_payload': real_sum * 8 / (24 * d.weights),
           'ideal_bpw_prod_model': ideal_sum * 8 / (24 * d.weights),
           'flush_bytes_per_stream': flush, 'flush_bpw': flush * 8 * d.streams / d.weights,
           'max_abs_rel_error_with_flush_3p5': max(abs(x['ideal_bytes_prod_model'] + 3.5 * d.streams -
                                                       x['real_weight_payload_bytes']) /
                                                   x['real_weight_payload_bytes'] for x in rows),
           'flush_per_stream_min_max': [min(x['flush_bytes_per_stream'] for x in rows),
                                        max(x['flush_bytes_per_stream'] for x in rows)],
           'ideal_bpw_own_model_all24_insample': bits(c.sum(0, dtype=np.uint64), c.sum(0, dtype=np.uint64)) / (24 * d.weights),
           'baseline': split_eval(d, c), 'rows': rows, 'checks': 'counts == codec, quantiser == codec: passed'}
    # order-0 and the nibble statistics of the coded domain
    tot = c.sum((0, 1, 2), dtype=np.uint64).astype(np.float64)
    p = tot / tot.sum()
    res['order0_entropy_bpw'] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    res['nibble_share'] = [float(x) for x in p]
    save('validate', res)
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))


VARIANTS = [
    # name, features, task item
    ('base', BASE, 0),
    ('above9', BASE + ['ABOVE'], 1), ('above3', BASE + ['ABOVE3'], 1), ('above_sign', BASE + ['ASIGN'], 1),
    ('pm2', BASE + ['PM2'], 2), ('sum3', BASE + ['SUM3'], 2),
    ('posclass', BASE + ['POSC'], 3), ('pos32', BASE + ['POS'], 3), ('parity', BASE + ['PAR'], 3),
    ('prev_sign', BASE + ['PSIGN'], 4),
    ('scale_next', BASE + ['DNEXT'], 5), ('scale_up', BASE + ['DUP'], 5), ('scale_left', BASE + ['DLEFT'], 5),
    ('scale_down', BASE + ['DDOWN'], 5), ('scale_next_up', BASE + ['DNEXT', 'DUP'], 5),
    ('scale_locmax', BASE + ['LOCMAX'], 5), ('scale_locavg', BASE + ['LOCAVG'], 5),
    ('scale_row_anchor', BASE + ['SROW'], 5), ('scale_col_anchor', BASE + ['SCOL'], 5),
    ('row_level', BASE + ['ROWLVL'], 5), ('col_level', BASE + ['COLLVL'], 5),
    ('row_anchor_instead', ['SROW', 'PM1', 'RMAX'], 5), ('srel5', ['SREL5', 'PM1', 'RMAX'], 5),
    ('zeros_instead_of_max', ['SREL', 'PM1', 'ZC'], 6), ('zeros_and_max', BASE + ['ZC'], 6),
    ('zeros4_and_max', BASE + ['ZC4'], 6), ('bigcount_and_max', BASE + ['BIG'], 6),
    ('side_blockmax', BASE + ['BMAX3'], 7), ('side_zero_bucket', BASE + ['ZBLK'], 7),
    ('side_zero_exact', ['SREL', 'PM1', 'RMAXC', 'REMZ', 'REMP'], 7),
    ('prev_block_max', BASE + ['PBMAX'], 10), ('pm1_no_block_reset', ['SREL', 'PM1X', 'RMAX'], 10),
]
SIDE_KIND = {'BMAX3': (0, 3), 'REMZ': (1, 33), 'ZBLK': (2, 6)}   # side feature -> (ctx_walk kind, symbols)


def cost_class(names):
    c = []
    if set(names) & CROSS:
        c.append('CROSS-STREAM (row above lives in an earlier stream: forces a stream order)')
    if set(names) & SIDE:
        c.append('two-pass (side symbol per block, extra symbol stream)')
    extra = [n for n in names if n in PER_SYMBOL]
    if extra:
        c.append('extra in-stream state: ' + '+'.join(extra))
    blockf = [n for n in names if n not in PER_SYMBOL and n not in CROSS and n not in SIDE and n not in BASE]
    if blockf:
        c.append('per-block context base from decoded scales: ' + '+'.join(blockf))
    if not c and len(names) < 3:
        return 'simpler than today: ' + ('context fixed per block' if names == ['SREL'] else 'fewer contexts')
    return '; '.join(c) or 'as today'


def run_variant(d, name, names):
    _, nctx = d.spec(names)
    c = d.counts(names)
    extra = None
    side_bpw = None
    for feature in set(names) & set(SIDE_KIND):   # the side symbol is paid for: order-0 per srel, 14-bit CDF
        kind, symbols = SIDE_KIND[feature]
        s = d.side(kind, symbols)
        extra, side_bpw = {}, {}
        for sp, (tr, te) in SPLITS.items():
            side_bpw[sp] = bits(s[te].sum(0, dtype=np.uint64), s[tr].sum(0, dtype=np.uint64)) / (len(te) * d.weights)
            extra[sp] = side_bpw[sp]
    r = {'features': names, 'contexts_per_matrix': nctx, 'model_bytes': 3 * nctx * 34,
         'used_contexts': int((c.sum((0, 3), dtype=np.uint64) > 0).sum(1).max()),
         'decode_cost': cost_class(names), 'splits': split_eval(d, c, extra)}
    if side_bpw:
        r['side_bpw'] = side_bpw
    return r, c


def with_delta(res, base):
    for sp in SPLITS:
        h, b = res['splits'][sp]['heldout_bpw'], base['splits'][sp]['heldout_bpw']
        res['splits'][sp]['delta_pct_of_raw'] = (h - b) / RAW_BPW * 100
    res['delta_pct_of_raw_mean_AB'] = (res['splits']['A']['delta_pct_of_raw'] + res['splits']['B']['delta_pct_of_raw']) / 2
    return res


def stage_contexts(d):
    out, base = {}, None
    for name, names, item in VARIANTS:
        r, c = run_variant(d, name, names)
        r['item'] = item
        if name == 'base':
            base = r
            out.update(sign_plane(d, c))
        if name == 'prev_sign':
            out.update(sign_context(d, c, 'sign_given_prev_sign', 3))
        if name == 'above_sign':
            out.update(sign_context(d, c, 'sign_given_above_sign', 3))
        out[name] = r
        for k in out:
            if 'delta_pct_of_raw_mean_AB' not in out[k]:
                with_delta(out[k], base)
        print(f"{name:22s} ctx {r['contexts_per_matrix']:6d}  A {r['splits']['A']['heldout_bpw']:.5f} "
              f"B {r['splits']['B']['heldout_bpw']:.5f} C {r['splits']['C']['heldout_bpw']:.5f}  "
              f"d%raw A {r['splits']['A']['delta_pct_of_raw']:+.4f} B {r['splits']['B']['delta_pct_of_raw']:+.4f} "
              f"C {r['splits']['C']['delta_pct_of_raw']:+.4f}", flush=True)
    save('contexts', out)
    for k in ('sign_plane_raw_bits', 'sign_plane_modelled', 'sign_given_prev_sign', 'sign_given_above_sign'):
        print(k, json.dumps(out[k]['splits']), out[k].get('note', ''))


def sign_plane(d, c):
    """(4) magnitudes as an 8-symbol alphabet + sign as its own plane. c = base counts [24][3][567][16]."""
    mag = c[..., :8].astype(np.uint64) + c[..., 8:]
    nonzero = mag[..., 1:].sum((1, 2, 3)).astype(np.float64)                 # per expert
    pairs = np.stack([c[..., 1:8], c[..., 9:16]], -1)                        # [24][3][567][7][2] (+, -)
    res = {}
    for key in ('sign_plane_raw_bits', 'sign_plane_modelled'):
        sp_res = {}
        for sp, (tr, te) in SPLITS.items():
            m = bits(mag[te].sum(0), mag[tr].sum(0))
            s = nonzero[te].sum() if key.endswith('raw_bits') else \
                bits(pairs[te].sum(0, dtype=np.uint64), pairs[tr].sum(0, dtype=np.uint64))
            sp_res[sp] = {'heldout_bpw': (m + s) / (len(te) * d.weights), 'magnitude_bpw': m / (len(te) * d.weights),
                          'sign_bits_per_nonzero': s / nonzero[te].sum(), 'insample_bpw': None}
        res[key] = {'features': BASE, 'item': 4, 'contexts_per_matrix': 567, 'splits': sp_res,
                    'model_bytes': 3 * 567 * 18 + (0 if key.endswith('raw_bits') else 3 * 567 * 7 * 2),
                    'decode_cost': '8-symbol rANS + one raw bit per nonzero weight' if key.endswith('raw_bits')
                    else '8-symbol rANS + one binary rANS symbol per nonzero weight (context = today x magnitude)'}
    return res


def sign_context(d, c, key, radix):
    """sign coded as a binary symbol with context (today's 567 x extra feature x magnitude); magnitudes as in
    sign_plane (8 symbols, today's 567 contexts). c = counts of BASE + [feature] = [24][3][567*radix][16]."""
    c5 = c.reshape(24, 3, 567, radix, 16)
    mag = (c5[..., :8].astype(np.uint64) + c5[..., 8:]).sum(3)
    nonzero = mag[..., 1:].sum((1, 2, 3)).astype(np.float64)
    pairs = np.stack([c5[..., 1:8], c5[..., 9:16]], -1)
    sp_res = {}
    for sp, (tr, te) in SPLITS.items():
        m = bits(mag[te].sum(0), mag[tr].sum(0))
        s = bits(pairs[te].sum(0, dtype=np.uint64), pairs[tr].sum(0, dtype=np.uint64))
        sp_res[sp] = {'heldout_bpw': (m + s) / (len(te) * d.weights), 'sign_bits_per_nonzero': s / nonzero[te].sum(),
                      'insample_bpw': None}
    return {key: {'features': BASE, 'item': 4, 'contexts_per_matrix': 567, 'splits': sp_res,
                  'model_bytes': 3 * 567 * 18 + 3 * 567 * radix * 7 * 2,
                  'decode_cost': 'binary sign symbol with its own context'
                  + (' (CROSS-STREAM)' if 'above' in key else '')}}


def stage_combo(d):
    """Combinations and ablations named on the command line (merged into results.json 'combo')."""
    base = json.load(open(os.path.join(HERE, 'results.json')))['contexts']['base']
    out = {}
    for arg in sys.argv[3:]:   # A+B = today's three features plus A, B; =A+B = exactly A, B
        names = arg[1:].split('+') if arg[0] == '=' else BASE + arg.split('+')
        name = arg
        r, _ = run_variant(d, name, names)
        r['item'] = 'combo'
        out[name] = with_delta(r, base)
        print(f"{name:34s} ctx {r['contexts_per_matrix']:6d} used {r['used_contexts']:6d}  "
              f"d%raw A {r['splits']['A']['delta_pct_of_raw']:+.4f} B {r['splits']['B']['delta_pct_of_raw']:+.4f} "
              f"C {r['splits']['C']['delta_pct_of_raw']:+.4f}  in-sample A "
              f"{(r['splits']['A']['insample_bpw'] - base['splits']['A']['insample_bpw']) / RAW_BPW * 100:+.4f}", flush=True)
    path = os.path.join(HERE, 'results.json')
    prev = json.load(open(path)).get('combo', {})
    prev.update(out)
    save('combo', prev)


def stage_kmix(d):
    """(8) K table sets, each 2048-nibble stream picks the cheapest (log2 K selector bits per stream).
    Lloyd iterations on the training experts only; the test streams then choose among the frozen sets."""
    names = sys.argv[3].split('+') if len(sys.argv) > 3 else BASE
    ids, nctx = d.spec(names)
    out = {}

    def stream_costs(idx, cost, K):
        sc = {i: np.zeros((d.streams, K), np.float64) for i in idx}
        d.pool(lambda i: d._walk(i, ids, 1, cost=cost, K=K, scost=sc[i]), idx)
        return sc

    for sp in ('A', 'B'):
        tr, te = SPLITS[sp]
        c = np.zeros((len(tr), 3, nctx, 16), np.uint32)
        d.pool(lambda j: d._walk(tr[j], ids, 0, counts=c[j]), range(len(tr)))
        one = np.ascontiguousarray((14.0 - np.log2(quantize(c.sum(0, dtype=np.uint64))))[None], np.float32)
        base_te = sum(v.sum() for v in stream_costs(te, one, 1).values()) / (len(te) * d.weights)
        s1 = stream_costs(tr, one, 1)
        for K in (2, 3, 4):
            # start: streams ranked by their cost under the single model, cut into K equal groups
            allc = np.concatenate([s1[i][:, 0] for i in tr])
            cuts = np.quantile(allc, [k / K for k in range(1, K)])
            assign = {i: np.searchsorted(cuts, s1[i][:, 0]).astype(np.uint8) for i in tr}
            hist = []
            for it in range(8):
                ck = {i: np.zeros((K, 3, nctx, 16), np.uint32) for i in tr}
                d.pool(lambda i: d._walk(i, ids, 2, counts=ck[i], K=K, assign=assign[i]), tr)
                tot = sum(ck[i].astype(np.uint64) for i in tr)
                cost = np.ascontiguousarray(14.0 - np.log2(quantize(tot)), np.float32)
                sc = stream_costs(tr, cost, K)
                assign = {i: sc[i].argmin(1).astype(np.uint8) for i in tr}
                hist.append(sum(sc[i].min(1).sum() for i in tr) / (len(tr) * d.weights) + np.log2(K) * d.streams / d.weights)
            ste = stream_costs(te, cost, K)
            payload = sum(v.min(1).sum() for v in ste.values()) / (len(te) * d.weights)
            share = np.bincount(np.concatenate([v.argmin(1) for v in ste.values()]), minlength=K) / (len(te) * d.streams)
            sel = np.log2(K) * d.streams / d.weights
            out.setdefault(f'K{K}', {'features': names, 'item': 8, 'contexts_per_matrix': nctx,
                                     'model_bytes': K * 3 * nctx * 34, 'selector_bpw': sel,
                                     'decode_cost': 'as today + one table-set index per stream', 'splits': {}})
            out[f'K{K}']['splits'][sp] = {'heldout_bpw': payload + sel, 'baseline_bpw': base_te,
                                          'delta_pct_of_raw': (payload + sel - base_te) / RAW_BPW * 100,
                                          'train_bpw_by_iteration': hist, 'test_share_per_set': share.tolist()}
            print(sp, K, f'{base_te:.5f} -> {payload + sel:.5f}  d%raw {(payload + sel - base_te) / RAW_BPW * 100:+.4f}',
                  'train', ' '.join(f'{h:.5f}' for h in hist), flush=True)
    for v in out.values():
        v['delta_pct_of_raw_mean_AB'] = (v['splits']['A']['delta_pct_of_raw'] + v['splits']['B']['delta_pct_of_raw']) / 2
    save('kmix_' + '+'.join(names), out)


def stage_layers(d):
    """(9) one model per layer / per group of layers instead of one global model. Train on two experts of
    each layer (or group), test on the third; same test experts as the global model of splits A and B."""
    c = d.counts(BASE).astype(np.uint64)
    out = {}
    groups = {'global': [list(range(8))], 'per_layer': [[i] for i in range(8)],
              'groups_of_2': [[0, 1], [2, 3], [4, 5], [6, 7]], 'groups_of_4': [[0, 1, 2, 3], [4, 5, 6, 7]]}
    for name, gs in groups.items():
        r = {'item': 9, 'models_for_8_sampled_layers': len(gs), 'splits': {}}
        for sp in ('A', 'B'):
            tr, te = SPLITS[sp]
            total = 0.0
            for g in gs:
                gtr = [i for i in tr if i // 3 in g]
                gte = [i for i in te if i // 3 in g]
                total += bits(c[gte].sum(0), c[gtr].sum(0))
            r['splits'][sp] = {'heldout_bpw': total / (len(te) * d.weights)}
        out[name] = r
    for name, r in out.items():
        for sp in ('A', 'B'):
            r['splits'][sp]['delta_pct_of_raw'] = (r['splits'][sp]['heldout_bpw'] - out['global']['splits'][sp]['heldout_bpw']) / RAW_BPW * 100
        r['delta_pct_of_raw_mean_AB'] = (r['splits']['A']['delta_pct_of_raw'] + r['splits']['B']['delta_pct_of_raw']) / 2
        print(name, json.dumps(r['splits']))
    out['per_layer']['model_bytes_40_layers'] = 40 * 59408
    out['groups_of_2']['model_bytes_if_20_groups'] = 20 * 59408
    out['groups_of_4']['model_bytes_if_10_groups'] = 10 * 59408
    # per-layer cost of the global model: how different are the layers
    tr, te = SPLITS['A']
    f = c[tr].sum(0)
    out['global_model_bpw_by_layer_split_A'] = {str(d.rows[i]['layer']): bits(c[i], f) / d.weights for i in te}
    save('layers', out)


def stage_precision(d):
    """Cost of smaller CDF precision (table-driven decoders want 2^bits-entry slot tables) for today's 567
    contexts and for the reduced context sets; 8-symbol magnitude alphabet + raw sign bit as a second form."""
    out = {}
    for label, names in (('SREL+PM1+RMAX', BASE), ('SREL+RMAX', ['SREL', 'RMAX']), ('SREL', ['SREL']),
                         ('SREL+ROWLVL', ['SREL', 'ROWLVL'])):
        c = d.counts(names)
        mag = c[..., :8].astype(np.uint64) + c[..., 8:]
        nonzero = mag[..., 1:].sum((1, 2, 3)).astype(np.float64)
        out[label] = {'contexts_per_matrix': c.shape[2], 'bits': {}}
        for nb in (14, 13, 12, 11, 10):
            r = {}
            for sp, (tr, te) in SPLITS.items():
                n = len(te) * d.weights
                r[sp] = {'joint16_bpw': bits(c[te].sum(0, dtype=np.uint64), c[tr].sum(0, dtype=np.uint64), 1 << nb) / n,
                         'mag8_plus_raw_sign_bpw': (bits(mag[te].sum(0), mag[tr].sum(0), 1 << nb) + nonzero[te].sum()) / n}
            out[label]['bits'][str(nb)] = r
            print(f"{label:14s} {nb} bit  joint A {r['A']['joint16_bpw']:.5f} B {r['B']['joint16_bpw']:.5f} "
                  f"C {r['C']['joint16_bpw']:.5f} | mag8+sign A {r['A']['mag8_plus_raw_sign_bpw']:.5f} "
                  f"B {r['B']['mag8_plus_raw_sign_bpw']:.5f} C {r['C']['mag8_plus_raw_sign_bpw']:.5f}", flush=True)
    save('precision', out)


def stage_decbench(d):
    """Run out/decbench.exe (single thread CPU decode-cost classes) on two experts and keep its output."""
    import subprocess
    out = {}
    for i in (10, 21):
        r = d.rows[i]
        text = subprocess.run([os.path.join(d.out, 'decbench.exe'), os.path.join(d.out, 'slabs', r['file']),
                               r['entry_hex']] + [str(a) for a in r['anchors']], capture_output=True, text=True,
                              check=True).stdout
        print(text)
        out[r['file']] = {'real_weight_payload_bytes': r['weight_payload_bytes'], 'lines': text.splitlines()}
    save('decbench', out)


if __name__ == '__main__':
    data = Data(sys.argv[1])
    {'validate': stage_validate, 'contexts': stage_contexts, 'combo': stage_combo, 'kmix': stage_kmix,
     'layers': stage_layers, 'precision': stage_precision, 'decbench': stage_decbench}[sys.argv[2]](data)
