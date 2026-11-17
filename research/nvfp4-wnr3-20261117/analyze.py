# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Raw sizes, entropy estimates, nibble order and slab export for the experts that fetch.py wrote.

  python -I analyze.py DATA_DIR OUT_DIR

DATA_DIR is fetch.py's output (DATA_DIR/<checkpoint>/experts.safetensors). For every checkpoint, every
fetched expert (layer, expert) and every matrix (WEX order gate/w1, down/w2, up/w3):
  nibbles q of the packed U8/I8 weights, low nibble = even element (the WEX convention; checked below);
  scales s, one byte per block of B values (MXFP4: B = 32, E8M0; NVFP4: B = 16, FP8 E4M3);
  NVFP4 only: the two F32 scalars weight_scale_2 and input_scale.
Writes OUT_DIR/slabs/<checkpoint>/L<layer>_E<expert>.nvs (input of nvwnr.c: a 64-byte header, then scales of
the three matrices, weights of the three matrices, then the NVFP4 scalars) and OUT_DIR/slabs/<checkpoint>/
L<layer>_E<expert>.slab (the same bytes without the header: the WEX slab for MXFP4, the slab plus 24 bytes of
scalars for NVFP4, the input of the general-purpose baselines). Writes OUT_DIR/estimates.json and prints
Markdown tables.

Estimates are empirical entropies of the counts pooled per matrix over the 16 experts of a checkpoint (the
WNR3 model is per matrix, trained once): H(q) order 0; H(q | scale class); H(q | scale class, pm, mx) with the
WNR3 contexts (pm = previous magnitude in the block, mx = running maximum, both 8 at the block start); scale
stream H(s) order 0, H(s | previous scale in the row), and the order-0 entropy of the difference to the previous
scale in the row (row starts coded order 0 apart). They are lower bounds of a static model fitted to the same
data: no model cost, no chunk overhead. Fractions are of the raw bytes of the same experts.
numpy and the standard library.
"""
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch  # noqa: E402

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def st_read(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        data = np.fromfile(f, dtype=np.uint8)
    out = {}
    for k, v in head.items():
        if k == '__metadata__':
            continue
        b, e = v['data_offsets']
        out[k] = (v['dtype'], v['shape'], data[b:e])
    return out


def e4m3(code):
    code = code.astype(np.int64)
    e, m = (code >> 3) & 15, code & 7
    val = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    return np.where(code & 128, -val, val)


def rule_g(values, n=256):
    h = np.bincount(values, minlength=n).astype(np.int64)
    pair = h[:-1] + h[1:]
    return int(np.argmax(pair))   # first maximum, as WNR3-SPEC 4.3


def experts(name, data_dir):
    repo, rev, fmt, prefix, mats, suff = fetch.CHECKPOINTS[name]
    t = st_read(os.path.join(data_dir, name, 'experts.safetensors'))
    keys = sorted({tuple(int(x) for x in k.split('layers.')[1].split('.')[0:1]) + (int(k.split('experts.')[1].split('.')[0]),)
                   for k in t})
    for L, e in keys:
        mm = []
        for m in mats:
            p = prefix.format(L=L, e=e) + m + '.'
            w = t[p + 'weight']
            s = t[p + suff[1]]
            rows, wcols = w[1]
            srows, scols = s[1]
            assert rows == srows and 2 * wcols % scols == 0
            B = 2 * wcols // scols
            assert B == (16 if fmt == 'nvfp4' else 32), (name, B)
            extra = b''
            g2 = inp = None
            if fmt == 'nvfp4':
                g2 = float(t[p + 'weight_scale_2'][2].view(np.float32)[0])
                inp = float(t[p + 'input_scale'][2].view(np.float32)[0])
                extra = t[p + 'weight_scale_2'][2].tobytes() + t[p + 'input_scale'][2].tobytes()
            mm.append({'rows': rows, 'wbytes': w[2], 'scales': s[2], 'B': B, 'g2': g2, 'inp': inp, 'extra': extra,
                       'sdtype': s[0], 'wdtype': w[0]})
        yield L, e, fmt, mm


def nibbles(wbytes, high_first=False):
    lo, hi = wbytes & 15, wbytes >> 4
    out = np.empty(2 * len(wbytes), dtype=np.uint8)
    out[0::2], out[1::2] = (hi, lo) if high_first else (lo, hi)
    return out


def ent(counts):
    """Entropy in bits of the total of a count table [contexts, symbols] (sum over contexts of n_c H_c)."""
    c = counts.astype(np.float64)
    tot = c.sum(axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(c > 0, c / tot, 1.0)
        return float(-(c * np.log2(p)).sum())


def wnr_ctx(q, B):
    """pm, mx of WNR3 section 4.1 for every nibble (block of B)."""
    mag = (q & 7).reshape(-1, B)
    pm = np.full_like(mag, 8)
    pm[:, 1:] = mag[:, :-1]
    mx = np.full_like(mag, 8)
    mx[:, 1:] = np.maximum.accumulate(mag, axis=1)[:, :-1]
    return pm.reshape(-1).astype(np.int64), mx.reshape(-1).astype(np.int64)


class Acc:
    def __init__(self):
        self.t = {}

    def add(self, key, ctx, sym, nctx, nsym):
        k = np.bincount(ctx.astype(np.int64) * nsym + sym, minlength=nctx * nsym).reshape(nctx, nsym)
        self.t[key] = self.t.get(key, 0) + k


def classes(fmt, s, cz_unused=None):
    """Scale classes per block: name -> (class array, number of classes)."""
    s = s.astype(np.int64)
    out = {}
    if fmt == 'mxfp4':
        a = rule_g(s)
        out['srel7 (WNR3)'] = (np.clip(s - a, -3, 3) + 3, 7)
        out['full code'] = (s, 256)
    else:
        e = (s >> 3) & 15
        a = rule_g(e, 16)
        out['exponent srel7'] = (np.clip(e - a, -3, 3) + 3, 7)
        out['exponent'] = (e, 16)
        out['mantissa'] = (s & 7, 8)
        ac = rule_g(s)
        out['code srel25'] = (np.clip(s - ac, -12, 12) + 12, 25)
        out['full code'] = (s, 256)
    out['none'] = (np.zeros_like(s), 1)
    return out


def analyze_ckpt(name, data_dir, slab_dir):
    acc = Acc()
    raw = {'weights': 0, 'scales': 0, 'global': 0}
    stats = {'experts': 0, 'neg0': 0, 'nibbles': 0, 'blocks': 0, 'max7_blocks': 0, 'max5_blocks': 0, 'even_blocks': 0,
             'scale_sign': 0, 'scale_zero': 0, 'scale_nan': 0, 'g2': [], 'inp': [], 'scale_codes_used': set()}
    os.makedirs(os.path.join(slab_dir, name), exist_ok=True)
    fmt = None
    for L, e, fmt, mm in experts(name, data_dir):
        stats['experts'] += 1
        payload_s, payload_w, extra = [], [], b''
        for m, x in enumerate(mm):
            q = nibbles(x['wbytes'])
            s = x['scales']
            B = x['B']
            raw['weights'] += len(x['wbytes'])
            raw['scales'] += len(s)
            raw['global'] += len(x['extra'])
            extra += x['extra']
            payload_s.append(s.tobytes())
            payload_w.append(x['wbytes'].tobytes())
            if x['g2'] is not None:
                stats['g2'].append(x['g2'])
                stats['inp'].append(x['inp'])
            mag = (q & 7).reshape(-1, B)
            M = mag.max(axis=1)
            has13 = ((mag == 1) | (mag == 3)).any(axis=1)
            stats['nibbles'] += len(q)
            stats['blocks'] += len(M)
            stats['neg0'] += int((q == 8).sum())
            stats['max7_blocks'] += int((M == 7).sum())
            stats['max5_blocks'] += int((M == 5).sum())
            stats['even_blocks'] += int(((M == 7) & ~has13).sum())
            if fmt == 'nvfp4':
                stats['scale_sign'] += int((s & 128).astype(bool).sum())
                stats['scale_zero'] += int((s == 0).sum())
                stats['scale_nan'] += int(((s & 127) == 127).sum())
            stats['scale_codes_used'] |= set(np.unique(s).tolist())
            pm, mx = wnr_ctx(q, B)
            qz = np.where(q == 8, 0, q)
            pmz, mxz = wnr_ctx(qz, B)
            acc.add(('q0', m), np.zeros(len(q), np.int64), q, 1, 16)
            acc.add(('q0z', m), np.zeros(len(q), np.int64), qz, 1, 16)
            for cname, (cls, nc) in classes(fmt, s).items():
                cb = np.repeat(cls, B)
                acc.add(('qs', cname, m), cb, q, nc, 16)
                acc.add(('qspm', cname, m), (cb * 9 + pm) * 9 + mx, q, nc * 81, 16)
                acc.add(('qspmz', cname, m), (cb * 9 + pmz) * 9 + mxz, qz, nc * 81, 16)
            # scale stream, rows of the scale tensor
            s2 = s.reshape(x['rows'], -1).astype(np.int64)
            prev = np.full_like(s2, 256)
            prev[:, 1:] = s2[:, :-1]
            acc.add(('s0', m), np.zeros(s2.size, np.int64), s2.reshape(-1), 1, 256)
            acc.add(('s1', m), prev.reshape(-1), s2.reshape(-1), 257, 256)
            d = (s2[:, 1:] - s2[:, :-1]) & 255
            acc.add(('sd', m), np.zeros(d.size, np.int64), d.reshape(-1), 1, 256)
            acc.add(('sd_start', m), np.zeros(s2.shape[0], np.int64), s2[:, 0], 1, 256)
            up = np.full_like(s2, 256)
            up[1:, :] = s2[:-1, :]
            acc.add(('s_up', m), up.reshape(-1), s2.reshape(-1), 257, 256)
        body = b''.join(payload_s) + b''.join(payload_w) + extra
        head = struct.pack('<4sIII', b'NVS1', mm[0]['B'], len(extra), 0)
        for x in mm:
            head += struct.pack('<II', x['rows'], 2 * len(x['wbytes']) // x['rows'])
        head += bytes(64 - len(head))
        base = os.path.join(slab_dir, name, f'L{L}_E{e}')
        with open(base + '.nvs', 'wb') as f:
            f.write(head + body)
        with open(base + '.slab', 'wb') as f:
            f.write(body)
    return fmt, raw, stats, acc


def summarize(name, fmt, raw, stats, acc):
    total = raw['weights'] + raw['scales'] + raw['global']
    n = stats['nibbles']
    r = {'checkpoint': name, 'format': fmt, 'experts': stats['experts'], 'raw': raw, 'raw_total': total,
         'nibbles': n, 'neg0_nibbles': stats['neg0'], 'neg0_fraction': stats['neg0'] / n,
         'blocks': stats['blocks'], 'max7_fraction': stats['max7_blocks'] / stats['blocks'],
         'max5_blocks': stats['max5_blocks'], 'even_fraction': stats['even_blocks'] / stats['blocks'],
         'scale_codes_used': len(stats['scale_codes_used'])}
    if fmt == 'nvfp4':
        r.update(scale_sign_bit=stats['scale_sign'], scale_zero=stats['scale_zero'], scale_nan=stats['scale_nan'],
                 g2_range=[min(stats['g2']), max(stats['g2'])], input_scale_range=[min(stats['inp']), max(stats['inp'])])
    sumk = lambda pre: sum(ent(v) for k, v in acc.t.items() if k[:len(pre)] == pre and len(k) == len(pre) + 1)
    w = {'order0': sumk(('q0',)), 'order0_cz': sumk(('q0z',))}
    for k in acc.t:
        if k[0] == 'qs' and k[2] == 0:
            c = k[1]
            w[f'| {c}'] = sumk(('qs', c))
            w[f'| {c}, pm, mx'] = sumk(('qspm', c))
            w[f'| {c}, pm, mx (CZ)'] = sumk(('qspmz', c))
    s = {'order0': sumk(('s0',)), 'order1 (left)': sumk(('s1',)), 'delta along row': sumk(('sd',)) + sumk(('sd_start',)),
         'order1 (row above)': sumk(('s_up',))}
    r['weight_bits_per_nibble'] = {k: v / n for k, v in w.items()}
    r['scale_bits_per_scale'] = {k: v / (raw['scales']) for k, v in s.items()}
    best_s = min(s.values())
    wnr = '| srel7 (WNR3), pm, mx' if fmt == 'mxfp4' else '| exponent srel7, pm, mx'
    est = {}
    for label, wk, sk in (('WNR3-like: srel7 contexts, scales order 0', wnr, 'order0'),
                          ('WNR3-like + CZ', wnr + ' (CZ)', 'order0'),
                          ('full scale code, pm, mx; best scale model', '| full code, pm, mx', None),
                          ('full scale code, pm, mx (CZ); best scale model', '| full code, pm, mx (CZ)', None)):
        bits = w[wk] + (best_s if sk is None else s[sk])
        est[label] = (bits / 8 + raw['global']) / total
    r['estimated_fraction_of_raw'] = est
    return r


def nibble_order(data_dir):
    """Dequantize the same expert matrices of the NVFP4 and the MXFP4 export; correlate with both nibble orders."""
    out = []
    for pair in fetch.PAIRS.values():
        if len(pair) < 2:
            continue
        nv, mx = pair[0], pair[1]
        a = {(L, e): mm for L, e, _, mm in experts(nv, data_dir)}
        b = {(L, e): mm for L, e, _, mm in experts(mx, data_dir)}
        (L, e) = sorted(a)[0]
        for m in range(3):
            x, y = a[(L, e)][m], b[(L, e)][m]
            rows = min(64, x['rows'])
            def deq(z, hf, fmtnv):
                q = nibbles(z['wbytes'][:rows * len(z['wbytes']) // z['rows']], hf)
                s = z['scales'][:rows * len(z['scales']) // z['rows']]
                sv = e4m3(s) * z['g2'] if fmtnv else 2.0 ** (s.astype(np.float64) - 127)
                return E2M1[q] * np.repeat(sv, z['B'])
            ref = deq(y, False, False)
            row = {'pair': f'{nv} vs {mx}', 'layer': L, 'expert': e, 'matrix': m, 'rows': rows}
            for hf in (False, True):
                v = deq(x, hf, True)
                row['corr_' + ('high_first' if hf else 'low_first')] = float(np.corrcoef(v, ref)[0, 1])
            v2 = deq(y, True, False)
            row['corr_mxfp4_high_first_vs_low_first'] = float(np.corrcoef(v2, ref)[0, 1])
            out.append(row)
    return out


def main(argv):
    data_dir, out_dir = argv[1], argv[2]
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for pair in fetch.PAIRS.values():
        for name in pair:
            if fetch.CHECKPOINTS[name][2] == 'nz5' or \
                    not os.path.exists(os.path.join(data_dir, name, 'experts.safetensors')):   # NZ5: extra.py
                continue
            fmt, raw, stats, acc = analyze_ckpt(name, data_dir, os.path.join(out_dir, 'slabs'))
            r = summarize(name, fmt, raw, stats, acc)
            results.append(r)
            print(json.dumps(r, indent=1), flush=True)
    order = nibble_order(data_dir)
    for row in order:
        print(json.dumps(row))
    # DeepSeek: is the Quark MXFP4 export the original checkpoint's bytes?
    same = None
    if os.path.exists(os.path.join(data_dir, 'ds-quark', 'experts.safetensors')):
        a = {(L, e): mm for L, e, _, mm in experts('ds-mxfp4', data_dir)}
        b = {(L, e): mm for L, e, _, mm in experts('ds-quark', data_dir)}
        same = {'experts': len(a), 'weights_equal': sum(all(np.array_equal(x['wbytes'], y['wbytes']) for x, y in zip(a[k], b[k])) for k in a),
                'scales_equal': sum(all(np.array_equal(x['scales'], y['scales']) for x, y in zip(a[k], b[k])) for k in a)}
        print('ds-mxfp4 vs ds-quark', json.dumps(same))
    with open(os.path.join(out_dir, 'estimates.json'), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'nibble_order': order, 'ds_native_vs_quark': same}, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
