"""ans-prior-art: measure per-stream rANS overheads of WNR2 variants on real V4.1 streams.

Usage: python -I flushx.py SAMPLE_DIR COUNTS_DIR OUT_JSON [streams_per_matrix] [experts]

Coder = WNR2 weight coder (32-bit state, 14-bit CDFs, byte renorm, shared in-sample 567-context model built
from the pass-1 counts exactly like wex_nr2_quantize). Variants of the encoder start state x0 (= the state the
decoder must END in) and of the flush:
  L        x0 = L = 2^23, 4-B flush                              (WNR2 today)
  Lh23     x0 = L + h23                                           (hash-in-initial-state idea, as measured before)
  small_k  x0 = 2^k + h_k  (k-bit hash, k = 0 -> x0 = 1)          ("minimal flush": decoder stops reading at
           the stream end, Bloom 2015; the last few symbols are decoded with state < L)
  h23lo    x0 = h23 in [1, 2^23)                                  (23-bit hash, but x0 < L)
  L16      x0 = 2^16, 16-bit renorm words (DietGPU/Recoil style) (coding loss with 14-bit probabilities)
Flush: fixed 4 B; 'sd34' = self-delimiting 3-or-4-byte flush (first byte >= 0x80 <=> 3-byte state in [2^23,2^24)).
Also: 32-lane interleaved single buffer (ryg_rans N-way / Giesen 2014 / DietGPU), round trip + sizes.
"""
import json, os, sys, random, zlib, time
import numpy as np

P = 16384
PB = 14
L = 1 << 23
W_CHUNK = 2048


def quantize(c):  # copy of the WNR2-exact quantizer used by the team's mxanalyze.py
    c = np.asarray(c, np.int64)
    n = c.shape[-1]
    flat = c.reshape(-1, n)
    tot = flat.sum(1, keepdims=True)
    safe = np.where(tot == 0, 1, tot)
    num = flat * (P - n)
    fl = num // safe
    rem = num % safe
    freq = 1 + fl
    left = P - freq.sum(1)
    order = np.argsort(-rem, axis=1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n)[None, :].repeat(flat.shape[0], 0), axis=1)
    freq = freq + (rank < left[:, None])
    freq[tot[:, 0] == 0] = P // n
    assert (freq.sum(1) == P).all() and (freq >= 1).all()
    return freq.reshape(c.shape)


def load_expert(path, tensors):
    raw = np.fromfile(path, dtype=np.uint8)
    out, off = {}, 0
    for t in tensors:
        n = t['bytes']
        out[t['name']] = (raw[off:off + n], t['shape'])
        off += n
    assert off == raw.size
    return out


def matrix_syms_ctx(wraw, wshape, sraw):
    R = wshape[0]
    wb = wraw.reshape(R, -1)
    sc = sraw.reshape(R, -1).astype(np.int16)
    C = wb.shape[1] * 2
    sym = np.empty((R, C), np.uint8)
    sym[:, 0::2] = wb & 15
    sym[:, 1::2] = wb >> 4
    nb = C // 32
    blk = sym.reshape(R, nb, 32)
    mag = blk & 7
    pm = np.empty_like(mag); pm[..., 0] = 8; pm[..., 1:] = mag[..., :-1]
    cm = np.maximum.accumulate(mag, axis=2)
    mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
    med = int(np.sort(sc.ravel())[(sc.size - 1) // 2])
    srel = (np.clip(sc - med, -3, 3) + 3).astype(np.int32)
    ctx = (srel[..., None] * 9 + pm.astype(np.int32)) * 9 + mx.astype(np.int32)
    return sym.ravel().astype(np.int64), ctx.ravel().astype(np.int64), np.repeat(srel.ravel(), 32)


class Model:
    def __init__(self, freq):
        self.f = freq.astype(np.int64).tolist()
        cum = np.zeros((freq.shape[0], 17), np.int64)
        cum[:, 1:] = np.cumsum(freq, 1)
        self.c = cum.tolist()
        lut = np.zeros((freq.shape[0], P), np.uint8)
        for r in range(freq.shape[0]):
            lut[r] = np.repeat(np.arange(16, dtype=np.uint8), freq[r])
        self.lut = [bytes(row) for row in lut]


def enc_core(syms, ctxs, m, x0, Lb=L, rb=8):
    """Returns (renorm words in DECODER read order, final encoder state). rb = renorm bits."""
    x = x0
    out = []
    fr, cu = m.f, m.c
    mask = (1 << rb) - 1
    xm_mul = (Lb >> PB) << rb
    for i in range(len(syms) - 1, -1, -1):
        s = syms[i]; c = ctxs[i]
        f = fr[c][s]
        x_max = xm_mul * f
        while x >= x_max:
            out.append(x & mask)
            x >>= rb
        x = ((x // f) << PB) + (x % f) + cu[c][s]
    out.reverse()
    return out, x


def ctx_stream(srel):
    """Decoder-side context generator state is kept inline in dec_core."""
    return srel


def dec_core(words, xinit, srel, n, m, Lb=L, rb=8, tail_ok=False):
    """Decode n symbols. tail_ok=False: WNR2 rule (reading past the end = error).
    tail_ok=True: minimal-flush rule (renorm only while words remain). Returns (syms, final x, pos) or None."""
    x = xinit
    pos = 0
    nw = len(words)
    fr, cu, lut = m.f, m.c, m.lut
    out = [0] * n
    pm = mx = 8
    for i in range(n):
        if (i & 31) == 0:
            pm = mx = 8
        c = (srel[i] * 9 + pm) * 9 + mx
        slot = x & (P - 1)
        s = lut[c][slot]
        x = fr[c][s] * (x >> PB) + slot - cu[c][s]
        while x < Lb:
            if pos >= nw:
                if tail_ok:
                    break
                return None
            x = (x << rb) | words[pos]
            pos += 1
        out[i] = s
        a = s & 7
        pm = a
        mx = a if mx == 8 else (a if a > mx else mx)
    if pos != nw:
        return None
    return out, x


def h_bits(syms, idx, rec, k):
    if k == 0:
        return 0
    v = zlib.crc32(bytes(syms) + idx.to_bytes(4, 'little') + rec.to_bytes(4, 'little'))
    v ^= zlib.crc32(v.to_bytes(4, 'little'), 0x9E3779B9) << 1  # 33 bits of mixing, then truncate
    return v & ((1 << k) - 1)


def flush_bytes(x, mode):
    if mode == 'fixed4':
        return 4
    if mode == 'sd34':  # self-delimiting: x in [2^23, 2^24) -> 3 bytes (first byte >= 0x80), else 4 (first byte < 0x80)
        return 3 if (1 << 23) <= x < (1 << 24) else 4
    raise ValueError(mode)


def main():
    sample_dir, counts_dir, out_json = sys.argv[1:4]
    per_m = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    n_exp = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    t0 = time.time()
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    cw = None
    for e in man['experts']:
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        cw = z['w_base'].astype(np.int64) if cw is None else cw + z['w_base']
    freq = quantize(cw)  # [3, 567, 16], in-sample like the team's baseline
    models = [Model(freq[mi]) for mi in range(3)]
    rng = random.Random(20261107)
    exps = rng.sample(range(len(man['experts'])), n_exp)
    streams = []  # (rec, mi, idx, syms, ctxs, srel)
    for rec in exps:
        e = man['experts'][rec]
        t = load_expert(os.path.join(sample_dir, e['file']), e['tensors'])
        for mi, w in enumerate(['w1', 'w2', 'w3']):
            wraw, wshape = t[f'{w}.weight']
            sraw, _ = t[f'{w}.scale']
            sym, ctx, srel = matrix_syms_ctx(wraw, wshape, sraw)
            ns = sym.size // W_CHUNK
            # one contiguous run of 32 streams (for the interleaved test) + random others
            g0 = rng.randrange(0, ns - 32)
            picks = list(range(g0, g0 + 32)) + rng.sample([i for i in range(ns) if not g0 <= i < g0 + 32], max(0, per_m - 32))
            for si in picks:
                a = si * W_CHUNK
                streams.append((rec, mi, si, sym[a:a + W_CHUNK].tolist(), ctx[a:a + W_CHUNK].tolist(), srel[a:a + W_CHUNK].tolist()))
    print(f'{len(streams)} streams loaded in {time.time() - t0:.1f}s', flush=True)

    # ---- sizes per variant ----
    variants = {
        'L': dict(x0=lambda S: L, Lb=L, rb=8, tail=False),
        'Lh23': dict(x0=lambda S: L + h_bits(S[3], S[2], S[0], 23), Lb=L, rb=8, tail=False),
        'small_k0': dict(x0=lambda S: 1, Lb=L, rb=8, tail=True),
        'small_k8': dict(x0=lambda S: (1 << 8) + h_bits(S[3], S[2], S[0], 8), Lb=L, rb=8, tail=True),
        'small_k16': dict(x0=lambda S: (1 << 16) + h_bits(S[3], S[2], S[0], 16), Lb=L, rb=8, tail=True),
        'small_k23': dict(x0=lambda S: (1 << 23) + h_bits(S[3], S[2], S[0], 23), Lb=L, rb=8, tail=True),
        'h23lo': dict(x0=lambda S: max(1, h_bits(S[3], S[2], S[0], 23)), Lb=L, rb=8, tail=True),
        'L16': dict(x0=lambda S: 1 << 16, Lb=1 << 16, rb=16, tail=False),
    }
    res = {'streams': len(streams), 'experts': [man['experts'][r]['file'] for r in exps]}
    ideal = 0.0
    for S in streams:
        m = models[S[1]]
        ideal += sum(-np.log2(m.f[c][s] / P) for s, c in zip(S[3], S[4]))
    res['ideal_quantized_model_bytes_per_stream'] = ideal / 8 / len(streams)
    enc_cache = {}
    for name, v in variants.items():
        tot4 = totsd = 0
        lfinal = []
        for k, S in enumerate(streams):
            m = models[S[1]]
            x0 = v['x0'](S)
            words, xf = enc_core(S[3], S[4], m, x0, v['Lb'], v['rb'])
            wbytes = len(words) * (v['rb'] // 8)
            tot4 += wbytes + 4
            totsd += wbytes + (flush_bytes(xf, 'sd34') if v['rb'] == 8 else 4)
            lfinal.append(np.log2(xf))
            if k < 24 or name in ('L', 'small_k0'):
                enc_cache[(name, k)] = (words, xf, x0)
        res[name] = {'bytes_per_stream_fixed4': tot4 / len(streams),
                     'bytes_per_stream_selfdelim34': totsd / len(streams),
                     'mean_log2_final_state': float(np.mean(lfinal)),
                     'frac_final_state_lt_2^24': float(np.mean(np.array(lfinal) < 24))}
        print(name, json.dumps(res[name]), f'{time.time() - t0:.0f}s', flush=True)
    base = res['L']['bytes_per_stream_fixed4']
    for name in variants:
        res[name]['delta_vs_L_bytes_per_stream'] = res[name]['bytes_per_stream_fixed4'] - base
        res[name]['delta_vs_L_pct_of_raw_record_at_19440_streams'] = 100 * 19440 * res[name]['delta_vs_L_bytes_per_stream'] / 18_800_640

    # ---- round trips (24 streams per variant) ----
    rt = {}
    for name, v in variants.items():
        ok = 0
        for k in range(24):
            S = streams[k]; m = models[S[1]]
            words, xf, x0 = enc_cache[(name, k)]
            d = dec_core(words, xf, S[5], W_CHUNK, m, v['Lb'], v['rb'], v['tail'])
            ok += d is not None and d[0] == S[3] and d[1] == x0
        rt[name] = f'{ok}/24'
    res['round_trips'] = rt
    print('round trips', rt, flush=True)

    # ---- corruption: false-accept rate of the end check for x0 = 1 (k = 0) and k = 8, L (today) ----
    trials = int(os.environ.get('TRIALS', '400'))
    cr = {}
    for name, k_bits in (('small_k0', 0), ('small_k8', 8), ('L', None)):
        v = variants[name]
        acc = acc_struct_only = struct_fail = 0
        for tr in range(trials):
            kk = rng.randrange(len(streams)) if name != 'small_k8' else rng.randrange(24)
            S = streams[kk]; m = models[S[1]]
            if (name, kk) in enc_cache:
                words, xf, x0 = enc_cache[(name, kk)]
            else:
                x0 = v['x0'](S); words, xf = enc_core(S[3], S[4], m, x0, v['Lb'], v['rb'])
            # flip one bit anywhere in [4-B state | payload]
            nbit = 32 + 8 * len(words)
            b = rng.randrange(nbit)
            if b < 32:
                xs = xf ^ (1 << b); ws = words
            else:
                b -= 32; ws = list(words); ws[b // 8] ^= 1 << (b % 8); xs = xf
            d = None if not (L <= xs < 256 * L) else dec_core(ws, xs, S[5], W_CHUNK, m, v['Lb'], v['rb'], v['tail'])
            if d is None:
                struct_fail += 1
                continue
            if name == 'L':
                exp = L
            elif k_bits == 0:
                exp = 1
            else:
                exp = (1 << 8) + h_bits(d[0], S[2], S[0], 8)
            acc += d[1] == exp
            acc_struct_only += 1
        cr[name] = {'trials': trials, 'caught_by_length_rule': struct_fail,
                    'reached_end_check': acc_struct_only, 'FALSE_ACCEPTS': acc}
        print('corruption', name, cr[name], f'{time.time() - t0:.0f}s', flush=True)
    res['corruption_single_bitflip'] = cr

    # ---- 32-lane interleaved single buffer (ryg_rans N-way, Giesen 2014, DietGPU warp layout) ----
    # group = the first 32 streams of each (expert, matrix) block (consecutive stream indices)
    il = []
    per_block = per_m
    for g in range(0, len(streams), per_block):
        grp = streams[g:g + 32]
        m = models[grp[0][1]]
        # encoder: t descending, lane descending; each lane x0 = L + h23 (integrity, no tail problem)
        xs = [L + h_bits(S[3], S[2], S[0], 23) for S in grp]
        x0s = list(xs)
        out = []
        fr, cu = m.f, m.c
        xm_mul = (L >> PB) << 8
        for t in range(W_CHUNK - 1, -1, -1):
            for ln in range(31, -1, -1):
                S = grp[ln]
                s = S[3][t]; c = S[4][t]
                f = fr[c][s]; x = xs[ln]
                x_max = xm_mul * f
                while x >= x_max:
                    out.append(x & 255); x >>= 8
                xs[ln] = ((x // f) << PB) + (x % f) + cu[c][s]
        out.reverse()
        buf = bytes(out)
        sep = sum(len(enc_core(S[3], S[4], m, x0s[i])[0]) for i, S in enumerate(grp))
        # decoder: lock-step; lane order within a step == __ballot/__popc prefix order
        x = list(xs); pos = 0
        fr, cu, lut = m.f, m.c, m.lut
        outsy = [[0] * W_CHUNK for _ in range(32)]
        pm = [8] * 32; mxv = [8] * 32
        need_hist = [0, 0, 0]
        ok = True
        for t in range(W_CHUNK):
            if (t & 31) == 0:
                pm = [8] * 32; mxv = [8] * 32
            # phase 1: every lane decodes its symbol (no memory dependence between lanes)
            needs = [0] * 32
            for ln in range(32):
                S = grp[ln]
                c = (S[5][t] * 9 + pm[ln]) * 9 + mxv[ln]
                xv = x[ln]; slot = xv & (P - 1); s = lut[c][slot]
                xv = fr[c][s] * (xv >> PB) + slot - cu[c][s]
                x[ln] = xv
                needs[ln] = (xv < L) + (xv < (L >> 8))  # 0/1/2 bytes, known BEFORE reading
                outsy[ln][t] = s
                a = s & 7; pm[ln] = a; mxv[ln] = a if mxv[ln] == 8 else max(a, mxv[ln])
            # phase 2: offsets = exclusive prefix sum of needs (= popc(b1 & lt) + popc(b2 & lt) on a GPU)
            off = pos
            for ln in range(32):
                for _ in range(needs[ln]):
                    if off >= len(buf):
                        ok = False; break
                    x[ln] = (x[ln] << 8) | buf[off]; off += 1
                need_hist[needs[ln]] += 1
            pos = off
        ok = ok and pos == len(buf) and all(outsy[i] == grp[i][3] for i in range(32)) and \
            all(x[i] == L + h_bits(outsy[i], grp[i][2], grp[i][0], 23) for i in range(32))
        il.append({'group_payload_bytes': len(buf), 'sum_of_separate_payloads': sep, 'round_trip_ok': ok,
                   'bytes_per_symbol_step_hist_0_1_2': need_hist})
        print('interleaved group', il[-1], f'{time.time() - t0:.0f}s', flush=True)
    res['interleaved_32'] = il
    json.dump(res, open(out_json, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
