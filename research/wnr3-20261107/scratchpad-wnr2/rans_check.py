"""Exact WNR2-style rANS (L = 2^23, 14-bit CDFs, byte renormalisation, 4-B big-endian flush) on real V4.1 streams.

Usage: python -I rans_check.py SAMPLE_DIR COUNTS_DIR [streams]
Compares stream sizes when the encoder starts at x0 = L (WNR2 today: decoder must end at L)
and at x0 = L + h, h = 23-bit hash of the stream's symbols + stream index (decoder recomputes h from what it
decoded and checks final state == L + h). Also checks round trips and that corruption is detected.
"""
import json, os, sys, random, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxanalyze import quantize, load_all, tot, P
from mxcount import load_expert

L = 1 << 23
PB = 14


def tables(freq):
    cum = np.zeros((freq.shape[0], 17), np.int64)
    cum[:, 1:] = np.cumsum(freq, 1)
    return freq.astype(np.int64), cum


def encode(syms, ctxs, freq, cum, x0):
    x = x0
    out = bytearray()
    for s, c in zip(reversed(syms), reversed(ctxs)):
        f = int(freq[c, s]); st = int(cum[c, s])
        x_max = ((L >> PB) << 8) * f
        while x >= x_max:
            out.append(x & 255)
            x >>= 8
        x = ((x // f) << PB) + (x % f) + st
    out += bytes([(x >> 0) & 255, (x >> 8) & 255, (x >> 16) & 255, (x >> 24) & 255])
    out.reverse()  # decoder reads forward: 4-B big-endian state, then renorm bytes
    return bytes(out)


def decode(buf, ctx_fn, n, freq, cum):
    x = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3]
    pos = 4
    syms = []
    for i in range(n):
        c = ctx_fn(i, syms)
        slot = x & (P - 1)
        row = cum[c]
        s = int(np.searchsorted(row, slot, side='right') - 1)
        x = int(freq[c, s]) * (x >> PB) + slot - int(row[s])
        while x < L:
            if pos >= len(buf):
                return None, None
            x = (x << 8) | buf[pos]
            pos += 1
        syms.append(s)
    if pos != len(buf):
        return None, None
    return syms, x


def h23(syms, idx):
    return zlib.crc32(bytes(syms) + idx.to_bytes(4, 'little')) & ((1 << 23) - 1)


def main():
    sample_dir, counts_dir = sys.argv[1:3]
    nstreams = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    ex = load_all(sample_dir, counts_dir)
    freq_all = quantize(tot(ex, 'w_base'))  # [3, 567, 16]
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    e = man['experts'][17]
    t = load_expert(os.path.join(sample_dir, e['file']), e['tensors'])
    wraw, wshape, _ = t['w1.weight']; sraw, _, _ = t['w1.scale']
    R = wshape[0]
    wb = wraw.reshape(R, -1); sc = sraw.reshape(R, -1).astype(np.int16)
    med = int(np.sort(sc.ravel())[(sc.size - 1) // 2])
    sym = np.empty((R, wb.shape[1] * 2), np.uint8); sym[:, 0::2] = wb & 15; sym[:, 1::2] = wb >> 4
    flat = sym.ravel().astype(np.int64)
    srel_flat = np.repeat((np.clip(sc - med, -3, 3) + 3).ravel(), 32)
    freq, cum = tables(freq_all[0])
    rng = random.Random(7)
    picks = sorted(rng.sample(range(flat.size // 2048), nstreams))
    tot0 = tot1 = 0
    ok = detected = trials = 0
    detected_old = misplaced_new = misplaced_old = 0
    for k, si in enumerate(picks):
        a = si * 2048
        syms = flat[a:a + 2048].tolist()
        srel = srel_flat[a:a + 2048].tolist()
        ctxs = []
        for i, s in enumerate(syms):
            if i % 32 == 0:
                pm = mx = 8
            ctxs.append((srel[i] * 9 + pm) * 9 + mx)
            pm = s & 7
            mx = (s & 7) if mx == 8 else max(mx, s & 7)

        def ctx_fn(i, dec):
            if i % 32 == 0:
                return (srel[i] * 9 + 8) * 9 + 8
            m = [d & 7 for d in dec[i - (i % 32):i]]
            return (srel[i] * 9 + m[-1]) * 9 + max(m)

        b0 = encode(syms, ctxs, freq, cum, L)
        h = h23(syms, si)
        b1 = encode(syms, ctxs, freq, cum, L + h)
        tot0 += len(b0); tot1 += len(b1)
        if k < 40:  # full decode checks on a subset (slow pure-Python decoder)
            d0, x0 = decode(b0, ctx_fn, 2048, freq, cum)
            d1, x1 = decode(b1, ctx_fn, 2048, freq, cum)
            if d0 == syms and x0 == L and d1 == syms and x1 == L + h23(d1, si):
                ok += 1
            # corrupt one payload byte (not the state) and check detection by each rule
            bb = bytearray(b1); j = 4 + rng.randrange(len(bb) - 4); bb[j] ^= 1 << rng.randrange(8)
            dc, xc = decode(bytes(bb), ctx_fn, 2048, freq, cum)
            trials += 1
            if dc is None or xc != L + h23(dc, si):
                detected += 1
            # today's rule on the same kind of damage to the x0 = L stream
            bb0 = bytearray(b0); j0 = 4 + rng.randrange(len(bb0) - 4); bb0[j0] ^= 1 << rng.randrange(8)
            d0c, x0c = decode(bytes(bb0), ctx_fn, 2048, freq, cum)
            if d0c is None or x0c != L:
                detected_old += 1
            # misplaced but valid data: stream si decoded as if it were stream si+1 (same model, same length rule)
            dm, xm = decode(b1, ctx_fn, 2048, freq, cum)
            if dm is not None and xm != L + h23(dm, si + 1):
                misplaced_new += 1
            d0m, x0m = decode(b0, ctx_fn, 2048, freq, cum)
            if d0m is None or x0m != L:
                misplaced_old += 1
    print(json.dumps({'streams': nstreams, 'bytes_x0_L': tot0, 'bytes_x0_L_plus_hash': tot1,
                      'extra_bytes_per_stream': (tot1 - tot0) / nstreams,
                      'mean_stream_bytes': tot0 / nstreams,
                      'round_trips_ok': f'{ok}/40', 'single_bitflip_detected_hash_rule': f'{detected}/{trials}',
                      'single_bitflip_detected_today_rule': f'{detected_old}/{trials}',
                      'misplaced_stream_detected_hash_rule': f'{misplaced_new}/{trials}',
                      'misplaced_stream_detected_today_rule': f'{misplaced_old}/{trials}'}, indent=1))


if __name__ == '__main__':
    main()
