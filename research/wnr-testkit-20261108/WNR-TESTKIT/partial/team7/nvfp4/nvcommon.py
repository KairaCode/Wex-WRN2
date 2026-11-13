"""Shared helpers for the NVFP4 analysis (copied into each script's namespace with exec; no package import under -I).

E2M1 code c: bit 3 = sign, bits 0-2 = magnitude index into E2M1_MAG.  E4M3 (fn) byte: s eeee mmm, bias 7.
"""
import json, os
import numpy as np

E2M1_MAG = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float64)
P = 16384


def e4m3_value(b):
    b = np.asarray(b, np.int64)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 15
    m = b & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7))
    v = np.where((b & 0x7F) == 0x7F, np.nan, v)
    return s * v


def load_item(fam_dir, it):
    raw = np.fromfile(os.path.join(fam_dir, it['file']), dtype=np.uint8)
    out, off = {}, 0
    for t in it['tensors']:
        n = t['bytes']
        out[(t['role'], t['part'])] = (raw[off:off + n], t['shape'], t['dtype'])
        off += n
    assert off == raw.size
    return out


def f32(buf):
    return float(np.frombuffer(buf.tobytes(), '<f4')[0])


def unpack(wb):
    """uint8 [R, C/2] -> uint8 codes [R, C], low nibble = even column."""
    R = wb.shape[0]
    sym = np.empty((R, wb.shape[1] * 2), np.uint8)
    sym[:, 0::2] = wb & 15
    sym[:, 1::2] = wb >> 4
    return sym


def matrices(fam_dir, it, roles=None):
    """Yield (role, codes [R, C] uint8, scale bytes [R, C/16] uint8, tscale float, iscale float, raw_bytes)."""
    t = load_item(fam_dir, it)
    if roles is None:
        roles = list(dict.fromkeys(x['role'] for x in it['tensors']))
    for r in roles:
        wraw, wshape, _ = t[(r, 'weight')]
        sraw, sshape, sdt = t[(r, 'scale')]
        R = wshape[0]
        codes = unpack(wraw.reshape(R, -1))
        sc = sraw.reshape(R, -1)
        nb = sum(t[(r, p)][0].size for p in ('weight', 'scale', 'tscale', 'iscale') if (r, p) in t)
        yield r, codes, sc, f32(t[(r, 'tscale')][0]), (f32(t[(r, 'iscale')][0]) if (r, 'iscale') in t else None), nb


def quantize(c):
    """WNR2 14-bit CDF quantizer (identical to wnr2 tools mxanalyze.py quantize)."""
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


def cost_bits(counts, freq):
    return float((np.asarray(counts, np.float64) * -np.log2(np.asarray(freq, np.float64) / P)).sum())


def ideal_bits(counts):
    c = np.asarray(counts, np.float64)
    t = c.sum(-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(c > 0, c / np.where(t == 0, 1, t), 1)
    return float(-(c * np.log2(p)).sum())
