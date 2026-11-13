"""Shared helpers (team4/ceil-origin): MXFP4 fingerprint, ceil-form transform T, WNR2 base-context counts, costs.

Copied/adapted from wnr2 tools mxceil.py (forward/inverse/ctx_counts) and mxanalyze.py (quantize/cost_bits).
Symbols: 4-bit E2M1 codes, bit 3 = sign, bits 0-2 = magnitude code (0,0.5,1,1.5,2,3,4,6). Blocks of 32 along a row.
"""
import numpy as np

P = 16384
HALF = np.array([0, 255, 1, 255, 2, 3, 4, 5], np.int16)
DOUBLE = np.array([0, 2, 4, 5, 6, 7, 255, 255], np.int16)


def unpack(wb):
    """wb uint8 [R, C/2] (low nibble = even column) -> sym uint8 [R, C]."""
    R = wb.shape[0]
    sym = np.empty((R, wb.shape[1] * 2), np.uint8)
    sym[:, 0::2] = wb & 15
    sym[:, 1::2] = wb >> 4
    return sym


def fingerprint(sym, sc):
    """Returns dict of int arrays: bmax_hist[8], even (max7 & no 1/3), mag_hist[8], mag_hist_even[8],
    max7_noodd_any (max 7, no code 1,3), max6_no13 (max 6, no 1/3), max5_blocks, neg0, pos0,
    scale_hist[256], even_by_maxpos? (not needed), 'max_lt6'."""
    R, C = sym.shape
    nb = C // 32
    blk = sym.reshape(R, nb, 32)
    mag = (blk & 7)
    bmax = mag.max(2)
    has13 = ((mag == 1) | (mag == 3)).any(2)
    even = (bmax == 7) & ~has13
    r = {}
    r['bmax_hist'] = np.bincount(bmax.ravel(), minlength=8)[:8]
    r['even'] = np.array(int(even.sum()))
    r['blocks'] = np.array(int(bmax.size))
    r['max6_no13'] = np.array(int(((bmax == 6) & ~has13).sum()))
    r['max5_no13'] = np.array(int(((bmax == 5) & ~has13).sum()))  # relevant for ceil-native storage
    mh = np.bincount(mag.ravel(), minlength=8)[:8]
    r['mag_hist'] = mh
    r['mag_hist_even'] = np.bincount(mag[even].ravel(), minlength=8)[:8]
    r['sym_hist'] = np.bincount(sym.ravel(), minlength=16)[:16]
    r['scale_hist'] = np.bincount(sc.ravel(), minlength=256)[:256]
    # i.i.d. reference: probability that a max-7 block lacks codes 1 and 3, given the matrix's magnitude shares
    # among the 31 non-max positions of max-7 blocks (empirical, excludes one max symbol per block)
    m7 = mag[bmax == 7]
    if m7.size:
        h = np.bincount(m7.ravel(), minlength=8)[:8].astype(np.float64)
        h[7] -= m7.shape[0]  # remove one max symbol per block
        p13 = (h[1] + h[3]) / h.sum()
        r['p_iid_even_given_max7'] = np.array((1 - p13) ** 31)
    else:
        r['p_iid_even_given_max7'] = np.array(np.nan)
    return r, even, bmax


def forward(sym, sc):
    R, C = sym.shape
    nb = C // 32
    blk = sym.reshape(R, nb, 32)
    mag = (blk & 7).astype(np.int16)
    bmax = mag.max(2)
    has13 = ((mag == 1) | (mag == 3)).any(2)
    even = (bmax == 7) & ~has13
    newmag = HALF[mag]
    tb = np.where(even[..., None], (blk & 8).astype(np.int16) | newmag, blk.astype(np.int16)).astype(np.uint8)
    tsc = np.where(even, sc.astype(np.int16) + 1, sc.astype(np.int16))
    return tb.reshape(R, C), tsc, even, bmax


def inverse(tsym, tsc):
    R, C = tsym.shape
    nb = C // 32
    blk = tsym.reshape(R, nb, 32)
    mag = (blk & 7).astype(np.int16)
    bmax = mag.max(2)
    odd = bmax == 5
    om = DOUBLE[np.minimum(mag, 7)]
    ob = np.where(odd[..., None], (blk & 8).astype(np.int16) | om, blk.astype(np.int16)).astype(np.uint8)
    osc = np.where(odd, tsc - 1, tsc)
    return ob.reshape(R, C), osc


def ctx_counts(sym, sc):
    """WNR2 base context: srel (scale - matrix median, clamped +-3) x previous magnitude (8=none) x running max (8=none)."""
    R, C = sym.shape
    nb = C // 32
    blk = sym.reshape(R, nb, 32)
    mag = blk & 7
    pm = np.empty_like(mag); pm[..., 0] = 8; pm[..., 1:] = mag[..., :-1]
    cm = np.maximum.accumulate(mag, axis=2)
    mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
    flat = np.sort(np.asarray(sc).ravel())
    med = int(flat[(flat.size - 1) // 2])
    srel = (np.clip(np.asarray(sc).astype(np.int32) - med, -3, 3) + 3)
    ctx = (srel[..., None] * 9 + pm.astype(np.int32)) * 9 + mx.astype(np.int32)
    return np.bincount((ctx * 16 + blk.astype(np.int32)).ravel(), minlength=567 * 16).reshape(567, 16)


def quantize(c):
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


def analyze_matrix(sym, sc):
    """Everything per matrix: fingerprint, transform checks, counts orig/ceil (weights base ctx, scales order-0)."""
    fp, even, bmax = fingerprint(sym, sc)
    tsym, tsc, _, _ = forward(sym, sc)
    isym, isc = inverse(tsym, tsc)
    fp['roundtrip_fail'] = np.array(int(not (np.array_equal(isym, sym) and np.array_equal(isc.astype(np.uint8), sc))))
    fp['max_lt6'] = np.array(int((bmax < 6).sum()))
    fp['scale_ovf'] = np.array(int((tsc > 254).sum()))
    fp['w_orig'] = ctx_counts(sym, sc)
    fp['w_ceil'] = ctx_counts(tsym, tsc.astype(np.int16))
    fp['s_orig'] = np.bincount(sc.ravel(), minlength=256)[:256]
    fp['s_ceil'] = np.bincount(tsc.ravel().astype(np.int64), minlength=257)[:257]
    return fp


def heldout(items, key, transform=lambda c: c):
    """items: list of dicts with 'split' (0/1) and count arrays under key. Returns (in-sample bits, held-out bits)."""
    a = transform(sum(np.asarray(e[key], np.int64) for e in items if e['split'] == 0))
    b = transform(sum(np.asarray(e[key], np.int64) for e in items if e['split'] == 1))
    c = a + b
    return cost_bits(c, quantize(c)), cost_bits(b, quantize(a)) + cost_bits(a, quantize(b))


def zmerge(c):
    c = np.array(c, np.int64, copy=True)
    c[..., 0] += c[..., 8]
    c[..., 8] = 0
    return c
