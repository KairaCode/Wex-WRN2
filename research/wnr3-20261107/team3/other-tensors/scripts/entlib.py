"""Entropy helpers shared by the analyses (pure numpy)."""
import numpy as np
import lzma, zlib
try:
    from compression import zstd  # Python 3.14 stdlib
except Exception:  # pragma: no cover
    zstd = None


def bc(idx, n):
    return np.bincount(np.asarray(idx).ravel(), minlength=n)[:n].astype(np.int64)


def ent_bits(counts):
    """Ideal in-sample code length (bits) of a [..., n] count table (each leading index = one context)."""
    c = np.asarray(counts, np.float64).reshape(-1, np.asarray(counts).shape[-1])
    t = c.sum(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(c > 0, c / np.where(t == 0, 1, t), 1)
    return float(-(c * np.log2(p)).sum())


def xent_bits(test, train, alpha=0.5):
    """Code length (bits) of `test` counts under the add-alpha smoothed model of `train` counts."""
    tr = np.asarray(train, np.float64)
    tr = tr.reshape(-1, tr.shape[-1]) + alpha
    p = tr / tr.sum(1, keepdims=True)
    te = np.asarray(test, np.float64).reshape(p.shape)
    return float(-(te * np.log2(p)).sum())


def ctx_counts(ctx, sym, nctx, nsym):
    return bc(np.asarray(ctx, np.int64) * nsym + np.asarray(sym, np.int64), nctx * nsym).reshape(nctx, nsym)


def generic(buf):
    """Generic codec ratios (compressed / raw) on a bytes-like buffer."""
    b = bytes(buf)
    r = {'zlib9': len(zlib.compress(b, 9)) / len(b),
         'lzma6': len(lzma.compress(b, preset=6)) / len(b)}
    if zstd is not None:
        r['zstd19'] = len(zstd.compress(b, level=19)) / len(b)
        r['zstd3'] = len(zstd.compress(b, level=3)) / len(b)
    return r


def fp8_fields(b):
    b = np.asarray(b, np.uint8)
    return b >> 7, (b >> 3) & 15, b & 7


def fp8_decompose(b):
    """Order-0 decomposition of E4M3 bytes into sign / exponent / mantissa|exponent (bits per byte)."""
    s, e, m = fp8_fields(b)
    n = b.size
    hb = ent_bits(bc(b, 256)) / n
    hs = ent_bits(bc(s, 2)) / n
    he = ent_bits(bc(e, 16)) / n
    hm_e = ent_bits(ctx_counts(e, m, 16, 8)) / n
    hs_em = ent_bits(ctx_counts(e * 8 + m, s, 128, 2)) / n
    return {'H_byte': hb, 'H_sign': hs, 'H_exp': he, 'H_mant_given_exp': hm_e, 'H_sign_given_exp_mant': hs_em}


E4M3_ZERO_CODES = (0x00, 0x80)
