"""team2/audit shared code: 14-bit quantizer (wex_nr2_quantize as described in WNR2-REFERENCE.md:237),
WNR2-style rANS (L = 2^23, 14-bit, byte renorm, x_max = ((L>>14)<<8)*f, 4-B big-endian state first),
vectorized over many streams. Independent of the scratchpad wnr2 scripts.
"""
import numpy as np

P = 1 << 14
PB = 14
L = 1 << 23


def quantize(counts):
    """Largest-remainder quantizer: f = 1 + floor(c*(P-n)/T); leftover slots to largest remainders,
    ties to the lower symbol; empty rows uniform P//n."""
    c = np.asarray(counts, np.int64)
    n = c.shape[-1]
    flat = c.reshape(-1, n)
    T = flat.sum(1)
    Ts = np.where(T == 0, 1, T)[:, None]
    num = flat * (P - n)
    f = 1 + num // Ts
    r = num % Ts
    left = P - f.sum(1)
    sym = np.broadcast_to(np.arange(n), flat.shape)
    order = np.lexsort((sym, -r), axis=-1)            # by remainder desc, then symbol asc
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, sym, axis=-1)
    f = f + (rank < left[:, None])
    f[T == 0] = P // n
    if not ((f.sum(1) == P).all() and (f >= 1).all()):
        raise AssertionError('quantizer invariant')
    return f.reshape(c.shape)


def quantize_scalar(row):
    n = len(row)
    T = sum(int(v) for v in row)
    if T == 0:
        return [P // n] * n
    f = [1 + (int(v) * (P - n)) // T for v in row]
    r = [(int(v) * (P - n)) % T for v in row]
    left = P - sum(f)
    for i in sorted(range(n), key=lambda i: (-r[i], i))[:left]:
        f[i] += 1
    return f


def cost_bits(counts, freq):
    c = np.asarray(counts, np.float64)
    return float((c * (PB - np.log2(np.asarray(freq, np.float64)))).sum())


def ideal_bits(counts):
    c = np.asarray(counts, np.float64)
    T = c.sum(-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        lg = np.where(c > 0, np.log2(np.where(T > 0, T, 1) / np.where(c > 0, c, 1)), 0.0)
    return float((c * lg).sum())


def cum_of(freq):
    f = np.asarray(freq, np.int64)
    cum = np.zeros(f.shape[:-1] + (f.shape[-1] + 1,), np.int64)
    cum[..., 1:] = np.cumsum(f, -1)
    return cum


def encode_vec(F, C, x0=None, want_bytes=False):
    """F, C: int64 [N, n] per-symbol freq and cumulative start, in stream order.
    Encodes backwards. Returns (lengths [N] incl. 4-B state, final state [N], bytes or None)."""
    N, n = F.shape
    x = np.full(N, L, np.int64) if x0 is None else np.asarray(x0, np.int64).copy()
    cnt = np.zeros(N, np.int64)
    out = np.zeros((N, 2 * n + 8), np.uint8) if want_bytes else None
    ar = np.arange(N)
    for j in range(n - 1, -1, -1):
        f = F[:, j]
        xmax = f << 17  # ((L >> 14) << 8) * f
        for _ in range(3):
            m = x >= xmax
            if not m.any():
                break
            if want_bytes:
                rows = ar[m]
                out[rows, cnt[rows]] = (x[rows] & 255).astype(np.uint8)
            cnt += m
            x = np.where(m, x >> 8, x)
        x = ((x // f) << PB) + (x % f) + C[:, j]
    if (x < L).any() or (x >= (L << 8)).any():
        raise AssertionError('encoder state out of range')
    lengths = cnt + 4
    if not want_bytes:
        return lengths, x, None
    for k in range(4):  # flush little-endian, the whole buffer is then reversed -> big-endian state first
        out[ar, cnt + k] = ((x >> (8 * k)) & 255).astype(np.uint8)
    W = out.shape[1]
    idx = lengths[:, None] - 1 - np.arange(W)[None, :]
    rev = np.where(idx >= 0, out[ar[:, None], np.clip(idx, 0, None)], 0).astype(np.uint8)
    return lengths, x, rev


def decode_vec(buf, lens, n, srel, Ftab, Ctab, INV, mag=None):
    """buf uint8 [N, W] (zero padded), lens [N], srel int [N, n//32], tables for one matrix type:
    Ftab/Ctab [567, 16], INV [567, P] slot -> symbol. mag: optional [16] coded-symbol -> magnitude map
    (default s & 7) for permuted alphabets. Returns (valid [N], symbols [N, n], final x [N])."""
    magv = np.arange(16, dtype=np.int64) & 7 if mag is None else np.asarray(mag, np.int64)
    N, W = buf.shape
    ar = np.arange(N)
    b = buf.astype(np.int64)
    x = (b[:, 0] << 24) | (b[:, 1] << 16) | (b[:, 2] << 8) | b[:, 3]
    ok = (x >= L) & (x < (L << 8)) & (lens >= 4)
    ptr = np.full(N, 4, np.int64)
    over = np.zeros(N, bool)
    syms = np.zeros((N, n), np.uint8)
    pm = mx = None
    for i in range(n):
        if i % 32 == 0:
            pm = np.full(N, 8, np.int64)
            mx = np.full(N, 8, np.int64)
            sr = srel[:, i // 32].astype(np.int64)
        ctx = (sr * 9 + pm) * 9 + mx
        slot = x & (P - 1)
        s = INV[ctx, slot].astype(np.int64)
        x = Ftab[ctx, s] * (x >> PB) + slot - Ctab[ctx, s]
        for _ in range(3):
            need = x < L
            over |= need & (ptr >= lens)
            byte = b[ar, np.clip(ptr, 0, W - 1)]
            x = np.where(need, (x << 8) | byte, x)
            ptr += need
        syms[:, i] = s
        m = magv[s]
        pm = m
        mx = np.where(mx == 8, m, np.maximum(mx, m))
    valid = ok & ~over & (ptr == lens) & (x == L)
    return valid, syms, x


def inv_table(freq):
    """freq [567, 16] -> INV [567, P] uint8."""
    inv = np.empty((freq.shape[0], P), np.uint8)
    for r in range(freq.shape[0]):
        inv[r] = np.repeat(np.arange(freq.shape[1], dtype=np.uint8), freq[r])
    return inv


def scalar_encode(syms, ctxs, freq, cum, x0=L):
    x = x0
    out = bytearray()
    for s, c in zip(reversed(syms), reversed(ctxs)):
        f = int(freq[c][s]); st = int(cum[c][s])
        while x >= ((L >> PB) << 8) * f:
            out.append(x & 255); x >>= 8
        x = ((x // f) << PB) + (x % f) + st
    for k in range(4):
        out.append((x >> (8 * k)) & 255)
    out.reverse()
    return bytes(out)


def scalar_decode(buf, n, srel_blocks, freq, cum):
    x = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3]
    if not (L <= x < (L << 8)):
        return None
    pos = 4
    out = []
    pm = mx = 8
    for i in range(n):
        if i % 32 == 0:
            pm = mx = 8
        c = (int(srel_blocks[i // 32]) * 9 + pm) * 9 + mx
        slot = x & (P - 1)
        s = 0
        while cum[c][s + 1] <= slot:
            s += 1
        x = int(freq[c][s]) * (x >> PB) + slot - int(cum[c][s])
        while x < L:
            if pos >= len(buf):
                return None
            x = (x << 8) | buf[pos]; pos += 1
        out.append(s)
        pm = s & 7
        mx = pm if mx == 8 else max(mx, pm)
    if pos != len(buf) or x != L:
        return None
    return out
