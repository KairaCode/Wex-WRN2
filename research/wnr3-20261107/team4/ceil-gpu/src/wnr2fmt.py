"""WNR2-style weight-stream formats for the GPU prototype (team2/gpu-proto).

Logic copied (not imported) from the team-1 scripts:
  quantize()      <- scratchpad\\wnr2\\mxanalyze.py:19-38 (wex_nr2_quantize rule)
  load_expert()   <- scratchpad\\wnr2\\mxcount.py:30-38
  encode_ref()    <- scratchpad\\wnr2\\rans_check.py:24-36 (scalar reference rANS encoder)
  context rule    <- scratchpad\\wnr2\\mxcount.py:45-67 / brief: ctx = (srel*9+pm)*9+mx

rANS: 32-bit state, L = 2^23, 14-bit CDFs, byte renormalisation, initial state(s) flushed as 4-byte
big-endian words at the start of each stream, final state(s) must equal L.

Variants (decode-step sequences; one stream = 2048 symbols = 64 blocks of 32):
  A    : 1 state, 567-row context table, 16-symbol alphabet (WNR2 today)
  A259 : same bytes as A, decoder uses compacted 259 reachable rows (row = srel*37 + tri(pm,mx))
  B1   : 2 states alternating per symbol (Giesen interleave), same contexts
  B2   : 2 states, state 0 = even blocks, state 1 = odd blocks, steps interleaved (2 independent chains)
  C    : 1 state, context + 'block max == 7' bit + block half; bit coded in-stream at each block start
         (binary CDF per srel); 1036 rows = 7 srel x 2 bit x 2 half x 37 tri
  D    : sign bypass: rANS codes magnitude (x&7) with an 8-symbol CDF per context (259 rows),
         sign bits stored raw as one u32 per 32-block (fixed-size plane, no lengths)
  DB2  : D + B2
  CB2  : C + B2
"""
import json, os
import numpy as np

P = 16384
PB = 14
L = 1 << 23
CHUNK = 2048
BPS = CHUNK // 32  # blocks per stream


# ---------------------------------------------------------------- copied helpers
def quantize(c):
    """c: int64 [..., n] counts -> int64 freq [..., n], each row sums to P, every freq >= 1."""
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
        out[t['name']] = (raw[off:off + n], t['shape'], t['dtype'])
        off += n
    assert off == raw.size, (path, off, raw.size)
    return out


def encode_ref(syms, ctxs, freq, cum, x0=L):
    """Scalar reference (rans_check.py:24-36). freq/cum int64 [rows, n]/[rows, n+1]."""
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
    out.reverse()
    return bytes(out)


# ---------------------------------------------------------------- models
def tri(pm, mx):
    pm = np.asarray(pm, np.int64); mx = np.asarray(mx, np.int64)
    return np.where(pm == 8, 36, mx * (mx + 1) // 2 + pm)


def ctx567_to_259():
    """index map: for each 567-row ctx, its 259-row index (or -1 if unreachable)."""
    m = np.full(567, -1, np.int64)
    for srel in range(7):
        for pm in range(9):
            for mx in range(9):
                ok = (pm == 8 and mx == 8) or (pm < 8 and mx < 8 and pm <= mx)
                if ok:
                    m[(srel * 9 + pm) * 9 + mx] = srel * 37 + int(tri(pm, mx))
    return m


def build_models(counts_dir, sample_dir):
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    wb = None; bmp = None; side = None
    for e in man['experts']:
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        a, b, c = z['w_base'].astype(np.int64), z['w_bmpos'].astype(np.int64), z['bm_side'].astype(np.int64)
        wb = a if wb is None else wb + a
        bmp = b if bmp is None else bmp + b
        side = c if side is None else side + c
    models = {}
    # A: 567 rows x 16
    fA = quantize(wb)  # [3,567,16]
    models['A'] = {'rows': 567, 'alpha': 16, 'freq': fA}
    # A259: same frequencies, compacted rows; unreachable rows must have zero counts
    mp = ctx567_to_259()
    assert (wb[:, mp < 0].sum() == 0), 'unreachable 567-row with counts'
    f259 = np.full((3, 259, 16), P // 16, np.int64)
    f259[:, mp[mp >= 0]] = fA[:, mp >= 0]
    models['A259'] = {'rows': 259, 'alpha': 16, 'freq': f259}
    # D: magnitude-only, 8 symbols, 259 rows
    mag = wb[..., :8] + wb[..., 8:]
    c259 = np.zeros((3, 259, 8), np.int64)
    c259[:, mp[mp >= 0]] = mag[:, mp >= 0]
    models['D'] = {'rows': 259, 'alpha': 8, 'freq': quantize(c259), 'counts': c259}
    # C: base x (bm==7) x half, 1036 rows; plus binary side model per srel
    b5 = bmp.reshape(3, 567, 8, 4, 16)
    cC = np.zeros((3, 1036, 16), np.int64)
    for base in range(567):
        if mp[base] < 0:
            assert b5[:, base].sum() == 0
            continue
        srel = base // 81; t = mp[base] - srel * 37
        for bm in range(8):
            for q in range(4):
                r = ((srel * 2 + (bm == 7)) * 2 + (q >= 2)) * 37 + t
                cC[:, r] += b5[:, base, bm, q]
    fC = quantize(cC)
    s2 = np.stack([side[..., :7].sum(-1), side[..., 7]], -1)  # [3,7,2] (max!=7, max==7)
    fbit = quantize(s2)
    models['C'] = {'rows': 1036, 'alpha': 16, 'freq': fC, 'bitfreq': fbit, 'counts': cC, 'bitcounts': s2}
    models['_counts'] = {'A': wb, 'D': c259, 'C': cC, 'Cbit': s2}
    return models


def cost_bits(counts, freq):
    return float((np.asarray(counts, np.float64) * -np.log2(np.asarray(freq, np.float64) / P)).sum())


# ---------------------------------------------------------------- per-matrix symbol/context data
def matrix_data(wb, sc):
    """wb: uint8 [R, C/2] packed E2M1 (low nibble = even column); sc uint8 [R, C/32]."""
    R = wb.shape[0]
    C = wb.shape[1] * 2
    nb = C // 32
    sym = np.empty((R, C), np.uint8)
    sym[:, 0::2] = wb & 15
    sym[:, 1::2] = wb >> 4
    blk = sym.reshape(-1, 32)  # [nblocks, 32] in row-major order
    mag = (blk & 7).astype(np.int64)
    pm = np.empty_like(mag); pm[:, 0] = 8; pm[:, 1:] = mag[:, :-1]
    cm = np.maximum.accumulate(mag, axis=1)
    mx = np.empty_like(mag); mx[:, 0] = 8; mx[:, 1:] = cm[:, :-1]
    s16 = sc.astype(np.int16).ravel()
    med = int(np.sort(s16)[(s16.size - 1) // 2])
    srel = (np.clip(s16 - med, -3, 3) + 3).astype(np.int64)  # [nblocks]
    bm7 = (cm[:, -1] == 7).astype(np.int64)
    nbk = blk.shape[0]
    assert nbk % BPS == 0
    return {'blk': blk.astype(np.int64), 'mag': mag, 'pm': pm, 'mx': mx, 'srel': srel, 'bm7': bm7,
            'med': med, 'nblocks': nbk, 'nstreams': nbk // BPS, 'scales': sc.ravel().copy(),
            'raw': wb.ravel().copy()}


def variant_steps(md, var, mtype, models):
    """Return (F, ST, sid, K, extra) with F/ST int64 [S, T] in decode order."""
    S = md['nstreams']
    srel = md['srel'][:, None]
    pm, mx = md['pm'], md['mx']
    half = (np.arange(32) >= 16)[None, :].astype(np.int64)
    t37 = tri(pm, mx)
    extra = {}
    if var in ('A', 'B1', 'B2'):
        rows = (srel * 9 + pm) * 9 + mx
        sy = md['blk']; fr = models['A']['freq'][mtype]
    elif var == 'A259':
        rows = srel * 37 + t37
        sy = md['blk']; fr = models['A259']['freq'][mtype]
    elif var in ('D', 'DB2'):
        rows = srel * 37 + t37
        sy = md['mag']; fr = models['D']['freq'][mtype]
        sg = (md['blk'] >> 3).astype(np.uint32)
        extra['signs'] = (sg << np.arange(32, dtype=np.uint32)[None, :]).sum(1, dtype=np.uint64).astype(np.uint32)
    elif var in ('C', 'CB2'):
        rows = ((srel * 2 + md['bm7'][:, None]) * 2 + half) * 37 + t37
        sy = md['blk']; fr = models['C']['freq'][mtype]
    else:
        raise ValueError(var)
    cum = np.zeros((fr.shape[0], fr.shape[1] + 1), np.int64); cum[:, 1:] = np.cumsum(fr, 1)
    Fb = fr[rows, sy]   # [nblocks, 32]
    Sb = cum[rows, sy]
    if var in ('C', 'CB2'):
        fb = models['C']['bitfreq'][mtype]  # [7,2]
        cb = np.zeros((7, 3), np.int64); cb[:, 1:] = np.cumsum(fb, 1)
        bf = fb[md['srel'], md['bm7']][:, None]; bs = cb[md['srel'], md['bm7']][:, None]
        Fb = np.concatenate([bf, Fb], 1); Sb = np.concatenate([bs, Sb], 1)  # [nblocks, 33]
    W = Fb.shape[1]
    Fs = Fb.reshape(S, BPS, W); Ss = Sb.reshape(S, BPS, W)
    if var in ('A', 'A259', 'C', 'D'):
        F = Fs.reshape(S, -1); ST = Ss.reshape(S, -1); sid = np.zeros(F.shape[1], np.int64); K = 1
    elif var == 'B1':
        F = Fs.reshape(S, -1); ST = Ss.reshape(S, -1); sid = np.arange(F.shape[1]) & 1; K = 2
    else:  # B2 / DB2 / CB2: pair p -> blocks 2p (state 0), 2p+1 (state 1), steps interleaved
        F = Fs.reshape(S, BPS // 2, 2, W).transpose(0, 1, 3, 2).reshape(S, -1)
        ST = Ss.reshape(S, BPS // 2, 2, W).transpose(0, 1, 3, 2).reshape(S, -1)
        sid = np.arange(F.shape[1]) & 1; K = 2
    return F, ST, sid, K, extra


def rans_encode_vec(F, ST, sid, K):
    """Vectorised over streams. Returns (payload uint8, lengths int64 [S])."""
    S, T = F.shape
    cap = 2 * T + 4 * K + 8
    out = np.zeros(S * cap, np.uint8)
    rowend = np.arange(S, dtype=np.int64) * cap + (cap - 1)   # next write position (moves backwards)
    X = np.full((K, S), L, np.int64)
    Ft = np.ascontiguousarray(F.T)    # [T, S]
    STt = np.ascontiguousarray(ST.T)
    for t in range(T - 1, -1, -1):
        k = sid[t]
        x = X[k]
        f = Ft[t].astype(np.int64)
        xmax = f << 17   # ((L >> 14) << 8) * f
        for _ in range(2):
            m = x >= xmax
            idx = np.flatnonzero(m)
            if idx.size:
                out[rowend[idx]] = (x[idx] & 255).astype(np.uint8)
                rowend[idx] -= 1
                x[idx] >>= 8
        q, r = np.divmod(x, f)
        X[k] = (q << PB) + r + STt[t]
    for k in range(K - 1, -1, -1):
        for sh in (0, 8, 16, 24):
            out[rowend] = ((X[k] >> sh) & 255).astype(np.uint8)
            rowend -= 1
    ptr = (np.arange(S, dtype=np.int64) * cap + (cap - 1)) - rowend
    mask = (np.arange(cap)[None, :] >= (cap - ptr)[:, None]).ravel()
    return out[mask], ptr


def table_u16(model, mtype):
    """Decoder rows: cum[1..alpha] as u16 per row ([rows, alpha]); cum[alpha] = 16384."""
    fr = model['freq'][mtype]
    return np.cumsum(fr, 1).astype(np.uint16)
