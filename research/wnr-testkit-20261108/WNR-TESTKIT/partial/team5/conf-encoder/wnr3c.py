"""team5 / conf-encoder: WNR3 DRAFT 0.2.1 v1 default profile (CEIL|LENC|HASH, anchor rule g0), written from the spec text
docs/WNR3-DRAFT-v2.1.md in the kit (cited as S:<line>) and DRAFT 0.1 (WNR3-DRAFT.md, cited D1:<line>)
where 0.2.1 defers to it.

Copied from team 4 ceil-ref wnr_common.py (cited CR:<line>): quantize() (= wex_nr2_quantize), the lock-step rANS encoder
rans_encode_lockstep, the lock-step decode loops (modified: they return the final state / pointer and do no hashing),
fit_alpha / pred_from_q (S:281 cites them as the normative predictor), load_expert.
Everything else (WNM3 blob, 259-row index, K=2 sets, 64-B header, header_check, LENC layout with base offsets,
per-matrix gate / skip-255 / ceil_mask, anchor0 check, H32 hash over ident || coded bytes) is new and follows the spec.
Limitations of this reference: Cw = 2048 and Cs = 512 only (V4.1 geometry), layout 0 only, no CZ / TAIL / DIGEST /
SIGNPLANE (rejected fail-closed by the decoder, S:270).
"""
import hashlib
import json
import os
import struct

import numpy as np

P, PB, L = 16384, 14, 1 << 23
M32 = 0xFFFFFFFF
REC_MAGIC, MODEL_MAGIC, DESC_MAGIC = 0x33524E57, 0x334D4E57, 0x33444E57      # S:141
SEED, HMASK = 0x33524E57, 0x7FFFFF                                          # S:147-148
GROUP, NBINS, DBC = 32, 25, 12                                              # S:146, 151
F_RES0, F_CZ, F_LENC, F_HASH, F_CEIL, F_TAIL, F_SIGN, F_DIGEST = 1, 2, 4, 8, 16, 32, 64, 128   # S:243
FULLCTX, ROWS, ALPH = 567, 259, 16
CW, CS = 2048, 512
BPS = CW // 32
ALIGN = 4096
CODEC_NR3 = 5
HDR_FLAGS_NR3 = 37                                                          # S:160 SCALE_TILES|DEDUP|NEURAL_RANS3
INDEX_SHA = 'e0aebac7d1b9b257737b92d1065bde74243f7f8f617a1619b077f6e6639d0738'   # S:194
U32 = np.uint32


class Invalid(Exception):
    """Decoder status -1 (destination untouched)."""


def pad4(n):
    return -(-n // 4) * 4


def align4096(n):
    return -(-n // ALIGN) * ALIGN


# ------------------------------------------------------------------------------------------------ H32 (D1:273)
def _rotl(x, r):
    return ((x << r) | (x >> (32 - r))) & M32


def h32(data, seed):
    c1, c2 = 0xCC9E2D51, 0x1B873593
    h = seed & M32
    n = len(data)
    nb = n // 4
    for i in range(nb):
        k = struct.unpack_from('<I', data, 4 * i)[0]
        k = (k * c1) & M32
        k = _rotl(k, 15)
        k = (k * c2) & M32
        h ^= k
        h = _rotl(h, 13)
        h = (h * 5 + 0xE6546B64) & M32
    t = data[4 * nb:]
    k = 0
    if len(t) >= 3:
        k ^= t[2] << 16
    if len(t) >= 2:
        k ^= t[1] << 8
    if len(t) >= 1:
        k ^= t[0]
        k = (k * c1) & M32
        k = _rotl(k, 15)
        k = (k * c2) & M32
        h ^= k
    h ^= n & M32
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & M32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & M32
    h ^= h >> 16
    return h


def smhasher_verification():
    key = bytearray(256)
    out = bytearray()
    for i in range(256):
        key[i] = i
        out += struct.pack('<I', h32(bytes(key[:i]), 256 - i))
    return h32(bytes(out), 0)


def _vrotl(x, r):
    return (x << U32(r)) | (x >> U32(32 - r))


def h32_lanes(data, seed):
    """Vectorised H32 over equal-length lanes: data uint8 [S, n] -> uint32 [S]."""
    S, n = data.shape
    c1, c2 = U32(0xCC9E2D51), U32(0x1B873593)
    h = np.full(S, seed, U32)
    nb = n // 4
    if nb:
        w = np.ascontiguousarray(data[:, :4 * nb]).view('<u4').astype(U32)
        for i in range(nb):
            k = w[:, i] * c1
            k = _vrotl(k, 15) * c2
            h ^= k
            h = _vrotl(h, 13) * U32(5) + U32(0xE6546B64)
    t = n - 4 * nb
    if t:
        k = np.zeros(S, U32)
        for j in range(t - 1, -1, -1):
            k ^= data[:, 4 * nb + j].astype(U32) << U32(8 * j)
        k = _vrotl(k * c1, 15) * c2
        h ^= k
    h ^= U32(n & M32)
    h ^= h >> U32(16)
    h *= U32(0x85EBCA6B)
    h ^= h >> U32(13)
    h *= U32(0xC2B2AE35)
    h ^= h >> U32(16)
    return h


def stream_idents(model_id, owner_layer, owner_expert, g, ceil_bit, flags, m, set_id, anchor):
    """S:296-299: 16-B identity per stream (vectors g, ceil_bit, m, set_id, anchor of equal length)."""
    S = g.size
    ident = np.zeros((S, 4), '<u4')
    ident[:, 0] = model_id
    ident[:, 1] = (owner_layer << 16) | owner_expert
    ident[:, 2] = g.astype(np.int64) | (ceil_bit.astype(np.int64) << 31)
    ident[:, 3] = (flags & 0xFF) | (m.astype(np.int64) << 8) | (set_id.astype(np.int64) << 16) | (anchor.astype(np.int64) << 24)
    return ident.view(np.uint8).reshape(S, 16)


def stream_h32(ident_u8, coded_u8):
    return h32_lanes(np.concatenate([ident_u8, coded_u8], 1), SEED)


# ------------------------------------------------------------------------------------------------ model
def canonical_index():
    """D1:143-149, S:194: full = (srel*9 + pm)*9 + mx; row = srel*37 + (start ? 36 : tri(pm, mx))."""
    idx = np.full(FULLCTX, 0xFFFF, np.int64)
    for srel in range(7):
        for pm in range(9):
            for mx in range(9):
                full = (srel * 9 + pm) * 9 + mx
                if pm == 8 or mx == 8:
                    if pm == 8 and mx == 8:
                        idx[full] = srel * 37 + 36
                elif pm <= mx:
                    idx[full] = srel * 37 + mx * (mx + 1) // 2 + pm
    assert sorted(idx[idx != 0xFFFF].tolist()) == list(range(ROWS))
    assert hashlib.sha256(idx.astype('<u2').tobytes()).hexdigest() == INDEX_SHA
    return idx


INDEX = canonical_index()


def quantize(c):
    """Copy of CR:68-93 (wex_nr2_quantize, S:216)."""
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


def fnv1a32(b):
    h = 2166136261
    for x in b:
        h = ((h ^ x) * 16777619) & M32
    return h


def cdf_of(freq):
    c = np.zeros(freq.shape[:-1] + (freq.shape[-1] + 1,), np.int64)
    c[..., 1:] = np.cumsum(freq, -1)
    return c


SET_BYTES = pad4(8 + 3 * ROWS * (ALPH + 1) * 2 + 3 * 257 * 2)               # S:187
INDEX_OFF = 64
SETS_OFF = INDEX_OFF + pad4(2 * FULLCTX)                                    # S:186


def model_sizes(K, lenpred=True):
    lp = SETS_OFF + K * SET_BYTES if lenpred else 0
    total = (lp + 4 * K * 3 * NBINS) if lenpred else SETS_OFF + K * SET_BYTES
    return lp, total


def build_wnm3(sets, etab):
    """sets: list of {form, role, anchor0[3], wf [3,259,16], scf [3,256]}; etab int [K,3,25] (1/256 bit). S:171-206."""
    K = len(sets)
    lp_off, total = model_sizes(K)
    hdr = struct.pack('<I8H7I4I', MODEL_MAGIC, 1, PB, 0, K, FULLCTX, ROWS, 3, ALPH,
                      1, INDEX_OFF, SETS_OFF, SET_BYTES, lp_off, total, 1, 0, 0, 0, 0)
    assert len(hdr) == 64
    b = bytearray(hdr)
    b += INDEX.astype('<u2').tobytes()
    b += b'\0' * (SETS_OFF - len(b))
    for s in sets:
        sb = struct.pack('<BB3B3B', s['form'], s['role'], *[int(a) for a in s['anchor0']], 0, 0, 0)
        sb += cdf_of(np.asarray(s['wf'], np.int64)).astype('<u2').tobytes()
        sb += cdf_of(np.asarray(s['scf'], np.int64)).astype('<u2').tobytes()
        sb += b'\0' * (SET_BYTES - len(sb))
        b += sb
    assert len(b) == lp_off
    et = np.asarray(etab, np.int64)
    assert et.shape == (K, 3, NBINS) and (et >= 0).all() and (et <= 114688).all()
    b += et.astype('<u4').tobytes()
    assert len(b) == total
    return bytes(b)


def model_id_of(blob):
    return fnv1a32(blob) or 1                                               # S:228, D1:178


def parse_wnm3(blob, impl_layout2=False):
    """wex_nr3_model_open, S:210-211. Returns dict with sets (freqs) and etab; raises Invalid."""
    b = bytes(blob)
    if len(b) < 64:
        raise Invalid('model: short')
    (magic, ver, prec, layout, K, fc, rows, mats, alph, feat, ioff, soff, sbytes, lpoff, total, ikind,
     r0, r1, r2, r3) = struct.unpack_from('<I8H7I4I', b, 0)
    if magic != MODEL_MAGIC or ver != 1 or prec != 14:
        raise Invalid('model: magic/version/precision')
    if layout not in (0, 2) or (layout == 0 and alph != 16) or (layout == 2 and alph != 8):
        raise Invalid('model: layout/alphabet')
    if layout == 2 and not impl_layout2:
        raise Invalid('model: layout 2 not implemented (fail closed, S:270)')
    if fc != FULLCTX or rows != ROWS or mats != 3 or not (1 <= K <= 16) or ikind != 1:
        raise Invalid('model: fixed fields')
    if feat & ~1:
        raise Invalid('model: unknown feature bits')
    if (r0, r1, r2, r3) != (0, 0, 0, 0):
        raise Invalid('model: reserved')
    sbx = pad4(8 + 3 * rows * (alph + 1) * 2 + 3 * 257 * 2)
    lpx = soff + K * sbytes if feat & 1 else 0
    totx = (lpx + 4 * K * 3 * NBINS) if feat & 1 else soff + K * sbytes
    if ioff != 64 or soff != ioff + pad4(2 * fc) or sbytes != sbx or lpoff != lpx or total != totx or total != len(b):
        raise Invalid('model: offsets/sizes')
    idx = np.frombuffer(b, '<u2', fc, ioff).astype(np.int64)
    if not (idx == INDEX).all():
        raise Invalid('model: index not canonical')
    if any(b[ioff + 2 * fc:soff]):
        raise Invalid('model: index padding')
    sets = []
    nwc = 3 * rows * (alph + 1)
    for k in range(K):
        o = soff + k * sbytes
        form, role = b[o], b[o + 1]
        a0 = list(b[o + 2:o + 5])
        if form not in (0, 1) or role not in (0, 1) or any(b[o + 5:o + 8]):
            raise Invalid('model: set form/role/reserved')
        wc = np.frombuffer(b, '<u2', nwc, o + 8).astype(np.int64).reshape(3, rows, alph + 1)
        scc = np.frombuffer(b, '<u2', 3 * 257, o + 8 + 2 * nwc).astype(np.int64).reshape(3, 257)
        if any(b[o + 8 + 2 * nwc + 2 * 3 * 257:o + sbytes]):
            raise Invalid('model: set padding')
        for c in (wc, scc):
            if (c[..., 0] != 0).any() or (c[..., -1] != P).any() or (np.diff(c, axis=-1) <= 0).any():
                raise Invalid('model: CDF not strictly increasing 0..16384')
        sets.append({'form': form, 'role': role, 'anchor0': a0, 'wf': np.diff(wc, axis=-1), 'scf': np.diff(scc, axis=-1)})
    etab = None
    if feat & 1:
        etab = np.frombuffer(b, '<u4', K * 3 * NBINS, lpoff).astype(np.int64).reshape(K, 3, NBINS)
        if (etab > 114688).any():
            raise Invalid('model: etab bound')
    return {'K': K, 'layout': layout, 'features': feat, 'sets': sets, 'etab': etab, 'model_id': model_id_of(b),
            'total_bytes': total}


def wnd3_descriptor(blob):
    """D1:66-76 (S:158 keeps it normative): 56-B descriptor at header byte 64."""
    md = parse_wnm3(blob)
    summ = md['layout'] | (md['K'] << 8) | (md['features'] << 16)            # S:163
    d = struct.pack('<IIQII', DESC_MAGIC, len(blob), 4096, md['model_id'], summ) + hashlib.sha256(blob).digest()
    assert len(d) == 56
    return d, summ


class RecTables:
    """Per-record expanded tables: matrix m uses sets[set_id[m]] (S:208); 259 rows expanded to the 567 full contexts
    through the canonical index so the ceil-ref context arithmetic can be reused (unreachable contexts: uniform rows,
    never selected because pm <= mx always holds and start is pm = mx = 8)."""

    def __init__(self, model, set_ids):
        wf = np.full((3, FULLCTX, 16), P // 16, np.int64)
        scf = np.zeros((3, 256), np.int64)
        reach = INDEX != 0xFFFF
        for m in range(3):
            s = model['sets'][set_ids[m]]
            wf[m][reach] = s['wf'][m][INDEX[reach]]
            scf[m] = s['scf'][m]
        self.wf = wf.reshape(-1, 16)
        self.wc = np.zeros_like(self.wf)
        self.wc[:, 1:] = np.cumsum(self.wf, 1)[:, :-1]
        self.scf = scf
        self.scc = np.zeros_like(scf)
        self.scc[:, 1:] = np.cumsum(scf, 1)[:, :-1]
        self.wf16, self.wc16 = self.wf.ravel().astype(np.uint16), self.wc.ravel().astype(np.uint16)
        self.wf32, self.wc32 = self.wf.ravel().astype(np.int32), self.wc.ravel().astype(np.int32)
        self.scf16, self.scc16 = self.scf.ravel().astype(np.uint16), self.scc.ravel().astype(np.uint16)
        self.scf32, self.scc32 = self.scf.ravel().astype(np.int32), self.scc.ravel().astype(np.int32)
        R = self.wf.shape[0]
        self.wlut = np.repeat(np.tile(np.arange(16, dtype=np.uint8), R), self.wf.ravel())
        self.slut = np.repeat(np.tile(np.arange(256, dtype=np.uint8), 3), self.scf.ravel())
        self.wcost = -np.log2(self.wf / P)


# ------------------------------------------------------------------------------------------------ data
def load_expert(path, e):
    """Copy of CR:178-198 (file order w1.s w2.s w3.s w1.w w2.w w3.w)."""
    raw = np.fromfile(path, dtype=np.uint8)
    t, off = {}, 0
    for d in e['tensors']:
        t[d['name']] = (raw[off:off + d['bytes']], tuple(d['shape']))
        off += d['bytes']
    assert off == raw.size
    mats = []
    for w in ('w1', 'w2', 'w3'):
        wb, ws = t[w + '.weight']
        sb, ss = t[w + '.scale']
        R, Cb = ws
        C = 2 * Cb
        assert ss == (R, C // 32)
        w2 = wb.reshape(R, Cb)
        sym = np.empty((R, C), np.uint8)
        sym[:, 0::2] = w2 & 15
        sym[:, 1::2] = w2 >> 4
        mats.append({'sym': sym.ravel(), 'sc': np.ascontiguousarray(sb), 'R': R, 'C': C})
    return raw, mats


def densest_pair(sc):
    h = np.bincount(sc, minlength=256)[:256]
    return int((h[:-1] + h[1:]).argmax())                                    # S:351 argmax_v h[v]+h[v+1], first on ties


# ------------------------------------------------------------------------------------------------ ceil-form (S:309-330)
HALF = np.array([0, 255, 1, 255, 2, 3, 4, 5], np.uint8)
DBL = np.array([0, 2, 4, 5, 6, 7, 255, 255], np.uint8)


def block_stats(sym, sc):
    blk = sym.reshape(-1, 32)
    mag = blk & 7
    M = mag.max(1)
    has13 = ((mag == 1) | (mag == 3)).any(1)
    even = (M == 7) & ~has13
    return blk, M, even


def gate_and_T(sym, sc):
    """Per-matrix gate 'no block with M == 5' and T with skip-255 (S:313-321). Returns (eligible, tsym, tsc, stats)."""
    blk, M, even = block_stats(sym, sc)
    max5 = int((M == 5).sum())
    st = {'blocks': int(M.size), 'max5_blocks': max5, 'even_blocks': int(even.sum()),
          'even_share': float(even.mean()), 'blocks_max_below5': int((M < 5).sum()),
          'even_scale255_skipped': int((even & (sc == 255)).sum())}
    if max5:
        return False, sym, sc, st
    tr = even & (sc <= 254)
    tblk = np.where(tr[:, None], (blk & 8) | HALF[blk & 7], blk).astype(np.uint8)
    tsc = (sc.astype(np.int16) + tr).astype(np.uint8)
    st['transformed_blocks'] = int(tr.sum())
    return True, tblk.ravel(), tsc, st


def T_inverse(csym, cs):
    """S:323-331: M' == 5 -> D and cs - 1 (cs == 0 invalid); M' in 0..4, 6, 7 pass unchanged."""
    blk = csym.reshape(-1, 32)
    Mp = (blk & 7).max(1)
    tr = Mp == 5
    if (cs[tr] == 0).any():
        raise Invalid('ceil: transformed block with coded scale 0')
    ob = np.where(tr[:, None], (blk & 8) | DBL[blk & 7], blk).astype(np.uint8)
    return ob.ravel(), (cs.astype(np.int16) - tr).astype(np.uint8)


def contexts(sym, cs, anchor):
    """Full contexts (S:366): at block start srel = clamp(cs - anchor, -3, 3) + 3, pm = mx = 8; then pm = x & 7,
    mx = running max. Returns (full [nblk,32], dbin [nblk])."""
    blk = sym.reshape(-1, 32)
    mag = (blk & 7).astype(np.int16)
    cm = np.maximum.accumulate(mag, axis=1)
    pm = np.empty_like(mag)
    pm[:, 0] = 8
    pm[:, 1:] = mag[:, :-1]
    mx = np.empty_like(mag)
    mx[:, 0] = 8
    mx[:, 1:] = cm[:, :-1]
    d = cs.astype(np.int16) - anchor
    srel = np.clip(d, -3, 3) + 3
    dbin = np.clip(d, -DBC, DBC) + DBC
    return (srel[:, None] * 9 + pm) * 9 + mx, dbin, srel


# ------------------------------------------------------------------------------------------------ rANS (copied CR:327-395)
def rans_encode_lockstep(FT, CT, x0):
    n, S = FT.shape
    x = np.asarray(x0, np.int64)
    assert ((x >= L) & (x < (L << 8))).all()
    x = x.astype(np.int32)
    W = 2 * n + 4
    assert W * S < 2 ** 31
    buf = np.zeros(W * S, np.uint8)
    rows = np.arange(S, dtype=np.int32)
    wpos = rows.copy()
    Sv = np.int32(S)
    for t in range(n - 1, -1, -1):
        f = FT[t].astype(np.int32)
        xmax = f << 17
        m = x >= xmax
        if m.any():
            buf[wpos] = x.astype(np.uint8)
            mi = m.astype(np.int32)
            wpos += mi * Sv
            x >>= mi << 3
            m = x >= xmax
            if m.any():
                buf[wpos] = x.astype(np.uint8)
                mi = m.astype(np.int32)
                wpos += mi * Sv
                x >>= mi << 3
        q = x // f
        x = (q << 14) + (x - q * f) + CT[t]
    for k in range(4):
        buf[wpos] = (x >> (8 * k)).astype(np.uint8)
        wpos += Sv
    lens = ((wpos - rows) // Sv).astype(np.int64)
    offs = np.zeros(S, np.int64)
    offs[1:] = np.cumsum(lens)[:-1]
    sid = np.repeat(rows.astype(np.int64), lens)
    j = np.arange(lens.sum(), dtype=np.int64) - offs[sid]
    j = (lens[sid] - 1 - j) * S + sid
    return buf[j], lens


def _read_state(pl, offs):
    o = offs
    x = (pl[o].astype(np.int64) << 24) | (pl[o + 1].astype(np.int64) << 16) | \
        (pl[o + 2].astype(np.int64) << 8) | pl[o + 3].astype(np.int64)
    ok = (x >= L) & (x < (L << 8))
    return np.where(ok, x, L).astype(np.int32), ok


def _renorm(x, ptr, pl):
    m = x < L
    if m.any():
        x = np.where(m, (x << 8) | pl[ptr], x)
        ptr += m
        m = x < L
        if m.any():
            x = np.where(m, (x << 8) | pl[ptr], x)
            ptr += m
    return x


def decode_scales(pl, offs, n, mat_of_stream, tab):
    """Returns (symT [n,S], x_final int64, ptr_final, init_ok)."""
    x, ok = _read_state(pl, offs)
    ptr = (offs + 4).astype(np.int64)
    lb = mat_of_stream.astype(np.int32) << 14
    fb = mat_of_stream.astype(np.int32) << 8
    LUT, F, C = tab.slut, tab.scf32, tab.scc32
    out = np.empty((n, offs.size), np.uint8)
    for t in range(n):
        slot = x & (P - 1)
        s = LUT[lb | slot]
        idx = fb | s
        x = F[idx] * (x >> PB) + slot - C[idx]
        x = _renorm(x, ptr, pl)
        out[t] = s
    return out, x.astype(np.int64), ptr, ok


def decode_weights(pl, offs, srelT, ctxoff, tab):
    """Weight streams of BPS blocks x 32 symbols; ctx = ctxoff + (srel*9 + pm)*9 + mx (start pm = mx = 8)."""
    S = offs.size
    x, ok = _read_state(pl, offs)
    ptr = (offs + 4).astype(np.int64)
    LUT, WF, WC = tab.wlut, tab.wf32, tab.wc32
    out = np.empty((BPS * 32, S), np.uint8)
    ctxoff = ctxoff.astype(np.int32)
    for b in range(BPS):
        rb = ctxoff + srelT[b].astype(np.int32) * np.int32(81)
        for j in range(32):
            ctx = rb + np.int32(80) if j == 0 else rb + pmx
            slot = x & (P - 1)
            s = LUT[(ctx << PB) | slot]
            idx = (ctx << 4) | s
            x = WF[idx] * (x >> PB) + slot - WC[idx]
            x = _renorm(x, ptr, pl)
            out[b * 32 + j] = s
            mag = (s & 7).astype(np.int32)
            mx = mag if j == 0 else np.maximum(mx, mag)
            pmx = mag * np.int32(9) + mx
    return out, x.astype(np.int64), ptr, ok


# ------------------------------------------------------------------------------------------------ LENC (S:276-283, D1:211-229)
def zigzag(r):
    r = np.asarray(r, np.int64)
    return np.where(r >= 0, 2 * r, -2 * r - 1)


def unzigzag(z):
    z = np.asarray(z, np.int64)
    return np.where(z & 1, -((z + 1) >> 1), z >> 1)


def pred_from_q(Q, alpha_q):                                                # CR:578-579 = S:281
    return 4 + ((np.asarray(Q, np.int64) * int(alpha_q) + (1 << 25)) >> 26)


def fit_alpha(Q, lens):                                                     # CR:582-599 (S:281 cites it)
    y = (lens - 4).astype(np.float64)
    q = Q.astype(np.float64) / (1 << 26)
    a = float((y * q).sum() / max(1e-9, (q * q).sum()))
    best = None
    a0 = int(round(a))
    for cand in range(max(0, a0 - 64), min(65535, a0 + 64) + 1, 4):
        cost = np.abs(lens - pred_from_q(Q, cand)).sum()
        if best is None or cost < best[0]:
            best = (cost, cand)
    a1 = best[1]
    for cand in range(max(0, a1 - 4), min(65535, a1 + 4) + 1):
        cost = np.abs(lens - pred_from_q(Q, cand)).sum()
        if cost < best[0]:
            best = (cost, cand)
    return best[1]


def pack_part(lens_per_m, preds_per_m):
    """One LENC part: u32 base_off[ng] | u8 width[ng] | pad4 | u32 words[sum(width)] (D1:215-223)."""
    base, widths, words = [], [], []
    off = 0
    for lens, preds in zip(lens_per_m, preds_per_m):
        z = zigzag(np.asarray(lens, np.int64) - np.asarray(preds, np.int64))
        for g0 in range(0, z.size, GROUP):
            zg = [int(v) for v in z[g0:g0 + GROUP]]
            mz = max(zg)
            w = mz.bit_length()
            if w > 17:
                raise ValueError('LENC width > 17')
            acc = 0
            for i, v in enumerate(zg):
                acc |= v << (i * w)
            words += [(acc >> (32 * q)) & M32 for q in range(w)]
            base.append(off)
            widths.append(w)
            off += int(np.sum(lens[g0:g0 + GROUP]))
    ng = len(base)
    out = struct.pack(f'<{ng}I', *base) + bytes(widths) + b'\0' * (pad4(ng) - ng) + struct.pack(f'<{len(words)}I', *words)
    return out, off


def unpack_part(buf, p0, counts, preds_fn, nsym, total_expected, what):
    """Parses one part starting at buf offset p0 for matrices with counts[m] streams. preds_fn(m) -> predictions of m.
    Returns (lens list per m, bytes consumed). Checks of D1:224 / S:278."""
    ngs = [-(-c // GROUP) for c in counts]
    ng = sum(ngs)
    if p0 + 4 * ng + pad4(ng) > len(buf):
        raise Invalid(f'LENC {what}: truncated')
    base = np.frombuffer(buf, '<u4', ng, p0).astype(np.int64)
    widths = np.frombuffer(buf, np.uint8, ng, p0 + 4 * ng).astype(np.int64)
    if any(buf[p0 + 4 * ng + ng:p0 + 4 * ng + pad4(ng)]):
        raise Invalid(f'LENC {what}: width padding')
    if (widths > 17).any():
        raise Invalid(f'LENC {what}: width > 17')
    wo = p0 + 4 * ng + pad4(ng)
    nwords = int(widths.sum())
    if wo + 4 * nwords > len(buf):
        raise Invalid(f'LENC {what}: truncated words')
    words = np.frombuffer(buf, '<u4', nwords, wo)
    out, gi, wi = [], 0, 0
    run = 0
    if base[0] != 0:
        raise Invalid(f'LENC {what}: base_off[0] != 0')
    for m, c in enumerate(counts):
        pr = preds_fn(m)
        lens = np.empty(c, np.int64)
        for g0 in range(0, c, GROUP):
            w = int(widths[gi])
            acc = 0
            for q in range(w):
                acc |= int(words[wi + q]) << (32 * q)
            wi += w
            mask = (1 << w) - 1
            zs = [(acc >> (i * w)) & mask for i in range(GROUP)]
            n = min(GROUP, c - g0)
            if any(zs[n:]):
                raise Invalid(f'LENC {what}: unused residual slot not 0')
            lens[g0:g0 + n] = pr[g0:g0 + n] + unzigzag(np.array(zs[:n], np.int64))
            if base[gi] != run:
                raise Invalid(f'LENC {what}: group continuity (group {gi})')
            run += int(lens[g0:g0 + n].sum())
            gi += 1
        if (lens < 4).any() or (lens > min(65535, 2 * nsym[m] + 8)).any():
            raise Invalid(f'LENC {what}: length out of [4, min(65535, 2n+8)]')
        out.append(lens)
    if run != total_expected:
        raise Invalid(f'LENC {what}: part total != {"SP" if what == "scale" else "WP"}')
    return out, wo + 4 * nwords - p0


# ------------------------------------------------------------------------------------------------ record header (S:237-260)
HDR_FMT = '<8I8B2H5I'
assert struct.calcsize(HDR_FMT) == 64


def pack_header(model_id, flags, LS, SP, WP, anchors, set_ids, ceil_mask, owner_layer, owner_expert,
                cz_count=0, record_digest=0, reserved39=0, reserved52=(0, 0)):
    h = bytearray(struct.pack(HDR_FMT, REC_MAGIC, model_id, flags, LS, SP, WP, CW, CS, *anchors, *set_ids, ceil_mask,
                              reserved39, owner_layer, owner_expert, cz_count, record_digest, *reserved52, 0))
    struct.pack_into('<I', h, 60, h32(bytes(h[:60]), SEED))
    return bytes(h)


def unpack_header(b):
    v = struct.unpack_from(HDR_FMT, b, 0)
    keys = ['magic', 'model_id', 'flags', 'LS', 'SP', 'WP', 'Cw', 'Cs', 'a0', 'a1', 'a2', 's0', 's1', 's2', 'ceil_mask',
            'res39', 'owner_layer', 'owner_expert', 'cz_count', 'record_digest', 'res52', 'res56', 'header_check']
    return dict(zip(keys, v))


# ------------------------------------------------------------------------------------------------ encoder (S:433-441)
def choose_and_transform(mats, model):
    """S:436 step 2: per matrix gate and even share; ceil set if gate passes, the model has a ceil set and even share >= 1%."""
    ceil_sets = [k for k, s in enumerate(model['sets']) if s['form'] == 1 and s['role'] == 0]
    plain_sets = [k for k, s in enumerate(model['sets']) if s['form'] == 0 and s['role'] == 0]
    syms, scs, set_ids, stats = [], [], [], []
    ceil_mask = 0
    for m, mt in enumerate(mats):
        ok, tsym, tsc, st = gate_and_T(mt['sym'], mt['sc'])
        use_ceil = ok and ceil_sets and st['even_share'] >= 0.01
        st['gate_pass'] = bool(ok)
        st['coded_plain_reason'] = None if use_ceil else ('gate: max-5 block' if not ok else
                                                          ('no ceil set' if not ceil_sets else 'even share < 1%'))
        if use_ceil:
            set_ids.append(ceil_sets[0])
            ceil_mask |= 1 << m
            syms.append(tsym)
            scs.append(tsc)
        else:
            if not plain_sets:
                raise ValueError('matrix needs a plain set but the model has none')
            set_ids.append(plain_sets[0])
            st['transformed_blocks'] = 0
            syms.append(mt['sym'])
            scs.append(mt['sc'])
        stats.append(st)
    return syms, scs, set_ids, ceil_mask, stats


def encode_record(mats, model, blob_model_id, owner_layer, owner_expert):
    """Returns (record bytes, build report). v1 default profile: HASH + LENC, CEIL iff ceil_mask != 0 (S:272)."""
    syms, scs, set_ids, ceil_mask, stats = choose_and_transform(mats, model)
    flags = F_LENC | F_HASH | (F_CEIL if ceil_mask else 0)
    anchors = [model['sets'][set_ids[m]]['anchor0'][m] for m in range(3)]       # g0 (S:351)
    for m in range(3):                                                         # encoder anchor rule (S:274)
        if anchors[m] != model['sets'][set_ids[m]]['anchor0'][m]:
            raise ValueError('anchor rule violated')
    tab = RecTables(model, set_ids)
    nb = [s.size for s in scs]
    assert all(mt['sym'].size % CW == 0 and s.size % CS == 0 for mt, s in zip(mats, scs))
    nws = [mats[m]['sym'].size // CW for m in range(3)]
    nss = [nb[m] // CS for m in range(3)]
    SS, WS = sum(nss), sum(nws)
    # ---- scale streams (S:363)
    scT = np.concatenate([scs[m].reshape(nss[m], CS) for m in range(3)]).T          # [512, SS]
    smat = np.repeat(np.arange(3), nss)
    sidx = (smat[None, :] << 8) | scT
    sF, sC = tab.scf16[sidx], tab.scc16[sidx]
    cbit = np.array([(ceil_mask >> m) & 1 for m in range(3)])
    sid_v, anc_v = np.array(set_ids), np.array(anchors)
    gS = np.arange(SS)
    idS = stream_idents(blob_model_id, owner_layer, owner_expert, gS, cbit[smat], flags, smat, sid_v[smat], anc_v[smat])
    hS = stream_h32(idS, np.ascontiguousarray(scT.T))
    x0s = L + (hS & U32(HMASK)).astype(np.int64)
    spay, slens = rans_encode_lockstep(sF, sC, x0s)
    # ---- weight streams (S:366)
    FT = np.empty((CW, WS), np.uint16)
    CT = np.empty((CW, WS), np.uint16)
    coded_bytes = np.empty((WS, CW // 2), np.uint8)
    dbins, col, est_w = [], 0, 0.0
    for m in range(3):
        full, dbin, _ = contexts(syms[m], scs[m], anchors[m])
        idx = ((full.astype(np.int64) + m * FULLCTX) << 4) | syms[m].reshape(-1, 32)
        n = nws[m]
        est_w += float(np.bincount(idx.ravel(), minlength=tab.wf.size) @ tab.wcost.ravel())
        FT[:, col:col + n] = tab.wf16[idx].reshape(n, CW).T
        CT[:, col:col + n] = tab.wc16[idx].reshape(n, CW).T
        s2 = syms[m].reshape(n, CW)
        coded_bytes[col:col + n] = s2[:, 0::2] | (s2[:, 1::2] << 4)
        dbins.append(dbin.reshape(n, BPS))
        col += n
    wmat = np.repeat(np.arange(3), nws)
    gW = SS + np.arange(WS)
    idW = stream_idents(blob_model_id, owner_layer, owner_expert, gW, cbit[wmat], flags, wmat, sid_v[wmat], anc_v[wmat])
    hW = stream_h32(idW, coded_bytes)
    x0w = L + (hW & U32(HMASK)).astype(np.int64)
    wpay, wlens = rans_encode_lockstep(FT, CT, x0w)
    del FT, CT
    # ---- LENC (S:276-282)
    sl_m = [slens[smat == m] for m in range(3)]
    wl_m = [wlens[wmat == m] for m in range(3)]
    sbase = [int(np.median(sl_m[m])) for m in range(3)]
    alphas, wpred = [], []
    for m in range(3):
        Q = model['etab'][set_ids[m]][m][dbins[m].astype(np.int64)].sum(1)
        a = fit_alpha(Q, wl_m[m])
        alphas.append(a)
        wpred.append(pred_from_q(Q, a))
    spart, stot = pack_part(sl_m, [np.full(c.size, sbase[m]) for m, c in enumerate(sl_m)])
    wpart, wtot = pack_part(wl_m, wpred)
    lsec = struct.pack('<6H', *sbase, *alphas) + spart + wpart
    SP, WP, LS = int(slens.sum()), int(wlens.sum()), len(lsec)
    assert stot == SP and wtot == WP
    hdr = pack_header(blob_model_id, flags, LS, SP, WP, anchors, set_ids, ceil_mask, owner_layer, owner_expert)
    used = 64 + LS + SP + WP
    total = align4096(used)
    rec = bytearray(total)
    rec[:64] = hdr
    rec[64:64 + LS] = lsec
    rec[64 + LS:64 + LS + SP] = spay.tobytes()
    rec[64 + LS + SP:used] = wpay.tobytes()
    report = {'flags': flags, 'entry_flags': CODEC_NR3 | (flags << 8), 'ceil_mask': ceil_mask, 'set_id': set_ids,
              'anchor': anchors, 'anchor0_of_selected_set': [model['sets'][set_ids[m]]['anchor0'][m] for m in range(3)],
              'densest_pair_coded': [densest_pair(scs[m]) for m in range(3)], 'matrix_stats': stats,
              'sizes': {'header': 64, 'LS': LS, 'LS_scale_part': len(spart), 'LS_weight_part': len(wpart), 'SP': SP,
                        'WP': WP, 'used': used, 'padding': total - used, 'total': total},
              'sbase': sbase, 'alpha': alphas, 'est_weight_bits': est_w,
              'header_check': struct.unpack_from('<I', hdr, 60)[0],
              'first_scale_stream_h32': int(hS[0]), 'first_weight_stream_h32': int(hW[0]),
              'x0_first_scale': int(x0s[0]), 'x0_first_weight': int(x0w[0])}
    return bytes(rec), report


# ------------------------------------------------------------------------------------------------ decoder (S:445-463)
def decode_record(rec, model, entry_flags, bind, shapes, header_flags=HDR_FLAGS_NR3, impl_cz=False, impl_tail=False):
    """CPU verify decoder (same author as the encoder: NOT an independent decoder for T13). Parses everything from the
    record bytes and the parsed WNM3 blob. Returns (slab bytes, info); raises Invalid (status -1, nothing written)."""
    rec = np.frombuffer(bytes(rec), np.uint8)
    if rec.size % ALIGN or rec.size < 64:
        raise Invalid('record size not 4096-aligned')
    # step 1: entry rules (S:161-162)
    if entry_flags & 0xFF != CODEC_NR3 or entry_flags >> 16:
        raise Invalid('entry flags: codec / bits 16-31')
    if entry_flags & 0x100:
        raise Invalid('entry flags: bit 8 reserved')
    if not (header_flags & 32):
        raise Invalid('container header flag 32 missing')
    # step 2: header
    hb = rec[:64].tobytes()
    H = unpack_header(hb)
    if H['magic'] != REC_MAGIC:
        raise Invalid('bad magic')
    if h32(hb[:60], SEED) != H['header_check']:
        raise Invalid('header_check mismatch')
    if H['model_id'] != model['model_id']:
        raise Invalid('model id mismatch')
    fl = H['flags']
    if fl >> 8 or fl != (entry_flags >> 8) & 0xFF:
        raise Invalid('flags != entry bits 8-15 / bits 8-31')
    cm = H['ceil_mask']
    sid = [H['s0'], H['s1'], H['s2']]
    anc = [H['a0'], H['a1'], H['a2']]
    if fl & F_RES0:
        raise Invalid('flag bit 0 reserved')
    if (fl & F_TAIL or fl & F_DIGEST) and not impl_tail:
        raise Invalid('TAIL/DIGEST not implemented (fail closed)')
    if fl & F_CZ and not impl_cz:
        raise Invalid('CZ not implemented (fail closed)')
    if (fl & F_HASH) and (fl & F_TAIL):
        raise Invalid('HASH and TAIL')
    if bool(fl & F_TAIL) != bool(fl & F_DIGEST):
        raise Invalid('TAIL <=> DIGEST')
    if bool(fl & F_SIGN) != (model['layout'] == 2):
        raise Invalid('SIGNPLANE <=> layout 2')
    if bool(fl & F_CEIL) != (cm != 0):
        raise Invalid('CEIL <=> ceil_mask != 0')
    if cm >> 3:
        raise Invalid('ceil_mask bits 3-7')
    if any(s >= model['K'] for s in sid):
        raise Invalid('set_id >= K')
    for m in range(3):
        if ((cm >> m) & 1) != model['sets'][sid[m]]['form']:
            raise Invalid(f'ceil_mask bit {m} != form of set {sid[m]}')
    if fl & F_LENC and not (model['features'] & 1):
        raise Invalid('LENC without model LENPRED')
    if not (fl & F_DIGEST) and H['record_digest']:
        raise Invalid('record_digest != 0 without DIGEST')
    if not (fl & F_CZ) and H['cz_count']:
        raise Invalid('cz_count != 0 without CZ')
    if H['Cw'] != CW or H['Cs'] != CS:
        raise Invalid('chunks not supported by this reference (2048:512 only)')
    if (H['owner_layer'], H['owner_expert']) != tuple(bind):
        raise Invalid('owner != bind')
    if H['res39'] or H['res52'] or H['res56']:
        raise Invalid('reserved header fields')
    LS, SP, WP = H['LS'], H['SP'], H['WP']
    used = 64 + LS + SP + WP
    if align4096(used) != rec.size:
        raise Invalid('total_bytes != align4096(64 + LS + SP + WP)')
    if rec[used:].any():
        raise Invalid('padding not zero')
    # step 3: geometry from the entry
    nsym = [R * C for R, C in shapes]
    nws = [n // CW for n in nsym]
    nss = [n // 32 // CS for n in nsym]
    SS, WS = sum(nss), sum(nws)
    buf = rec.tobytes()
    tab = RecTables(model, sid)
    # step 4: length section, scale part
    p = 64
    if fl & F_LENC:
        if LS < 12:
            raise Invalid('LENC: LS < 12')
        sbase = struct.unpack_from('<3H', buf, p)
        alpha = struct.unpack_from('<3H', buf, p + 6)
        lsend = 64 + LS
        sl_m, nsp = unpack_part(buf[:lsend], p + 12, nss, lambda m: np.full(nss[m], sbase[m], np.int64),
                                [CS] * 3, SP, 'scale')
        slens = np.concatenate(sl_m)
        pw = p + 12 + nsp
    else:
        if LS != pad4(2 * (SS + WS)):
            raise Invalid('u16 length table size')
        lt = np.frombuffer(buf, '<u2', SS + WS, p).astype(np.int64)
        slens, wl_fixed = lt[:SS], lt[SS:]
        if any(buf[p + 2 * (SS + WS):64 + LS]):
            raise Invalid('length table padding')
        if (slens < 4).any() or (slens > 2 * CS + 8).any() or slens.sum() != SP:
            raise Invalid('scale lengths')
    # step 5: scale streams -> coded scales
    sbeg, wbeg = 64 + LS, 64 + LS + SP
    pad = np.zeros(2 * CW + 64, np.uint8)
    pl_s = np.concatenate([rec[sbeg:wbeg], pad])
    soffs = np.zeros(SS, np.int64)
    soffs[1:] = np.cumsum(slens)[:-1]
    smat = np.repeat(np.arange(3), nss)
    scT, xs, ps, oks = decode_scales(pl_s, soffs, CS, smat, tab)
    cbit = np.array([(cm >> m) & 1 for m in range(3)])
    sid_v, anc_v = np.array(sid), np.array(anc)
    if fl & F_HASH:
        idS = stream_idents(H['model_id'], H['owner_layer'], H['owner_expert'], np.arange(SS), cbit[smat], fl, smat,
                            sid_v[smat], anc_v[smat])
        wantS = L + (stream_h32(idS, np.ascontiguousarray(scT.T)) & U32(HMASK)).astype(np.int64)
    else:
        wantS = np.full(SS, L, np.int64)
    badS = ~(oks & (ps == soffs + slens) & (xs == wantS))
    if badS.any():
        g = int(np.flatnonzero(badS)[0])
        raise Invalid(f'scale stream final-state/length/hash check failed: {int(badS.sum())} streams, first g={g}')
    cs = []
    a = 0
    for m in range(3):
        cs.append(np.ascontiguousarray(scT[:, a:a + nss[m]].T).ravel())
        a += nss[m]
    # step 6: weight lengths from coded scales
    srelT = np.empty((BPS, WS), np.int64)
    dbins = []
    col = 0
    for m in range(3):
        d = cs[m].astype(np.int16) - anc[m]
        srelT[:, col:col + nws[m]] = (np.clip(d, -3, 3) + 3).reshape(nws[m], BPS).T
        dbins.append((np.clip(d, -DBC, DBC) + DBC).reshape(nws[m], BPS))
        col += nws[m]
    if fl & F_LENC:
        def wp(m):
            Q = model['etab'][sid[m]][m][dbins[m].astype(np.int64)].sum(1)
            return pred_from_q(Q, alpha[m])
        wl_m, nwp = unpack_part(buf[:64 + LS], pw, nws, wp, [CW] * 3, WP, 'weight')
        if pw + nwp != 64 + LS:
            raise Invalid('LENC: LS != 12 + scale part + weight part')
        wlens = np.concatenate(wl_m)
    else:
        wlens = wl_fixed
        if (wlens < 4).any() or (wlens > 2 * CW + 8).any() or wlens.sum() != WP:
            raise Invalid('weight lengths')
    # step 7-8: weight streams, hash on coded words, then T^-1
    pl_w = np.concatenate([rec[wbeg:wbeg + WP], pad])
    woffs = np.zeros(WS, np.int64)
    woffs[1:] = np.cumsum(wlens)[:-1]
    wmat = np.repeat(np.arange(3), nws)
    wT, xw, pw_, okw = decode_weights(pl_w, woffs, srelT, wmat * FULLCTX, tab)
    cw = wT.T                                                                   # [WS, 2048]
    coded_bytes = np.ascontiguousarray(cw[:, 0::2] | (cw[:, 1::2] << 4))
    if fl & F_HASH:
        idW = stream_idents(H['model_id'], H['owner_layer'], H['owner_expert'], SS + np.arange(WS), cbit[wmat], fl,
                            wmat, sid_v[wmat], anc_v[wmat])
        wantW = L + (stream_h32(idW, coded_bytes) & U32(HMASK)).astype(np.int64)
    else:
        wantW = np.full(WS, L, np.int64)
    badW = ~(okw & (pw_ == woffs + wlens) & (xw == wantW))
    if badW.any():
        g = int(np.flatnonzero(badW)[0])
        raise Invalid(f'weight stream final-state/length/hash check failed: {int(badW.sum())} streams, first g={SS + g}')
    out_sc, out_w = [], []
    col = 0
    for m in range(3):
        sym = np.ascontiguousarray(cw[col:col + nws[m]]).ravel()
        osc = cs[m]
        if (cm >> m) & 1:
            sym, osc = T_inverse(sym, cs[m])
        out_sc.append(osc)
        out_w.append((sym[0::2] | (sym[1::2] << 4)).astype(np.uint8))
        col += nws[m]
    slab = np.concatenate(out_sc + out_w)
    return slab.tobytes(), {'flags': fl, 'ceil_mask': cm, 'set_id': sid, 'anchor': anc, 'SS': SS, 'WS': WS,
                            'wbeg': wbeg, 'woffs': woffs, 'wlens': wlens, 'srelT': srelT, 'wantW': wantW,
                            'coded_weights': cw, 'nws': nws, 'LS': LS}
