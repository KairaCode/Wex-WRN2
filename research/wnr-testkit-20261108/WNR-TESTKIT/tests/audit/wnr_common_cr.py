"""Vectorised NumPy reference encoder/decoder for WNR2 and WNR3/ceil candidates.

team4 / ceil-ref copy of team3/ref-impl/wnr_common.py (2026-11-07), extended with:
  F  (16) ceil-form transform: per 32-block, 'even' block (max magnitude 7, no magnitude 1 or 3) -> magnitudes
          halved (0,2,4,5,6,7 -> 0,1,2,3,4,5, sign bit 8 kept), its scale byte + 1. Decoder: block whose decoded
          max magnitude is 5 -> doubled, scale - 1. The encoder checks per record that no block has max < 6 and
          that no transformed scale reaches 255 (original 254/255 in an even block); otherwise the record is coded
          plain (flag F cleared, same model) -> 'fallback'.
  G       srel anchor = lower value of the densest adjacent scale pair (argmax h[v]+h[v+1], first on ties) of the
          (transformed) scales of the matrix, stored in the existing median byte. Decoder unchanged (it only reads
          the byte). Without G the byte is the lower median, as in WNR2.
  A  (1)  now a 3-valued side symbol per 32-block: bm3 = clip(block max magnitude - 5, 0, 2) (on ceil-form records
          the block max is exactly 5, 6 or 7; in a non-ceil record max <= 5 maps to 0, so the rule is total).
          3-symbol 14-bit CDF per (model or matrix type) x srel; weight context = base*6 + bm3*2 + half (3402 rows).
  T  (32) raw tail seeding: the last TAIL_W = 5 weight codes of each weight stream (4 bits each) and the last
          TAIL_S = 2 scale bytes of each scale stream are not rANS-coded; they are put in the initial encoder state
          x0 = L + sum(sym[n-k+j] << (b*j)). The decoder decodes n-k symbols and requires L <= x_final < L + 2^(b*k)
          (20 / 16 bits). Mutually exclusive with E (same start-state bits).
Unchanged from team 3: B (K models, k-means), C (canonical zero), D (coded length table), E (23-bit hash in x0).
quantize(): empty rows of a 3-symbol alphabet get 5462,5461,5461 (team 3's P//n would not sum to P for n = 3);
16- and 256-symbol rows are unchanged, so the WNR2 model is bit-identical to team 3's.
Header: team 3's 48-B WNR3 header; flags byte 27 now also carries F (16) and T (32).

Original team-3 description follows.

All streams of one kind in a record (2160 scale streams, 17280 weight streams) advance in lock-step:
one NumPy step per symbol position, per-stream byte buffers with write pointers. No per-symbol Python loop
over streams.

WNR2 (exactly as in wnr2-brief.md / WNR2-REFERENCE.md sections 5-7):
  rANS, 32-bit state, L = 2^23, 14-bit CDFs (quantised like wex_nr2_quantize), byte renormalisation,
  x_max = ((L >> 14) << 8) * freq, 4-B big-endian initial state, final state == L.
  Streams: 2048 weights / 512 scales, never crossing a matrix. Scales order-0 per matrix type,
  weights 567 contexts per matrix type: ctx = (srel*9 + pm)*9 + mx.
  Record: 32-B header {u32 'WNR2', model_id, scale_payload_bytes, weight_payload_bytes, weight_chunk,
  scale_chunk, u8 median[3], u8 0, u32 0}, u16 lengths (scales m0..m2 then weights m0..m2),
  scale payload, weight payload, zero padding to 4096.
"""
import json
import os
import struct
import time

import numpy as np

P = 16384
PB = 14
L = 1 << 23
W_CHUNK, S_CHUNK = 2048, 512
BPS = W_CHUNK // 32            # 32-blocks per weight stream (64)
NCTX = 567
NSUB = 6                       # A: bm3 (3) x half (2)
NACTX = NCTX * NSUB            # 3402
NSIDE = 3
NDB, DBC = 25, 12              # dbin = clamp(scale - anchor, -12, 12) + 12 (length predictor only)
MAT = ('w1', 'w2', 'w3')
WNR2_MAGIC, WNR3_MAGIC = 0x32524E57, 0x33524E57
WNM2_MAGIC, WNM3_MAGIC = 0x324D4E57, 0x334D4E57
FNV_OFF, FNV_PRIME = 2166136261, 16777619
GROUP = 32
FLAG_A, FLAG_C, FLAG_D, FLAG_E, FLAG_F, FLAG_T = 1, 2, 4, 8, 16, 32
TAIL_W, TAIL_S = 5, 2          # raw tail symbols in the start state (4-bit codes / 8-bit scales)
ALIGN = 4096
U32 = np.uint32


# ----------------------------------------------------------------------------------------------- model
def quantize(c):
    """Copy of mxanalyze.quantize = wex_nr2_quantize: 1 + floor(c*(P-n)/total), the remaining slots to the
    largest remainders (ties to the lower symbol), empty rows near-uniform (P//n)."""
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
    if P % n:
        # near-uniform empty rows (P//n leaves a remainder for n = 3): give it to the first symbols
        u = np.full(n, P // n, np.int64)
        u[:P - int(u.sum())] += 1
        freq[tot[:, 0] == 0] = u
    else:
        freq[tot[:, 0] == 0] = P // n
    assert (freq.sum(1) == P).all() and (freq >= 1).all()
    return freq.reshape(c.shape)


def fnv1a32(b):
    h = FNV_OFF
    for x in b:
        h = ((h ^ x) * FNV_PRIME) & 0xFFFFFFFF
    return h


def cum_excl(freq):
    c = np.zeros_like(freq)
    c[..., 1:] = np.cumsum(freq, -1)[..., :-1]
    return c


class Tables:
    """Decoder/encoder tables of one model set.
    wf [Rw,16] weight freqs; sf [Rs,3] side freqs (or None); scf [3,256] scale freqs; etab [3,25] int (1/256 bit)."""

    def __init__(self, wf, scf, sf=None, etab=None, K=0, nctx=NCTX):
        self.wf = np.ascontiguousarray(wf, np.int64)
        self.wc = cum_excl(self.wf)
        self.scf = np.ascontiguousarray(scf, np.int64)
        self.scc = cum_excl(self.scf)
        self.sf = None if sf is None else np.ascontiguousarray(sf, np.int64)
        if self.sf is not None:
            assert self.sf.shape[1] == NSIDE and (self.sf.sum(1) == P).all()
            self.sfc = cum_excl(self.sf)
            self.sf32, self.sfc32 = self.sf.ravel().astype(np.int32), self.sfc.ravel().astype(np.int32)
        self.etab = None if etab is None else np.asarray(etab, np.int64)
        self.K, self.nctx = K, nctx
        self.wf_flat, self.wc_flat = self.wf.ravel(), self.wc.ravel()
        self.scf_flat, self.scc_flat = self.scf.ravel(), self.scc.ravel()
        assert self.wf.max() < P and self.scf.max() < P and (self.sf is None or self.sf.max() < P)
        self.wf16, self.wc16 = self.wf_flat.astype(np.uint16), self.wc_flat.astype(np.uint16)
        self.wf32, self.wc32 = self.wf_flat.astype(np.int32), self.wc_flat.astype(np.int32)
        self.scf16, self.scc16 = self.scf_flat.astype(np.uint16), self.scc_flat.astype(np.uint16)
        self.scf32, self.scc32 = self.scf_flat.astype(np.int32), self.scc_flat.astype(np.int32)
        self._wlut = self._slut = None
        self.wcost = -np.log2(self.wf / P)
        self.sidecost = None if sf is None else -np.log2(self.sf / P)
        self.blob = self.serialize()
        self.model_id = fnv1a32(self.blob) or 1

    @property
    def wlut(self):
        if self._wlut is None:
            R = self.wf.shape[0]
            self._wlut = np.repeat(np.tile(np.arange(16, dtype=np.uint8), R), self.wf_flat)
            assert self._wlut.size == R * P
        return self._wlut

    @property
    def slut(self):
        if self._slut is None:
            self._slut = np.repeat(np.tile(np.arange(256, dtype=np.uint8), 3), self.scf_flat)
            assert self._slut.size == 3 * P
        return self._slut

    def serialize(self):
        def cdf(freq):
            c = np.zeros(freq.shape[:-1] + (freq.shape[-1] + 1,), np.int64)
            c[..., 1:] = np.cumsum(freq, -1)
            return c.astype('<u2').tobytes()
        if self.sf is None and self.etab is None and self.K == 0 and self.nctx == NCTX:
            # exact WNM2 blob: header, 3*567*17 u16 weight CDFs, 3*257 u16 scale CDFs = 59,408 B
            hdr = struct.pack('<8I', WNM2_MAGIC, 1, PB, NCTX, 3, 0, 0, 0)
            b = hdr + cdf(self.wf.reshape(3, NCTX, 16)) + cdf(self.scf)
            assert len(b) == 59408
            return b
        flags = (FLAG_A if self.sf is not None else 0)
        hdr = struct.pack('<8I', WNM3_MAGIC, 1, PB, self.nctx, self.wf.shape[0] // self.nctx, flags, self.K,
                          NSIDE if self.sf is not None else 0)
        b = hdr + cdf(self.wf) + (cdf(self.sf) if self.sf is not None else b'') + cdf(self.scf)
        if self.etab is not None:
            b += self.etab.astype('<u4').tobytes()
        return b


# ----------------------------------------------------------------------------------------------- data
def load_manifest(sample_dir):
    return json.load(open(os.path.join(sample_dir, 'manifest.json')))


def load_expert(sample_dir, e):
    """Returns raw bytes (uint8, file order = w1.s w2.s w3.s w1.w w2.w w3.w) and per-matrix symbols/scales."""
    raw = np.fromfile(os.path.join(sample_dir, e['file']), dtype=np.uint8)
    t, off = {}, 0
    for d in e['tensors']:
        t[d['name']] = (raw[off:off + d['bytes']], tuple(d['shape']))
        off += d['bytes']
    assert off == raw.size
    mats = []
    for w in MAT:
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


def lower_median(sc):
    k = (sc.size - 1) // 2
    return int(np.partition(sc, k)[k])


def anchor_pair(sc):
    """G: lower value of the densest adjacent pair of scale values (first on ties)."""
    h = np.bincount(sc, minlength=256)[:256]
    return int((h[:-1] + h[1:]).argmax())


# ---- F: ceil-form transform
HALF = np.array([0, 255, 1, 255, 2, 3, 4, 5], np.uint8)       # magnitude -> halved (255 = impossible in even block)
DOUBLE = np.array([0, 2, 4, 5, 6, 7, 255, 255], np.uint8)     # halved magnitude -> stored magnitude


def ceil_forward(sym, sc):
    """sym flat uint8 (row-major), sc flat uint8 (one per 32-block). Returns (tsym, tsc, info)."""
    blk = sym.reshape(-1, 32)
    mag = blk & 7
    bmax = mag.max(1)
    has13 = ((mag == 1) | (mag == 3)).any(1)
    even = (bmax == 7) & ~has13
    info = {'blocks': int(even.size), 'even': int(even.sum()), 'below6': int((bmax < 6).sum()),
            'overflow': int((sc[even] >= 254).sum())}
    tblk = np.where(even[:, None], (blk & 8) | HALF[mag], blk).astype(np.uint8)
    tsc = np.minimum(sc.astype(np.int16) + even, 255).astype(np.uint8)   # only used when info['overflow'] == 0
    return tblk.ravel(), tsc, info


def ceil_inverse(tsym, tsc):
    """Inverse of ceil_forward; raises ValueError on input no valid ceil-form record can produce."""
    blk = tsym.reshape(-1, 32)
    mag = blk & 7
    bmax = mag.max(1)
    if (bmax < 5).any():
        raise ValueError('ceil: decoded block max < 5')
    odd = bmax == 5
    if (tsc[odd] == 0).any():
        raise ValueError('ceil: scale 0 in a halved block')
    ob = np.where(odd[:, None], (blk & 8) | DOUBLE[mag], blk).astype(np.uint8)
    osc = (tsc.astype(np.int16) - odd).astype(np.uint8)
    return ob.ravel(), osc


def canon(sym):
    return np.where(sym == 8, np.uint8(0), sym)


def prepare(mats, f, c, g):
    """Coding-domain symbols, scales and anchors of a record. Returns (syms, scs, anchors, finfo).
    finfo['applied'] is False when the ceil transform was requested but is not bijective for this record."""
    syms = [canon(mt['sym']) if c else mt['sym'] for mt in mats]
    scs = [mt['sc'] for mt in mats]
    finfo = None
    if f:
        out = [ceil_forward(syms[m], scs[m]) for m in range(3)]
        finfo = {k: [o[2][k] for o in out] for k in ('blocks', 'even', 'below6', 'overflow')}
        ok = all(o[2]['below6'] == 0 and o[2]['overflow'] == 0 for o in out)
        finfo['applied'] = bool(ok)
        if ok:
            syms = [o[0] for o in out]
            scs = [o[1] for o in out]
    anchors = [anchor_pair(s) if g else lower_median(s) for s in scs]
    return syms, scs, anchors, finfo


def block_features(sym, sc, anchor=None):
    """Encoder-side contexts of one matrix. sym flat uint8 (row-major, 32-blocks), sc flat uint8 per block."""
    blk = sym.reshape(-1, 32)
    mag = (blk & 7).astype(np.int16)
    cm = np.maximum.accumulate(mag, axis=1)
    pm = np.empty_like(mag)
    pm[:, 0] = 8
    pm[:, 1:] = mag[:, :-1]
    mx = np.empty_like(mag)
    mx[:, 0] = 8
    mx[:, 1:] = cm[:, :-1]
    if anchor is None:
        anchor = lower_median(sc)
    d = sc.astype(np.int16) - anchor
    srel = (np.clip(d, -3, 3) + 3).astype(np.int16)
    dbin = (np.clip(d, -DBC, DBC) + DBC).astype(np.int16)
    base = (srel[:, None] * 9 + pm) * 9 + mx
    bm3 = np.clip(cm[:, -1] - 5, 0, 2).astype(np.int16)
    half = (np.arange(32) >= 16).astype(np.int16)
    actx = base * NSUB + bm3[:, None] * 2 + half[None, :]
    return {'blk': blk, 'base': base, 'actx': actx, 'srel': srel, 'dbin': dbin, 'bm3': bm3, 'med': anchor}


# ----------------------------------------------------------------------------------------------- hash
def lowbias32(h):
    h = h.astype(U32)
    h ^= h >> U32(16)
    h *= U32(0x7FEB352D)
    h ^= h >> U32(15)
    h *= U32(0x846CA68B)
    h ^= h >> U32(16)
    return h


def hash_seeds(layer, expert, model_id, kind, mat_idx, stream_idx, kvec):
    """FNV-1a over 24 LE bytes (layer, expert, model_id, kind | k<<8, matrix, stream index), vectorised."""
    S = stream_idx.size
    rec = np.zeros((S, 6), '<u4')
    rec[:, 0] = layer
    rec[:, 1] = expert
    rec[:, 2] = model_id
    rec[:, 3] = kind | (kvec.astype(np.int64) << 8)
    rec[:, 4] = mat_idx
    rec[:, 5] = stream_idx
    by = rec.view(np.uint8).reshape(S, 24)
    h = np.full(S, FNV_OFF, U32)
    for j in range(24):
        h = (h ^ by[:, j].astype(U32)) * U32(FNV_PRIME)
    return h


def hash_symbols(seed, symT):
    h = seed.copy()
    for t in range(symT.shape[0]):
        h = (h ^ symT[t].astype(U32)) * U32(FNV_PRIME)
    return lowbias32(h) & U32((1 << 23) - 1)


# ----------------------------------------------------------------------------------------------- rANS core
def rans_encode_lockstep(FT, CT, x0):
    """FT, CT: [n, S] freq / start of every coded symbol in stream (decode) order. x0: [S] initial states
    (= decoder final states). All S streams are encoded backwards in lock-step.
    Returns (payload uint8: streams concatenated in order, lens int64 [S])."""
    # int32 is exact here: x < 256L = 2^31, f <= P-1 so f<<17 < 2^31, (x//f)<<14 < 2^31.
    n, S = FT.shape
    x = np.asarray(x0, np.int64)
    assert ((x >= L) & (x < (L << 8))).all()
    x = x.astype(np.int32)
    W = 2 * n + 4
    assert W * S < 2 ** 31
    buf = np.zeros(W * S, np.uint8)
    rows = np.arange(S, dtype=np.int32)
    wpos = rows.copy()                      # = write pointer * S + stream
    Sv = np.int32(S)
    for t in range(n - 1, -1, -1):
        f = FT[t].astype(np.int32)
        xmax = f << 17                      # ((L >> 14) << 8) * f
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
    for k in range(4):                      # flush: emitted LE, the reversed stream starts with BE state
        buf[wpos] = (x >> (8 * k)).astype(np.uint8)
        wpos += Sv
    lens = ((wpos - rows) // Sv).astype(np.int64)
    offs = np.zeros(S, np.int32)
    offs[1:] = np.cumsum(lens)[:-1]
    sid = np.repeat(rows.astype(np.int32), lens)
    j = np.arange(lens.sum(), dtype=np.int32)
    j -= offs[sid]
    j = (lens.astype(np.int32)[sid] - 1 - j)
    j *= np.int32(S)
    j += sid
    del sid
    payload = buf[j]
    return payload, lens


def _read_state(pl, offs):
    """4-B big-endian initial states; returns (x int32 with bad lanes set to L, ok-mask of the range check)."""
    o = offs
    x = (pl[o].astype(np.int64) << 24) | (pl[o + 1].astype(np.int64) << 16) | \
        (pl[o + 2].astype(np.int64) << 8) | pl[o + 3].astype(np.int64)
    ok = (x >= L) & (x < (L << 8))
    return np.where(ok, x, L).astype(np.int32), ok


def _renorm(x, ptr, pl):
    # int32 exact: after a decode step x < 2^31; (x << 8) is only kept where x < L.
    m = x < L
    if m.any():
        x = np.where(m, (x << 8) | pl[ptr], x)
        ptr += m
        m = x < L
        if m.any():
            x = np.where(m, (x << 8) | pl[ptr], x)
            ptr += m
    return x


def _final_check(x, ptr, offs, lens, h, ntail, bits):
    """Final-state rule. ntail == 0: x == L (+ h23 with E). ntail > 0: L <= x < L + 2^(bits*ntail); returns tail value."""
    if ntail:
        v = x.astype(np.int64) - L
        okf = (ptr == offs + lens) & (v >= 0) & (v < (np.int64(1) << (bits * ntail)))
        return okf, np.where(okf, v, 0)
    want = np.full(x.size, L, np.int64) if h is None else L + (lowbias32(h) & U32((1 << 23) - 1)).astype(np.int64)
    return (ptr == offs + lens) & (x == want), None


def decode_scales_lockstep(pl, offs, lens, n, mat_of_stream, tab, hseed=None, ntail=0):
    """Order-0 scale streams; pl padded. Returns (symT [n,S] uint8, ok [S] bool)."""
    S = offs.size
    x, ok = _read_state(pl, offs)
    ok &= (lens >= 4)
    ptr = (offs + 4).astype(np.int32)
    lb = mat_of_stream.astype(np.int32) << 14
    fb = mat_of_stream.astype(np.int32) << 8
    LUT, F, C = tab.slut, tab.scf32, tab.scc32
    out = np.empty((n, S), np.uint8)
    h = None if hseed is None else hseed.copy()
    for t in range(n - ntail):
        slot = x & (P - 1)
        s = LUT[lb | slot]
        idx = fb | s
        x = F[idx] * (x >> PB) + slot - C[idx]
        x = _renorm(x, ptr, pl)
        out[t] = s
        if h is not None:
            h = (h ^ s.astype(U32)) * U32(FNV_PRIME)
    okf, v = _final_check(x, ptr, offs, lens, h, ntail, 8)
    if ntail:
        for j in range(ntail):
            out[n - ntail + j] = ((v >> (8 * j)) & 255).astype(np.uint8)
    return out, ok & okf


def decode_weights_lockstep(pl, offs, lens, srelT, ctxoff, sideoff, tab, side, hseed=None, ntail=0):
    """Weight streams of 64 blocks x 32 symbols (+1 3-valued side symbol per block when side=True).
    srelT [64,S] block srel; ctxoff [S] row offset of the stream's model; sideoff [S] side-row offset (model*7).
    Context rule (decoder side): base = (srel*9 + pm)*9 + mx with pm = mx = 8 at block start;
    with side: row = ctxoff + base*6 + bm3*2 + half. ntail: last ntail codes of the stream come from the final state."""
    S = offs.size
    x, ok = _read_state(pl, offs)
    ok &= (lens >= 4)
    ptr = (offs + 4).astype(np.int32)
    LUT, WF, WC = tab.wlut, tab.wf32, tab.wc32
    assert tab.wf.shape[0] * P < 2 ** 31
    out = np.empty((BPS * 32, S), np.uint8)
    h = None if hseed is None else hseed.copy()
    mul = NSUB if side else 1
    m9, m1 = np.int32(9 * mul), np.int32(mul)
    ctxoff = ctxoff.astype(np.int32)
    if side:
        SF, SFC = tab.sf32, tab.sfc32
        sideoff = sideoff.astype(np.int32)
    for b in range(BPS):
        sr = srelT[b].astype(np.int32)
        if side:
            r3 = (sideoff + sr) * np.int32(NSIDE)
            c1 = SF[r3]
            c2 = c1 + SF[r3 + 1]
            slot = x & (P - 1)
            si = (slot >= c1).astype(np.int32) + (slot >= c2).astype(np.int32)
            idx = r3 + si
            x = SF[idx] * (x >> PB) + slot - SFC[idx]
            x = _renorm(x, ptr, pl)
            if h is not None:
                h = (h ^ si.astype(U32)) * U32(FNV_PRIME)
            rb = ctxoff + sr * np.int32(81 * NSUB) + si * np.int32(2)
        else:
            rb = ctxoff + sr * np.int32(81)
        rb_half = rb + 1 if side else rb
        jend = 32 - ntail if b == BPS - 1 else 32
        for j in range(jend):
            if j == 0:
                ctx = rb + np.int32(80 * mul)
            else:
                ctx = (rb_half if j >= 16 else rb) + pmx
            slot = x & (P - 1)
            s = LUT[(ctx << PB) | slot]
            idx = (ctx << 4) | s
            x = WF[idx] * (x >> PB) + slot - WC[idx]
            x = _renorm(x, ptr, pl)
            out[b * 32 + j] = s
            mag = (s & 7).astype(np.int32)
            if j == 0:
                mx = mag
            else:
                mx = np.maximum(mx, mag)
            pmx = mag * m9 + mx * m1         # (pm*9 + mx)*mul for the next symbol
            if h is not None:
                h = (h ^ s.astype(U32)) * U32(FNV_PRIME)
    okf, v = _final_check(x, ptr, offs, lens, h, ntail, 4)
    if ntail:
        for j in range(ntail):
            out[BPS * 32 - ntail + j] = ((v >> (4 * j)) & 15).astype(np.uint8)
    return out, ok & okf


# ----------------------------------------------------------------------------------------------- length table (D)
def zigzag(r):
    r = np.asarray(r, np.int64)
    return (r << 1) ^ (r >> 63)


def unzigzag(z):
    z = np.asarray(z, np.int64)
    return (z >> 1) ^ -(z & 1)


def pack_groups(segments):
    """segments: list of int64 residual arrays. Returns bytes: u8 width per 32-group (all segments, in order),
    then all groups' zigzag values bit-packed LSB-first, padded to a 4-byte multiple."""
    widths, vals, wv = [], [], []
    for r in segments:
        z = zigzag(r)
        ng = -(-z.size // GROUP)
        zp = np.zeros(ng * GROUP, np.int64)
        zp[:z.size] = z
        gmax = zp.reshape(ng, GROUP).max(1)
        w = np.zeros(ng, np.int64)
        for k in range(1, 25):
            w += gmax >= (1 << (k - 1))
        assert (gmax < (np.int64(1) << w)).all() and (w <= 24).all()
        widths.append(w)
        vals.append(z)
        wv.append(np.repeat(w, GROUP)[:z.size])
    widths = np.concatenate(widths)
    z = np.concatenate(vals)
    wv = np.concatenate(wv)
    off = np.zeros(z.size, np.int64)
    off[1:] = np.cumsum(wv)[:-1]
    nbits = int(wv.sum())
    bits = np.zeros(nbits + 7, np.uint8)
    for k in range(int(wv.max()) if wv.size else 0):
        m = wv > k
        bits[off[m] + k] = ((z[m] >> k) & 1).astype(np.uint8)
    body = np.packbits(bits[:(nbits + 7) // 8 * 8], bitorder='little').tobytes()
    out = widths.astype(np.uint8).tobytes() + body
    out += b'\0' * (-len(out) % 4)
    return out, nbits


def unpack_groups(buf, sizes):
    """Inverse of pack_groups for segments of the given sizes. Returns (list of residual arrays, bytes used)."""
    ngs = [-(-n // GROUP) for n in sizes]
    ng = sum(ngs)
    widths = np.frombuffer(buf, np.uint8, ng, 0).astype(np.int64)
    wv = []
    gi = 0
    for n, g in zip(sizes, ngs):
        wv.append(np.repeat(widths[gi:gi + g], GROUP)[:n])
        gi += g
    wv = np.concatenate(wv)
    off = np.zeros(wv.size, np.int64)
    off[1:] = np.cumsum(wv)[:-1]
    nbits = int(wv.sum())
    nbytes = (nbits + 7) // 8
    bits = np.unpackbits(np.frombuffer(buf, np.uint8, nbytes, ng), bitorder='little')
    bits = np.concatenate([bits, np.zeros(8, np.uint8)])
    z = np.zeros(wv.size, np.int64)
    for k in range(int(wv.max()) if wv.size else 0):
        m = wv > k
        z[m] |= bits[off[m] + k].astype(np.int64) << k
    r = unzigzag(z)
    out, a = [], 0
    for n in sizes:
        out.append(r[a:a + n])
        a += n
    used = ng + nbytes
    used += -used % 4
    return out, used


def predict_wlen_q(dbin_streams, etab_m):
    """dbin_streams [nstreams, 64] -> Q (sum of etab, 1/256 bit units) per stream."""
    return etab_m[dbin_streams.astype(np.int64)].sum(1)


def pred_from_q(Q, alpha_q):
    return 4 + ((Q * int(alpha_q) + (1 << 25)) >> 26)


def fit_alpha(Q, lens):
    y = (lens - 4).astype(np.float64)
    q = Q.astype(np.float64) / (1 << 26)
    a = float((y * q).sum() / max(1e-9, (q * q).sum()))
    best = None
    a0 = int(round(a))
    for cand in range(max(0, a0 - 64), min(65535, a0 + 64) + 1, 4):
        r = lens - pred_from_q(Q, cand)
        cost = np.abs(r).sum()
        if best is None or cost < best[0]:
            best = (cost, cand)
    a1 = best[1]
    for cand in range(max(0, a1 - 4), min(65535, a1 + 4) + 1):
        r = lens - pred_from_q(Q, cand)
        cost = np.abs(r).sum()
        if cost < best[0]:
            best = (cost, cand)
    return best[1]


# ----------------------------------------------------------------------------------------------- record encoder
def align(n, a=ALIGN):
    return -(-n // a) * a


class Config:
    def __init__(self, name, a=False, K=0, c=False, d=False, e=False, f=False, g=False, t=False, wnr2=False):
        self.name, self.a, self.K, self.c, self.d, self.e, self.wnr2 = name, a, K, c, d, e, wnr2
        self.f, self.g, self.t = f, g, t
        if wnr2:
            assert not (a or K or c or d or e or f or g or t)
        assert not (e and t), 'E and T use the same start-state bits'
        assert not K or a, 'K models only with A (as team 3)'

    @property
    def flags(self):
        return ((FLAG_A if self.a else 0) | (FLAG_C if self.c else 0) | (FLAG_D if self.d else 0) |
                (FLAG_E if self.e else 0) | (FLAG_F if self.f else 0) | (FLAG_T if self.t else 0))

    def family(self):
        return 'wnr2' if not self.f else ('fg' if self.g else 'f')

    def model_key(self):
        return f'{self.family()}_a{int(self.a)}_K{self.K}_c{int(self.c)}'


def choose_models(feats, syms, tab, cfg):
    """Per-matrix model index (B): argmin of the exact code length under each of the K models."""
    ks, costs = [], []
    for m in range(3):
        f = feats[m]
        ctx = (f['actx'] if cfg.a else f['base']).astype(np.int64)
        nct = NACTX if cfg.a else NCTX
        cnt = np.bincount((ctx * 16 + syms[m].reshape(-1, 32)).ravel(), minlength=nct * 16).reshape(nct, 16)
        wc = tab.wcost.reshape(cfg.K, nct, 16)
        cost = (cnt[None].astype(np.float64) * wc).sum((1, 2))
        if cfg.a:
            sc = np.bincount(f['srel'].astype(np.int64) * NSIDE + f['bm3'], minlength=7 * NSIDE).reshape(7, NSIDE)
            cost += (sc[None].astype(np.float64) * tab.sidecost.reshape(cfg.K, 7, NSIDE)).sum((1, 2))
        ks.append(int(np.argmin(cost)))
        costs.append(cost.tolist())
    return ks, costs


def encode_record(mats, tab, cfg, layer, expert, timing=None):
    """Returns (record bytes, info dict, stream state, base info)."""
    t0 = time.perf_counter()
    syms, scs, anchors, finfo = prepare(mats, cfg.f, cfg.c, cfg.g)
    rec_flags = cfg.flags
    if cfg.f and not finfo['applied']:
        rec_flags &= ~FLAG_F                  # fallback: plain coding with the same model
    feats = [block_features(syms[m], scs[m], anchors[m]) for m in range(3)]
    meds = [f['med'] for f in feats]
    nb = [f['srel'].size for f in feats]
    nws = [n // BPS for n in nb]
    assert all(n % BPS == 0 for n in nb)
    nss = [scs[m].size // S_CHUNK for m in range(3)]
    assert all(scs[m].size % S_CHUNK == 0 for m in range(3))
    ks, kcost = (choose_models(feats, syms, tab, cfg) if cfg.K else ([0, 0, 0], None))
    model_id = tab.model_id
    ntw, nts = (TAIL_W, TAIL_S) if cfg.t else (0, 0)
    # ---- scales: [512, 2160]
    SS = sum(nss)
    scT = np.concatenate([scs[m].reshape(nss[m], S_CHUNK) for m in range(3)]).T
    smat = np.repeat(np.arange(3), nss)
    sidx = np.concatenate([np.arange(n) for n in nss])
    sidx_t = (smat[None, :] << 8) | scT
    sF = tab.scf16[sidx_t]
    sC = tab.scc16[sidx_t]
    x0s = np.full(SS, L, np.int64)
    if cfg.e:
        seed = hash_seeds(layer, expert, model_id, 0, smat, sidx, np.zeros(SS, np.int64))
        x0s = L + hash_symbols(seed, scT).astype(np.int64)
    if nts:
        x0s = L + sum(scT[S_CHUNK - nts + j].astype(np.int64) << (8 * j) for j in range(nts))
    t1 = time.perf_counter()
    nsc = S_CHUNK - nts
    spay, slens = rans_encode_lockstep(sF[:nsc], sC[:nsc], x0s)
    t2 = time.perf_counter()
    # ---- weights: [64*(32+side), 17280], uint16 freq/start per coded symbol, stream-interleaved by column
    nq = 33 if cfg.a else 32
    nper = BPS * nq
    WS = sum(nws)
    FT = np.empty((nper, WS), np.uint16)
    CT = np.empty((nper, WS), np.uint16)
    FT3, CT3 = FT.reshape(BPS, nq, WS), CT.reshape(BPS, nq, WS)
    need_sym = cfg.e or cfg.t
    symT = np.empty((nper, WS), np.uint8) if need_sym else None
    sym3 = symT.reshape(BPS, nq, WS) if need_sym else None
    q0 = 1 if cfg.a else 0
    col = 0
    est_w = est_side = 0.0
    est_s = float(-np.log2(sF[:nsc] / P).sum())
    for m in range(3):
        f = feats[m]
        k = ks[m]
        n = nws[m]
        if cfg.a:
            roff = (k if cfg.K else m) * NACTX
            idx = (f['actx'].astype(np.int32) + np.int32(roff)) << 4
        else:
            roff = (k if cfg.K else m) * NCTX
            idx = (f['base'].astype(np.int32) + np.int32(roff)) << 4
        s2 = syms[m].reshape(-1, 32)
        idx |= s2
        est_w += float(np.bincount(idx.ravel(), minlength=tab.wf_flat.size) @ tab.wcost.ravel())
        FT3[:, q0:, col:col + n] = tab.wf16[idx].reshape(n, BPS, 32).transpose(1, 2, 0)
        CT3[:, q0:, col:col + n] = tab.wc16[idx].reshape(n, BPS, 32).transpose(1, 2, 0)
        if need_sym:
            sym3[:, q0:, col:col + n] = s2.reshape(n, BPS, 32).transpose(1, 2, 0)
        del idx
        if cfg.a:
            srow = (k if cfg.K else m) * 7 + f['srel'].astype(np.int64)
            bm = f['bm3'].astype(np.int64)
            Fs = tab.sf[srow, bm]
            Cs = tab.sfc[srow, bm]
            est_side += float(-np.log2(Fs / P).sum())
            FT3[:, 0, col:col + n] = Fs.reshape(n, BPS).T
            CT3[:, 0, col:col + n] = Cs.reshape(n, BPS).T
            if need_sym:
                sym3[:, 0, col:col + n] = bm.reshape(n, BPS).T
        col += n
    wmat = np.repeat(np.arange(3), nws)
    widx = np.concatenate([np.arange(n) for n in nws])
    x0w = np.full(WS, L, np.int64)
    if cfg.e:
        kvec = np.repeat(np.array(ks), nws)
        seed = hash_seeds(layer, expert, model_id, 1, wmat, widx, kvec)
        x0w = L + hash_symbols(seed, symT).astype(np.int64)
    if ntw:
        x0w = L + sum(symT[nper - ntw + j].astype(np.int64) << (4 * j) for j in range(ntw))
        est_w -= float(-np.log2(FT[nper - ntw:].astype(np.float64) / P).sum())
    t3 = time.perf_counter()
    nwc = nper - ntw
    wpay, wlens = rans_encode_lockstep(FT[:nwc], CT[:nwc], x0w)
    del FT, CT
    t4 = time.perf_counter()
    assert (slens <= 65535).all() and (wlens <= 65535).all()
    base_info = {'meds': meds, 'ks': ks, 'kcost': kcost, 'scale_payload': int(slens.sum()), 'weight_payload': int(wlens.sum()),
                 'n_scale_streams': SS, 'n_weight_streams': WS,
                 'est_weight_bits': est_w, 'est_side_bits': est_side, 'est_scale_bits': est_s,
                 'slens': slens, 'wlens': wlens, 'rec_flags': rec_flags, 'ceil': finfo}
    st = {'meds': meds, 'ks': ks, 'slens': slens, 'wlens': wlens, 'spay': spay, 'wpay': wpay, 'smat': smat, 'wmat': wmat,
          'dbins': [feats[m]['dbin'].reshape(nws[m], BPS) for m in range(3)], 'nws': nws, 'rec_flags': rec_flags}
    rec, info = build_record(st, tab, cfg, base_info)
    t5 = time.perf_counter()
    if timing is not None:
        timing.update({'enc_prep_s': t1 - t0 + (t3 - t2), 'enc_rans_scales_s': t2 - t1, 'enc_rans_weights_s': t4 - t3,
                       'enc_pack_s': t5 - t4, 'enc_total_s': t5 - t0})
    return rec, info, st, base_info


def build_record(st, tab, cfg, base_info):
    """Serialise a record from encoded streams (the D variant reuses the same payload).
    Record flags = the encoder's (F cleared on fallback) with D taken from cfg."""
    meds, ks, slens, wlens, spay, wpay = st['meds'], st['ks'], st['slens'], st['wlens'], st['spay'], st['wpay']
    smat, wmat, nws = st['smat'], st['wmat'], st['nws']
    model_id = tab.model_id
    info = dict(base_info)
    t0 = time.perf_counter()
    rflags = (st['rec_flags'] & ~FLAG_D) | (FLAG_D if cfg.d else 0)
    info['rec_flags'] = rflags
    if cfg.wnr2:
        hdr = struct.pack('<6I3BBI', WNR2_MAGIC, model_id, int(slens.sum()), int(wlens.sum()), W_CHUNK, S_CHUNK,
                          meds[0], meds[1], meds[2], 0, 0)
        assert len(hdr) == 32
        lt = np.concatenate([slens, wlens]).astype('<u2').tobytes()
        info['header'] = 32
    else:
        if cfg.d:
            sbase = [int(np.median(slens[smat == m])) for m in range(3)]
            alphas, sres, wres = [], [], []
            for m in range(3):
                sres.append(slens[smat == m] - sbase[m])
                Q = predict_wlen_q(st['dbins'][m], tab.etab[m])
                a_q = fit_alpha(Q, wlens[wmat == m])
                alphas.append(a_q)
                wres.append(wlens[wmat == m] - pred_from_q(Q, a_q))
            lt, nbits = pack_groups(sres + wres)
            info['len_bits'] = nbits
            info['len_resid_std_w'] = float(np.concatenate(wres).std())
            info['len_resid_std_s'] = float(np.concatenate(sres).std())
        else:
            sbase, alphas = [0, 0, 0], [0, 0, 0]
            lt = np.concatenate([slens, wlens]).astype('<u2').tobytes()
        hdr = struct.pack('<6I3BB3BBI6H', WNR3_MAGIC, model_id, int(slens.sum()), int(wlens.sum()), W_CHUNK, S_CHUNK,
                          meds[0], meds[1], meds[2], rflags, ks[0], ks[1], ks[2], cfg.K, len(lt),
                          alphas[0], alphas[1], alphas[2], sbase[0], sbase[1], sbase[2])
        assert len(hdr) == 48
        info['header'] = 48
    used = len(hdr) + len(lt) + spay.size + wpay.size
    total = align(used)
    rec = bytearray(total)
    rec[:len(hdr)] = hdr
    p = len(hdr)
    rec[p:p + len(lt)] = lt
    p += len(lt)
    rec[p:p + spay.size] = spay.tobytes()
    p += spay.size
    rec[p:p + wpay.size] = wpay.tobytes()
    info.update({'length_table': len(lt), 'used': used, 'total': total, 'padding': total - used,
                 'build_s': time.perf_counter() - t0})
    return bytes(rec), info


# ----------------------------------------------------------------------------------------------- record decoder
class DecodeError(Exception):
    pass


def decode_record(rec, tab, layer, expert, shapes, timing=None):
    """rec: record bytes; shapes: [(R, C)] symbols per matrix (from the WexEntry). Returns the decoded slab
    (uint8, scales w1..w3 then packed weights w1..w3) and info. Raises DecodeError on any failed check."""
    t0 = time.perf_counter()
    rec = np.frombuffer(rec, np.uint8)
    if rec.size % ALIGN:
        raise DecodeError('size not 4096-aligned')
    magic = struct.unpack_from('<I', rec, 0)[0]
    if magic == WNR2_MAGIC:
        _, mid, spb, wpb, wch, sch, m0, m1, m2, z0, z1 = struct.unpack_from('<6I3BBI', rec, 0)
        if z0 or z1:
            raise DecodeError('reserved header bytes')
        flags, ks, K, ltb, alphas, sbase, hlen = 0, [0, 0, 0], 0, None, None, None, 32
    elif magic == WNR3_MAGIC:
        (_, mid, spb, wpb, wch, sch, m0, m1, m2, flags, k0, k1, k2, K, ltb,
         a0, a1, a2, b0, b1, b2) = struct.unpack_from('<6I3BB3BBI6H', rec, 0)
        ks, alphas, sbase, hlen = [k0, k1, k2], [a0, a1, a2], [b0, b1, b2], 48
    else:
        raise DecodeError('bad magic')
    if mid != tab.model_id:
        raise DecodeError('model id mismatch')
    if (wch, sch) != (W_CHUNK, S_CHUNK):
        raise DecodeError('chunk not supported by this reference')
    if flags & ~(FLAG_A | FLAG_C | FLAG_D | FLAG_E | FLAG_F | FLAG_T):
        raise DecodeError('unknown flags')
    if (flags & FLAG_E) and (flags & FLAG_T):
        raise DecodeError('E and T are exclusive')
    side = bool(flags & FLAG_A)
    if side != (tab.sf is not None) or K != tab.K or any(k >= max(1, K) for k in ks):
        raise DecodeError('flags do not match the model')
    ntw, nts = (TAIL_W, TAIL_S) if flags & FLAG_T else (0, 0)
    meds = [m0, m1, m2]
    nsym = [R * C for R, C in shapes]
    nws = [n // W_CHUNK for n in nsym]
    nss = [n // 32 // S_CHUNK for n in nsym]
    SS, WS = sum(nss), sum(nws)
    p = hlen
    if magic == WNR2_MAGIC or not (flags & FLAG_D):
        lt = np.frombuffer(rec, '<u2', SS + WS, p).astype(np.int64)
        slens, wlens = lt[:SS], lt[SS:]
        p += 2 * (SS + WS)
        if magic == WNR3_MAGIC and ltb != 2 * (SS + WS):
            raise DecodeError('length table size')
        wres = None
    else:
        segs, used = unpack_groups(rec[p:p + ltb].tobytes(), nss + nws)
        if used != ltb:
            raise DecodeError('length table size')
        slens = np.concatenate([segs[m] + sbase[m] for m in range(3)])
        wres = segs[3:]
        p += ltb
    if (slens < 4).any() or slens.sum() != spb:
        raise DecodeError('scale lengths')
    sbeg = p
    wbeg = p + spb
    used_end = wbeg + wpb
    if align(used_end) != rec.size or rec[used_end:].any():
        raise DecodeError('size / padding')
    smat = np.repeat(np.arange(3), nss)
    sidx = np.concatenate([np.arange(n) for n in nss])
    pad = np.zeros(2 * (W_CHUNK + BPS) + 16, np.uint8)
    pl_s = np.concatenate([rec[sbeg:wbeg], pad])
    soffs = np.zeros(SS, np.int64)
    soffs[1:] = np.cumsum(slens)[:-1]
    hs = hash_seeds(layer, expert, mid, 0, smat, sidx, np.zeros(SS, np.int64)) if flags & FLAG_E else None
    t1 = time.perf_counter()
    scT, sok = decode_scales_lockstep(pl_s, soffs, slens, S_CHUNK, smat, tab, hs, nts)
    t2 = time.perf_counter()
    if not sok.all():
        raise DecodeError(f'scale streams failed: {int((~sok).sum())}')
    scs = []
    a = 0
    for m in range(3):
        scs.append(np.ascontiguousarray(scT[:, a:a + nss[m]].T).ravel())
        a += nss[m]
    # weight-stream lengths (D: predicted from the decoded scales)
    srelT = np.empty((BPS, WS), np.int64)
    col = 0
    wl = []
    for m in range(3):
        d = scs[m].astype(np.int16) - meds[m]
        srel = np.clip(d, -3, 3) + 3
        srelT[:, col:col + nws[m]] = srel.reshape(nws[m], BPS).T
        if wres is not None:
            dbin = (np.clip(d, -DBC, DBC) + DBC).reshape(nws[m], BPS)
            wl.append(pred_from_q(predict_wlen_q(dbin, tab.etab[m]), alphas[m]) + wres[m])
        col += nws[m]
    if wres is not None:
        wlens = np.concatenate(wl)
    if (wlens < 4).any() or (wlens > 65535).any() or wlens.sum() != wpb:
        raise DecodeError('weight lengths')
    pl_w = np.concatenate([rec[wbeg:used_end], pad])
    woffs = np.zeros(WS, np.int64)
    woffs[1:] = np.cumsum(wlens)[:-1]
    wmat = np.repeat(np.arange(3), nws)
    widx = np.concatenate([np.arange(n) for n in nws])
    kvec = np.repeat(np.array(ks), nws) if K else np.zeros(WS, np.int64)
    nct = NACTX if side else NCTX
    ctxoff = (kvec if K else wmat).astype(np.int64) * nct
    sideoff = (kvec if K else wmat).astype(np.int64) * 7
    hw = hash_seeds(layer, expert, mid, 1, wmat, widx, kvec) if flags & FLAG_E else None
    t3 = time.perf_counter()
    wT, wok = decode_weights_lockstep(pl_w, woffs, wlens, srelT, ctxoff, sideoff, tab, side, hw, ntw)
    t4 = time.perf_counter()
    if not wok.all():
        raise DecodeError(f'weight streams failed: {int((~wok).sum())}')
    wparts = []
    col = 0
    for m in range(3):
        sym = np.ascontiguousarray(wT[:, col:col + nws[m]].T).ravel()
        if flags & FLAG_F:
            try:
                sym, scs[m] = ceil_inverse(sym, scs[m])
            except ValueError as ex:
                raise DecodeError(str(ex))
        wparts.append((sym[0::2] | (sym[1::2] << 4)).astype(np.uint8))
        col += nws[m]
    slab = np.concatenate(scs + wparts)
    t5 = time.perf_counter()
    if timing is not None:
        timing.update({'dec_parse_s': t1 - t0 + (t3 - t2), 'dec_rans_scales_s': t2 - t1, 'dec_rans_weights_s': t4 - t3,
                       'dec_unpack_s': t5 - t4, 'dec_total_s': t5 - t0})
    return slab, {'flags': flags, 'ks': ks}
