"""Test kit / audit (from team5 audit-ceil): corruption audit of REAL ceil-form records (WNR3 spec v2.1, 5.8 / test T3).

Coder: wnr_common_cr.py = byte copy of team 4 ceil-ref wnr_common.py (sha256 F311E27A...0586), never edited.
Paths are not hard-coded: the runner calls configure(sample_dir, manifest, models, rec_dir) first.
Records are produced with its encode_record()/build_record() (anchor rule g0 = constant 120, monkey-patched exactly as
ceil-ref run_fixed_anchor.py does) with the held-out model fg_a0_K0_c0 of ceil-ref models\\even.npz (80 even experts).

Record kinds (one per integrity rule):
  fg0d   CEIL|LENC            x0 = L                      -> rule R0   (today: final state == L, exact length)
  fg0de  CEIL|LENC|HASH       x0 = L + h23 (ceil-ref e: FNV-1a seed + FNV-1a symbols + lowbias32)  -> rule HASH
  fg0dh  CEIL|LENC|HASH       x0 = L + (H32(ident16 || coded bytes) & 0x7FFFFF), spec 5.1 normative  -> rule HASH32
  fg0dt  CEIL|LENC|TAIL       x0 = L + 5 tail nibbles / 2 tail bytes, window check (ceil-ref t)      -> rule TAIL
  fg0dtD CEIL|LENC|TAIL|DIGEST same streams as fg0dt; 48-B ceil-ref header + u32 record_digest (spec 5.8:
         XOR over all streams of H32(ident16 || coded bytes), flags 0xB4) -> 52-B header           -> rule TAILD
Decoder: lane decoders below are ceil-ref's lock-step loops (decode_scales_lockstep / decode_weights_lockstep, side=False)
with the in-loop hash removed; the final-state rule is applied afterwards from (x, ptr, decoded symbols), which is the same
computation (the in-loop FNV fold == hash_symbols over the decoded symbols).
Ceil inverse: 'spec' (5.2: M' == 5 -> D, s - 1; cs == 0 with M' == 5 invalid; M' 0..4 legal) and 'cr' (ceil-ref
ceil_inverse: additionally rejects any decoded block with max < 5).
"""
import hashlib, json, math, os, struct, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wnr_common_cr as W

SAMPLE = None                                          # dir with the expert .bin files (configure())
MANIFEST = None                                        # 160-expert manifest.json (configure())
MODELS = os.path.join(HERE, 'models', 'even.npz')      # ceil-ref models/even.npz (trained on the 80 even experts)
MODEL_KEY = 'fg_a0_K0_c0'
EXPERT_IDX = [1, 25, 47, 71, 95, 113, 137, 159]      # CR held-out (odd) experts, layers 0,6,11,17,23,28,34,39
ANCHOR0 = 120
REC_DIR = None                                         # record cache dir (configure())
_MAN = None


def configure(sample_dir=None, manifest=None, models=None, rec_dir=None):
    global SAMPLE, MANIFEST, MODELS, REC_DIR, _MAN
    if sample_dir is not None:
        SAMPLE = sample_dir
    if manifest is not None:
        MANIFEST, _MAN = manifest, None
    if models is not None:
        MODELS = models
    if rec_dir is not None:
        REC_DIR = rec_dir


def manifest():
    global _MAN
    if _MAN is None:
        with open(MANIFEST, 'r', encoding='utf-8') as f:
            _MAN = json.load(f)
    return _MAN
L, P, PB, BPS, S_CHUNK, W_CHUNK = W.L, W.P, W.PB, W.BPS, W.S_CHUNK, W.W_CHUNK
NWS, NSS = 5760, 720
NS_TOT = 3 * NSS
FLAG_DIGEST = 128
MASK23 = (1 << 23) - 1
H32_SEED = 0x33524E57
U32 = np.uint32
KINDS = ('fg0d', 'fg0de', 'fg0dh', 'fg0dt', 'fg0dtD')
RULE_KIND = {'R0': 'fg0d', 'HASH': 'fg0de', 'HASH32': 'fg0dh', 'TAIL': 'fg0dt', 'TAILD': 'fg0dtD'}
DEC_KINDS = ('fg0d', 'fg0de', 'fg0dh', 'fg0dt')       # fg0dt streams serve TAIL and TAILD
RULES_OF_DEC = {'fg0d': ('R0',), 'fg0de': ('HASH',), 'fg0dh': ('HASH32',), 'fg0dt': ('TAIL', 'TAILD')}
RULES = ('R0', 'HASH', 'HASH32', 'TAIL', 'TAILD')
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)
DOUBLE = W.DOUBLE

W.anchor_pair = lambda sc: ANCHOR0      # rule g0, as ceil-ref run_fixed_anchor.py (prepare() looks it up at call time)


# ------------------------------------------------------------------------------------------------ H32 (MurmurHash3_x86_32)
def _rotl(x, r):
    return (x << U32(r)) | (x >> U32(32 - r))


def h32_lanes(data, seed=H32_SEED):
    """MurmurHash3_x86_32 of every row of data (uint8 [S, n], same n), seed scalar or [S]. Returns uint32 [S]."""
    data = np.ascontiguousarray(data, np.uint8)
    S, n = data.shape
    c1, c2 = U32(0xcc9e2d51), U32(0x1b873593)
    h = np.empty(S, U32)
    h[:] = seed
    nb = n // 4
    if nb:
        k = np.ascontiguousarray(data[:, :nb * 4]).view('<u4').reshape(S, nb)
        for i in range(nb):
            k1 = k[:, i] * c1
            k1 = _rotl(k1, 15) * c2
            h ^= k1
            h = _rotl(h, 13) * U32(5) + U32(0xe6546b64)
    r = n & 3
    if r:
        t = data[:, nb * 4:].astype(U32)
        k1 = np.zeros(S, U32)
        if r == 3:
            k1 ^= t[:, 2] << U32(16)
        if r >= 2:
            k1 ^= t[:, 1] << U32(8)
        k1 ^= t[:, 0]
        k1 = _rotl(k1 * c1, 15) * c2
        h ^= k1
    h ^= U32(n & 0xFFFFFFFF)
    h ^= h >> U32(16)
    h *= U32(0x85ebca6b)
    h ^= h >> U32(13)
    h *= U32(0xc2b2ae35)
    h ^= h >> U32(16)
    return h


def h32_verification():
    """SMHasher VerificationTest for MurmurHash3_x86_32; expected 0xB0F57EE3 (spec 5.1)."""
    key = np.arange(256, dtype=np.uint8)
    hs = bytearray()
    for i in range(256):
        hs += struct.pack('<I', int(h32_lanes(key[None, :i], 256 - i)[0]))
    return int(h32_lanes(np.frombuffer(bytes(hs), np.uint8)[None, :], 0)[0])


def ident16(model_id, layer, expert, g, m, flags, ceil_bit=1, set_id=0, anchor=ANCHOR0):
    """Spec 5.1 per-stream identity (4 x u32 LE), vectorised over g (layer/expert/m scalars or arrays)."""
    g = np.asarray(g, np.int64).ravel()
    S = g.size
    lay = np.broadcast_to(np.asarray(layer, np.int64), (S,))
    exp = np.broadcast_to(np.asarray(expert, np.int64), (S,))
    mm = np.broadcast_to(np.asarray(m, np.int64), (S,))
    a = np.empty((S, 4), '<u4')
    a[:, 0] = model_id
    a[:, 1] = (lay << 16) | exp
    a[:, 2] = g | (np.int64(ceil_bit) << 31)
    a[:, 3] = (np.int64(flags) & 0xFF) | (mm << 8) | (np.int64(set_id) << 16) | (np.int64(anchor) << 24)
    return a.view(np.uint8).reshape(S, 16)


def pack_nib(sym2d):
    """[S, n] nibbles in stream order -> [S, n/2] coded bytes, low nibble = even symbol (spec 5.1)."""
    return (sym2d[:, 0::2] | (sym2d[:, 1::2] << 4)).astype(np.uint8)


def h32_streams(model_id, layer, expert, g, m, flags, data):
    return h32_lanes(np.concatenate([ident16(model_id, layer, expert, g, m, flags), data], 1))


def gw(m, j):
    return NS_TOT + np.asarray(m, np.int64) * NWS + np.asarray(j, np.int64)


def gs(m, j):
    return np.asarray(m, np.int64) * NSS + np.asarray(j, np.int64)


# FNV seeds of ceil-ref feature e, vectorised over lanes (copy of W.hash_seeds with array layer/expert, K = 0)
def fnv_seeds(layer, expert, model_id, kind, mat, idx):
    idx = np.asarray(idx, np.int64).ravel()
    S = idx.size
    rec = np.zeros((S, 6), '<u4')
    rec[:, 0] = np.broadcast_to(np.asarray(layer, np.int64), (S,))
    rec[:, 1] = np.broadcast_to(np.asarray(expert, np.int64), (S,))
    rec[:, 2] = model_id
    rec[:, 3] = kind
    rec[:, 4] = np.broadcast_to(np.asarray(mat, np.int64), (S,))
    rec[:, 5] = idx
    by = rec.view(np.uint8).reshape(S, 24)
    h = np.full(S, W.FNV_OFF, U32)
    for j in range(24):
        h = (h ^ by[:, j].astype(U32)) * U32(W.FNV_PRIME)
    return h


# ------------------------------------------------------------------------------------------------ model, sources
def load_tables():
    Z = np.load(MODELS)
    k = MODEL_KEY
    return W.Tables(Z[k + '_wf'], Z[k + '_scf'], None, Z[k + '_etab'], K=0, nctx=W.NCTX)


def load_source(idx):
    e = manifest()['experts'][idx]
    raw, mats = W.load_expert(SAMPLE, e)
    sha_ok = hashlib.sha256(raw.tobytes()).hexdigest() == e['sha256']
    if not sha_ok:
        raise RuntimeError(f"sha256 mismatch for {e['file']} (expected {e['sha256']})")
    syms, scs, anchors, finfo = W.prepare(mats, True, False, True)
    assert finfo['applied'] and anchors == [ANCHOR0] * 3
    return {'idx': idx, 'file': e['file'], 'layer': e['layer'], 'expert': e['expert'], 'raw': raw, 'mats': mats,
            'syms': syms, 'scs': scs, 'sha_ok': sha_ok, 'finfo': finfo,
            'shapes': [(mt['R'], mt['C']) for mt in mats]}


# ------------------------------------------------------------------------------------------------ encoding (ceil-ref coder)
class _H32Hash:
    """Swap ceil-ref's FNV/lowbias32 x0 hash for the normative spec 5.1 H32 (ident16 || coded bytes), flags 0x1C.
    encode_record() calls hash_seeds()/hash_symbols() by module-global name, so patching the module is enough."""

    def __enter__(self):
        self.old = (W.hash_seeds, W.hash_symbols)

        def seeds(layer, expert, model_id, kind, mat_idx, stream_idx, kvec):
            return ('h32', layer, expert, model_id, kind, np.asarray(mat_idx, np.int64), np.asarray(stream_idx, np.int64))

        def symbols(seed, symT):
            _, layer, expert, model_id, kind, mat, idx = seed
            if kind == 0:
                g, data = gs(mat, idx), np.ascontiguousarray(symT.T)
            else:
                g, data = gw(mat, idx), pack_nib(np.ascontiguousarray(symT.T))
            h = h32_streams(model_id, layer, expert, g, mat, 0x1C, data)
            return (h & U32(MASK23)).astype(U32)

        W.hash_seeds, W.hash_symbols = seeds, symbols
        return self

    def __exit__(self, *a):
        W.hash_seeds, W.hash_symbols = self.old


def stream_h32_all(src, model_id, flags):
    """Clean per-stream H32 (coded domain, incl. tail symbols) of every scale and weight stream: ([3,NSS], [3,NWS])."""
    hs = np.zeros((3, NSS), U32)
    hw = np.zeros((3, NWS), U32)
    for m in range(3):
        hs[m] = h32_streams(model_id, src['layer'], src['expert'], gs(m, np.arange(NSS)), m, flags,
                            src['scs'][m].reshape(NSS, S_CHUNK))
        hw[m] = h32_streams(model_id, src['layer'], src['expert'], gw(m, np.arange(NWS)), m, flags,
                            pack_nib(src['syms'][m].reshape(NWS, W_CHUNK)))
    return hs, hw


def record_digest(src, model_id, flags=0xB4):
    hs, hw = stream_h32_all(src, model_id, flags)
    return int(np.bitwise_xor.reduce(hs.ravel()) ^ np.bitwise_xor.reduce(hw.ravel()))


def add_digest(rec48, used, digest):
    b = bytearray(rec48[:used])
    assert struct.unpack_from('<I', b, 0)[0] == W.WNR3_MAGIC
    b[27] |= FLAG_DIGEST
    b = b[:48] + struct.pack('<I', digest) + b[48:]
    total = W.align(len(b))
    return bytes(b) + bytes(total - len(b))


def encode_kinds(src, tab, kinds=KINDS):
    """Encode the record kinds asked for (fg0dtD implies fg0dt). Returns {kind: (record bytes, info)}."""
    cfg = {'fg0d': W.Config('fgd', d=True, f=True, g=True),
           'fg0de': W.Config('fgde', d=True, e=True, f=True, g=True),
           'fg0dt': W.Config('fgdt', d=True, t=True, f=True, g=True)}
    kinds = set(kinds)
    out = {}
    for k, c in cfg.items():
        if k in kinds or (k == 'fg0dt' and 'fg0dtD' in kinds):
            rec, info, st, bi = W.encode_record(src['mats'], tab, c, src['layer'], src['expert'])
            out[k] = (rec, info)
    if 'fg0dh' in kinds:
        with _H32Hash():
            rec, info, st, bi = W.encode_record(src['mats'], tab, cfg['fg0de'], src['layer'], src['expert'])
        out['fg0dh'] = (rec, info)
    if 'fg0dtD' in kinds:
        dg = record_digest(src, tab.model_id, 0xB4)
        rec_t, info_t = out['fg0dt']
        out['fg0dtD'] = (add_digest(rec_t, info_t['used'], dg), dict(info_t, digest=dg))
    return {k: v for k, v in out.items() if k in kinds}


# ------------------------------------------------------------------------------------------------ record parsing
class DecodeError(Exception):
    pass


def parse_record(rec, tab, shapes):
    """Front part of ceil-ref decode_record (copy) + DIGEST: header, flags, length section, size/padding checks.
    Returns dict; raises DecodeError(stage, msg)."""
    rec = np.frombuffer(rec, np.uint8) if not isinstance(rec, np.ndarray) else rec
    if rec.size % W.ALIGN or rec.size < 64:
        raise DecodeError('size', 'size not 4096-aligned')
    magic = struct.unpack_from('<I', rec, 0)[0]
    if magic != W.WNR3_MAGIC:
        raise DecodeError('header', 'bad magic')
    (_, mid, spb, wpb, wch, sch, m0, m1, m2, flags, k0, k1, k2, K, ltb,
     a0, a1, a2, b0, b1, b2) = struct.unpack_from('<6I3BB3BBI6H', rec, 0)
    hlen, digest = 48, None
    if flags & FLAG_DIGEST:
        if not flags & W.FLAG_T:
            raise DecodeError('header', 'DIGEST without TAIL')
        digest = struct.unpack_from('<I', rec, 48)[0]
        hlen = 52
    if mid != tab.model_id:
        raise DecodeError('header', 'model id mismatch')
    if (wch, sch) != (W_CHUNK, S_CHUNK):
        raise DecodeError('header', 'chunk')
    if flags & ~(W.FLAG_A | W.FLAG_C | W.FLAG_D | W.FLAG_E | W.FLAG_F | W.FLAG_T | FLAG_DIGEST):
        raise DecodeError('header', 'unknown flags')
    if (flags & W.FLAG_E) and (flags & W.FLAG_T):
        raise DecodeError('header', 'E and T are exclusive')
    if bool(flags & W.FLAG_A) or K != 0 or any(k >= 1 for k in (k0, k1, k2)):
        raise DecodeError('header', 'flags do not match the model')
    ks, alphas, sbase = [k0, k1, k2], [a0, a1, a2], [b0, b1, b2]
    meds = [m0, m1, m2]
    nsym = [R * C for R, C in shapes]
    nws = [n // W_CHUNK for n in nsym]
    nss = [n // 32 // S_CHUNK for n in nsym]
    SS, WS = sum(nss), sum(nws)
    p = hlen
    if not (flags & W.FLAG_D):
        lt = np.frombuffer(rec, '<u2', SS + WS, p).astype(np.int64)
        slens, wlens0 = lt[:SS], lt[SS:]
        p += 2 * (SS + WS)
        if ltb != 2 * (SS + WS):
            raise DecodeError('lengths', 'length table size')
        wres = None
    else:
        try:
            segs, used = W.unpack_groups(rec[p:p + ltb].tobytes(), nss + nws)
        except Exception as ex:                      # malformed table (e.g. truncated buffer)
            raise DecodeError('lengths', f'length table parse: {type(ex).__name__}')
        if used != ltb:
            raise DecodeError('lengths', 'length table size')
        slens = np.concatenate([segs[m] + sbase[m] for m in range(3)])
        wres = segs[3:]
        wlens0 = None
        p += ltb
    if (slens < 4).any() or slens.sum() != spb:
        raise DecodeError('lengths', 'scale lengths')
    sbeg = p
    wbeg = p + spb
    used_end = wbeg + wpb
    if W.align(used_end) != rec.size or rec[used_end:].any():
        raise DecodeError('size', 'size / padding')
    soffs = np.zeros(SS, np.int64)
    soffs[1:] = np.cumsum(slens)[:-1]
    return {'rec': rec, 'mid': mid, 'spb': spb, 'wpb': wpb, 'flags': flags, 'meds': meds, 'ks': ks, 'K': K,
            'alphas': alphas, 'sbase': sbase, 'hlen': hlen, 'digest': digest, 'ltb': ltb, 'nws': nws, 'nss': nss,
            'SS': SS, 'WS': WS, 'slens': slens, 'soffs': soffs, 'wres': wres, 'wlens0': wlens0,
            'sbeg': sbeg, 'wbeg': wbeg, 'used_end': used_end}


def weight_lengths(H, scs, tab):
    """Weight-stream lengths from the (coded) scales (LENC) or the u16 table; ceil-ref decode_record logic (copy)."""
    if H['wres'] is None:
        wl = H['wlens0']
    else:
        wl = []
        for m in range(3):
            d = scs[m].astype(np.int16) - H['meds'][m]
            dbin = (np.clip(d, -W.DBC, W.DBC) + W.DBC).reshape(H['nws'][m], BPS)
            wl.append(W.pred_from_q(W.predict_wlen_q(dbin, tab.etab[m]), H['alphas'][m]) + H['wres'][m])
        wl = np.concatenate(wl)
    if (wl < 4).any() or (wl > 65535).any() or wl.sum() != H['wpb']:
        raise DecodeError('wlengths', 'weight lengths')
    wo = np.zeros(wl.size, np.int64)
    wo[1:] = np.cumsum(wl)[:-1]
    return wl, wo


def srel_of(sc, med):
    return (np.clip(sc.astype(np.int16) - med, -3, 3) + 3).astype(np.int16)


# ------------------------------------------------------------------------------------------------ lane decoders
PAD = np.zeros(2 * (W_CHUNK + BPS) + 16, np.uint8)


def dec_scales(pl, offs, lens, mat, tab, ntail):
    """Copy of ceil-ref decode_scales_lockstep without the in-loop hash. Returns (out [512,S], ok_init, x, ptr)."""
    S = offs.size
    x, ok = W._read_state(pl, offs)
    ok = ok & (lens >= 4)
    ptr = (offs + 4).astype(np.int32)
    lb = mat.astype(np.int32) << 14
    fb = mat.astype(np.int32) << 8
    LUT, F, C = tab.slut, tab.scf32, tab.scc32
    out = np.zeros((S_CHUNK, S), np.uint8)
    for t in range(S_CHUNK - ntail):
        slot = x & (P - 1)
        s = LUT[lb | slot]
        idx = fb | s
        x = F[idx] * (x >> PB) + slot - C[idx]
        x = W._renorm(x, ptr, pl)
        out[t] = s
    return out, ok, x, ptr


def dec_weights(pl, offs, lens, srelT, ctxoff, tab, ntail):
    """Copy of ceil-ref decode_weights_lockstep (side=False) without the in-loop hash."""
    S = offs.size
    x, ok = W._read_state(pl, offs)
    ok = ok & (lens >= 4)
    ptr = (offs + 4).astype(np.int32)
    LUT, WF, WC = tab.wlut, tab.wf32, tab.wc32
    out = np.zeros((BPS * 32, S), np.uint8)
    ctxoff = ctxoff.astype(np.int32)
    m9 = np.int32(9)
    pmx = mx = None
    for b in range(BPS):
        sr = srelT[b].astype(np.int32)
        rb = ctxoff + sr * np.int32(81)
        jend = 32 - ntail if b == BPS - 1 else 32
        for j in range(jend):
            ctx = rb + np.int32(80) if j == 0 else rb + pmx
            slot = x & (P - 1)
            s = LUT[(ctx << PB) | slot]
            idx = (ctx << 4) | s
            x = WF[idx] * (x >> PB) + slot - WC[idx]
            x = W._renorm(x, ptr, pl)
            out[b * 32 + j] = s
            mag = (s & 7).astype(np.int32)
            mx = mag if j == 0 else np.maximum(mx, mag)
            pmx = mag * m9 + mx
    return out, ok, x, ptr


def apply_rules(dk, kind, out, ok, x, ptr, offs, lens, ids, model_id):
    """Final-state rules of record kind dk on lanes. kind 's' or 'w'. ids: dict lay, exp, m, j (identity arrays).
    Mutates out (tail symbols). Returns dict rule -> bool [S]; for fg0dt also 'TAILD_h' (uint32 H32, flags 0xB4)."""
    n = out.shape[0]
    end_ok = ok & (ptr == offs + lens)
    res = {}
    if dk == 'fg0d':
        res['R0'] = end_ok & (x == L)
    elif dk == 'fg0de':
        seed = fnv_seeds(ids['lay'], ids['exp'], model_id, 0 if kind == 's' else 1, ids['m'], ids['j'])
        want = L + W.hash_symbols(seed, out).astype(np.int64)
        res['HASH'] = end_ok & (x.astype(np.int64) == want)
    elif dk == 'fg0dh':
        g = gs(ids['m'], ids['j']) if kind == 's' else gw(ids['m'], ids['j'])
        data = np.ascontiguousarray(out.T) if kind == 's' else pack_nib(np.ascontiguousarray(out.T))
        h = h32_streams(model_id, ids['lay'], ids['exp'], g, ids['m'], 0x1C, data)
        res['HASH32'] = end_ok & (x.astype(np.int64) == L + (h & U32(MASK23)).astype(np.int64))
    elif dk == 'fg0dt':
        nt, bits = (W.TAIL_S, 8) if kind == 's' else (W.TAIL_W, 4)
        v = x.astype(np.int64) - L
        okw = end_ok & (v >= 0) & (v < (np.int64(1) << (bits * nt)))
        v = np.where(okw, v, 0)
        for j in range(nt):
            out[n - nt + j] = ((v >> (bits * j)) & ((1 << bits) - 1)).astype(np.uint8)
        res['TAIL'] = okw
        g = gs(ids['m'], ids['j']) if kind == 's' else gw(ids['m'], ids['j'])
        data = np.ascontiguousarray(out.T) if kind == 's' else pack_nib(np.ascontiguousarray(out.T))
        res['TAILD_h'] = h32_streams(model_id, ids['lay'], ids['exp'], g, ids['m'], 0xB4, data)
    else:
        raise ValueError(dk)
    return res


def ntail_of(dk, kind):
    if dk != 'fg0dt':
        return 0
    return W.TAIL_S if kind == 's' else W.TAIL_W


def run_lanes(dk, kind, data_list, lens, srel, mmodel, ids, tab, chunk=24000):
    """Decode job lanes. data_list: list of uint8 arrays (bytes the decoder sees from each lane's start),
    lens [J], srel [J,64] (weights), mmodel [J] (model matrix), ids dict of [J] arrays. Returns (res dict, out [J,n])."""
    J = len(data_list)
    n = S_CHUNK if kind == 's' else W_CHUNK
    outs = np.zeros((J, n), np.uint8)
    res = {}
    for a in range(0, J, chunk):
        b = min(J, a + chunk)
        dl = data_list[a:b]
        sz = np.array([d.size for d in dl], np.int64)
        offs = np.zeros(b - a, np.int64)
        offs[1:] = np.cumsum(sz)[:-1]
        pl = np.concatenate(list(dl) + [PAD])
        ln = np.asarray(lens[a:b], np.int64)
        nt = ntail_of(dk, kind)
        if kind == 's':
            out, ok, x, ptr = dec_scales(pl, offs, ln, np.asarray(mmodel[a:b]), tab, nt)
        else:
            out, ok, x, ptr = dec_weights(pl, offs, ln, np.ascontiguousarray(srel[a:b].T), np.asarray(mmodel[a:b], np.int64) * W.NCTX, tab, nt)
        sub = {k: np.asarray(v)[a:b] for k, v in ids.items()}
        r = apply_rules(dk, kind, out, ok, x, ptr, offs, ln, sub, tab.model_id)
        for k, v in r.items():
            res.setdefault(k, []).append(v)
        outs[a:b] = out.T
    return {k: np.concatenate(v) for k, v in res.items()}, outs


# ------------------------------------------------------------------------------------------------ ceil inverse + damage
def inverse(dec, cs):
    """dec uint8 [N, 32*nb] coded nibbles, cs int16 [N, nb] coded scales -> (out_n, out_s, invalid_spec [N], invalid_cr [N], odd)."""
    N = dec.shape[0]
    blk = dec.reshape(N, -1, 32)
    mag = blk & 7
    bm = mag.max(2)
    odd = bm == 5
    inv_spec = (odd & (cs == 0)).any(1)
    inv_cr = inv_spec | (bm < 5).any(1)
    out_n = np.where(odd[..., None], (blk & 8) | DOUBLE[mag], blk).reshape(N, -1).astype(np.uint8)
    out_s = cs.astype(np.int16) - odd
    return out_n, out_s, inv_spec, inv_cr, odd


DAMAGE_KEYS = ('coded_sym_diff', 'coded_scale_diff', 'nib_diff', 'val_diff', 'scale_diff', 'block_flips',
               'err_units', 'err_abs', 'sign_only', 'nan_scale')


def damage(dec, cs, clean_c, clean_cs, src_n, src_s):
    """Per-lane damage after the spec ceil inverse. All [N, ...] arrays of one weight stream (or block range) per lane.
    err_units: max |v_out - v_src| in units of 2^(s_src - 127) (E2M1 units of the source block scale)."""
    out_n, out_s, inv_spec, inv_cr, odd = inverse(dec, cs)
    N, nb = cs.shape
    codd = (clean_c.reshape(N, nb, 32) & 7).max(2) == 5
    src_s16 = src_s.astype(np.int16)
    sh = (out_s - src_s16).astype(np.float64)
    nan = (out_s >= 255).any(1) | (out_s < 0).any(1)
    vo = E2M1[out_n].reshape(N, nb, 32) * np.exp2(sh)[..., None]
    vs = E2M1[src_n].reshape(N, nb, 32)
    dv = np.abs(vo - vs)
    absdv = dv * np.exp2(src_s16.astype(np.float64) - 127)[..., None]
    nd = out_n != src_n
    so = (nd.sum(1) > 0) & ((out_s != src_s16).sum(1) == 0) & ~((nd & ((out_n ^ src_n) != 8)).any(1))
    return {'coded_sym_diff': (dec != clean_c).sum(1), 'coded_scale_diff': (cs != clean_cs).sum(1),
            'nib_diff': nd.sum(1), 'val_diff': (dv.reshape(N, -1) > 0).sum(1), 'scale_diff': (out_s != src_s16).sum(1),
            'block_flips': (odd != codd).sum(1), 'err_units': dv.reshape(N, -1).max(1), 'err_abs': absdv.reshape(N, -1).max(1),
            'sign_only': so, 'nan_scale': nan, 'inv_spec': inv_spec, 'inv_cr': inv_cr}


# ------------------------------------------------------------------------------------------------ statistics
def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided (1-alpha) CI for a binomial proportion, by bisection on P(X <= k | p)."""
    if n == 0:
        return (0.0, 1.0)
    ii = np.arange(0, k + 1, dtype=np.float64)
    lgam = np.frompyfunc(math.lgamma, 1, 1)
    lc = (math.lgamma(n + 1) - lgam(ii + 1) - lgam(n - ii + 1)).astype(np.float64)

    def cdf_le(kk, p):          # P(X <= kk), kk in {k-1, k}
        if kk < 0:
            return 0.0
        if kk >= n:
            return 1.0
        t = lc[:kk + 1] + ii[:kk + 1] * math.log(p) + (n - ii[:kk + 1]) * math.log1p(-p)
        mx = t.max()
        return min(1.0, math.exp(mx) * float(np.exp(t - mx).sum()))

    def bis(f, lo, hi):
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    if k == 0:
        lo = 0.0
    else:   # lower: P(X >= k | p) = alpha/2  <=>  1 - cdf(k-1) = alpha/2
        lo = bis(lambda p: 1 - cdf_le(k - 1, p) < alpha / 2, 0.0, k / n)
    if k == n:
        hi = 1.0
    else:   # upper: P(X <= k | p) = alpha/2
        hi = bis(lambda p: cdf_le(k, p) > alpha / 2, k / n, 1.0)
    return (lo, hi)


def ci_fast(k, n, alpha=0.05):
    """Clopper-Pearson for small k (exact), Wilson for large k (k > 2000 or n-k small handled by symmetry)."""
    if n == 0:
        return (0.0, 1.0)
    if k <= 2000:
        return clopper_pearson(k, n, alpha)
    if n - k <= 2000:
        lo, hi = clopper_pearson(n - k, n, alpha)
        return (1 - hi, 1 - lo)
    z = 1.959963984540054
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - hw), min(1.0, c + hw))


def jdump(o, path):
    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.bool_,)):
            return bool(x)
        raise TypeError(type(x))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(o, f, indent=1, default=conv)
