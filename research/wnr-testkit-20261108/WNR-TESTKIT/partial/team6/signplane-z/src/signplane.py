"""signplane-z (team 6): value-exact layout = canonical zero + sign bypass with a COMPACTED raw sign plane.

Built on a verbatim copy of team 4 ceil-ref's coder (wnr_common.py in this folder, sha256 F311E27A...). Nothing of the
original is modified; this module adds the Z layout next to it.

Coding domain: ceil-form (f), constant srel anchor 120 (g0), canonical zero (c is implied: a zero has no sign to store).
  * rANS codes only the 3-bit magnitude a = q & 7 with an 8-symbol 14-bit CDF per context; contexts are the WNR2 ones
    (ctx = (srel*9 + pm)*9 + mx, 567 rows per matrix type, pm = mx = 8 at a block start), so they never depend on signs.
  * the sign of every NONZERO magnitude is stored raw; zeros carry no sign bit (they decode as +0).

Weight stream layout (Z1, 'tail plane'):
    [4-B big-endian rANS state][rANS renorm bytes ...][sign plane, ceil(nsg/8) bytes, stored in DESCENDING address order]
  Sign bit k (k-th nonzero weight of the stream in decode order) is bit (k & 7) of the byte at address end-1-(k >> 3),
  end = stream offset + stream length. So a decoder reads rANS bytes forward from the stream start and sign bits backward
  from the stream end (a backward bit reader, as in zstd's FSE/Huffman streams); no sign-plane length or offset is stored:
  nsg is known when the stream is decoded and the two readers must meet exactly:
      rANS bytes consumed == len - ceil(nsg/8)   and   the unused high bits of the last (partial) sign byte are 0.
Z2 ('tail plane + x0 tail'): as Z1, but the last r = min(nsg & 7, nonzeros in the stream's last block) signs are not in
  the plane; they ride in the start state above the hash: x0 = L + h23 + (t << 23), t < 2^r (x0 < 2^31 holds).
  The plane then holds nP = nsg - r bits in ceil(nP/8) bytes (= floor(nsg/8) bytes except in the r < nsg & 7 corner).
  Decoder: final v = x - L, t = v >> 23 must be < 2^r, the low 23 bits must equal h23 of the decoded nibbles.

HASH (E) as in ceil-ref: x0 = L + h23, h23 = lowbias32(FNV-1a(identity seed, every coded NIBBLE q = a | sign << 3)) & (2^23-1).
  The hash input is the full nibble incl. the raw sign bits, so a flipped sign-plane bit changes h23 while the rANS final
  state stays the same -> mismatch (miss probability 2^-23 per corrupted stream).
Scales: unchanged from ceil-ref (order-0 per matrix type, hash kind 0).
Record: ceil-ref's 48-B WNR3 header; flags F|C|E|Z (+D) (+Z2); Z = 64, Z2 = 128 (this experiment's own bits).
"""
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wnr_common as W  # noqa: E402  (verbatim copy of team 4 ceil-ref)
from wnr_common import P, PB, L, BPS, NCTX, NACTX, NSUB, NDB, DBC, S_CHUNK, W_CHUNK, U32  # noqa: E402

FLAG_Z, FLAG_Z2 = 64, 128
ANCHOR0 = 120
FRONT = 512        # front padding of the payload copy used by the lockstep decoder (backward sign reads)
M23 = (1 << 23) - 1


def use_g0(a0=ANCHOR0):
    """g0: constant srel anchor (as team 4 run_fixed_anchor.py: prepare() looks anchor_pair up at call time)."""
    W.anchor_pair = lambda sc: a0


# ------------------------------------------------------------------------------------------------- model
class TablesZ:
    """8-symbol magnitude tables wf8 [3*567, 8] + order-0 scale tables scf [3, 256] + Z length predictor etab [3, 25]."""

    def __init__(self, wf8, scf, etab):
        self.wf8 = np.ascontiguousarray(wf8, np.int64)
        assert self.wf8.shape == (3 * NCTX, 8) and (self.wf8.sum(1) == P).all() and (self.wf8 >= 1).all()
        self.wc8 = W.cum_excl(self.wf8)
        self.scf = np.ascontiguousarray(scf, np.int64)
        self.scc = W.cum_excl(self.scf)
        self.etab = np.asarray(etab, np.int64)
        self.wf8_flat, self.wc8_flat = self.wf8.ravel(), self.wc8.ravel()
        self.wf16, self.wc16 = self.wf8_flat.astype(np.uint16), self.wc8_flat.astype(np.uint16)   # names as in W.Tables
        self.wf32, self.wc32 = self.wf8_flat.astype(np.int32), self.wc8_flat.astype(np.int32)
        self.scf_flat, self.scc_flat = self.scf.ravel(), self.scc.ravel()
        self.scf16, self.scc16 = self.scf_flat.astype(np.uint16), self.scc_flat.astype(np.uint16)
        self.scf32, self.scc32 = self.scf_flat.astype(np.int32), self.scc_flat.astype(np.int32)
        self.wcost = -np.log2(self.wf8 / P)
        self.sf = None
        self.K = 0
        self._wlut = self._slut = None
        self.blob = self.serialize()
        self.model_id = W.fnv1a32(self.blob) or 1

    @property
    def wlut(self):
        if self._wlut is None:
            R = self.wf8.shape[0]
            self._wlut = np.repeat(np.tile(np.arange(8, dtype=np.uint8), R), self.wf8_flat)
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
        hdr = struct.pack('<8I', W.WNM3_MAGIC, 1, PB, NCTX, 3, FLAG_Z, 0, 8)     # alphabet 8 = magnitude layout
        return hdr + cdf(self.wf8) + cdf(self.scf) + self.etab.astype('<u4').tobytes()


def fold_mag(c16):
    """[..., 16] nibble counts -> [..., 8] magnitude counts."""
    return c16[..., :8] + c16[..., 8:]


def train(sample_dir, counts_dir, out_npz, ref_models=None):
    """Z model from team 4's training counts (80 even-indexed experts, fg family = ceil-form, anchor 120 on all of them)."""
    man = W.load_manifest(sample_dir)
    cb = 0
    cs = 0
    n = 0
    anchors = []
    for i, e in enumerate(man['experts']):
        if i % 2:
            continue
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        cb = cb + z['fg_a'].reshape(3, NCTX, NSUB, 16).sum(2)
        cs = cs + z['fg_s']
        anchors.append(z['fg_anchor'].tolist())
        n += 1
    assert n == 80 and all(a == [ANCHOR0] * 3 for a in anchors)
    cm = fold_mag(cb)                                     # [3, 567, 8]
    wf8 = W.quantize(cm).reshape(-1, 8)
    scf = W.quantize(cs)
    acc = {}
    for tg in ('p0', 'p1'):
        z = np.load(os.path.join(counts_dir, f'accum_{tg}.npz'))
        for k in ('A_fg', 'SA_fg'):
            acc[k] = acc.get(k, 0) + z[k]
    A = acc['A_fg'].reshape(3, NDB, NCTX, NSUB, 16).sum(3)   # [3, 25, 567, 16]
    assert (A.sum(1) == cb).all(), 'accum_p0+p1 != sum of the 80 training experts'
    Am = fold_mag(A)                                      # [3, 25, 567, 8]
    nblk = acc['SA_fg'].sum(2)                            # [3, 25]
    cost = -np.log2(wf8.reshape(3, NCTX, 8) / P)
    etab = np.zeros((3, NDB))
    for m in range(3):
        bits = (Am[m].astype(np.float64) * cost[m][None]).sum((1, 2)) + Am[m][..., 1:].sum((1, 2))   # magnitudes + raw signs
        with np.errstate(invalid='ignore', divide='ignore'):
            e_m = bits / nblk[m]
        good = nblk[m] > 0
        idx = np.arange(NDB)
        gi = idx[good]
        near = gi[np.abs(idx[:, None] - gi[None, :]).argmin(1)]
        etab[m] = e_m[near]
    etab_q = np.round(etab * 256).astype(np.int64)
    nW = 80 * 3 * 2304 * 5120
    info = {'train_mag_bits_per_w': float((cm * cost).sum() / nW),
            'train_sign_bits_per_w': float(cm[..., 1:].sum() / nW),
            'train_zero_share': float(cm[..., 0].sum() / cm.sum()),
            'rows_with_counts': int((cm.sum(-1) > 0).sum())}
    info['train_z_bits_per_w'] = info['train_mag_bits_per_w'] + info['train_sign_bits_per_w']
    if ref_models:
        R = np.load(ref_models)
        info['scf_equals_ceilref_fg_c1'] = bool((R['fg_a0_K0_c1_scf'] == scf).all())
        wf16 = R['fg_a0_K0_c1_wf'].reshape(3, NCTX, 16)
        info['train_c1_16sym_bits_per_w (same counts, ceil-ref fg_a0_K0_c1 model)'] = float(
            (W.merge_c(cb) * -np.log2(wf16 / P)).sum() / nW) if hasattr(W, 'merge_c') else float(
            (_merge_c(cb) * -np.log2(wf16 / P)).sum() / nW)
    np.savez_compressed(out_npz, wf8=wf8, scf=scf, etab=etab_q, etab_float=etab)
    return info


def _merge_c(x):
    x = x.copy()
    x[..., 0] += x[..., 8]
    x[..., 8] = 0
    return x


def load_tables(npz):
    z = np.load(npz)
    return TablesZ(z['wf8'], z['scf'], z['etab'])


# ------------------------------------------------------------------------------------------------- encoder
def sign_planes(symT, z2):
    """symT [2048, WS] coded nibbles (stream order). Returns (planes [WS, 257] uint8 LSB-first, nplane_bits [WS],
    tail value t [WS], tail count r [WS], nsg [WS])."""
    S = symT.shape[1]
    q = np.ascontiguousarray(symT.T)                       # [WS, 2048]
    nz = (q & 7) != 0
    sg = (q >> 3).astype(np.uint8)
    assert not (sg.astype(bool) & ~nz).any(), 'non-canonical zero in the coded domain'
    nsg = nz.sum(1)
    k = np.cumsum(nz, 1) - 1
    rows, cols = np.nonzero(nz)
    if z2:
        nzl = nz[:, -32:].sum(1)
        r = np.minimum(nsg & 7, nzl)
    else:
        r = np.zeros(S, np.int64)
    nP = nsg - r
    kk = k[rows, cols]
    comp = np.zeros((S, 2048 + 8), np.uint8)
    comp[rows, kk] = sg[rows, cols]
    # tail value: bits nP .. nsg-1 -> t (bit i = sign index nP + i)
    t = np.zeros(S, np.int64)
    for i in range(7):
        sel = r > i
        if sel.any():
            t[sel] |= comp[np.flatnonzero(sel), nP[sel] + i].astype(np.int64) << i
    # zero the bits that are not in the plane (Z2), then pack
    if z2:
        pos = np.arange(2048 + 8)[None, :]
        comp[pos >= nP[:, None]] = 0
    planes = np.packbits(comp, axis=1, bitorder='little')   # [WS, 257]
    return planes, nP, t, r, nsg


def assemble_streams(rpay, rlens, planes, nP):
    """Per stream: rANS bytes, then ceil(nP/8) sign bytes in descending address order."""
    S = rlens.size
    nsb = (nP + 7) >> 3
    lens = rlens + nsb
    offs = np.zeros(S, np.int64)
    offs[1:] = np.cumsum(lens)[:-1]
    out = np.zeros(int(lens.sum()), np.uint8)
    roffs = np.zeros(S, np.int64)
    roffs[1:] = np.cumsum(rlens)[:-1]
    sid = np.repeat(np.arange(S), rlens)
    j = np.arange(rpay.size, dtype=np.int64)
    out[offs[sid] + (j - roffs[sid])] = rpay
    rows, cols = np.nonzero(np.arange(planes.shape[1])[None, :] < nsb[:, None])
    out[offs[rows] + lens[rows] - 1 - cols] = planes[rows, cols]
    return out, lens, nsb


def encode_record_z(mats, tab, layer, expert, d=True, e=True, z2=False, timing=None, keep=False):
    t0 = time.perf_counter()
    syms, scs, anchors, finfo = W.prepare(mats, True, True, True)
    rec_flags = W.FLAG_F | W.FLAG_C | FLAG_Z | (W.FLAG_E if e else 0) | (FLAG_Z2 if z2 else 0)
    if not finfo['applied']:
        rec_flags &= ~W.FLAG_F
    feats = [W.block_features(syms[m], scs[m], anchors[m]) for m in range(3)]
    meds = [f['med'] for f in feats]
    nb = [f['srel'].size for f in feats]
    nws = [n // BPS for n in nb]
    nss = [scs[m].size // S_CHUNK for m in range(3)]
    model_id = tab.model_id
    # ---- scales (as ceil-ref, kind 0)
    SS = sum(nss)
    scT = np.concatenate([scs[m].reshape(nss[m], S_CHUNK) for m in range(3)]).T
    smat = np.repeat(np.arange(3), nss)
    sidx = np.concatenate([np.arange(n) for n in nss])
    sidx_t = (smat[None, :] << 8) | scT
    sF, sC = tab.scf16[sidx_t], tab.scc16[sidx_t]
    x0s = np.full(SS, L, np.int64)
    if e:
        seed = W.hash_seeds(layer, expert, model_id, 0, smat, sidx, np.zeros(SS, np.int64))
        x0s = L + W.hash_symbols(seed, scT).astype(np.int64)
    spay, slens = W.rans_encode_lockstep(sF, sC, x0s)
    t1 = time.perf_counter()
    # ---- weights: magnitudes through rANS
    WS = sum(nws)
    FT = np.empty((2048, WS), np.uint16)
    CT = np.empty((2048, WS), np.uint16)
    symT = np.empty((2048, WS), np.uint8)
    FT3, CT3, sym3 = FT.reshape(BPS, 32, WS), CT.reshape(BPS, 32, WS), symT.reshape(BPS, 32, WS)
    col = 0
    est_mag = 0.0
    for m in range(3):
        f = feats[m]
        n = nws[m]
        s2 = syms[m].reshape(-1, 32)
        idx = ((f['base'].astype(np.int32) + np.int32(m * NCTX)) << 3) | (s2 & 7)
        est_mag += float(np.bincount(idx.ravel(), minlength=tab.wf8_flat.size) @ tab.wcost.ravel())
        FT3[:, :, col:col + n] = tab.wf16[idx].reshape(n, BPS, 32).transpose(1, 2, 0)
        CT3[:, :, col:col + n] = tab.wc16[idx].reshape(n, BPS, 32).transpose(1, 2, 0)
        sym3[:, :, col:col + n] = s2.reshape(n, BPS, 32).transpose(1, 2, 0)
        col += n
    wmat = np.repeat(np.arange(3), nws)
    widx = np.concatenate([np.arange(n) for n in nws])
    planes, nP, tval, rtail, nsg = sign_planes(symT, z2)
    x0w = np.full(WS, L, np.int64)
    if e:
        seed = W.hash_seeds(layer, expert, model_id, 1, wmat, widx, np.zeros(WS, np.int64))
        x0w = L + W.hash_symbols(seed, symT).astype(np.int64)
    if z2:
        x0w = x0w + (tval << 23)
    t2 = time.perf_counter()
    rpay, rlens = W.rans_encode_lockstep(FT, CT, x0w)
    del FT, CT
    t3 = time.perf_counter()
    wpay, wlens, nsb = assemble_streams(rpay, rlens, planes, nP)
    assert (slens <= 65535).all() and (wlens <= 65535).all()
    pad_bits = nsb * 8 - nP
    base_info = {'meds': meds, 'ks': [0, 0, 0], 'kcost': None, 'scale_payload': int(slens.sum()),
                 'weight_payload': int(wlens.sum()), 'n_scale_streams': SS, 'n_weight_streams': WS,
                 'est_weight_bits': est_mag + float(nsg.sum()), 'est_side_bits': 0.0, 'est_scale_bits':
                     float(-np.log2(sF / P).sum()), 'slens': slens, 'wlens': wlens, 'rec_flags': rec_flags, 'ceil': finfo,
                 'z': {'rans_bytes': int(rlens.sum()), 'sign_plane_bytes': int(nsb.sum()), 'sign_bits': int(nsg.sum()),
                       'plane_bits': int(nP.sum()), 'x0_tail_bits': int(rtail.sum()),
                       'pad_bits': int(pad_bits.sum()), 'est_mag_bits': est_mag,
                       'tail_corner_streams (r < nsg&7)': int((rtail < (nsg & 7)).sum()) if z2 else 0,
                       'mean_rans_stream': float(rlens.mean()), 'mean_plane_bytes': float(nsb.mean())}}
    st = {'meds': meds, 'ks': [0, 0, 0], 'slens': slens, 'wlens': wlens, 'spay': spay, 'wpay': wpay, 'smat': smat,
          'wmat': wmat, 'dbins': [feats[m]['dbin'].reshape(nws[m], BPS) for m in range(3)], 'nws': nws,
          'rec_flags': rec_flags}
    cfg = W.Config('z', d=d)
    rec, info = W.build_record(st, tab, cfg, base_info)
    t4 = time.perf_counter()
    if timing is not None:
        timing.update({'enc_scales_s': t1 - t0, 'enc_prep_w_s': t2 - t1, 'enc_rans_w_s': t3 - t2,
                       'enc_pack_s': t4 - t3, 'enc_total_s': t4 - t0})
    if keep:
        info['_keep'] = {'symT': symT, 'rpay': rpay, 'rlens': rlens, 'x0w': x0w, 'feats': feats, 'syms': syms,
                         'scs': scs, 'nws': nws, 'planes': planes, 'nP': nP, 'nsg': nsg}
    return rec, info


# ------------------------------------------------------------------------------------------------- decoder
def decode_weights_z(pl, offs, lens, srelT, ctxoff, tab, hseed, z2):
    """Lockstep decoder of Z weight streams. pl = payload with FRONT bytes of front padding and tail padding;
    offs are offsets into the unpadded payload. Returns (nibbles [2048, S] uint8, ok [S] bool, stats)."""
    S = offs.size
    po = offs + FRONT
    x, ok = W._read_state(pl, po)
    ok &= (lens >= 4)
    ptr = (po + 4).astype(np.int32)
    send = (po + lens).astype(np.int64)
    LUT, WF, WC = tab.wlut, tab.wf32, tab.wc32
    out = np.empty((2048, S), np.uint8)
    k = np.zeros(S, np.int64)
    ctxoff = ctxoff.astype(np.int32)
    for b in range(BPS):
        rb = ctxoff + srelT[b].astype(np.int32) * np.int32(81)
        for j in range(32):
            ctx = rb + np.int32(80) if j == 0 else rb + pmx
            slot = x & (P - 1)
            s = LUT[(ctx << PB) | slot]
            idx = (ctx << 3) | s
            x = WF[idx] * (x >> PB) + slot - WC[idx]
            x = W._renorm(x, ptr, pl)
            nz = s != 0
            byte = pl[send - 1 - (k >> 3)]
            sgn = ((byte >> (k & 7).astype(np.uint8)) & 1) & nz
            out[b * 32 + j] = s | (sgn << 3).astype(np.uint8)
            k += nz
            mag = s.astype(np.int32)
            mx = mag if j == 0 else np.maximum(mx, mag)
            pmx = mag * np.int32(9) + mx
    nsg = k
    if z2:
        nzl = ((out[-32:] & 7) != 0).sum(0)
        r = np.minimum(nsg & 7, nzl)
        v = x.astype(np.int64) - L
        t = v >> 23
        ok &= (v >= 0) & (t < (np.int64(1) << r))
        nP = nsg - r
        # the last r nonzeros of the last block take their signs from t (bit i -> sign index nP + i)
        lastnz = ((out[-32:] & 7) != 0)
        rank = np.cumsum(lastnz, 0) - 1 + (nsg - nzl)[None, :]      # global sign index of each last-block nonzero
        take = lastnz & (rank >= nP[None, :])
        bit = ((t[None, :] >> np.clip(rank - nP[None, :], 0, 6)) & 1).astype(np.uint8)
        blk = out[-32:]
        out[-32:] = np.where(take, (blk & 7) | (bit << 3), blk)
    else:
        nP = nsg
        r = np.zeros(S, np.int64)
        v = x.astype(np.int64) - L
    nsb = (nP + 7) >> 3
    ok &= (ptr == send - nsb)
    part = nP & 7
    lastb = pl[send - 1 - (nP >> 3)].astype(np.int64)
    ok &= np.where(part > 0, (lastb >> part) == 0, True)
    if hseed is not None:
        h23 = W.hash_symbols(hseed, out).astype(np.int64)
        ok &= ((v & M23) == h23) if z2 else (v == h23)
    else:
        ok &= (v == 0) if not z2 else ((v & M23) == 0)
    return out, ok, {'nsg': nsg, 'nP': nP, 'r': r}


def decode_record_z(rec, tab, layer, expert, shapes, timing=None):
    t0 = time.perf_counter()
    rec = np.frombuffer(rec, np.uint8)
    if rec.size % W.ALIGN:
        raise W.DecodeError('size not 4096-aligned')
    (magic, mid, spb, wpb, wch, sch, m0, m1, m2, flags, k0, k1, k2, K, ltb,
     a0, a1, a2, b0, b1, b2) = struct.unpack_from('<6I3BB3BBI6H', rec, 0)
    if magic != W.WNR3_MAGIC:
        raise W.DecodeError('bad magic')
    if mid != tab.model_id:
        raise W.DecodeError('model id mismatch')
    if (wch, sch) != (W_CHUNK, S_CHUNK) or K or k0 or k1 or k2:
        raise W.DecodeError('chunk / K')
    known = W.FLAG_C | W.FLAG_D | W.FLAG_E | W.FLAG_F | FLAG_Z | FLAG_Z2
    if flags & ~known or not (flags & FLAG_Z) or not (flags & W.FLAG_C):
        raise W.DecodeError('flags')
    z2 = bool(flags & FLAG_Z2)
    alphas, sbase, hlen = [a0, a1, a2], [b0, b1, b2], 48
    meds = [m0, m1, m2]
    nsym = [R * C for R, C in shapes]
    nws = [n // W_CHUNK for n in nsym]
    nss = [n // 32 // S_CHUNK for n in nsym]
    SS, WS = sum(nss), sum(nws)
    p = hlen
    if not (flags & W.FLAG_D):
        lt = np.frombuffer(rec, '<u2', SS + WS, p).astype(np.int64)
        slens, wlens = lt[:SS], lt[SS:]
        p += 2 * (SS + WS)
        if ltb != 2 * (SS + WS):
            raise W.DecodeError('length table size')
        wres = None
    else:
        segs, used = W.unpack_groups(rec[p:p + ltb].tobytes(), nss + nws)
        if used != ltb:
            raise W.DecodeError('length table size')
        slens = np.concatenate([segs[m] + sbase[m] for m in range(3)])
        wres = segs[3:]
        p += ltb
    if (slens < 4).any() or slens.sum() != spb:
        raise W.DecodeError('scale lengths')
    sbeg, wbeg = p, p + spb
    used_end = wbeg + wpb
    if W.align(used_end) != rec.size or rec[used_end:].any():
        raise W.DecodeError('size / padding')
    smat = np.repeat(np.arange(3), nss)
    sidx = np.concatenate([np.arange(n) for n in nss])
    pad = np.zeros(2 * (W_CHUNK + BPS) + 16, np.uint8)
    pl_s = np.concatenate([rec[sbeg:wbeg], pad])
    soffs = np.zeros(SS, np.int64)
    soffs[1:] = np.cumsum(slens)[:-1]
    hs = W.hash_seeds(layer, expert, mid, 0, smat, sidx, np.zeros(SS, np.int64)) if flags & W.FLAG_E else None
    t1 = time.perf_counter()
    scT, sok = W.decode_scales_lockstep(pl_s, soffs, slens, S_CHUNK, smat, tab, hs, 0)
    if not sok.all():
        raise W.DecodeError(f'scale streams failed: {int((~sok).sum())}')
    scs, a = [], 0
    for m in range(3):
        scs.append(np.ascontiguousarray(scT[:, a:a + nss[m]].T).ravel())
        a += nss[m]
    srelT = np.empty((BPS, WS), np.int64)
    col, wl = 0, []
    for m in range(3):
        dd = scs[m].astype(np.int16) - meds[m]
        srelT[:, col:col + nws[m]] = (np.clip(dd, -3, 3) + 3).reshape(nws[m], BPS).T
        if wres is not None:
            dbin = (np.clip(dd, -DBC, DBC) + DBC).reshape(nws[m], BPS)
            wl.append(W.pred_from_q(W.predict_wlen_q(dbin, tab.etab[m]), alphas[m]) + wres[m])
        col += nws[m]
    if wres is not None:
        wlens = np.concatenate(wl)
    if (wlens < 4).any() or (wlens > 65535).any() or wlens.sum() != wpb:
        raise W.DecodeError('weight lengths')
    pl_w = np.concatenate([np.zeros(FRONT, np.uint8), rec[wbeg:used_end], pad])
    woffs = np.zeros(WS, np.int64)
    woffs[1:] = np.cumsum(wlens)[:-1]
    wmat = np.repeat(np.arange(3), nws)
    widx = np.concatenate([np.arange(n) for n in nws])
    hw = W.hash_seeds(layer, expert, mid, 1, wmat, widx, np.zeros(WS, np.int64)) if flags & W.FLAG_E else None
    t2 = time.perf_counter()
    wT, wok, zst = decode_weights_z(pl_w, woffs, wlens, srelT, wmat.astype(np.int64) * NCTX, tab, hw, z2)
    t3 = time.perf_counter()
    if not wok.all():
        raise W.DecodeError(f'weight streams failed: {int((~wok).sum())}')
    wparts, col = [], 0
    for m in range(3):
        sym = np.ascontiguousarray(wT[:, col:col + nws[m]].T).ravel()
        if flags & W.FLAG_F:
            try:
                sym, scs[m] = W.ceil_inverse(sym, scs[m])
            except ValueError as ex:
                raise W.DecodeError(str(ex))
        wparts.append((sym[0::2] | (sym[1::2] << 4)).astype(np.uint8))
        col += nws[m]
    slab = np.concatenate(scs + wparts)
    if timing is not None:
        timing.update({'dec_scales_s': t1 - t0 + 0.0, 'dec_rans_w_s': t3 - t2, 'dec_total_s': time.perf_counter() - t0})
    return slab, {'flags': flags, 'nsg': zst['nsg'], 'wok': wok}


def canon_bytes(raw, nscale_bytes):
    """Copy of run_ceil.canon_bytes: every FP4 nibble 8 (-0) -> 0 (+0)."""
    out = raw.copy()
    w = out[nscale_bytes:]
    lo = w & 15
    hi = w >> 4
    lo = np.where(lo == 8, 0, lo)
    hi = np.where(hi == 8, 0, hi)
    out[nscale_bytes:] = (lo | (hi << 4)).astype(np.uint8)
    return out
