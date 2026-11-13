"""Test kit / audit (from team5 audit-ceil): experts, clean records and the record-level decoder (path R).

decode_r() runs ceil-ref decode_record's order of operations (parse header + length section, size/padding, scale streams,
LENC weight lengths from the decoded coded scales, weight streams, ceil inverse) plus DIGEST, on a possibly modified
record. With use_clean=True it decodes only the streams whose byte range, offset, length, contexts (srel) or identity
differ from the clean record (every other stream provably decodes exactly as in the clean record, which make_records.py
verified byte-exact), so it is exact and fast. use_clean=False decodes every stream (full decoder, builds the slab).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as A
from audit_lib import W, NWS, NSS, S_CHUNK, W_CHUNK, BPS, DecodeError

DK_OF_KIND = {'fg0d': 'fg0d', 'fg0de': 'fg0de', 'fg0dh': 'fg0dh', 'fg0dt': 'fg0dt', 'fg0dtD': 'fg0dt'}
RULE_OF_KIND = {'fg0d': 'R0', 'fg0de': 'HASH', 'fg0dh': 'HASH32', 'fg0dt': 'TAIL', 'fg0dtD': 'TAILD'}
SMAT = np.repeat(np.arange(3), NSS)
SIDX = np.tile(np.arange(NSS), 3)
WMAT = np.repeat(np.arange(3), NWS)
WIDX = np.tile(np.arange(NWS), 3)


class CleanRec:
    def __init__(self, E, kind, rec, tab):
        self.kind = kind
        self.rec = np.frombuffer(rec, np.uint8) if not isinstance(rec, np.ndarray) else rec
        self.H = A.parse_record(self.rec, tab, E.shapes)
        self.sstart = self.H['sbeg'] + self.H['soffs']
        self.slens = self.H['slens']
        self.wlens, wo = A.weight_lengths(self.H, E.cs, tab)
        self.wstart = self.H['wbeg'] + wo

    def wbytes(self, m, j):
        g = m * NWS + j
        return self.rec[self.wstart[g]:self.wstart[g] + self.wlens[g]]

    def sbytes(self, m, j):
        g = m * NSS + j
        return self.rec[self.sstart[g]:self.sstart[g] + self.slens[g]]


class Expert:
    def __init__(self, idx, tab, kinds=A.KINDS, rec_dir=None):
        rec_dir = A.REC_DIR if rec_dir is None else rec_dir      # (kit fix: the default was bound at import time)
        src = A.load_source(idx)
        self.idx, self.file, self.layer, self.expert = idx, src['file'], src['layer'], src['expert']
        self.stem = self.file[:-4]
        self.shapes = src['shapes']
        self.raw = src['raw']
        self.sha_ok = src['sha_ok']
        self.cs = [src['scs'][m].astype(np.int16) for m in range(3)]                  # coded scales, flat per matrix
        self.csym = [src['syms'][m].reshape(NWS, W_CHUNK) for m in range(3)]          # coded nibbles per weight stream
        self.srcn = [src['mats'][m]['sym'].reshape(NWS, W_CHUNK) for m in range(3)]   # source nibbles
        self.srcs = [src['mats'][m]['sc'].astype(np.int16) for m in range(3)]         # source scales, flat
        self.srel = [A.srel_of(self.cs[m], A.ANCHOR0).reshape(NWS, BPS) for m in range(3)]
        if 'fg0dt' in kinds or 'fg0dtD' in kinds:                # per-stream H32 (flags 0xB4), only for TAILD
            self.hs_d, self.hw_d = A.stream_h32_all(src, tab.model_id, 0xB4)
        else:
            self.hs_d = self.hw_d = None
        self.recs = {}
        for k in kinds:
            p = os.path.join(rec_dir, f'{self.stem}_{k}.rec')
            if not os.path.exists(p):
                raise FileNotFoundError(f'clean record missing: {p}')
            self.recs[k] = CleanRec(self, k, np.fromfile(p, np.uint8), tab)

    def cs_stream(self, m, j):
        return self.cs[m][j * BPS:(j + 1) * BPS]

    def srcs_stream(self, m, j):
        return self.srcs[m][j * BPS:(j + 1) * BPS]


def _changed(diff, start, ln):
    if diff is None:
        return np.ones(start.size, bool)
    return np.searchsorted(diff, start) < np.searchsorted(diff, start + ln)


def _bad_init(rec, start, ln):
    """Exact early exit: a lane whose 4-B initial state is outside [L, 256L) or whose length < 4 fails in the decoder
    (W._read_state / lens >= 4), so the record is rejected whatever the other lanes do."""
    st = np.clip(start, 0, max(0, rec.size - 4))
    pad = np.concatenate([rec, np.zeros(4, np.uint8)])
    x = (pad[st].astype(np.int64) << 24) | (pad[st + 1].astype(np.int64) << 16) | (pad[st + 2].astype(np.int64) << 8) | pad[st + 3]
    return (x < A.L) | (x >= (A.L << 8)) | (ln < 4) | (start + 4 > rec.size)


def decode_r(E, kind, rec, tab, bind=None, use_clean=True, build_slab=False):
    """Decode record bytes `rec` (np.uint8) of format `kind` in the slot `bind` = (layer, expert) (default E's own).
    Returns dict: status 'accepted' | 'rejected', stage, msg, ceilref_reject (bool), n_dec_s / n_dec_w (streams decoded),
    and for accepted records the damage vs E's source (aggregated over affected streams) and, with build_slab, the slab."""
    C = E.recs[kind]
    dk = DK_OF_KIND[kind]
    rule = A.RULES_OF_DEC[dk][0]
    lay, exp = bind if bind is not None else (E.layer, E.expert)
    same_id = (lay, exp) == (E.layer, E.expert)
    out = {'status': 'rejected', 'stage': None, 'msg': None, 'ceilref_reject': None, 'n_dec_s': 0, 'n_dec_w': 0}
    try:
        H = A.parse_record(rec, tab, E.shapes)
    except DecodeError as ex:
        out['stage'], out['msg'] = ex.args[0], ex.args[1]
        return out
    diff = None
    if use_clean and rec.size == C.rec.size:
        diff = np.flatnonzero(rec != C.rec)
    id_rule = dk in ('fg0de', 'fg0dh')
    # ---- scale streams
    sstart = H['sbeg'] + H['soffs']
    slen = H['slens']
    if diff is None:
        sch = np.ones(sstart.size, bool)
    else:
        sch = (sstart != C.sstart) | (slen != C.slens) | _changed(diff, sstart, slen)
    if id_rule and not same_id:
        sch[:] = True
    cs2 = [E.cs[m].copy() for m in range(3)]
    si = np.flatnonzero(sch)
    hs_new = {}
    if si.size and _bad_init(rec, sstart[si], slen[si]).any():
        out['stage'], out['msg'] = 'scales', 'scale stream initial state / length (early exit)'
        return out
    if si.size:
        dl = [rec[sstart[g]:sstart[g] + slen[g]] for g in si]
        ids = {'lay': np.full(si.size, lay), 'exp': np.full(si.size, exp), 'm': SMAT[si], 'j': SIDX[si]}
        res, outs = A.run_lanes(dk, 's', dl, slen[si], None, SMAT[si], ids, tab)
        out['n_dec_s'] = int(si.size)
        if not res[rule].all():
            out['stage'], out['msg'] = 'scales', f'scale streams failed: {int((~res[rule]).sum())}'
            return out
        for t, g in enumerate(si):
            m, j = SMAT[g], SIDX[g]
            cs2[m][j * S_CHUNK:(j + 1) * S_CHUNK] = outs[t]
        if 'TAILD_h' in res:
            hs_new = dict(zip(si.tolist(), res['TAILD_h'].tolist()))
    # ---- weight lengths (LENC, from coded scales)
    try:
        wl, wo = A.weight_lengths(H, cs2, tab)
    except DecodeError as ex:
        out['stage'], out['msg'] = ex.args[0], ex.args[1]
        return out
    wstart = H['wbeg'] + wo
    srel2 = [A.srel_of(cs2[m], H['meds'][m]).reshape(NWS, BPS) for m in range(3)]
    if diff is None:
        wch = np.ones(wstart.size, bool)
    else:
        sr_ch = np.concatenate([(srel2[m] != E.srel[m]).any(1) for m in range(3)])
        wch = (wstart != C.wstart) | (wl != C.wlens) | _changed(diff, wstart, wl) | sr_ch
    if id_rule and not same_id:
        wch[:] = True
    wi = np.flatnonzero(wch)
    dec_w = {}
    hw_new = {}
    if wi.size and _bad_init(rec, wstart[wi], wl[wi]).any():
        out['stage'], out['msg'] = 'weights', 'weight stream initial state / length (early exit)'
        return out
    if wi.size:
        dl = [rec[wstart[g]:wstart[g] + wl[g]] for g in wi]
        srel = np.stack([srel2[WMAT[g]][WIDX[g]] for g in wi])
        ids = {'lay': np.full(wi.size, lay), 'exp': np.full(wi.size, exp), 'm': WMAT[wi], 'j': WIDX[wi]}
        res, outs = A.run_lanes(dk, 'w', dl, wl[wi], srel, WMAT[wi], ids, tab)
        out['n_dec_w'] = int(wi.size)
        if not res[rule].all():
            out['stage'], out['msg'] = 'weights', f'weight streams failed: {int((~res[rule]).sum())}'
            return out
        dec_w = {int(g): outs[t] for t, g in enumerate(wi)}
        if 'TAILD_h' in res:
            hw_new = dict(zip(wi.tolist(), res['TAILD_h'].tolist()))
    # ---- DIGEST (TAILD records)
    if H['digest'] is not None:
        if same_id:
            hs, hw = E.hs_d.copy().ravel(), E.hw_d.copy().ravel()
        else:   # unchanged streams decode to the clean coded data; their h_g under the slot identity:
            hs = np.concatenate([A.h32_streams(tab.model_id, lay, exp, A.gs(m, np.arange(NSS)), m, 0xB4,
                                               E.cs[m].astype(np.uint8).reshape(NSS, S_CHUNK)) for m in range(3)])
            hw = np.concatenate([A.h32_streams(tab.model_id, lay, exp, A.gw(m, np.arange(NWS)), m, 0xB4,
                                               A.pack_nib(E.csym[m])) for m in range(3)])
        for g, h in hs_new.items():
            hs[g] = h
        for g, h in hw_new.items():
            hw[g] = h
        dg = int(np.bitwise_xor.reduce(hs) ^ np.bitwise_xor.reduce(hw))
        if dg != H['digest']:
            out['stage'], out['msg'] = 'digest', 'record digest mismatch'
            return out
    # ---- ceil inverse on affected streams
    sc_ch = np.concatenate([(cs2[m] != E.cs[m]).reshape(NWS, BPS).any(1) for m in range(3)])
    aff = np.flatnonzero(wch | sc_ch) if diff is not None else np.arange(3 * NWS)
    dmg_tot = None
    inv_spec = inv_cr = False
    if aff.size:
        decs = np.stack([dec_w[int(g)] if int(g) in dec_w else E.csym[WMAT[g]][WIDX[g]] for g in aff])
        cl = np.stack([E.csym[WMAT[g]][WIDX[g]] for g in aff])
        csb = np.stack([cs2[WMAT[g]][WIDX[g] * BPS:(WIDX[g] + 1) * BPS] for g in aff])
        ccsb = np.stack([E.cs[WMAT[g]][WIDX[g] * BPS:(WIDX[g] + 1) * BPS] for g in aff])
        sn = np.stack([E.srcn[WMAT[g]][WIDX[g]] for g in aff])
        ss = np.stack([E.srcs[WMAT[g]][WIDX[g] * BPS:(WIDX[g] + 1) * BPS] for g in aff])
        dmg = {}
        CH = 2048
        for a in range(0, aff.size, CH):
            d = A.damage(decs[a:a + CH], csb[a:a + CH], cl[a:a + CH], ccsb[a:a + CH], sn[a:a + CH], ss[a:a + CH])
            for k, v in d.items():
                dmg.setdefault(k, []).append(v)
        dmg = {k: np.concatenate(v) for k, v in dmg.items()}
        inv_spec, inv_cr = bool(dmg['inv_spec'].any()), bool(dmg['inv_cr'].any())
        dmg_tot = agg_damage(dmg)
        if build_slab:
            on, osc, _, _, _ = A.inverse(decs, csb)
            out['_slab_parts'] = (aff, on, osc)
    out['ceilref_reject'] = inv_cr
    if inv_spec:
        out['stage'], out['msg'] = 'ceil', 'cs == 0 in a halved block'
        return out
    out['status'] = 'accepted'
    out['damage'] = dmg_tot
    out['n_aff_streams'] = int(aff.size)
    if build_slab:
        aff, on, osc = out.pop('_slab_parts')
        assert aff.size == 3 * NWS
        sc_parts, w_parts = [], []
        for m in range(3):
            sel = slice(m * NWS, (m + 1) * NWS)
            sc_parts.append(osc[sel].reshape(-1).astype(np.uint8))
            sy = on[sel].reshape(-1)
            w_parts.append((sy[0::2] | (sy[1::2] << 4)).astype(np.uint8))
        out['slab'] = np.concatenate(sc_parts + w_parts)
    return out


def agg_damage(d):
    """Aggregate per-stream damage arrays of one trial to totals / maxima."""
    return {'coded_sym_diff': int(d['coded_sym_diff'].sum()), 'coded_scale_diff': int(d['coded_scale_diff'].sum()),
            'nib_diff': int(d['nib_diff'].sum()), 'val_diff': int(d['val_diff'].sum()), 'scale_diff': int(d['scale_diff'].sum()),
            'block_flips': int(d['block_flips'].sum()), 'err_units': float(d['err_units'].max()), 'err_abs': float(d['err_abs'].max()),
            'sign_only': bool(d['sign_only'].any() and (d['nib_diff'][~d['sign_only']] == 0).all() and d['scale_diff'].sum() == 0),
            'nan_scale': bool(d['nan_scale'].any())}
