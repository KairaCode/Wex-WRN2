"""team5/gpu-v1: encode the 8 team-2/team-4 GPU experts as WNR3 v1-profile streams (CEIL | LENC | HASH, anchor g0 = 120).

Usage: python -I v1enc.py OUT_DIR

Model: team 4 ceil-ref's held-out ceil model 'fg_a0_K0_c0' (trained on the 80 even-indexed sample experts; densest-pair
anchor = 120 on all 240 training matrices, so it is exactly the g0 = 120 model [C ceil-ref run_fixed_anchor.py:2-3]),
read from partial/_shared/ceil-ref/models/even.npz (copy of team 4 ceil-ref, unmodified) and compacted from 567 to 259 rows (exact).
WNM3 blob of the spec (3.1-3.2, layout 0, K = 1, LENPRED) is built only to derive model_id = FNV-1a 32 (0 -> 1).

Per expert / matrix (V4.1 shapes, Cw = 2048, Cs = 512):
  ceil transform T (spec 5.2; gate per matrix: no source block with max 5),
  contexts on CODED data with anchor 120, rANS streams:
    G   : x0 = L                          (CEIL; the same streams serve CEIL|LENC = 0x14)
    H18 : x0 = L + h23, flags 0x18        (CEIL|HASH)
    H1C : x0 = L + h23, flags 0x1C        (CEIL|LENC|HASH = v1 default)
  h23 = H32(ident || coded bytes, seed 0x33524E57) & 0x7FFFFF (spec 5.1), ident = model_id, layer<<16|expert,
  g | ceil<<31, flags | m<<8 | set<<16 | anchor<<24; g = global stream index (scale streams of m0..m2, then weights).
  LENC section (spec 4.3, LENC = 1): u16 sbase[3], u16 alpha[3], then scale part and weight part, each
  u32 base_off[ng], u8 width[ng], pad4, u32 words[sum(width)] (zigzag residuals, LSB-first, 32-stream groups per
  matrix), predictor pred = sbase[m] (scales) / 4 + ((Q*alpha + 2^25) >> 26), Q = sum etab[m][dbin] (weights).
"""
import hashlib, json, os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wnr2fmt as W
import h32 as H

SAMPLE = os.environ.get('WNR_SAMPLE', 'v41-sample')  # kit: dir with the 160-expert sample (manifest.json + L*_E*.bin)
MODEL_NPZ = os.environ.get('WNR_CEILREF_MODEL') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '_shared', 'ceil-ref', 'models', 'even.npz')
MODEL_KEY = 'fg_a0_K0_c0'
PICK = [17, 0, 40, 80, 120, 159, 60, 100]   # team 4 ceil-gpu / team 2 gpu-proto experts
ANCHOR = 120
SET_ID = 0
SPS = 512
NDB, DBC = 25, 12
P, L = 16384, 1 << 23
INDEX_SHA = 'e0aebac7d1b9b257737b92d1065bde74243f7f8f617a1619b077f6e6639d0738'
HALF = np.array([0, 255, 1, 255, 2, 3, 4, 5], np.uint8)
DOUBLE = np.array([0, 2, 4, 5, 6, 7, 255, 255], np.uint8)
FLAGS = {'G': 0x10, 'G14': 0x14, 'H18': 0x18, 'H1C': 0x1C}


def fnv1a32(b):
    a = np.frombuffer(b, np.uint8)
    h = 2166136261
    for x in a.tolist():
        h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return h


def canonical_index():
    idx = np.full(567, 0xFFFF, np.uint16)
    for srel in range(7):
        for pm in range(9):
            for mx in range(9):
                full = (srel * 9 + pm) * 9 + mx
                if pm == 8 and mx == 8:
                    idx[full] = srel * 37 + 36
                elif pm < 8 and mx < 8 and pm <= mx:
                    idx[full] = srel * 37 + mx * (mx + 1) // 2 + pm
    return idx


def build_wnm3(cum259_17, scum, etab, anchor0):
    """spec 3.1-3.2: layout 0, K = 1, LENPRED. cum259_17 [3,259,17] u16, scum [3,257] u16, etab [3,25]."""
    idx = canonical_index()
    ib = idx.astype('<u2').tobytes()
    ib += b'\0' * (-len(ib) % 4)
    sets_off = 64 + len(ib)
    sb = struct.pack('<BB3B3B', 1, 0, *anchor0, 0, 0, 0) + cum259_17.astype('<u2').tobytes() + scum.astype('<u2').tobytes()
    sb += b'\0' * (-len(sb) % 4)
    set_bytes = len(sb)
    lenpred_off = sets_off + set_bytes
    eb = np.asarray(etab, np.int64).astype('<u4').tobytes()
    total = lenpred_off + len(eb)
    hdr = struct.pack('<I8H7I4I', 0x334D4E57, 1, 14, 0, 1, 567, 259, 3, 16, 1, 64, sets_off, set_bytes, lenpred_off,
                      total, 1, 0, 0, 0, 0)
    assert len(hdr) == 64
    blob = hdr + ib + sb + eb
    assert len(blob) == total
    return blob, idx


def ceil_forward(sym, sc):
    """sym flat (row-major nibbles), sc flat (one per block). Spec 5.2 T with skip-255."""
    blk = sym.reshape(-1, 32)
    mag = blk & 7
    bmax = mag.max(1)
    has13 = ((mag == 1) | (mag == 3)).any(1)
    even = (bmax == 7) & ~has13 & (sc <= 254)
    tb = np.where(even[:, None], (blk & 8) | HALF[mag], blk).astype(np.uint8)
    ts = (sc.astype(np.int16) + even).astype(np.uint8)
    st = {'blocks': int(even.size), 'even': int(even.sum()), 'max5_blocks_gate': int((bmax == 5).sum()),
          'max_below6': int((bmax < 6).sum()), 'even_scale255_skipped': int(((bmax == 7) & ~has13 & (sc == 255)).sum())}
    return tb.ravel(), ts, even, st


def ceil_inverse(tsym, tsc):
    blk = tsym.reshape(-1, 32)
    mag = blk & 7
    odd = mag.max(1) == 5
    assert not (tsc[odd] == 0).any()
    ob = np.where(odd[:, None], (blk & 8) | DOUBLE[mag], blk).astype(np.uint8)
    return ob.ravel(), (tsc.astype(np.int16) - odd).astype(np.uint8)


def zigzag(r):
    r = np.asarray(r, np.int64)
    return np.where(r >= 0, 2 * r, -2 * r - 1)


def unzigzag(z):
    z = np.asarray(z, np.int64)
    return np.where(z & 1, -(z >> 1) - 1, z >> 1)


def pred_from_q(Q, a):
    return 4 + ((np.asarray(Q, np.int64) * int(a) + (1 << 25)) >> 26)


def fit_alpha(Q, lens):
    """copy of ceil-ref wnr_common.py:582-599 (coarse step 4 within +-64 of least squares, then +-4)."""
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
    """spec 4.3 part: u32 base_off[ng], u8 width[ng], pad4, u32 words[sum(width)]. Groups of 32 per matrix."""
    base, widths, words = [], [], []
    acc = 0
    for lens, preds in zip(lens_per_m, preds_per_m):
        n = lens.size
        ng = -(-n // 32)
        z = np.zeros(ng * 32, np.int64)
        z[:n] = zigzag(lens - preds)
        lp = np.zeros(ng * 32, np.int64); lp[:n] = lens
        for g in range(ng):
            base.append(acc)
            acc += int(lp[32 * g:32 * g + 32].sum())
            zg = z[32 * g:32 * g + 32]
            w = int(zg.max()).bit_length()
            assert w <= 17, ('width > 17', w)
            widths.append(w)
            if w:
                bits = 0
                for i in range(32):
                    bits |= int(zg[i]) << (i * w)
                for k in range(w):
                    words.append((bits >> (32 * k)) & 0xFFFFFFFF)
    b = np.array(base, '<u4').tobytes() + np.array(widths, np.uint8).tobytes()
    b += b'\0' * (-len(b) % 4)
    b += np.array(words, '<u4').tobytes()
    return b, acc, widths


def unpack_part_ref(buf, ng_per_m, n_per_m, preds_per_m):
    """independent CPU reader of one part (bit by bit). Returns lengths per matrix and the bytes used."""
    ng = sum(ng_per_m)
    base = np.frombuffer(buf, '<u4', ng, 0)
    widths = np.frombuffer(buf, np.uint8, ng, 4 * ng).astype(np.int64)
    p = 4 * ng + ng + (-ng % 4)
    wbase = p // 4
    words = np.frombuffer(buf, '<u4', (len(buf) - p) // 4, p)
    out, gi, woff = [], 0, 0
    for ngm, n, preds in zip(ng_per_m, n_per_m, preds_per_m):
        ln = np.zeros(ngm * 32, np.int64)
        for g in range(ngm):
            w = int(widths[gi])
            for i in range(32):
                z = 0
                for k in range(w):
                    bit = i * w + k
                    z |= ((int(words[woff + (bit >> 5)]) >> (bit & 31)) & 1) << k
                j = 32 * g + i
                if j < n:
                    ln[j] = int(preds[j]) + int(unzigzag(z))
                else:
                    assert z == 0
            woff += w
            gi += 1
        out.append(ln[:n])
    return out, base, p + 4 * woff


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    rep = {'pick': PICK, 'anchor': ANCHOR, 'model': f'{MODEL_NPZ}:{MODEL_KEY}'}
    # ------------------------------------------------------------------ model
    Z = np.load(MODEL_NPZ)
    wf567 = Z[MODEL_KEY + '_wf'].reshape(3, 567, 16).astype(np.int64)
    scf = Z[MODEL_KEY + '_scf'].astype(np.int64)
    etab = Z[MODEL_KEY + '_etab'].astype(np.int64)
    assert (wf567.sum(-1) == P).all() and (scf.sum(-1) == P).all() and (etab <= 114688).all()
    idx = canonical_index()
    mp = W.ctx567_to_259()
    assert (idx.astype(np.int64)[mp >= 0] == mp[mp >= 0]).all() and (idx[mp < 0] == 0xFFFF).all()
    f259 = np.zeros((3, 259, 16), np.int64)
    f259[:, mp[mp >= 0]] = wf567[:, mp >= 0]
    assert (f259.sum(-1) == P).all() and (f259 >= 1).all()
    cum259 = np.zeros((3, 259, 17), np.int64); cum259[..., 1:] = np.cumsum(f259, -1)
    scum = np.zeros((3, 257), np.int64); scum[:, 1:] = np.cumsum(scf, -1)
    slut = np.stack([np.repeat(np.arange(256), scf[m]).astype(np.uint8) for m in range(3)])
    blob, _ = build_wnm3(cum259, scum, etab, [ANCHOR] * 3)
    idx_sha = hashlib.sha256(idx.astype('<u2').tobytes()).hexdigest()
    assert idx_sha == INDEX_SHA, idx_sha
    model_id = fnv1a32(blob) or 1
    rep['wnm3'] = {'bytes': len(blob), 'index_sha256_ok': idx_sha == INDEX_SHA, 'model_id': f'0x{model_id:08X}',
                   'blob_sha256': hashlib.sha256(blob).hexdigest()}
    open(os.path.join(out_dir, 'wnm3_k1.bin'), 'wb').write(blob)
    rep['h32_self_test'] = H.self_test()
    print(json.dumps(rep, indent=1), flush=True)
    # ------------------------------------------------------------------ records
    man = json.load(open(os.path.join(SAMPLE, 'manifest.json')))
    var = ('G', 'H18', 'H1C')
    acc = {v: {'wpay': [], 'wlen': [], 'spay': [], 'slen': []} for v in var}
    raw_all, sco_all, scc_all, recs = [], [], [], []
    tstats = []
    est_bits = 0.0
    for r, ei in enumerate(PICK):
        e = man['experts'][ei]
        fpath = os.path.join(SAMPLE, e['file'])
        sha_ok = hashlib.sha256(open(fpath, 'rb').read()).hexdigest() == e['sha256']
        assert sha_ok
        t = W.load_expert(fpath, e['tensors'])
        layer, expert = int(e['layer']), int(e['expert'])
        rec = {'file': e['file'], 'layer': layer, 'expert': expert, 'sha256_ok': sha_ok, 'matrices': []}
        dbins = []
        per = {v: {'w': [], 'wl': [], 's': [], 'sl': []} for v in var}
        for m, wn in enumerate(['w1', 'w2', 'w3']):
            wraw, wshape, _ = t[f'{wn}.weight']; sraw, _, _ = t[f'{wn}.scale']
            R = wshape[0]
            wb = wraw.reshape(R, -1)
            sc = sraw.reshape(-1)
            sym = np.empty((R, wb.shape[1] * 2), np.uint8)
            sym[:, 0::2] = wb & 15; sym[:, 1::2] = wb >> 4
            tsym, tsc, even, st = ceil_forward(sym.ravel(), sc)
            assert st['max5_blocks_gate'] == 0, 'gate fails: matrix would be coded plain (not implemented here)'
            isym, isc = ceil_inverse(tsym, tsc)
            st['cpu_roundtrip_ok'] = bool(np.array_equal(isym, sym.ravel()) and np.array_equal(isc, sc))
            assert st['cpu_roundtrip_ok']
            twb = (tsym.reshape(R, -1)[:, 0::2] | (tsym.reshape(R, -1)[:, 1::2] << 4)).astype(np.uint8)
            raw_all.append(wb.ravel().copy()); sco_all.append(sc.copy()); scc_all.append(tsc.copy())
            # contexts on coded data, anchor 120
            blk = tsym.reshape(-1, 32).astype(np.int64)
            mag = blk & 7
            cm = np.maximum.accumulate(mag, axis=1)
            pm = np.empty_like(mag); pm[:, 0] = 8; pm[:, 1:] = mag[:, :-1]
            mx = np.empty_like(mag); mx[:, 0] = 8; mx[:, 1:] = cm[:, :-1]
            d = tsc.astype(np.int64) - ANCHOR
            srel = np.clip(d, -3, 3) + 3
            dbins.append((np.clip(d, -DBC, DBC) + DBC).reshape(-1, 64))
            rows = srel[:, None] * 37 + W.tri(pm, mx)
            del pm, mx, cm, mag
            nb = blk.shape[0]; S = nb // 64
            F = f259[m][rows, blk].reshape(S, 2048).astype(np.int32)
            ST = cum259[m][rows, blk].reshape(S, 2048).astype(np.int32)
            est_bits += float(-np.log2(F.astype(np.float64) / P).sum())
            # hashes (coded bytes, ident per spec 5.1); weight stream g = 2160 + m*5760 + j, scale stream g = m*720 + j
            ww = twb.ravel().view('<u4').reshape(S, 256)
            sw = tsc.view('<u4').reshape(-1, 128)
            SS = sw.shape[0]
            gw = 3 * 720 + m * S + np.arange(S)
            gs = m * SS + np.arange(SS)
            x0 = {'G': (np.full(S, L, np.int64), np.full(SS, L, np.int64))}
            for v in ('H18', 'H1C'):
                fl = FLAGS[v]
                i3 = (fl | (m << 8) | (SET_ID << 16) | (ANCHOR << 24))
                iw = np.stack([np.full(S, model_id), np.full(S, (layer << 16) | expert), gw | (1 << 31), np.full(S, i3)], 1).astype(np.uint32)
                isw = np.stack([np.full(SS, model_id), np.full(SS, (layer << 16) | expert), gs | (1 << 31), np.full(SS, i3)], 1).astype(np.uint32)
                hw = H.h32_words_vec(np.concatenate([iw, ww], 1), H.SEED) & np.uint32(H.MASK23)
                hs = H.h32_words_vec(np.concatenate([isw, sw], 1), H.SEED) & np.uint32(H.MASK23)
                if r == 0 and m == 0:   # scalar H32 on a few streams
                    for j in (0, 1, S - 1):
                        hb = H.ident_bytes(model_id, layer, expert, int(gw[j]), 1, fl, m, SET_ID, ANCHOR) + twb.ravel()[1024 * j:1024 * j + 1024].tobytes()
                        assert (H.h32(hb, H.SEED) & H.MASK23) == int(hw[j])
                    for j in (0, SS - 1):
                        hb = H.ident_bytes(model_id, layer, expert, int(gs[j]), 1, fl, m, SET_ID, ANCHOR) + tsc[512 * j:512 * j + 512].tobytes()
                        assert (H.h32(hb, H.SEED) & H.MASK23) == int(hs[j])
                x0[v] = (L + hw.astype(np.int64), L + hs.astype(np.int64))
            s1 = tsc.astype(np.int64).reshape(-1, SPS)
            sF = scf[m][s1]; sST = scum[m][s1]
            for v in var:
                xw, xs = x0[v]
                pay, ln = W.rans_encode_vec(F, ST, np.zeros(2048, np.int64), 1, x0=xw)
                assert ln.min() >= 4 and ln.max() <= 4104
                spay, sln = W.rans_encode_vec(sF, sST, np.zeros(SPS, np.int64), 1, x0=xs)
                if r == 0 and m == 0:   # scalar reference encoder on 4 weight + 2 scale streams
                    offs = np.concatenate([[0], np.cumsum(ln)]); so = np.concatenate([[0], np.cumsum(sln)])
                    for j in (0, 1, S - 2, S - 1):
                        ref = W.encode_ref(blk[64 * j:64 * j + 64].ravel().tolist(), rows[64 * j:64 * j + 64].ravel().tolist(),
                                           f259[m], cum259[m], x0=int(xw[j]))
                        assert ref == pay[offs[j]:offs[j + 1]].tobytes(), ('scalar mismatch', v, j)
                    for j in (0, SS - 1):
                        ref = W.encode_ref(s1[j].tolist(), [0] * SPS, scf[m:m + 1], scum[m:m + 1], x0=int(xs[j]))
                        assert ref == spay[so[j]:so[j + 1]].tobytes(), ('scalar scale mismatch', v, j)
                    rep[f'scalar_reference_check_{v}'] = '4 weight + 2 scale streams byte-identical (record 0, w1)'
                per[v]['w'].append(pay); per[v]['wl'].append(ln); per[v]['s'].append(spay); per[v]['sl'].append(sln)
            del F, ST, rows, blk
            st['even_pct'] = 100.0 * st['even'] / st['blocks']
            rec['matrices'].append(st)
        rec['dbins'] = dbins
        for v in var:
            acc[v]['wpay'].append(np.concatenate(per[v]['w'])); acc[v]['wlen'].append(np.concatenate(per[v]['wl']))
            acc[v]['spay'].append(np.concatenate(per[v]['s'])); acc[v]['slen'].append(np.concatenate(per[v]['sl']))
        recs.append(rec)
        print(e['file'], 'elapsed', round(time.time() - t0, 1), flush=True)
    # ------------------------------------------------------------------ LENC sections (per record)
    lenc = {}
    for v, lv in (('G', 'G14'), ('H1C', 'H1C')):
        secs, info = [], []
        for r in range(len(PICK)):
            wl = acc[v]['wlen'][r].reshape(3, -1); sl = acc[v]['slen'][r].reshape(3, -1)
            sbase = [int(np.median(sl[m])) for m in range(3)]
            Q = [etab[m][recs[r]['dbins'][m]].sum(1) for m in range(3)]
            alpha = [fit_alpha(Q[m], wl[m]) for m in range(3)]
            spred = [np.full(sl.shape[1], sbase[m], np.int64) for m in range(3)]
            wpred = [pred_from_q(Q[m], alpha[m]) for m in range(3)]
            sp, SP, swid = pack_part([sl[m] for m in range(3)], spred)
            wp, WP, wwid = pack_part([wl[m] for m in range(3)], wpred)
            sec = struct.pack('<3H3H', *sbase, *alpha) + sp + wp
            assert len(sec) % 4 == 0
            # independent CPU read-back
            got_s, bs, us = unpack_part_ref(sp, [23] * 3, [720] * 3, spred)
            got_w, bw, uw = unpack_part_ref(wp, [180] * 3, [5760] * 3, wpred)
            assert us == len(sp) and uw == len(wp)
            assert all((got_s[m] == sl[m]).all() for m in range(3)) and all((got_w[m] == wl[m]).all() for m in range(3))
            assert SP == acc[v]['spay'][r].size and WP == acc[v]['wpay'][r].size
            secs.append(np.frombuffer(sec, np.uint8))
            wres = np.concatenate([wl[m] - wpred[m] for m in range(3)]); sres = np.concatenate([sl[m] - sbase[m] for m in range(3)])
            info.append({'LS': len(sec), 'scale_part': len(sp), 'weight_part': len(wp), 'sbase': sbase, 'alpha': alpha,
                         'max_width_s': max(swid), 'max_width_w': max(wwid), 'sum_width_s': sum(swid), 'sum_width_w': sum(wwid),
                         'resid_std_w': float(wres.std()), 'resid_std_s': float(sres.std()),
                         'resid_absmax_w': int(np.abs(wres).max()), 'resid_absmax_s': int(np.abs(sres).max()),
                         'SP': SP, 'WP': WP})
        lenc[lv] = {'secs': secs, 'info': info}
    # ------------------------------------------------------------------ save
    np.savez(os.path.join(out_dir, 'common_v1.npz'), raw=np.concatenate(raw_all), scales=np.concatenate(sco_all),
             scales_coded=np.concatenate(scc_all), mtype=np.tile(np.arange(3, dtype=np.uint8), len(PICK)),
             anchor=np.full(3 * len(PICK), ANCHOR, np.uint8), spm=np.int64(5760), sspm=np.int64(720),
             layer=np.array([rc['layer'] for rc in recs], np.int64), expert=np.array([rc['expert'] for rc in recs], np.int64),
             model_id=np.int64(model_id), tab259=cum259[..., 1:].astype(np.uint16), scum=scum.astype(np.uint16), slut=slut,
             etab=etab.astype(np.uint32), names=np.array([rc['file'] for rc in recs]))
    for v in var:
        np.savez(os.path.join(out_dir, f'str_{v}.npz'), wpay=np.concatenate(acc[v]['wpay']), wlen=np.concatenate(acc[v]['wlen']).astype(np.uint16),
                 spay=np.concatenate(acc[v]['spay']), slen=np.concatenate(acc[v]['slen']).astype(np.uint16),
                 wp_rec=np.array([a.size for a in acc[v]['wpay']], np.int64), sp_rec=np.array([a.size for a in acc[v]['spay']], np.int64))
    for lv in lenc:
        secs = lenc[lv]['secs']
        np.savez(os.path.join(out_dir, f'lenc_{lv}.npz'), sec=np.concatenate(secs), sec_bytes=np.array([s.size for s in secs], np.int64))
    # ------------------------------------------------------------------ sizes
    nrec = len(PICK)
    sizes = {}
    for v in var:
        wp = np.array([a.size for a in acc[v]['wpay']]); sp = np.array([a.size for a in acc[v]['spay']])
        sizes[v] = {'weight_payload_mean': float(wp.mean()), 'scale_payload_mean': float(sp.mean()),
                    'payload_mean': float((wp + sp).mean()), 'u16_length_table': 2 * (17280 + 2160)}
    def rec_total(v, ls):
        wp = np.array([a.size for a in acc[v]['wpay']]); sp = np.array([a.size for a in acc[v]['spay']])
        used = 64 + np.asarray(ls) + wp + sp
        return {'used_mean': float(used.mean()), 'total_mean_aligned4096': float((-(-used // 4096) * 4096).mean()),
                'of_raw_used': float(used.mean() / 18800640)}
    u16 = [38880] * nrec
    prof = {'CEIL 0x10 (u16 lengths, x0=L)': rec_total('G', u16),
            'CEIL|HASH 0x18 (u16)': rec_total('H18', u16),
            'CEIL|LENC 0x14': rec_total('G', [i['LS'] for i in lenc['G14']['info']]),
            'CEIL|LENC|HASH 0x1C (v1)': rec_total('H1C', [i['LS'] for i in lenc['H1C']['info']])}
    rep['sizes_streams'] = sizes
    rep['record_sizes_spec_layout_64B_header'] = prof
    rep['hash_cost_bytes_per_record'] = {
        'H18_minus_G_payload': sizes['H18']['payload_mean'] - sizes['G']['payload_mean'],
        'H1C_minus_G_payload': sizes['H1C']['payload_mean'] - sizes['G']['payload_mean'],
        'per_stream_H1C': (sizes['H1C']['payload_mean'] - sizes['G']['payload_mean']) / 19440}
    rep['lenc'] = {lv: lenc[lv]['info'] for lv in lenc}
    rep['lenc_LS_mean'] = {lv: float(np.mean([i['LS'] for i in lenc[lv]['info']])) for lv in lenc}
    rep['est_weight_bits_per_w_G (model code length, no flush)'] = est_bits / (nrec * 3 * 5760 * 2048)
    for rc in recs:
        rc.pop('dbins')
    rep['records'] = recs
    rep['total_s'] = time.time() - t0
    json.dump(rep, open(os.path.join(out_dir, 'encode_report_v1.json'), 'w'), indent=1)
    print(json.dumps({k: rep[k] for k in ('record_sizes_spec_layout_64B_header', 'hash_cost_bytes_per_record', 'lenc_LS_mean')}, indent=1))


if __name__ == '__main__':
    main()
