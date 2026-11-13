"""WNR3 draft support measurements (team3 / wnr3-spec), on the real V4.1 sample.

Usage: python -I -B lenres.py SAMPLE_DIR COUNTS_DIR COMBO_DIR OUT_JSON [eval_experts]

1. Structural compaction check: every row USED by the c_bm2h counts (base x bm7 x half, 2268 full contexts)
   must lie in the structurally reachable set (pm <= mx, bm7=0 => mx <= 6, position 0 only in half 0);
   reports the size of that set (expected 910) and the used rows per matrix.
2. Length-table residuals for the WNR3 weight streams, held out: model + length predictor trained on
   split-0 experts, residuals measured on split-1 experts. Two models:
     base : WNR2 567 contexts (as mxstreams.py, but held out)
     w3a  : c_bm2h contexts + the bm7 symbol coded in-stream with an srel context (feature a)
   Stream bytes are estimated as round(bits/8) + 4 (same convention as mxstreams.py).
   Predictor (integer, as in the draft): pred = 4 + ((sum_b E_q[srel_b] + 1024) >> 11), E_q in 1/256 bit.
   Scale streams: constant predictor 4 + ((n * Es_q + 1024) >> 11).
3. Coding cost of the residuals: in-sample entropy, held-out static order-0 (64-symbol zigzag + escape,
   14-bit, like a rANS table), and grouped bit-packing (G streams share one width byte).
Reads only; writes OUT_JSON.
"""
import json, os, sys, time
import numpy as np

P = 16384
WCH, SCH = 2048, 512


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


def load_expert(path, tensors):
    raw = np.fromfile(path, dtype=np.uint8)
    out, off = {}, 0
    for t in tensors:
        n = t['bytes']
        out[t['name']] = (raw[off:off + n], t['shape'])
        off += n
    assert off == raw.size
    return out


def structural_rows():
    """full context id ((base*2+bm7)*2+half), base=(srel*9+pm)*9+mx -> reachable?"""
    ok = np.zeros(2268, bool)
    for srel in range(7):
        for pm in range(9):
            for mx in range(9):
                for bm in range(2):
                    for half in range(2):
                        cid = ((((srel * 9 + pm) * 9 + mx) * 2 + bm) * 2 + half)
                        if pm == 8 or mx == 8:
                            r = (pm == 8 and mx == 8 and half == 0)
                        else:
                            r = pm <= mx and (bm == 1 or mx <= 6)
                        ok[cid] = r
    return ok


def zz(r):
    r = r.astype(np.int64)
    return np.where(r >= 0, 2 * r, -2 * r - 1)


def ent_bits(v):
    _, c = np.unique(v, return_counts=True)
    p = c / c.sum()
    return float(-(c * np.log2(p)).sum())


def bitpack_bits(z, G):
    """z: zigzag residuals of one matrix's streams in stream order. 8-bit width per group + G*w bits."""
    n = z.size
    ng = -(-n // G)
    zp = np.concatenate([z, np.zeros(ng * G - n, np.int64)]).reshape(ng, G)
    mx = zp.max(1)
    w = np.where(mx == 0, 0, np.floor(np.log2(np.maximum(mx, 1))).astype(np.int64) + 1)
    return float((8 + G * w).sum()), np.bincount(w, minlength=18)


def main():
    sample_dir, counts_dir, combo_dir, out_json = sys.argv[1:5]
    neval = int(sys.argv[5]) if len(sys.argv) > 5 else 24
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    exs = man['experts']
    t0 = time.time()
    # ---- training on split 0
    cw = np.zeros((3, 567, 16), np.int64); cs = np.zeros((3, 256), np.int64); bms = np.zeros((3, 7, 8), np.int64)
    cz = np.zeros((3, 2268, 16), np.int64); cb = np.zeros((3, 7, 2), np.int64)
    used_all = np.zeros((3, 2268), np.int64)
    for i, e in enumerate(exs):
        zc = np.load(os.path.join(combo_dir, e['file'].replace('.bin', '.npz')))
        used_all += zc['c_bm2h'].sum(-1)
        if i % 2:
            continue
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        cw += z['w_base']; cs += z['s_o0']; bms += z['bm_side']
        cz += zc['c_bm2h']; cb += zc['bm2_side']
    st = structural_rows()
    used = used_all > 0
    out = {'structural_rows': int(st.sum()),
           'used_rows_per_matrix_all160': used.sum(1).tolist(),
           'used_rows_outside_structural': int((used & ~st[None, :]).sum()),
           'structural_rows_never_used_any_matrix': int((st & ~used.any(0)).sum())}
    Lw = -np.log2(quantize(cw) / P)
    Lz = -np.log2(quantize(cz) / P)
    Lb = -np.log2(quantize(cb) / P)
    Ls = -np.log2(quantize(cs) / P)
    blocks_srel = bms.sum(2)  # [3, 7]
    # expected bits per block given srel, 1/256-bit integer units
    Eb = np.zeros((3, 7)); E3 = np.zeros((3, 7))
    for m in range(3):
        cc = (cw[m] * Lw[m]).sum(1)
        c3 = (cz[m] * Lz[m]).sum(1)
        for s in range(7):
            nb = max(1, blocks_srel[m, s])
            Eb[m, s] = cc[s * 81:(s + 1) * 81].sum() / nb
            E3[m, s] = (c3[s * 324:(s + 1) * 324].sum() + (cb[m, s] * Lb[m, s]).sum()) / nb
    Ebq = np.round(Eb * 256).astype(np.int64); E3q = np.round(E3 * 256).astype(np.int64)
    Esq = np.round((cs * Ls).sum(1) / cs.sum(1) * 256).astype(np.int64)
    out['E_q_base'] = Ebq.tolist(); out['E_q_w3a'] = E3q.tolist(); out['Es_q'] = Esq.tolist()
    # ---- evaluation on split 1
    ev = [e for i, e in enumerate(exs) if i % 2 == 1]
    step = max(1, len(ev) // neval)
    ev = ev[::step][:neval]
    res = {k: [[[] for _ in range(3)] for _ in range(len(ev))] for k in ('base', 'w3a', 'scale')}
    act = {k: [[] for _ in range(3)] for k in ('base', 'w3a')}
    for ei, e in enumerate(ev):
        t = load_expert(os.path.join(sample_dir, e['file']), e['tensors'])
        for m, w in enumerate(['w1', 'w2', 'w3']):
            wraw, wshape = t[f'{w}.weight']; sraw, _ = t[f'{w}.scale']
            R = wshape[0]
            wb = wraw.reshape(R, -1); sc = sraw.reshape(R, -1).astype(np.int16)
            C = wb.shape[1] * 2; nb = C // 32
            sym = np.empty((R, C), np.uint8); sym[:, 0::2] = wb & 15; sym[:, 1::2] = wb >> 4
            blk = sym.reshape(R, nb, 32); mag = blk & 7
            pm = np.empty_like(mag); pm[..., 0] = 8; pm[..., 1:] = mag[..., :-1]
            cm = np.maximum.accumulate(mag, axis=2)
            mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
            bm = (cm[..., -1] == 7).astype(np.int32)
            med = int(np.sort(sc.ravel())[(sc.size - 1) // 2])
            srel = (np.clip(sc - med, -3, 3) + 3).astype(np.int32)
            base = (srel[..., None] * 9 + pm.astype(np.int32)) * 9 + mx.astype(np.int32)
            half = (np.arange(32) >= 16).astype(np.int32)[None, None, :]
            c3 = (base * 2 + bm[..., None]) * 2 + half
            b32 = blk.astype(np.int32)
            bits_b = Lw[m][base, b32].sum(2).ravel()                      # per block
            bits_3 = (Lz[m][c3, b32].sum(2) + Lb[m][srel, bm]).ravel()    # per block incl. bm7 symbol
            nbs = WCH // 32
            ns = -(-bits_b.size // nbs)
            padn = ns * nbs - bits_b.size
            sr = srel.ravel()
            for key, bits, Eq in (('base', bits_b, Ebq), ('w3a', bits_3, E3q)):
                sb = np.concatenate([bits, np.zeros(padn)]).reshape(ns, nbs).sum(1)
                a = np.round(sb / 8).astype(np.int64) + 4
                pq = np.concatenate([Eq[m][sr], np.zeros(padn, np.int64)]).reshape(ns, nbs).sum(1)
                pred = 4 + ((pq + 1024) >> 11)
                res[key][ei][m] = (a - pred)
                act[key][m].append(a)
            sbits = Ls[m][sc.astype(np.int32).ravel()]
            nss = -(-sbits.size // SCH)
            nsym = np.full(nss, SCH, np.int64); nsym[-1] = sbits.size - SCH * (nss - 1)
            sbits = np.concatenate([sbits, np.zeros(nss * SCH - sbits.size)]).reshape(nss, SCH).sum(1)
            sa = np.round(sbits / 8).astype(np.int64) + 4
            spred = 4 + ((nsym * Esq[m] + 1024) >> 11)
            res['scale'][ei][m] = sa - spred
        print(f'{ei + 1}/{len(ev)} {e["file"]} {time.time() - t0:.0f}s', flush=True)
    out['eval_experts'] = [e['file'] for e in ev]
    # ---- coding cost
    nstreams = {'base': 5760, 'w3a': 5760, 'scale': 720}  # per matrix on V4.1 at 2048:512
    half_a = list(range(0, len(ev), 2)); half_b = list(range(1, len(ev), 2))
    for key in ('base', 'w3a', 'scale'):
        o = {}
        tot_ent = tot_o0 = 0.0
        bp = {G: 0.0 for G in (8, 16, 32, 64)}
        for m in range(3):
            allr = np.concatenate([res[key][i][m] for i in range(len(ev))])
            o[f'm{m}'] = {'mean': float(allr.mean()), 'std': float(allr.std()), 'min': int(allr.min()), 'max': int(allr.max()),
                          'p_abs_le_7': float((np.abs(allr) <= 7).mean()), 'p_abs_gt_31': float((np.abs(allr) > 31).mean())}
            if key != 'scale':
                o[f'm{m}']['actual_mean_bytes'] = float(np.concatenate(act[key][m]).mean())
            tot_ent += ent_bits(allr) / allr.size * nstreams[key]
            # held-out order-0 table over zigzag(r - bias): 63 direct symbols + escape (+16 raw bits)
            ho = 0.0
            for tr, te in ((half_a, half_b), (half_b, half_a)):
                rtr = np.concatenate([res[key][i][m] for i in tr]); rte = np.concatenate([res[key][i][m] for i in te])
                bias = int(np.round(np.median(rtr)))
                ztr = zz(rtr - bias); zte = zz(rte - bias)
                c = np.bincount(np.minimum(ztr, 63), minlength=64)
                q = quantize(c[None, :])[0]
                ho += float((-np.log2(q[np.minimum(zte, 63)] / P)).sum() + 16 * (zte >= 63).sum())
            nr = sum(res[key][i][m].size for i in range(len(ev)))
            tot_o0 += ho / nr * nstreams[key]
            for G in bp:
                bsum = 0.0
                for i in range(len(ev)):
                    r = res[key][i][m]
                    bias = 0
                    b, _ = bitpack_bits(zz(r - bias), G)
                    bsum += b
                bp[G] += bsum / nr * nstreams[key]
        o['record_bytes_entropy_insample'] = tot_ent / 8
        o['record_bytes_order0_heldout_no_flush'] = tot_o0 / 8
        o['record_bytes_bitpack'] = {G: v / 8 for G, v in bp.items()}
        o['record_bytes_u16'] = 2 * 3 * nstreams[key]
        out[key] = o
    json.dump(out, open(out_json, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
