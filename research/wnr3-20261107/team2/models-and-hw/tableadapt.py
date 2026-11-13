"""Static 567-context table + tiny per-matrix / per-record parameters (held-out).

Usage: python -I tableadapt.py SAMPLE_DIR COUNTS_DIR OUT_JSON
Reads the pass-1 counts (w_base [3, 567, 16] per expert) written by mxcount.py.
Protocol (same as mxcombo_an.py): split = expert index % 2; tables are trained on one split and the
cost is measured on the other, both ways, summed. Per-matrix parameters are chosen by the ENCODER on
the test matrix itself (that is legal: the parameter is transmitted, 1 byte per matrix).
Costs: 'ideal' = float probabilities; 'q14' = 14-bit CDFs quantized like wex_nr2_quantize.
All bit/w figures include the 4-B flush per 2048-weight stream (0.015625 bit/w).
"""
import json, os, sys, time
import numpy as np

P = 16384
W_MAT = 2304 * 5120
FLUSH_BPW = 32 / 2048


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
    return freq.reshape(c.shape)


def quantize_prob(p, scale=1 << 30):
    """float probabilities -> 14-bit freq via the same rule (counts = round(p * 2^30))."""
    return quantize(np.rint(np.asarray(p) * scale).astype(np.int64))


def load(sample_dir, counts_dir):
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    U, lay, split = [], [], []
    for i, e in enumerate(man['experts']):
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')), allow_pickle=False)
        U.append(z['w_base'].astype(np.int64))
        lay.append(e['layer']); split.append(i % 2)
    return np.stack(U), np.array(lay), np.array(split)


def norm(c, alpha=0.0):
    c = np.asarray(c, np.float64) + alpha
    t = c.sum(-1, keepdims=True)
    t[t == 0] = 1
    p = c / t
    p[(c.sum(-1) == 0)] = 1.0 / c.shape[-1]
    return p


def table_from_counts(c):
    """the production model's probabilities (14-bit quantized), as float"""
    return quantize(c) / P


def bits(counts, p):
    """counts [..., 567, 16], p broadcastable -> total bits"""
    with np.errstate(divide='ignore'):
        lp = np.log2(np.maximum(p, 1e-300))
    return float((counts * -lp).sum())


def tilt(p, v, theta):
    """p [567,16] -> p * 2^(theta*v) normalised; theta scalar or [T] -> [T,567,16]"""
    th = np.atleast_1d(theta)[:, None, None]
    q = p[None] * np.exp2(th * v[None, None, :])
    return q / q.sum(-1, keepdims=True)


def best_param(cnt, fam):
    """cnt [567,16]; fam [T,567,16] probabilities -> (bits per member [T])"""
    lp = -np.log2(np.maximum(fam, 1e-300))
    return np.einsum('cs,tcs->t', cnt.astype(np.float64), lp)


SRC = {}


def main():
    sample_dir, counts_dir, out_json = sys.argv[1:4]
    U, lay, split = load(sample_dir, counts_dir)  # U [n, 3, 567, 16]
    n = U.shape[0]
    Wtot = n * 3 * W_MAT
    bpw = lambda b: b / Wtot + FLUSH_BPW
    out = {'experts': n}
    mag = np.arange(16) & 7
    vals = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])[mag]
    V = {
        'mag': mag.astype(np.float64),
        'value': vals,
        'log2v': np.where(vals > 0, np.log2(np.maximum(vals, 1e-9)), -2.0),
        'zero': (mag == 0).astype(np.float64),
    }
    srel_of_ctx = np.arange(567) // 81

    # ---- base held-out, ideal vs q14 ----
    res = {}
    tot_ideal = tot_q = 0.0
    tables = {}
    for tr in (0, 1):
        A = U[split == tr].sum(0)  # [3,567,16]
        T = U[split != tr]
        pq = table_from_counts(A)
        pi = norm(A)
        tables[tr] = pq
        tot_q += bits(T, pq[None])
        tot_ideal += bits(T, pi[None])
    base_q = tot_q
    res['base_heldout_q14'] = bpw(tot_q)
    res['base_heldout_ideal'] = bpw(tot_ideal)
    # in-sample ideal for reference
    Aall = U.sum(0)
    res['base_insample_ideal'] = bpw(bits(Aall, norm(Aall)))
    out['base'] = res

    def report(name, total_bits, extra_side_bits=0.0, note=''):
        b = total_bits + extra_side_bits
        out.setdefault('variants', {})[name] = {
            'bit_per_w': bpw(b), 'delta_vs_base_q14_pct': 100 * (b - base_q) / (base_q + FLUSH_BPW * Wtot),
            'side_bits_total': extra_side_bits, 'note': note}
        print(f'{name:48s} {bpw(b):.5f}  {100 * (b - base_q) / (base_q + FLUSH_BPW * Wtot):+.4f}%  {note}', flush=True)

    print('base held-out q14', res['base_heldout_q14'], 'ideal', res['base_heldout_ideal'], flush=True)

    # ---- per-matrix one-parameter tilts ----
    grid = np.linspace(-0.6, 0.6, 241)
    theta_store = {}
    for vname, v in V.items():
        tb = 0.0
        tb_rec = 0.0
        th_all = []
        for tr in (0, 1):
            pq = tables[tr]
            fam = [tilt(pq[m], v, grid) for m in range(3)]  # 3 x [T,567,16]
            idx = np.where(split != tr)[0]
            for i in idx:
                per = np.stack([best_param(U[i, m], fam[m]) for m in range(3)])  # [3, T]
                tb += per.min(1).sum()
                tb_rec += per.sum(0).min()
                th_all.append((i, grid[per.argmin(1)].tolist()))
        theta_store[vname] = th_all
        report(f'tilt_{vname} per matrix (1 B)', tb, 8 * 3 * n, 'ideal float probs, grid 241')
        report(f'tilt_{vname} per record (1 B)', tb_rec, 8 * n, 'one theta shared by w1,w2,w3')

    # 2-parameter tilt: mag + zero
    g2a = np.linspace(-0.4, 0.4, 41)
    g2b = np.linspace(-0.6, 0.6, 41)
    tb = 0.0
    for tr in (0, 1):
        pq = tables[tr]
        for m in range(3):
            q = pq[m][None, None] * np.exp2(g2a[:, None, None, None] * V['mag'] + g2b[None, :, None, None] * V['zero'])
            q = q / q.sum(-1, keepdims=True)
            lp = -np.log2(q)
            for i in np.where(split != tr)[0]:
                tb += np.einsum('cs,abcs->ab', U[i, m].astype(np.float64), lp).min()
    report('tilt mag+zero per matrix (2 B)', tb, 16 * 3 * n, '41x41 grid')

    # per-srel tilt (7 thetas per matrix, 7 B)
    tb = 0.0
    for tr in (0, 1):
        pq = tables[tr]
        for m in range(3):
            fam = tilt(pq[m], V['mag'], grid)  # [T,567,16]
            lp = -np.log2(fam)
            for i in np.where(split != tr)[0]:
                c = U[i, m].astype(np.float64)
                per_ctx = np.einsum('cs,tcs->tc', c, lp)  # [T,567]
                for s in range(7):
                    tb += per_ctx[:, srel_of_ctx == s].sum(1).min()
    report('tilt_mag per (matrix, srel) (7 B)', tb, 8 * 7 * 3 * n)

    # power / temperature
    tgrid = np.linspace(0.8, 1.2, 81)
    tb = 0.0
    for tr in (0, 1):
        pq = tables[tr]
        for m in range(3):
            fam = pq[m][None] ** tgrid[:, None, None]
            fam = fam / fam.sum(-1, keepdims=True)
            for i in np.where(split != tr)[0]:
                tb += best_param(U[i, m], fam).min()
    report('temperature per matrix (1 B)', tb, 8 * 3 * n)

    # srel shift: mix with the neighbouring srel row (alpha in [-1,1])
    agrid = np.linspace(-1, 1, 81)
    tb = 0.0
    for tr in (0, 1):
        pq = tables[tr]
        for m in range(3):
            t3 = pq[m].reshape(7, 81, 16)
            up = np.concatenate([t3[1:], t3[-1:]], 0).reshape(567, 16)
            dn = np.concatenate([t3[:1], t3[:-1]], 0).reshape(567, 16)
            fam = np.stack([(1 - abs(a)) * pq[m] + abs(a) * (up if a > 0 else dn) for a in agrid])
            for i in np.where(split != tr)[0]:
                tb += best_param(U[i, m], fam).min()
    report('srel half-step shift per matrix (1 B)', tb, 8 * 3 * n, 'mix with srel+-1 row')

    # ---- K clusters (reproduce) + soft mixture of the K=2 tables ----
    for K in (2, 4, 8):
        tbq = tbi = tbmix = tbclt = 0.0
        for tr in (0, 1):
            idx_tr = np.where(split == tr)[0]
            Ut = np.stack([U[i, m] for i in idx_tr for m in range(3)])
            lt = np.array([lay[i] for i in idx_tr for m in range(3)])
            assign = lt * K // 40
            for _ in range(15):
                Lm = np.stack([-np.log2(quantize(Ut[assign == k].sum(0) if (assign == k).any() else np.zeros((567, 16), np.int64)) / P) for k in range(K)])
                new = np.einsum('ucs,kcs->uk', Ut.astype(np.float64), Lm).argmin(1)
                if (new == assign).all():
                    break
                assign = new
            Pk = np.exp2(-Lm)  # [K,567,16]
            Tt = np.stack([U[i, m] for i in np.where(split != tr)[0] for m in range(3)]).astype(np.float64)
            tbq += np.einsum('ucs,kcs->uk', Tt, Lm).min(1).sum()
            if K == 2:
                ag = np.linspace(0, 1, 33)
                fam = np.stack([(1 - a) * Pk[0] + a * Pk[1] for a in ag])
                lpf = -np.log2(fam)
                tbmix += np.einsum('ucs,tcs->ut', Tt, lpf).min(1).sum()
                # K=2 hard choice, then a magnitude tilt on top
                fam2 = np.concatenate([tilt(Pk[k], V['mag'], np.linspace(-0.3, 0.3, 61)) for k in range(2)])
                tbclt += np.einsum('ucs,tcs->ut', Tt, -np.log2(fam2)).min(1).sum()
        report(f'K={K} clusters (shared over types, q14)', tbq, 0, 'reproduces mxcombo_an clusters_heldout')
        if K == 2:
            report('K=2 soft mixture alpha (33 levels)', tbmix, 6 * 3 * n)
            report('K=2 hard + mag tilt (61 levels)', tbclt, 7 * 3 * n)

    # ---- a family of PREBUILT tilted tables (what a decoder would store): 16 levels, q14 ----
    for nlev in (4, 8, 16):
        lev = np.linspace(-0.3, 0.3, nlev)
        tb = 0.0
        for tr in (0, 1):
            pq = tables[tr]
            for m in range(3):
                fam = quantize_prob(tilt(pq[m], V['mag'], lev)) / P
                for i in np.where(split != tr)[0]:
                    tb += best_param(U[i, m], fam).min()
        report(f'{nlev} prebuilt mag-tilted q14 tables, select per matrix', tb, np.log2(nlev) * 3 * n, f'levels {lev[0]:.2f}..{lev[-1]:.2f}')

    # ---- what drives theta? correlate per-matrix best theta (tilt_mag) with simple statistics ----
    th = {i: t for i, t in theta_store['mag']}
    rows = []
    for i in range(n):
        for m in range(3):
            c = U[i, m].sum(0)  # [16]
            mg = np.bincount(np.arange(16) & 7, weights=c, minlength=8)
            frac = mg / mg.sum()
            srel_hist = U[i, m].reshape(7, 81, 16).sum((1, 2))
            rows.append({'i': i, 'layer': int(lay[i]), 'm': m, 'theta': th[i][m], 'mean_mag': float((frac * np.arange(8)).sum()),
                         'zero_frac': float(frac[0]), 'srel_mean': float((srel_hist * np.arange(7)).sum() / srel_hist.sum() - 3),
                         'srel_spread': float(np.sqrt((srel_hist * (np.arange(7) - 3) ** 2).sum() / srel_hist.sum()))})
    import itertools
    thv = np.array([r['theta'] for r in rows])
    corr = {}
    for k in ['layer', 'm', 'mean_mag', 'zero_frac', 'srel_mean', 'srel_spread']:
        x = np.array([r[k] for r in rows], np.float64)
        corr[k] = float(np.corrcoef(x, thv)[0, 1])
    out['theta_mag_stats'] = {'mean': float(thv.mean()), 'std': float(thv.std()), 'min': float(thv.min()), 'max': float(thv.max()),
                              'corr_with': corr,
                              'by_matrix_type_mean': [float(thv[[r['m'] == m for r in rows]].mean()) for m in range(3)],
                              'by_layer_mean': [float(thv[[r['layer'] == L for r in rows]].mean()) for L in range(40)]}
    print(json.dumps(out['theta_mag_stats'], indent=1)[:3000])
    json.dump(out, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
