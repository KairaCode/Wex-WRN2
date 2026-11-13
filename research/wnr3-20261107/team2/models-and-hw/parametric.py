"""Parametric latent models for E2M1 codes given the UE8M0 block scale (OCP MX quantiser), on V4.1 data.

Generative model per 32-block with scale X = 2^(s-127):  u_i = w_i / X, |u_i| iid from a family f(.; a), a = 2^lam,
conditioned on the OCP rule  max|u| in [4, 8)  (scale = floor(log2 amax) - 2).  Code magnitude = RNE(|u|) on the
E2M1 grid {0,.5,1,1.5,2,3,4,6} with saturation (|u| in (5,8) -> 6); sign = fair coin (measured).
Exact block probability (order-independent, so a sequential coder reaches it):
   P(codes | lam) = 2^-32 * prod p(m_i) * (1 - [n7 == 0] (q6/p6)^n6) / (1 - Q^32)
   p(m) = P(|u| in bin m | |u| < 8), q6 = P(3.5 <= |u| < 4 | |u|<8), Q = P(|u| < 4 | |u| < 8).
Models:
   'single'  : one lam per (matrix type, srel)            -> 2 numbers per row of 7
   'mixture' : lam hidden per block, prior pi(lam | type, srel) on a grid (EM), shape shared per (type, srel)
Compared with the static 567-context table on the SAME test experts (table trained on the training split).

Usage: python -I parametric.py SAMPLE_DIR HIST_DIR COUNTS_DIR OUT_JSON NTRAIN NTEST
"""
import json, os, sys, time
import numpy as np
from scipy import special, stats

T = np.array([0, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, 8.0])
LAM = np.arange(-3.5, 5.0001, 0.125)
LN2 = np.log(2)


def fam_cdf(kind, shape, x, a):
    """(CDF, survival) of |u| at x (array) for scale a (array [G]) -> two [G, len(x)] arrays"""
    z = x[None, :] / a[:, None]
    if kind == 'gg':
        return special.gammainc(1.0 / shape, z ** shape), special.gammaincc(1.0 / shape, z ** shape)
    if kind == 't':
        return 2 * stats.t.cdf(z, shape) - 1, 2 * stats.t.sf(z, shape)
    raise ValueError(kind)


def interval(F, S, lo, hi):
    """P(lo <= |u| < hi) with the less cancellation-prone formula"""
    return np.where(F[:, lo] > 0.5, S[:, lo] - S[:, hi], F[:, hi] - F[:, lo])


def fam_tables(kind, shape, lam=LAM):
    a = np.exp2(lam)
    F, S = fam_cdf(kind, shape, np.append(T, 4.0), a)  # [G, 10] (index 9 = 4.0)
    F8 = F[:, 8]
    p = np.stack([interval(F, S, k, k + 1) for k in range(8)], 1) / F8[:, None]  # [G, 8]
    q6 = interval(F, S, 6, 9) / F8
    oneminusQ = interval(F, S, 9, 8) / F8  # P(4 <= |u| < 8 | |u| < 8)
    logp = np.log(np.maximum(p, 1e-300))
    logr = np.log(np.maximum(q6, 1e-300)) - logp[:, 6]
    lognorm = np.log(-np.expm1(32 * np.log1p(-np.minimum(oneminusQ, 1 - 1e-16))))
    return logp, logr, lognorm


def decode_keys(keys):
    srel = (keys >> 48).astype(np.int64)
    N = np.stack([(keys >> (6 * m)) & 63 for m in range(8)], 1).astype(np.float64)
    return srel, N


def loglik(N, tabs):
    logp, logr, lognorm = tabs
    LL = N @ logp.T  # [U, G]
    n6 = N[:, 6:7]; no7 = (N[:, 7:8] == 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(no7, np.log(-np.expm1(np.minimum(n6 * logr[None, :], -1e-300))), 0.0)
    return LL + corr - lognorm[None, :]


def lse(x, axis=1):
    m = x.max(axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)


def em(LL, w, iters=400, tol=1e-10):
    """EM for the mixing weights on a fixed lambda grid (multiplicative update on the likelihood matrix)"""
    G = LL.shape[1]
    mx = LL.max(1, keepdims=True)
    Lk = np.exp(LL - mx)
    pi = np.full(G, 1.0 / G)
    W = w.sum()
    prev = -np.inf
    for it in range(iters):
        z = Lk @ pi
        ll = float((w * np.log(z)).sum())
        pi = pi * (Lk.T @ (w / z)) / W
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    return np.log(np.maximum(pi, 1e-300)), ll


def pool(files, hist_dir, m):
    ks, cs = [], []
    for f in files:
        z = np.load(os.path.join(hist_dir, f.replace('.bin', '.npz')), allow_pickle=False)
        ks.append(z[f'key_m{m}']); cs.append(z[f'cnt_m{m}'])
    k = np.concatenate(ks); c = np.concatenate(cs).astype(np.float64)
    uk, inv = np.unique(k, return_inverse=True)
    return uk, np.bincount(inv, weights=c)


def table_cost(files_tr, files_te, counts_dir):
    """567-context table (ideal float probabilities from training counts) on test experts: bits per (type, srel)"""
    A = sum(np.load(os.path.join(counts_dir, f.replace('.bin', '.npz')))['w_base'].astype(np.float64) for f in files_tr)
    Bc = sum(np.load(os.path.join(counts_dir, f.replace('.bin', '.npz')))['w_base'].astype(np.float64) for f in files_te)
    p = (A + 1e-3) / (A + 1e-3).sum(-1, keepdims=True)  # tiny smoothing for unseen
    bits = -(Bc * np.log2(p)).reshape(3, 7, 81, 16).sum((2, 3))  # [3,7]
    nsym = Bc.reshape(3, 7, 81, 16).sum((2, 3))
    pin = np.where(Bc > 0, Bc / np.maximum(Bc.sum(-1, keepdims=True), 1), 1)
    ins = -(Bc * np.log2(pin)).reshape(3, 7, 81, 16).sum((2, 3))
    return bits, nsym, ins


def main():
    sample_dir, hist_dir, counts_dir, out_json = sys.argv[1:5]
    ntr, nte = int(sys.argv[5]), int(sys.argv[6])
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    ex = man['experts']
    have = lambda f: os.path.exists(os.path.join(hist_dir, f.replace('.bin', '.npz')))
    s0 = [e['file'] for i, e in enumerate(ex) if i % 2 == 0 and have(e['file'])]
    s1 = [e['file'] for i, e in enumerate(ex) if i % 2 == 1 and have(e['file'])]
    pick = lambda L, k: [L[int(j)] for j in np.linspace(0, len(L) - 1, k).round()]
    ftr, fte = pick(s0, ntr), pick(s1, nte)
    out = {'train': ftr, 'test': fte, 'lam_grid': [float(LAM[0]), float(LAM[-1]), 0.125]}
    tbits_small, _, _ = table_cost(ftr, fte, counts_dir)
    out['table_heldout_bpw_trained_on_same_ntrain'] = float(tbits_small.sum() / tbits_small.size and tbits_small.sum())
    all0 = [e['file'] for i, e in enumerate(ex) if i % 2 == 0]
    tbits, tnsym, tins = table_cost(all0, fte, counts_dir)
    out['table_heldout_bpw_trained_on_same_ntrain'] = float(tbits_small.sum() / tnsym.sum())
    out['table_heldout_bpw'] = float(tbits.sum() / tnsym.sum())
    out['table_insample_on_test_bpw'] = float(tins.sum() / tnsym.sum())
    out['table_per_type_srel_bpw'] = (tbits / np.maximum(tnsym, 1)).round(5).tolist()
    out['test_weight_share_per_srel'] = (tnsym.sum(0) / tnsym.sum()).round(5).tolist()
    print('table held-out (ideal, no flush)', out['table_heldout_bpw'], 'in-sample on test', out['table_insample_on_test_bpw'], flush=True)
    fam_arg = os.environ.get('FAMS', 'gg0.7,gg1.0,gg1.5,gg2.0,t3,t6')
    families = [(f[:2] if f.startswith('gg') else 't', float(f[2:] if f.startswith('gg') else f[1:])) for f in fam_arg.split(',')]
    tabs = {fa: fam_tables(*fa) for fa in families}
    res = {}
    t0 = time.time()
    for m in range(3):
        ktr, wtr = pool(ftr, hist_dir, m)
        kte, wte = pool(fte, hist_dir, m)
        str_, Ntr = decode_keys(ktr)
        ste, Nte = decode_keys(kte)
        print(f'm{m}: unique train {len(ktr)}, test {len(kte)}  ({time.time() - t0:.0f}s)', flush=True)
        for s in range(7):
            itr = str_ == s; ite = ste == s
            if wte[ite].sum() == 0:
                continue
            nsym = 32 * wte[ite].sum()
            row = {}
            for fa in families:
                LLtr = loglik(Ntr[itr], tabs[fa])
                LLte = loglik(Nte[ite], tabs[fa])
                # single lam (best grid point on train)
                g = int(np.argmax((wtr[itr][:, None] * LLtr).sum(0)))
                single = float(-(wte[ite] * LLte[:, g]).sum() / LN2 + nsym) / nsym
                lpi, _ = em(LLtr, wtr[itr])
                mix = float(-(wte[ite] * lse(LLte + lpi[None, :])).sum() / LN2 + nsym) / nsym
                row[f'{fa[0]}{fa[1]}'] = {'single_bpw': single, 'single_lam': float(LAM[g]), 'mixture_bpw': mix}
            best = min(row, key=lambda k: row[k]['mixture_bpw'])
            bests = min(row, key=lambda k: row[k]['single_bpw'])
            res[f'm{m}_srel{s - 3:+d}'] = {'families': row, 'best_mixture': best, 'best_single': bests,
                                          'table_bpw': float(tbits[m, s] / tnsym[m, s]), 'weights': float(nsym)}
            print(f'm{m} srel{s - 3:+d} table {tbits[m, s] / tnsym[m, s]:.4f} | best single {bests} {row[bests]["single_bpw"]:.4f} '
                  f'| best mixture {best} {row[best]["mixture_bpw"]:.4f} | all-mix ' + ' '.join(f'{k}:{v["mixture_bpw"]:.4f}' for k, v in row.items()) +
                  f' | single lam {row[bests]["single_lam"]} ({time.time() - t0:.0f}s)', flush=True)
    out['per_group'] = res
    # totals
    W = sum(v['weights'] for v in res.values())
    for kind in ['single_bpw', 'mixture_bpw']:
        for fa in families:
            k = f'{fa[0]}{fa[1]}'
            out.setdefault('totals', {})[f'{kind}:{k}'] = sum(v['families'][k][kind] * v['weights'] for v in res.values()) / W
        out['totals'][f'{kind}:best_per_group'] = sum(v['families'][v['best_mixture' if kind == 'mixture_bpw' else 'best_single']][kind] * v['weights'] for v in res.values()) / W
    out['totals']['table'] = sum(v['table_bpw'] * v['weights'] for v in res.values()) / W
    print(json.dumps(out['totals'], indent=1), flush=True)
    json.dump(out, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
