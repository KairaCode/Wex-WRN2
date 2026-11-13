"""Exchangeable latent-class model for E2M1 magnitudes in a 32-block (nonparametric shape), vs the 567 table.

Per (matrix type, srel): P(block) = 2^-32 * sum_k pi_k * prod_i th_k(m_i) * 1[n6+n7>=1] / (1 - (1-th_k6-th_k7)^32)
(valid normalised code; an exact sequential coder reaches -log2 P(block)). Fitted by (generalised) EM on block
magnitude histograms of the training experts; cost measured on test experts (held-out).

Usage: python -I lclass.py SAMPLE_DIR HIST_DIR COUNTS_DIR OUT_JSON NTRAIN NTEST K1,K2,...
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parametric import pool, decode_keys, table_cost

LN2 = np.log(2)


def block_ll(N, logth, lognorm):
    return N @ logth.T - lognorm[None, :]  # [U, K]


TOP = 5 if os.environ.get('FORM', 'canonical') == 'ceil' else 6  # block max is always >= this magnitude


def norm_terms(th):
    rest = np.clip(1 - th[:, TOP:].sum(1), 0, 1 - 1e-15)
    return np.log(-np.expm1(32 * np.log(np.maximum(rest, 1e-300))))  # log(1 - rest^32)


def fit(N, w, K, iters=300, seed=0):
    W = w.sum()
    mean = (N * np.arange(8)).sum(1) / 32
    order = np.argsort(mean, kind='stable')
    cw = np.cumsum(w[order]) / W
    grp = np.empty(len(N), np.int64)
    grp[order] = np.minimum((cw * K).astype(np.int64), K - 1)
    th = np.stack([(N[grp == k] * w[grp == k, None]).sum(0) + 1.0 for k in range(K)])
    th /= th.sum(1, keepdims=True)
    pi = np.bincount(grp, weights=w, minlength=K) / W
    best = (-np.inf, None)
    prev = -np.inf
    for it in range(iters):
        logth = np.log(th); ln = norm_terms(th)
        X = block_ll(N, logth, ln) + np.log(np.maximum(pi, 1e-300))[None, :]
        mx = X.max(1, keepdims=True)
        z = np.log(np.exp(X - mx).sum(1, keepdims=True)) + mx
        ll = float((w * z[:, 0]).sum())
        if ll > best[0]:
            best = (ll, (th.copy(), pi.copy()))
        if ll - prev < 1e-9 * abs(ll):
            break
        prev = ll
        r = np.exp(X - z) * w[:, None]
        pi = r.sum(0) / W
        th = r.T @ N + 1e-6
        th /= th.sum(1, keepdims=True)
    return best[1], best[0], it + 1


def cost(N, w, th, pi):
    X = block_ll(N, np.log(th), norm_terms(th)) + np.log(np.maximum(pi, 1e-300))[None, :]
    mx = X.max(1, keepdims=True)
    z = (np.log(np.exp(X - mx).sum(1, keepdims=True)) + mx)[:, 0]
    return float(-(w * z).sum() / LN2 + 32 * w.sum())  # bits incl. 32 sign bits


def main():
    sample_dir, hist_dir, counts_dir, out_json = sys.argv[1:5]
    ntr, nte = int(sys.argv[5]), int(sys.argv[6])
    Ks = [int(k) for k in sys.argv[7].split(',')]
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    ex = man['experts']
    have = lambda f: os.path.exists(os.path.join(hist_dir, f.replace('.bin', '.npz')))
    s0 = [e['file'] for i, e in enumerate(ex) if i % 2 == 0 and have(e['file'])]
    s1 = [e['file'] for i, e in enumerate(ex) if i % 2 == 1 and have(e['file'])]
    pick = lambda L, k: [L[int(j)] for j in np.linspace(0, len(L) - 1, min(k, len(L))).round()]
    ftr, fte = pick(s0, ntr), pick(s1, nte)
    all0 = [e['file'] for i, e in enumerate(ex) if i % 2 == 0]
    tbits, tnsym, _ = table_cost(all0, fte, counts_dir)
    tsmall, _, _ = table_cost(ftr, fte, counts_dir)
    out = {'train': ftr, 'test': fte, 'table_heldout_bpw (trained on 80 split-0 experts, ideal)': float(tbits.sum() / tnsym.sum()),
           'table_heldout_bpw (trained on the same NTRAIN experts)': float(tsmall.sum() / tnsym.sum())}
    print(out, flush=True)
    tot = {K: 0.0 for K in Ks}
    totsym = 0.0
    per = {}
    t0 = time.time()
    form = os.environ.get('FORM', 'canonical')
    out['form'] = form
    if form == 'ceil':
        from param2 import pool_form, table_bits
        ceil_dir = os.environ['CEIL_DIR']
        tbits, tnsym = table_bits(ceil_dir, all0, fte, 'ceil')
        out['table_heldout_bpw (ceil form, 80 split-0)'] = float(tbits.sum() / tnsym.sum())
    for m in range(3):
        if form == 'ceil':
            str_, Ntr, wtr = pool_form(ftr, hist_dir, ceil_dir, m, 'ceil')
            ste, Nte, wte = pool_form(fte, hist_dir, ceil_dir, m, 'ceil')
        else:
            ktr, wtr = pool(ftr, hist_dir, m)
            kte, wte = pool(fte, hist_dir, m)
            str_, Ntr = decode_keys(ktr)
            ste, Nte = decode_keys(kte)
        for s in range(7):
            itr = str_ == s; ite = ste == s
            if wte[ite].sum() == 0:
                continue
            nsym = 32 * wte[ite].sum()
            totsym += nsym
            row = {'table_bpw': float(tbits[m, s] / tnsym[m, s])}
            for K in Ks:
                (th, pi), ll, its = fit(Ntr[itr], wtr[itr], K)
                b = cost(Nte[ite], wte[ite], th, pi)
                tot[K] += b
                row[f'K{K}'] = b / nsym
            per[f'm{m}_srel{s - 3:+d}'] = row
            print(f'm{m} srel{s - 3:+d} ' + ' '.join(f'{k}:{v:.4f}' for k, v in row.items()) + f' ({time.time() - t0:.0f}s)', flush=True)
    out['per_group'] = per
    out['totals_bpw'] = {f'K{K}': tot[K] / totsym for K in Ks}
    out['totals_bpw']['table'] = float(tbits.sum() / tnsym.sum())
    print(json.dumps(out['totals_bpw'], indent=1), flush=True)
    json.dump(out, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
