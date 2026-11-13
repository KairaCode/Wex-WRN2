"""Parametric latent models through the quantiser, on canonical OCP blocks AND on the reversible ceil form.

Canonical OCP form : |u| < 8, max|u| >= 4  (block max code 4.0 or 6.0)
Ceil form          : |u| <= 6, max|u| > 3  (scale = 2^ceil(log2(amax/6)); max code 3.0, 4.0 or 6.0)
Ceil-form keys are derived from the canonical block histograms (blockhist.py) exactly: even-only blocks with max 6.0
(no 0.5/1.5) are halved and their srel recomputed with the ceil-form median (ceilform.py meta). Clamping at srel -3
is approximated (affects ~5e-7 of blocks).
For each (form, matrix type, srel) and family: single lambda (best on train) and a mixture over a lambda grid (EM).
Compared on the same test experts with the 567 table of the same form (trained on all 80 split-0 experts).

Usage: python -I param2.py SAMPLE_DIR HIST_DIR CEIL_DIR OUT_JSON NTRAIN NTEST FAMS
"""
import json, os, sys, time
import numpy as np
from scipy import special, stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parametric import decode_keys, em

LAM = np.arange(-3.5, 5.0001, 0.125)
LN2 = np.log(2)
FORMS = {'canonical': (8.0, 4.0, 6), 'ceil': (6.0, 3.0, 5)}  # (upper U, lower L, bin containing L)
EDGES = [0, .25, .75, 1.25, 1.75, 2.5, 3.5, 5]


def cdf_sf(kind, shape, x, a):
    z = x[None, :] / a[:, None]
    if kind == 'gg':
        return special.gammainc(1.0 / shape, z ** shape), special.gammaincc(1.0 / shape, z ** shape)
    return 2 * stats.t.cdf(z, shape) - 1, 2 * stats.t.sf(z, shape)


def tables(kind, shape, form):
    U, L, bl = FORMS[form]
    x = np.array(EDGES + [U, L])  # indices 0..7 bin lower edges, 8 = U, 9 = L
    a = np.exp2(LAM)
    F, S = cdf_sf(kind, shape, x, a)
    def iv(lo, hi):
        return np.where(F[:, lo] > 0.5, S[:, lo] - S[:, hi], F[:, hi] - F[:, lo])
    FU = F[:, 8]
    p = np.stack([iv(k, k + 1) if k < 7 else iv(7, 8) for k in range(8)], 1) / FU[:, None]
    if form == 'ceil':
        p[:, 7] = iv(7, 8) / FU  # (5, 6]
    qL = iv(bl, 9) / FU  # P(lower edge of bin bl <= |u| < L)
    upper = iv(9, 8) / FU  # P(L <= |u| < U)
    logp = np.log(np.maximum(p, 1e-300))
    logr = np.log(np.maximum(qL, 1e-300)) - logp[:, bl]
    with np.errstate(divide='ignore'):
        lognorm = np.where(upper > 1e-280, np.log(-np.expm1(32 * np.log1p(-np.minimum(upper, 1 - 1e-16)))), np.inf)
    return logp, logr, lognorm, bl  # lognorm = +inf marks grid points where the block constraint has ~0 mass


def loglik(N, tab):
    logp, logr, lognorm, bl = tab
    LL = N @ logp.T
    nb = N[:, bl:bl + 1]
    none_above = (N[:, bl + 1:].sum(1, keepdims=True) == 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(none_above, np.log(-np.expm1(np.minimum(nb * logr[None, :], -1e-300))), 0.0)
    return LL + corr - lognorm[None, :]


def ceil_keys(keys, medc, medl):
    s, N = decode_keys(keys)
    ev = (N[:, 1] == 0) & (N[:, 3] == 0) & (N[:, 7] > 0)
    N2 = N.copy()
    N2[ev] = np.stack([N[ev, 0], N[ev, 2], N[ev, 4], N[ev, 5], N[ev, 6], N[ev, 7], 0 * N[ev, 0], 0 * N[ev, 0]], 1)
    raw = s - 3 + (ev.astype(np.int64)) + (medc - medl)  # s - medl (unclamped approx)
    s2 = np.clip(raw, -3, 3) + 3
    return s2, N2


def pool_form(files, hist_dir, ceil_dir, m, form):
    S_, N_, W_ = [], [], []
    for f in files:
        z = np.load(os.path.join(hist_dir, f.replace('.bin', '.npz')), allow_pickle=False)
        k = z[f'key_m{m}']; w = z[f'cnt_m{m}'].astype(np.float64)
        if form == 'canonical':
            s, N = decode_keys(k)
        else:
            meta = np.load(os.path.join(ceil_dir, f.replace('.bin', '.npz')), allow_pickle=False)['meta']
            s, N = ceil_keys(k, int(meta[m, 2]), int(meta[m, 3]))
        S_.append(s); N_.append(N); W_.append(w)
    return np.concatenate(S_), np.concatenate(N_), np.concatenate(W_)


def table_bits(ceil_dir, train, test, form):
    key = 'w_can' if form == 'canonical' else 'w_ceil'
    A = sum(np.load(os.path.join(ceil_dir, f.replace('.bin', '.npz')))[key].astype(np.float64) for f in train)
    B = sum(np.load(os.path.join(ceil_dir, f.replace('.bin', '.npz')))[key].astype(np.float64) for f in test)
    p = (A + 1e-3) / (A + 1e-3).sum(-1, keepdims=True)
    return -(B * np.log2(p)).reshape(3, 7, 81, 16).sum((2, 3)), B.reshape(3, 7, 81, 16).sum((2, 3))


def main():
    sample_dir, hist_dir, ceil_dir, out_json = sys.argv[1:5]
    ntr, nte = int(sys.argv[5]), int(sys.argv[6])
    fams = [(f[:2], float(f[2:])) if f.startswith('gg') else ('t', float(f[1:])) for f in sys.argv[7].split(',')]
    ex = json.load(open(os.path.join(sample_dir, 'manifest.json')))['experts']
    have = lambda f: os.path.exists(os.path.join(ceil_dir, f.replace('.bin', '.npz')))
    s0 = [e['file'] for i, e in enumerate(ex) if i % 2 == 0 and have(e['file'])]
    s1 = [e['file'] for i, e in enumerate(ex) if i % 2 == 1 and have(e['file'])]
    pick = lambda L, k: [L[int(j)] for j in np.linspace(0, len(L) - 1, min(k, len(L))).round()]
    ftr, fte = pick(s0, ntr), pick(s1, nte)
    out = {'train': ftr, 'test': fte, 'train_table_on': f'{len(s0)} split-0 experts'}
    t0 = time.time()
    for form in FORMS:
        tb, tn = table_bits(ceil_dir, s0, fte, form)
        tabs = {fa: tables(fa[0], fa[1], form) for fa in fams}
        tot = {'table': 0.0}
        for fa in fams:
            tot[f'single:{fa[0]}{fa[1]}'] = 0.0; tot[f'mix:{fa[0]}{fa[1]}'] = 0.0
        tot['single:best'] = tot['mix:best'] = 0.0
        nsym_tot = 0.0
        groups = {}
        for m in range(3):
            str_, Ntr, wtr = pool_form(ftr, hist_dir, ceil_dir, m, form)
            ste, Nte, wte = pool_form(fte, hist_dir, ceil_dir, m, form)
            for s in range(7):
                it = str_ == s; ie = ste == s
                if wte[ie].sum() == 0 or wtr[it].sum() == 0:
                    if wte[ie].sum() > 0:
                        tot['table'] += tb[m, s]; nsym_tot += 32 * wte[ie].sum()
                        for k in tot:
                            if k != 'table':
                                tot[k] += tb[m, s]  # no training data: fall back to the table cost
                    continue
                nsym = 32 * wte[ie].sum()
                nsym_tot += nsym
                tot['table'] += tb[m, s]
                row = {'table': tb[m, s] / nsym}
                bs = bm = np.inf
                for fa in fams:
                    LLtr = loglik(Ntr[it], tabs[fa]); LLte = loglik(Nte[ie], tabs[fa])
                    g = int(np.argmax((wtr[it][:, None] * LLtr).sum(0)))
                    single = float(-(wte[ie] * LLte[:, g]).sum() / LN2 + nsym)
                    lpi, _ = em(LLtr, wtr[it])
                    X = LLte + lpi[None, :]
                    mx = X.max(1, keepdims=True)
                    z = (np.log(np.exp(X - mx).sum(1, keepdims=True)) + mx)[:, 0]
                    mix = float(-(wte[ie] * z).sum() / LN2 + nsym)
                    tot[f'single:{fa[0]}{fa[1]}'] += single; tot[f'mix:{fa[0]}{fa[1]}'] += mix
                    row[f'single:{fa[0]}{fa[1]}'] = single / nsym; row[f'mix:{fa[0]}{fa[1]}'] = mix / nsym
                    row[f'lam:{fa[0]}{fa[1]}'] = float(LAM[g])
                    bs = min(bs, single); bm = min(bm, mix)
                tot['single:best'] += bs; tot['mix:best'] += bm
                groups[f'm{m}_srel{s - 3:+d}'] = row
                print(f'{form} m{m} srel{s - 3:+d} ' + ' '.join(f'{k}={v:.4f}' for k, v in row.items() if not k.startswith('lam')) +
                      f' ({time.time() - t0:.0f}s)', flush=True)
        out[form] = {'totals_bpw_ideal_no_flush': {k: v / nsym_tot for k, v in tot.items()}, 'groups': groups}
        print(form, json.dumps(out[form]['totals_bpw_ideal_no_flush'], indent=1), flush=True)
    json.dump(out, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
