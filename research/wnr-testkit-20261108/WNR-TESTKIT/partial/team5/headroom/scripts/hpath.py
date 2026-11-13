"""headroom / exact held-out evaluation of given context sets on the full stride-16 feature files.

Usage: python -I hpath.py SETS_JSON OUT_JSON [KEEP_EVERY]
SETS_JSON: {"name": [variants...], ...}
Train even-indexed experts -> test odd-indexed (primary), plus the reverse fold, plus the team-4 40-expert test subset
(4l+1 for even l, 4l+3 for odd l), per-expert costs (for a bootstrap CI of the gain vs the first set) and rows used.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hctx import VARIANTS, quantize, P, PCT_RAW_PER_BPW, field_values, CH
from hadapt import split_from_words, dense_ctx

SAMPLE = os.environ.get('WNR_SAMPLE', 'v41-sample')  # kit: dir with the 160-expert sample (manifest.json + L*_E*.bin)
FEAT = os.environ.get('WNR_HEADROOM_FEAT', os.path.join('feat', 's16'))  # kit: regenerate with hfeat.py (2.9 GB)


def main():
    sets = json.load(open(sys.argv[1]))
    out_json = sys.argv[2]
    keep = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    man = json.load(open(os.path.join(SAMPLE, 'manifest.json')))
    ex = man['experts']
    t0 = time.time()
    # per set, matrix, expert: (uniq dense ids, counts)
    per = {k: [[None] * len(ex) for _ in range(3)] for k in sets}
    for i, e in enumerate(ex):
        z = np.load(os.path.join(FEAT, e['file'].replace('.bin', '.npz')))
        for m in range(3):
            W, B = z[f'W{m}'], z[f'B{m}']
            if keep > 1:
                nbr = 160 if m != 1 else 72
                nrows = B.size // nbr
                kp = np.arange(nrows) % keep == 0
                B = B.reshape(nrows, nbr)[kp].ravel(); W = W.reshape(nrows, nbr * 32)[kp].ravel()
            d = split_from_words(W, B)
            for k, names in sets.items():
                ctx = dense_ctx(d, names)
                u, inv = np.unique(ctx, return_inverse=True)
                per[k][m][i] = (u, np.bincount(inv * 16 + d['sym'], minlength=u.size * 16).reshape(-1, 16))
        if i % 20 == 0:
            print(i, e['file'], f'{time.time() - t0:.0f}s', flush=True)
    t4 = set()
    for i, e in enumerate(ex):
        L = e['layer']
        j = i - 4 * L
        if (L % 2 == 0 and j == 1) or (L % 2 == 1 and j == 3):
            t4.add(i)
    out = {'keep_every': keep, 'sets': sets, 'team4_subset': sorted(ex[i]['file'] for i in t4), 'results': {}}
    for k in sets:
        r = {'rows_train_even': [], 'per_expert_bits': {}, 'per_expert_n': {}}
        tot_ho = 0.0; tot_rev = 0.0; n_ho = 0; n_rev = 0
        for m in range(3):
            allu = np.unique(np.concatenate([per[k][m][i][0] for i in range(len(ex))]))
            C = [np.zeros((allu.size, 16), np.int64) for _ in range(2)]
            pos = {}
            for i in range(len(ex)):
                u, c = per[k][m][i]
                p = np.searchsorted(allu, u); pos[i] = p
                C[i % 2][p] += c
            costs = []
            for sp in range(2):
                ct = C[sp]
                used = ct.sum(1) > 0
                lc = np.full((allu.size, 16), 4.0)
                lc[used] = -np.log2(quantize(ct[used]) / P)
                costs.append(lc)
                if sp == 0:
                    r['rows_train_even'].append(int(used.sum()))
            for i in range(len(ex)):
                u, c = per[k][m][i]
                b = float((c * costs[1 - (i % 2)][pos[i]]).sum())
                n = int(c.sum())
                r['per_expert_bits'][ex[i]['file']] = r['per_expert_bits'].get(ex[i]['file'], 0.0) + b
                r['per_expert_n'][ex[i]['file']] = r['per_expert_n'].get(ex[i]['file'], 0) + n
                if i % 2:
                    tot_ho += b; n_ho += n
                else:
                    tot_rev += b; n_rev += n
        r['heldout_even_to_odd_bpw'] = tot_ho / n_ho
        r['heldout_odd_to_even_bpw'] = tot_rev / n_rev
        b4 = sum(r['per_expert_bits'][ex[i]['file']] for i in t4); n4 = sum(r['per_expert_n'][ex[i]['file']] for i in t4)
        r['team4_subset_bpw'] = b4 / n4
        out['results'][k] = r
    first = list(sets)[0]
    rng = np.random.default_rng(0)
    odd = [ex[i]['file'] for i in range(len(ex)) if i % 2]
    for k in sets:
        r = out['results'][k]
        r['delta_vs_first_bpw'] = r['heldout_even_to_odd_bpw'] - out['results'][first]['heldout_even_to_odd_bpw']
        r['delta_vs_first_pct_raw'] = r['delta_vs_first_bpw'] * PCT_RAW_PER_BPW
        r['delta_rev_vs_first_bpw'] = r['heldout_odd_to_even_bpw'] - out['results'][first]['heldout_odd_to_even_bpw']
        r['delta_team4_vs_first_bpw'] = r['team4_subset_bpw'] - out['results'][first]['team4_subset_bpw']
        dpe = np.array([(r['per_expert_bits'][f] - out['results'][first]['per_expert_bits'][f]) / r['per_expert_n'][f] for f in odd])
        bs = [dpe[rng.integers(0, dpe.size, dpe.size)].mean() for _ in range(2000)]
        r['delta_ci95_bpw'] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        r['delta_per_expert_min_max_bpw'] = [float(dpe.min()), float(dpe.max())]
        r['experts_improved'] = int((dpe < 0).sum())
        print(f"{k:24s} ho {r['heldout_even_to_odd_bpw']:.6f} d {r['delta_vs_first_bpw']:+.6f} bpw "
              f"({r['delta_vs_first_pct_raw']:+.4f}% raw) CI {r['delta_ci95_bpw']} rev d {r['delta_rev_vs_first_bpw']:+.6f} "
              f"t4 d {r['delta_team4_vs_first_bpw']:+.6f} rows {r['rows_train_even']} improved {r['experts_improved']}/80")
    out['sec'] = time.time() - t0
    json.dump(out, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
