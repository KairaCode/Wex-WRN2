"""Distribution, outliers and correlations of per-expert WNR2 bit/w (from units.json of countstats.py).

Usage: python -I analyze_units.py UNITS_JSON [RAWSTATS_JSON]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wnrlib import spearman, pearson

FL = 0.015625

d = json.load(open(sys.argv[1]))
U = d['units']
raw = None
if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    raw = {r['file']: r for r in json.load(open(sys.argv[2]))}
    for u in U:
        rm = raw[u['file']]['m'][u['m']]
        for k in ['s_var_row', 's_var_col', 's_var_res', 's_var_tot', 'kurt_code', 'kurt_deq', 'blk_ties', 'deq_std',
                  's_distinct', 'rows_allzero', 's_rows_const']:
            u['raw_' + k] = rm[k]
        u['raw_row_share'] = rm['s_var_row'] / rm['s_var_tot'] if rm['s_var_tot'] else 0
        u['raw_col_share'] = rm['s_var_col'] / rm['s_var_tot'] if rm['s_var_tot'] else 0

files = sorted({u['file'] for u in U}, key=lambda f: [u['i'] for u in U if u['file'] == f][0])
E = []
for f in files:
    us = sorted([u for u in U if u['file'] == f], key=lambda u: u['m'])
    e = {'file': f, 'layer': us[0]['layer'], 'expert': us[0]['expert'], 'orig_id': us[0]['orig_id']}
    for k in ['bpw_shared', 'bpw_shared_loo', 'bpw_layer_loo', 'bpw_own', 'bpw_own_ideal', 'excess_shared_vs_own', 'zneg', 'zpos',
              'zero', 'h0_own', 'mean_absval_code', 'frac_mag67', 'frac_mag_le1', 's_std', 's_iqr', 's_range99', 'blk_max7',
              'sbits_shared', 's_clamp_lo', 's_clamp_hi', 'neg_share_nonzero', 's_med', 's_skew'] + \
             ([k for k in us[0] if k.startswith('raw_')]):
        e[k] = float(np.mean([u[k] for u in us]))   # matrices have equal weight counts -> mean = per-record value
    for m in range(3):
        e[f'bpw_m{m}'] = us[m]['bpw_shared']
        e[f'smed_m{m}'] = us[m]['s_med']
    e['srel_hist'] = np.mean([u['srel_hist'] for u in us], 0).tolist()
    e['srel_p1'] = e['srel_hist'][4]       # share of weights in blocks with scale = median + 1
    e['srel_m1'] = e['srel_hist'][2]       # scale = median - 1
    e['srel_out'] = e['srel_hist'][0] + e['srel_hist'][1] + e['srel_hist'][5] + e['srel_hist'][6]   # |s - med| >= 2
    E.append(e)
for u in U:
    h = u['srel_hist']
    u['srel_p1'] = h[4]; u['srel_m1'] = h[2]; u['srel_out'] = h[0] + h[1] + h[5] + h[6]

out = {}
b = np.array([e['bpw_shared'] for e in E]) + FL
out['expert_bpw_shared_fl'] = {'min': b.min(), 'p10': np.percentile(b, 10), 'median': np.median(b), 'p90': np.percentile(b, 90),
                               'max': b.max(), 'mean': b.mean(), 'std': b.std()}
bo = np.array([e['bpw_own'] for e in E]) + FL
out['expert_bpw_own_fl'] = {'min': bo.min(), 'median': np.median(bo), 'max': bo.max(), 'mean': bo.mean()}
ex_ = np.array([e['excess_shared_vs_own'] for e in E])
out['expert_excess_shared_minus_own'] = {'min': ex_.min(), 'median': np.median(ex_), 'max': ex_.max(), 'mean': ex_.mean()}
for m in range(3):
    bm = np.array([u['bpw_shared'] for u in U if u['m'] == m]) + FL
    out[f'unit_m{m}_bpw_shared_fl'] = {'min': bm.min(), 'median': np.median(bm), 'max': bm.max(), 'std': bm.std()}

# variance decomposition: share of between-expert variance explained by layer means
lay = np.array([e['layer'] for e in E])
lm = np.array([b[lay == L].mean() for L in lay])
out['var_explained_by_layer_mean'] = 1 - ((b - lm) ** 2).sum() / ((b - b.mean()) ** 2).sum()

stats = ['layer', 'orig_id', 'zero', 'zneg', 'h0_own', 'mean_absval_code', 'frac_mag67', 'frac_mag_le1', 's_std', 's_iqr', 's_range99',
         'srel_p1', 'srel_m1', 'srel_out', 'blk_max7', 's_med', 's_skew', 'excess_shared_vs_own', 'bpw_own', 'neg_share_nonzero'] + \
        [k for k in E[0] if k.startswith('raw_')]
cor = {}
for s in stats:
    x = np.array([e[s] for e in E])
    cor[s] = {'spearman_vs_bpw_shared': round(spearman(x, b), 3), 'pearson': round(pearson(x, b), 3),
              'spearman_vs_excess': round(spearman(x, ex_), 3)}
out['expert_correlations'] = cor
# within matrix type, units
ucor = {}
for m in range(3):
    um = [u for u in U if u['m'] == m]
    bb = np.array([u['bpw_shared'] for u in um])
    ucor[f'm{m}'] = {s: round(spearman([u[s] for u in um], bb), 3) for s in
                     ['layer', 'zero', 'h0_own', 'srel_p1', 'srel_m1', 's_std', 'blk_max7', 'mean_absval_code', 'frac_mag67'] +
                     [k for k in um[0] if k.startswith('raw_')]}
out['unit_spearman_vs_bpw_shared'] = ucor

# OLS: bpw ~ a few cheap stats
def ols(cols, y):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, np.float64) for c in cols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return beta.tolist(), float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())

reg = {}
for name, ks in [('zero', ['zero']), ('h0_own', ['h0_own']), ('srel_p1', ['srel_p1']), ('blk_max7', ['blk_max7']),
                 ('zero+srel_p1', ['zero', 'srel_p1']), ('zero+srel_p1+blk_max7', ['zero', 'srel_p1', 'blk_max7']),
                 ('h0_own+srel_p1', ['h0_own', 'srel_p1']), ('mag_hist(8)', None)]:
    if ks is None:
        mh = np.array([np.mean([u['mag_hist'] for u in U if u['file'] == e['file']], 0) for e in E])
        beta, r2 = ols([mh[:, k] for k in range(7)], b)
    else:
        beta, r2 = ols([[e[k] for e in E] for k in ks], b)
    reg[name] = {'R2': round(r2, 3), 'beta': [round(x, 4) for x in beta]}
out['expert_ols_R2'] = reg

# top / bottom 10 experts
cols = ['file', 'layer', 'orig_id', 'bpw_shared', 'bpw_own', 'excess_shared_vs_own', 'bpw_m0', 'bpw_m1', 'bpw_m2', 'zero', 'zneg', 'h0_own',
        'srel_p1', 'srel_m1', 'srel_out', 's_std', 'blk_max7', 'mean_absval_code', 'frac_mag67', 'frac_mag_le1', 's_med'] + \
       [k for k in E[0] if k.startswith('raw_')]
order = np.argsort(b)
pick = lambda e: {k: (round(e[k] + FL, 4) if k.startswith('bpw') else (round(e[k], 5) if isinstance(e[k], float) else e[k])) for k in cols}
out['best10'] = [pick(E[j]) for j in order[:10]]
out['worst10'] = [pick(E[j]) for j in order[::-1][:10]]
# population medians for reference
out['population_median'] = {k: float(np.median([e[k] for e in E])) for k in cols if k not in ('file',)}
# per layer means
out['per_layer'] = [{'layer': int(L), 'bpw_shared_fl': round(float(b[lay == L].mean()), 4),
                     'min': round(float(b[lay == L].min()), 4), 'max': round(float(b[lay == L].max()), 4),
                     'zero': round(float(np.mean([e['zero'] for e in E if e['layer'] == L])), 4),
                     'srel_p1': round(float(np.mean([e['srel_p1'] for e in E if e['layer'] == L])), 4),
                     'h0_own': round(float(np.mean([e['h0_own'] for e in E if e['layer'] == L])), 4),
                     'excess': round(float(np.mean([e['excess_shared_vs_own'] for e in E if e['layer'] == L])), 4)}
                    for L in sorted(set(lay))]
out['experts'] = E
json.dump(out, open(os.path.join(os.path.dirname(sys.argv[1]), 'expert_analysis.json'), 'w'), indent=1, default=float)
pr = {k: v for k, v in out.items() if k != 'experts'}
print(json.dumps(pr, indent=1, default=float)[:30000])
