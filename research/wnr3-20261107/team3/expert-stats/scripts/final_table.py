"""Final per-expert table: WNR2 (median anchor) vs densest-pair anchor, explanatory statistics, raw-data tails.

Usage: python -I final_table.py SAMPLE_DIR COUNTS_DIR UNITS_JSON RAWSTATS_JSON KEEPSET_JSON OUT_JSON OUT_CSV
"""
import json, os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wnrlib import NW, quantize, codelen, cost_bits, load_all, spearman, pearson

FL = 0.015625
sample_dir, counts_dir, units_json, raw_json, keep_json, out_json, out_csv = sys.argv[1:8]
ex = load_all(sample_dir, counts_dir, keys=('w_base', 'w_srel15', 's_o0', 'meta', 'bm_side'))
n = len(ex)
U = {(u['i'], u['m']): u for u in json.load(open(units_json))['units']}
R = {r['file']: r for r in json.load(open(raw_json))} if os.path.exists(raw_json) else None
keep = json.load(open(keep_json, encoding='utf-8'))['keep']
units = [(i, m) for i in range(n) for m in range(3)]
nu = len(units)
mtype = np.array([m for i, m in units])
H = np.stack([ex[i]['s_o0'][m] for i, m in units])
MED = np.array([int(ex[i]['meta'][m, 2]) for i, m in units])
A = (H[:, :-1] + H[:, 1:]).argmax(1)
W15 = np.stack([ex[i]['w_srel15'][m] for i, m in units]).reshape(nu, 15, 81, 16)
C = np.zeros((nu, 7, 81, 16), np.int64)
for j in range(15):
    C[np.arange(nu), np.clip(j - 7 - (A - MED), -3, 3) + 3] += W15[:, j]
C = C.reshape(nu, 567, 16)
L = [codelen(C[mtype == m].sum(0)) for m in range(3)]
Ctot = [C[mtype == m].sum(0) for m in range(3)]
fix_in = np.array([(C[k] * L[mtype[k]]).sum() / NW for k in range(nu)])
fix_loo = np.array([cost_bits(C[k], quantize(Ctot[mtype[k]] - C[k])) / NW for k in range(nu)])
fix_own = np.array([cost_bits(C[k], quantize(C[k])) / NW for k in range(nu)])
p_up = H[np.arange(nu), A + 1] / H.sum(1)
flip = A != MED

rows = []
for i, e in enumerate(ex):
    ks = [i * 3 + m for m in range(3)]
    us = [U[(i, m)] for m in range(3)]
    r = {'file': e['file'], 'layer': e['layer'], 'expert': e['expert'], 'orig_id': keep[str(e['layer'])][e['expert']],
         'bpw_wnr2': np.mean([u['bpw_shared'] for u in us]) + FL,
         'bpw_wnr2_loo': np.mean([u['bpw_shared_loo'] for u in us]) + FL,
         'bpw_own': np.mean([u['bpw_own'] for u in us]) + FL,
         'bpw_fix': fix_in[ks].mean() + FL, 'bpw_fix_loo': fix_loo[ks].mean() + FL, 'bpw_fix_own': fix_own[ks].mean() + FL,
         'flipped': ''.join('123'[m] for m in range(3) if flip[i * 3 + m]) or '-',
         'p_up': [round(float(p_up[k]), 3) for k in ks],
         'bpw_m': [round(us[m]['bpw_shared'] + FL, 4) for m in range(3)],
         's_med': [int(MED[k]) for k in ks],
         's_std': np.mean([u['s_std'] for u in us]), 's_out': np.mean([u['srel_hist'][0] + u['srel_hist'][1] + u['srel_hist'][5] + u['srel_hist'][6] for u in us]),
         'zero': np.mean([u['zero'] for u in us]), 'zneg': np.mean([u['zneg'] for u in us]),
         'blk_max7': np.mean([u['blk_max7'] for u in us]), 'h0': np.mean([u['h0_own'] for u in us]),
         'frac_mag67': np.mean([u['frac_mag67'] for u in us])}
    if R:
        rm = R[e['file']]['m']
        r['kurt_deq'] = float(np.mean([x['kurt_deq'] for x in rm]))
        r['kurt_code'] = float(np.mean([x['kurt_code'] for x in rm]))
        r['s_row_share'] = float(np.mean([x['s_var_row'] / x['s_var_tot'] for x in rm]))
        r['s_col_share'] = float(np.mean([x['s_var_col'] / x['s_var_tot'] for x in rm]))
        r['deq_std'] = float(np.mean([x['deq_std'] for x in rm]))
    rows.append(r)

b = np.array([r['bpw_wnr2'] for r in rows]); bf = np.array([r['bpw_fix'] for r in rows]); bo = np.array([r['bpw_own'] for r in rows])
summ = lambda x: {'min': round(float(x.min()), 4), 'p10': round(float(np.percentile(x, 10)), 4), 'median': round(float(np.median(x)), 4),
                  'p90': round(float(np.percentile(x, 90)), 4), 'max': round(float(x.max()), 4), 'mean': round(float(x.mean()), 4),
                  'std': round(float(x.std()), 4)}
out = {'dist_expert_bpw': {'wnr2_shared_insample': summ(b), 'wnr2_shared_loo': summ(np.array([r['bpw_wnr2_loo'] for r in rows])),
                           'own_model_payload': summ(bo), 'pair_anchor_shared_insample': summ(bf),
                           'pair_anchor_shared_loo': summ(np.array([r['bpw_fix_loo'] for r in rows])),
                           'pair_anchor_own': summ(np.array([r['bpw_fix_own'] for r in rows]))}}
# unit-level: residual dependence on phase after the fix (unflipped and all)
fu = fix_in + FL
out['unit_fix_vs_p_up'] = {'spearman_all': spearman(p_up, fu), 'pearson_all': pearson(p_up, fu)}
X = np.column_stack([np.ones(nu), p_up])
beta, *_ = np.linalg.lstsq(X, fu, rcond=None)
res = fu - X @ beta
out['unit_fix_ols_p_up'] = {'beta': beta.tolist(), 'R2': float(1 - (res ** 2).sum() / ((fu - fu.mean()) ** 2).sum())}
for m in range(3):
    g = mtype == m
    Xg = np.column_stack([np.ones(g.sum()), p_up[g]]); bg, *_ = np.linalg.lstsq(Xg, fu[g], rcond=None)
    rg = fu[g] - Xg @ bg
    out[f'unit_fix_ols_p_up_m{m}'] = {'beta': bg.tolist(), 'R2': float(1 - (rg ** 2).sum() / ((fu[g] - fu[g].mean()) ** 2).sum())}
# expert-level correlations after the fix
stats = ['layer', 'orig_id', 'expert', 'zero', 'zneg', 's_std', 's_out', 'blk_max7', 'h0', 'frac_mag67'] + (['kurt_deq', 'kurt_code', 's_row_share', 's_col_share', 'deq_std'] if R else [])
pmean = np.array([np.mean(r['p_up']) for r in rows])
out['expert_corr'] = {s: {'rho_wnr2': round(spearman([r[s] for r in rows], b), 3), 'rho_fix': round(spearman([r[s] for r in rows], bf), 3)} for s in stats}
out['expert_corr']['p_up_mean'] = {'rho_wnr2': round(spearman(pmean, b), 3), 'rho_fix': round(spearman(pmean, bf), 3)}
out['expert_corr']['n_flipped'] = {'rho_wnr2': round(spearman([0 if r['flipped'] == '-' else len(r['flipped']) for r in rows], b), 3)}
o = np.argsort(b)
out['worst10_wnr2'] = [rows[j] for j in o[::-1][:10]]
out['best10_wnr2'] = [rows[j] for j in o[:10]]
of = np.argsort(bf)
out['worst10_after_fix'] = [rows[j] for j in of[::-1][:10]]
out['best10_after_fix'] = [rows[j] for j in of[:10]]
json.dump(out, open(out_json, 'w'), indent=1, default=float)
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    keys = list(rows[0].keys())
    w.writerow(keys)
    for r in rows:
        w.writerow([(round(r[k], 5) if isinstance(r[k], float) else r[k]) for k in keys])
pr = {k: v for k, v in out.items() if not k.startswith(('worst', 'best'))}
print(json.dumps(pr, indent=1, default=float))
for k in ['worst10_wnr2', 'best10_wnr2', 'worst10_after_fix', 'best10_after_fix']:
    print('---', k)
    for r in out[k]:
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()})
