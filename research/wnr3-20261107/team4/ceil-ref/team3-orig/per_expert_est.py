"""Per-expert estimated weight bit/w (code length under the held-out 'even' models, from own counts) for all 160 experts:
are the experts that gain most from K models (B) whole layers or single experts?
Usage: python -I per_expert_est.py SAMPLE_DIR COUNTS_DIR MODELS_DIR OUT_JSON"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wnr_common import load_manifest, P, NACTX

sample_dir, counts_dir, models_dir, out_json = sys.argv[1:5]
man = load_manifest(sample_dir)
Z = np.load(os.path.join(models_dir, 'even.npz'))
Lb = -np.log2(Z['a0_K0_c0_wf'].reshape(3, 567, 16) / P)
La = -np.log2(Z['a1_K0_c0_wf'].reshape(3, NACTX, 16) / P)
Ls = -np.log2(Z['a1_K0_c0_sf'].reshape(3, 7, 2) / P)
LK = -np.log2(Z['a1_K4_c0_wf'].reshape(4, NACTX, 16) / P)
LKs = -np.log2(Z['a1_K4_c0_sf'].reshape(4, 7, 2) / P)
Wm = 2304 * 5120
rows = []
for i, e in enumerate(man['experts']):
    z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
    r = {'idx': i, 'file': e['file'], 'layer': e['layer'], 'train': i % 2 == 0, 'per_matrix': []}
    for m in range(3):
        b = float((z['base'][m] * Lb[m]).sum()) / Wm
        a = float((z['a'][m] * La[m]).sum() + (z['side'][m] * Ls[m]).sum()) / Wm
        k = [float((z['a'][m] * LK[j]).sum() + (z['side'][m] * LKs[j]).sum()) / Wm for j in range(4)]
        r['per_matrix'].append({'base': b, 'a': a, 'aK4_best': min(k), 'k': int(np.argmin(k)), 'neg0_frac': float(z['neg0'][m] / Wm),
                                'med_scale': int(z['med'][m])})
    r['base'] = np.mean([p['base'] for p in r['per_matrix']])
    r['gain_aK4_vs_base_pct'] = 100 * (np.mean([p['aK4_best'] for p in r['per_matrix']]) - r['base']) / r['base']
    rows.append(r)
rows.sort(key=lambda r: r['gain_aK4_vs_base_pct'])
json.dump(rows, open(out_json, 'w'), indent=1)
for r in rows[:16]:
    print(r['file'], 'train' if r['train'] else 'test ', f"base {r['base']:.4f} bit/w  aK4 gain {r['gain_aK4_vs_base_pct']:+.3f}%",
          [(p['k'], round(p['base'], 3), p['med_scale']) for p in r['per_matrix']])
g = np.array([r['gain_aK4_vs_base_pct'] for r in rows])
print('all 160: mean gain %.3f%% median %.3f%%; experts with gain < -1%%: %d' % (g.mean(), np.median(g), (g < -1).sum()))
lay = {}
for r in rows:
    lay.setdefault(r['layer'], []).append(round(r['gain_aK4_vs_base_pct'], 2))
print({L: sorted(v) for L, v in sorted(lay.items()) if min(v) < -0.5})
