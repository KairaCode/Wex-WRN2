"""Distribution of the per-expert ceil-form gain (estimates from per_expert.py).
Usage: python -I -B per_expert_an.py PER_EXPERT_SAMPLE_JSON PER_EXPERT_EXTRA_JSON OUT_JSON"""
import json, sys
import numpy as np


def main():
    a, b, outp = sys.argv[1:4]
    rows = json.load(open(a)) + json.load(open(b))
    for r in rows:  # per_expert.py divided by 552,960 instead of 1,105,920 blocks per record: fix the share here
        r['even_pct'] = r['even_pct'] / 2
    d = np.array([r['d_record_pct_of_raw'] for r in rows])
    ev = np.array([r['even_pct'] for r in rows])
    ob = np.array([r['orig_bpw'] for r in rows])
    lay = np.array([r['layer'] for r in rows])
    out = {'n': len(rows), 'mean': float(d.mean()), 'median': float(np.median(d)), 'min_(largest_gain)': float(d.min()),
           'max_(smallest_gain)': float(d.max()), 'p10': float(np.percentile(d, 10)), 'p90': float(np.percentile(d, 90)),
           'n_with_gain_smaller_than_1pct': int((d > -1).sum()), 'n_worse_than_today': int((d > 0).sum()),
           'corr_gain_vs_even_pct': float(np.corrcoef(d, ev)[0, 1]), 'corr_gain_vs_orig_bpw': float(np.corrcoef(d, ob)[0, 1]),
           'per_layer_mean': {int(L): round(float(d[lay == L].mean()), 3) for L in sorted(set(lay.tolist()))},
           'per_layer_even_pct': {int(L): round(float(ev[lay == L].mean()), 2) for L in sorted(set(lay.tolist()))},
           'per_layer_orig_bpw': {int(L): round(float(ob[lay == L].mean()), 4) for L in sorted(set(lay.tolist()))},
           'worst5': sorted([(round(r['d_record_pct_of_raw'], 3), r['file'], round(r['even_pct'], 2)) for r in rows])[-5:],
           'best5': sorted([(round(r['d_record_pct_of_raw'], 3), r['file'], round(r['even_pct'], 2)) for r in rows])[:5]}
    json.dump(out, open(outp, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
