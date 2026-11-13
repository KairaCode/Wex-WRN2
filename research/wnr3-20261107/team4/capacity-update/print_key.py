"""Compact print of the requested numbers from capacity_update_results.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, 'capacity_update_results.json')))
R = lambda v, n=2: [round(x, n) for x in v]
for mode in ('W2', 'noW2_single_helper'):
    for d in ('U', 'E'):
        c = r['mc'][mode][d]
        print(mode, d, 'WEX1 tok/s', R(c['WEX1']['tok_s'], 3), 'GB/t', R(c['WEX1']['GB_token'], 3),
              'reads', R(c['WEX1']['reads_token'], 1))
        for f in ('WNR2', 'WNR3', 'WNR3t', 'WNR3z'):
            e = c[f]
            a = e['gain_vs_WEX1_%']
            s = (f"  {f} tok/s {R(e['tok_s'], 3)} vsWEX1 {R(a['p5_p50_p95'])} P3 {a['P(>=+3%)']:.4f} "
                 f"bytesWEX1 {e['bytes_vs_WEX1_%(fewer)'][1]:.3f} elWEX1 {R(e['elasticity_vs_WEX1'], 3)}")
            if 'gain_vs_WNR2_%' in e:
                b = e['gain_vs_WNR2_%']
                s += (f" vsWNR2 {R(b['p5_p50_p95'])} P3 {b['P(>=+3%)']:.4f} bytesWNR2 {R(e['bytes_vs_WNR2_%(fewer)'], 3)} "
                      f"elWNR2 {R(e['elasticity_vs_WNR2'], 3)}")
            print(s)
print(r['equivalence_simulate2_vs_team3'])
for k, v in r['per_expert_delta_vs_wnr2_bytes'].items():
    print(k, {kk: round(vv, 4) for kk, vv in v.items()})
print(r['reproduction_team3'])
