"""Cross-entropy estimate (same tables as e2e_real.py) for exactly the records encoded there, to compare the real
rANS record sizes with the estimate. Usage: python -I -B e2e_est.py COUNTS_SAMPLE_NPZ SAMPLE_DIR EXTRA_DIR E2E_JSON OUT_JSON"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refute_count as rc
from refute_an import quantize, cost_bits


def main():
    cnp, sdir, edir, e2e, outp = sys.argv[1:6]
    z = np.load(cnp)
    q = {k: quantize(z[k][[0, 2]].sum(0)) for k in ['o_base', 'c_base', 's_o', 's_c']}
    rec = json.load(open(e2e))['records']
    mans = {sdir: json.load(open(os.path.join(sdir, 'manifest.json'))), edir: json.load(open(os.path.join(edir, 'manifest.json')))}
    out = []
    for r in rec:
        d = sdir if r['set'] == 'sample_odd' else edir
        e = [x for x in mans[d]['experts'] if x['file'] == r['file']][0]
        acc = rc.new_acc()
        raw = np.fromfile(os.path.join(d, e['file']), dtype=np.uint8)
        off, parts = 0, {}
        for t in e['tensors']:
            parts[t['name']] = raw[off:off + t['bytes']]; off += t['bytes']
        for m, w in enumerate(['w1', 'w2', 'w3']):
            rc.do_matrix(parts[f'{w}.weight'], parts[f'{w}.scale'], 0, acc, 0, m)
        eo = cost_bits(acc['o_base'][0], q['o_base']) / 8; ec = cost_bits(acc['c_base'][0], q['c_base']) / 8
        so = cost_bits(acc['s_o'][0], q['s_o']) / 8; scl = cost_bits(acc['s_c'][0], q['s_c']) / 8
        est = 100 * ((ec + scl) - (eo + so)) / raw.size
        real_payload = 100 * ((r['ceil']['weight_payload'] + r['ceil']['scale_payload']) - (r['orig']['weight_payload'] + r['orig']['scale_payload'])) / raw.size
        out.append({'file': r['file'], 'set': r['set'], 'est_payload_delta_pct_of_raw': est, 'real_payload_delta_pct_of_raw': real_payload,
                    'real_used_delta_pct_of_raw': r['ceil_vs_orig_used_pct_of_raw'],
                    'est_weight_bytes_orig': eo, 'real_weight_payload_orig': r['orig']['weight_payload'],
                    'est_weight_bytes_ceil': ec, 'real_weight_payload_ceil': r['ceil']['weight_payload']})
        print(json.dumps(out[-1]), flush=True)
    json.dump(out, open(outp, 'w'), indent=1)


if __name__ == '__main__':
    main()
