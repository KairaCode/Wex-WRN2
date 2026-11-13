"""Held-out gains of extra contexts on ceil-form data (from mxceil2.py npz files).

Usage: python -I mxceil2_an.py SAMPLE_DIR CEIL2_DIR OUT_JSON
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxanalyze import quantize, cost_bits, W_PER_REC, RAW_RECORD

FLUSH_PER_REC = 28 * (5760 * 3)  # exact rANS flush is ~3.5 B/stream (team 2 audit), same for every variant


def main():
    sample_dir, d, out_json = sys.argv[1:4]
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    ex = []
    for i, e in enumerate(man['experts']):
        z = np.load(os.path.join(d, e['file'].replace('.bin', '.npz')))
        r = {'split': i % 2, **{k: z[k] for k in z.files}}
        # 3-symbol side alphabet: pad to 4 (unused symbol gets freq 1) so the shared 14-bit quantizer's empty-row rule holds
        r['bm_side'] = np.concatenate([r['bm_side'], np.zeros(r['bm_side'].shape[:-1] + (1,), np.int64)], -1)
        ex.append(r)
    n = len(ex)
    W = n * W_PER_REC; fl = n * FLUSH_PER_REC

    def S(k, sp=None):
        return sum(e[k].astype(np.int64) for e in ex if sp is None or e['split'] == sp)

    def ho(k, tr=lambda c: c):
        a = tr(S(k, 0)); b = tr(S(k, 1)); c = a + b
        return cost_bits(c, quantize(c)), cost_bits(b, quantize(a)) + cost_bits(a, quantize(b))

    def zm(c):
        c = c.copy(); c[..., 0] += c[..., 8]; c[..., 8] = 0; return c

    si, sh = ho('bm_side')
    out = {'experts': n, 'side_bits_per_w_ho': sh / W}
    bi, bh = ho('base')
    out['base'] = {'in_bpw': (bi + fl) / W, 'ho_bpw': (bh + fl) / W}
    for k, side in [('pm2', False), ('pm2b', False), ('bmh', True), ('bmh_pm2b', True), ('pos', False)]:
        i, h = ho(k)
        if side:
            i += si; h += sh
        used = int((S(k).sum(-1) > 0).sum(-1).max())
        out[k] = {'in_bpw': (i + fl) / W, 'ho_bpw': (h + fl) / W, 'delta_ho_pct_of_weight_bits': 100 * (h - bh) / (bh + fl),
                  'delta_ho_pct_of_raw_record': 100 * (h - bh) / n / 8 / RAW_RECORD,
                  'rows_used_max': used, 'table_kb': used * 32 / 1024}
    for k, side in [('base', False), ('bmh_pm2b', True)]:
        i, h = ho(k, zm)
        if side:
            h += sh
        out[f'{k}_z'] = {'ho_bpw': (h + fl) / W, 'delta_ho_pct_of_weight_bits': 100 * (h - bh) / (bh + fl)}
    json.dump(out, open(out_json, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
