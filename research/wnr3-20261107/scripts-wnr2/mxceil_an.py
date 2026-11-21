"""Held-out comparison of WNR2 base model on original vs ceil-form data (+ zero-sign merge), from mxceil.py npz files.

Usage: python -I mxceil_an.py SAMPLE_DIR CEIL_DIR OUT_JSON
Held-out = shared static model trained on split A (expert index even/odd), costed on split B, both ways.
Bits per weight include the 4-B flush per 2048-weight stream (same for both forms).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxanalyze import quantize, cost_bits, P, W_PER_REC, S_PER_REC, RAW_RECORD

FLUSH_PER_REC = 32 * (5760 * 3)


def main():
    sample_dir, d, out_json = sys.argv[1:4]
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    ex = []
    for i, e in enumerate(man['experts']):
        z = np.load(os.path.join(d, e['file'].replace('.bin', '.npz')))
        ex.append({'split': i % 2, 'layer': e['layer'], **{k: z[k] for k in z.files}})
    n = len(ex)

    def S(k, f=lambda e: True):
        return sum(e[k].astype(np.int64) for e in ex if f(e))

    def ho(k, transform=lambda c: c):
        a = transform(S(k, lambda e: e['split'] == 0)); b = transform(S(k, lambda e: e['split'] == 1)); c = a + b
        return cost_bits(c, quantize(c)), cost_bits(b, quantize(a)) + cost_bits(a, quantize(b))

    def zmerge(c):
        c = c.copy(); c[..., 0] += c[..., 8]; c[..., 8] = 0; return c

    W = n * W_PER_REC; Sn = n * S_PER_REC; fl = n * FLUSH_PER_REC
    out = {'experts': n}
    for name, k, tr in [('orig', 'w_orig', lambda c: c), ('ceil', 'w_ceil', lambda c: c),
                        ('orig_z', 'w_orig', zmerge), ('ceil_z', 'w_ceil', zmerge)]:
        i, h = ho(k, tr)
        out[f'w_{name}'] = {'in_bpw': (i + fl) / W, 'ho_bpw': (h + fl) / W}
    for name, k in [('orig', 's_orig'), ('ceil', 's_ceil')]:
        i, h = ho(k)
        out[f's_{name}'] = {'in_bps': i / Sn, 'ho_bps': h / Sn}
    base = out['w_orig']['ho_bpw']
    for name in ['ceil', 'orig_z', 'ceil_z']:
        out[f'w_{name}']['delta_pct_vs_orig_ho'] = 100 * (out[f'w_{name}']['ho_bpw'] - base) / base
    dw = (out['w_ceil']['ho_bpw'] - base) * W_PER_REC / 8
    ds = (out['s_ceil']['ho_bps'] - out['s_orig']['ho_bps']) * S_PER_REC / 8
    out['record_delta_bytes_ceil'] = dw + ds
    out['record_delta_pct_of_raw_ceil'] = 100 * (dw + ds) / RAW_RECORD
    dwz = (out['w_ceil_z']['ho_bpw'] - base) * W_PER_REC / 8
    out['record_delta_pct_of_raw_ceil_z'] = 100 * (dwz + ds) / RAW_RECORD
    st = S('stats')  # [3, 12]
    out['block_max_hist_per_matrix'] = st[:, :8].tolist()
    out['even_block_pct_per_matrix'] = (100 * st[:, 8] / st[:, :8].sum(1)).tolist()
    out['blocks_with_max_below_6'] = int(st[:, 9].sum())
    out['scale_overflow_255'] = int(st[:, 10].sum())
    out['roundtrip_failures'] = int(st[:, 11].sum())
    # i.i.d. sanity: probability that a max-7 block has no magnitude 1 or 3 if its other 31 symbols were drawn
    # from the matrix's overall magnitude distribution restricted to max-7 blocks is not available from counts;
    # report the global fraction of magnitudes 1 and 3 instead.
    wc = S('w_orig').sum((0, 1))
    mag = np.array([wc[m] + wc[m + 8] for m in range(8)], np.float64)
    out['magnitude_share'] = (mag / mag.sum()).tolist()
    p13 = (mag[1] + mag[3]) / mag.sum()
    out['p_no_1_or_3_in_31_iid'] = float((1 - p13) ** 31)
    json.dump(out, open(out_json, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
