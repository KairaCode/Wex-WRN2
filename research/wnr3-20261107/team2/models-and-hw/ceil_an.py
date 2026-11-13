"""Held-out costs of the static WNR2 model on canonical vs ceil-form blocks (counts from ceilform.py).
Same protocol as mxanalyze/mxcombo_an: split = manifest index % 2, train on one, test on the other, both ways;
14-bit CDFs quantized like wex_nr2_quantize; flush 4 B per 2048-weight / 512-scale stream added.
Usage: python -I ceil_an.py SAMPLE_DIR CEIL_DIR OUT_JSON"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tableadapt import quantize, P

RAW = 18_800_640
W_REC = 3 * 2304 * 5120
S_REC = W_REC // 32


def cost(c, f):
    return float((c * -np.log2(f / P)).sum())


def main():
    sample_dir, d, out_json = sys.argv[1:4]
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    E = []
    for i, e in enumerate(man['experts']):
        p = os.path.join(d, e['file'].replace('.bin', '.npz'))
        if os.path.exists(p):
            z = np.load(p, allow_pickle=False)
            E.append({'split': i % 2, 'layer': e['layer'], **{k: z[k].astype(np.int64) for k in z.files}})
    n = len(E)
    out = {'experts': n}
    W = n * W_REC; S = n * S_REC
    flush_w = 32 * n * 3 * 5760; flush_s = 32 * n * 3 * 720
    def ho(key):
        a = sum(e[key] for e in E if e['split'] == 0); b = sum(e[key] for e in E if e['split'] == 1)
        return cost(b, quantize(a)) + cost(a, quantize(b)), cost(a + b, quantize(a + b))
    r = {}
    for form in ['can', 'ceil']:
        wh, wi = ho(f'w_{form}')
        sh, si = ho(f's_{form}')
        side_h, _ = ho(f'side_{form}')
        r[form] = {'weights_bpw_heldout': (wh + flush_w) / W, 'weights_bpw_insample': (wi + flush_w) / W,
                   'scales_bit_per_scale_heldout': (sh + flush_s) / S,
                   'record_payload_bytes_heldout': (wh + flush_w + sh + flush_s) / 8 / n,
                   'blockmax_side_bits_per_w (if coded as side info, not used here)': side_h / W}
    r['delta_weights_pct'] = 100 * (r['ceil']['weights_bpw_heldout'] / r['can']['weights_bpw_heldout'] - 1)
    r['delta_scales_pct'] = 100 * (r['ceil']['scales_bit_per_scale_heldout'] / r['can']['scales_bit_per_scale_heldout'] - 1)
    lt = 2 * 3 * (5760 + 720) + 32 + 2048
    for form in ['can', 'ceil']:
        r[form]['record_bytes_est'] = r[form]['record_payload_bytes_heldout'] + lt
        r[form]['ratio_vs_raw'] = r[form]['record_bytes_est'] / RAW
    r['delta_record_pct'] = 100 * (r['ceil']['record_bytes_est'] / r['can']['record_bytes_est'] - 1)
    r['even_blocks_fraction'] = float(sum(e['meta'][:, 4].sum() for e in E) / (n * S_REC))
    r['median_changed_matrices'] = int(sum((e['meta'][:, 2] != e['meta'][:, 3]).sum() for e in E))
    out['result'] = r
    print(json.dumps(out, indent=1))
    json.dump(out, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
