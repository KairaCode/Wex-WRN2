"""Fingerprint + ceil-form held-out gain on DeepSeek-layout expert files (w1.scale w2.scale w3.scale w1.weight w2.weight w3.weight).

Usage: python -I ds_analyze.py DATA_DIR OUT_JSON [SAMPLE_DIR]
If SAMPLE_DIR is given, experts with a 'sample_file' entry are compared byte for byte with that file of the v41 sample.
Held-out split: expert position in the manifest, even -> split 0, odd -> split 1 (both directions summed).
"""
import json, os, sys, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mxfp

KEYS_SUM = ['bmax_hist', 'even', 'blocks', 'max6_no13', 'max5_no13', 'mag_hist', 'mag_hist_even', 'sym_hist', 'scale_hist',
            'roundtrip_fail', 'max_lt6', 'scale_ovf']


def load(path, tensors):
    raw = np.fromfile(path, dtype=np.uint8)
    out, off = {}, 0
    for t in tensors:
        out[t['name']] = (raw[off:off + t['bytes']], t['shape'])
        off += t['bytes']
    assert off == raw.size
    return out


def main():
    d, out_json = sys.argv[1], sys.argv[2]
    sample = sys.argv[3] if len(sys.argv) > 3 else None
    man = json.load(open(os.path.join(d, 'manifest.json')))
    per = []
    items = []
    tot = {}
    for i, e in enumerate(man['experts']):
        t = load(os.path.join(d, e['file']), e['tensors'])
        row = {k: v for k, v in e.items() if k not in ('tensors', 'sha256')}
        if sample and e.get('sample_file'):
            sp = os.path.join(sample, e['sample_file'])
            row['identical_to_sample'] = open(sp, 'rb').read() == open(os.path.join(d, e['file']), 'rb').read()
        acc = {'split': i % 2}
        mats = {}
        for w in ['w1', 'w2', 'w3']:
            wraw, wshape = t[f'{w}.weight']
            sraw, sshape = t[f'{w}.scale']
            R = wshape[0]
            sym = mxfp.unpack(wraw.reshape(R, -1))
            sc = sraw.reshape(R, -1)
            fp = mxfp.analyze_matrix(sym, sc)
            mats[w] = {'even_pct': 100 * int(fp['even']) / int(fp['blocks']), 'bmax_hist': fp['bmax_hist'].tolist(),
                       'max_lt6': int(fp['max_lt6']), 'p_iid_even_given_max7': float(fp['p_iid_even_given_max7']),
                       'median_scale': int(np.sort(sc.ravel())[(sc.size - 1) // 2])}
            for k in KEYS_SUM + ['w_orig', 'w_ceil', 's_orig', 's_ceil']:
                acc[k] = acc.get(k, 0) + np.asarray(fp[k], np.int64)
            for k in KEYS_SUM:
                tot[k] = tot.get(k, 0) + np.asarray(fp[k], np.int64)
        row['matrices'] = mats
        row['even_pct'] = 100 * int(acc['even']) / int(acc['blocks'])
        per.append(row)
        items.append(acc)
        print(e['file'], round(row['even_pct'], 3), 'max<6:', int(acc['max_lt6']), 'rt_fail:', int(acc['roundtrip_fail']),
              'same_as_sample:', row.get('identical_to_sample'), flush=True)
    nW = int(sum(int(x['w_orig'].sum()) for x in items))
    nS = int(sum(int(x['s_orig'].sum()) for x in items))
    flush_bits = 32 * nW / 2048  # 4 B per 2048-weight stream, same for both forms
    res = {'n_experts': len(items), 'weights': nW, 'scales': nS}
    if len(items) >= 2:
        for name, k, tr in [('orig', 'w_orig', lambda c: c), ('ceil', 'w_ceil', lambda c: c),
                            ('orig_z', 'w_orig', mxfp.zmerge), ('ceil_z', 'w_ceil', mxfp.zmerge)]:
            i_, h_ = mxfp.heldout(items, k, tr)
            res[f'w_{name}'] = {'in_bpw': (i_ + flush_bits) / nW, 'ho_bpw': (h_ + flush_bits) / nW}
        for name, k in [('orig', 's_orig'), ('ceil', 's_ceil')]:
            i_, h_ = mxfp.heldout(items, k)
            res[f's_{name}'] = {'in_bps': i_ / nS, 'ho_bps': h_ / nS}
        base = res['w_orig']['ho_bpw']
        for name in ['ceil', 'orig_z', 'ceil_z']:
            res[f'w_{name}']['delta_pct_vs_orig_ho'] = 100 * (res[f'w_{name}']['ho_bpw'] - base) / base
        raw_bits = nW * 4 + nS * 8
        dw = (res['w_ceil']['ho_bpw'] - base) * nW
        ds = (res['s_ceil']['ho_bps'] - res['s_orig']['ho_bps']) * nS
        res['ceil_delta_pct_of_raw'] = 100 * (dw + ds) / raw_bits
        res['orig_ratio_est'] = (base * nW + res['s_orig']['ho_bps'] * nS) / raw_bits
        res['ceil_ratio_est'] = res['orig_ratio_est'] + (dw + ds) / raw_bits
    T = {k: (v.tolist() if v.ndim else int(v)) for k, v in tot.items()}
    T['even_pct'] = 100 * T['even'] / T['blocks']
    T['bmax_share_pct'] = [100 * x / T['blocks'] for x in T['bmax_hist']]
    sh = np.array(T['sym_hist'], np.float64)
    T['zero_pos_pct'] = 100 * sh[0] / sh.sum(); T['zero_neg_pct'] = 100 * sh[8] / sh.sum()
    T['scale_hist'] = {str(i): c for i, c in enumerate(T['scale_hist']) if c}
    res['totals'] = T
    res['experts'] = per
    json.dump(res, open(out_json, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'experts'}, indent=1))


if __name__ == '__main__':
    main()
