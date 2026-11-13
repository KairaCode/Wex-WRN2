"""Fingerprint + ceil-form held-out gain for item files written by fetch_other.py (any MXFP4 + E8M0 layout).

Usage: python -I other_analyze.py DATA_DIR OUT_JSON
Per item, tensors are paired by 'role' (weight part + scale part). Weight bytes are viewed as [R, bytes_per_row]
(R = first dim of the sliced shape), low nibble = even column; scales as [R, blocks_per_row].
Held-out: shared static WNR2 base model (567 contexts x 16) trained on items with even manifest position, costed on
odd ones, and vice versa; ceil form retrained on transformed data with the median recomputed. Flush 4 B / 2048 weights.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mxfp

KEYS_SUM = ['bmax_hist', 'even', 'blocks', 'max6_no13', 'max5_no13', 'mag_hist', 'mag_hist_even', 'sym_hist', 'scale_hist',
            'roundtrip_fail', 'max_lt6', 'scale_ovf']


def main():
    d, out_json = sys.argv[1], sys.argv[2]
    man = json.load(open(os.path.join(d, 'manifest.json')))
    per, items, tot, byrole = [], [], {}, {}
    for i, e in enumerate(man['experts']):
        raw = np.fromfile(os.path.join(d, e['file']), dtype=np.uint8)
        off = 0
        parts = {}
        for t in e['tensors']:
            parts[(t['role'], t['part'])] = (raw[off:off + t['bytes']], t['shape'], t['dtype'])
            off += t['bytes']
        assert off == raw.size
        acc = {'split': i % 2}
        mats = {}
        for role in sorted({t['role'] for t in e['tensors']}):
            wraw, wshape, wdt = parts[(role, 'weight')]
            sraw, sshape, sdt = parts[(role, 'scale')]
            R = wshape[0]
            wb = wraw.reshape(R, -1)
            sc = sraw.reshape(R, -1)
            assert wb.shape[1] * 2 == sc.shape[1] * 32, (role, wb.shape, sc.shape)
            sym = mxfp.unpack(wb)
            fp = mxfp.analyze_matrix(sym, sc)
            mats[role] = {'shape': [R, wb.shape[1] * 2], 'dtype_w': wdt, 'dtype_s': sdt,
                          'even_pct': 100 * int(fp['even']) / int(fp['blocks']), 'bmax_hist': fp['bmax_hist'].tolist(),
                          'max_lt6': int(fp['max_lt6']), 'p_iid_even_given_max7': float(fp['p_iid_even_given_max7']),
                          'median_scale': int(np.sort(sc.ravel())[(sc.size - 1) // 2])}
            for k in KEYS_SUM + ['w_orig', 'w_ceil', 's_orig', 's_ceil']:
                acc[k] = acc.get(k, 0) + np.asarray(fp[k], np.int64)
            for k in KEYS_SUM:
                tot[k] = tot.get(k, 0) + np.asarray(fp[k], np.int64)
            br = byrole.setdefault(role, {'blocks': 0, 'even': 0, 'bmax_hist': np.zeros(8, np.int64)})
            br['blocks'] += int(fp['blocks']); br['even'] += int(fp['even']); br['bmax_hist'] += fp['bmax_hist']
        row = {k: v for k, v in e.items() if k not in ('tensors', 'sha256')}
        row['matrices'] = mats
        row['even_pct'] = 100 * int(acc['even']) / int(acc['blocks'])
        per.append(row)
        items.append(acc)
        print(e['file'], {r: (round(m['even_pct'], 3), m['bmax_hist']) for r, m in mats.items()}, 'rt_fail', int(acc['roundtrip_fail']),
              flush=True)
    nW = int(sum(int(x['w_orig'].sum()) for x in items))
    nS = int(sum(int(x['s_orig'].sum()) for x in items))
    flush_bits = 32 * nW / 2048
    res = {'family': man.get('family'), 'repo': man.get('repo'), 'revision': man.get('revision'),
           'n_items': len(items), 'weights': nW, 'scales': nS}
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
    # i.i.d. reference over all max-7 blocks: magnitude shares among non-max positions of max-7 blocks are not
    # aggregated here; per-matrix values are in experts[].matrices[].p_iid_even_given_max7
    res['totals'] = T
    res['by_role'] = {r: {'blocks': v['blocks'], 'even_pct': 100 * v['even'] / v['blocks'],
                          'bmax_share_pct': (100 * v['bmax_hist'] / v['blocks']).tolist()} for r, v in byrole.items()}
    res['experts'] = per
    json.dump(res, open(out_json, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('experts',)}, indent=1))


if __name__ == '__main__':
    main()
