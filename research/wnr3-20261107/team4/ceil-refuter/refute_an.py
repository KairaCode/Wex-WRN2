"""ceil-refuter analysis: invariant summary + held-out costs of ceil-form vs side-info alternatives.

Usage: python -I refute_an.py OUT_DIR OUT_JSON
Reads OUT_DIR/counts_sample.npz, counts_extra.npz, stats_sample.json, stats_extra.json (from refute_count.py).
Costs: 14-bit CDFs quantized like wex_nr2_quantize (copy of mxanalyze.quantize), one model per matrix type, trained on
the train groups, costed on the test groups. Weight bit/w includes side symbols and the 4-B flush per 2048-weight stream.
Scale bits exclude the flush (identical for all variants).
"""
import json, os, sys
import numpy as np

P = 16384
RAW_RECORD = 18_800_640
W_PER_REC = 3 * 2304 * 5120
S_PER_REC = W_PER_REC // 32
FLUSH_BPW = 32 / 2048


def quantize(c):
    c = np.asarray(c, np.int64)
    n = c.shape[-1]
    flat = c.reshape(-1, n)
    tot = flat.sum(1, keepdims=True)
    safe = np.where(tot == 0, 1, tot)
    num = flat * (P - n)
    fl = num // safe
    rem = num % safe
    freq = 1 + fl
    left = P - freq.sum(1)
    order = np.argsort(-rem, axis=1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n)[None, :].repeat(flat.shape[0], 0), axis=1)
    freq = freq + (rank < left[:, None])
    freq[tot[:, 0] == 0] = P // n if P % n == 0 else freq[tot[:, 0] == 0]
    if P % n:
        z = tot[:, 0] == 0
        if z.any():
            u = np.full((int(z.sum()), n), P // n, np.int64); u[:, :P - (P // n) * n] += 1
            freq[z] = u
    assert (freq.sum(1) == P).all() and (freq >= 1).all()
    return freq.reshape(c.shape)


def cost_bits(counts, freq):
    return float((np.asarray(counts, np.float64) * -np.log2(np.asarray(freq, np.float64) / P)).sum())


def zmerge(c):
    c = c.copy(); c[..., 0] += c[..., 8]; c[..., 8] = 0; return c


def magonly(c):
    return c[..., :8] + c[..., 8:]


# variant = list of (key, transform) ; key may be 'cls0:<side key>' (class order-0 from a side table)
def comp(acc, key, groups):
    if key.startswith('order0:'):
        k = key.split(':')[1]
        return acc[k][groups].sum(0).sum(1)  # [3, k]  (sum over srel)
    return acc[key][groups].sum(0)


VARIANTS = {
    'V0_today_o_base': (['o_base'], ['s_o']),
    'V7_o_pm2b': (['o_pm2b'], ['s_o']),
    'V1_o_bm7h(+bm7 side)': (['o_bm7h', 'o_side_bm7'], ['s_o']),
    'V2_o_ev(+even flag side, all blocks)': (['o_ev', 'o_side_ev'], ['s_o']),
    'V3_o_cls(+3-class side = bm7 + even bit per max-7 block)': (['o_cls', 'o_side_cls'], ['s_o']),
    'V4_o_clsh(+3-class side)': (['o_clsh', 'o_side_cls'], ['s_o']),
    'V5_o_bm7h_pm2b(team2 best non-ceil)': (['o_bm7h_pm2b', 'o_side_bm7'], ['s_o']),
    'V6_o_clsh_pm2b(+3-class side, scale first)': (['o_clsh_pm2b', 'o_side_cls'], ['s_o']),
    'V6b_o_clsh_pm2b(class order-0 first, scale|class)': (['o_clsh_pm2b', 'order0:o_side_cls'], ['s_o_cls']),
    'H_o_clsh_pm2b + ceil scales (s+even) + bm3 side': (['o_clsh_pm2b', 'c_side_bm3'], ['s_c']),
    'C0_ceil_base(implicit)': (['c_base'], ['s_c']),
    'C1_ceil_pm2b(implicit)': (['c_pm2b'], ['s_c']),
    'C2_ceil_bm3(+bm3 side)': (['c_bm3', 'c_side_bm3'], ['s_c']),
    'C3_ceil_bm3h(+bm3 side)': (['c_bm3h', 'c_side_bm3'], ['s_c']),
    'C4_ceil_bm3h_pm2b(+bm3 side)': (['c_bm3h_pm2b', 'c_side_bm3'], ['s_c']),
    'C4b_ceil_bm3h_pm2b(bm3 order-0 first, scale|bm3)': (['c_bm3h_pm2b', 'order0:c_side_bm3'], ['s_c_bm3']),
}
WEIGHT_KEYS = {'o_base', 'o_pm2b', 'o_ev', 'o_bm7h', 'o_cls', 'o_clsh', 'o_bm7h_pm2b', 'o_clsh_pm2b',
               'c_base', 'c_pm2b', 'c_bm3', 'c_bm3h', 'c_bm3h_pm2b'}


def eval_variant(acc, var, train, test, mode):
    """mode: 'byte' | 'z' (canonical zero) | 'signbypass' (8-symbol magnitudes + 1 raw bit per weight).
    Returns (weight+side bits, scale bits, n_test_experts_weights)."""
    wk, sk = var
    wb = 0.0
    nzero = 0
    for k in wk:
        a = comp(acc, k, train); b = comp(acc, k, test)
        if k in WEIGHT_KEYS:
            if mode == 'z':
                a, b = zmerge(a), zmerge(b)
            elif mode in ('signbypass', 'z_signbypass'):
                a, b = magonly(a), magonly(b)
                nzero = int(b[..., 0].sum())
        wb += cost_bits(b, quantize(a))
    nw = comp(acc, wk[0], test).sum()
    if mode == 'signbypass':
        wb += float(nw)  # 1 raw sign bit per weight (byte-exact, includes the sign of zero)
    elif mode == 'z_signbypass':
        wb += float(nw - nzero)  # canonical zero: raw sign bit only for non-zero magnitudes
    sb = 0.0
    for k in sk:
        a = comp(acc, k, train); b = comp(acc, k, test)
        sb += cost_bits(b, quantize(a))
    return wb, sb, nw


def protocols():
    return {
        'P1_two_way_sample(even<->odd, 160)': [((0, 2), (1, 3)), ((1, 3), (0, 2))],
        'P2_one_way_train_even80_test_odd80': [((0, 2), (1, 3))],
        'P3_team3_train_even80_test_odd40(i%4==1)': [((0, 2), (1,))],
        'P4_train_sample160_test_extra70_routed': [((0, 1, 2, 3), (4,))],
        'P5_train_sample160_test_mtp6': [((0, 1, 2, 3), (5,))],
        'P0_insample_all_routed230': [((0, 1, 2, 3, 4), (0, 1, 2, 3, 4))],
    }


def summarize_stats(stats_all):
    out = {}
    for kind in ['sample', 'extra_routed', 'extra_mtp']:
        rows = [s for s in stats_all if s['_set'] == kind]
        if not rows:
            continue
        ms = [m for s in rows for m in s['m']]
        tot = lambda k: int(sum(m[k] for m in ms))
        hist = np.sum([m['bmax_hist'] for m in ms], 0).tolist()
        per_exp_even = [sum(m['even'] for m in s['m']) / sum(m['blocks'] for m in s['m']) for s in rows]
        per_mat_even = [[m['even'] / m['blocks'] for s in rows for m in [s['m'][j]]] for j in range(3)]
        out[kind] = {
            'experts': len(rows), 'matrices': len(ms), 'blocks': tot('blocks'), 'block_max_hist': hist,
            'blocks_max_lt6': tot('max_lt6'), 'blocks_stored_max5': tot('stored_max5'), 'blocks_allzero': tot('allzero'),
            'even_blocks': tot('even'), 'even_pct': 100 * tot('even') / tot('blocks'),
            'even_pct_per_matrix_type_mean': [100 * float(np.mean(v)) for v in per_mat_even],
            'even_pct_per_expert_min_max': [100 * min(per_exp_even), 100 * max(per_exp_even)],
            'max6_blocks_without_1_or_3': tot('max6_no13'),
            'scale_min': min(m['scale_min'] for m in ms), 'scale_max': max(m['scale_max'] for m in ms),
            'scale_n0': tot('scale_n0'), 'scale_n254': tot('scale_n254'), 'scale_n255': tot('scale_n255'),
            'even_scale_max': max(m['even_scale_max'] for m in ms), 'even_at_254': tot('even_at_254'),
            'even_at_255_overflow': tot('even_at_255_overflow'), 'transformed_scale_max': max(m['ts_max'] for m in ms),
            'roundtrip_fail_matrices': sum(not m['rt_ok'] for m in ms),
            'inverse_selects_exactly_even_fail': sum(not m['inv_selects_exactly_even'] for m in ms),
            'canonical_zero_commute_fail': sum(not m['z_commute'] for m in ms),
            'canonical_zero_roundtrip_fail': sum(not m['z_rt_ok'] for m in ms),
            'sign_plane_changed': sum(not m['sign_plane_kept'] for m in ms),
            'neg0_fraction': tot('neg0') / (tot('blocks') * 32),
            'median_changed_matrices': sum(m['med_o'] != m['med_c'] for m in ms),
            'median_shift_hist': {str(d): sum((m['med_c'] - m['med_o']) == d for m in ms) for d in (-1, 0, 1, 2)},
            'not_counted_matrices': sum(not m.get('counted') for m in ms),
        }
        if kind != 'sample':
            out[kind]['per_expert'] = [{'f': s['_file'], 'even_pct': round(100 * sum(m['even'] for m in s['m']) / sum(m['blocks'] for m in s['m']), 3),
                                        'lt6': sum(m['max_lt6'] for m in s['m']), 'rt': all(m['rt_ok'] for m in s['m']),
                                        'smin': min(m['scale_min'] for m in s['m']), 'smax': max(m['scale_max'] for m in s['m'])} for s in rows]
    shas = [m['sha256'] for s in stats_all for m in s['m']]
    out['duplicate_matrices_all_sets'] = len(shas) - len(set(shas))
    # rule-of-three upper bounds for invariant violations (per record / over 7680 records)
    nrec = sum(len([s for s in stats_all if s['_set'] == k]) for k in ['sample', 'extra_routed'])
    nblk = sum(out[k]['blocks'] for k in ['sample', 'extra_routed'] if k in out)
    viol = sum(out[k]['blocks_stored_max5'] + out[k]['even_at_255_overflow'] for k in ['sample', 'extra_routed'] if k in out)
    out['fallback_estimate'] = {
        'records_checked_routed': nrec, 'blocks_checked_routed': nblk, 'violating_blocks': viol,
        'records_violating': sum(any(m['stored_max5'] or m['even_at_255_overflow'] for m in s['m']) for s in stats_all),
        'upper95_records_violating_fraction_rule_of_3': 3 / nrec,
        'upper95_expected_fallback_records_of_7680': 7680 * 3 / nrec,
        'point_estimate_fallback_records_of_7680': 0 if viol == 0 else None,
    }
    return out


def main():
    d, out_json = sys.argv[1:3]
    acc = None
    for f in ['counts_sample.npz', 'counts_extra.npz']:
        z = np.load(os.path.join(d, f))
        if acc is None:
            acc = {k: z[k].astype(np.int64) for k in z.files}
        else:
            for k in z.files:
                acc[k] += z[k]
    stats_all = []
    for f, kind in [('stats_sample.json', 'sample'), ('stats_extra.json', None)]:
        st = json.load(open(os.path.join(d, f)))
        for fn, s in st.items():
            s['_file'] = fn
            s['_set'] = kind or ('extra_routed' if s['kind'] == 'layers' else 'extra_mtp')
            stats_all.append(s)
    res = {'invariants': summarize_stats(stats_all)}
    res['group_weights'] = [int(acc['o_base'][g].sum()) for g in range(6)]
    tabs = {}
    for pname, pairs in protocols().items():
        for mode in ['byte', 'z', 'signbypass', 'z_signbypass']:
            rows = {}
            for vname, var in VARIANTS.items():
                wb = sb = nw = 0.0
                for tr, te in pairs:
                    a, b, n = eval_variant(acc, var, list(tr), list(te), mode)
                    wb += a; sb += b; nw += n
                ns = nw / 32
                rows[vname] = {'bpw_incl_side_flush': wb / nw + FLUSH_BPW, 'bit_per_scale': sb / ns,
                               'record_bytes_payload': (wb / nw + FLUSH_BPW) * W_PER_REC / 8 + sb / ns * S_PER_REC / 8}
            base = rows['V0_today_o_base']
            c0 = rows['C0_ceil_base(implicit)']
            for vname, r in rows.items():
                r['d_weight_pct_vs_V0'] = 100 * (r['bpw_incl_side_flush'] - base['bpw_incl_side_flush']) / base['bpw_incl_side_flush']
                r['d_record_pct_of_raw_vs_V0'] = 100 * (r['record_bytes_payload'] - base['record_bytes_payload']) / RAW_RECORD
                gain_c0 = base['record_bytes_payload'] - c0['record_bytes_payload']
                r['share_of_C0_ceil_gain_captured'] = (base['record_bytes_payload'] - r['record_bytes_payload']) / gain_c0
            non = {k: v for k, v in rows.items() if k[0] in 'V'}
            cei = {k: v for k, v in rows.items() if k[0] == 'C'}
            bn = min(non, key=lambda k: non[k]['record_bytes_payload'])
            bc = min(cei, key=lambda k: cei[k]['record_bytes_payload'])
            hy = rows['H_o_clsh_pm2b + ceil scales (s+even) + bm3 side']
            summary = {'best_non_ceil': bn, 'best_ceil': bc,
                       'ceil_on_top_of_best_non_ceil_record_pct_of_raw': 100 * (cei[bc]['record_bytes_payload'] - non[bn]['record_bytes_payload']) / RAW_RECORD,
                       'ceil_on_top_of_best_non_ceil_weight_pct': 100 * (cei[bc]['bpw_incl_side_flush'] - non[bn]['bpw_incl_side_flush']) / non[bn]['bpw_incl_side_flush'],
                       'ceil_C4_vs_hybrid_H_record_pct_of_raw': 100 * (rows['C4_ceil_bm3h_pm2b(+bm3 side)']['record_bytes_payload'] - hy['record_bytes_payload']) / RAW_RECORD}
            tabs[f'{pname}|{mode}'] = {'summary': summary, 'rows': rows}
            print(pname, mode, json.dumps(summary), flush=True)
    res['heldout'] = tabs
    # per-record fallback penalty: a record left in original form (T off) but coded with the ceil-trained base model
    # (srel vs its own median), vs the same record coded with a model trained on original-form data.
    fb = {}
    for pname, tr, te in [('P2', [0, 2], [1, 3]), ('P4', [0, 1, 2, 3], [4])]:
        b = comp(acc, 'o_base', te)
        nw = b.sum()
        own = cost_bits(b, quantize(comp(acc, 'o_base', tr))) / nw + FLUSH_BPW
        cross = cost_bits(b, quantize(comp(acc, 'c_base', tr))) / nw + FLUSH_BPW
        sc_own = cost_bits(comp(acc, 's_o', te), quantize(comp(acc, 's_o', tr))) / (nw / 32)
        sc_cross = cost_bits(comp(acc, 's_o', te), quantize(comp(acc, 's_c', tr))) / (nw / 32)
        fb[pname] = {'orig_form_with_orig_model_bpw': own, 'orig_form_with_ceil_model_bpw': cross,
                     'orig_scales_with_orig_model_bps': sc_own, 'orig_scales_with_ceil_model_bps': sc_cross,
                     'penalty_record_pct_of_raw': 100 * ((cross - own) * W_PER_REC / 8 + (sc_cross - sc_own) * S_PER_REC / 8) / RAW_RECORD}
    res['fallback_record_coded_with_ceil_model'] = fb
    print('fallback', json.dumps(fb))
    json.dump(res, open(out_json, 'w'), indent=1)
    # compact print
    for key in ['P1_two_way_sample(even<->odd, 160)|byte', 'P4_train_sample160_test_extra70_routed|byte']:
        print(key)
        for v, r in tabs[key]['rows'].items():
            print(f"  {v:60s} bpw {r['bpw_incl_side_flush']:.4f} bps {r['bit_per_scale']:.4f} dW {r['d_weight_pct_vs_V0']:+.3f}% dRec {r['d_record_pct_of_raw_vs_V0']:+.3f}% share {r['share_of_C0_ceil_gain_captured']:.3f}")


if __name__ == '__main__':
    main()
