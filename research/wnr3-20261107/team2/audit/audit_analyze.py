"""team2/audit pass 2: entropies, 14-bit costs and held-out gains from audit_count.py output.

Usage: python -I audit_analyze.py SAMPLE_DIR COUNTS_DIR OUT_JSON
Conventions: W = weights; flush = 32 bits per 2048-weight stream (the measured-v41 convention; exact rANS
bytes are measured separately in audit_rans.py). 'delta %' = 100*(variant_bits + side_bits - base_bits)/(base_bits + flush_bits),
the same normalisation as mxanalyze.py:124 / mxcombo_an.py:45, so numbers are directly comparable.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import quantize, cost_bits, ideal_bits

W_PER_MAT = 2304 * 5120
P2B = np.array([1, 1, 1, 1, 2, 2, 3, 3, 0])


def load(sample_dir, counts_dir):
    with open(os.path.join(sample_dir, 'manifest.json'), 'r', encoding='utf-8') as f:
        man = json.load(f)
    ex = []
    for i, e in enumerate(man['experts']):
        p = os.path.join(counts_dir, e['file'].replace('.bin', '.npz'))
        if not os.path.exists(p):
            continue
        z = np.load(p)
        ex.append({'i': i, 'layer': e['layer'], 'expert': e['expert'], **{k: z[k] for k in z.files}})
    return ex


def variants(e):
    """Per-expert count tables [3, rows, 16] for every model variant (derived by marginalising J1/J2)."""
    J1 = e['J1'].reshape(3, 567, 2, 32, 16)
    J2 = e['J2'].reshape(3, 567, 2, 2, 9, 16)
    v = {}
    v['base'] = J1.sum((2, 3))
    v['bm7'] = J1.sum(3).reshape(3, 567 * 2, 16)
    v['half'] = J1.sum(2).reshape(3, 567, 2, 16, 16).sum(3).reshape(3, 567 * 2, 16)
    v['bm7_half'] = J1.reshape(3, 567, 2, 2, 16, 16).sum(4).reshape(3, 567 * 4, 16)
    v['pos32'] = J1.sum(2).reshape(3, 567 * 32, 16)
    v['parity'] = J1.sum(2).reshape(3, 567, 16, 2, 16).sum(2).reshape(3, 567 * 2, 16)
    v['bm7_pos32'] = J1.reshape(3, 567 * 64, 16)
    pm2 = J2.sum((2, 3))  # [3, 567, 9, 16]
    v['pm2'] = pm2.reshape(3, 567 * 9, 16)
    b = np.zeros((3, 567, 4, 16), np.int64)
    for k in range(9):
        b[:, :, P2B[k]] += pm2[:, :, k]
    v['p2b'] = b.reshape(3, 567 * 4, 16)
    bb = np.zeros((3, 567, 2, 2, 4, 16), np.int64)
    for k in range(9):
        bb[..., P2B[k], :] += J2[..., k, :]
    v['bm7_p2b'] = bb.sum(3).reshape(3, 567 * 8, 16)
    v['bm7_p2b_half'] = bb.reshape(3, 567 * 16, 16)
    v['bm7_pm2_half'] = J2.reshape(3, 567 * 36, 16)
    z = v['bm7_p2b'].copy()
    z[..., 0] += z[..., 8]; z[..., 8] = 0
    v['z_bm7_p2b'] = z
    z0 = v['base'].copy()
    z0[..., 0] += z0[..., 8]; z0[..., 8] = 0
    v['z_base'] = z0
    zh = v['bm7_half'].copy()
    zh[..., 0] += zh[..., 8]; zh[..., 8] = 0
    v['z_bm7_half'] = zh
    # side information: bm7 per block given srel, [3, 7, 2]
    bh = e['bmhist']
    v['side_bm7'] = np.stack([bh[..., :7].sum(-1), bh[..., 7]], -1)
    v['side_bm7_nosrel'] = v['side_bm7'].sum(1, keepdims=True)
    return {k: a.astype(np.int32) for k, a in v.items()}  # per-expert cells < 2^31; sums are done in int64


NEEDS_SIDE = {'bm7', 'bm7_half', 'bm7_pos32', 'bm7_p2b', 'bm7_p2b_half', 'bm7_pm2_half', 'z_bm7_p2b', 'z_bm7_half'}


def heldout(tables, folds):
    """tables: list of per-expert arrays; folds: list of fold ids. Train on all other folds, test on each fold."""
    tot = sum(t.astype(np.int64) for t in tables)
    bits = 0.0
    for f in sorted(set(folds)):
        test = sum(t.astype(np.int64) for t, g in zip(tables, folds) if g == f)
        bits += cost_bits(test, quantize(tot - test))
    return bits


def main():
    sample_dir, counts_dir, out_json = sys.argv[1:4]
    ex = load(sample_dir, counts_dir)
    n = len(ex)
    W = n * 3 * W_PER_MAT
    flush = 32.0 * W / 2048
    S = n * 3 * W_PER_MAT // 32
    sflush = 32.0 * S / 512
    V = []
    for e in ex:
        V.append(variants(e))
        del e['J1'], e['J2']
    keys = list(V[0].keys())
    T = {k: sum(v[k].astype(np.int64) for v in V) for k in keys}
    out = {'experts': n, 'layers': len({e['layer'] for e in ex}), 'weights': W,
           'sha_all_ok': bool(all(int(e['meta'][:, 6].min()) == 1 for e in ex))}
    meta = np.stack([e['meta'] for e in ex])  # [n, 3, 7]
    out['median_lower_ne_upper_matrices'] = int((meta[:, :, 2] != meta[:, :, 3]).sum())
    out['frac_code0'] = float(meta[:, :, 4].sum() / W)
    out['frac_code8'] = float(meta[:, :, 5].sum() / W)
    pe0 = meta[:, :, 4].sum(1) / (3 * W_PER_MAT); pe8 = meta[:, :, 5].sum(1) / (3 * W_PER_MAT)
    out['per_expert_code8_minus_code0_range'] = [float((pe8 - pe0).min()), float((pe8 - pe0).max())]
    bh = sum(e['bmhist'] for e in ex)  # [3, 7, 8]
    out['blockmax_hist_per_matrix'] = {f'm{m}': (bh[m].sum(0) / bh[m].sum()).round(6).tolist() for m in range(3)}
    out['blockmax_hist_all'] = (bh.sum((0, 1)) / bh.sum()).round(6).tolist()
    out['blocks_per_srel_frac'] = (bh.sum((0, 2)) / bh.sum()).round(6).tolist()
    out['p_bm7_given_srel'] = (bh[..., 7].sum(0) / bh.sum((0, 2))).round(4).tolist()

    base = T['base']
    ib = ideal_bits(base)
    qb = cost_bits(base, quantize(base))
    out['base'] = {'ideal_bit_per_w': ib / W, 'quant14_bit_per_w': qb / W, 'quant_cost_bit_per_w': (qb - ib) / W,
                   'quant14_plus_4B_flush_bit_per_w': (qb + flush) / W,
                   'per_matrix_quant14_bit_per_w_noflush': [cost_bits(base[m], quantize(base[m])) / (n * W_PER_MAT) for m in range(3)],
                   'order0_quant14_bit_per_w': cost_bits(base.sum(1), quantize(base.sum(1))) / W,
                   'order0_ideal_bit_per_w': ideal_bits(base.sum(1)) / W}
    # sensitivity: rule variants
    sw = sum(e['base_sw'] for e in ex)
    b4 = base.reshape(3, 7, 9, 9, 16)
    out['sensitivity'] = {
        'nibble_order_swapped_ideal_bit_per_w': ideal_bits(sw) / W,
        'mx_never_updates_ie_context_srel_pm_only_ideal': ideal_bits(b4.sum(3)) / W,
        'srel_only_ideal': ideal_bits(b4.sum((2, 3))) / W,
        'no_srel_ideal': ideal_bits(b4.sum(1)) / W,
    }
    # WNR2z
    zb = T['z_base']
    out['wnr2z_base'] = {'ideal_bit_per_w': ideal_bits(zb) / W, 'quant14_plus_flush': (cost_bits(zb, quantize(zb)) + flush) / W}
    # sign-of-zero information per context: sum_ctx n_zero * H(sign | ctx)
    n0 = base[..., 0].astype(np.float64); n8 = base[..., 8].astype(np.float64)
    nz = n0 + n8
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(nz > 0, n8 / nz, 0.5)
        h2 = np.where((p > 0) & (p < 1), -(p * np.log2(p) + (1 - p) * np.log2(1 - p)), 0.0)
    out['sign_of_zero_bits_per_w_given_ctx'] = float((nz * h2).sum() / W)
    out['sign_of_zero_max_abs_p_minus_half_ctx_with_1e5_zeros'] = float(np.abs(p - 0.5)[nz > 1e5].max())

    # side info
    sb = T['side_bm7']
    out['bm7_side'] = {'ideal_bits_per_block_given_srel': ideal_bits(sb) / S,
                       'ideal_bits_per_block_no_srel': ideal_bits(T['side_bm7_nosrel']) / S,
                       'quant_bits_per_w_given_srel': cost_bits(sb, quantize(sb)) / W}

    # in-sample + held-out for every variant
    splits = {
        'S1_manifest_parity (theirs)': [e['i'] % 2 for e in ex],
        'S2_layer_parity': [e['layer'] % 2 for e in ex],
        'S3_slot_in_layer_4fold': [e['i'] % 4 for e in ex],
        'S4_layer_blocks_4fold (0-9,10-19,20-29,30-39)': [e['layer'] // 10 for e in ex],
    }
    res = {}
    base_ho = {s: heldout([v['base'] for v in V], f) for s, f in splits.items()}
    side_ho = {s: heldout([v['side_bm7'] for v in V], f) for s, f in splits.items()}
    side_in = cost_bits(sb, quantize(sb))
    for k in ['base', 'bm7', 'half', 'bm7_half', 'pos32', 'parity', 'bm7_pos32', 'pm2', 'p2b', 'bm7_p2b',
              'bm7_p2b_half', 'bm7_pm2_half', 'z_base', 'z_bm7_p2b', 'z_bm7_half']:
        t = T[k]
        si = side_in if k in NEEDS_SIDE else 0.0
        ins = cost_bits(t, quantize(t)) + si
        r = {'rows': int(t.shape[1]), 'rows_used_max': int((t.sum(-1) > 0).sum(-1).max()),
             'ideal_in_bit_per_w': (ideal_bits(t) + (ideal_bits(sb) if k in NEEDS_SIDE else 0.0)) / W,
             'in_bit_per_w': (ins + flush) / W,
             'in_delta_pct': 100 * (ins - qb) / (qb + flush)}
        for s, f in splits.items():
            h = heldout([v[k] for v in V], f) + (side_ho[s] if k in NEEDS_SIDE else 0.0)
            r[f'ho_{s}_bit_per_w'] = (h + flush) / W
            r[f'ho_{s}_delta_pct'] = 100 * (h - base_ho[s]) / (base_ho[s] + flush)
        res[k] = r
    out['variants'] = res
    # per-expert spread under the in-sample shared model (sampling error of the 160-expert mean)
    qf = quantize(base)
    pe = np.array([cost_bits(v['base'], qf) for v in V]) / (3 * W_PER_MAT) + 32 / 2048
    out['per_expert_bit_per_w'] = {'mean': float(pe.mean()), 'std': float(pe.std(ddof=1)), 'se_mean': float(pe.std(ddof=1) / np.sqrt(n)),
                                   'min': float(pe.min()), 'max': float(pe.max())}
    s_o0 = sum(e['s_o0'] for e in ex)
    out['scales'] = {'ideal_bit_per_scale': ideal_bits(s_o0) / S, 'quant14_bit_per_scale': cost_bits(s_o0, quantize(s_o0)) / S,
                     'quant14_plus_4B_flush_per_512': (cost_bits(s_o0, quantize(s_o0)) + sflush) / S}
    pes = np.array([cost_bits(e['s_o0'], quantize(s_o0)) for e in ex]) / (3 * W_PER_MAT // 32) + 32 / 512
    out['per_expert_scale_bits'] = {'mean': float(pes.mean()), 'std': float(pes.std(ddof=1)), 'se_mean': float(pes.std(ddof=1) / np.sqrt(n))}
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
