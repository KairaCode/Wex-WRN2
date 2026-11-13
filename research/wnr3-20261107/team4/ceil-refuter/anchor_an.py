"""ceil-refuter, resumed run: is the ceil-form gain partly an srel-anchor effect? Held-out costs with fixed anchor rules.

Usage: python -I -B anchor_an.py OUT_DIR OUT_JSON
Reads OUT_DIR/pe/*.npz + OUT_DIR/agg_sample.{npz,json} + agg_extra.{npz,json} (anchor_count.py).
Model = one 567-context 14-bit table per matrix type (WNR2 today) + order-0 scales per matrix type; quantizer and
cost = refute_an.quantize / cost_bits (copy of wex_nr2_quantize rule). Flush 4 B per 2048-weight stream added.
Splits: P2 train sample groups {0,2} (even manifest index) -> test {1,3}; P3 test only group 1 (team 3's 40);
P4 train all 160 sample -> test 70 extra routed. MTP: own models, leave-one-MTP-layer-out.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refute_an import quantize, cost_bits

S0, NS = 104, 24
W_PER_REC = 3 * 2304 * 5120
S_PER_REC = W_PER_REC // 32
RAW = 18_800_640
FLUSH = 32 / 2048


def srel_map(a):
    """[7, 24] one-hot: abs bin j (s = 104 + j) -> srel index clip(s - a, -3, 3) + 3."""
    M = np.zeros((7, NS), np.int64)
    s = S0 + np.arange(NS)
    M[np.clip(s - a, -3, 3) + 3, np.arange(NS)] = 1
    return M


MAPS = {a: srel_map(a) for a in range(100, 135)}


def ctx_counts(A, a):
    """A int [24, 81, 16] abs-scale counts of one matrix -> [567, 16] srel counts for anchor a."""
    return np.einsum('kj,jcs->kcs', MAPS[a], A.astype(np.int64)).reshape(567, 16)


def load(out_dir):
    info = {}
    for g in ('sample', 'extra'):
        info.update(json.load(open(os.path.join(out_dir, f'agg_{g}.json'))))
    pe = {}
    for f in sorted(info):
        z = np.load(os.path.join(out_dir, 'pe', f.replace('.bin', '.npz')))
        pe[f] = {k: z[k] for k in z.files}
    agg = None
    for g in ('sample', 'extra'):
        z = np.load(os.path.join(out_dir, f'agg_{g}.npz'))
        if agg is None:
            agg = {k: z[k].astype(np.int64) for k in z.files}
        else:
            for k in z.files:
                agg[k] += z[k]
    return info, pe, agg


def anchors_for(info, f, form, rule):
    out = []
    for m in range(3):
        mi = info[f]['m'][m]
        if rule.startswith('fixed'):
            out.append(int(rule[5:]))
        elif rule.endswith('+1o'):  # ceil form, anchor from the ORIGINAL scales' rule + 1
            out.append(mi['anchors_o'][rule[:-3]] + 1)
        elif rule.endswith('_o'):
            out.append(mi['anchors_o'][rule[:-2]])
        else:
            out.append(mi[f'anchors_{form}'][rule])
    return out


def train(pe, files, key, anc):
    c = np.zeros((3, 567, 16), np.int64)
    for f in files:
        for m in range(3):
            c[m] += ctx_counts(pe[f][key][m], anc[f][m])
    return quantize(c)


def bits_matrix(pe, f, key, m, a, logq):
    return float((ctx_counts(pe[f][key][m], a) * logq[m]).sum())


def search(pe, train_files, test_files, key, anc0, iters=3, span=2):
    """Encoder-side anchor search (the anchor is stored per record, so any choice is decodable): alternate between
    training on the train experts and picking, per matrix, the anchor in anc0 +- span that minimises the coded bits."""
    anc = {f: list(anc0[f]) for f in train_files + test_files}
    for it in range(iters):
        q = train(pe, train_files, key, anc)
        logq = -np.log2(q / 16384.0)
        for f in train_files + test_files:
            for m in range(3):
                a0 = anc0[f][m]
                best = min(range(a0 - span, a0 + span + 1), key=lambda a: bits_matrix(pe, f, key, m, a, logq))
                anc[f][m] = best
    q = train(pe, train_files, key, anc)
    return q, anc


def cost_set(pe, files, key, anc, q, skey, sq):
    logq = -np.log2(q / 16384.0)
    wb = sum(bits_matrix(pe, f, key, m, anc[f][m], logq) for f in files for m in range(3))
    sb = sum(cost_bits(pe[f][skey][m], sq[m]) for f in files for m in range(3))
    n = len(files)
    bpw = wb / (n * W_PER_REC) + FLUSH
    bps = sb / (n * S_PER_REC)
    return {'bpw': bpw, 'bps': bps, 'rec_bytes': bpw * W_PER_REC / 8 + bps * S_PER_REC / 8}


def scale_model(pe, files, skey):
    return quantize(sum(pe[f][skey].astype(np.int64) for f in files))


def per_expert(pe, f, key, anc, q, skey, sq):
    return cost_set(pe, [f], key, anc, q, skey, sq)['rec_bytes']


def main():
    out_dir, out_json = sys.argv[1:3]
    info, pe, agg = load(out_dir)
    files = sorted(info)
    grp = {f: info[f]['group'] for f in files}
    G = lambda gs: [f for f in files if grp[f] in gs]
    res = {'n_experts': len(files), 'clip_total': sum(info[f]['clip_o'] + info[f]['clip_c'] for f in files)}
    # anchor statistics
    st = {}
    for form in ('o', 'c'):
        for rule in ('med', 'pair', 'mode'):
            v = [info[f]['m'][m][f'anchors_{form}'][rule] for f in files if grp[f] <= 4 for m in range(3)]
            st[f'{form}_{rule}'] = {str(a): int(c) for a, c in zip(*np.unique(v, return_counts=True))}
    d = [info[f]['m'][m]['anchors_c']['med'] - info[f]['m'][m]['anchors_o']['med'] for f in files if grp[f] <= 4 for m in range(3)]
    st['ceil_med_minus_orig_med'] = {str(a): int(c) for a, c in zip(*np.unique(d, return_counts=True))}
    res['anchor_hist_routed_matrices'] = st
    splits = {'P2_train_even80_test_odd80': (G((0, 2)), G((1, 3))),
              'P3_train_even80_test_odd40': (G((0, 2)), G((1,))),
              'P4_train_sample160_test_extra70': (G((0, 1, 2, 3)), G((4,)))}
    rules_o = ['med', 'pair', 'mode', 'fixed120', 'fixed121']
    rules_c = ['med', 'pair', 'mode', 'fixed120', 'fixed121', 'fixed122', 'med+1o', 'pair+1o']
    tabs = {}
    for sp, (tr, te) in splits.items():
        rows = {}
        sq_o = scale_model(pe, tr, 'so'); sq_c = scale_model(pe, tr, 'sc')
        for form, key, rules, skey, sq in [('o', 'ao', rules_o, 'so', sq_o), ('c', 'ac', rules_c, 'sc', sq_c)]:
            for rule in rules + ['search']:
                if rule == 'search':
                    base_rule = 'pair' if form == 'o' else 'med'
                    anc0 = {f: anchors_for(info, f, form, base_rule) for f in tr + te}
                    q, anc = search(pe, tr, te, key, anc0)
                    sh = [anc[f][m] - anc0[f][m] for f in te for m in range(3)]
                    extra = {'test_anchor_shift_hist': {str(a): int(c) for a, c in zip(*np.unique(sh, return_counts=True))}}
                else:
                    anc = {f: anchors_for(info, f, form, rule) for f in tr + te}
                    q = train(pe, tr, key, anc)
                    extra = {}
                r = cost_set(pe, te, key, anc, q, skey, sq)
                r.update(extra)
                rows[f'{form}_{rule}'] = r
        b0 = rows['o_med']
        for k, r in rows.items():
            r['d_w_pct_vs_today'] = 100 * (r['bpw'] - b0['bpw']) / b0['bpw']
            r['d_rec_pct_raw_vs_today'] = 100 * (r['rec_bytes'] - b0['rec_bytes']) / RAW
        # class-side alternatives (V6b family) from the group sums, anchors med / pair
        gi = {'P2_train_even80_test_odd80': ((0, 2), (1, 3)), 'P3_train_even80_test_odd40': ((0, 2), (1,)),
              'P4_train_sample160_test_extra70': ((0, 1, 2, 3), (4,))}[sp]
        for rule in ('med', 'pair'):
            A = agg[f'o_clsh_pm2b_{rule}']
            a_tr = A[list(gi[0])].sum(0); a_te = A[list(gi[1])].sum(0)
            soc_tr = sum(pe[f]['soc'].astype(np.int64) for f in tr); soc_te = sum(pe[f]['soc'].astype(np.int64) for f in te)
            ncls_tr = soc_tr.sum(2); ncls_te = soc_te.sum(2)  # [3 m, 3 cls]
            side = cost_bits(ncls_te, quantize(ncls_tr))  # class order-0 per matrix type
            sbits = cost_bits(soc_te, quantize(soc_tr))  # scale | class
            nw = len(te) * W_PER_REC
            for vname, red in [('V6b_clsh_pm2b', lambda x: x), ('V4b_clsh', lambda x: x.reshape(3, 3402, 6, 16).sum(2)),
                               ('V3b_cls', lambda x: x.reshape(3, 3, 2, 567, 6, 16).sum((2, 4)).reshape(3, 1701, 16))]:
                wb = cost_bits(red(a_te), quantize(red(a_tr)))
                bpw = (wb + side) / nw + FLUSH
                bps = sbits / (nw / 32)
                rb = bpw * W_PER_REC / 8 + bps * S_PER_REC / 8
                rows[f'{vname}_{rule}'] = {'bpw': bpw, 'bps': bps, 'rec_bytes': rb,
                                           'd_w_pct_vs_today': 100 * (bpw - b0['bpw']) / b0['bpw'],
                                           'd_rec_pct_raw_vs_today': 100 * (rb - b0['rec_bytes']) / RAW}
        non = {k: v for k, v in rows.items() if not k.startswith('c_')}
        cei = {k: v for k, v in rows.items() if k.startswith('c_')}
        bn = min(non, key=lambda k: non[k]['rec_bytes']); bc = min(cei, key=lambda k: cei[k]['rec_bytes'])
        bo = min([k for k in rows if k.startswith('o_')], key=lambda k: rows[k]['rec_bytes'])
        summ = {'best_non_ceil': bn, 'best_ceil': bc, 'best_orig_base_anchor_rule': bo,
                'ceil_on_top_of_best_non_ceil_rec_pct_raw': 100 * (cei[bc]['rec_bytes'] - non[bn]['rec_bytes']) / RAW,
                'ceil_on_top_of_best_non_ceil_w_pct': 100 * (cei[bc]['bpw'] - non[bn]['bpw']) / non[bn]['bpw'],
                'ceil_best_vs_orig_best_anchor_base_rec_pct_raw': 100 * (cei[bc]['rec_bytes'] - rows[bo]['rec_bytes']) / RAW,
                'ceil_best_vs_orig_best_anchor_base_w_pct': 100 * (cei[bc]['bpw'] - rows[bo]['bpw']) / rows[bo]['bpw'],
                'anchor_fix_alone_rec_pct_raw': 100 * (rows[bo]['rec_bytes'] - b0['rec_bytes']) / RAW}
        tabs[sp] = {'summary': summ, 'rows': rows}
        print(sp, json.dumps(summ), flush=True)
        for k, r in rows.items():
            print(f"   {k:24s} bpw {r['bpw']:.4f} bps {r['bps']:.4f} dW {r['d_w_pct_vs_today']:+.3f}% dRec {r['d_rec_pct_raw_vs_today']:+.3f}%", flush=True)
    res['heldout'] = tabs
    # per-expert distribution of the ceil gain, two-way on the sample (+ extra with the all-sample model)
    pe_rows = []
    combos = [('today: o_med vs c_med', ('o', 'ao', 'med', 'so'), ('c', 'ac', 'med', 'sc')),
              ('fixed anchors: o_pair vs c_fixed121', ('o', 'ao', 'pair', 'so'), ('c', 'ac', 'fixed121', 'sc')),
              ('o_pair vs c_med+1o', ('o', 'ao', 'pair', 'so'), ('c', 'ac', 'pair+1o', 'sc'))]
    dist = {}
    for cname, A, B in combos:
        dl = []
        for tr, te in [(G((1, 3)), G((0, 2))), (G((0, 2)), G((1, 3))), (G((0, 1, 2, 3)), G((4,)))]:
            mods = {}
            for form, key, rule, skey in (A, B):
                anc = {f: anchors_for(info, f, form, rule) for f in tr + te}
                mods[form] = (key, anc, train(pe, tr, key, anc), skey, scale_model(pe, tr, skey))
            for f in te:
                ro = per_expert(pe, f, *mods['o']); rcc = per_expert(pe, f, *mods['c'])
                dl.append((100 * (rcc - ro) / RAW, f))
        v = np.array([x[0] for x in dl])
        dist[cname] = {'n': len(v), 'mean': float(v.mean()), 'median': float(np.median(v)), 'std': float(v.std()),
                       'min': float(v.min()), 'max': float(v.max()), 'p5': float(np.percentile(v, 5)), 'p95': float(np.percentile(v, 95)),
                       'n_worse_than_orig': int((v > 0).sum()), 'n_gain_below_1pct': int((v > -1).sum()),
                       'smallest_gain5': sorted(dl)[-5:], 'largest_gain5': sorted(dl)[:5]}
        print(cname, json.dumps({k: x for k, x in dist[cname].items() if 'gain5' not in k}), flush=True)
    res['per_expert_gain_distribution'] = dist
    # MTP with its own models (leave-one-MTP-layer-out), and with the routed (all sample) model
    mtp = [f for f in files if grp[f] == 5]
    mres = {}
    for rule_o, rule_c in [('med', 'med'), ('pair', 'fixed121'), ('pair', 'pair+1o')]:
        tot = {'o': 0.0, 'c': 0.0}
        rows = []
        for L in sorted({info[f]['layer'] for f in mtp}):
            te = [f for f in mtp if info[f]['layer'] == L]; tr = [f for f in mtp if info[f]['layer'] != L]
            r = {}
            for form, key, rule, skey in [('o', 'ao', rule_o, 'so'), ('c', 'ac', rule_c, 'sc')]:
                anc = {f: anchors_for(info, f, form, rule) for f in tr + te}
                c = cost_set(pe, te, key, anc, train(pe, tr, key, anc), skey, scale_model(pe, tr, skey))
                r[form] = c; tot[form] += c['rec_bytes'] * len(te)
            rows.append({'test_mtp_layer': L, 'orig_ratio': r['o']['rec_bytes'] / RAW, 'ceil_ratio': r['c']['rec_bytes'] / RAW,
                         'orig_bpw': r['o']['bpw'], 'ceil_bpw': r['c']['bpw'], 'orig_bps': r['o']['bps'], 'ceil_bps': r['c']['bps'],
                         'd_rec_pct_raw': 100 * (r['c']['rec_bytes'] - r['o']['rec_bytes']) / RAW})
        mres[f'{rule_o}/{rule_c}'] = {'per_layer': rows, 'mean_d_rec_pct_raw': 100 * (tot['c'] - tot['o']) / len(mtp) / RAW,
                                      'mean_orig_ratio': tot['o'] / len(mtp) / RAW, 'mean_ceil_ratio': tot['c'] / len(mtp) / RAW}
        print('MTP own', rule_o, rule_c, json.dumps(mres[f'{rule_o}/{rule_c}']), flush=True)
    tr = G((0, 1, 2, 3))
    r = {}
    for form, key, rule, skey in [('o', 'ao', 'med', 'so'), ('c', 'ac', 'med', 'sc')]:
        anc = {f: anchors_for(info, f, form, rule) for f in tr + mtp}
        r[form] = cost_set(pe, mtp, key, anc, train(pe, tr, key, anc), skey, scale_model(pe, tr, skey))
    mres['routed_model_on_mtp_med'] = {'orig_ratio': r['o']['rec_bytes'] / RAW, 'ceil_ratio': r['c']['rec_bytes'] / RAW,
                                       'orig_bpw': r['o']['bpw'], 'ceil_bpw': r['c']['bpw'], 'orig_bps': r['o']['bps'], 'ceil_bps': r['c']['bps']}
    print('MTP routed model', json.dumps(mres['routed_model_on_mtp_med']), flush=True)
    res['mtp'] = mres
    json.dump(res, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
