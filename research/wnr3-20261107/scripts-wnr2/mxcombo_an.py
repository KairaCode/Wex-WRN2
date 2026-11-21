"""Analysis of pass 1b (combined, shared-memory-budgeted contexts) + held-out clustered models.

Usage: python -I mxcombo_an.py SAMPLE_DIR COUNTS_DIR COMBO_DIR OUT_JSON
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxanalyze import quantize, cost_bits, P, W_PER_REC

FLUSH_PER_REC = 32 * (5760 + 5760 + 5760)  # 2048-weight streams: 2304*5120/2048 = 5120*2304/2048 = 5760


def load(sample_dir, d):
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    out = []
    for i, e in enumerate(man['experts']):
        z = np.load(os.path.join(d, e['file'].replace('.bin', '.npz')))
        out.append({'layer': e['layer'], 'split': i % 2, **{k: z[k] for k in z.files}})
    return out


def S(ex, k, f=lambda e: True):
    return sum(e[k].astype(np.int64) for e in ex if f(e))


def ev(ex, k):
    a = S(ex, k, lambda e: e['split'] == 0); b = S(ex, k, lambda e: e['split'] == 1); c = a + b
    return cost_bits(c, quantize(c)), cost_bits(b, quantize(a)) + cost_bits(a, quantize(b))


def main():
    sample_dir, counts_dir, combo_dir, out_json = sys.argv[1:5]
    base = load(sample_dir, counts_dir)
    co = load(sample_dir, combo_dir)
    n = len(co); W = n * W_PER_REC; fl = n * FLUSH_PER_REC
    bpw = lambda bits: (bits + fl) / W
    bi, bh = ev(base, 'w_base')
    out = {'base': {'in': bpw(bi), 'ho': bpw(bh)}}
    si, sh = ev(co, 'bm2_side')
    for k, side in [('c_bm2', True), ('c_p2b', False), ('c_bm2p2b', True), ('c_bm2h', True), ('c_bm2p2bh', True), ('c_z', True)]:
        i, h = ev(co, k)
        if side:
            i += si; h += sh
        used = int((S(co, k).sum(-1) > 0).sum(-1).max())
        out[k] = {'in': bpw(i), 'ho': bpw(h), 'delta_ho_pct': 100 * (h - bh) / (bh + fl),
                  'rows_total': int(S(co, k).shape[1]), 'rows_used_max_over_matrices': used,
                  'table_kb_compacted': used * 32 / 1024}
    out['side_bits_per_w'] = si / W
    z = S(co, 'zeros')
    out['zeros_plus_minus_fraction'] = (z.sum(0) / W).tolist()
    # held-out K clusters on the base context: fit on split A units, assign+cost on split B units, and vice versa
    units = [(e['split'], e['layer'], e['w_base'][m]) for e in base for m in range(3)]
    res = {}
    for K in [1, 2, 4, 8]:
        tot_bits = 0.0
        for tr in (0, 1):
            U = np.stack([u[2] for u in units if u[0] == tr]).astype(np.int64)
            lay = np.array([u[1] for u in units if u[0] == tr])
            assign = (lay * K // 40)
            for _ in range(15):
                Lm = np.stack([-np.log2(quantize(U[assign == k].sum(0) if (assign == k).any() else np.zeros((567, 16), np.int64)) / P) for k in range(K)])
                new = np.einsum('ucs,kcs->uk', U.astype(np.float64), Lm).argmin(1)
                if (new == assign).all():
                    break
                assign = new
            T = np.stack([u[2] for u in units if u[0] != tr]).astype(np.float64)
            tot_bits += np.einsum('ucs,kcs->uk', T, Lm).min(1).sum()  # encoder picks the best model per matrix
        res[K] = bpw(tot_bits)
    out['clusters_heldout_bit_per_w'] = res

    # clusters on top of the richer contexts: per-matrix model choice, held-out, side bits added
    def clusters(key, side_key, rows, Ks=(1, 2, 4)):
        un = [(e['split'], e['layer'], e[key][m], e[side_key][m] if side_key else None) for e in co for m in range(3)]
        r = {}
        for K in Ks:
            tb = 0.0
            for tr in (0, 1):
                U = np.stack([u[2] for u in un if u[0] == tr]).astype(np.int64)
                lay = np.array([u[1] for u in un if u[0] == tr])
                assign = lay * K // 40
                for _ in range(15):
                    Lm = np.stack([-np.log2(quantize(U[assign == k].sum(0) if (assign == k).any() else np.zeros((rows, 16), np.int64)) / P) for k in range(K)])
                    new = np.einsum('ucs,kcs->uk', U.astype(np.float64), Lm).argmin(1)
                    if (new == assign).all():
                        break
                    assign = new
                T = np.stack([u[2] for u in un if u[0] != tr]).astype(np.float64)
                tb += np.einsum('ucs,kcs->uk', T, Lm).min(1).sum()
            r[K] = bpw(tb + sh)  # block-max side bits (held-out cost), same for every K
        return r

    out['clusters_on_c_bm2h_heldout'] = clusters('c_bm2h', None, 2268)
    out['clusters_on_c_z_heldout (value-exact)'] = clusters('c_z', None, 4536)
    json.dump(out, open(out_json, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
