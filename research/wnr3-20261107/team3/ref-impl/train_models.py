"""Train the WNR2 / WNR3-candidate models from count_pass.py output.

Usage: python -I train_models.py SAMPLE_DIR COUNTS_DIR OUT_DIR
Writes OUT_DIR/even.npz (trained on even-indexed experts = held-out models for the odd ones) and OUT_DIR/all.npz
(trained on all 160 = in-sample). Keys per combo 'a{0,1}_K{k}_c{0,1}': _wf, _sf, _scf, _etab, _assign, _trainbits.
K-means (B): units = (expert, matrix) of the training split, models shared across matrix types, init by layer bands
(layer*K//40, as mxanalyze.py), iterate to convergence (max 40), cost = exact code length of weights (+ side bits).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wnr_common import quantize, load_manifest, NCTX, NACTX, NDB, DBC, P

COMBOS = [(0, 0, 0), (1, 0, 0), (1, 2, 0), (1, 4, 0), (0, 0, 1), (1, 4, 1), (1, 2, 1)]


def merge_c(x):
    x = x.copy()
    x[..., 0] += x[..., 8]
    x[..., 8] = 0
    return x


def cost(counts, freq):
    return float((counts.astype(np.float64) * -np.log2(freq / P)).sum())


def main():
    sample_dir, counts_dir, out_dir = sys.argv[1:4]
    man = load_manifest(sample_dir)
    ex = []
    for i, e in enumerate(man['experts']):
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        ex.append({'i': i, 'layer': e['layer'], 'expert': e['expert'], **{k: z[k] for k in z.files}})
    acc = np.load(os.path.join(counts_dir, 'accum.npz'))
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for split in ('even', 'all'):
        sel = [e for e in ex if split == 'all' or e['i'] % 2 == 0]
        out = {}
        A = acc['A_' + split]          # [3, 25, 2268, 16]
        SA = acc['SA_' + split]        # [3, 25, 2]
        scf = quantize(sum(e['s'] for e in sel))
        srel_of_db = np.clip(np.arange(NDB) - DBC, -3, 3) + 3
        for a, K, c in COMBOS:
            key = f'a{a}_K{K}_c{c}'
            t0 = time.time()
            ca = sum(e['a'] for e in sel)                 # [3, 2268, 16]
            cs = sum(e['side'] for e in sel)              # [3, 7, 2]
            if c:
                ca = merge_c(ca)
            cb = ca.reshape(3, NCTX, 4, 16).sum(2)
            # per-type (unclustered) model of these features: also used for the length predictor table
            if a:
                wf_t = quantize(ca)
                sf_t = quantize(cs)
            else:
                wf_t = quantize(cb)
                sf_t = None
            Ac = merge_c(A) if c else A
            etab = np.zeros((3, NDB))
            nblk = SA.sum(2)
            for m in range(3):
                if a:
                    bits = (Ac[m].astype(np.float64) * -np.log2(wf_t[m] / P)[None]).sum((1, 2))
                    bits += (SA[m].astype(np.float64) * -np.log2(sf_t[m][srel_of_db] / P)).sum(1)
                else:
                    Ab = Ac[m].reshape(NDB, NCTX, 4, 16).sum(2)
                    bits = (Ab.astype(np.float64) * -np.log2(wf_t[m] / P)[None]).sum((1, 2))
                with np.errstate(invalid='ignore', divide='ignore'):
                    e_m = bits / nblk[m]
                good = nblk[m] > 0
                idx = np.arange(NDB)
                gi = idx[good]
                near = gi[np.abs(idx[:, None] - gi[None, :]).argmin(1)]
                etab[m] = e_m[near]
            etab_q = np.round(etab * 256).astype(np.int64)
            info = {'K': K}
            if K == 0:
                wf = wf_t.reshape(-1, 16)
                sf = None if sf_t is None else sf_t.reshape(-1, 2)
                tb = cost(ca if a else cb, wf_t) + (cost(cs, sf_t) if a else 0.0)
                assign = np.zeros(0, np.int64)
            else:
                assert a
                units = [(e, m) for e in sel for m in range(3)]
                U = np.stack([(merge_c(e['a'][m]) if c else e['a'][m]).reshape(-1) for e, m in units]).astype(np.float64)
                US = np.stack([e['side'][m].reshape(-1) for e, m in units]).astype(np.float64)
                lay = np.array([e['layer'] for e, m in units])
                assign = (lay * K // 40).astype(np.int64)
                for it in range(40):
                    wfk = np.stack([quantize(U[assign == k].sum(0).reshape(NACTX, 16).astype(np.int64)) for k in range(K)])
                    sfk = np.stack([quantize(US[assign == k].sum(0).reshape(7, 2).astype(np.int64)) for k in range(K)])
                    Cst = U @ (-np.log2(wfk.reshape(K, -1) / P)).T + US @ (-np.log2(sfk.reshape(K, -1) / P)).T
                    new = Cst.argmin(1)
                    if (new == assign).all():
                        break
                    assign = new
                wfk = np.stack([quantize(U[assign == k].sum(0).reshape(NACTX, 16).astype(np.int64)) for k in range(K)])
                sfk = np.stack([quantize(US[assign == k].sum(0).reshape(7, 2).astype(np.int64)) for k in range(K)])
                Cst = U @ (-np.log2(wfk.reshape(K, -1) / P)).T + US @ (-np.log2(sfk.reshape(K, -1) / P)).T
                tb = float(Cst[np.arange(len(units)), assign].sum())
                wf = wfk.reshape(-1, 16)
                sf = sfk.reshape(-1, 2)
                info.update({'iterations': it + 1, 'cluster_sizes': np.bincount(assign, minlength=K).tolist(),
                             'cluster_by_matrix': [np.bincount(assign[np.arange(len(units)) % 3 == m], minlength=K).tolist() for m in range(3)]})
            out[key + '_wf'] = wf
            if sf is not None:
                out[key + '_sf'] = sf
            out[key + '_scf'] = scf
            out[key + '_etab'] = etab_q
            out[key + '_assign'] = assign
            nW = len(sel) * 3 * 2304 * 5120
            # reachable rows: structurally (a: 130 per srel) and observed (non-zero counts in the training split)
            used_rows = [int(((ca if a else cb)[m].sum(1) > 0).sum()) for m in range(3)]
            info.update({'train_payload_bits_per_w (no flush)': tb / nW, 'rows_stored': int(wf.shape[0]),
                         'structurally_reachable_rows_per_model': 910 if a else 259,
                         'rows_with_training_counts_per_matrix_type': used_rows, 'seconds': time.time() - t0,
                         'etab_bits_per_block': np.round(etab, 2).tolist()})
            summary[f'{split}/{key}'] = info
            print(split, key, json.dumps({k: v for k, v in info.items() if k != 'etab_bits_per_block'}), flush=True)
        np.savez_compressed(os.path.join(out_dir, f'{split}.npz'), **out)
    json.dump(summary, open(os.path.join(out_dir, 'train_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
