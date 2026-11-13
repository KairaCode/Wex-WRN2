"""Train WNR2 / ceil-form model sets from count_ceil.py output (team4 / ceil-ref; adapted from team 3 train_models.py).

Usage: python -I train_ceil.py SAMPLE_DIR COUNTS_DIR OUT_DIR TAGS
  TAGS: comma list of count_ceil.py tags whose accum_TAG.npz are summed (training experts = all per-expert npz present
  for the even manifest indices). Writes OUT_DIR/even.npz with keys '{fam}_a{a}_K{K}_c{c}' + _wf/_sf/_scf/_etab/_assign
  and OUT_DIR/train_summary.json. Checks that the WNR2 model equals team 3's even.npz a0_K0_c0 (same training split).
K-means (B): units = (expert, matrix), models shared across matrix types, init by layer bands (layer*K//40), as team 3.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wnr_common import quantize, load_manifest, NCTX, NACTX, NDB, DBC, P, NSIDE, NSUB

COMBOS = [('wnr2', 0, 0, 0), ('f', 0, 0, 0), ('fg', 0, 0, 0), ('fg', 1, 0, 0), ('fg', 1, 2, 0), ('fg', 0, 0, 1),
          ('fg', 1, 2, 1)]
T3_EVEN = r'<RESEARCH-PC>\team3\ref-impl\models\even.npz'


def merge_c(x):
    x = x.copy()
    x[..., 0] += x[..., 8]
    x[..., 8] = 0
    return x


def cost(counts, freq):
    return float((counts.astype(np.float64) * -np.log2(freq / P)).sum())


def main():
    sample_dir, counts_dir, out_dir, tags = sys.argv[1:5]
    man = load_manifest(sample_dir)
    sel = []
    for i, e in enumerate(man['experts']):
        if i % 2:
            continue
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        sel.append({'i': i, 'layer': e['layer'], 'expert': e['expert'], **{k: z[k] for k in z.files}})
    assert len(sel) == 80
    acc = {}
    for tg in tags.split(','):
        z = np.load(os.path.join(counts_dir, f'accum_{tg}.npz'))
        for k in z.files:
            acc[k] = acc.get(k, 0) + z[k]
    os.makedirs(out_dir, exist_ok=True)
    out, summary = {}, {}
    srel_of_db = np.clip(np.arange(NDB) - DBC, -3, 3) + 3
    nW = len(sel) * 3 * 2304 * 5120
    for fam, a, K, c in COMBOS:
        key = f'{fam}_a{a}_K{K}_c{c}'
        t0 = time.time()
        ca = sum(e[fam + '_a'] for e in sel)            # [3, 3402, 16]
        cs = sum(e[fam + '_side'] for e in sel)         # [3, 7, 3]
        scf = quantize(sum(e[fam + '_s'] for e in sel))
        if c:
            ca = merge_c(ca)
        cb = ca.reshape(3, NCTX, NSUB, 16).sum(2)
        if a:
            wf_t, sf_t = quantize(ca), quantize(cs)
        else:
            wf_t, sf_t = quantize(cb), None
        A = acc['A_' + fam]                               # [3, 25, 3402, 16]
        SA = acc['SA_' + fam]                             # [3, 25, 3]
        Ac = merge_c(A) if c else A
        etab = np.zeros((3, NDB))
        nblk = SA.sum(2)
        for m in range(3):
            if a:
                bits = (Ac[m].astype(np.float64) * -np.log2(wf_t[m] / P)[None]).sum((1, 2))
                bits += (SA[m].astype(np.float64) * -np.log2(sf_t[m][srel_of_db] / P)).sum(1)
            else:
                Ab = Ac[m].reshape(NDB, NCTX, NSUB, 16).sum(2)
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
            sf = None if sf_t is None else sf_t.reshape(-1, NSIDE)
            tb = cost(ca if a else cb, wf_t) + (cost(cs, sf_t) if a else 0.0)
            tside = cost(cs, sf_t) if a else 0.0
            assign = np.zeros(0, np.int64)
            used_rows = int(((ca if a else cb).reshape(-1, 16).sum(1) > 0).sum())
        else:
            assert a
            units = [(e, m) for e in sel for m in range(3)]
            U = np.stack([(merge_c(e[fam + '_a'][m]) if c else e[fam + '_a'][m]).reshape(-1) for e, m in units]).astype(np.float64)
            US = np.stack([e[fam + '_side'][m].reshape(-1) for e, m in units]).astype(np.float64)
            lay = np.array([e['layer'] for e, m in units])
            assign = (lay * K // 40).astype(np.int64)
            for it in range(40):
                wfk = np.stack([quantize(U[assign == k].sum(0).reshape(NACTX, 16).astype(np.int64)) for k in range(K)])
                sfk = np.stack([quantize(US[assign == k].sum(0).reshape(7, NSIDE).astype(np.int64)) for k in range(K)])
                Cst = U @ (-np.log2(wfk.reshape(K, -1) / P)).T + US @ (-np.log2(sfk.reshape(K, -1) / P)).T
                new = Cst.argmin(1)
                if (new == assign).all():
                    break
                assign = new
            wfk = np.stack([quantize(U[assign == k].sum(0).reshape(NACTX, 16).astype(np.int64)) for k in range(K)])
            sfk = np.stack([quantize(US[assign == k].sum(0).reshape(7, NSIDE).astype(np.int64)) for k in range(K)])
            Cst = U @ (-np.log2(wfk.reshape(K, -1) / P)).T + US @ (-np.log2(sfk.reshape(K, -1) / P)).T
            tb = float(Cst[np.arange(len(units)), assign].sum())
            tside = float((US @ (-np.log2(sfk.reshape(K, -1) / P)).T)[np.arange(len(units)), assign].sum())
            wf = wfk.reshape(-1, 16)
            sf = sfk.reshape(-1, NSIDE)
            used_rows = int(sum((U[assign == k].sum(0).reshape(NACTX, 16).sum(1) > 0).sum() for k in range(K)))
            info.update({'iterations': it + 1, 'cluster_sizes': np.bincount(assign, minlength=K).tolist(),
                         'cluster_by_matrix': [np.bincount(assign[np.arange(len(units)) % 3 == m], minlength=K).tolist() for m in range(3)]})
        out[key + '_wf'] = wf
        if sf is not None:
            out[key + '_sf'] = sf
        out[key + '_scf'] = scf
        out[key + '_etab'] = etab_q
        out[key + '_assign'] = assign
        rows = wf.shape[0]
        info.update({'train_payload_bits_per_w (in-sample on the 80 training experts, no flush, incl side)': tb / nW,
                     'train_side_bits_per_w': tside / nW,
                     'train_scale_bits_per_scale (order-0, in-sample)': cost(sum(e[fam + '_s'] for e in sel), scf) / (nW / 32),
                     'rows_stored': int(rows), 'rows_with_training_counts': used_rows,
                     'weight_cdf_bytes_stored (17 u16 per row)': int(rows * 34),
                     'weight_cdf_bytes_compacted (used rows only)': int(used_rows * 34),
                     'side_cdf_bytes': 0 if sf is None else int(sf.shape[0] * (NSIDE + 1) * 2),
                     'scale_cdf_bytes': 3 * 257 * 2, 'etab_bytes': 3 * NDB * 4, 'seconds': time.time() - t0,
                     'etab_bits_per_block': np.round(etab, 2).tolist()})
        summary[key] = info
        print(key, json.dumps({k: v for k, v in info.items() if k != 'etab_bits_per_block'}), flush=True)
    # check WNR2 model vs team 3
    t3 = np.load(T3_EVEN)
    summary['check_wnr2_equals_team3_even_a0_K0_c0'] = {
        'wf': bool((t3['a0_K0_c0_wf'] == out['wnr2_a0_K0_c0_wf']).all()),
        'scf': bool((t3['a0_K0_c0_scf'] == out['wnr2_a0_K0_c0_scf']).all())}
    print(summary['check_wnr2_equals_team3_even_a0_K0_c0'])
    np.savez_compressed(os.path.join(out_dir, 'even.npz'), **out)
    json.dump(summary, open(os.path.join(out_dir, 'train_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
