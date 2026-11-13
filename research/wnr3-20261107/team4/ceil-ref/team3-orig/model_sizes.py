"""Model blob sizes (as serialised by wnr_common.Tables) and compacted sizes (structurally reachable rows only).
Usage: python -I model_sizes.py MODELS_DIR"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wnr_common as W


def reachable_a():
    """Rows (base*4 + bm7*2 + half) reachable by the A context rule, per model."""
    ok = np.zeros((W.NCTX, 2, 2), bool)
    for sr in range(7):
        for bm in range(2):
            ok[(sr * 9 + 8) * 9 + 8, bm, 0] = True          # block start
            top = 7 if bm else 6
            for mx in range(top + 1):
                for pm in range(mx + 1):
                    for h in range(2):
                        ok[(sr * 9 + pm) * 9 + mx, bm, h] = True
    return ok.reshape(-1)


def main():
    d = sys.argv[1]
    out = {}
    ra = reachable_a()
    for split in ('even', 'all'):
        Z = np.load(os.path.join(d, f'{split}.npz'))
        for key in sorted({k.rsplit('_', 1)[0] for k in Z.files}):
            a = key.startswith('a1')
            K = int(key.split('_')[1][1:])
            sf = Z[key + '_sf'] if key + '_sf' in Z.files else None
            t = W.Tables(Z[key + '_wf'], Z[key + '_scf'], sf, Z[key + '_etab'], K=K, nctx=W.NACTX if a else W.NCTX)
            nmod = K if K else 3
            reach = int(ra.sum()) if a else 259
            row_b = 17 * 2
            comp = 32 + nmod * reach * row_b + (nmod * 7 * 3 * 2 if a else 0) + 3 * 257 * 2 + (3 * W.NDB * 4)
            out[f'{split}/{key}'] = {'blob_bytes_full_rows': len(t.blob), 'model_id': t.model_id, 'models': nmod,
                                     'reachable_rows_per_model': reach,
                                     'compacted_blob_bytes': comp,
                                     'shared_mem_per_matrix_bytes (one model, reachable rows, 17 u16)': reach * row_b}
        if split == 'even':
            t2 = W.Tables(Z['a0_K0_c0_wf'], Z['a0_K0_c0_scf'])
            out['even/WNM2 (exact WNR2 blob)'] = {'blob_bytes': len(t2.blob), 'model_id': t2.model_id}
    json.dump(out, open(os.path.join(d, 'model_sizes.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
