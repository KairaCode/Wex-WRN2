"""Own training counts for every expert of the sample, cross-checked against <RESEARCH-PC>\\v41-counts.

Usage: python -I count_pass.py SAMPLE_DIR NPZ_DIR OUT_DIR
Per expert (OUT_DIR/L.._E...npz): base [3,567,16], a [3,2268,16] (row = base*4 + bm7*2 + half),
side [3,7,2] (srel x bm7), s [3,256], neg0 [3], med [3].
OUT_DIR/accum.npz: A_even/A_all [3,25,2268,16] (dbin x a-row x symbol), SA_even/SA_all [3,25,2] (dbin x bm7).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wnr_common import load_manifest, load_expert, block_features, NCTX, NACTX, NDB


def main():
    sample_dir, npz_dir, out_dir = sys.argv[1:4]
    man = load_manifest(sample_dir)
    os.makedirs(out_dir, exist_ok=True)
    A = {'even': np.zeros((3, NDB * NACTX * 16), np.int64), 'all': np.zeros((3, NDB * NACTX * 16), np.int64)}
    SA = {'even': np.zeros((3, NDB * 2), np.int64), 'all': np.zeros((3, NDB * 2), np.int64)}
    checks = []
    t0 = time.time()
    for i, e in enumerate(man['experts']):
        raw, mats = load_expert(sample_dir, e)
        base = np.zeros((3, NCTX * 16), np.int64)
        a = np.zeros((3, NACTX * 16), np.int64)
        side = np.zeros((3, 14), np.int64)
        s = np.zeros((3, 256), np.int64)
        neg0 = np.zeros(3, np.int64)
        meds = np.zeros(3, np.int64)
        for m, mt in enumerate(mats):
            f = block_features(mt['sym'], mt['sc'])
            blk = f['blk'].astype(np.int64)
            base[m] = np.bincount((f['base'].astype(np.int64) * 16 + blk).ravel(), minlength=NCTX * 16)
            ai = f['actx'].astype(np.int64) * 16 + blk
            a[m] = np.bincount(ai.ravel(), minlength=NACTX * 16)
            side[m] = np.bincount(f['srel'].astype(np.int64) * 2 + f['bm7'], minlength=14)
            s[m] = np.bincount(mt['sc'], minlength=256)
            neg0[m] = int((mt['sym'] == 8).sum())
            meds[m] = f['med']
            db = f['dbin'].astype(np.int64)
            acc = np.bincount((db[:, None] * (NACTX * 16) + ai).ravel(), minlength=NDB * NACTX * 16)
            sacc = np.bincount(db * 2 + f['bm7'], minlength=NDB * 2)
            A['all'][m] += acc
            SA['all'][m] += sacc
            if i % 2 == 0:
                A['even'][m] += acc
                SA['even'][m] += sacc
        name = e['file'].replace('.bin', '.npz')
        np.savez_compressed(os.path.join(out_dir, name), base=base.reshape(3, NCTX, 16), a=a.reshape(3, NACTX, 16),
                            side=side.reshape(3, 7, 2), s=s, neg0=neg0, med=meds)
        # cross-check against the team-1 counts (mxcount.py)
        z = np.load(os.path.join(npz_dir, name))
        wb = z['w_bmpos'].reshape(3, NCTX, 8, 4, 16)
        ref_a = np.zeros((3, NCTX, 2, 2, 16), np.int64)
        ref_a[:, :, 0, 0] = wb[:, :, :7, 0:2].sum((2, 3))
        ref_a[:, :, 0, 1] = wb[:, :, :7, 2:4].sum((2, 3))
        ref_a[:, :, 1, 0] = wb[:, :, 7, 0:2].sum(2)
        ref_a[:, :, 1, 1] = wb[:, :, 7, 2:4].sum(2)
        bs = z['bm_side']
        ref_side = np.stack([bs[:, :, :7].sum(2), bs[:, :, 7]], 2)
        ok = {'base': bool((z['w_base'] == base.reshape(3, NCTX, 16)).all()),
              'a': bool((ref_a.reshape(3, NACTX, 16) == a.reshape(3, NACTX, 16)).all()),
              'side': bool((ref_side == side.reshape(3, 7, 2)).all()),
              's': bool((z['s_o0'] == s).all()),
              'meta_med': bool((z['meta'][:, 2] == meds).all()), 'meta_neg0': bool((z['meta'][:, 3] == neg0).all())}
        checks.append({'file': e['file'], **ok})
        if not all(ok.values()):
            print('MISMATCH', e['file'], ok, flush=True)
        if i % 10 == 0:
            print(i, e['file'], f'{time.time() - t0:.1f}s', ok, flush=True)
    np.savez_compressed(os.path.join(out_dir, 'accum.npz'),
                        A_even=A['even'].reshape(3, NDB, NACTX, 16), A_all=A['all'].reshape(3, NDB, NACTX, 16),
                        SA_even=SA['even'].reshape(3, NDB, 2), SA_all=SA['all'].reshape(3, NDB, 2))
    summ = {k: sum(c[k] for c in checks) for k in ['base', 'a', 'side', 's', 'meta_med', 'meta_neg0']}
    json.dump({'experts': len(checks), 'all_equal_counts': summ, 'seconds': time.time() - t0, 'rows': checks},
              open(os.path.join(out_dir, 'count_check.json'), 'w'), indent=1)
    print('done', summ, f'{time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
