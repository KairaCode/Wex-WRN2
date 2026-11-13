"""Training counts for the ceil-form families (team4 / ceil-ref), cross-checked against team 3 and mxceil.py counts.

Usage: python -I count_ceil.py SAMPLE_DIR OUT_DIR TAG IDX_LIST
  IDX_LIST: comma list of manifest indices (training experts). Families:
    wnr2 = plain data, median anchor; f = ceil-form, median of transformed scales; fg = ceil-form + densest-pair anchor.
  Canonical zero (C) is derived at training time (contexts do not depend on the sign, so C = merge symbol 8 into 0).
Per expert OUT_DIR/L.._E...npz: {fam}_a [3,3402,16] (row = base*6 + bm3*2 + half), {fam}_side [3,7,3] (srel x bm3),
  {fam}_s [3,256], {fam}_anchor [3], ceil_even/ceil_below6/ceil_overflow [3], neg0 [3].
OUT_DIR/accum_TAG.npz: A_{fam} [3,25,3402,16] (dbin x a-row x symbol), SA_{fam} [3,25,3] (dbin x bm3).
OUT_DIR/check_TAG.json: equality checks vs team 3 counts (wnr2 base/a/side/s/med) and mxceil.py (f: w_ceil, s_ceil).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wnr_common import load_manifest, load_expert, block_features, prepare, NCTX, NACTX, NDB, NSIDE, NSUB

FAMS = {'wnr2': (0, 0), 'f': (1, 0), 'fg': (1, 1)}
T3_COUNTS = r'<RESEARCH-PC>\team3\ref-impl\counts'
CEIL_COUNTS = r'<RESEARCH-PC>\v41-ceil'


def main():
    sample_dir, out_dir, tag, idxs = sys.argv[1:5]
    idxs = [int(x) for x in idxs.split(',')]
    man = load_manifest(sample_dir)
    os.makedirs(out_dir, exist_ok=True)
    A = {k: np.zeros((3, NDB * NACTX * 16), np.int64) for k in FAMS}
    SA = {k: np.zeros((3, NDB * NSIDE), np.int64) for k in FAMS}
    checks = []
    t0 = time.time()
    for n, i in enumerate(idxs):
        e = man['experts'][i]
        raw, mats = load_expert(sample_dir, e)
        out = {'neg0': np.array([int((mt['sym'] == 8).sum()) for mt in mats])}
        for fam, (f, g) in FAMS.items():
            syms, scs, anchors, finfo = prepare(mats, f, 0, g)
            if f:
                assert finfo['applied'], (e['file'], finfo)
                out['ceil_even'] = np.array(finfo['even'])
                out['ceil_below6'] = np.array(finfo['below6'])
                out['ceil_overflow'] = np.array(finfo['overflow'])
            a = np.zeros((3, NACTX * 16), np.int64)
            side = np.zeros((3, 7 * NSIDE), np.int64)
            s = np.zeros((3, 256), np.int64)
            for m in range(3):
                ft = block_features(syms[m], scs[m], anchors[m])
                ai = ft['actx'].astype(np.int64) * 16 + ft['blk']
                a[m] = np.bincount(ai.ravel(), minlength=NACTX * 16)
                side[m] = np.bincount(ft['srel'].astype(np.int64) * NSIDE + ft['bm3'], minlength=7 * NSIDE)
                s[m] = np.bincount(scs[m], minlength=256)[:256]
                db = ft['dbin'].astype(np.int64)
                A[fam][m] += np.bincount((db[:, None] * (NACTX * 16) + ai).ravel(), minlength=NDB * NACTX * 16)
                SA[fam][m] += np.bincount(db * NSIDE + ft['bm3'], minlength=NDB * NSIDE)
            out[fam + '_a'] = a.reshape(3, NACTX, 16)
            out[fam + '_side'] = side.reshape(3, 7, NSIDE)
            out[fam + '_s'] = s
            out[fam + '_anchor'] = np.array(anchors)
        name = e['file'].replace('.bin', '.npz')
        np.savez_compressed(os.path.join(out_dir, name), **out)
        # ---- cross-checks
        ck = {'file': e['file']}
        z = np.load(os.path.join(T3_COUNTS, name))
        a3 = out['wnr2_a'].reshape(3, NCTX, 3, 2, 16)
        ck['t3_base'] = bool((a3.sum((2, 3)) == z['base']).all())
        ref_a = np.stack([a3[:, :, 0] + a3[:, :, 1], a3[:, :, 2]], 2).reshape(3, NCTX * 4, 16)   # team 3: base*4 + bm7*2 + half
        ck['t3_a_bm7'] = bool((ref_a == z['a']).all())
        sd = out['wnr2_side']
        ck['t3_side'] = bool((np.stack([sd[..., 0] + sd[..., 1], sd[..., 2]], 2) == z['side']).all())
        ck['t3_s'] = bool((out['wnr2_s'] == z['s']).all())
        ck['t3_med'] = bool((out['wnr2_anchor'] == z['med']).all())
        zc = np.load(os.path.join(CEIL_COUNTS, name))
        ck['mxceil_w_ceil'] = bool((out['f_a'].reshape(3, NCTX, NSUB, 16).sum(2) == zc['w_ceil']).all())
        ck['mxceil_s_ceil'] = bool((out['f_s'] == zc['s_ceil']).all())
        ck['mxceil_even'] = bool((out['ceil_even'] == zc['stats'][:, 8]).all())
        # bm3 = 0 on ceil-form <=> block max exactly 5 (transformed); count must equal the even blocks
        ck['f_bm3_0_equals_even'] = bool(out['f_side'].sum(1)[:, 0].tolist() == out['ceil_even'].tolist())
        ck['wnr2_bm3_0_is_zero'] = bool(out['wnr2_side'].sum(1)[:, 0].sum() == 0)
        checks.append(ck)
        bad = [k for k, v in ck.items() if v is False]
        if bad:
            print('MISMATCH', e['file'], bad, flush=True)
        print(n, e['file'], f'{time.time() - t0:.1f}s', 'even', out['ceil_even'].tolist(),
              'anchors', {k: out[k + '_anchor'].tolist() for k in FAMS}, flush=True)
    np.savez_compressed(os.path.join(out_dir, f'accum_{tag}.npz'),
                        **{f'A_{k}': A[k].reshape(3, NDB, NACTX, 16) for k in FAMS},
                        **{f'SA_{k}': SA[k].reshape(3, NDB, NSIDE) for k in FAMS})
    keys = [k for k in checks[0] if k != 'file']
    json.dump({'experts': len(checks), 'all_true': {k: sum(c[k] for c in checks) for k in keys},
               'seconds': time.time() - t0, 'rows': checks}, open(os.path.join(out_dir, f'check_{tag}.json'), 'w'), indent=1)
    print('done', {k: sum(c[k] for c in checks) for k in keys}, f'{time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
