"""Compare item files of two fetch_other.py folders tensor by tensor (same tags): bytes, then dequantized values.

Usage: python -I cmp_pairs.py DIR_A DIR_B OUT_JSON
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mxfp

GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def load(d, e):
    raw = np.fromfile(os.path.join(d, e['file']), dtype=np.uint8)
    off, parts = 0, {}
    for t in e['tensors']:
        parts[(t['role'], t['part'])] = (raw[off:off + t['bytes']], t['shape'])
        off += t['bytes']
    return parts


def main():
    da, db, out = sys.argv[1:4]
    ma = {e['tag']: e for e in json.load(open(os.path.join(da, 'manifest.json')))['experts']}
    mb = {e['tag']: e for e in json.load(open(os.path.join(db, 'manifest.json')))['experts']}
    res = []
    for tag in sorted(set(ma) & set(mb)):
        pa, pb = load(da, ma[tag]), load(db, mb[tag])
        row = {'tag': tag}
        for role in sorted({k[0] for k in pa}):
            wa, sa = pa[(role, 'weight')], pa[(role, 'scale')]
            wb_, sb = pb[(role, 'weight')], pb[(role, 'scale')]
            R = wa[1][0]
            A = mxfp.unpack(wa[0].reshape(R, -1)); B = mxfp.unpack(wb_[0].reshape(R, -1))
            SA = sa[0].reshape(R, -1).astype(np.int32); SB = sb[0].reshape(R, -1).astype(np.int32)
            va = GRID[A].reshape(R, -1, 32) * np.exp2(SA - 127)[..., None]
            vb = GRID[B].reshape(R, -1, 32) * np.exp2(SB - 127)[..., None]
            ne_blk = (A.reshape(R, -1, 32) != B.reshape(R, -1, 32)).any(-1) | (SA != SB)
            val_eq = (va == vb)  # -0 == +0 here
            row[role] = {'weight_bytes_identical': bool(np.array_equal(wa[0], wb_[0])),
                         'scale_bytes_identical': bool(np.array_equal(sa[0], sb[0])),
                         'blocks_differing_repr': int(ne_blk.sum()), 'blocks': int(ne_blk.size),
                         'values_equal_pct': float(100 * val_eq.mean()),
                         'scale_diff_hist': {str(k): int(v) for k, v in zip(*np.unique(SB - SA, return_counts=True))},
                         'neg_zero_A': int((A == 8).sum()), 'neg_zero_B': int((B == 8).sum())}
        res.append(row)
        print(json.dumps(row), flush=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
