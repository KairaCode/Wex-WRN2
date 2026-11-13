"""Exchangeability diagnostics from blockhist.py output.
MI(mag_k ; mag_{k-d} | srel) within a 32-block for several distances d, and across adjacent blocks of a row
(same position, block b vs b-1, conditioned on the srel of block b). If the codes of a block were exchangeable
given a hidden per-block scale, MI would not depend on d.
Also: srel-only order-0 table cost (from the pass-1 counts) to compare with the single-lambda parametric model.

Usage: python -I pairs_mi.py HIST_DIR COUNTS_DIR OUT_JSON
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockhist import DISTS


def mi(j):
    """j [..., 8, 8] counts -> conditional MI in bits summed with weights over leading dims / total"""
    j = j.astype(np.float64)
    tot = j.sum((-1, -2), keepdims=True)
    pj = j / np.maximum(tot, 1)
    pa = pj.sum(-1, keepdims=True); pb = pj.sum(-2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(pj > 0, pj * np.log2(pj / (pa * pb)), 0.0)
    m = t.sum((-1, -2))  # per leading index
    w = tot[..., 0, 0]
    return float((m * w).sum() / w.sum())


def main():
    hist_dir, counts_dir, out_json = sys.argv[1:4]
    files = sorted(f for f in glob.glob(os.path.join(hist_dir, '*.npz')) if not f.endswith('.tmp.npz'))
    pair = left = None
    for f in files:
        z = np.load(f, allow_pickle=False)
        pair = z['pair'].astype(np.int64) if pair is None else pair + z['pair']
        left = z['left'].astype(np.int64) if left is None else left + z['left']
    out = {'experts': len(files)}
    out['within_block_MI_bits_given_srel'] = {str(d): mi(pair[:, :, j]) for j, d in enumerate(DISTS)}
    out['adjacent_block_same_position_MI_bits_given_srel'] = mi(left)
    # srel-only order-0 vs base 567 table (in-sample, ideal) on the counts of the same experts
    names = [os.path.basename(f) for f in files]
    C = sum(np.load(os.path.join(counts_dir, n), allow_pickle=False)['w_base'].astype(np.float64) for n in names)
    def H(c):
        t = c.sum(-1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            return float(-(np.where(c > 0, c * np.log2(c / t), 0)).sum())
    W = C.sum()
    out['insample_ideal_bpw'] = {
        'order0_per_type': H(C.sum(1)) / W,
        'srel_only_per_type': H(C.reshape(3, 7, 81, 16).sum(2)) / W,
        'srel_mx (no pm)': H(C.reshape(3, 7, 9, 9, 16).sum(2).reshape(3, 63, 16)) / W,
        'srel_pm (no mx)': H(C.reshape(3, 7, 9, 9, 16).sum(3).reshape(3, 63, 16)) / W,
        'base_567': H(C) / W,
    }
    print(json.dumps(out, indent=1))
    json.dump(out, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
