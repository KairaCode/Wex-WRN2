"""headroom / scale-stream contexts on ceil-form (anchor 120), held-out (train even-indexed, test odd-indexed experts).

Usage: python -I hscale.py SAMPLE_DIR OUT_JSON [workers]
Scale streams: 512 scales of the row-major [R, nb] scale matrix, never crossing a matrix (REF:205); all scale streams of a
record are decoded independently (team-4 wnr_common.py decode_scales_lockstep), so a stream-causal context may only use
earlier scales of the same stream. '_any' contexts use the full previous row / the partner matrix (needs sequential
or phased scale decode) and are upper bounds.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxcount import load_expert
from hfeat import ceil_mats
from hctx import quantize, RAW_RECORD

ANCHOR = 120
S_CHUNK = 512
S_PER_REC = 1_105_920


def q7(x):
    return (np.clip(x - ANCHOR, -3, 3) + 3).astype(np.int64)


def contexts(S, Sp, Sp_known):
    """S [R,nb] int16 ceil-form scales; Sp partner scales (or None). Returns dict name -> (ctx int64 [R*nb], nctx)."""
    R, nb = S.shape
    flat = np.arange(R * nb).reshape(R, nb)
    pos_in_stream = flat % S_CHUNK
    s = S
    NONE = 7
    left = np.full((R, nb), NONE, np.int64); left[:, 1:] = q7(s[:, :-1])
    left_ok = (np.arange(nb)[None, :] > 0) & (pos_in_stream > 0)
    L = np.where(left_ok, left, NONE)
    left2 = np.full((R, nb), NONE, np.int64); left2[:, 2:] = q7(s[:, :-2])
    L2 = np.where((np.arange(nb)[None, :] > 1) & (pos_in_stream > 1), left2, NONE)
    up = np.full((R, nb), NONE, np.int64); up[1:] = q7(s[:-1])
    Ua = up
    Us = np.where(pos_in_stream >= nb, up, NONE)
    ul = np.full((R, nb), NONE, np.int64); ul[1:, 1:] = q7(s[:-1, :-1])
    ULs = np.where((pos_in_stream >= nb + 1) & (np.arange(nb)[None, :] > 0), ul, NONE)
    ur = np.full((R, nb), NONE, np.int64); ur[1:, :-1] = q7(s[:-1, 1:])
    URs = np.where((pos_in_stream >= nb - 1) & (np.arange(nb)[None, :] < nb - 1) & (np.arange(R)[:, None] > 0), ur, NONE)
    # running share of scales > ANCHOR in the row so far, within the stream: buckets none,0..4
    hi = (s > ANCHOR).astype(np.int64)
    cs = np.cumsum(hi, axis=1)
    prev_hi = np.zeros((R, nb), np.int64); prev_hi[:, 1:] = cs[:, :-1]
    rowstart = np.maximum(np.arange(nb)[None, :] - np.minimum(pos_in_stream, np.arange(nb)[None, :]), 0)
    # number of in-stream predecessors in this row
    npred = np.arange(nb)[None, :] - rowstart
    hi_before_start = np.where(rowstart > 0, np.take_along_axis(np.concatenate([np.zeros((R, 1), np.int64), cs], 1),
                                                                rowstart, 1), 0)
    cnt = prev_hi - hi_before_start
    frac = np.where(npred > 0, cnt / np.maximum(npred, 1), -1)
    ROW = np.where(npred > 0, np.minimum((frac * 5).astype(np.int64), 4), 5)
    out = {
        'o0': (np.zeros(R * nb, np.int64), 1),
        'L': (L.ravel(), 8),
        'L_L2': ((L * 8 + L2).ravel(), 64),
        'U_stream': (Us.ravel(), 8),
        'U_any': (Ua.ravel(), 8),
        'LU_stream': ((L * 8 + Us).ravel(), 64),
        'LU_any': ((L * 8 + Ua).ravel(), 64),
        'LUUL_stream': (((L * 8 + Us) * 8 + ULs).ravel(), 512),
        'LUUR_stream': (((L * 8 + Us) * 8 + URs).ravel(), 512),
        'ROW': (ROW.ravel(), 6),
        'L_ROW': ((L * 6 + ROW).ravel(), 48),
        'LU_ROW_stream': (((L * 8 + Us) * 6 + ROW).ravel(), 384),
    }
    if Sp is not None and Sp_known:
        Pq = q7(Sp)
        out['PART_any'] = (Pq.ravel(), 8)
        out['L_PART_any'] = ((L * 8 + Pq).ravel(), 64)
        out['LU_PART_any'] = (((L * 8 + Ua) * 8 + Pq).ravel(), 512)
    else:
        z = np.zeros(R * nb, np.int64)
        out['PART_any'] = (z, 8)
        out['L_PART_any'] = (L.ravel(), 64)
        out['LU_PART_any'] = ((L * 8 + Ua).ravel() * 8, 512)
    return out


def do_one(args):
    sample_dir, e = args
    t = load_expert(os.path.join(sample_dir, e['file']), e['tensors'])
    mats = ceil_mats(t)
    res = {}
    for m in range(3):
        S = mats[m][1]
        Sp, known = (None, False)
        if m == 2:
            Sp, known = mats[0][1], True     # w3 scale given w1 scale (w1 scale streams first)
        for k, (ctx, n) in contexts(S, Sp, known).items():
            c = np.bincount(ctx * 256 + S.ravel().astype(np.int64), minlength=n * 256).reshape(n, 256)
            res.setdefault(k, [None] * 3)[m] = c
    return e['file'], res


def main():
    sample_dir, out_json = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    t0 = time.time()
    acc = {}
    jobs = [(sample_dir, e) for e in man['experts']]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(workers); it = pool.map(do_one, jobs)
    else:
        it = map(do_one, jobs)
    for i, (f, res) in enumerate(it):
        sp = i % 2
        for k, v in res.items():
            if k not in acc:
                acc[k] = [[np.zeros_like(x) for x in v] for _ in range(2)]
            for m in range(3):
                acc[k][sp][m] += v[m]
        if i % 20 == 0:
            print(i, f, f'{time.time() - t0:.0f}s', flush=True)
    ntest = sum(int(acc['o0'][1][m].sum()) for m in range(3))
    out = {'experts': len(jobs), 'test_scales': ntest, 'contexts': {}}
    base = None
    for k in acc:
        bits = 0.0; ins = 0.0; rows = []
        per_m = []
        for m in range(3):
            tr, te = acc[k][0][m], acc[k][1][m]
            used = tr.sum(1) > 0
            f = quantize(tr[used])
            lc = -np.log2(f / 16384)
            b = float((te[used] * lc).sum()) + float(te[~used].sum()) * 8.0
            bits += b; per_m.append(b / te.sum())
            ins += float((tr[used] * lc).sum())
            rows.append(int(used.sum()))
        out['contexts'][k] = {'heldout_bit_per_scale': bits / ntest, 'train_insample_bit_per_scale': ins / ntest,
                              'per_matrix': per_m, 'rows_used': rows, 'table_bytes_256x_u16': [r * 512 for r in rows]}
        if k == 'o0':
            base = bits / ntest
    for k, v in out['contexts'].items():
        v['delta_bit_per_scale'] = v['heldout_bit_per_scale'] - base
        v['delta_pct_scale_bits'] = 100 * v['delta_bit_per_scale'] / base
        v['delta_pct_raw'] = 100 * v['delta_bit_per_scale'] * S_PER_REC / 8 / RAW_RECORD
    # ideal (unquantised) entropy of the in-sample order-0 for reference
    out['sec'] = time.time() - t0
    json.dump(out, open(out_json, 'w'), indent=1)
    for k, v in sorted(out['contexts'].items(), key=lambda kv: kv[1]['heldout_bit_per_scale']):
        print(f"{k:16s} {v['heldout_bit_per_scale']:.5f} bit/scale  d {v['delta_pct_scale_bits']:+.2f}% of scale bits "
              f"{v['delta_pct_raw']:+.4f}% raw rows {v['rows_used']}")


if __name__ == '__main__':
    main()
