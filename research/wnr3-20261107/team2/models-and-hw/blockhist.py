"""Per-block magnitude histograms + exchangeability diagnostics for V4.1 MXFP4 experts.

Usage: python -I blockhist.py SAMPLE_DIR OUT_DIR [workers] [limit]
For every expert in SAMPLE_DIR/manifest.json writes OUT_DIR/<name>.npz:
  key_m{0,1,2}, cnt_m{0,1,2}: unique block keys and counts. key = srel (3 bits, 0..6, srel exactly as WNR2:
      clamp(s - lower_median, -3, 3) + 3) << 48 | sum_m n_m << (6*m), n_m = count of magnitude m (x&7) in the 32-block.
  pair  [3, 7, ND, 8, 8]: within-block pair counts (srel, distance d, mag_k, mag_{k-d}) summed over valid k.
  left  [3, 7, 8, 8]:      (srel of block b, mag at position k of block b, mag at position k of block b-1, same row).
  sign  [3, 8, 2]:         magnitude x sign counts.
  meta  [3, 3]:            rows, cols, median.
"""
import json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

DISTS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 31]


def load_expert(path, tensors):
    raw = np.fromfile(path, dtype=np.uint8)
    out, off = {}, 0
    for t in tensors:
        n = t['bytes']
        out[t['name']] = (raw[off:off + n], t['shape'])
        off += n
    assert off == raw.size
    return out


def matrix(wb, sc):
    R = wb.shape[0]
    C = wb.shape[1] * 2
    nb = C // 32
    sym = np.empty((R, C), np.uint8)
    sym[:, 0::2] = wb & 15
    sym[:, 1::2] = wb >> 4
    blk = sym.reshape(R * nb, 32)
    mag = (blk & 7).astype(np.int64)
    s16 = sc.astype(np.int16).ravel()
    med = int(np.sort(s16)[(s16.size - 1) // 2])
    srel = (np.clip(s16 - med, -3, 3) + 3).astype(np.int64)  # [R*nb]
    nblk = R * nb
    hist = np.bincount((np.arange(nblk, dtype=np.int64)[:, None] * 8 + mag).ravel(), minlength=nblk * 8).reshape(nblk, 8)
    key = srel << 48
    for m in range(8):
        key |= hist[:, m] << (6 * m)
    uk, uc = np.unique(key, return_counts=True)
    pair = np.zeros((7, len(DISTS), 8, 8), np.int64)
    for j, d in enumerate(DISTS):
        a = mag[:, d:]
        b = mag[:, :-d]
        idx = (srel[:, None] * 64 + a * 8 + b).ravel()
        pair[:, j] = np.bincount(idx, minlength=7 * 64).reshape(7, 8, 8)
    mg = mag.reshape(R, nb, 32)
    sr = srel.reshape(R, nb)
    idx = (sr[:, 1:, None] * 64 + mg[:, 1:, :] * 8 + mg[:, :-1, :]).ravel()
    left = np.bincount(idx, minlength=7 * 64).reshape(7, 8, 8)
    sign = np.bincount(((blk & 7).astype(np.int64) * 2 + (blk >> 3)).ravel(), minlength=16).reshape(8, 2)
    return uk, uc.astype(np.int32), pair, left, sign, (R, C, med)


def do_expert(args):
    sample_dir, out_dir, ex = args
    outp = os.path.join(out_dir, ex['file'].replace('.bin', '.npz'))
    if os.path.exists(outp):
        return ex['file'], 0.0
    t0 = time.time()
    t = load_expert(os.path.join(sample_dir, ex['file']), ex['tensors'])
    res = {}
    pairs, lefts, signs, metas = [], [], [], []
    for m, w in enumerate(['w1', 'w2', 'w3']):
        wraw, wshape = t[f'{w}.weight']
        sraw, sshape = t[f'{w}.scale']
        R = wshape[0]
        uk, uc, pair, left, sign, meta = matrix(wraw.reshape(R, -1), sraw.reshape(R, -1))
        res[f'key_m{m}'] = uk
        res[f'cnt_m{m}'] = uc
        pairs.append(pair); lefts.append(left); signs.append(sign); metas.append(meta)
    res['pair'] = np.stack(pairs)
    res['left'] = np.stack(lefts)
    res['sign'] = np.stack(signs)
    res['meta'] = np.array(metas, np.int64)
    np.savez_compressed(outp + '.tmp.npz', **res)
    os.replace(outp + '.tmp.npz', outp)
    return ex['file'], time.time() - t0


def main():
    sample_dir, out_dir = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    exs = man['experts'][:limit] if limit else man['experts']
    jobs = [(sample_dir, out_dir, ex) for ex in exs]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for i, (f, dt) in enumerate(pool.map(do_expert, jobs)):
            if i % 10 == 0:
                print(f'{i + 1}/{len(jobs)} {f} {dt:.1f}s elapsed {time.time() - t0:.0f}s', flush=True)
    print('done', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
