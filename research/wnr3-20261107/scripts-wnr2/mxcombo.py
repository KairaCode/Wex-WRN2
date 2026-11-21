"""Pass 1b: combined contexts that respect the GPU shared-memory budget, plus zero statistics.

Usage: python -I mxcombo.py SAMPLE_DIR OUT_DIR [workers]
The WNR2 kernel keeps a matrix's CDF rows in shared memory (32 B per row: 16 cumulative u16).
At ~99 KB per block that is ~3,100 rows; 567 today (18 KB). Only rows that are REACHABLE need storage
(an index map compacts them), so we also report used rows per variant.
Variants (per matrix, counts [rows, 16]):
  c_bm2      base x (final block max == 7)            1,134 rows  + side bit per block (bm2_side [7,2])
  c_p2b      base x pm2 bucket {none, 0-3, 4-5, 6-7}  2,268 rows
  c_bm2p2b   base x bm2 x pm2 bucket                  4,536 rows (needs compaction)
  c_bm2h     base x bm2 x half of block               2,268 rows
  c_bm2p2bh  base x bm2 x pm2 bucket x half           9,072 rows (compaction / L1)
  c_z        WNR2z (-0 -> +0) on c_bm2p2b             value-exact
  zeros      [3, 2] counts of +0 and -0
"""
import json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxcount import load_expert, bc

P2B = np.array([1, 1, 1, 1, 2, 2, 3, 3, 0], np.int32)  # magnitude 0..7, 8 = none -> bucket


def stats(wb, sc):
    R = wb.shape[0]; C = wb.shape[1] * 2; nb = C // 32
    sym = np.empty((R, C), np.uint8)
    sym[:, 0::2] = wb & 15; sym[:, 1::2] = wb >> 4
    blk = sym.reshape(R, nb, 32)
    mag = blk & 7
    pm = np.empty_like(mag); pm[..., 0] = 8; pm[..., 1:] = mag[..., :-1]
    pm2 = np.empty_like(mag); pm2[..., :2] = 8; pm2[..., 2:] = mag[..., :-2]
    cm = np.maximum.accumulate(mag, axis=2)
    mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
    bm2 = (cm[..., -1] == 7).astype(np.int32)  # block max is always 6 or 7 under the OCP scale rule
    s16 = sc.astype(np.int16)
    med = int(np.sort(s16.ravel())[(s16.size - 1) // 2])
    srel = (np.clip(s16 - med, -3, 3) + 3).astype(np.int32)
    base = (srel[..., None] * 9 + pm.astype(np.int32)) * 9 + mx.astype(np.int32)
    b = blk.astype(np.int32)
    p2b = P2B[pm2]
    half = (np.arange(32) >= 16).astype(np.int32)[None, None, :]
    bm = bm2[..., None]
    r = {}
    r['c_bm2'] = bc((base * 2 + bm) * 16 + b, 1134 * 16).reshape(1134, 16)
    r['bm2_side'] = bc(srel * 2 + bm2, 14).reshape(7, 2)
    r['c_p2b'] = bc((base * 4 + p2b) * 16 + b, 2268 * 16).reshape(2268, 16)
    k3 = (base * 2 + bm) * 4 + p2b
    r['c_bm2p2b'] = bc(k3 * 16 + b, 4536 * 16).reshape(4536, 16)
    r['c_bm2h'] = bc(((base * 2 + bm) * 2 + half) * 16 + b, 2268 * 16).reshape(2268, 16)
    r['c_bm2p2bh'] = bc((k3 * 2 + half) * 16 + b, 9072 * 16).reshape(9072, 16)
    bz = np.where(b == 8, 0, b)
    r['c_z'] = bc(k3 * 16 + bz, 4536 * 16).reshape(4536, 16)
    r['zeros'] = np.array([int((sym == 0).sum()), int((sym == 8).sum())], np.int64)
    return r


def do(args):
    sample_dir, out_dir, ex = args
    outp = os.path.join(out_dir, ex['file'].replace('.bin', '.npz'))
    if os.path.exists(outp):
        return 0
    t = load_expert(os.path.join(sample_dir, ex['file']), ex['tensors'])
    res = {}
    for w in ['w1', 'w2', 'w3']:
        wraw, wshape, _ = t[f'{w}.weight']; sraw, _, _ = t[f'{w}.scale']
        R = wshape[0]
        for k, v in stats(wraw.reshape(R, -1), sraw.reshape(R, -1)).items():
            res.setdefault(k, []).append(v)
    np.savez_compressed(outp + '.tmp.npz', **{k: np.stack(v) for k, v in res.items()})
    os.replace(outp + '.tmp.npz', outp)
    return 1


def main():
    sample_dir, out_dir = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        n = sum(pool.map(do, [(sample_dir, out_dir, e) for e in man['experts']]))
    print('done', n, time.time() - t0)


if __name__ == '__main__':
    main()
