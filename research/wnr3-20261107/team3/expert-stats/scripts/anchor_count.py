"""Pass: team-1 combined contexts (c_bm2h, c_bm2p2b, c_z) counted with a WIDE srel (s - median clamped to +-7, 15 values)
so that any per-matrix anchor shift |d| <= 3 can be re-mapped exactly afterwards (srel = clamp(s - med - d, -3, 3)).

Usage: python -I anchor_count.py SAMPLE_DIR OUT_DIR
Per expert npz, per matrix (decode order w1, w2, w3):
  b15h   [3, 15*81*2*2, 16]  (srel15, pm, mx) x bm2 x half      -> c_bm2h after re-map
  b15p   [3, 15*81*2*4, 16]  (srel15, pm, mx) x bm2 x pm2 bucket -> c_bm2p2b after re-map
  b15z   [3, 15*81*2*4, 16]  same with -0 mapped to +0          -> c_z after re-map
  side15 [3, 15, 2]          srel15 x (block max == 7)
  med    [3]
Context definitions copied from team 1 mxcombo.py (P2B buckets, bm2 = final block max == 7, half = position >= 16).
"""
import json, os, sys, time
import numpy as np

P2B = np.array([1, 1, 1, 1, 2, 2, 3, 3, 0], np.int32)


def bc(idx, n):
    return np.bincount(idx.ravel(), minlength=n)[:n]


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
    bm2 = (cm[..., -1] == 7).astype(np.int32)
    s16 = sc.astype(np.int16)
    med = int(np.sort(s16.ravel())[(s16.size - 1) // 2])
    srel = (np.clip(s16 - med, -7, 7) + 7).astype(np.int32)
    base = (srel[..., None] * 9 + pm.astype(np.int32)) * 9 + mx.astype(np.int32)   # 1215 values
    b = blk.astype(np.int32)
    p2b = P2B[pm2]
    half = (np.arange(32) >= 16).astype(np.int32)[None, None, :]
    bm = bm2[..., None]
    k3 = (base * 2 + bm) * 4 + p2b
    r = {'b15h': bc(((base * 2 + bm) * 2 + half) * 16 + b, 4860 * 16).reshape(4860, 16),
         'b15p': bc(k3 * 16 + b, 9720 * 16).reshape(9720, 16),
         'b15z': bc(k3 * 16 + np.where(b == 8, 0, b), 9720 * 16).reshape(9720, 16),
         'side15': bc(srel * 2 + bm2, 30).reshape(15, 2),
         'med': np.int64(med)}
    return r


def main():
    sample_dir, out_dir = sys.argv[1:3]
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    t0 = time.time()
    for k, ex in enumerate(man['experts']):
        outp = os.path.join(out_dir, ex['file'].replace('.bin', '.npz'))
        if os.path.exists(outp):
            continue
        raw = np.fromfile(os.path.join(sample_dir, ex['file']), dtype=np.uint8)
        t, off = {}, 0
        for tt in ex['tensors']:
            t[tt['name']] = (raw[off:off + tt['bytes']], tt['shape'])
            off += tt['bytes']
        assert off == raw.size
        res = {}
        for w in ['w1', 'w2', 'w3']:
            wr, ws = t[w + '.weight']; sr, ss = t[w + '.scale']
            for kk, v in stats(wr.reshape(ws[0], -1), sr.reshape(ss[0], -1)).items():
                res.setdefault(kk, []).append(v)
        np.savez_compressed(outp + '.tmp.npz', **{kk: np.stack(v) for kk, v in res.items()})
        os.replace(outp + '.tmp.npz', outp)
        if k % 20 == 0:
            print(k, ex['file'], f'{time.time() - t0:.0f}s', flush=True)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
