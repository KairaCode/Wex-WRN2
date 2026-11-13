"""Pass: anchor fix (densest pair) + row phase bins on top of team 1's (a) c_bm2h and (c) c_z contexts.

Usage: python -I phase_combo_count.py SAMPLE_DIR OUT_DIR
Per matrix: h_row8 [2268*8,16] = c_bm2h(fix) x rowbin8; z_row8 [4536*8,16] = c_z(fix) x rowbin8; side_fix [7,2].
Context code copied from team 1 mxcombo.py; anchor and row bins from phase_count.py.
"""
import json, os, sys, time
import numpy as np

P2B = np.array([1, 1, 1, 1, 2, 2, 3, 3, 0], np.int32)
EDGES = np.array([0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6])


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
    h = np.bincount(sc.ravel(), minlength=256)
    A = int((h[:-1] + h[1:]).argmax())
    s16 = sc.astype(np.int16)
    srel = (np.clip(s16 - A, -3, 3) + 3).astype(np.int32)
    base = (srel[..., None] * 9 + pm.astype(np.int32)) * 9 + mx.astype(np.int32)
    b = blk.astype(np.int32)
    p2b = P2B[pm2]
    half = (np.arange(32) >= 16).astype(np.int32)[None, None, :]
    bm = bm2[..., None]
    rbin = np.searchsorted(EDGES, (s16 >= A + 1).mean(1), side='right').astype(np.int32)[:, None, None]
    kh = ((base * 2 + bm) * 2 + half)
    k3 = (base * 2 + bm) * 4 + p2b
    return {'h_row8': bc((kh * 8 + rbin) * 16 + b, 2268 * 8 * 16).reshape(2268 * 8, 16),
            'z_row8': bc((k3 * 8 + rbin) * 16 + np.where(b == 8, 0, b), 4536 * 8 * 16).reshape(4536 * 8, 16),
            'side_fix': bc(srel * 2 + bm2, 14).reshape(7, 2)}


def main():
    sample_dir, out_dir = sys.argv[1:3]
    start, step = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    t0 = time.time()
    for k, ex in enumerate(man['experts']):
        if k % step != start:
            continue
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
