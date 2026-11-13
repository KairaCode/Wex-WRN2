"""Pass: "local phase" contexts derived from the decoded scales only (no side info).
anchor A = lower value of the densest adjacent pair of the matrix's scales; srel = clamp(s - A, -3, 3).
phase of a row    = share of that row's blocks with s >= A+1    (known once the row's scales are decoded)
phase of a column = share of that block-column's blocks with s >= A+1 (over all rows)
Both bucketed by fixed edges EDGES (8 bins). Contexts: base_fix x rowbin, base_fix x colbin, and the plain base_fix.

Usage: python -I phase_count.py SAMPLE_DIR OUT_DIR
"""
import json, os, sys, time
import numpy as np

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
    cm = np.maximum.accumulate(mag, axis=2)
    mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
    h = np.bincount(sc.ravel(), minlength=256)
    A = int((h[:-1] + h[1:]).argmax())
    s16 = sc.astype(np.int16)
    srel = (np.clip(s16 - A, -3, 3) + 3).astype(np.int32)
    base = (srel[..., None] * 9 + pm.astype(np.int32)) * 9 + mx.astype(np.int32)
    b = blk.astype(np.int32)
    up = (s16 >= A + 1)
    rph = up.mean(1); cph = up.mean(0)
    rbin = np.searchsorted(EDGES, rph, side='right').astype(np.int32)[:, None, None]
    cbin = np.searchsorted(EDGES, cph, side='right').astype(np.int32)[None, :, None]
    return {'base_fix': bc(base * 16 + b, 567 * 16).reshape(567, 16),
            'row8': bc((base * 8 + rbin) * 16 + b, 4536 * 16).reshape(4536, 16),
            'col8': bc((base * 8 + cbin) * 16 + b, 4536 * 16).reshape(4536, 16),
            'rowbin_hist': np.bincount(rbin.ravel(), minlength=8), 'colbin_hist': np.bincount(cbin.ravel(), minlength=8),
            'row_phase_std': np.float64(rph.std()), 'col_phase_std': np.float64(cph.std()), 'anchor': np.int64(A)}


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
