"""Reversible 'ceil-form' block transform (byte-exact, lossless) and its effect on the static WNR2 model.

Canonical OCP blocks whose codes use no odd half-steps (no |x| = 0.5 or 1.5) and whose max is 6.0 are re-expressed
as (codes / 2, scale + 1): max becomes 3.0. Inverse (decoder): a block whose max magnitude is 3.0 (mag 5) is
mapped back (codes * 2, scale - 1). Bijective because canonical blocks never have max 3.0 (measured: max is always
4.0 or 6.0) and halving an even-only block is exact.
Per expert, writes counts for the WNR2 567-context rule computed on (a) the canonical data and (b) the ceil form
(srel recomputed with the ceil-form median), plus scale order-0 counts and block-max side counts in both forms.

Usage: python -I ceilform.py SAMPLE_DIR OUT_DIR [workers]
"""
import json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockhist import load_expert

HALF = np.array([0, 0, 1, 0, 2, 3, 4, 5], np.uint8)  # magnitude index of value/2 for even magnitudes {0,2,4,5,6,7}


def ctx_counts(blk, s16):
    mag = (blk & 7).astype(np.int32)
    med = int(np.sort(s16)[(s16.size - 1) // 2])
    srel = (np.clip(s16.astype(np.int32) - med, -3, 3) + 3)[:, None]
    pm = np.empty_like(mag); pm[:, 0] = 8; pm[:, 1:] = mag[:, :-1]
    cm = np.maximum.accumulate(mag, 1)
    mx = np.empty_like(mag); mx[:, 0] = 8; mx[:, 1:] = cm[:, :-1]
    base = (srel * 9 + pm) * 9 + mx
    w = np.bincount((base * 16 + blk.astype(np.int32)).ravel(), minlength=567 * 16).reshape(567, 16)
    bm = cm[:, -1]
    side = np.bincount(srel[:, 0] * 8 + bm, minlength=56).reshape(7, 8)
    return w, side, med


def do_expert(args):
    sample_dir, out_dir, ex = args
    outp = os.path.join(out_dir, ex['file'].replace('.bin', '.npz'))
    if os.path.exists(outp):
        return ex['file'], 0.0, None
    t0 = time.time()
    t = load_expert(os.path.join(sample_dir, ex['file']), ex['tensors'])
    res = {k: [] for k in ['w_can', 'w_ceil', 'side_can', 'side_ceil', 's_can', 's_ceil', 'meta']}
    for m, wn in enumerate(['w1', 'w2', 'w3']):
        wraw, wshape = t[f'{wn}.weight']; sraw, _ = t[f'{wn}.scale']
        R = wshape[0]
        wb = wraw.reshape(R, -1)
        C = wb.shape[1] * 2
        sym = np.empty((R, C), np.uint8); sym[:, 0::2] = wb & 15; sym[:, 1::2] = wb >> 4
        blk = sym.reshape(-1, 32)
        s = sraw.astype(np.int16).copy()
        mag = blk & 7
        has_odd = ((mag == 1) | (mag == 3)).any(1)
        bmax = mag.max(1)
        assert (bmax >= 6).all(), 'canonical block with max < 4.0: transform would not be bijective'
        ev = (~has_odd) & (bmax == 7)
        blk2 = blk.copy()
        blk2[ev] = (blk[ev] & 8) | HALF[mag[ev]]
        s2 = s.copy(); s2[ev] += 1
        assert (s2 <= 254).all()
        # inverse check (decoder rule): blocks with max magnitude 5 (3.0) are expanded back
        mag2 = blk2 & 7
        back = mag2.max(1) == 5
        assert (back == ev).all()
        inv = blk2.copy()
        DOUBLE = np.array([0, 2, 4, 5, 6, 7, 255, 255], np.uint8)
        inv[back] = (blk2[back] & 8) | DOUBLE[mag2[back]]
        assert (inv == blk).all() and ((s2 - back) == s).all()
        wc, sc_, medc = ctx_counts(blk, s)
        wl, sl, medl = ctx_counts(blk2, s2)
        res['w_can'].append(wc); res['w_ceil'].append(wl); res['side_can'].append(sc_); res['side_ceil'].append(sl)
        res['s_can'].append(np.bincount(s.astype(np.int64), minlength=256))
        res['s_ceil'].append(np.bincount(s2.astype(np.int64), minlength=256))
        res['meta'].append([R, C, medc, medl, int(ev.sum())])
    out = {k: np.array(v, np.int64) for k, v in res.items()}
    np.savez_compressed(outp + '.tmp.npz', **out)
    os.replace(outp + '.tmp.npz', outp)
    return ex['file'], time.time() - t0, out['meta'][:, 4].sum()


def main():
    sample_dir, out_dir = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    jobs = [(sample_dir, out_dir, ex) for ex in man['experts']]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for i, (f, dt, nev) in enumerate(pool.map(do_expert, jobs)):
            if i % 10 == 0:
                print(f'{i + 1}/{len(jobs)} {f} {dt:.1f}s even-blocks {nev} elapsed {time.time() - t0:.0f}s', flush=True)
    print('done', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
