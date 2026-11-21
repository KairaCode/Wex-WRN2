"""Do the extra contexts still pay on ceil-form data? Counts per expert for several context sets on the ceil-form transform.

Usage: python -I mxceil2.py SAMPLE_DIR OUT_DIR
Keys (each [3, rows, 16] int64 unless noted):
  base    : (srel7 x pm9 x mx9) = 567
  pm2     : base x pm2 (9: none, 0..7) = 5103
  pm2b    : base x B6[pm2] (none, 0, 1, 2, 3, 4-7) = 3402
  bmh     : (bm3 x half2) x base = 3402, bm3 = ceil-form block max magnitude - 5 (0..2), coded as side info
  bmh_pm2b: bmh x B6[pm2] = 20412
  pos     : base x position-in-block (32) = 18144
  bm_side : [3, 7, 3] side-symbol counts (block max given srel)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxcount import load_expert
from mxceil import forward

B6 = np.array([1, 2, 3, 4, 5, 5, 5, 5, 0], np.int32)  # index = pm2 value (8 = none)


def counts(ctx, sym, rows):
    return np.bincount((ctx.astype(np.int64) * 16 + sym.astype(np.int64)).ravel(), minlength=rows * 16)[:rows * 16].reshape(rows, 16)


def main():
    sample_dir, out_dir = sys.argv[1:3]
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    for e in man['experts']:
        outp = os.path.join(out_dir, e['file'].replace('.bin', '.npz'))
        if os.path.exists(outp):
            continue
        t = load_expert(os.path.join(sample_dir, e['file']), e['tensors'])
        acc = {k: [] for k in ['base', 'pm2', 'pm2b', 'bmh', 'bmh_pm2b', 'pos']}
        side = np.zeros((3, 7, 3), np.int64)
        for m, w in enumerate(['w1', 'w2', 'w3']):
            wraw, wshape, _ = t[f'{w}.weight']
            sraw, _, _ = t[f'{w}.scale']
            R = wshape[0]
            wb = wraw.reshape(R, -1)
            sc = sraw.reshape(R, -1)
            C = wb.shape[1] * 2
            sym = np.empty((R, C), np.uint8)
            sym[:, 0::2] = wb & 15
            sym[:, 1::2] = wb >> 4
            tsym, tsc, even, _ = forward(sym, sc)
            nb = C // 32
            blk = tsym.reshape(R, nb, 32)
            mag = (blk & 7).astype(np.int32)
            pm = np.empty_like(mag); pm[..., 0] = 8; pm[..., 1:] = mag[..., :-1]
            pm2 = np.empty_like(mag); pm2[..., :2] = 8; pm2[..., 2:] = mag[..., :-2]
            cm = np.maximum.accumulate(mag, axis=2)
            mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
            flat = np.sort(tsc.ravel())
            med = int(flat[(flat.size - 1) // 2])
            srel = (np.clip(tsc.astype(np.int32) - med, -3, 3) + 3)
            base = (srel[..., None] * 9 + pm) * 9 + mx
            bm3 = mag.max(2) - 5
            assert bm3.min() >= 0 and bm3.max() <= 2
            half = np.zeros(32, np.int32); half[16:] = 1
            pos = np.arange(32, dtype=np.int32)
            bmh = (bm3[..., None] * 2 + half) * 567 + base
            acc['base'].append(counts(base, blk, 567))
            acc['pm2'].append(counts(base * 9 + pm2, blk, 5103))
            acc['pm2b'].append(counts(base * 6 + B6[pm2], blk, 3402))
            acc['bmh'].append(counts(bmh, blk, 3402))
            acc['bmh_pm2b'].append(counts(bmh * 6 + B6[pm2], blk, 20412))
            acc['pos'].append(counts(base * 32 + pos, blk, 18144))
            side[m] = np.bincount((srel * 3 + bm3).ravel(), minlength=21).reshape(7, 3)
        np.savez_compressed(outp, bm_side=side, **{k: np.stack(v) for k, v in acc.items()})
        print(e['file'], flush=True)


if __name__ == '__main__':
    main()
