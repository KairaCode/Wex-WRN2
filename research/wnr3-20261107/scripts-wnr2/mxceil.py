"""Independent check of the 'ceil-form' claim on the real V4.1 sample.

Usage: python -I mxceil.py SAMPLE_DIR OUT_DIR

Per expert and matrix:
  - fingerprint: blocks with max magnitude 7 and no magnitude 1 or 3 ('even' blocks), block-max histogram over 0..7
  - forward transform T: even block -> magnitudes halved (0,2,4,5,6,7 -> 0,1,2,3,4,5; sign nibble bit kept), scale + 1
  - checks: no original block has max magnitude 5 (or any max < 6), no transformed scale would hit 255 (E8M0 NaN),
    inverse(T(x)) == x byte for byte on weights and scales
  - counts for the WNR2 base context (srel7 x pm9 x mx9) x 16 on the original and on the transformed data
    (median recomputed on the transformed scales, as a retrained model would), plus order-0 scale counts.
Writes one npz per expert into OUT_DIR: w_orig, w_ceil [3,567,16], s_orig, s_ceil [3,256], stats [3, 12].
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxcount import load_expert

HALF = np.array([0, 255, 1, 255, 2, 3, 4, 5], np.int16)    # magnitude -> halved magnitude (255 = not allowed)
DOUBLE = np.array([0, 2, 4, 5, 6, 7, 255, 255], np.int16)  # halved magnitude -> original magnitude


def ctx_counts(sym, sc):
    R, C = sym.shape
    nb = C // 32
    blk = sym.reshape(R, nb, 32)
    mag = blk & 7
    pm = np.empty_like(mag); pm[..., 0] = 8; pm[..., 1:] = mag[..., :-1]
    cm = np.maximum.accumulate(mag, axis=2)
    mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
    flat = np.sort(sc.ravel())
    med = int(flat[(flat.size - 1) // 2])
    srel = (np.clip(sc.astype(np.int32) - med, -3, 3) + 3)
    ctx = (srel[..., None] * 9 + pm.astype(np.int32)) * 9 + mx.astype(np.int32)
    c = np.bincount((ctx * 16 + blk.astype(np.int32)).ravel(), minlength=567 * 16).reshape(567, 16)
    return c


def forward(sym, sc):
    """sym [R, C] uint8 nibbles, sc [R, nb] uint8. Returns transformed copies and the even-block mask."""
    R, C = sym.shape
    nb = C // 32
    blk = sym.reshape(R, nb, 32)
    mag = (blk & 7).astype(np.int16)
    bmax = mag.max(2)
    has13 = ((mag == 1) | (mag == 3)).any(2)
    even = (bmax == 7) & ~has13
    newmag = HALF[mag]
    tb = np.where(even[..., None], (blk & 8).astype(np.int16) | newmag, blk.astype(np.int16)).astype(np.uint8)
    tsc = np.where(even, sc.astype(np.int16) + 1, sc.astype(np.int16))
    return tb.reshape(R, C), tsc, even, bmax


def inverse(tsym, tsc):
    R, C = tsym.shape
    nb = C // 32
    blk = tsym.reshape(R, nb, 32)
    mag = (blk & 7).astype(np.int16)
    bmax = mag.max(2)
    odd = bmax == 5  # transformed blocks are exactly the blocks whose decoded max is 5
    om = DOUBLE[np.minimum(mag, 7)]
    ob = np.where(odd[..., None], (blk & 8).astype(np.int16) | om, blk.astype(np.int16)).astype(np.uint8)
    osc = np.where(odd, tsc - 1, tsc)
    return ob.reshape(R, C), osc


def main():
    sample_dir, out_dir = sys.argv[1:3]
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    agg = {'blocks': 0, 'even': 0, 'max_hist': np.zeros(8, np.int64), 'bad_max': 0, 'scale_overflow': 0, 'roundtrip_fail': 0,
           'even_and_zero_only_sign': 0}
    for e in man['experts']:
        outp = os.path.join(out_dir, e['file'].replace('.bin', '.npz'))
        t = load_expert(os.path.join(sample_dir, e['file']), e['tensors'])
        wo = np.zeros((3, 567, 16), np.int64); wc = np.zeros_like(wo)
        so = np.zeros((3, 256), np.int64); scl = np.zeros_like(so)
        stats = np.zeros((3, 12), np.int64)
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
            tsym, tsc, even, bmax = forward(sym, sc)
            hist = np.bincount(bmax.ravel(), minlength=8)
            bad = int((bmax < 6).sum())
            ovf = int((tsc > 254).sum())
            isym, isc = inverse(tsym, tsc)
            rt = int(not (np.array_equal(isym, sym) and np.array_equal(isc.astype(np.uint8), sc)))
            wo[m] = ctx_counts(sym, sc)
            wc[m] = ctx_counts(tsym, tsc.astype(np.int16))
            so[m] = np.bincount(sc.ravel(), minlength=256)
            scl[m] = np.bincount(tsc.ravel().astype(np.int64), minlength=256)[:256]
            stats[m, :8] = hist
            stats[m, 8] = int(even.sum()); stats[m, 9] = bad; stats[m, 10] = ovf; stats[m, 11] = rt
            agg['blocks'] += even.size; agg['even'] += int(even.sum()); agg['max_hist'] += hist
            agg['bad_max'] += bad; agg['scale_overflow'] += ovf; agg['roundtrip_fail'] += rt
        np.savez_compressed(outp, w_orig=wo, w_ceil=wc, s_orig=so, s_ceil=scl, stats=stats)
        print(e['file'], 'even%', round(100 * stats[:, 8].sum() / stats[:, :8].sum(), 3), 'bad', stats[:, 9].sum(),
              'ovf', stats[:, 10].sum(), 'rt_fail', stats[:, 11].sum(), flush=True)
    agg['max_hist'] = agg['max_hist'].tolist()
    agg['even_pct'] = 100 * agg['even'] / agg['blocks']
    print(json.dumps(agg, indent=1))
    json.dump(agg, open(os.path.join(out_dir, 'fingerprint.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
