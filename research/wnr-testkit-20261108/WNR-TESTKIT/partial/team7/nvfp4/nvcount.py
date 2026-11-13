"""Pass 1 for NVFP4 experts: per-item context counts on 16-blocks (WNR-style srel / pm / mx) and scale statistics.

Usage: python -I nvcount.py FAM_DIR OUT_DIR
Writes OUT_DIR/<tag>.npz (int64) with, per matrix m in decode order (w1 gate, w2 down, w3 up):
  meta     [3, 8]   R, C, anchor byte, live blocks, zero-scale blocks, +0 count, -0 count, blocks with max 0
  w_o0     [3, 16]          order-0 codes
  w_b{W}   [3, 648, 16]     srel8 x pm9 x mx9, srel8 = clamp(floor((sb - anchor)/W), -3, 3) + 3, 7 = 'far' (sb < anchor - 24)
  w_b{W}p  [3, 648*16, 16]  same x position in the 16-block                         (W in 1, 2, 4, 8)
  w_s15    [3, 16*81, 16]   srel16 = clamp(sb - anchor, -7, 7) + 7, 15 = far
  w_b2pm2  [3, 648*9, 16]   b2 x magnitude two back (8 = none)
  w_b2lm   [3, 648*9, 16]   b2 x max magnitude of the previous 16-block in the row (8 = none)
  w_b2ls   [3, 648*3, 16]   b2 x sign(sb - sb_left) (0 none/equal, 1 lower, 2 higher)
  w_b2par  [3, 648*2, 16]   b2 x block parity (first / second half of a 32-group)
  v_b2 / v_b2p              value-exact variant: -0 -> +0, and codes of zero-VALUE-scale blocks -> 0
  bmax     [3, 256, 8]      scale byte x block max magnitude
  n13      [3, 17]          max-7 blocks by number of magnitudes in {1, 3} ('doubled' 16-blocks have 0)
  magmax7  [3, 8]           magnitude histogram inside max-7 blocks (non-max positions estimate)
  s_o0 [3,256]; s_left/s_up [3,257,256]; s_lu [3,256,256]; s_par [3,514,256]
  zsgn     [3, 2, 2]        (+0, -0) counts in live vs zero-scale blocks
"""
import importlib.util, json, os, sys, time
import numpy as np

_s = importlib.util.spec_from_file_location('nvcommon', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nvcommon.py'))
nc = importlib.util.module_from_spec(_s); _s.loader.exec_module(nc)


def bc(idx, n):
    return np.bincount(idx.ravel(), minlength=n)[:n]


def wctx(codes, sb, anchor):
    """Return dict of context arrays (int32, shape [R, nb, 16]) and helpers."""
    R, C = codes.shape
    nb = C // 16
    blk = codes.reshape(R, nb, 16)
    mag = (blk & 7).astype(np.int32)
    pm = np.empty_like(mag); pm[..., 0] = 8; pm[..., 1:] = mag[..., :-1]
    pm2 = np.empty_like(mag); pm2[..., :2] = 8; pm2[..., 2:] = mag[..., :-2]
    cm = np.maximum.accumulate(mag, axis=2)
    mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
    bm = cm[..., -1]
    d = sb.astype(np.int32) - anchor
    far = d < -24
    pmmx = pm * 9 + mx
    out = {}
    for W in (1, 2, 4, 8):
        s8 = np.where(far, 7, np.clip(np.floor_divide(d, W), -3, 3) + 3)
        out[W] = s8[..., None] * 81 + pmmx
    s16 = np.where(far, 15, np.clip(d, -7, 7) + 7)
    out['s15'] = s16[..., None] * 81 + pmmx
    lm = np.empty_like(bm); lm[:, 0] = 8; lm[:, 1:] = bm[:, :-1]
    sl = np.zeros_like(d); sl[:, 1:] = np.sign(sb[:, 1:].astype(np.int32) - sb[:, :-1].astype(np.int32))
    sl = np.where(sl < 0, 1, np.where(sl > 0, 2, 0))
    par = np.broadcast_to((np.arange(nb) & 1)[None, :], (R, nb))
    return out, blk.astype(np.int32), mag, pm2, bm, lm, sl, par


def matrix_counts(codes, sb):
    R, C = codes.shape
    nb = C // 16
    sv = nc.e4m3_value(sb)
    mag = (codes.reshape(R, nb, 16) & 7)
    bmax = mag.max(2)
    live = (sv > 0) & (bmax > 0)
    if live.any():
        ls = np.sort(sb[live].ravel())
        anchor = int(ls[(ls.size - 1) // 2])
    else:
        anchor = int(np.sort(sb.ravel())[(sb.size - 1) // 2])
    ctx, b32, mg, pm2, bm, lm, sl, par = wctx(codes, sb, anchor)
    r = {}
    r['w_o0'] = bc(b32, 16)
    for W in (1, 2, 4, 8):
        r[f'w_b{W}'] = bc(ctx[W] * 16 + b32, 648 * 16).reshape(648, 16)
        pos = np.arange(16, dtype=np.int32)[None, None, :]
        r[f'w_b{W}p'] = bc((ctx[W] * 16 + pos) * 16 + b32, 648 * 16 * 16).reshape(648 * 16, 16)
    r['w_s15'] = bc(ctx['s15'] * 16 + b32, 16 * 81 * 16).reshape(16 * 81, 16)
    c2 = ctx[2]
    r['w_b2pm2'] = bc((c2 * 9 + pm2) * 16 + b32, 648 * 9 * 16).reshape(648 * 9, 16)
    r['w_b2lm'] = bc((c2 * 9 + lm[..., None]) * 16 + b32, 648 * 9 * 16).reshape(648 * 9, 16)
    r['w_b2ls'] = bc((c2 * 3 + sl[..., None]) * 16 + b32, 648 * 3 * 16).reshape(648 * 3, 16)
    r['w_b2par'] = bc((c2 * 2 + par[..., None]) * 16 + b32, 648 * 2 * 16).reshape(648 * 2, 16)
    # value-exact variant
    vc = codes.copy()
    vc[vc == 8] = 0
    vb = vc.reshape(R, nb, 16)
    vb[(sv == 0)] = 0
    vctx, vb32 = wctx(vc, sb, anchor)[:2]
    r['v_b2'] = bc(vctx[2] * 16 + vb32, 648 * 16).reshape(648, 16)
    pos = np.arange(16, dtype=np.int32)[None, None, :]
    r['v_b2p'] = bc((vctx[2] * 16 + pos) * 16 + vb32, 648 * 16 * 16).reshape(648 * 16, 16)
    r['v_o0'] = bc(vb32, 16)
    # block max / doubled structure
    r['bmax'] = bc(sb.astype(np.int32) * 8 + bmax, 256 * 8).reshape(256, 8)
    m7 = bmax == 7
    n13 = ((mag == 1) | (mag == 3)).sum(2)
    r['n13'] = bc(n13[m7], 17)
    r['magmax7'] = bc(mag[m7], 8)
    zl = codes.reshape(R, nb, 16)
    zs = (sv == 0)
    r['zsgn'] = np.array([[int((zl[~zs] == 0).sum()), int((zl[~zs] == 8).sum())],
                          [int((zl[zs] == 0).sum()), int((zl[zs] == 8).sum())]], np.int64)
    # scales
    s32 = sb.astype(np.int32)
    left = np.empty_like(s32); left[:, 0] = 256; left[:, 1:] = s32[:, :-1]
    up = np.empty_like(s32); up[0] = 256; up[1:] = s32[:-1]
    r['s_o0'] = bc(s32, 256)
    r['s_left'] = bc(left * 256 + s32, 257 * 256).reshape(257, 256)
    r['s_up'] = bc(up * 256 + s32, 257 * 256).reshape(257, 256)
    lq = np.where(left == 256, 15, np.clip(left - anchor, -7, 7) + 7)
    uq = np.where(up == 256, 15, np.clip(up - anchor, -7, 7) + 7)
    r['s_lu'] = bc((lq * 16 + uq) * 256 + s32, 256 * 256).reshape(256, 256)
    parr = np.broadcast_to((np.arange(s32.shape[1]) & 1)[None, :], s32.shape)
    r['s_par'] = bc((parr * 257 + left) * 256 + s32, 514 * 256).reshape(514, 256)
    r['meta'] = np.array([R, C, anchor, int(live.sum()), int((sv == 0).sum()), int((codes == 0).sum()),
                          int((codes == 8).sum()), int((bmax == 0).sum())], np.int64)
    return r


def main():
    d, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    man = json.load(open(os.path.join(d, 'manifest.json')))
    t0 = time.time()
    tsc = []
    for i, it in enumerate(man['experts']):
        p = os.path.join(out, it['tag'] + '.npz')
        mats = list(nc.matrices(d, it))
        tsc.append({'tag': it['tag'], 'tscale': [m[3] for m in mats], 'iscale': [m[4] for m in mats],
                    'raw_bytes': sum(m[5] for m in mats)})
        if os.path.exists(p):
            continue
        res = {}
        for r, codes, sb, ts, isc, nb in mats:
            for k, v in matrix_counts(codes, sb).items():
                res.setdefault(k, []).append(v)
        np.savez_compressed(p + '.tmp.npz', **{k: np.stack(v) for k, v in res.items()})
        os.replace(p + '.tmp.npz', p)
        if i % 10 == 0:
            print(i, it['tag'], round(time.time() - t0, 1), 's', flush=True)
    json.dump({'repo': man['repo'], 'revision': man['revision'], 'items': tsc}, open(os.path.join(out, 'tensor_scales.json'), 'w'), indent=1)
    print('done', round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
