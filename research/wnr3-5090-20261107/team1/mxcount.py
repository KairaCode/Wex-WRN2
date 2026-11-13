"""Pass 1: per-expert symbol counts under WNR2's context rule and candidate variants.

Usage: python -I mxcount.py SAMPLE_DIR OUT_DIR [workers]
For every expert file in SAMPLE_DIR/manifest.json writes OUT_DIR/<name>.npz with int64 counts:
  w_base   [3, 567, 16]        WNR2 context (srel7, pm9, mx9) x symbol      (decode order m0=w1, m1=w2, m2=w3)
  w_srel15 [3, 15*81, 16]      srel clamped to +-7 instead of +-3
  w_pos    [3, 567*32, 16]     base x position in the 32-block
  w_pm2    [3, 567*9, 16]      base x magnitude two symbols back (8 = none)
  w_psign  [3, 567*3, 16]      base x sign of previous symbol (0 none, 1 +, 2 -)
  w_bm     [3, 567*8, 16]      base x FINAL block max magnitude (side info coded separately)
  bm_side  [3, 7, 8]           srel7 x final block max magnitude (cost of coding the block max)
  w_bmpos  [3, 567*8*4, 16]    base x final block max x position quartile
  w_up     [3, 567*9, 16]      base x magnitude at the same column in the previous row (8 = row 0)
  w_upsort [3, 567*9, 16]      same, after sorting the rows of each matrix by (mean scale, row index)
  w_part   [567*9, 16]         w3 only: base x magnitude of w1 at the same position
  s_o0     [3, 256]            scales order-0
  s_left   [3, 257, 256]       scale given the previous scale in the row (256 = none)
  s_up     [3, 257, 256]       scale given the scale above (same block column, previous row)
  s_lu     [3, 16*16, 256]     scale given (left-med, up-med) each clamped to -7..7, 15 = none
  s_lusort [3, 16*16, 256]     s_lu after the row sort above
  meta     [3, 4]              rows, cols, median scale, count of -0 codes (symbol 8)
"""
import json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ORDER = ['w1.scale', 'w2.scale', 'w3.scale', 'w1.weight', 'w2.weight', 'w3.weight']


def load_expert(path, tensors):
    raw = np.fromfile(path, dtype=np.uint8)
    out, off = {}, 0
    for t in tensors:
        n = t['bytes']
        out[t['name']] = (raw[off:off + n], t['shape'], t['dtype'])
        off += n
    assert off == raw.size, (path, off, raw.size)
    return out


def bc(idx, n):
    return np.bincount(idx.ravel(), minlength=n)[:n]


def matrix_stats(wb, sc):
    """wb: uint8 [R, C/2] packed E2M1 (low nibble = even column); sc: uint8 [R, C/32] UE8M0."""
    R = wb.shape[0]
    C = wb.shape[1] * 2
    nb = C // 32
    assert sc.shape == (R, nb), (sc.shape, R, nb)
    sym = np.empty((R, C), np.uint8)
    sym[:, 0::2] = wb & 15
    sym[:, 1::2] = wb >> 4
    blk = sym.reshape(R, nb, 32)
    mag = blk & 7
    pm = np.empty_like(mag); pm[..., 0] = 8; pm[..., 1:] = mag[..., :-1]
    pm2 = np.empty_like(mag); pm2[..., :2] = 8; pm2[..., 2:] = mag[..., :-2]
    cm = np.maximum.accumulate(mag, axis=2)
    mx = np.empty_like(mag); mx[..., 0] = 8; mx[..., 1:] = cm[..., :-1]
    bm = cm[..., -1]  # final block max magnitude, [R, nb]
    sgn = np.empty_like(mag); sgn[..., 0] = 0; sgn[..., 1:] = 1 + (blk[..., :-1] >> 3)
    s16 = sc.astype(np.int16)
    med = int(np.sort(s16.ravel())[(s16.size - 1) // 2])
    srel = (np.clip(s16 - med, -3, 3) + 3).astype(np.int32)[..., None]
    srel15 = (np.clip(s16 - med, -7, 7) + 7).astype(np.int32)[..., None]
    pm32 = pm.astype(np.int32); mx32 = mx.astype(np.int32); b32 = blk.astype(np.int32)
    base = (srel * 9 + pm32) * 9 + mx32  # [R, nb, 32]
    pos = np.arange(32, dtype=np.int32)[None, None, :]
    up = np.empty_like(mag); up[0] = 8; up[1:] = mag[:-1]
    order = np.lexsort((np.arange(R), s16.mean(axis=1)))
    mag_s = mag[order]
    up_s = np.empty_like(mag); up_s[0] = 8; up_s[1:] = mag_s[:-1]
    base_s = base[order]
    b_s = b32[order]
    r = {}
    r['w_base'] = bc(base * 16 + b32, 567 * 16).reshape(567, 16)
    r['w_srel15'] = bc(((srel15 * 9 + pm32) * 9 + mx32) * 16 + b32, 1215 * 16).reshape(1215, 16)
    r['w_pos'] = bc((base * 32 + pos) * 16 + b32, 567 * 32 * 16).reshape(567 * 32, 16)
    r['w_pm2'] = bc((base * 9 + pm2.astype(np.int32)) * 16 + b32, 567 * 9 * 16).reshape(567 * 9, 16)
    r['w_psign'] = bc((base * 3 + sgn.astype(np.int32)) * 16 + b32, 567 * 3 * 16).reshape(567 * 3, 16)
    bm32 = bm.astype(np.int32)[..., None]
    r['w_bm'] = bc((base * 8 + bm32) * 16 + b32, 567 * 8 * 16).reshape(567 * 8, 16)
    r['bm_side'] = bc(srel[..., 0] * 8 + bm.astype(np.int32), 56).reshape(7, 8)
    r['w_bmpos'] = bc(((base * 8 + bm32) * 4 + pos // 8) * 16 + b32, 567 * 32 * 16).reshape(567 * 32, 16)
    r['w_up'] = bc((base * 9 + up.astype(np.int32)) * 16 + b32, 567 * 9 * 16).reshape(567 * 9, 16)
    r['w_upsort'] = bc((base_s * 9 + up_s.astype(np.int32)) * 16 + b_s, 567 * 9 * 16).reshape(567 * 9, 16)
    # scales
    s32 = sc.astype(np.int32)
    left = np.empty_like(s32); left[:, 0] = 256; left[:, 1:] = s32[:, :-1]
    upv = np.empty_like(s32); upv[0] = 256; upv[1:] = s32[:-1]
    r['s_o0'] = bc(s32, 256)
    r['s_left'] = bc(left * 256 + s32, 257 * 256).reshape(257, 256)
    r['s_up'] = bc(upv * 256 + s32, 257 * 256).reshape(257, 256)

    def lu_ctx(s, lft, u):
        lq = np.where(lft == 256, 15, np.clip(lft - med, -7, 7) + 7)
        uq = np.where(u == 256, 15, np.clip(u - med, -7, 7) + 7)
        return bc((lq * 16 + uq) * 256 + s, 256 * 256).reshape(256, 256)

    r['s_lu'] = lu_ctx(s32, left, upv)
    ss = s32[order]
    lft_s = np.empty_like(ss); lft_s[:, 0] = 256; lft_s[:, 1:] = ss[:, :-1]
    up_ss = np.empty_like(ss); up_ss[0] = 256; up_ss[1:] = ss[:-1]
    r['s_lusort'] = lu_ctx(ss, lft_s, up_ss)
    r['meta'] = np.array([R, C, med, int((sym == 8).sum())], np.int64)
    return r, mag, base, b32


def do_expert(args):
    sample_dir, out_dir, ex = args
    outp = os.path.join(out_dir, ex['file'].replace('.bin', '.npz'))
    if os.path.exists(outp):
        return ex['file'], 0.0
    t0 = time.time()
    t = load_expert(os.path.join(sample_dir, ex['file']), ex['tensors'])
    res = {}
    keep = {}
    for m, w in enumerate(['w1', 'w2', 'w3']):
        wraw, wshape, wdt = t[f'{w}.weight']
        sraw, sshape, sdt = t[f'{w}.scale']
        R = wshape[0]
        wb = wraw.reshape(R, -1)
        sc = sraw.reshape(R, -1)
        r, mag, base, b32 = matrix_stats(wb, sc)
        if m == 0:
            keep['mag1'] = mag
        if m == 2:
            part = keep['mag1'].astype(np.int32)
            if part.shape == mag.shape:
                res['w_part'] = bc((base * 8 + part) * 16 + b32, 567 * 8 * 16).reshape(567 * 8, 16)
        for k, v in r.items():
            res.setdefault(k, []).append(v)
    out = {k: (np.stack(v) if isinstance(v, list) else v) for k, v in res.items()}
    np.savez_compressed(outp + '.tmp.npz', **out)
    os.replace(outp + '.tmp.npz', outp)
    return ex['file'], time.time() - t0


def main():
    sample_dir, out_dir = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    ex0 = man['experts'][0]
    print('tensors:', [(x['name'], x['dtype'], x['shape']) for x in ex0['tensors']], flush=True)
    jobs = [(sample_dir, out_dir, ex) for ex in man['experts']]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for i, (f, dt) in enumerate(pool.map(do_expert, jobs)):
            if i % 20 == 0:
                print(f'{i + 1}/{len(jobs)} {f} {dt:.1f}s elapsed {time.time() - t0:.0f}s', flush=True)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
