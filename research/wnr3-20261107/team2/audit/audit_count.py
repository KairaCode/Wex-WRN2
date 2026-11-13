"""team2/audit pass 1: independent symbol counts under the WNR2 context rule.

Written from the brief's rule only; does NOT import the scratchpad wnr2 scripts.
Rule: symbols row-major, symbol 2k = low nibble of byte k. Per 32-block with UE8M0 scale s:
  srel = clamp(s - lower_median(matrix scales), -3, 3) + 3, pm = 8, mx = 8 at block start,
  context = (srel*9 + pm)*9 + mx; after symbol x: pm = x&7; mx = x&7 if mx == 8 else max(mx, x&7).

Usage: python -I audit_count.py SAMPLE_DIR OUT_DIR [workers] [stride]
Per expert writes OUT_DIR/<file>.npz (int64 unless noted):
  J1      [3, 567*2*32, 16]   ((ctx*2 + bm7)*32 + pos) x symbol      bm7 = (final block max magnitude == 7)
  J2      [3, 567*2*2*9, 16]  (((ctx*2 + bm7)*2 + half)*9 + pm2) x symbol   pm2 = magnitude two back (8 = none)
  base_sw [3, 567, 16]        base context with the nibble order swapped (high nibble first) - sensitivity only
  bmhist  [3, 7, 8]           srel x final block max magnitude
  s_o0    [3, 256]            scale histogram
  meta    [3, 7]              R, C, med_lower, med_upper, count code0 (+0), count code8 (-0), sha_ok
"""
import hashlib, json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

NJ1 = 567 * 2 * 32
NJ2 = 567 * 2 * 2 * 9


def read_expert(sample_dir, ex):
    path = os.path.join(sample_dir, ex['file'])
    with open(path, 'rb') as f:
        data = f.read()
    sha_ok = hashlib.sha256(data).hexdigest() == ex['sha256']
    tens, off = {}, 0
    for t in ex['tensors']:
        n = int(t['bytes'])
        tens[t['name']] = (np.frombuffer(data, np.uint8, n, off), tuple(int(v) for v in t['shape']))
        off += n
    if off != len(data):
        raise ValueError(f'{path}: tensors cover {off} of {len(data)} bytes')
    return tens, sha_ok


def unpack(wbytes, low_first=True):
    sym = np.empty(wbytes.size * 2, np.uint8)
    lo, hi = wbytes & 0x0F, wbytes >> 4
    if low_first:
        sym[0::2], sym[1::2] = lo, hi
    else:
        sym[0::2], sym[1::2] = hi, lo
    return sym


def lower_median(v):
    v = np.sort(v.astype(np.int64))
    return int(v[(v.size - 1) // 2]), int(v[v.size // 2])


def block_contexts(S, srel):
    """S: uint8 [nblk, 32] symbols, srel: int32 [nblk]. Literal loop over the 32 positions.
    Returns ctx, pm2 as int32 [nblk, 32] and the final block max magnitude [nblk]."""
    nblk = S.shape[0]
    ST = np.ascontiguousarray(S.T)  # [32, nblk]
    ctx = np.empty((32, nblk), np.int32)
    pm2 = np.empty((32, nblk), np.int32)
    pm = np.full(nblk, 8, np.int32)
    prev2 = np.full(nblk, 8, np.int32)
    mx = np.full(nblk, 8, np.int32)
    for k in range(32):
        ctx[k] = (srel * 9 + pm) * 9 + mx
        pm2[k] = prev2
        m = (ST[k] & 7).astype(np.int32)
        prev2 = pm
        pm = m
        mx = np.where(mx == 8, m, np.maximum(mx, m))
    return np.ascontiguousarray(ctx.T), np.ascontiguousarray(pm2.T), mx  # mx = final block max magnitude


def do_expert(args):
    sample_dir, out_dir, ex = args
    outp = os.path.join(out_dir, ex['file'].replace('.bin', '.npz'))
    if os.path.exists(outp):
        return ex['file'], -1.0
    t0 = time.time()
    tens, sha_ok = read_expert(sample_dir, ex)
    J1 = np.zeros((3, NJ1, 16), np.int64)
    J2 = np.zeros((3, NJ2, 16), np.int64)
    base_sw = np.zeros((3, 567, 16), np.int64)
    bmhist = np.zeros((3, 7, 8), np.int64)
    s_o0 = np.zeros((3, 256), np.int64)
    meta = np.zeros((3, 7), np.int64)
    pos = np.arange(32, dtype=np.int32)[None, :]
    half = (pos >= 16).astype(np.int32)
    for m, w in enumerate(('w1', 'w2', 'w3')):
        wb, wshape = tens[f'{w}.weight']
        sc, sshape = tens[f'{w}.scale']
        R = wshape[0]
        C = wshape[1] * 2
        if sshape != (R, C // 32):
            raise ValueError(f'scale shape {sshape} vs weights {wshape}')
        med_lo, med_hi = lower_median(sc)
        srel = (np.clip(sc.astype(np.int32) - med_lo, -3, 3) + 3).astype(np.int32)
        sym = unpack(wb, True).reshape(-1, 32)
        assert sym.shape[0] == sc.size
        ctx, pm2, bmax = block_contexts(sym, srel)
        bm7 = (bmax == 7).astype(np.int32)[:, None]
        s32 = sym.astype(np.int32)
        k12 = ctx * 2 + bm7
        J1[m] = np.bincount((((k12 * 32 + pos) * 16) + s32).ravel(), minlength=NJ1 * 16).reshape(NJ1, 16)
        J2[m] = np.bincount(((((k12 * 2 + half) * 9 + pm2) * 16) + s32).ravel(), minlength=NJ2 * 16).reshape(NJ2, 16)
        bmhist[m] = np.bincount(srel * 8 + bmax, minlength=56).reshape(7, 8)
        s_o0[m] = np.bincount(sc, minlength=256)
        sw = unpack(wb, False).reshape(-1, 32)
        ctx_sw, _, _ = block_contexts(sw, srel)
        base_sw[m] = np.bincount((ctx_sw * 16 + sw.astype(np.int32)).ravel(), minlength=567 * 16).reshape(567, 16)
        meta[m] = [R, C, med_lo, med_hi, int((sym == 0).sum()), int((sym == 8).sum()), int(sha_ok)]
    np.savez_compressed(outp + '.tmp.npz', J1=J1, J2=J2, base_sw=base_sw, bmhist=bmhist, s_o0=s_o0, meta=meta)
    os.replace(outp + '.tmp.npz', outp)
    return ex['file'], time.time() - t0


def main():
    sample_dir, out_dir = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(sample_dir, 'manifest.json'), 'r', encoding='utf-8') as f:
        man = json.load(f)
    exs = man['experts'][::stride]
    jobs = [(sample_dir, out_dir, e) for e in exs]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for i, (fn, dt) in enumerate(pool.map(do_expert, jobs)):
            if i % 10 == 0:
                print(f'{i + 1}/{len(jobs)} {fn} {dt:.1f}s total {time.time() - t0:.0f}s', flush=True)
    print('done', len(jobs), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
