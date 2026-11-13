"""headroom / feature extraction on ceil-form V4.1 experts (anchor 120).

Per weight a uint64 word W and per 32-block a uint64 word B of decoder-known (causal) fields, see FIELDS.
Decode-order assumptions (WNR2, REF:200-205 + team-4 wnr_common.py:26-38): all scale streams are decoded before
any weight stream, so every scale of all 3 matrices is known when weights are decoded; weight streams are 2048
weights = 64 blocks of the row-major matrix and are decoded independently, so a weight may only use symbols of its
own stream: in-block history, and the previous block of the same row when it is in the same stream.
'pmag'/'pbmx' (w1 weights used for w3) additionally need w1 weight streams decoded before w3's (phase split).

Usage: python -I hfeat.py SAMPLE_DIR OUT_DIR ROW_STRIDE [workers]
  writes OUT_DIR/<expert>.npz with W0,B0,W1,B1,W2,B2 for the rows r with (r + idx) % ROW_STRIDE == 0
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxcount import load_expert
from mxceil import forward

ANCHOR = 120
BPS = 64  # blocks per weight stream (2048 weights)

# (word, name, shift, bits)
FIELDS = [
    ('W', 'sym', 0, 4), ('W', 'pm', 4, 4), ('W', 'mx', 8, 4), ('W', 'pm2', 12, 4), ('W', 'pos', 16, 5),
    ('W', 'psign', 21, 2), ('W', 'nz', 23, 4), ('W', 'run', 27, 3), ('W', 'seen13', 30, 1), ('W', 'msum', 31, 3),
    ('W', 'nmax', 34, 2), ('W', 'pmag', 36, 4),
    ('B', 'srel15', 0, 4), ('B', 'bpar', 4, 1), ('B', 'bcolq', 5, 2), ('B', 'pbmax', 7, 2), ('B', 'pbsum', 9, 3),
    ('B', 'pbzero', 12, 3), ('B', 'dl', 15, 3), ('B', 'dr', 18, 3), ('B', 'du', 21, 3), ('B', 'dd', 24, 3),
    ('B', 'rowp', 27, 3), ('B', 'colp', 30, 3), ('B', 'mlev', 33, 3), ('B', 'pbmx', 36, 2), ('B', 'pscale', 38, 3),
    ('B', 'rowrel', 41, 3),
]
FIELD = {f[1]: f for f in FIELDS}
PBSUM_T = np.array([62, 68, 76, 90, 104, 114])  # block magnitude-sum thresholds (bimodal: halved blocks ~60-75)


def to_sym(wraw, R):
    wb = wraw.reshape(R, -1)
    C = wb.shape[1] * 2
    sym = np.empty((R, C), np.uint8)
    sym[:, 0::2] = wb & 15
    sym[:, 1::2] = wb >> 4
    return sym


def ceil_mats(t):
    """Ceil-form symbols [R,C] uint8 and scales [R,nb] int16 of the 3 matrices (order w1, w2, w3)."""
    out = []
    for w in ('w1', 'w2', 'w3'):
        wraw, wshape, _ = t[f'{w}.weight']
        sraw, _, _ = t[f'{w}.scale']
        R = wshape[0]
        sym = to_sym(wraw, R)
        sc = sraw.reshape(R, -1)
        tsym, tsc, even, bmax = forward(sym, sc)
        assert int(bmax.min()) >= 6 and int(tsc.max()) <= 254
        out.append((tsym, tsc.astype(np.int16)))
    return out


def weight_fields(blk, pmagblk=None):
    """blk uint8 [n, nb, 32] ceil-form symbols. Returns uint64 word per weight (fields of FIELDS 'W')."""
    mag = (blk & 7).astype(np.uint8)
    sh = mag.shape
    pm = np.full(sh, 8, np.uint8); pm[..., 1:] = mag[..., :-1]
    pm2 = np.full(sh, 8, np.uint8); pm2[..., 2:] = mag[..., :-2]
    cm = np.maximum.accumulate(mag, axis=2)
    mx = np.full(sh, 8, np.uint8); mx[..., 1:] = cm[..., :-1]
    pos = np.broadcast_to(np.arange(32, dtype=np.uint8), sh)
    psign = np.zeros(sh, np.uint8); psign[..., 1:] = 1 + (blk[..., :-1] >> 3)
    zc = np.cumsum(mag == 0, axis=2, dtype=np.int16)
    nz = np.zeros(sh, np.uint8); nz[..., 1:] = np.minimum(zc[..., :-1], 15)
    run = np.zeros(sh, np.uint8)
    run[..., 1] = 1
    for j in range(2, 32):
        run[..., j] = np.where(mag[..., j - 1] == mag[..., j - 2], np.minimum(run[..., j - 1] + 1, 7), 1)
    s13 = ((mag == 1) | (mag == 3))
    seen13 = np.zeros(sh, np.uint8); seen13[..., 1:] = np.maximum.accumulate(s13, axis=2)[..., :-1]
    cs = np.cumsum(mag, axis=2, dtype=np.int16)
    sumb = np.zeros(sh, np.int16); sumb[..., 1:] = cs[..., :-1]
    jj = np.arange(32, dtype=np.int16); jj[0] = 1
    msum = (np.clip((2 * sumb) // jj, 2, 8) - 2).astype(np.uint8); msum[..., 0] = 7
    nmax = np.zeros(sh, np.uint8)
    c = np.zeros(sh[:2], np.int16); cmx = np.full(sh[:2], -1, np.int16)
    for j in range(1, 32):
        v = mag[..., j - 1].astype(np.int16)
        c = np.where(v > cmx, 1, np.where(v == cmx, c + 1, c))
        cmx = np.maximum(cmx, v)
        nmax[..., j] = np.minimum(c, 3)
    pmag = np.full(sh, 8, np.uint8) if pmagblk is None else (pmagblk & 7).astype(np.uint8)
    W = blk.astype(np.uint64)
    for name, arr in (('pm', pm), ('mx', mx), ('pm2', pm2), ('pos', pos), ('psign', psign), ('nz', nz), ('run', run),
                      ('seen13', seen13), ('msum', msum), ('nmax', nmax), ('pmag', pmag)):
        W |= arr.astype(np.uint64) << np.uint64(FIELD[name][2])
    return W.ravel()


def dclip(a, b, valid):
    d = (np.clip(a - b, -2, 2) + 2).astype(np.uint8)
    return np.where(valid, d, 5).astype(np.uint8)


def block_fields(sym, S, rows, partner=None):
    """sym [R,C] ceil-form, S [R,nb] int16 ceil-form scales, rows: sampled row indices.
    partner: (sym_p, S_p, use_weights) of the up/gate partner (w3 for w1, w1 for w3) or None for w2."""
    R, C = sym.shape
    nb = C // 32
    magF = (sym.reshape(R, nb, 32) & 7)
    bmax = magF.max(2).astype(np.int16)
    bsum = magF.sum(2, dtype=np.int16)
    bzero = (magF == 0).sum(2, dtype=np.int16)
    r = rows[:, None].astype(np.int64)
    c = np.arange(nb, dtype=np.int64)[None, :]
    s = S[rows]
    srel15 = (np.clip(s - ANCHOR, -7, 7) + 7).astype(np.uint8)
    bpar = (c % 2).astype(np.uint8) + np.zeros_like(srel15)
    bcolq = (c * 4 // nb).astype(np.uint8) + np.zeros_like(srel15)
    pvalid = (c > 0) & (((r * nb + c) % BPS) != 0)
    prev = np.maximum(c - 1, 0)
    pbmax = np.where(pvalid, bmax[rows][:, prev[0]] - 5, 3).astype(np.uint8)
    pbsum = np.where(pvalid, np.searchsorted(PBSUM_T, bsum[rows][:, prev[0]], side='right'), 7).astype(np.uint8)
    pbzero = np.where(pvalid, np.minimum(bzero[rows][:, prev[0]], 6), 7).astype(np.uint8)
    cl = np.maximum(c - 1, 0)[0]; cr = np.minimum(c + 1, nb - 1)[0]
    ru = np.maximum(rows - 1, 0); rd = np.minimum(rows + 1, R - 1)
    dl = dclip(S[rows][:, cl], s, c > 0)
    dr = dclip(S[rows][:, cr], s, c < nb - 1)
    du = dclip(S[ru], s, (r > 0))
    dd = dclip(S[rd], s, (r < R - 1))
    Sf = S.astype(np.float64)
    mm = Sf.mean()
    rm = Sf.mean(1)[rows][:, None]
    cmn = Sf.mean(0)[None, :]
    rowp = np.clip(np.floor((rm - ANCHOR - 0.4) * 10), 0, 7).astype(np.uint8) + np.zeros_like(srel15)
    rowrel = np.clip(np.floor((rm - mm) / 0.05) + 4, 0, 7).astype(np.uint8) + np.zeros_like(srel15)
    colp = np.clip(np.floor((cmn - mm) / 0.01) + 4, 0, 7).astype(np.uint8) + np.zeros_like(srel15)
    mlev = np.uint8(np.clip(np.floor((mm - ANCHOR - 0.5) * 10), 0, 7)) + np.zeros_like(srel15)
    if partner is None:
        pbmx = np.full_like(srel15, 3)
        pscale = np.full_like(srel15, 5)
    else:
        psym, pS, use_w = partner
        pscale = dclip(pS[rows], s, np.ones_like(s, bool))
        if use_w:
            pb = (psym.reshape(R, nb, 32) & 7).max(2).astype(np.int16)
            pbmx = (pb[rows] - 5).astype(np.uint8)
        else:
            pbmx = np.full_like(srel15, 3)
    B = np.zeros(srel15.shape, np.uint64)
    for name, arr in (('srel15', srel15), ('bpar', bpar), ('bcolq', bcolq), ('pbmax', pbmax), ('pbsum', pbsum),
                      ('pbzero', pbzero), ('dl', dl), ('dr', dr), ('du', du), ('dd', dd), ('rowp', rowp),
                      ('colp', colp), ('mlev', mlev), ('pbmx', pbmx), ('pscale', pscale), ('rowrel', rowrel)):
        assert arr.shape == srel15.shape, name
        B |= arr.astype(np.uint64) << np.uint64(FIELD[name][2])
    return B.ravel()


def expert_words(t, rowsel):
    """rowsel(m, R) -> sorted row indices. Returns [(W, B) for m in 0..2]."""
    mats = ceil_mats(t)
    out = []
    for m in range(3):
        sym, S = mats[m]
        R, C = sym.shape
        nb = C // 32
        rows = rowsel(m, R)
        partner = None
        pmagblk = None
        if m == 0:
            partner = (mats[2][0], mats[2][1], False)      # w1: w3 scales known, w3 weights not yet
        elif m == 2:
            partner = (mats[0][0], mats[0][1], True)       # w3: w1 scales and (phase split) w1 weights
            pmagblk = mats[0][0][rows].reshape(len(rows), nb, 32)
        B = block_fields(sym, S, rows, partner)
        W = weight_fields(sym[rows].reshape(len(rows), nb, 32), pmagblk)
        out.append((W, B))
    return out


def get(word, name):
    _, _, sh, bits = FIELD[name]
    return ((word >> np.uint64(sh)) & np.uint64((1 << bits) - 1)).astype(np.uint8)


def do_one(args):
    sample_dir, out_dir, idx, e, stride = args
    outp = os.path.join(out_dir, e['file'].replace('.bin', '.npz'))
    if os.path.exists(outp):
        return e['file'], 0.0
    t0 = time.time()
    t = load_expert(os.path.join(sample_dir, e['file']), e['tensors'])
    res = expert_words(t, lambda m, R: np.arange((-idx) % stride, R, stride))
    d = {}
    for m, (W, B) in enumerate(res):
        d[f'W{m}'] = W; d[f'B{m}'] = B
    np.savez(outp + '.tmp.npz', **d)
    os.replace(outp + '.tmp.npz', outp)
    return e['file'], time.time() - t0


def main():
    sample_dir, out_dir, stride = sys.argv[1], sys.argv[2], int(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    jobs = [(sample_dir, out_dir, i, e, stride) for i, e in enumerate(man['experts'])]
    t0 = time.time()
    if workers <= 1:
        for i, j in enumerate(jobs):
            f, dt = do_one(j)
            print(i, f, round(dt, 1), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            for i, (f, dt) in enumerate(pool.map(do_one, jobs)):
                if i % 10 == 0:
                    print(f'{i + 1}/{len(jobs)} {f} {dt:.1f}s elapsed {time.time() - t0:.0f}s', flush=True)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
