"""Follow-up: the per-matrix tilt correlates 0.98 with the matrix's mean clamped srel.
Test tables selected/interpolated by a DECODER-COMPUTABLE per-matrix statistic (zero side info):
  f_m = mean over the matrix's blocks of clamp(s - lower_median, -3, 3)   (scales are decoded before weights)
Held-out as in tableadapt.py (split = index % 2, both ways). q14 = 14-bit quantized CDFs, flush included.

Usage: python -I tableadapt2.py SAMPLE_DIR COUNTS_DIR OUT_JSON
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tableadapt import quantize, load, P, W_MAT, FLUSH_BPW, tilt


def lq(c):
    return -np.log2(quantize(c) / P)


def main():
    sample_dir, counts_dir, out_json = sys.argv[1:4]
    U, lay, split = load(sample_dir, counts_dir)
    n = U.shape[0]
    Wtot = n * 3 * W_MAT
    bpw = lambda b: b / Wtot + FLUSH_BPW
    srel_w = U.reshape(n, 3, 7, 81, 16).sum((3, 4)).astype(np.float64)  # weights per srel
    f = (srel_w * (np.arange(7) - 3)).sum(-1) / srel_w.sum(-1)  # [n,3]
    out = {'f_stats': {'mean': float(f.mean()), 'std': float(f.std()), 'min': float(f.min()), 'max': float(f.max()),
                       'per_type_mean': f.mean(0).tolist(), 'last3_layers_mean': float(f[lay >= 37].mean()),
                       'others_mean': float(f[lay < 37].mean())}}
    print(out['f_stats'], flush=True)

    base = 0.0
    for tr in (0, 1):
        A = U[split == tr].sum(0)
        L = lq(A)
        base += sum(float((U[i].astype(np.float64) * L).sum()) for i in np.where(split != tr)[0])
    out['base'] = bpw(base)
    res = {}

    def rep(name, b, side=0.0):
        res[name] = {'bit_per_w': bpw(b + side), 'delta_pct': 100 * (b + side - base) / (base + FLUSH_BPW * Wtot)}
        print(f'{name:60s} {bpw(b + side):.5f} {res[name]["delta_pct"]:+.4f}%', flush=True)

    # 1. bucket by f (quantile edges from the training split), per matrix type or shared over types
    for B in (2, 3, 4, 6, 8):
        for shared in (False, True):
            tb = 0.0
            for tr in (0, 1):
                itr = np.where(split == tr)[0]; ite = np.where(split != tr)[0]
                ftr = f[itr].ravel()
                edges = np.quantile(ftr, np.linspace(0, 1, B + 1)[1:-1])
                btr = np.searchsorted(edges, f[itr])  # [ntr,3]
                bte = np.searchsorted(edges, f[ite])
                if shared:
                    Ls = []
                    for b in range(B):
                        c = sum(U[i, m] for k, i in enumerate(itr) for m in range(3) if btr[k, m] == b)
                        Ls.append(lq(c))
                    for k, i in enumerate(ite):
                        for m in range(3):
                            tb += float((U[i, m] * Ls[bte[k, m]]).sum())
                else:
                    for m in range(3):
                        Ls = []
                        for b in range(B):
                            c = sum(U[i, m] for k, i in enumerate(itr) if btr[k, m] == b)
                            Ls.append(lq(c))
                        for k, i in enumerate(ite):
                            tb += float((U[i, m] * Ls[bte[k, m]]).sum())
            rep(f'f-bucket B={B} {"shared over types" if shared else "per type"} (0 side bits)', tb)

    # 2. f-bucket (per type, B=4) + K=2 clusters inside each bucket (1 bit per matrix)
    # 3. f-bucket + per-matrix srel half-step shift (1 B) + per-matrix mag tilt (1 B)
    for B in (4,):
        tb_s = tb_st = tb_k = 0.0
        ag = np.linspace(-1, 1, 41)
        tg = np.linspace(-0.2, 0.2, 41)
        for tr in (0, 1):
            itr = np.where(split == tr)[0]; ite = np.where(split != tr)[0]
            edges = np.quantile(f[itr].ravel(), np.linspace(0, 1, B + 1)[1:-1])
            btr = np.searchsorted(edges, f[itr]); bte = np.searchsorted(edges, f[ite])
            for m in range(3):
                for b in range(B):
                    sel_tr = [i for k, i in enumerate(itr) if btr[k, m] == b]
                    sel_te = [i for k, i in enumerate(ite) if bte[k, m] == b]
                    if not sel_te:
                        continue
                    c = sum(U[i, m] for i in sel_tr)
                    p = quantize(c) / P
                    t3 = p.reshape(7, 81, 16)
                    up = np.concatenate([t3[1:], t3[-1:]], 0).reshape(567, 16)
                    dn = np.concatenate([t3[:1], t3[:-1]], 0).reshape(567, 16)
                    fam = np.stack([(1 - abs(a)) * p + abs(a) * (up if a > 0 else dn) for a in ag])  # [A,567,16]
                    lf = -np.log2(fam)
                    fam2 = np.stack([tilt(fa, (np.arange(16) & 7).astype(float), tg) for fa in fam])  # [A,T,567,16]
                    lf2 = -np.log2(fam2)
                    for i in sel_te:
                        x = U[i, m].astype(np.float64)
                        tb_s += np.einsum('cs,acs->a', x, lf).min()
                        tb_st += np.einsum('cs,atcs->at', x, lf2).min()
                    # K=2 clusters inside the bucket
                    Ut = np.stack([U[i, m] for i in sel_tr])
                    assign = np.arange(len(sel_tr)) % 2
                    for _ in range(15):
                        Lm = np.stack([lq(Ut[assign == k].sum(0)) if (assign == k).any() else lq(np.zeros((567, 16), np.int64)) for k in range(2)])
                        new = np.einsum('ucs,kcs->uk', Ut.astype(np.float64), Lm).argmin(1)
                        if (new == assign).all():
                            break
                        assign = new
                    Te = np.stack([U[i, m] for i in sel_te]).astype(np.float64)
                    tb_k += np.einsum('ucs,kcs->uk', Te, Lm).min(1).sum()
        rep(f'f-bucket B={B} per type + srel shift (1 B, float)', tb_s, 8 * 3 * n)
        rep(f'f-bucket B={B} per type + srel shift + mag tilt (2 B, float)', tb_st, 16 * 3 * n)
        rep(f'f-bucket B={B} per type + K=2 clusters inside (1 bit)', tb_k, 3 * n)

    # 4. last-3-layers flag (layer known to the engine) + f-bucket
    tb = 0.0
    for tr in (0, 1):
        itr = np.where(split == tr)[0]; ite = np.where(split != tr)[0]
        edges = np.quantile(f[itr].ravel(), [0.25, 0.5, 0.75])
        for m in range(3):
            for late in (0, 1):
                for b in range(4):
                    sel_tr = [i for i in itr if (lay[i] >= 37) == late and np.searchsorted(edges, f[i, m]) == b]
                    sel_te = [i for i in ite if (lay[i] >= 37) == late and np.searchsorted(edges, f[i, m]) == b]
                    if not sel_te:
                        continue
                    if not sel_tr:
                        sel_tr = [i for i in itr if np.searchsorted(edges, f[i, m]) == b]
                    L = lq(sum(U[i, m] for i in sel_tr))
                    tb += sum(float((U[i, m] * L).sum()) for i in sel_te)
    rep('f-bucket B=4 per type x (layer>=37) (0 side bits)', tb)
    out['variants'] = res
    json.dump(out, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
