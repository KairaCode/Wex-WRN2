"""Held-out conditional entropies of the coded nibbles and scales under many context features, on real records.

Method: for each (record, matrix) the rows are split by parity; counts of (context, symbol) from one half
(KT add-0.5 smoothing) score the other half, both ways (2-fold). The result is a held-out code length in bits per
symbol, so a feature only "helps" if its gain survives table estimation on half a matrix (an UPPER bound on the
true conditional entropy given that feature; the in-sample plug-in value would be a lower, biased one).
MI(feature) = H(base) - H(base + feature), held-out, in bit/weight (or bit/scale).

  python3 mi.py weights L:E,...      -> out/mi_weights.json
  python3 mi.py scales  L:E,...      -> out/mi_scales.json
Side records (other experts of the same layer) are taken from the 121-record sample."""
import sys, json, os
import numpy as np
from load import load, MAN, H as HERE

def ho_entropy(ctx, x, A, rows_parity):
    """held-out 2-fold cross entropy, bits/symbol. ctx int64 array, x int array (values < A)."""
    ctx = ctx.astype(np.int64); x = x.astype(np.int64)
    _, cid = np.unique(ctx, return_inverse=True)
    K = cid.max() + 1
    total = 0.0
    for f in (0, 1):
        tr = rows_parity == f; te = ~tr
        cnt = np.bincount(cid[tr] * A + x[tr], minlength=K * A).reshape(K, A).astype(np.float64) + 0.5
        p = cnt / cnt.sum(1, keepdims=True)
        total += -np.log2(p[cid[te], x[te]]).sum()
    return total / x.size

def clampi(v, lo, hi):
    return np.clip(v.astype(np.int64), lo, hi) - lo

def qbins(v, n):
    """quantile bins of a float array -> 0..n-1"""
    edges = np.quantile(v, np.linspace(0, 1, n + 1)[1:-1])
    return np.searchsorted(edges, v, side="right")

def base_ctx(s, q, anchor):
    """WNR3 context per nibble (567), plus srel per nibble."""
    R, C = q.shape; nb = C // 32
    a = (q & 7).reshape(R, nb, 32).astype(np.int64)
    srel = np.clip(s.astype(np.int64) - anchor, -3, 3) + 3                        # R, nb
    pm = np.empty_like(a); pm[:, :, 0] = 8; pm[:, :, 1:] = a[:, :, :-1]
    mx = np.empty_like(a); mx[:, :, 0] = 8; mx[:, :, 1:] = np.maximum.accumulate(a, axis=2)[:, :, :-1]
    ctx = (srel[:, :, None] * 9 + pm) * 9 + mx
    return ctx.reshape(R, C), np.repeat(srel, 32, axis=1)

def weights(recs):
    res = []
    for (l, e) in recs:
        d, anc, _ = load(l, e)
        others = [k for k in MAN if k[0] == l and k[1] != e and k != (39, 120)]
        od = load(*others[0])[0] if others else None
        for m in (0, 1, 2):
            s, q = d[m]; R, C = q.shape
            ctx, srel = base_ctx(s, q, anc[m])
            par = np.broadcast_to((np.arange(R) % 2)[:, None], (R, C)).ravel()
            x = q.ravel().astype(np.int64); mag = x & 7
            B = ctx.ravel()
            row = {"layer": l, "expert": e, "m": m, "n": int(x.size)}
            row["order0"] = ho_entropy(np.zeros_like(B), x, 16, par)
            row["srel"] = ho_entropy(srel.ravel(), x, 16, par)
            row["base567"] = hb = ho_entropy(B, x, 16, par)
            def add(name, f, k):
                row[name] = hb - ho_entropy(B * k + f.ravel().astype(np.int64), x, 16, par)
            pos = np.broadcast_to(np.arange(C) % 32, (R, C))
            add("pos32", pos, 32)
            # column importance: mean magnitude of column j over the OTHER parity half (so it is not fit to the
            # scored row; leave-half-out), relative to its block's srel via the block-normalised magnitude
            amag = (q & 7).astype(np.float64)
            colm = np.stack([amag[(np.arange(R) % 2) == f].mean(0) for f in (0, 1)])  # 2, C
            colf = colm[1 - (np.arange(R) % 2)]                                        # R, C : other half
            add("col_mean_q8(other half)", qbins(colf.ravel(), 8).reshape(R, C), 8)
            rowm = amag.mean(1)
            add("row_mean_q8(own row)", np.broadcast_to(qbins(rowm, 8)[:, None], (R, C)), 8)
            # scale structure: srel of the row anchor (row median scale - anchor), neighbouring blocks
            rowmed = np.median(s, axis=1)
            add("row_anchor5", np.broadcast_to(clampi(np.round(rowmed - anc[m]), -2, 2)[:, None], (R, C)), 5)
            if m in (0, 2):
                ss, sq = d[2 - m]
                sctx, ssrel = base_ctx(ss, sq, anc[2 - m])
                add("sib_mag8", sq & 7, 8)
                add("sib_nib16", sq, 16)
                add("sib_srel7", ssrel, 7)
                add("sib_srel7_x_mag8", ssrel * 8 + (sq & 7), 56)
                # sign agreement among nonzero pairs
                nz = (mag > 0) & ((sq.ravel() & 7) > 0)
                sg = (x >> 3)[nz]; ssg = (sq.ravel().astype(np.int64) >> 3)[nz]
                p_agree = float((sg == ssg).mean())
                row["sib_sign_agree"] = p_agree
                pa = p_agree; hs = -(pa * np.log2(pa) + (1 - pa) * np.log2(1 - pa))
                row["sib_sign_MI_per_nonzero_pair"] = 1 - hs
                row["sib_sign_MI_bit_per_weight"] = (1 - hs) * nz.mean()
                scol = (sq & 7).astype(np.float64).mean(0)
                add("sib_col_mean_q8", np.broadcast_to(qbins(scol, 8)[None, :], (R, C)), 8)
                srow = (sq & 7).astype(np.float64).mean(1)
                add("sib_row_mean_q8", np.broadcast_to(qbins(srow, 8)[:, None], (R, C)), 8)
            if m == 1:
                # w2 is [hidden, inter]; its column i is neuron i = row i of w1/w3
                n1 = (d[0][1] & 7).astype(np.float64).mean(1) + (d[2][1] & 7).astype(np.float64).mean(1)
                add("w13_row_of_neuron_q8", np.broadcast_to(qbins(n1, 8)[None, :], (R, C)), 8)
            if od is not None:
                os_, oq = od[m]
                add("other_expert_mag8", oq & 7, 8)
                add("other_expert_nib16", oq, 16)
                ocol = (oq & 7).astype(np.float64).mean(0)
                add("other_expert_col_mean_q8", np.broadcast_to(qbins(ocol, 8)[None, :], (R, C)), 8)
                orow = (oq & 7).astype(np.float64).mean(1)
                add("other_expert_row_mean_q8", np.broadcast_to(qbins(orow, 8)[:, None], (R, C)), 8)
                row["other_expert"] = list(others[0])
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            res.append(row)
    return res

def scales(recs):
    res = []
    for (l, e) in recs:
        d, anc, _ = load(l, e)
        others = [k for k in MAN if k[0] == l and k[1] != e and k != (39, 120)]
        od = load(*others[0])[0] if others else None
        for m in (0, 1, 2):
            s, q = d[m]; R, NB = s.shape
            x = s.ravel().astype(np.int64)
            par = np.broadcast_to((np.arange(R) % 2)[:, None], (R, NB)).ravel()
            row = {"layer": l, "expert": e, "m": m, "n": int(x.size)}
            Z = np.zeros_like(x)
            row["order0"] = h0 = ho_entropy(Z, x, 256, par)
            def add(name, f, k=None):
                row[name] = h0 - ho_entropy(f.ravel().astype(np.int64), x, 256, par)
            si = s.astype(np.int64)
            left = np.empty_like(si); left[:, 0] = 300; left[:, 1:] = si[:, :-1]
            up = np.empty_like(si); up[0, :] = 300; up[1:, :] = si[:-1, :]
            add("left", left); add("up", up); add("left_up", left * 512 + up)
            left2 = np.empty_like(si); left2[:, :2] = 300; left2[:, 2:] = si[:, :-2]
            add("left_left2", left * 512 + left2)
            add("left_up_upleft", (left * 512 + up) * 512 + np.pad(si, ((1, 0), (1, 0)), constant_values=300)[:-1, :-1])
            # block max of the block's own coded nibbles (if nibbles came first)
            bm = (q & 7).reshape(R, NB, 32).max(2)
            add("own_blockmax", bm)
            # row / column statistics (leave-half-out for columns, own row for rows: the row median could be
            # sent as side info at R bytes)
            colmean = np.stack([si[(np.arange(R) % 2) == f].mean(0) for f in (0, 1)])[1 - (np.arange(R) % 2)]
            add("col_mean_q8(other half)", qbins(colmean.ravel(), 8).reshape(R, NB))
            rmed = np.round(np.median(si, axis=1)).astype(np.int64)
            add("row_median(side info)", np.broadcast_to(rmed[:, None], (R, NB)))
            add("row_median+left", rmed[:, None] * 512 + left)
            add("running_row_max_left", np.maximum.accumulate(np.where(left == 300, 0, left), axis=1) * 512 + left)
            if m in (0, 2):
                ss = d[2 - m][0].astype(np.int64)
                add("sib_scale", ss)
                add("sib_scale+left", ss * 512 + left)
                add("sib_scale+left+up", (ss * 512 + left) * 512 + up)
            if m == 1:
                # w2 block (r, c) covers neurons 32c..32c+31 = rows of w1/w3; their mean scale, 8 bins
                n1 = (d[0][0].astype(np.float64).mean(1) + d[2][0].astype(np.float64).mean(1)).reshape(NB, 32).mean(1)
                add("w13_scale_of_neuron_group_q8", np.broadcast_to(qbins(n1, 8)[None, :], (R, NB)))
            if od is not None:
                add("other_expert_scale", od[m][0].astype(np.int64))
                add("other_expert_scale+left", od[m][0].astype(np.int64) * 512 + left)
                ocol = od[m][0].astype(np.float64).mean(0)
                add("other_expert_col_mean_q8", np.broadcast_to(qbins(ocol, 8)[None, :], (R, NB)))
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            res.append(row)
    return res

if __name__ == "__main__":
    what = sys.argv[1]
    recs = [tuple(map(int, t.split(":"))) for t in sys.argv[2].split(",")]
    res = weights(recs) if what == "weights" else scales(recs)
    json.dump(res, open(os.path.join(HERE, "out", "mi_%s.json" % what), "w"), indent=0)
