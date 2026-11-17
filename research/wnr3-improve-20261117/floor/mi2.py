"""Cross-tensor mutual information on every record of the sample (in-sample plug-in with Miller-Madow bias
correction; the conditioning is srel only, so the bias term is tiny: <= 1e-4 bit). Bits per weight (or scale).

  I(a3 ; a1 | srel3)            magnitudes of w3[i,j] and w1[i,j] (same position in the sibling matrix)
  I(q3 ; q1 | srel3)            full nibbles (magnitude + sign)
  I(sign3 ; sign1 | both != 0)  per weight
  I(a ; a_other | srel)         same position, another expert of the same layer (w1, w2, w3)
  I(s3 ; s1), I(s ; s_other)    scales, unconditioned
  I(a ; colbin | srel)          column mean magnitude (8 quantile bins) of the same matrix, other expert, sibling
  python3 mi2.py JOBS -> out/mi2.jsonl"""
import sys, json, os
import numpy as np
from multiprocessing import Pool
from load import load, MAN, H as HERE

def cmi(x, y, c, A, B, K):
    """plug-in I(X;Y|C) with Miller-Madow correction, bits."""
    n = x.size
    j = np.bincount((c.astype(np.int64) * A + x) * B + y, minlength=K * A * B).reshape(K, A, B).astype(np.float64)
    def H(t):
        t = t[t > 0]; return -(t * np.log2(t / n)).sum() / n, t.size
    hxc, nxc = H(j.sum(2)); hyc, nyc = H(j.sum(1)); hxyc, nxyc = H(j); hc, ncc = H(j.sum((1, 2)))
    mi = hxc + hyc - hxyc - hc
    corr = (nxyc - nxc - nyc + ncc) / (2 * n * np.log(2))   # MM bias of the plug-in CMI
    return mi - corr, mi

def qb(v, n=8):
    e = np.quantile(v, np.linspace(0, 1, n + 1)[1:-1]); return np.searchsorted(e, v, side="right")

def one(k):
    l, e = k
    d, anc, _ = load(l, e)
    oth = [o for o in sorted(MAN) if o[0] == l and o[1] != e and o != (39, 120)][0]
    od, oanc, _ = load(*oth)
    r = {"layer": l, "expert": e, "other": list(oth)}
    sr = {}
    for m in range(3):
        s, q = d[m]
        sr[m] = np.repeat(np.clip(s.astype(np.int64) - anc[m], -3, 3) + 3, 32, axis=1).ravel()
    a1 = (d[0][1] & 7).ravel().astype(np.int64); a3 = (d[2][1] & 7).ravel().astype(np.int64)
    q1 = d[0][1].ravel().astype(np.int64); q3 = d[2][1].ravel().astype(np.int64)
    r["I_a3_a1|srel3"] = cmi(a3, a1, sr[2], 8, 8, 7)[0]
    r["I_q3_q1|srel3"] = cmi(q3, q1, sr[2], 16, 16, 7)[0]
    r["I_a3_a1|srel3,srel1"] = cmi(a3, a1, sr[2] * 7 + sr[0], 8, 8, 49)[0]
    nz = (a1 > 0) & (a3 > 0)
    r["I_sign3_sign1_per_weight"] = cmi(q3[nz] >> 3, q1[nz] >> 3, np.zeros(nz.sum(), np.int64), 2, 2, 1)[0] * nz.mean()
    jt = np.bincount(sr[2] * 8 + a3, minlength=56).reshape(7, 8).astype(np.float64)
    r["H_a3|srel3"] = float(-(jt[jt > 0] * np.log2((jt / jt.sum(1, keepdims=True))[jt > 0])).sum() / a3.size)
    r["I_s3_s1"] = cmi(d[2][0].ravel().astype(np.int64), d[0][0].ravel().astype(np.int64), np.zeros(d[0][0].size, np.int64), 256, 256, 1)[0]
    for m in range(3):
        a = (d[m][1] & 7).astype(np.int64); ao = (od[m][1] & 7).astype(np.int64)
        R, C = a.shape
        r[f"I_a_aother|srel m{m}"] = cmi(a.ravel(), ao.ravel(), sr[m], 8, 8, 7)[0]
        r[f"I_s_sother m{m}"] = cmi(d[m][0].ravel().astype(np.int64), od[m][0].ravel().astype(np.int64), np.zeros(d[m][0].size, np.int64), 256, 256, 1)[0]
        cb = np.broadcast_to(qb(a.mean(0))[None, :], (R, C)).ravel()
        r[f"I_a_colbin_own|srel m{m}"] = cmi(a.ravel(), cb, sr[m], 8, 8, 7)[0]
        cbo = np.broadcast_to(qb(ao.mean(0))[None, :], (R, C)).ravel()
        r[f"I_a_colbin_otherexpert|srel m{m}"] = cmi(a.ravel(), cbo, sr[m], 8, 8, 7)[0]
        rb = np.broadcast_to(qb(a.mean(1))[:, None], (R, C)).ravel()
        r[f"I_a_rowbin_own|srel m{m}"] = cmi(a.ravel(), rb, sr[m], 8, 8, 7)[0]
        pos = np.broadcast_to(np.arange(C) % 32, (R, C)).ravel()
        r[f"I_a_pos32|srel m{m}"] = cmi(a.ravel(), pos, sr[m], 8, 32, 7)[0]
    cb1 = np.broadcast_to(qb((d[0][1] & 7).mean(0))[None, :], d[2][1].shape).ravel()
    r["I_a3_colbin_w1|srel3"] = cmi(a3, cb1, sr[2], 8, 8, 7)[0]
    # w2 column i = neuron i: neuron magnitude from w1 and w3 rows
    nm = qb((d[0][1] & 7).mean(1) + (d[2][1] & 7).mean(1))
    a2 = (d[1][1] & 7).astype(np.int64)
    r["I_a2_neuronbin_w13|srel2"] = cmi(a2.ravel(), np.broadcast_to(nm[None, :], a2.shape).ravel(), sr[1], 8, 8, 7)[0]
    r = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
    print(json.dumps(r), flush=True)
    return r

if __name__ == "__main__":
    with Pool(int(sys.argv[1])) as p:
        res = p.map(one, sorted(MAN), chunksize=1)
    with open(os.path.join(HERE, "out", "mi2.jsonl"), "w") as f:
        for r in res: f.write(json.dumps(r) + "\n")
