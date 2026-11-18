"""Small neural predictor (torch, CPU) of the coded MAGNITUDE (8 classes) of each nibble from a wide causal
neighbourhood, trained on TRAIN records and scored on TEST records; compared on the same TEST samples with the
static WNR3 context model (base567, magnitude part, counts from the same TRAIN rows, KT smoothing, no
quantisation: a slightly optimistic static baseline). The sign is not modelled (it is a coin: mi2.py).

Features per weight (i, j) of matrix m: srel one-hot, exact scale offset, position in block, last 4 magnitudes in
the block, running max, zeros so far, row above (j-1, j, j+1), column mean of the rows above (causal), and for
w3 the sibling w1: magnitudes j-2..j+2, sign at j, block max of the w1 block, clamp(s1 - s3); for w2 the neuron
magnitude (w1 + w3 row mean of neuron j).
  python3 nn.py M ROWS_PER_MATRIX EPOCHS -> out/nn_mM.json"""
import sys, json, os, time
import numpy as np, torch
from load import load, H as HERE, train_set, test_set
torch.set_num_threads(int(os.environ.get("NN_THREADS", "1")))

def onehot(v, n):
    o = np.zeros((v.size, n), np.float32); o[np.arange(v.size), v.ravel()] = 1; return o

def feats(k, m, rows_pick, rng):
    d, anc, _ = load(*k)
    s, q = d[m]; R, C = q.shape; nb = C // 32
    a = (q & 7).astype(np.int64)
    rows = np.sort(rng.choice(np.arange(1, R), rows_pick, replace=False))
    colcum = np.cumsum(a, 0)
    F = []; Y = []; B = []
    ab = a.reshape(R, nb, 32)
    neu = ((d[0][1] & 7).mean(1) + (d[2][1] & 7).mean(1)) / 2 if m == 1 else None
    for i in rows:
        ar = ab[i]                                                     # nb, 32
        srel = (np.clip(s[i].astype(np.int64) - anc[m], -3, 3) + 3)    # nb
        sx = np.clip(s[i].astype(np.int64) - anc[m], -7, 7)
        pms = []
        for back in (1, 2, 3, 4):
            p = np.full_like(ar, 8); p[:, back:] = ar[:, :-back]; pms.append(p.ravel())
        mx = np.full_like(ar, 8); mx[:, 1:] = np.maximum.accumulate(ar, 1)[:, :-1]
        zc = np.zeros_like(ar); zc[:, 1:] = np.cumsum(ar == 0, 1)[:, :-1]
        up = a[i - 1]; upl = np.r_[8, up[:-1]]; upr = np.r_[up[1:], 8]
        colm = colcum[i - 1] / i
        k32 = np.tile(np.arange(32), nb)
        f = [onehot(np.repeat(srel, 32), 7), np.repeat(sx, 32)[:, None].astype(np.float32) / 4, onehot(k32, 32)]
        f += [onehot(p, 9) for p in pms] + [onehot(mx.ravel(), 9), zc.ravel()[:, None].astype(np.float32) / 16]
        f += [onehot(up, 9), onehot(upl, 9), onehot(upr, 9), (colm[:, None] - 2.5).astype(np.float32)]
        if m == 2:
            a1 = (d[0][1][i] & 7).astype(np.int64); sg = (d[0][1][i] >> 3).astype(np.int64)
            pad = np.r_[8, 8, a1, 8, 8]
            f += [onehot(pad[o:o + C], 9) for o in range(5)]
            f += [onehot(np.where(a1 > 0, 1 + sg, 0), 3)]
            bm1 = np.repeat(a1.reshape(nb, 32).max(1), 32); f += [onehot(bm1, 8)]
            sd = np.clip(d[0][0][i].astype(np.int64) - s[i].astype(np.int64), -2, 2) + 2; f += [onehot(np.repeat(sd, 32), 5)]
        if m == 1:
            f += [(neu[:, None] - 2.5).astype(np.float32).reshape(-1, 1)[:C]]
        F.append(np.concatenate(f, 1)); Y.append(ar.ravel())
        B.append((np.repeat(srel, 32) * 9 + pms[0]) * 9 + mx.ravel())
    return np.concatenate(F), np.concatenate(Y), np.concatenate(B)

if __name__ == "__main__":
    m, rp, ep = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(20261117)
    tr = [k for k in train_set() if k[0] % 4 == 0]          # 20 TRAIN records, every 4th layer
    te = [k for k in test_set() if k[0] % 5 == 0]           # 8 TEST records
    t0 = time.time()
    Xtr, Ytr, Btr = map(np.concatenate, zip(*[feats(k, m, rp, rng) for k in tr]))
    tes = [(k, feats(k, m, rp, rng)) for k in te]
    print("features", Xtr.shape, time.time() - t0, flush=True)
    # static base567 magnitude model from ALL rows of the same TRAIN records (KT)
    cnt = np.zeros((567, 8)) + 0.5
    for k in tr:
        d, anc, _ = load(*k); s, q = d[m]; R, C = q.shape; nb = C // 32
        a = (q & 7).reshape(R, nb, 32).astype(np.int64)
        srel = (np.clip(s.astype(np.int64) - anc[m], -3, 3) + 3)[:, :, None]
        pm = np.empty_like(a); pm[:, :, 0] = 8; pm[:, :, 1:] = a[:, :, :-1]
        mx = np.empty_like(a); mx[:, :, 0] = 8; mx[:, :, 1:] = np.maximum.accumulate(a, axis=2)[:, :, :-1]
        cnt += np.bincount((((srel * 9 + pm) * 9 + mx) * 8 + a).ravel(), minlength=567 * 8).reshape(567, 8)
    lp = np.log2(cnt / cnt.sum(1, keepdims=True))
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 64),
                              torch.nn.ReLU(), torch.nn.Linear(64, 8))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    X = torch.from_numpy(Xtr); Y = torch.from_numpy(Ytr)
    for e in range(ep):
        perm = torch.randperm(X.shape[0]); tot = 0
        for b in range(0, X.shape[0], 4096):
            idx = perm[b:b + 4096]
            loss = torch.nn.functional.cross_entropy(net(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * idx.numel()
        for g in opt.param_groups: g["lr"] *= 0.7
        print("epoch", e, "train bits/mag", tot / X.shape[0] / np.log(2), time.time() - t0, flush=True)
    res = {"m": m, "rows_per_matrix": rp, "epochs": ep, "train": tr, "ntrain": int(X.shape[0]), "test": []}
    with torch.no_grad():
        for k, (Xt, Yt, Bt) in tes:
            lo = torch.log_softmax(net(torch.from_numpy(Xt)), 1).numpy() / np.log(2)
            nn_b = float(-lo[np.arange(Yt.size), Yt].mean()); st_b = float(-lp[Bt, Yt].mean())
            res["test"].append(dict(layer=k[0], expert=k[1], n=int(Yt.size), nn_bits_mag=nn_b, static567_bits_mag=st_b,
                                    gain_bit_per_weight=st_b - nn_b))
            print(res["test"][-1], flush=True)
    g = np.array([t["gain_bit_per_weight"] for t in res["test"]])
    res["mean_gain_bit_per_weight"] = float(g.mean())
    print("mean gain bit/weight", g.mean(), g.min(), g.max())
    json.dump(res, open(os.path.join(HERE, "out", f"nn_m{m}.json"), "w"), indent=0)
