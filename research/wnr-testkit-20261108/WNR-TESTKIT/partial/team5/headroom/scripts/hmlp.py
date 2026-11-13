"""headroom / rich static upper bound: residual MLP over all causal fields on top of a static context table.

Usage: python -I hmlp.py SETS_JSON BASESET OUT_JSON [KEEP_EVERY] [HIDDEN] [EPOCHS]
logits(s) = log q_BASESET(s | ctx) + x.E[:, s] + relu(x.W1 + b1).W2[:, s] + b2[s]
x = one-hot of every raw field of hfeat (256 cols) + matrix type (3 cols); q = quantized static table trained on the
train split (even-indexed experts, stride-16 rows, KEEP_EVERY). Trained by Adam on the train split, cost on the test split
(odd-indexed experts). Zero init of E and W2 -> starts exactly at the static table.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hctx import VARIANTS, load_split, ctx_of, compact, quantize, P, PCT_RAW_PER_BPW
from hfeat import FIELDS as FIELDS_ALL

FEAT = os.environ.get('WNR_HEADROOM_FEAT', os.path.join('feat', 's16'))  # kit: regenerate with hfeat.py (2.9 GB)
SAMPLE = os.environ.get('WNR_SAMPLE', 'v41-sample')  # kit: dir with the 160-expert sample (manifest.json + L*_E*.bin)


def main():
    sets = json.load(open(sys.argv[1]))
    base = sys.argv[2]
    out_json = sys.argv[3]
    keep = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    H = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    epochs = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    t0 = time.time()
    data = load_split(FEAT, SAMPLE, keep)
    fields = [f for f in FIELDS_ALL if f[1] != 'sym']
    offs, o = {}, 0
    for word, name, sh, bits in fields:
        offs[name] = o; o += 1 << bits
    D = o + 3
    # static base logits per matrix type
    LQ = []; idx = []
    for m in range(3):
        ctr, _ = ctx_of(data[m][0], sets[base]); cte, _ = ctx_of(data[m][1], sets[base])
        a, b, K = compact(ctr, cte)
        ct = np.bincount(a.astype(np.int64) * 16 + data[m][0]['sym'], minlength=K * 16).reshape(K, 16)
        q = np.where(ct.sum(1, keepdims=True) > 0, quantize(ct), P // 16)
        LQ.append(np.log(q / P).astype(np.float32))
        idx.append([a, b])
    print('static ready', f'{time.time() - t0:.0f}s', flush=True)

    def batch_X(m, sp, sel):
        d = data[m][sp]
        Xb = np.zeros((sel.size, D), np.float32)
        r = np.arange(sel.size)
        for word, name, sh, bits in fields:
            v = d['F'][name][sel] if word == 'W' else d['F'][name][sel // 32]
            Xb[r, offs[name] + v] = 1.0
        Xb[:, o + m] = 1.0
        return Xb, LQ[m][idx[m][sp][sel]], d['sym'][sel].astype(np.int64)

    rng = np.random.default_rng(0)
    E = np.zeros((D, 16), np.float32)
    W1 = (rng.standard_normal((D, H)) * (1.0 / np.sqrt(28))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = np.zeros((H, 16), np.float32)
    b2 = np.zeros(16, np.float32)
    params = [E, W1, b1, W2, b2]
    mom = [np.zeros_like(p) for p in params]; vel = [np.zeros_like(p) for p in params]
    lr, beta1, beta2, eps = 2e-3, 0.9, 0.999, 1e-8
    BS = 32768
    step = 0

    def forward(Xb, lq, H_only=False):
        Z = Xb @ W1 + b1
        A = np.maximum(Z, 0)
        logit = lq + Xb @ E + A @ W2 + b2
        return Z, A, logit

    def nll_bits(logit, y):
        mx = logit.max(1, keepdims=True)
        lse = np.log(np.exp(logit - mx).sum(1)) + mx[:, 0]
        return (lse - logit[np.arange(y.size), y]) / np.log(2)

    def evaluate(sp, maxn=None):
        tot = 0.0; totb = 0.0; n = 0
        for m in range(3):
            N = data[m][sp]['sym'].size
            lim = N if maxn is None else min(N, maxn)
            for lo in range(0, lim, 65536):
                sel = np.arange(lo, min(lim, lo + 65536))
                Xb, lq, y = batch_X(m, sp, sel)
                _, _, logit = forward(Xb, lq)
                tot += float(nll_bits(logit, y).sum())
                totb += float(nll_bits(lq, y).sum())
                n += sel.size
        return tot / n, totb / n

    log = {'base': base, 'set': sets[base], 'keep_every': keep, 'hidden': H, 'epochs': epochs, 'D': D, 'history': []}
    Ns = [data[m][0]['sym'].size for m in range(3)]
    for ep in range(epochs):
        order = [(m, rng.permutation(Ns[m])) for m in range(3)]
        nb = max(Ns) // BS
        for bi in range(nb):
            for m, perm in order:
                sel = np.sort(perm[bi * BS:(bi + 1) * BS])
                Xb, lq, y = batch_X(m, 0, sel)
                Z, A, logit = forward(Xb, lq)
                mx = logit.max(1, keepdims=True)
                pr = np.exp(logit - mx); pr /= pr.sum(1, keepdims=True)
                g = pr; g[np.arange(y.size), y] -= 1.0; g /= y.size
                gE = Xb.T @ g
                gW2 = A.T @ g; gb2 = g.sum(0)
                gA = g @ W2.T; gZ = gA * (Z > 0)
                gW1 = Xb.T @ gZ; gb1 = gZ.sum(0)
                step += 1
                for p, gp, mo, ve in zip(params, [gE, gW1, gb1, gW2, gb2], mom, vel):
                    mo *= beta1; mo += (1 - beta1) * gp
                    ve *= beta2; ve += (1 - beta2) * gp * gp
                    p -= lr * (mo / (1 - beta1 ** step)) / (np.sqrt(ve / (1 - beta2 ** step)) + eps)
            if bi % 100 == 0:
                print(f'ep {ep} batch {bi}/{nb} {time.time() - t0:.0f}s', flush=True)
        tr, trb = evaluate(0, maxn=2_000_000)
        te, teb = evaluate(1)
        h = {'epoch': ep, 'train_sub_bpw': tr, 'train_sub_static_bpw': trb, 'test_bpw': te, 'test_static_bpw': teb,
             'gain_bpw': teb - te, 'gain_pct_raw': (teb - te) * PCT_RAW_PER_BPW, 'sec': time.time() - t0}
        log['history'].append(h)
        print(json.dumps(h), flush=True)
        json.dump(log, open(out_json, 'w'), indent=1)
        lr *= 0.5
    json.dump(log, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
