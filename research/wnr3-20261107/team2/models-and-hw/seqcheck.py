"""Independent check on RAW ordered blocks (not histograms):
 (a) 567-context table cost recomputed from raw codes (must match table_cost per group),
 (b) exact SEQUENTIAL latent-class coder: predictive P(m_k | m_<k) with the block-max constraint, float and with
     per-symbol 14-bit CDFs quantised like wex_nr2_quantize (realistic rANS cost), sign = 1 bit,
 (c) lookup-table approximations of the latent-class posterior: context = (srel, position bucket, bucket of the
     running magnitude sum, seen-6/7 flag), trained on the training experts (held-out),
Usage: python -I seqcheck.py SAMPLE_DIR HIST_DIR COUNTS_DIR OUT_JSON NTRAIN NTEST K
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parametric import pool, decode_keys
from lclass import fit
from blockhist import load_expert

P = 16384


def quantize_rows(p):
    """p [n,16] float probs -> freq [n,16] ints summing to P, >=1 (wex_nr2_quantize rule on counts = p*2^30)"""
    c = np.rint(p * (1 << 30)).astype(np.int64)
    tot = c.sum(1, keepdims=True)
    num = c * (P - 16)
    fl = num // tot
    rem = num % tot
    freq = 1 + fl
    left = P - freq.sum(1)
    order = np.argsort(-rem, axis=1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(16), order.shape).copy(), axis=1)
    return freq + (rank < left[:, None])


def blocks_of(ex, sample_dir, m):
    t = load_expert(os.path.join(sample_dir, ex['file']), ex['tensors'])
    w = ['w1', 'w2', 'w3'][m]
    wraw, wshape = t[f'{w}.weight']; sraw, _ = t[f'{w}.scale']
    R = wshape[0]
    wb = wraw.reshape(R, -1)
    C = wb.shape[1] * 2
    sym = np.empty((R, C), np.uint8); sym[:, 0::2] = wb & 15; sym[:, 1::2] = wb >> 4
    s16 = sraw.astype(np.int16)
    med = int(np.sort(s16)[(s16.size - 1) // 2])
    srel = (np.clip(s16 - med, -3, 3) + 3).astype(np.int64)
    return sym.reshape(-1, 32), srel


def base_ctx(blk, srel):
    mag = (blk & 7).astype(np.int64)
    pm = np.empty_like(mag); pm[:, 0] = 8; pm[:, 1:] = mag[:, :-1]
    cm = np.maximum.accumulate(mag, 1)
    mx = np.empty_like(mag); mx[:, 0] = 8; mx[:, 1:] = cm[:, :-1]
    return (srel[:, None] * 9 + pm) * 9 + mx


SUMB = 24


def run_ctx(blk, srel):
    """(srel, posb 8, running-mean bucket SUMB, seen67) -> index"""
    mag = (blk & 7).astype(np.int64)
    cs = np.cumsum(mag, 1)
    prev = np.zeros_like(cs); prev[:, 1:] = cs[:, :-1]
    k = np.arange(32)[None, :]
    mean = np.where(k > 0, prev / np.maximum(k, 1), 0)
    mb = np.where(k > 0, np.minimum((mean * 4).astype(np.int64), SUMB - 2) + 1, 0)  # 0 = none
    s67 = (mag >= 6).astype(np.int64)
    seen = np.zeros_like(mag); seen[:, 1:] = np.maximum.accumulate(s67, 1)[:, :-1]
    posb = k // 4
    return ((srel[:, None] * 8 + posb) * SUMB + mb) * 2 + seen


def seq_latent(blk, srel, models, quant=True):
    """exact sequential latent-class coder; returns (float bits, q14 bits) over all blocks"""
    mag = (blk & 7).astype(np.int64)
    nb = len(mag)
    fb = qb = 0.0
    avail = np.array(sorted(models))
    for s in range(7):
        sel = srel == s
        if not sel.any():
            continue
        th, pi = models[int(avail[np.abs(avail - s).argmin()])]  # nearest trained srel if s unseen in training
        M = mag[sel]
        CODES = blk[sel].astype(np.int64)
        n = len(M)
        logth = np.log(th)  # [K,8]
        rest = 1 - th[:, 6] - th[:, 7]
        logZ = np.log(-np.expm1(32 * np.log(rest)))
        logW = np.log(np.maximum(pi, 1e-300))[None, :] - logZ[None, :] + np.zeros((n, 1))  # prefix weight without g
        seen = np.zeros(n, bool)
        for k in range(32):
            R = 31 - k  # symbols remaining after this one
            # g before (prefix of length k): seen or 1-rest^(32-k)
            g_prev = np.where(seen[:, None], 0.0, np.log(-np.expm1((32 - k) * np.log(rest)))[None, :])
            den = logW + g_prev  # [n,K]
            # candidate m
            g_new_67 = np.zeros((n, 1))
            g_new_no = np.where(seen[:, None], 0.0, (np.log(-np.expm1(R * np.log(rest))) if R > 0 else np.full(len(rest), -np.inf))[None, :])
            cand = logW[:, :, None] + logth[None, :, :]  # [n,K,8]
            g = np.concatenate([np.repeat(g_new_no[:, :, None], 6, 2), np.repeat(g_new_67[:, :, None] + np.zeros((1, len(rest), 1)), 2, 2)], 2)
            num = cand + g
            mxv = den.max(1, keepdims=True)
            lden = np.log(np.exp(den - mxv).sum(1)) + mxv[:, 0]
            mxn = num.max(1, keepdims=True)
            mxn = np.where(np.isfinite(mxn), mxn, 0.0)  # all classes impossible for this m -> prob 0, not nan
            with np.errstate(divide='ignore'):
                lnum = np.log(np.exp(num - mxn).sum(1)) + mxn[:, 0]  # [n,8]
            pm = np.exp(lnum - lden[:, None])
            pm = pm / pm.sum(1, keepdims=True)  # guard rounding
            x = M[:, k]
            fb += float(-np.log2(pm[np.arange(n), x]).sum()) + n  # + sign bit
            if quant:
                p16 = np.concatenate([pm, pm], 1) / 2  # codes 0..7 (+), 8..15 (-)
                fr = quantize_rows(p16)
                qb += float(-np.log2(fr[np.arange(n), CODES[:, k]] / P).sum())  # actual 4-bit code
            logW = logW + logth[:, x].T
            seen = seen | (x >= 6)
    return fb, qb


def main():
    sample_dir, hist_dir, counts_dir, out_json = sys.argv[1:5]
    ntr, nte, K = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    ex = man['experts']
    have = lambda f: os.path.exists(os.path.join(hist_dir, f.replace('.bin', '.npz')))
    i0 = [i for i, e in enumerate(ex) if i % 2 == 0 and have(e['file'])]
    i1 = [i for i, e in enumerate(ex) if i % 2 == 1 and have(e['file'])]
    pick = lambda L, k: [L[int(j)] for j in np.linspace(0, len(L) - 1, min(k, len(L))).round()]
    itr, ite = pick(i0, ntr), pick(i1, nte)
    all0 = [i for i, e in enumerate(ex) if i % 2 == 0]
    A80 = sum(np.load(os.path.join(counts_dir, ex[i]['file'].replace('.bin', '.npz')))['w_base'].astype(np.float64) for i in all0)
    Atr = sum(np.load(os.path.join(counts_dir, ex[i]['file'].replace('.bin', '.npz')))['w_base'].astype(np.float64) for i in itr)
    out = {'train': [ex[i]['file'] for i in itr], 'test': [ex[i]['file'] for i in ite], 'K': K}
    t0 = time.time()
    res = {'weights': 0}
    acc = {k: 0.0 for k in ['table80_ideal', 'table80_q14', 'tableNtr_ideal', 'latent_seq_float', 'latent_seq_q14', 'runctx_ideal', 'runctx_q14']}
    for m in range(3):
        # latent-class models per srel from the training histograms
        ktr, wtr = pool([ex[i]['file'] for i in itr], hist_dir, m)
        st, Nt = decode_keys(ktr)
        models = {}
        for s in range(7):
            sel = st == s
            if wtr[sel].sum() > 0:
                (th, pi), _, _ = fit(Nt[sel], wtr[sel], K)
                models[s] = (th, pi)
        # running-context table trained on raw training blocks
        nctx = 7 * 8 * SUMB * 2
        Crun = np.zeros((nctx, 16))
        for i in itr:
            blk, srel = blocks_of(ex[i], sample_dir, m)
            Crun += np.bincount((run_ctx(blk, srel) * 16 + blk).ravel(), minlength=nctx * 16).reshape(nctx, 16)
        prun = (Crun + 0.02) / (Crun + 0.02).sum(1, keepdims=True)
        frun = quantize_rows(prun)
        p80 = (A80[m] + 1e-3) / (A80[m] + 1e-3).sum(1, keepdims=True)
        f80 = quantize_rows(p80)
        ptr = (Atr[m] + 1e-3) / (Atr[m] + 1e-3).sum(1, keepdims=True)
        for i in ite:
            blk, srel = blocks_of(ex[i], sample_dir, m)
            b = blk.astype(np.int64)
            c = base_ctx(blk, srel)
            acc['table80_ideal'] += float(-np.log2(p80[c, b]).sum())
            acc['table80_q14'] += float(-np.log2(f80[c, b] / P).sum())
            acc['tableNtr_ideal'] += float(-np.log2(ptr[c, b]).sum())
            rc = run_ctx(blk, srel)
            acc['runctx_ideal'] += float(-np.log2(prun[rc, b]).sum())
            acc['runctx_q14'] += float(-np.log2(frun[rc, b] / P).sum())
            fb, qb = seq_latent(blk, srel, models)
            acc['latent_seq_float'] += fb; acc['latent_seq_q14'] += qb
            res['weights'] += b.size
            print(f'm{m} {ex[i]["file"]}: ' + ' '.join(f'{k}={v / res["weights"]:.5f}' for k, v in acc.items()) + f' ({time.time() - t0:.0f}s)', flush=True)
    out['bpw_no_flush'] = {k: v / res['weights'] for k, v in acc.items()}
    out['runctx_rows'] = 7 * 8 * SUMB * 2
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
