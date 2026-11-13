"""Shared helpers: 14-bit CDF quantizer identical to wex_nr2_quantize (copied from the team-1 harness
mxanalyze.py, which test_mxcount/rans_check validated), costs, loading of the per-expert counts."""
import json, os
import numpy as np

P = 16384
NW = 2304 * 5120            # weights per matrix (all three matrices have 11,796,480)
FLUSH_W_PER_MATRIX = 32 * (NW // 2048)   # 4-byte flush per 2048-weight stream = 0.015625 bit/w
W_PER_REC = 3 * NW
FLUSH_PER_REC = 3 * FLUSH_W_PER_MATRIX


def quantize(c):
    c = np.asarray(c, np.int64)
    n = c.shape[-1]
    flat = c.reshape(-1, n)
    tot = flat.sum(1, keepdims=True)
    safe = np.where(tot == 0, 1, tot)
    num = flat * (P - n)
    fl = num // safe
    rem = num % safe
    freq = 1 + fl
    left = P - freq.sum(1)
    order = np.argsort(-rem, axis=1, kind='stable')
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n)[None, :].repeat(flat.shape[0], 0), axis=1)
    freq = freq + (rank < left[:, None])
    freq[tot[:, 0] == 0] = P // n
    assert (freq.sum(1) == P).all() and (freq >= 1).all()
    return freq.reshape(c.shape)


def codelen(c):
    """-log2 of the quantized probabilities of count table c."""
    return -np.log2(quantize(c) / P)


def cost_bits(counts, freq):
    return float((np.asarray(counts, np.float64) * -np.log2(np.asarray(freq, np.float64) / P)).sum())


def ideal_bits(counts):
    c = np.asarray(counts, np.float64)
    t = c.sum(-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(c > 0, c / np.where(t == 0, 1, t), 1)
    return float(-(c * np.log2(p)).sum())


def load_all(sample_dir, counts_dir, keys=('w_base', 's_o0', 'meta', 'bm_side')):
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    ex = []
    for i, e in enumerate(man['experts']):
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        ex.append({'i': i, 'file': e['file'], 'layer': e['layer'], 'expert': e['expert'], 'split': i % 2,
                   **{k: z[k].astype(np.int64) for k in keys}})
    return ex


def rankdata(x):
    x = np.asarray(x, np.float64)
    o = np.argsort(x, kind='stable')
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    # average ties
    xs = x[o]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float('nan')


def pearson(a, b):
    a = np.asarray(a, np.float64) - np.mean(a); b = np.asarray(b, np.float64) - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float('nan')
