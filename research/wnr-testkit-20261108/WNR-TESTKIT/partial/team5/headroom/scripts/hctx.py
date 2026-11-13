"""headroom / shared helpers: feature variants (LUTs over hfeat fields), data loading, held-out table cost."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hfeat import FIELD, FIELDS as FIELDS_ALL

P = 16384
RAW_RECORD = 18_800_640
W_PER_REC = 3 * 2304 * 5120
PCT_RAW_PER_BPW = 100 * W_PER_REC / 8 / RAW_RECORD   # 1 bit/w = 23.53 % of raw


def quantize(c):
    """Exact copy of mxanalyze.quantize (= wex_nr2_quantize rule)."""
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


def _lut(bits, f):
    return np.array([f(v) for v in range(1 << bits)], np.int64)


def build_variants():
    V = {}

    def add(name, field, f):
        bits = FIELD[field][3]
        lut = _lut(bits, f)
        V[name] = (field, lut, int(lut.max()) + 1)

    ident = lambda v: v
    add('srel7', 'srel15', lambda v: min(max(v - 7, -3), 3) + 3)
    add('srel3', 'srel15', lambda v: min(max(v - 7, -1), 1) + 1)
    add('srel4a', 'srel15', lambda v: min(max(v - 7, -1), 2) + 1)
    add('srel5', 'srel15', lambda v: min(max(v - 7, -2), 2) + 2)
    add('srel15', 'srel15', lambda v: min(v, 14))
    mb = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 4, 8: 5}
    for fld in ('pm', 'pm2', 'pmag'):
        add(fld, fld, lambda v: min(v, 8))
        add(fld + 'b', fld, lambda v: mb.get(v, 5))
    add('mx', 'mx', lambda v: min(v, 8))
    add('mxb', 'mx', lambda v: {5: 1, 6: 2, 7: 3, 8: 4}.get(v, 0))
    add('pos32', 'pos', ident)
    add('pos16', 'pos', lambda v: v // 2)
    add('pos8', 'pos', lambda v: v // 4)
    add('pos4', 'pos', lambda v: v // 8)
    add('pos2', 'pos', lambda v: v // 16)
    add('par', 'pos', lambda v: v % 2)
    add('last', 'pos', lambda v: int(v == 31))
    add('end5', 'pos', lambda v: 0 if v < 24 else (1 if v < 28 else (2 if v < 30 else v - 27)))
    add('psign', 'psign', lambda v: min(v, 2))
    add('nz16', 'nz', ident)
    add('nz6', 'nz', lambda v: v if v < 4 else (4 if v < 6 else 5))
    add('nz3', 'nz', lambda v: 0 if v == 0 else (1 if v < 3 else 2))
    add('run8', 'run', ident)
    add('run3', 'run', lambda v: min(v, 2))
    add('seen13', 'seen13', ident)
    add('msum8', 'msum', ident)
    add('msum4', 'msum', lambda v: 3 if v == 7 else min(v // 2, 2))
    add('nmax4', 'nmax', ident)
    add('nmax3', 'nmax', lambda v: min(v, 2))
    add('bpar', 'bpar', ident)
    add('bcolq', 'bcolq', ident)
    add('pbmax', 'pbmax', ident)
    for fld in ('pbsum', 'pbzero'):
        add(fld + '8', fld, ident)
        add(fld + '4', fld, lambda v: 3 if v == 7 else min(v // 2, 2))
    for fld in ('dl', 'dr', 'du', 'dd', 'pscale'):
        add(fld + '6', fld, lambda v: min(v, 5))
        add(fld + '4', fld, lambda v: {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}.get(v, 3))
    for fld in ('rowp', 'rowrel', 'colp', 'mlev'):
        add(fld + '8', fld, ident)
        add(fld + '4', fld, lambda v: v // 2)
    add('pbmx', 'pbmx', ident)
    return V


VARIANTS = build_variants()
BASE = ['srel7', 'pm', 'mx']   # WNR2 567-context rule


def load_split(feat_dir, sample_dir, keep_every=1):
    """Returns data[m][sp] = dict(W uint64 [N], B uint64 [N/32], sym uint8 [N]); sp 0 = even-indexed (train)."""
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    acc = [[{'W': [], 'B': []} for _ in range(2)] for _ in range(3)]
    for i, e in enumerate(man['experts']):
        z = np.load(os.path.join(feat_dir, e['file'].replace('.bin', '.npz')))
        for m in range(3):
            W, B = z[f'W{m}'], z[f'B{m}']
            if keep_every > 1:
                nbr = 160 if m != 1 else 72
                nrows = B.size // nbr
                keep = np.arange(nrows) % keep_every == 0
                B = B.reshape(nrows, nbr)[keep].ravel()
                W = W.reshape(nrows, nbr * 32)[keep].ravel()
            acc[m][i % 2]['W'].append(W)
            acc[m][i % 2]['B'].append(B)
    data = [[None, None] for _ in range(3)]
    for m in range(3):
        for sp in range(2):
            W = np.concatenate(acc[m][sp]['W']); B = np.concatenate(acc[m][sp]['B'])
            acc[m][sp] = None
            F = {}
            for word, name, sh, bits in FIELDS_ALL:
                src = W if word == 'W' else B
                F[name] = ((src >> np.uint64(sh)) & np.uint64((1 << bits) - 1)).astype(np.uint8)
            data[m][sp] = {'sym': F.pop('sym'), 'F': F}
            del W, B
    return data


CH = 1 << 23  # chunk (weights), multiple of 32


def field_values(d, vname, lo, hi):
    """Variant values for weights [lo, hi) of split dict d (int64)."""
    field, lut, card = VARIANTS[vname]
    word = FIELD[field][0]
    if word == 'W':
        return lut[d['F'][field][lo:hi]]
    return np.repeat(lut[d['F'][field][lo // 32:hi // 32]], 32)


def ctx_of(d, names):
    """Dense context id (int64) of a variant list over the whole split; returns (ctx, dense_size)."""
    N = d['sym'].size
    out = np.zeros(N, np.int64)
    size = 1
    for v in names:
        card = VARIANTS[v][2]
        for lo in range(0, N, CH):
            hi = min(N, lo + CH)
            out[lo:hi] = out[lo:hi] * card + field_values(d, v, lo, hi)
        size *= card
    return out, size


def compact(ctx_tr, ctx_te):
    u, inv = np.unique(np.concatenate([ctx_tr, ctx_te]), return_inverse=True)
    inv = inv.astype(np.int32)
    return inv[:ctx_tr.size], inv[ctx_tr.size:], u.size


def table_cost(ct, ce):
    """ct, ce [K,16] train/test counts -> (held-out bits, train-used rows, in-sample bits of train)."""
    used = ct.sum(1) > 0
    bits = 0.0
    if used.any():
        f = quantize(ct[used])
        lc = -np.log2(f / P)
        bits += float((ce[used] * lc).sum())
        ins = float((ct[used] * lc).sum())
    else:
        ins = 0.0
    bits += float(ce[~used].sum()) * 4.0
    return bits, int(used.sum()), ins


def counts(ctx, sym, K):
    return np.bincount(ctx * 16 + sym, minlength=K * 16).reshape(K, 16)
