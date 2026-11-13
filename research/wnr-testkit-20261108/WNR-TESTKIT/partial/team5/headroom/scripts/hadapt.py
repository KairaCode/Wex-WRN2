"""headroom / ADAPTIVE upper bound (offline, sequential; a GPU stream-parallel decoder could not use it).

Usage: python -I hadapt.py SETS_JSON OUT_JSON [N_TEST_EXPERTS]
SETS_JSON: {"A0": [...variants], "A1": [...], "A2": [...]}   (A0 = WNR2 base, A1 = best in-budget set, A2 = big set)

Static priors per matrix type: counts of each context set on the stride-16 rows of the 80 even-indexed experts.
Test: full matrices (all rows) of N odd-indexed experts spread over layers, coded sequentially in raster order.
Per matrix and set we report (bits per weight):
  static      quantized static table (exact 14-bit quantizer; unseen context -> 4 bits)
  own         the matrix's own quantized table, in-sample, table cost NOT charged (semi-static lower bound)
  adapt(l)    Dirichlet/KT adaptive counts with the static prior as pseudo-counts:
              p_t(s) = (l * pi[ctx,s] + n_t[ctx,s]) / (l + n_t[ctx]),  pi = smoothed static distribution
  chains      hierarchical back-off: each level p = (l * p_parent(s) + n_cs) / (l + n_c) over its own context
              (big set, row band, stream) with the previous level as parent; l chosen per level on the test matrices
  mix         linear mixture of the chain levels with EM-fitted weights per matrix type (5 params, optimistic)
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxcount import load_expert
import hfeat
from hctx import VARIANTS, quantize, P, PCT_RAW_PER_BPW, field_values, CH
from hfeat import FIELD, FIELDS as FIELDS_ALL

SAMPLE = os.environ.get('WNR_SAMPLE', 'v41-sample')  # kit: dir with the 160-expert sample (manifest.json + L*_E*.bin)
FEAT = os.environ.get('WNR_HEADROOM_FEAT', os.path.join('feat', 's16'))  # kit: regenerate with hfeat.py (2.9 GB)
LAMS = [4, 16, 64, 256, 1024, 4096, 16384, 65536]


def split_from_words(W, B):
    F = {}
    for word, name, sh, bits in FIELDS_ALL:
        src = W if word == 'W' else B
        F[name] = ((src >> np.uint64(sh)) & np.uint64((1 << bits) - 1)).astype(np.uint8)
    return {'sym': F.pop('sym'), 'F': F}


def dense_ctx(d, names):
    N = d['sym'].size
    out = np.zeros(N, np.int64)
    for v in names:
        card = VARIANTS[v][2]
        for lo in range(0, N, CH):
            hi = min(N, lo + CH)
            out[lo:hi] = out[lo:hi] * card + field_values(d, v, lo, hi)
    return out


def train_priors(sets):
    """prior[set][m] = (uniq dense ids, counts [K,16])."""
    man = json.load(open(os.path.join(SAMPLE, 'manifest.json')))
    acc = {k: [dict() for _ in range(3)] for k in sets}
    for i, e in enumerate(man['experts']):
        if i % 2:
            continue
        z = np.load(os.path.join(FEAT, e['file'].replace('.bin', '.npz')))
        for m in range(3):
            d = split_from_words(z[f'W{m}'], z[f'B{m}'])
            for k, names in sets.items():
                ctx = dense_ctx(d, names)
                u, inv = np.unique(ctx, return_inverse=True)
                c = np.bincount(inv * 16 + d['sym'], minlength=u.size * 16).reshape(-1, 16)
                acc[k][m][i] = (u, c)
    pri = {}
    for k in sets:
        pri[k] = []
        for m in range(3):
            us = np.unique(np.concatenate([v[0] for v in acc[k][m].values()]))
            C = np.zeros((us.size, 16), np.int64)
            for u, c in acc[k][m].values():
                C[np.searchsorted(us, u)] += c
            pri[k].append((us, C))
    return pri


def ranks(key):
    """Number of earlier positions with the same key (stable, time order)."""
    order = np.argsort(key, kind='stable')
    ks = key[order]
    n = ks.size
    chg = np.ones(n, bool); chg[1:] = ks[1:] != ks[:-1]
    start = np.maximum.accumulate(np.where(chg, np.arange(n), 0))
    r = np.empty(n, np.int64)
    r[order] = np.arange(n) - start
    return r


def level(ctxkey, sym, p_parent, lams):
    """Adaptive level given parent probabilities of the actual symbols.
    Returns ({lam: bits}, best lam, p(sym_t) at the best lam)."""
    n_c = ranks(ctxkey).astype(np.float64)
    n_cs = ranks(ctxkey * 16 + sym).astype(np.float64)
    bl = {}
    for l in lams:
        bl[l] = bits((l * p_parent + n_cs) / (l + n_c))
    b = min(bl, key=bl.get)
    return bl, b, (b * p_parent + n_cs) / (b + n_c)


def bits(p):
    return float(-np.log2(p).sum())


def main():
    sets = json.load(open(sys.argv[1]))
    out_json = sys.argv[2]
    ntest = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    t0 = time.time()
    pri = train_priors(sets)
    print('priors', {k: [p[0].size for p in v] for k, v in pri.items()}, f'{time.time() - t0:.0f}s', flush=True)
    man = json.load(open(os.path.join(SAMPLE, 'manifest.json')))
    odd = [i for i in range(len(man['experts'])) if i % 2 == 1]
    pick = [odd[int(round(j * (len(odd) - 1) / max(ntest - 1, 1)))] for j in range(ntest)]
    res = {'sets': sets, 'test_experts': [man['experts'][i]['file'] for i in pick], 'per_matrix': []}
    tot = {}
    for i in pick:
        e = man['experts'][i]
        t = load_expert(os.path.join(SAMPLE, e['file']), e['tensors'])
        words = hfeat.expert_words(t, lambda m, R: np.arange(R))
        for m in range(3):
            W, B = words[m]
            d = split_from_words(W, B)
            del W, B
            sym = d['sym'].astype(np.int64)
            N = sym.size
            nb = 160 if m != 1 else 72
            row = np.arange(N) // (nb * 32)
            stream = np.arange(N) // 2048
            rec = {'expert': e['file'], 'm': m, 'N': N}
            pk = {}
            for k, names in sets.items():
                ctx = dense_ctx(d, names)
                us, C = pri[k][m]
                idx = np.searchsorted(us, ctx); idx = np.minimum(idx, us.size - 1)
                seen = us[idx] == ctx
                q = quantize(C)
                qs = np.where(seen, q[idx, sym] / P, 1.0 / 16)
                rec[f'{k}_static'] = bits(qs) / N
                u2, inv = np.unique(ctx, return_inverse=True)
                co = np.bincount(inv * 16 + sym, minlength=u2.size * 16).reshape(-1, 16)
                qo = quantize(co)
                rec[f'{k}_own'] = bits(qo[inv, sym] / P) / N
                pis = (C + 0.25) / (C.sum(1, keepdims=True) + 4.0)
                pi = np.where(seen, pis[idx, sym], 1.0 / 16)
                bl, lb, pbest = level(inv.astype(np.int64), sym, pi, LAMS)
                rec[f'{k}_adapt'] = {str(l): b / N for l, b in bl.items()}
                pk[k] = (inv.astype(np.int64), pi, pbest, lb)
                del ctx
            # chains on top of the best adaptive A1: + big set (A2) + row band + stream
            chain = {}
            if 'A1' in pk and 'A2' in pk:
                p1 = pk['A1'][2]
                bl2, b2, p2 = level(pk['A2'][0], sym, p1, LAMS)
                chain['A1>A2'] = {str(l): b / N for l, b in bl2.items()}
                blr, br, p3 = level(pk['A1'][0] * 100000 + row, sym, p2, LAMS)
                chain['A1>A2>row(A1)'] = {str(l): b / N for l, b in blr.items()}
                bl16, _, _ = level(pk['A1'][0] * 100000 + row // 16, sym, p2, LAMS)
                chain['A1>A2>band16(A1)'] = {str(l): b / N for l, b in bl16.items()}
                bls, _, _ = level(pk['A1'][0] * 1000000 + stream, sym, pk['A1'][1], LAMS)
                chain['static(A1)>stream(A1)'] = {str(l): b / N for l, b in bls.items()}
                # linear mixture of components (EM)
                comps = np.stack([pk['A0'][1] if 'A0' in pk else p1, p1, p2, p3])
                w = np.full(comps.shape[0], 1.0 / comps.shape[0])
                for _ in range(30):
                    mix = (w[:, None] * comps).sum(0)
                    w = (w[:, None] * comps / mix).mean(1)
                chain['mix_EM'] = bits((w[:, None] * comps).sum(0)) / N
                chain['mix_weights'] = w.tolist()
                chain['best_lams'] = {'A1': pk['A1'][3], 'A2': b2, 'row': br}
            rec['chains'] = chain
            res['per_matrix'].append(rec)
            print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 5) if isinstance(vv, float) else vv)
                                                                      for kk, vv in list(v.items())[:4]})
                              for k, v in rec.items() if k != 'chains'}), flush=True)
            print('   chains', {k: (min(v.values()) if isinstance(v, dict) and k != 'best_lams' else v)
                                for k, v in chain.items() if k != 'mix_weights'}, f'{time.time() - t0:.0f}s', flush=True)
            json.dump(res, open(out_json, 'w'), indent=1)
    res['sec'] = time.time() - t0
    json.dump(res, open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
