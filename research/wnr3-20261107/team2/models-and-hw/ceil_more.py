"""On top of the ceil form: held-out K clusters (per-matrix model id), f-buckets (decoder-computable), and the
value-exact WNR2z variant (-0 -> +0), all with 14-bit CDFs + flush, same split protocol.
Usage: python -I ceil_more.py SAMPLE_DIR CEIL_DIR OUT_JSON"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tableadapt import quantize, P, W_MAT, FLUSH_BPW


def lq(c):
    return -np.log2(quantize(c) / P)


def main():
    sample_dir, d, out_json = sys.argv[1:4]
    man = json.load(open(os.path.join(sample_dir, 'manifest.json')))
    U = {'can': [], 'ceil': []}; lay = []; split = []
    for i, e in enumerate(man['experts']):
        z = np.load(os.path.join(d, e['file'].replace('.bin', '.npz')), allow_pickle=False)
        U['can'].append(z['w_can']); U['ceil'].append(z['w_ceil']); lay.append(e['layer']); split.append(i % 2)
    lay = np.array(lay); split = np.array(split)
    n = len(lay)
    Wt = n * 3 * W_MAT
    bpw = lambda b: b / Wt + FLUSH_BPW
    out = {}
    for form in ['can', 'ceil']:
        X = np.stack(U[form]).astype(np.int64)  # [n,3,567,16]
        for zero in (False, True):
            Y = X.copy()
            if zero:
                Y[..., 0] += Y[..., 8]; Y[..., 8] = 0
            tag = form + ('+z' if zero else '')
            r = {}
            # base per type
            b = 0.0
            for tr in (0, 1):
                L = lq(Y[split == tr].sum(0))
                b += float((Y[split != tr] * L[None]).sum())
            r['base'] = bpw(b)
            # K clusters shared over types (as mxcombo_an)
            for K in (2, 4, 8):
                tb = 0.0
                for tr in (0, 1):
                    itr = np.where(split == tr)[0]
                    Ut = np.stack([Y[i, m] for i in itr for m in range(3)])
                    lt = np.array([lay[i] for i in itr for m in range(3)])
                    assign = lt * K // 40
                    for _ in range(15):
                        Lm = np.stack([lq(Ut[assign == k].sum(0)) if (assign == k).any() else lq(np.zeros((567, 16), np.int64)) for k in range(K)])
                        new = np.einsum('ucs,kcs->uk', Ut.astype(np.float64), Lm).argmin(1)
                        if (new == assign).all():
                            break
                        assign = new
                    Te = np.stack([Y[i, m] for i in np.where(split != tr)[0] for m in range(3)]).astype(np.float64)
                    tb += np.einsum('ucs,kcs->uk', Te, Lm).min(1).sum()
                r[f'K{K}'] = bpw(tb)
            # f-buckets per type (decoder-computable mean clamped srel), B = 4, 8
            sw = Y.reshape(n, 3, 7, 81, 16).sum((3, 4)).astype(np.float64)
            f = (sw * (np.arange(7) - 3)).sum(-1) / sw.sum(-1)
            for B in (4, 8):
                tb = 0.0
                for tr in (0, 1):
                    itr = np.where(split == tr)[0]; ite = np.where(split != tr)[0]
                    edges = np.quantile(f[itr].ravel(), np.linspace(0, 1, B + 1)[1:-1])
                    btr = np.searchsorted(edges, f[itr]); bte = np.searchsorted(edges, f[ite])
                    for m in range(3):
                        Ls = [lq(sum((Y[i, m] for k_, i in enumerate(itr) if btr[k_, m] == bb), np.zeros((567, 16), np.int64))) for bb in range(B)]
                        for k_, i in enumerate(ite):
                            tb += float((Y[i, m] * Ls[bte[k_, m]]).sum())
                r[f'fbucket{B}'] = bpw(tb)
            r = {k: {'bpw': v, 'delta_vs_canonical_base_pct': None} for k, v in r.items()}
            out[tag] = r
            print(tag, {k: round(v['bpw'], 5) for k, v in r.items()}, flush=True)
    ref = out['can']['base']['bpw']
    for tag in out:
        for k in out[tag]:
            out[tag][k]['delta_vs_canonical_base_pct'] = 100 * (out[tag][k]['bpw'] / ref - 1)
    json.dump(out, open(out_json, 'w'), indent=1)
    print(json.dumps({t: {k: round(v['delta_vs_canonical_base_pct'], 3) for k, v in out[t].items()} for t in out}, indent=1))


if __name__ == '__main__':
    main()
