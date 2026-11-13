"""Task 2b: anchor rule g (lower value of the densest adjacent scale pair, first on ties) as an encoder-only
WNR2 change, on the same split and the same 12 largest production records as t2_g0.py. Model C trained with
rule g (scratch copy, g0_anchor = -2); decoded with the UNCHANGED src decoder. Compares with A (median) and
B (fixed 120) from t2_g0.json.

    python -I t2b_rule_g.py OUT_DIR
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_g0 as G

RULE_G = -2


def densest_pair(s):
    h = np.bincount(s, minlength=257).astype(np.int64)
    return int(np.argmax(h[:-1] + h[1:]))   # argmax returns the first index on ties


def main(out_dir):
    t0 = time.time()
    prev = json.load(open(os.path.join(out_dir, 't2_g0.json')))
    s0 = prev['summary']
    tr = [tuple(x) for x in s0['train']]
    sh = G.Shards()
    blobC, _, _ = G.train(sh, tr, RULE_G)
    idC = G.fnv(blobC)
    blobC.tofile(os.path.join(out_dir, 'model_C_rule_g.wnm2'))
    rows = []
    for r in prev['rows']:
        sc, wt, _ = sh.expert(r['layer'], r['expert'])
        anchors = [densest_pair(s) for s in sc]
        recC, uC, tC, _ = G.encode(blobC, RULE_G, sc, wt)
        ok, rc, med, mid, _ = G.decode_check(blobC, recC, sc, wt)
        assert ok and med == anchors and mid == idC, (r['layer'], r['expert'], rc, med, anchors)
        row = dict(set=r['set'], layer=r['layer'], expert=r['expert'], medians=r['medians'], anchors_g=anchors,
                   A_total=r['A_total'], B_total=r['B_total'], C_total=tC, C_used=uC, cpu_decode_C=ok,
                   prod_total=r['prod_total'])
        rows.append(row)
        print(json.dumps(row), flush=True)
    raw = G.RAW
    T = [r for r in rows if r['set'] == 'test']
    dC = np.array([r['C_total'] - r['A_total'] for r in T], float)
    Bg = [r for r in rows if r['set'] == 'big']
    summ = dict(model_C_id=idC, n_test=len(T), all_cpu_decode_exact=all(r['cpu_decode_C'] for r in rows),
                C_mean_total=float(np.mean([r['C_total'] for r in T])), C_ratio=float(np.mean([r['C_total'] for r in T]) / raw),
                delta_C_vs_A_pct_raw=float(dC.mean() / raw * 100),
                delta_C_vs_A_pct_raw_CI95=[x / raw * 100 for x in G.boot(dC)],
                median_expert_delta_pct_raw=float(np.median(dC) / raw * 100),
                experts_C_smaller=int((dC < 0).sum()), experts_C_larger=int((dC > 0).sum()),
                matrices_anchor_g_ne_median=sum(sum(a != m for a, m in zip(r['anchors_g'], r['medians'])) for r in T),
                big=dict(A_max=max(r['A_total'] for r in Bg), B_max=max(r['B_total'] for r in Bg),
                         C_max=max(r['C_total'] for r in Bg), prod_max=max(r['prod_total'] for r in Bg),
                         delta_C_vs_A_pct_raw_mean=float(np.mean([(r['C_total'] - r['A_total']) / raw * 100 for r in Bg])),
                         rows=[(r['layer'], r['expert'], r['medians'], r['anchors_g'],
                                round((r['B_total'] - r['A_total']) / raw * 100, 3),
                                round((r['C_total'] - r['A_total']) / raw * 100, 3)) for r in Bg]),
                seconds=time.time() - t0)
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(out_dir, 't2b_rule_g.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
