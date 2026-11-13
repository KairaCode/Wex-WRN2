"""team2/audit: why does the pm2 bucket of mxcombo.py (P2B = {none, 0-3, 4-5, 6-7}) keep only -0.023% of the
-0.284% that full pm2 gives? Evaluate other pm2 bucketings (in-sample 14-bit cost + held-out S1 split).

Usage: python -I audit_pm2.py SAMPLE_DIR COUNTS_DIR OUT_JSON
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import quantize, cost_bits, ideal_bits


def main():
    sample_dir, counts_dir, out_json = sys.argv[1:4]
    with open(os.path.join(sample_dir, 'manifest.json'), 'r', encoding='utf-8') as f:
        man = json.load(f)
    A = np.zeros((3, 567, 2, 2, 9, 16), np.int64)  # split 0
    B = np.zeros_like(A)
    for i, e in enumerate(man['experts']):
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        (A if i % 2 == 0 else B)[...] += z['J2'].reshape(3, 567, 2, 2, 9, 16)
    T = A + B
    W = T.sum()
    flush = 32.0 * W / 2048
    side = None
    # bm7 side cost (held-out S1) from bmhist
    sa = np.zeros((3, 7, 2), np.int64); sb = np.zeros_like(sa)
    for i, e in enumerate(man['experts']):
        z = np.load(os.path.join(counts_dir, e['file'].replace('.bin', '.npz')))
        bh = z['bmhist']
        (sa if i % 2 == 0 else sb)[...] += np.stack([bh[..., :7].sum(-1), bh[..., 7]], -1)
    side_ho = cost_bits(sb, quantize(sa)) + cost_bits(sa, quantize(sb))
    base = T.sum((2, 3, 4)); baseA = A.sum((2, 3, 4)); baseB = B.sum((2, 3, 4))
    q_base = cost_bits(base, quantize(base))
    ho_base = cost_bits(baseB, quantize(baseA)) + cost_bits(baseA, quantize(baseB))
    # pm2 index: 0..7 magnitude, 8 none
    bucketings = {
        'full9': list(range(9)),
        'P2B_theirs {none,0-3,4-5,6-7}': [1, 1, 1, 1, 2, 2, 3, 3, 0],
        '{none,0,1-3,4-5,6-7}': [1, 2, 2, 2, 3, 3, 4, 4, 0],
        '{none,0-1,2-3,4-5,6-7}': [1, 1, 2, 2, 3, 3, 4, 4, 0],
        '{none,0,1,2,3,4-7}': [1, 2, 3, 4, 5, 5, 5, 5, 0],
        '{none,0,1,2,3+}': [1, 2, 3, 4, 4, 4, 4, 4, 0],
        '{none,0,1,2-3,4+}': [1, 2, 3, 3, 4, 4, 4, 4, 0],
        '{none,0,1+}': [1, 2, 2, 2, 2, 2, 2, 2, 0],
        '{none,0-1,2+}': [1, 1, 2, 2, 2, 2, 2, 2, 0],
        '{none,0,1,2+}': [1, 2, 3, 3, 3, 3, 3, 3, 0],
        '{0-7 merged, none} (= position 1 flag)': [1, 1, 1, 1, 1, 1, 1, 1, 0],
    }
    res = {}
    for name, bk in bucketings.items():
        nb = max(bk) + 1
        for withbm in (False, True):
            def red(X):
                Y = np.zeros((3, 567, 2, 2, nb, 16), np.int64)
                for k in range(9):
                    Y[..., bk[k], :] += X[..., k, :]
                if withbm:
                    return Y.reshape(3, -1, 16)            # bm7 x half x bucket
                return Y.sum((2, 3)).reshape(3, -1, 16)    # bucket only
            t, a, b = red(T), red(A), red(B)
            q = cost_bits(t, quantize(t)); h = cost_bits(b, quantize(a)) + cost_bits(a, quantize(b))
            if withbm:
                q += cost_bits(sa + sb, quantize(sa + sb)); h += side_ho
            res[name + (' x bm7 x half' if withbm else '')] = {
                'rows': int(t.shape[1]), 'rows_used_max': int((t.sum(-1) > 0).sum(-1).max()),
                'in_delta_pct': 100 * (q - q_base) / (q_base + flush), 'ho_S1_delta_pct': 100 * (h - ho_base) / (ho_base + flush),
                'table_KB_32B_rows': int((t.sum(-1) > 0).sum(-1).max()) * 32 / 1024}
    # where does the full-pm2 gain come from? information I(sym; pm2 | ctx) split by pm (prev magnitude)
    b4 = T.sum((2, 3))  # [3, 567, 9, 16]
    gain_by_pm = []
    r5 = b4.reshape(3, 7, 9, 9, 9, 16)  # srel, pm, mx, pm2, sym
    for pm in range(9):
        sub = r5[:, :, pm]
        gain_by_pm.append((ideal_bits(sub.sum(3)) - ideal_bits(sub)) / W)
    res['_ideal_gain_bit_per_w_by_pm'] = gain_by_pm
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        print(k, v)


if __name__ == '__main__':
    main()
