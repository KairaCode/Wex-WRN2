"""headroom / greedy forward selection of a static context (product of feature variants) on ceil-form data.

Usage: python -I hgreedy.py FEAT_DIR OUT_JSON BUDGET_ROWS START(empty|base) [KEEP_EVERY] [MAX_STEPS] [EXCLUDE]
  Train counts on even-indexed experts, held-out cost (exact 14-bit quantizer) on odd-indexed experts, per matrix type
  (3 tables, one shared feature set). Constraint: max over matrix types of train-observed rows <= BUDGET_ROWS (0 = none).
  EXCLUDE: comma list of fields never to use (e.g. pmag,pbmx = no w1->w3 phase dependency).
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hctx import VARIANTS, BASE, PCT_RAW_PER_BPW, load_split, field_values, ctx_of, compact, table_cost, CH

SAMPLE = os.environ.get('WNR_SAMPLE', 'v41-sample')  # kit: dir with the 160-expert sample (manifest.json + L*_E*.bin)
DENSE_CAP = 1 << 22


def eval_add(data, cur, Ks, v):
    """Held-out bits and rows per matrix for current contexts x variant v (None = current)."""
    res = []
    for m in range(3):
        card = 1 if v is None else VARIANTS[v][2]
        K2 = Ks[m] * card
        cts = []
        for sp in range(2):
            d = data[m][sp]
            N = d['sym'].size
            acc = np.zeros(K2 * 16, np.int64)
            for lo in range(0, N, CH):
                hi = min(N, lo + CH)
                c = cur[m][sp][lo:hi].astype(np.int64)
                if v is not None:
                    c = c * card + field_values(d, v, lo, hi)
                acc += np.bincount(c * 16 + d['sym'][lo:hi], minlength=K2 * 16)
            cts.append(acc.reshape(K2, 16))
        bits, rows, ins = table_cost(cts[0], cts[1])
        res.append((bits, rows, ins))
    return res


def recompute(data, names):
    cur, Ks = [], []
    for m in range(3):
        ctr, _ = ctx_of(data[m][0], names)
        cte, _ = ctx_of(data[m][1], names)
        a, b, K = compact(ctr, cte)
        cur.append([a, b]); Ks.append(K)
    return cur, Ks


def extend(data, cur, Ks, v):
    newc, newK = [], []
    card = VARIANTS[v][2]
    for m in range(3):
        parts = []
        for sp in range(2):
            d = data[m][sp]
            N = d['sym'].size
            c = np.empty(N, np.int64)
            for lo in range(0, N, CH):
                hi = min(N, lo + CH)
                c[lo:hi] = cur[m][sp][lo:hi].astype(np.int64) * card + field_values(d, v, lo, hi)
            parts.append(c)
        a, b, K = compact(parts[0], parts[1])
        newc.append([a, b]); newK.append(K)
    return newc, newK


def main():
    feat_dir, out_json, budget, start = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    keep = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    max_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 12
    excl = set(sys.argv[7].split(',')) if len(sys.argv) > 7 and sys.argv[7] else set()
    t0 = time.time()
    data = load_split(feat_dir, SAMPLE, keep)
    Ntest = sum(data[m][1]['sym'].size for m in range(3))
    print('loaded', [[data[m][sp]['sym'].size for sp in range(2)] for m in range(3)], f'{time.time() - t0:.0f}s', flush=True)
    cand = [v for v in VARIANTS if VARIANTS[v][0] not in excl]
    log = {'feat_dir': feat_dir, 'budget_rows': budget, 'start': start, 'keep_every': keep, 'exclude': sorted(excl),
           'test_weights': Ntest, 'steps': []}
    # reference: WNR2 base
    cur, Ks = recompute(data, BASE)
    r = eval_add(data, cur, Ks, None)
    base_bits = sum(x[0] for x in r)
    log['base'] = {'set': BASE, 'bpw': base_bits / Ntest, 'rows': [x[1] for x in r],
                   'bpw_per_matrix': [x[0] / data[m][1]['sym'].size for m, x in enumerate(r)]}
    print('base', log['base'], flush=True)
    names = list(BASE) if start == 'base' else []
    if start != 'base':
        cur, Ks = recompute(data, [])
    r = eval_add(data, cur, Ks, None)
    best_bits = sum(x[0] for x in r)
    log['start_bpw'] = best_bits / Ntest
    for step in range(max_steps):
        ts = time.time()
        scores = []
        for v in cand:
            if v in names:
                continue
            if max(Ks) * VARIANTS[v][2] > DENSE_CAP:
                continue
            r = eval_add(data, cur, Ks, v)
            bits = sum(x[0] for x in r)
            rows = [x[1] for x in r]
            scores.append((bits, v, rows, [x[0] / data[m][1]['sym'].size for m, x in enumerate(r)]))
        scores.sort()
        ok = [s for s in scores if budget == 0 or max(s[2]) <= budget]
        top = [{'v': s[1], 'bpw': s[0] / Ntest, 'delta_bpw': (s[0] - best_bits) / Ntest, 'rows': s[2]} for s in scores[:12]]
        if not ok or ok[0][0] >= best_bits - 1e-6 * Ntest:
            log['steps'].append({'step': step, 'stop': True, 'top': top, 'sec': time.time() - ts})
            print('stop', top[:5], flush=True)
            break
        bits, v, rows, perm = ok[0]
        names.append(v)
        cur, Ks = extend(data, cur, Ks, v)
        rec = {'step': step, 'add': v, 'set': list(names), 'bpw': bits / Ntest, 'gain_vs_prev_bpw': (best_bits - bits) / Ntest,
               'delta_vs_base_bpw': (bits - base_bits) / Ntest,
               'delta_vs_base_pct_raw': (bits - base_bits) / Ntest * PCT_RAW_PER_BPW,
               'rows': rows, 'K_compact': Ks, 'bpw_per_matrix': perm, 'top': top, 'sec': time.time() - ts}
        best_bits = bits
        log['steps'].append(rec)
        print(json.dumps({k: rec[k] for k in ('step', 'add', 'bpw', 'delta_vs_base_bpw', 'delta_vs_base_pct_raw', 'rows', 'sec')}),
              flush=True)
        json.dump(log, open(out_json, 'w'), indent=1)
    log['final_set'] = names
    log['total_sec'] = time.time() - t0
    json.dump(log, open(out_json, 'w'), indent=1)
    print('final', names, best_bits / Ntest, (best_bits - base_bits) / Ntest, flush=True)


if __name__ == '__main__':
    main()
