"""headroom / greedy forward selection, field-level counting (all variants of a field from one pass).

Usage: python -I hgreedy2.py FEAT_DIR OUT_JSON BUDGET_ROWS START(empty|base) [KEEP_EVERY] [MAX_STEPS] [EXCLUDE]
Train = even-indexed experts, held-out cost (exact 14-bit quantizer, 4 bits for contexts unseen in training)
on odd-indexed experts, 3 tables (per matrix type) with one shared feature set.
Constraint: max over matrix types of train-observed rows <= BUDGET_ROWS (0 = unlimited).
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hctx import VARIANTS, BASE, PCT_RAW_PER_BPW, load_split, ctx_of, compact, table_cost, field_values, CH
from hfeat import FIELD

SAMPLE = os.environ.get('WNR_SAMPLE', 'v41-sample')  # kit: dir with the 160-expert sample (manifest.json + L*_E*.bin)
MAXBINS = 1 << 26


def field_tables(data, cur64, Ks, field):
    """[m][sp] counts [K, 2^bits, 16] of (current context, raw field value, symbol)."""
    word, _, _, bits = FIELD[field]
    R = 1 << bits
    out = []
    for m in range(3):
        if Ks[m] * R * 16 > MAXBINS:
            return None
        pair = []
        for sp in range(2):
            d = data[m][sp]
            N = d['sym'].size
            acc = np.zeros(Ks[m] * R * 16, np.int64)
            for lo in range(0, N, CH):
                hi = min(N, lo + CH)
                raw = d['F'][field][lo:hi] if word == 'W' else np.repeat(d['F'][field][lo // 32:hi // 32], 32)
                rs = raw.astype(np.int64) * 16 + d['sym'][lo:hi]
                rs += cur64[m][sp][lo:hi]
                acc += np.bincount(rs, minlength=Ks[m] * R * 16)
            pair.append(acc.reshape(Ks[m], R, 16))
        out.append(pair)
    return out


def variant_cost(tabs, v):
    field, lut, card = VARIANTS[v]
    R = tabs[0][0].shape[1]
    M = np.zeros((R, card), np.int64)
    M[np.arange(R), lut[:R]] = 1
    res = []
    for m in range(3):
        ct = np.einsum('krs,rv->kvs', tabs[m][0], M).reshape(-1, 16)
        ce = np.einsum('krs,rv->kvs', tabs[m][1], M).reshape(-1, 16)
        res.append(table_cost(ct, ce))
    return res


def make_cur64(cur, Ks, fields_R):
    return [[c.astype(np.int64) * (fields_R * 16) for c in cur[m]] for m in range(3)]


def main():
    feat_dir, out_json, budget, start = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    keep = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    max_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 12
    excl = set(sys.argv[7].split(',')) if len(sys.argv) > 7 and sys.argv[7] else set()
    t0 = time.time()
    data = load_split(feat_dir, SAMPLE, keep)
    Nt = [data[m][1]['sym'].size for m in range(3)]
    Ntest = sum(Nt)
    print('loaded', [[data[m][sp]['sym'].size for sp in range(2)] for m in range(3)], f'{time.time() - t0:.0f}s', flush=True)
    fields = sorted({VARIANTS[v][0] for v in VARIANTS if VARIANTS[v][0] not in excl})
    log = {'feat_dir': feat_dir, 'budget_rows': budget, 'start': start, 'keep_every': keep, 'exclude': sorted(excl),
           'test_weights_per_matrix': Nt, 'steps': []}

    def state(names):
        cur, Ks = [], []
        for m in range(3):
            ctr, _ = ctx_of(data[m][0], names)
            cte, _ = ctx_of(data[m][1], names)
            a, b, K = compact(ctr, cte)
            cur.append([a, b]); Ks.append(K)
        return cur, Ks

    def cost_now(cur, Ks):
        res = []
        for m in range(3):
            ct = np.bincount(cur[m][0].astype(np.int64) * 16 + data[m][0]['sym'], minlength=Ks[m] * 16).reshape(-1, 16)
            ce = np.bincount(cur[m][1].astype(np.int64) * 16 + data[m][1]['sym'], minlength=Ks[m] * 16).reshape(-1, 16)
            res.append(table_cost(ct, ce))
        return res

    cur, Ks = state(BASE)
    r = cost_now(cur, Ks)
    base_bits = sum(x[0] for x in r)
    log['base'] = {'set': BASE, 'bpw': base_bits / Ntest, 'rows': [x[1] for x in r],
                   'bpw_per_matrix': [x[0] / Nt[m] for m, x in enumerate(r)], 'insample_bpw': sum(x[2] for x in r) / Ntest}
    print('base', log['base'], flush=True)
    names = list(BASE) if start == 'base' else []
    if start != 'base':
        cur, Ks = state(names)
        r = cost_now(cur, Ks)
    best_bits = sum(x[0] for x in r)
    log['start_bpw'] = best_bits / Ntest
    for step in range(max_steps):
        ts = time.time()
        scores = []
        for fld in fields:
            R = 1 << FIELD[fld][3]
            cur64 = [[c.astype(np.int64) * (R * 16) for c in cur[m]] for m in range(3)]
            tabs = field_tables(data, cur64, Ks, fld)
            del cur64
            if tabs is None:
                continue
            for v in VARIANTS:
                if VARIANTS[v][0] != fld or v in names:
                    continue
                res = variant_cost(tabs, v)
                scores.append((sum(x[0] for x in res), v, [x[1] for x in res], [x[0] / Nt[m] for m, x in enumerate(res)]))
            del tabs
        scores.sort()
        ok = [s for s in scores if budget == 0 or max(s[2]) <= budget]
        top = [{'v': s[1], 'bpw': s[0] / Ntest, 'delta_bpw': (s[0] - best_bits) / Ntest, 'rows': s[2]} for s in scores[:16]]
        if not ok or ok[0][0] >= best_bits - 1e-6 * Ntest:
            log['steps'].append({'step': step, 'stop': True, 'top': top, 'sec': time.time() - ts})
            print('stop', top[:5], flush=True)
            break
        bits, v, rows, perm = ok[0]
        names.append(v)
        cur, Ks = state(names)
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
