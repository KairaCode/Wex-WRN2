"""Paired B vs A decode rate over the first K decode tokens of each prompt (decprof timestamps), so W3's
256-token runs and the short ABBA's 64-token runs are compared on the same token window.

  python -I cut_tokens.py [--k 64] [--ids code1,...] [--pairs 1,3] RUNDIR [RUNDIR...]

--pairs keeps only those pairs (W3 two-pair subsets: what a short ABBA's 2 pairs would have seen). Same-arm
spread, as in analyze_short.py: (max-min)/mean of each arm's run means over the prompts (the verdict guard,
2 %) and the largest per-prompt spread (a printed warning only).

A run is RUNDIR/<name>.jsonl + .stderr.log with A<i>/B<i> in the name (smoke runs skipped); prompt spans come
from READY + cumulative wall, as in real_cost.py. Rate = (n-1)/(t_n - t_1) over decode tokens 2..K (token 1
comes with the prefill); wait = mean moeprof wait over the same tokens. On the short ABBA's own runs this
reproduces analyze_short.py (-2.69 +- 1.87 vs -2.69 +- 1.85 %, iid intervals, 128 tokens).
Three 95% intervals for the mean: prompts fixed (pair-to-pair residuals, df = n_pairs*n_prompts - n_prompts,
analyze_short.py's verdict interval), pair means (one mean per pair over the prompts, df = n_pairs - 1, W3
results.md's; it also absorbs a run-level effect) and iid over every delta (overstates n: the deltas of one
prompt share its level). W3 at K=64 over the 7 default prompts: -1.89, pair means +- 0.70.
"""
import json
import math
import re
import sys
from pathlib import Path

T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042]   # two-sided 95% t quantile, df 1..30 (analyze_short.py's table); 2.0 above
DEC = re.compile(r'\[\s*([\d.]+)\] decprof position=')
MOE = re.compile(r'\[\s*([\d.]+)\] moeprof wait=([\d.]+)')
NAME = re.compile(r'(?<![A-Za-z0-9])([AB])(\d+)(?![0-9])')


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float('nan')
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, (T975[n - 2] if n <= 31 else 2.0) * sd / math.sqrt(n)


def fixed_ci(cells):
    """cells {prompt: [delta of each pair]}: mean and 95% half-width with the prompts as fixed effects."""
    xs = [x for v in cells.values() for x in v]
    n, df = len(xs), len(xs) - len(cells)
    if df < 1:
        return sum(xs) / n, float('nan'), df
    ss = sum((x - sum(v) / len(v)) ** 2 for v in cells.values() for x in v)
    return sum(xs) / n, (T975[df - 1] if df <= 30 else 2.0) * math.sqrt(ss / df / n), df


def run(prefix, ids, k):
    recs = [json.loads(l) for l in open(str(prefix) + '.jsonl', encoding='utf-8') if l.strip()]
    log = open(str(prefix) + '.stderr.log', encoding='utf-8', errors='replace').read()
    dec = [float(m[1]) for m in DEC.finditer(log)]
    moe = [(float(m[1]), float(m[2])) for m in MOE.finditer(log)]
    t = next(r['secs'] for r in recs if r.get('event') == 'ready')
    out = {}
    for r in recs:
        if 'id' not in r:
            continue
        a, b = t, t + r['wall']
        t = b
        if r['id'] not in ids:
            continue
        ts = [x for x in dec if a <= x < b][:k - 1]
        w = [v for x, v in moe if a <= x < b][:k - 1]
        out[r['id']] = ((len(ts) - 1) / (ts[-1] - ts[0]), sum(w) / len(w), len(ts) + 1)
    return out


def main(argv):
    k, ids, keep = 64, {'code1', 'code2', 'chat_it1', 'chat_it2', 'reason1', 'reason2', 'short1'}, None
    while argv and argv[0].startswith('--'):
        if argv[0] == '--k':
            k = int(argv[1])
        elif argv[0] == '--ids':
            ids = set(argv[1].split(','))
        elif argv[0] == '--pairs':
            keep = {int(x) for x in argv[1].split(',')}
        else:
            raise SystemExit(__doc__)
        argv = argv[2:]
    if not argv:
        raise SystemExit(__doc__)
    for d in argv:
        runs = {}
        for p in sorted(Path(d).glob('*.jsonl')):
            m = NAME.search(p.stem)
            if m and 'smoke' not in p.stem and (keep is None or int(m[2]) in keep):
                runs[(m[1], int(m[2]))] = run(p.with_suffix(''), ids, k)
        pairs = sorted(i for (arm, i) in runs if arm == 'A' and ('B', i) in runs)
        used = sorted(set.intersection(ids, *(set(runs[x, i]) for i in pairs for x in 'AB')))   # balanced cells
        per = {pid: [100 * (runs['B', i][pid][0] / runs['A', i][pid][0] - 1) for i in pairs] for pid in used}
        wait = {pid: [runs['B', i][pid][1] - runs['A', i][pid][1] for i in pairs] for pid in used}
        arm = lambda a: [v[0] for (x, i), r in runs.items() if x == a and i in pairs for v in r.values()]
        mr, cr, df = fixed_ci(per)
        mw, cw, _ = fixed_ci(wait)
        pm = [sum(per[p][j] for p in used) / len(used) for j in range(len(pairs))]
        _, cp = mean_ci(pm)
        _, ci = mean_ci([x for v in per.values() for x in v])
        ntok = sorted({v[2] for r in runs.values() for v in r.values()})
        print(f'{d}: first {k} tokens (got {ntok}), pairs {pairs}: A {sum(arm("A")) / len(arm("A")):.3f} '
              f'B {sum(arm("B")) / len(arm("B")):.3f} tok/s; B vs A {mr:+.2f} % +- {cr:.2f} prompts fixed (df={df}), '
              f'+- {cp:.2f} pair means (df={len(pairs) - 1}), +- {ci:.2f} iid (n={len(used) * len(pairs)}); '
              f'wait {mw:+.1f} +- {cw:.1f} ms/tok (prompts fixed); '
              f'per prompt {{{", ".join(f"{p}: {[round(x, 2) for x in v]}" for p, v in per.items())}}}')
        sp = lambda v: 100 * (max(v) - min(v)) / (sum(v) / len(v))
        runm = {x: sp([sum(runs[x, i][p][0] for p in used) / len(used) for i in pairs]) for x in 'AB'}
        top = max(((sp([runs[x, i][p][0] for i in pairs]), f'{x} {p}') for x in 'AB' for p in used), default=(0, '-'))
        print(f'   same-arm spread: run means A {runm["A"]:.2f} % B {runm["B"]:.2f} %; '
              f'largest per prompt {top[1]} {top[0]:.2f} %; pair means {[round(x, 2) for x in pm]}')
        if ids - set(used):
            print(f'   left out (missing from a paired run): {sorted(ids - set(used))}')


if __name__ == '__main__':
    assert mean_ci([1.0, 3.0])[0] == 2.0 and abs(mean_ci([1.0, 3.0])[1] - 12.706) < 1e-9
    assert abs(fixed_ci({'a': [1.0, 3.0], 'b': [5.0, 7.0]})[1] - 4.303 * math.sqrt(0.5)) < 1e-9
    assert abs(mean_ci([float(x) for x in range(40)])[1] - 2.0 * math.sqrt(40 * 41 / 12) / math.sqrt(40)) < 1e-9   # n > 31: 2.0
    main(sys.argv[1:])
