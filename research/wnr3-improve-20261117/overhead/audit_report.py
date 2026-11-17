# Aggregate out/audit.json: per profile and corruption type, over all audited records.
# Accepted-wrong upper bound: exact one-sided 95% Clopper-Pearson (0 events: 1 - 0.05**(1/n)).
# usage: python3 audit_report.py out/audit.json
import json, sys
from math import lgamma, log, exp
a = json.load(open(sys.argv[1]))
T = a["types"]
def ub(k, n, conf=0.95):
    # smallest p with P(X <= k | n, p) <= 1 - conf, by bisection
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        cdf = sum(exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log(1 - p)) for i in range(k + 1))
        if cdf > 1 - conf: lo = p
        else: hi = p
    return hi
profs = list(a["records"][0]["profiles"].keys())
print(f"records: {[r['record'] for r in a['records']]}, N per type per record = {a['N']}, seed {a['seed']}")
print(f"{'profile':20s} {'type':18s} {'trials':>7s} {'rejected':>8s} {'acc_same':>8s} {'acc_WRONG':>9s} {'digest_only':>11s} {'95% UB wrong':>12s} {'xcheck(mism)':>12s}")
summ = {}
for p in profs:
    tot = [0] * 7
    for t, tn in enumerate(T):
        s = [sum(r["profiles"][p][t][c] for r in a["records"]) for c in range(7)]
        tot = [x + y for x, y in zip(tot, s)]
        print(f"{p:20s} {tn:18s} {s[0]:7d} {s[1]:8d} {s[2]:8d} {s[3]:9d} {s[4]:11d} {ub(s[3], s[0]):12.2e} {s[5]:6d}({s[6]})")
    print(f"{p:20s} {'ALL':18s} {tot[0]:7d} {tot[1]:8d} {tot[2]:8d} {tot[3]:9d} {tot[4]:11d} {ub(tot[3], tot[0]):12.2e} {tot[5]:6d}({tot[6]})")
    summ[p] = dict(trials=tot[0], rejected=tot[1], accepted_identical=tot[2], accepted_wrong=tot[3], digest_only=tot[4],
                   ub95_wrong=ub(tot[3], tot[0]), crosschecked=tot[5], cross_mismatch=tot[6])
json.dump(summ, open(sys.argv[1].replace(".json", "_summary.json"), "w"), indent=1)
