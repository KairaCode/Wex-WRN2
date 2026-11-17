"""Ideal code lengths (bounds, no rANS overhead) on TEST records for model variants: training data,
quantiser, precision, scale symbol mode, in-sample bound. Writes screen.json."""
import json, numpy as np, time
from common import *
train, test = split()
t40a = [t for t in train if BY[t]["expert"] == (37 * BY[t]["layer"] + 11) % 192]
t40b = [t for t in train if t not in t40a]
TC = {t: counts(t) for t in TAGS}
S1 = {t: np.load(W + "/counts/%s.s1.npy" % t) for t in TAGS}
def agg(tags, s1=False):
    wc = sum(TC[t][0] for t in tags); sc = sum((S1[t] if s1 else TC[t][1]) for t in tags); return wc, sc
z8 = np.ones(16, np.int64); z8[8] = 0
def mk(wc, sc, q, P):
    if q == "rule": return q_rule(wc, P), q_rule(sc, P)
    if q == "opt": return q_opt(wc, P), q_opt(sc, P)
    if q == "optz": return q_opt(wc, P, z8), q_opt(sc, P)
res = {}
def score(name, wf, sf, s1=False, per=None):
    rows = []
    for t in test:
        if per is not None: wf, sf = per(t)
        P = int(round(np.log2(wf[0, 0].sum())))
        bw = ideal_bits(TC[t][0], wf, P); bs = ideal_bits(S1[t] if s1 else TC[t][1], sf, P)
        rows.append((bw, bs))
    a = np.array(rows); res[name] = dict(w_bits=a[:, 0].tolist(), s_bits=a[:, 1].tolist())
    print("%-28s weights %.6f bit/w  scales %.5f bit/s  total %.1f B/rec" % (name, a[:, 0].mean() / 35389440, a[:, 1].mean() / 1105920, a.sum(1).mean() / 8), flush=True)
wf, sf = blob_freqs(PROD_BLOB); score("PROD", wf, sf)
for nm, tg in (("T40a", t40a), ("T40b", t40b), ("T80", train)):
    wc, sc = agg(tg); f = mk(wc, sc, "rule", 14); score("rule14_" + nm, *f)
wc, sc = agg(train); wc1, sc1 = agg(train, True)
for P in (14, 15, 16):
    for q in ("rule", "opt", "optz"):
        t0 = time.time(); f = mk(wc, sc, q, P); score("%s%d_T80" % (q, P), *f)
    f1 = (mk(wc1, sc1, "opt", P)); score("opt%d_T80_scalerel" % P, f1[0], f1[1], s1=True)
# in-sample bound: model trained on the record itself
score("rule14_self", None, None, per=lambda t: mk(*TC[t], "rule", 14))
score("optz16_self", None, None, per=lambda t: mk(*TC[t], "optz", 16))
score("optz16_self_scalerel", None, None, s1=True, per=lambda t: mk(TC[t][0], S1[t], "optz", 16))
json.dump(dict(test=test, res=res), open(W + "/screen.json", "w"))
