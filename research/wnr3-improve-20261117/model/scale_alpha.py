"""Ideal code length (bound) of the scales with a reduced alphabet: anchor-relative value, only the values
seen in TRAIN get a symbol, plus one escape symbol (freq 1 floor) followed by the raw byte coded uniformly
(8 bits). WNR3 rule at P = 14 / 16. Compared with the full 256-symbol absolute and relative tables. TEST set."""
import numpy as np
from common import *
train, test = split()
S1 = {t: np.load(W + "/counts/%s.s1.npy" % t) for t in TAGS}
tr = sum(S1[t] for t in train)
for P in (14, 16):
    full = q_rule(tr, P)
    seen = tr.sum(0) > 0
    keep = np.flatnonzero(seen)
    red = np.zeros((3, len(keep) + 1), np.int64); red[:, :-1] = tr[:, keep]; red[:, -1] = 0
    fr = q_rule(red, P)
    b_full = []; b_red = []
    for t in test:
        c = S1[t]; b_full.append(ideal_bits(c, full, P))
        cr = np.zeros((3, len(keep) + 1)); cr[:, :-1] = c[:, keep]; esc = c[:, ~seen].sum(1); cr[:, -1] = esc
        b_red.append(ideal_bits(cr, fr, P) + 8 * esc.sum())
    no = np.array([t != "39_120" for t in test])
    b_full = np.array(b_full) / 8; b_red = np.array(b_red) / 8
    print("P=%d alphabet %d+esc: full rel %.0f B/rec, reduced %.0f B/rec (no outlier: %.0f vs %.0f)" % (
        P, len(keep), b_full.mean(), b_red.mean(), b_full[no].mean(), b_red[no].mean()))
