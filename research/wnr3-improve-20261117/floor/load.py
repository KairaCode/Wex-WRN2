"""Coded-domain loader for out/coded/rec_LL_EEE.coded (slab layout: scales w1,w2,w3 then packed FP4 w1,w2,w3)."""
import numpy as np, json, os
H = os.path.dirname(os.path.abspath(__file__))
MAN = {(r["layer"], r["expert"]): r for r in json.load(open(os.path.join(H, "out", "manifest.json")))}
RAW = 18800640

def sample():
    return sorted(MAN)

def test_set():
    s = [(l, (53 * l + 7) % 192) for l in range(40)] + [(39, 120)]
    return s

def train_set():
    return [(l, (37 * l + 11) % 192) for l in range(40)] + [(l, (37 * l + 101) % 192) for l in range(40)]

def load(l, e):
    """returns dict m -> (scales[rows, nb] uint8, nibbles[rows, nb*32] uint8), anchors, record row"""
    r = MAN[(l, e)]; ent = r["entry"]
    sshape = [(ent[4 + 2 * m], ent[5 + 2 * m]) for m in range(3)]
    wshape = [(ent[10 + 2 * m], ent[11 + 2 * m]) for m in range(3)]
    soff = ent[16:19]; woff = ent[19:22]
    b = np.fromfile(os.path.join(H, "out", "coded", "rec_%02d_%03d.coded" % (l, e)), dtype=np.uint8)
    out = {}
    for m in range(3):
        R, NB = sshape[m]
        s = b[soff[m]:soff[m] + R * NB].reshape(R, NB)
        wr, wc = wshape[m]
        w = b[woff[m]:woff[m] + wr * wc].reshape(wr, wc)
        q = np.empty((wr, wc * 2), np.uint8); q[:, 0::2] = w & 15; q[:, 1::2] = w >> 4
        assert q.shape[1] == NB * 32 and wr == R
        out[m] = (s, q)
    return out, r["anchors"], r
