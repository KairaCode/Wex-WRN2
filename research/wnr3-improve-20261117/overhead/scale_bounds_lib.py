# shared helpers (quantiser = wex_nr2_quantize, record loading, TRAIN/TEST split)
import numpy as np, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(HERE + "/meta.json"))
def is_test(r): return r["expert"] == (53 * r["layer"] + 7) % 192 or (r["layer"], r["expert"]) == (39, 120)
RAW = 18800640

def quantize(counts):
    """wex_nr2_quantize for one row of counts (len n). returns frequencies (sum 16384)."""
    n = len(counts); total = int(counts.sum()); spare = 16384 - n
    if total:
        scaled = counts.astype(object) * spare
        freq = np.array([1 + int(x // total) for x in scaled]); rem = np.array([int(x % total) for x in scaled])
    else:
        freq = np.full(n, 1 + spare // n); rem = np.full(n, spare % n)
    extra = 16384 - int(freq.sum())
    order = sorted(range(n), key=lambda s: (-rem[s], s))
    for s in order[:extra]: freq[s] += 1
    return freq

def load(r):
    s = np.fromfile(f"{HERE}/scales/{r['layer']:02d}_{r['expert']:03d}.bin", np.uint8)
    out = []; off = 0
    for (R, C) in r["scale_shapes"]:
        out.append(s[off:off + R * C].reshape(R, C).astype(np.int32)); off += R * C
    return out

