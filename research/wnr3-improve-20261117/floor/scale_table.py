"""Static alternative for the scale streams: a per-record (per-matrix) order-0 table stored in the record, against
the production model's global scale tables. Cost = ideal code length with the record's own 14-bit quantised table
(wex_nr2_quantize rule) + the table's bytes (2 B per used symbol + 2 B header per matrix, generous) - production
ideal cost. No stream-overhead change (same streams). -> out/scale_table.jsonl"""
import json, os
import numpy as np
from load import load, MAN, H
from static_sib import quantize
RAW = 18800640
blob = np.fromfile("<TMP>/wnr3-data/recs/model.bin", dtype=np.uint8)
S = np.frombuffer(blob[57866:57866 + 3 * 514].tobytes(), dtype="<u2").reshape(3, 257).astype(np.float64)
SL = -np.log2(np.diff(S, axis=1) / 16384.0)

def q256(c):
    """wex_nr2_quantize for 256 symbols (one row)"""
    c = c.astype(np.int64)[None, :]
    total = c.sum(); spare = 16384 - 256
    f = 1 + c * spare // total; rem = c * spare % total
    extra = 16384 - f.sum()
    order = np.lexsort((np.arange(256), -rem[0]))
    f[0, order[:extra]] += 1
    return f[0]

def sparse_q(c):
    """table over the used symbols only (unused symbols get no slot: the table lists the used ones)"""
    used = np.nonzero(c)[0]; cu = c[used].astype(np.int64)
    total = cu.sum(); spare = 16384 - used.size
    f = 1 + cu * spare // total; rem = cu * spare % total
    f[np.lexsort((np.arange(used.size), -rem))[:16384 - f.sum()]] += 1
    return used, f

with open(os.path.join(H, "out", "scale_table.jsonl"), "w") as fo:
    for k in sorted(MAN):
        d, anc, _ = load(*k)
        prod = own = tb = 0.0
        for m in range(3):
            c = np.bincount(d[m][0].ravel(), minlength=256)
            prod += (c * SL[m]).sum()
            used, f = sparse_q(c)
            own += -(c[used] * np.log2(f / 16384.0)).sum()
            tb += 2 + 3 * used.size          # count byte + (symbol, u16 freq) per used symbol
        r = dict(layer=k[0], expert=k[1], prod_bytes=prod / 8, own_bytes=own / 8, table_bytes=tb,
                 delta_bytes=own / 8 + tb - prod / 8, delta_pct_raw=100 * (own / 8 + tb - prod / 8) / RAW)
        fo.write(json.dumps(r) + "\n")
        print(json.dumps(r))
