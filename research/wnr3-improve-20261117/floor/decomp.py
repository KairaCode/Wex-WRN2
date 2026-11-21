"""Decompose each stored record of the sample: ideal cost under the PRODUCTION model (WNM2 in recs/model.bin),
the per-stream rANS overhead (payload - ideal), the u16 length table, the 32-B header and the 4096-B padding.
  python3 decomp.py JOBS -> out/decomp.jsonl"""
import sys, json, os
import numpy as np
from multiprocessing import Pool
from load import load, MAN, H as HERE

blob = np.fromfile("<TMP>/wnr3-data/recs/model.bin", dtype=np.uint8)
W = np.frombuffer(blob[32:32 + 3 * 567 * 34].tobytes(), dtype="<u2").reshape(3, 567, 17).astype(np.float64)
S = np.frombuffer(blob[57866:57866 + 3 * 514].tobytes(), dtype="<u2").reshape(3, 257).astype(np.float64)
WL = -np.log2(np.diff(W, axis=2) / 16384.0)     # 3, 567, 16
SL = -np.log2(np.diff(S, axis=1) / 16384.0)     # 3, 256

def one(k):
    d, anc, r = load(*k)
    out = dict(layer=k[0], expert=k[1], total_bytes=r["total_bytes"], SP=r["SP"], WP=r["WP"], Cw=r["Cw"], Cs=r["Cs"])
    wb = sb = 0.0; nstreams = 0; n = ns = 0
    for m in range(3):
        s, q = d[m]; R, C = q.shape; nb = C // 32
        a = (q & 7).reshape(R, nb, 32).astype(np.int64)
        srel = (np.clip(s.astype(np.int64) - anc[m], -3, 3) + 3)[:, :, None]
        pm = np.empty_like(a); pm[:, :, 0] = 8; pm[:, :, 1:] = a[:, :, :-1]
        mx = np.empty_like(a); mx[:, :, 0] = 8; mx[:, :, 1:] = np.maximum.accumulate(a, axis=2)[:, :, :-1]
        ctx = ((srel * 9 + pm) * 9 + mx).ravel()
        c = np.bincount(ctx * 16 + q.ravel().astype(np.int64), minlength=567 * 16).reshape(567, 16)
        wb += float((c * WL[m]).sum())
        cs = np.bincount(s.ravel(), minlength=256)
        sb += float((cs * SL[m]).sum())
        nstreams += -(-q.size // r["Cw"]) + -(-s.size // r["Cs"])
        n += q.size; ns += s.size
    used = 32 + 2 * nstreams + r["SP"] + r["WP"]
    out.update(ideal_w_bytes=wb / 8, ideal_s_bytes=sb / 8, streams=nstreams, n=n, ns=ns, used=used,
               pad=r["total_bytes"] - used, w_stream_overhead=r["WP"] - wb / 8, s_stream_overhead=r["SP"] - sb / 8,
               ideal_bpw=wb / n, real_bpw=r["WP"] * 8 / n, ideal_bps=sb / ns, real_bps=r["SP"] * 8 / ns)
    print(json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    with Pool(int(sys.argv[1])) as p:
        res = p.map(one, sorted(MAN), chunksize=1)
    with open(os.path.join(HERE, "out", "decomp.jsonl"), "w") as f:
        for r in res: f.write(json.dumps(r) + "\n")
