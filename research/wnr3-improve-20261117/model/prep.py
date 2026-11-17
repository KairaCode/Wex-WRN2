#!/usr/bin/env python3
"""Stage 1: for each of the 121 sample records: reference decode (libwex_nr3_tools.so) -> Z(source),
prepare(cz=1) -> coded slab, check flags/anchors vs the record header, re-encode with the production model
(must reproduce the stored record byte for byte), training counts.  Writes coded/<LL_EEE>.bin,
counts/<LL_EEE>.npz, prep.json (baseline sizes)."""
import ctypes, struct, os, sys, glob, hashlib, json, numpy as np
from multiprocessing import Pool
D = "/tmp/wnr3-data"; W = "/tmp/wnr3-research/model"
RAW = 18800640
os.makedirs(W + "/coded", exist_ok=True); os.makedirs(W + "/counts", exist_ok=True)
model = open(D + "/recs/model.bin", "rb").read(); table = open(D + "/recs/entries.bin", "rb").read()

def one(p):
    lib = ctypes.CDLL(D + "/libwex_nr3_tools.so")
    l, e = map(int, os.path.basename(p)[4:-4].split("_"))
    ent = table[(l * 192 + e) * 72:(l * 192 + e + 1) * 72]; rec = open(p, "rb").read()
    out = ctypes.create_string_buffer(RAW)
    assert lib.wex_nr3_decode(model, ent, rec, l, e, out) == 0
    z = out.raw
    coded = ctypes.create_string_buffer(RAW); st = (ctypes.c_uint32 * 16)()
    fl = lib.wex_nr3_prepare(ent, z, 1, coded, st)
    assert fl == rec[27], (fl, rec[27])
    assert bytes(st[9:12]) == rec[24:27]
    cap = 2 * RAW; buf = ctypes.create_string_buffer(cap); used = ctypes.c_size_t(); tot = ctypes.c_size_t()
    r = lib.wex_nr3_encode_record(model, ent, coded, fl, (l << 16) | e, 2048, 512, buf, cap, ctypes.byref(used), ctypes.byref(tot))
    same = (r == 0 and buf.raw[:tot.value] == rec)
    wc = np.zeros(3 * 567 * 16, np.uint64); sc = np.zeros(3 * 256, np.uint64)
    assert lib.wex_nr3_count_expert(ent, coded, wc.ctypes.data_as(ctypes.c_void_p), sc.ctypes.data_as(ctypes.c_void_p)) == 0
    tag = "%02d_%03d" % (l, e)
    open(W + "/coded/%s.bin" % tag, "wb").write(coded.raw)
    np.savez(W + "/counts/%s.npz" % tag, wc=wc, sc=sc)
    sp, wp = struct.unpack("<II", rec[8:16])
    return dict(tag=tag, layer=l, expert=e, total=len(rec), used=used.value, sp=sp, wp=wp, flags=fl,
                anchors=list(rec[24:27]), reencode_identical=same, zsha=hashlib.sha256(z).hexdigest())

if __name__ == "__main__":
    ps = sorted(glob.glob(sys.argv[1] + "/rec_*.bin")) if len(sys.argv) > 1 else sorted(glob.glob(D + "/recs/rec_*.bin"))
    with Pool(2) as pool: res = pool.map(one, ps)
    json.dump(res, open(W + ("/prep_extra.json" if len(sys.argv) > 1 else "/prep.json"), "w"), indent=1)
    print(len(res), "records; reencode identical:", sum(r["reencode_identical"] for r in res))
    print("mean total/raw %.5f  mean used/raw %.5f" % (np.mean([r["total"] for r in res]) / RAW, np.mean([r["used"] for r in res]) / RAW))
