#!/usr/bin/env python3
"""Checker's own prep: for each sample record (121) and each fetched shifted record (37):
reference decode (tool library) -> Z(source) sha; prepare(cz=1) -> coded slab, compared byte-for-byte with the
lane's coded/<tag>.bin (which the copied scripts then read); re-encode with production model == stored record;
library counts and numpy anchor-relative scale counts written to THIS directory (counts/).
Writes prep.json, prep_extra.json here."""
import ctypes, struct, os, sys, glob, hashlib, json, numpy as np
from multiprocessing import Pool
D = "<TMP>/wnr3-data"; W = "<TMP>/wnr3-research/check-model"; WI = "<TMP>/wnr3-research/model"
RAW = 18800640
os.makedirs(W + "/counts", exist_ok=True)
model = open(D + "/recs/model.bin", "rb").read(); table = open(D + "/recs/entries.bin", "rb").read()

def one(p):
    lib = ctypes.CDLL(D + "/libwex_nr3_tools.so")
    l, e = map(int, os.path.basename(p)[4:-4].split("_"))
    ent = table[(l * 192 + e) * 72:(l * 192 + e + 1) * 72]; rec = open(p, "rb").read()
    E = struct.unpack("<QIII6H6H3I3II", ent)
    assert E[1] == len(rec), "entry total_bytes != record file size"
    out = ctypes.create_string_buffer(RAW)
    assert lib.wex_nr3_decode(model, ent, rec, l, e, out) == 0
    z = out.raw
    coded = ctypes.create_string_buffer(RAW); st = (ctypes.c_uint32 * 16)()
    fl = lib.wex_nr3_prepare(ent, z, 1, coded, st)
    tag = "%02d_%03d" % (l, e)
    lane_same = open(WI + "/coded/%s.bin" % tag, "rb").read() == coded.raw
    cap = 2 * RAW; buf = ctypes.create_string_buffer(cap); used = ctypes.c_size_t(); tot = ctypes.c_size_t()
    r = lib.wex_nr3_encode_record(model, ent, coded, fl, (l << 16) | e, 2048, 512, buf, cap, ctypes.byref(used), ctypes.byref(tot))
    same = (r == 0 and buf.raw[:tot.value] == rec)
    wc = np.zeros(3 * 567 * 16, np.uint64); sc = np.zeros(3 * 256, np.uint64)
    assert lib.wex_nr3_count_expert(ent, coded, wc.ctypes.data_as(ctypes.c_void_p), sc.ctypes.data_as(ctypes.c_void_p)) == 0
    np.savez(W + "/counts/%s.npz" % tag, wc=wc, sc=sc)
    # anchor-relative scale counts, numpy, anchors from the record header
    anch = list(rec[24:27]); s1 = np.zeros((3, 256), np.int64); cb = coded.raw
    for m in range(3):
        n = E[4 + 2 * m] * E[5 + 2 * m]; s = np.frombuffer(cb, np.uint8, n, E[16 + m]).astype(np.int64)
        s1[m] = np.bincount((s - anch[m]) & 255, minlength=256)
        assert s1[m].sum() == sc.reshape(3, 256)[m].sum()
    np.save(W + "/counts/%s.s1.npy" % tag, s1)
    return dict(tag=tag, layer=l, expert=e, total=len(rec), used=used.value, flags=fl, hdr_flags=rec[27],
                anchors=anch, prep_anchors=list(st[9:12]), reencode_identical=same, lane_coded_identical=lane_same,
                zsha=hashlib.sha256(z).hexdigest())

if __name__ == "__main__":
    for src, name in ((D + "/recs", "prep.json"), (WI + "/extra", "prep_extra.json")):
        ps = sorted(glob.glob(src + "/rec_*.bin"))
        with Pool(2) as pool: res = pool.map(one, ps)
        json.dump(res, open(W + "/" + name, "w"), indent=1)
        print(name, len(res), "records; reencode identical:", sum(r["reencode_identical"] for r in res),
              "lane coded identical:", sum(r["lane_coded_identical"] for r in res),
              "flags==hdr:", sum(r["flags"] == r["hdr_flags"] for r in res),
              "anchors==hdr:", sum(r["anchors"] == r["prep_anchors"] for r in res),
              "flags set:", sorted(set(r["flags"] for r in res)), flush=True)
        print("  mean total/raw %.5f  mean used/raw %.5f" % (np.mean([r["total"] for r in res]) / RAW, np.mean([r["used"] for r in res]) / RAW))
