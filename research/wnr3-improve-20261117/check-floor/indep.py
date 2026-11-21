"""Independent checks (check-floor lane), not reusing the floor lane's context code.
For each of the 121 sample records:
 1. re-encode the coded slab with the PRODUCTION encoder (libwex_nr3_tools wex_nr3_encode_record, header's flags,
    owner, Cw, Cs) and compare to the stored record bytes -> proves the coded slab is exactly the WNR3-CZ domain.
 2. production ideal code length from the PRODUCTION counter (wex_nr3_count_expert) x model CDFs.
 3. stored ratio = entry total_bytes / raw.
-> out/indep.jsonl ; summary to stdout.
  python3 indep.py JOBS"""
import ctypes as C, struct, os, sys, json, glob
import numpy as np
from multiprocessing import Pool
D = "<TMP>/wnr3-data"; H = os.path.dirname(os.path.abspath(__file__))
ENTRY = struct.Struct('<QIII6H6H3I3II'); RAW = 18800640

def work(p):
    lib = C.CDLL(D + "/libwex_nr3_tools.so"); P = C.c_void_p
    lib.wex_nr3_encode_record.argtypes = [P, P, P, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, P, C.c_size_t,
                                          C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
    lib.wex_nr3_count_expert.argtypes = [P, P, P, P]
    model = open(D + "/recs/model.bin", "rb").read(); table = open(D + "/recs/entries.bin", "rb").read()
    l, e = map(int, os.path.basename(p)[4:-4].split("_"))
    eb = table[(l * 192 + e) * 72:(l * 192 + e + 1) * 72]; ent = ENTRY.unpack(eb)
    rec = open(p, "rb").read()
    coded = open(os.path.join(H, "out/coded/rec_%02d_%03d.coded" % (l, e)), "rb").read()
    magic, mid, SP, WP, Cw, Cs = struct.unpack_from('<6I', rec, 0)
    flags = rec[27]; owner = struct.unpack_from('<I', rec, 28)[0]
    cap = 4 * RAW; out = C.create_string_buffer(cap); used = C.c_size_t(); tot = C.c_size_t()
    r = lib.wex_nr3_encode_record(model, eb, coded, flags, owner, Cw, Cs, out, cap, C.byref(used), C.byref(tot))
    identical = (r == 0 and tot.value == ent[1] and out.raw[:tot.value] == rec[:ent[1]])
    wc = np.zeros(3 * 567 * 16, np.uint64); sc = np.zeros(3 * 256, np.uint64)
    assert lib.wex_nr3_count_expert(eb, coded, wc.ctypes.data, sc.ctypes.data) == 0
    blob = np.frombuffer(model, np.uint8)
    W = np.frombuffer(blob[32:32 + 3 * 567 * 34].tobytes(), "<u2").reshape(3 * 567, 17).astype(np.float64)
    S = np.frombuffer(blob[32 + 3 * 567 * 34:32 + 3 * 567 * 34 + 3 * 514].tobytes(), "<u2").reshape(3, 257).astype(np.float64)
    wf = np.diff(W, axis=1).ravel(); sf = np.diff(S, axis=1).ravel()
    wcf = wc.astype(np.float64); scf = sc.astype(np.float64)
    assert not (wcf[wf == 0].any() or scf[sf == 0].any())
    wbits = float((wcf[wf > 0] * -np.log2(wf[wf > 0] / 16384)).sum())
    sbits = float((scf[sf > 0] * -np.log2(sf[sf > 0] / 16384)).sum())
    nstreams = sum(-(-ent[4 + 2 * m] * ent[5 + 2 * m] // Cs) + -(-ent[4 + 2 * m] * ent[5 + 2 * m] * 32 // Cw) for m in range(3))
    return dict(layer=l, expert=e, total_bytes=ent[1], reenc_identical=bool(identical), enc_status=r, SP=SP, WP=WP,
                ideal_w=wbits / 8, ideal_s=sbits / 8, nstreams=nstreams, used=32 + 2 * nstreams + SP + WP,
                nweights=int(wc.sum()))

if __name__ == "__main__":
    files = sorted(glob.glob(D + "/recs/rec_*.bin"))
    with Pool(int(sys.argv[1])) as pool: rows = pool.map(work, files, chunksize=1)
    with open(os.path.join(H, "out/indep.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    test = {(l, (53 * l + 7) % 192) for l in range(40)}
    def agg(sel, name):
        rr = [r for r in rows if sel(r)]
        st = np.mean([r["total_bytes"] / RAW for r in rr])
        fr = np.mean([100 * (r["total_bytes"] - r["ideal_w"] - r["ideal_s"]) / RAW for r in rr])
        lens = np.mean([100 * 2 * r["nstreams"] / RAW for r in rr]); pad = np.mean([100 * (r["total_bytes"] - r["used"]) / RAW for r in rr])
        wo = np.mean([100 * (r["WP"] - r["ideal_w"]) / RAW for r in rr]); so = np.mean([100 * (r["SP"] - r["ideal_s"]) / RAW for r in rr])
        bpw_i = np.mean([8 * r["ideal_w"] / r["nweights"] for r in rr]); bpw_s = np.mean([8 * r["WP"] / r["nweights"] for r in rr])
        print(f"{name}: n={len(rr)} reencode_identical={sum(r['reenc_identical'] for r in rr)} stored_ratio={st:.5f} "
              f"framing%={fr:.4f} (lens {lens:.4f} w-ovh {wo:.4f} s-ovh {so:.4f} pad {pad:.4f}) bpw stored {bpw_s:.4f} prod-ideal {bpw_i:.4f}")
    agg(lambda r: True, "ALL121")
    agg(lambda r: (r["layer"], r["expert"]) != (39, 120), "ALL120")
    agg(lambda r: (r["layer"], r["expert"]) in test, "TEST40")
