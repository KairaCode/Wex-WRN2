# Independent same-training-set baseline: WNR3-CZ re-encoded with a WNM2 model trained on the 80 TRAIN records
# (production WCHUNK/SCHUNK = 2048/512), real records, reference decode == Z(source). Writes logs/wnr3_train.txt.
import ctypes as C, os, struct, sys
D = "/tmp/wnr3-data"; H = os.path.dirname(os.path.abspath(__file__))
lib = C.CDLL(D + "/libwex_nr3_tools.so")
P = C.c_void_p
lib.wex_nr3_encode_record.argtypes = [P, P, P, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, P, C.c_size_t, P, P]
ENT = struct.Struct("<QIII6H6H3I3II")
RAW = 18800640
def load(path):
    b = os.path.basename(path)[2:-4]; l, e = map(int, b.split("_"))
    z = open(f"{H}/data/Z_{b}.bin", "rb").read(); eb = open(f"{H}/data/E_{b}.bin", "rb").read()
    return l, e, z, eb
def prep(eb, z):
    v = list(ENT.unpack(eb)); v[0], v[1], v[-1] = 0, 0, 0
    e0 = ENT.pack(*v); coded = C.create_string_buffer(len(z))
    fl = lib.wex_nr3_prepare(e0, z, 1, coded, None); assert fl >= 0
    return e0, coded, fl
wc = (C.c_uint64 * (3 * 567 * 16))(); sc = (C.c_uint64 * (3 * 256))()
for p in open(H + "/train.txt").read().split():
    l, e, z, eb = load(p); e0, coded, fl = prep(eb, z)
    assert lib.wex_nr3_count_expert(e0, coded, wc, sc) == 0
blob = C.create_string_buffer(59408 + 4096); mid = C.c_uint32()
assert lib.wex_nr3_build_model(wc, sc, blob, C.byref(mid)) == 0
blob = blob.raw[:59408]
out = open(H + "/logs/wnr3_train.txt", "w")
s_st = s_new = 0; n = 0
for p in open(H + "/test.txt").read().split():
    l, e, z, eb = load(p); e0, coded, fl = prep(eb, z)
    cap = 4 * RAW + (1 << 20); buf = C.create_string_buffer(cap); used = C.c_size_t(); total = C.c_size_t()
    rc = lib.wex_nr3_encode_record(blob, e0, coded, fl, l << 16 | e, 2048, 512, buf, cap, C.byref(used), C.byref(total))
    assert rc == 0, rc
    rec = buf.raw[:total.value]
    v = list(ENT.unpack(eb)); v[1] = total.value; e1 = ENT.pack(*v)
    o = C.create_string_buffer(RAW)
    ok = lib.wex_nr3_decode(blob, e1, rec, l, e, o) == 0 and o.raw == z
    st = os.path.getsize(f"{D}/recs/rec_{l:02d}_{e:03d}.bin")
    s_st += st; s_new += total.value; n += 1
    print(f"L{l:02d} E{e:03d} stored {st} wnr3-trainmodel {total.value} delta {100*(total.value-st)/RAW:+.4f}% ok={ok}", file=out, flush=True)
print(f"mean stored {s_st/n:.1f} ({s_st/n/RAW:.5f}) wnr3 TRAIN-model {s_new/n:.1f} ({s_new/n/RAW:.5f}) delta {100*(s_new-s_st)/n/RAW:+.4f}% of raw, n={n}", file=out)
