"""Decode the 121 sample records with the reference decoder, check the output is Z-canonical (no 0x8 nibble),
bring each slab to the coded domain (wex_nr3_prepare cz=1) and write out/coded/rec_LL_EEE.coded (raw slab layout,
coded scales + coded nibbles). Writes out/manifest.json with the stored record sizes and header fields.
  python3 extract.py [jobs]"""
import ctypes as C, struct, os, sys, glob, json
from multiprocessing import Pool
D = "<TMP>/wnr3-data"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
ENTRY = struct.Struct('<QIII6H6H3I3II')
RAW = 18800640

def work(p):
    lib = C.CDLL(D + "/libwex_nr3_tools.so"); P = C.c_void_p
    lib.wex_nr3_prepare.argtypes = [P, P, C.c_int, P, P]
    lib.wex_nr3_decode.argtypes = [P, P, P, C.c_uint32, C.c_uint32, P]
    model = open(D + "/recs/model.bin", "rb").read(); table = open(D + "/recs/entries.bin", "rb").read()
    l, e = map(int, os.path.basename(p)[4:-4].split("_"))
    eb = table[(l * 192 + e) * 72:(l * 192 + e + 1) * 72]; ent = ENTRY.unpack(eb)
    rec = open(p, "rb").read()
    out = C.create_string_buffer(RAW)
    r = lib.wex_nr3_decode(model, eb, rec, l, e, out)
    assert r == 0, (l, e, r)
    coded = C.create_string_buffer(RAW); st = (C.c_uint32 * 16)()
    flags = lib.wex_nr3_prepare(eb, out, 1, coded, st)
    magic, mid, SP, WP, Cw, Cs = struct.unpack_from('<6I', rec, 0)
    anchors = list(rec[24:27]); rflags = rec[27]
    assert flags == rflags and anchors == list(st[9:12]), (l, e)
    assert st[13] == 0 and st[14] == 0  # decoded slab already canonical: no -0 left
    open(os.path.join(OUT, "coded", "rec_%02d_%03d.coded" % (l, e)), "wb").write(coded.raw)
    return dict(layer=l, expert=e, total_bytes=ent[1], stored_len=len(rec), SP=SP, WP=WP, Cw=Cw, Cs=Cs,
                anchors=anchors, flags=rflags, entry=list(ent))

if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "coded"), exist_ok=True)
    files = sorted(glob.glob(D + "/recs/rec_*.bin"))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
        rows = pool.map(work, files)
    json.dump(rows, open(os.path.join(OUT, "manifest.json"), "w"), indent=0)
    tot = sum(r["total_bytes"] for r in rows)
    print(len(rows), "records, mean total_bytes", tot / len(rows), "ratio", tot / len(rows) / RAW)
