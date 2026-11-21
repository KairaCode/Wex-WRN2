# Decode each sample record with the reference decoder, bring it back to the coded domain
# (wex_nr3_prepare cz=1) and save its CODED scales (3 matrices, slab order) to scales/LL_EEE.bin,
# plus per-record facts (stored size, SP, WP, flags, anchors) to meta.json.
# usage: python3 extract.py   (reads <TMP>/wnr3-data, writes ./scales and ./meta.json)
import ctypes, struct, os, glob, json, sys
from multiprocessing import Pool
D = "<TMP>/wnr3-data"; HERE = os.path.dirname(os.path.abspath(__file__))
RAW = 18800640

def work(p):
    lib = ctypes.CDLL(D + "/libwex_nr3_tools.so")
    model = open(D + "/recs/model.bin", "rb").read(); table = open(D + "/recs/entries.bin", "rb").read()
    l, e = map(int, os.path.basename(p)[4:-4].split("_"))
    ent = table[(l * 192 + e) * 72:(l * 192 + e + 1) * 72]
    v = struct.unpack('<QIII6H6H3I3II', ent)
    rec = open(p, "rb").read()
    out = ctypes.create_string_buffer(RAW); coded = ctypes.create_string_buffer(RAW)
    st = (ctypes.c_uint32 * 16)()
    r = lib.wex_nr3_decode(model, ent, rec, l, e, out)
    assert r == 0, (l, e, r)
    flags = lib.wex_nr3_prepare(ent, out, 1, coded, st)
    assert flags == rec[27], (flags, rec[27])
    assert list(st[9:12]) == list(rec[24:27])
    sb = v[2]
    open(f"{HERE}/scales/{l:02d}_{e:03d}.bin", "wb").write(coded.raw[:sb])
    magic, mid, SP, WP, Cw, Cs = struct.unpack('<6I', rec[:24])
    nstreams = 0
    sc = [v[4 + 2 * m] * v[5 + 2 * m] for m in range(3)]
    nstreams = sum((c + Cs - 1) // Cs for c in sc) + sum((32 * c + Cw - 1) // Cw for c in sc)
    used = 32 + 2 * nstreams + SP + WP
    lens = struct.unpack(f'<{nstreams}H', rec[32:32 + 2 * nstreams])
    nss = sum((c + Cs - 1) // Cs for c in sc)
    assert sum(lens[:nss]) == SP and sum(lens[nss:]) == WP
    return dict(layer=l, expert=e, total=len(rec), used=used, SP=SP, WP=WP, streams=nstreams, scale_streams=nss,
                flags=flags, anchors=list(rec[24:27]), scale_counts=sc,
                scale_shapes=[[v[4 + 2 * m], v[5 + 2 * m]] for m in range(3)],
                scale_stream_lens=list(lens[:nss]))

if __name__ == "__main__":
    os.makedirs(HERE + "/scales", exist_ok=True)
    files = sorted(glob.glob(D + "/recs/rec_*.bin"))
    with Pool(2) as pool: res = pool.map(work, files)
    json.dump(res, open(HERE + "/meta.json", "w"))
    tot = sum(r["total"] for r in res) / len(res)
    print(len(res), "records, mean stored", tot, tot / RAW)
