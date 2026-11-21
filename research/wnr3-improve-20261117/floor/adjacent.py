"""Cross-LAYER relation (same expert index in the adjacent layer): the 121-record sample has no such pair, so this
fetches 2 extra records over HTTP range requests straight into out/adj/ (nothing written to <TMP>/wnr3-data):
L1 E7 (next to TEST L0 E7) and L23 E21 (next to TEST L22 E21). Each is decoded with the reference decoder (stream
hashes checked), brought to the coded domain, and I(a ; a_adjacent | srel) / I(s ; s_adjacent) are measured as in
mi2.py.  python3 adjacent.py -> out/adjacent.json"""
import ctypes as C, struct, os, json, urllib.request
import numpy as np
from load import load, H
from mi2 import cmi
URL = "https://huggingface.co/Kaira666/Kaira-WEX4-Experts/resolve/main/v41-reap/v4_flash_experts.nr3.wex"
D = "<TMP>/wnr3-data"; ENTRY = struct.Struct('<QIII6H6H3I3II'); RAW = 18800640
PAIRS = [((0, 7), (1, 7)), ((22, 21), (23, 21))]

def fetch(l, e):
    table = open(D + "/recs/entries.bin", "rb").read()
    eb = table[(l * 192 + e) * 72:(l * 192 + e + 1) * 72]; ent = ENTRY.unpack(eb)
    p = os.path.join(H, "out", "adj", "rec_%02d_%03d.bin" % (l, e))
    if not os.path.exists(p):
        req = urllib.request.Request(URL, headers={"Range": f"bytes={ent[0]}-{ent[0] + ent[1] - 1}"})
        data = urllib.request.urlopen(req, timeout=300).read(); assert len(data) == ent[1]
        open(p, "wb").write(data)
    rec = open(p, "rb").read()
    lib = C.CDLL(D + "/libwex_nr3_tools.so"); P = C.c_void_p
    lib.wex_nr3_prepare.argtypes = [P, P, C.c_int, P, P]; lib.wex_nr3_decode.argtypes = [P, P, P, C.c_uint32, C.c_uint32, P]
    out = C.create_string_buffer(RAW); assert lib.wex_nr3_decode(open(D + "/recs/model.bin", "rb").read(), eb, rec, l, e, out) == 0
    coded = C.create_string_buffer(RAW); lib.wex_nr3_prepare(eb, out, 1, coded, None)
    b = np.frombuffer(coded.raw, dtype=np.uint8); res = {}
    for m in range(3):
        R, NB = ent[4 + 2 * m], ent[5 + 2 * m]; wr, wc = ent[10 + 2 * m], ent[11 + 2 * m]
        s = b[ent[16 + m]:ent[16 + m] + R * NB].reshape(R, NB); w = b[ent[19 + m]:ent[19 + m] + wr * wc].reshape(wr, wc)
        q = np.empty((wr, wc * 2), np.uint8); q[:, 0::2] = w & 15; q[:, 1::2] = w >> 4
        res[m] = (s, q)
    return res, list(rec[24:27])

if __name__ == "__main__":
    os.makedirs(os.path.join(H, "out", "adj"), exist_ok=True)
    out = []
    for a, b in PAIRS:
        da, anc, _ = load(*a); db, ancb = fetch(*b)
        r = {"record": list(a), "adjacent": list(b)}
        for m in range(3):
            s, q = da[m]; sb, qb = db[m]
            sr = np.repeat(np.clip(s.astype(np.int64) - anc[m], -3, 3) + 3, 32, axis=1).ravel()
            r[f"I_a_aadj|srel m{m}"] = float(cmi((q & 7).ravel().astype(np.int64), (qb & 7).ravel().astype(np.int64), sr, 8, 8, 7)[0])
            r[f"I_s_sadj m{m}"] = float(cmi(s.ravel().astype(np.int64), sb.ravel().astype(np.int64), np.zeros(s.size, np.int64), 256, 256, 1)[0])
        print(json.dumps(r)); out.append(r)
    json.dump(out, open(os.path.join(H, "out", "adjacent.json"), "w"), indent=0)
