"""Round trip of the scale escape path (no real record has a scale outside the 10 TRAIN values, so the path is
exercised synthetically): in a copy of a coded slab, 1000 blocks whose coded max is not 5 (the ceil inverse
leaves them alone) get scale anchor+40 / anchor-30 / 255 / 0; encode with P14_sesc and decode; the output must
equal Z(source) with exactly those scale bytes changed."""
import ctypes, numpy as np, hashlib
from common import *
import real
mdl = real.build("P14_sesc")[1]
for tag in ["10_153", "39_120"]:
    e = struct.unpack("<QIII6H6H3I3II", entry(tag)); c = bytearray(coded(tag)); rng = np.random.default_rng(1)
    zs = ctypes.create_string_buffer(RAW)
    rec0 = open(D + "/recs/rec_%s.bin" % tag, "rb").read()
    assert TOOLS.wex_nr3_decode(PROD_BLOB, entry(tag), rec0, BY[tag]["layer"], BY[tag]["expert"], zs) == 0
    z = bytearray(zs.raw); changed = 0
    for m in range(3):
        n = e[4 + 2 * m] * e[5 + 2 * m]; so = e[16 + m]; wo = e[19 + m]; a = BY[tag]["anchors"][m]
        w = np.frombuffer(bytes(c[wo:wo + n * 16]), np.uint8).reshape(n, 16)
        mx = np.maximum(w & 7, (w >> 4) & 7).max(1)
        cand = np.flatnonzero(mx != 5); pick = rng.choice(cand, 1000 // 3, replace=False)
        for i, b in enumerate(pick):
            v = [(a + 40) & 255, (a - 30) & 255, 255, 0][i % 4]
            c[so + b] = v; z[so + b] = v; changed += 1
    out = ctypes.create_string_buffer(2 * RAW); used = ctypes.c_size_t()
    r = BY[tag]
    assert LIB.m_encode(ctypes.byref(mdl.m), entry(tag), bytes(c), 2048, 512, r["flags"], (r["layer"] << 16) | r["expert"], mdl.model_id,
                        out, ctypes.c_size_t(2 * RAW), ctypes.byref(used), None) == 0
    rec = out.raw[:used.value]; dec = ctypes.create_string_buffer(RAW)
    rc = LIB.m_decode(ctypes.byref(mdl.m), entry(tag), rec, ctypes.c_size_t(len(rec)), mdl.model_id, dec)
    print(tag, "escaped scales", changed, "decode rc", rc, "output == expected:", dec.raw == bytes(z), "record bytes", len(rec))
