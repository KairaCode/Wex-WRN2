"""Harness checks: (1) the prototype with the production WNM2 tables reproduces stored records byte for byte
and decodes them to Z(source); (2) q_rule reproduces the library's wex_nr3_build_model on TRAIN counts."""
import ctypes, numpy as np, sys
from common import *
train, test = split()
wf, sf = blob_freqs(PROD_BLOB)
M = Model(wf, sf, 14)
for tag in [test[0], "39_120"]:
    rec, _ = M.encode(tag)
    stored = open(D + "/recs/rec_%s.bin" % tag, "rb").read()
    print(tag, "identical to stored:", rec == stored[:len(rec)] and not any(stored[len(rec):]), "decode ok:", M.decode_ok(tag, rec))
wc, sc = sum_counts(train)
blob = ctypes.create_string_buffer(59408); mid = ctypes.c_uint32()
TOOLS.wex_nr3_build_model(np.ascontiguousarray(wc.reshape(-1).astype(np.uint64)).ctypes.data_as(ctypes.c_void_p),
                          np.ascontiguousarray(sc.reshape(-1).astype(np.uint64)).ctypes.data_as(ctypes.c_void_p), blob, ctypes.byref(mid))
a, b = blob_freqs(blob.raw)
print("q_rule == library weights:", np.array_equal(q_rule(wc, 14), a), " scales:", np.array_equal(q_rule(sc, 14), b))
