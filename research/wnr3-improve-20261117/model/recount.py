"""Counts with scale symbols relative to the anchor (smode 1), via mlib m_count; also cross-checks smode 0
against the library counts for every record. Writes counts/<tag>.s1.npy (scale counts, smode 1)."""
import ctypes, numpy as np
from common import *
bad = 0
for tag in TAGS + [r["tag"] for r in EXTRA]:
    c = coded(tag); e = entry(tag)
    for sm in (0, 1):
        wc = np.zeros(3 * NC * 16, np.uint64); sc = np.zeros(768, np.uint64)
        LIB.m_count(e, c, sm, wc.ctypes.data_as(ctypes.c_void_p), sc.ctypes.data_as(ctypes.c_void_p))
        if sm == 0:
            a, b = counts(tag); bad += not (np.array_equal(a.reshape(-1), wc.astype(np.int64)) and np.array_equal(b.reshape(-1), sc.astype(np.int64)))
        else:
            np.save(W + "/counts/%s.s1.npy" % tag, sc.astype(np.int64).reshape(3, 256))
print("records", len(TAGS) + len(EXTRA), "count mismatches vs library:", bad)
