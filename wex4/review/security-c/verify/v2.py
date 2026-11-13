# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
import sys, os, struct, ctypes
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(H)
h = ctypes.CDLL(os.path.join(R, 'out', 'rv_harness.dll'))
h.rv_bp_parse.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
rec = open(os.path.join(R, 'findings', 'wbp1_decoded_ffffffff_parse_accepts.bin'), 'rb').read()
n = (ctypes.c_uint32 * 8)()
print('parse', h.rv_bp_parse(rec, len(rec), 0xFFFFFFFF, 0, n), list(n), sum(n))
# container with decoded 2^30+1 through real DLLs is refused?  header check only
for nm in ('t_py.wex4','tiny_nr3_cz.wex4'):
    b = open(os.path.join(R, '..', '..', 'out', nm), 'rb').read()
    print(nm, 'layers,experts', struct.unpack_from('<II', b, 60))
