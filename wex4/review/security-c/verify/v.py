# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
import sys, os, struct, ctypes, hashlib
sys.dont_write_bytecode = True
H = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(H)
sys.path.insert(0, R)
import rvlib as rv
from rvlib import wex4, C, CREF
F = os.path.join(R, 'findings')
def tri(label, b, flags=3):
    a, r, p = rv.cv(C, b, flags), rv.cv(CREF, b, flags), rv.py(b, bool(flags & 1))
    print(label, 'fast', a[0], 'ref', r[0], 'py', p[0], p[1] if p[0] else '', 'digest_eq', a[1] == r[1] == (p[1] if p[0] == 0 else a[1]))
    return a
# S1
for n in ('grid_65536x65536_no_experts.wex4', 'grid_65536x65536_one_expert.wex4'):
    b = open(os.path.join(F, n), 'rb').read()
    print(n, len(b), 'layers,experts', struct.unpack_from('<II', b, 60), 'records', struct.unpack_from('<I', b, 52))
    tri('S1 ' + n, b)
# own S1: one raw tensor, grid 1x1 and 65536x65536, no experts
data = bytes(range(16))
for g in ((1, 1), (65536, 65536), (65536, 1)):
    b = rv.build(records=[rv.raw_rec(data, 0, 0, 1)], tensors=[rv.T('a', 2, [16], 0, 0, 16)], layers=g[0], experts=g[1])
    tri('S1 own grid %dx%d no experts' % g, b)
# S3
b0 = open(os.path.join(rv.NOUT, 'tiny_nr3_cz.wex4'), 'rb').read()
b1 = open(os.path.join(F, 'wnr3_scale_chunk_16384_accepted_without_sha.wex4'), 'rb').read()
diff = [i for i in range(len(b0)) if b0[i] != b1[i]]
print('S3 diff offsets', diff, len(b0) == len(b1))
a0 = tri('S3 orig flags2', b0, 2); a1 = tri('S3 mut flags2', b1, 2); print('S3 same digest', a0[1] == a1[1]); tri('S3 mut flags3', b1, 3)
# own: weight_chunk (offset 16) mutate too
rd_off = 2191360
print('S3 rec hdr', b0[rd_off:rd_off+24].hex())
for off, val in ((16, 8192), (16, 1 << 14), (20, 256), (20, 1)):
    m = bytearray(b0); struct.pack_into('<I', m, rd_off + off, val)
    a = tri('S3 own off %d val %d flags2' % (off, val), bytes(m), 2); print('   same digest', a[1] == a0[1])
# S5
for dt, shape in ((23, [1 << 40, 1 << 40]), (23, [1 << 20, 1 << 40]), (255, [1 << 40, 1 << 40])):
    b = rv.build(records=[rv.raw_rec(b'1234567', 0, 0, 1)], tensors=[rv.T('a', dt, shape, 0, 0, 7)])
    tri('S5 dtype %d shape %s' % (dt, shape), b)
# S2
rec = open(os.path.join(F, 'wbp1_decoded_ffffffff_parse_accepts.bin'), 'rb').read()
print('S2 rec', len(rec), rec[:16].hex())
h = ctypes.CDLL(os.path.join(R, 'out', 'rv_harness.dll'))
print([x for x in dir(h)][:0])
try:
    wex4.bp_decode(rec, 0xFFFFFFFF, struct.unpack_from('<I', rec, 4)[0]); print('S2 py accepted')
except wex4.FormatError as e: print('S2 py refuses:', e)
except Exception as e: print('S2 py other exc', repr(e))
