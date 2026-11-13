# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Verification of spec-vs-code findings S1..S7.  python -I v.py"""
import ctypes, hashlib, io, os, struct, sys
HERE = os.path.dirname(os.path.abspath(__file__))
NEXT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, NEXT)
sys.dont_write_bytecode = True
import wex4
C = wex4.CLib(os.path.join(NEXT, 'out', 'wex4_tools_ref.dll'))

def reseal(b):
    io_, ib = struct.unpack_from('<QQ', b, 32)
    if io_ + ib <= len(b):
        b[96:128] = hashlib.sha256(b[io_:io_ + ib]).digest()
    b[4064:4096] = hashlib.sha256(b[:4064]).digest()
    return bytes(b)
def patch(base, edits):
    b = bytearray(base)
    for at, fmt, value in edits:
        struct.pack_into(fmt, b, at, value)
    return reseal(b)
def verdicts(data):
    io_, ib = struct.unpack_from('<QQ', data, 32)
    index = data[io_:io_ + wex4.align(ib)]
    c_open = C.lib.wex4_validate(data[:4096], len(data), index, len(index))
    try:
        rd = wex4.Reader(io.BytesIO(data), C); p_open = 0
    except wex4.FormatError as e:
        rd, p_open = None, f'FormatError({e})'
    c_full, c_dig = C.check(data, 3, True)
    try:
        p_dig = rd.content_digest(True) if rd else None
        p_full = 0 if rd else p_open
    except wex4.FormatError as e:
        p_dig, p_full = None, f'FormatError({e})'
    return dict(c_open=c_open, py_open=p_open, c_full=c_full, py_full=p_full,
                dig=(c_dig.hex()[:12] if c_dig else None, p_dig.hex()[:12] if p_dig else None))
def show(label, data): print(label, '\n   ', verdicts(data))

def load(name):
    base = open(os.path.join(NEXT, 'out', name), 'rb').read()
    rd = wex4.Reader(io.BytesIO(base), C)
    return base, rd
base, rd = load('t_py.wex4')
nb, nr, nt = len(rd.blobs), len(rd.records), len(rd.tensors)
r0 = rd.index_offset + nb * 64; t0 = r0 + nr * 72
print('t_py grid', rd.layers, rd.experts, 'records', [(r[3], r[4], r[8]) for r in rd.records], 'required', rd.required)
show('baseline', base)
show('S1 A 3x5', patch(base, [(60, '<I', 3), (64, '<I', 5)]))
show('S1 A2 65536^2', patch(base, [(60, '<I', 65536), (64, '<I', 65536)]))
print('   table bytes', 65536 * 65536 * 72)
for n in ('tiny_nr3_cz.wex4', 'up_tiny2_cz.wex'):
    try:
        b2, rd2 = load(n)
        ex = [r[6] for r in rd2.records if r[4] & 2]
        print(n, 'grid', rd2.layers, rd2.experts, 'expert recs', len(ex), 'max owner', hex(max(ex)) if ex else None, 'required', rd2.required)
        show('  S1 grid+1 on ' + n, patch(b2, [(60, '<I', rd2.layers + 1), (64, '<I', rd2.experts + 1)]))
        show('  S7 required^1 on ' + n, patch(b2, [(8, '<Q', rd2.required ^ 1)]))
    except Exception as e:
        print(n, 'ERR', repr(e))
wbp = next(i for i, r in enumerate(rd.records) if r[3] == 6)
for codec in (1, 2, 3, 4, 7, 0xffff):
    show(f'S2/S3 record {wbp} codec->{codec}', patch(base, [(r0 + wbp * 72 + 16, '<H', codec)]))
opaque = next(i for i, t in enumerate(rd.tensors) if t[1] == 0 or t[1] > 22)
at = t0 + opaque * 96
show('S4 E (7,9)', patch(base, [(at + 7, '<B', 2), (at + 8, '<Q', 7), (at + 16, '<Q', 9)]))
show('S4 E2 overflow', patch(base, [(at + 7, '<B', 2), (at + 8, '<Q', 1 << 62), (at + 16, '<Q', 1 << 62)]))
show('S7 G required=1', patch(base, [(8, '<Q', 1)]))
# S4 b: names_bytes / meta cap: meta_bytes 2^28+1 with otherwise-consistent index_bytes -> both refuse at header
show('S4b meta_bytes 2^28+1', patch(base, [(80, '<Q', (1 << 28) + 1)]))
