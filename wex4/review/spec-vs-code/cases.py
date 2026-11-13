# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Targeted spec-vs-code cases, each run through the C reader and the Python reader.  python -I cases.py"""
import ctypes, hashlib, io, os, struct, sys
HERE = os.path.dirname(os.path.abspath(__file__))
NEXT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, NEXT)
import wex4
C = wex4.CLib(os.path.join(NEXT, 'out', 'wex4_tools_ref.dll'))
OUT = os.path.join(HERE, 'out')


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
    """(C open, Python open, C full check, Python full check, digests equal)"""
    io_, ib = struct.unpack_from('<QQ', data, 32)
    index = data[io_:io_ + wex4.align(ib)]
    c_open = C.lib.wex4_validate(data[:4096], len(data), index, len(index))
    try:
        rd = wex4.Reader(io.BytesIO(data), C)
        p_open = 0
    except wex4.FormatError as e:
        rd, p_open = None, f'FormatError({e})'
    c_full, c_dig = C.check(data, 3, True)
    try:
        p_dig = rd.content_digest(True) if rd else None
        p_full = 0 if rd else p_open
    except wex4.FormatError as e:
        p_dig, p_full = None, f'FormatError({e})'
    return dict(c_open=c_open, py_open=p_open, c_full=c_full, py_full=p_full,
                same_digest=(c_dig == p_dig) if c_full == 0 and p_full == 0 else None)


def show(label, data):
    print(f'{label}\n    {verdicts(data)}')


base = open(os.path.join(NEXT, 'out', 't_py.wex4'), 'rb').read()
rd = wex4.Reader(io.BytesIO(base))
nb, nr, nt = len(rd.blobs), len(rd.records), len(rd.tensors)
r0 = rd.index_offset + nb * 64
t0 = r0 + nr * 72
print('t_py.wex4 records (codec, flags, tensor_count):', [(r[3], r[4], r[8]) for r in rd.records])
print('layers/experts in t_py.wex4:', rd.layers, rd.experts, ' expert records:', sum(1 for r in rd.records if r[4] & 2))
show('baseline', base)

# A. spec 4: num_layers "0 when the file has no expert records"
show('A. expert grid 3 x 5 declared in a file with no expert record', patch(base, [(60, '<I', 3), (64, '<I', 5)]))
show('A2. expert grid 65536 x 65536, no expert record', patch(base, [(60, '<I', 65536), (64, '<I', 65536)]))

# B. unknown / legacy codec ids on a compressed record (spec 5.2, 7, 10.6)
wbp = next(i for i, r in enumerate(rd.records) if r[3] == 6)
for codec in (7, 0xffff, 1, 2, 3, 4):
    show(f'B. record {wbp} codec 6 -> {codec}', patch(base, [(r0 + wbp * 72 + 16, '<H', codec)]))
# unknown codec may point at any blob? (no blob in this file) and keep flags?
show('B2. unknown codec 7 with EXPERT flag', patch(base, [(r0 + wbp * 72 + 16, '<H', 7), (r0 + wbp * 72 + 18, '<H', 2)]))

# E. opaque dtype: spec 5.4 "its shape is not checked against its size"
opaque = next((i for i, t in enumerate(rd.tensors) if t[1] == 0 or t[1] > 22), None)
print('opaque tensor index:', opaque, rd.tensors[opaque][:4] if opaque is not None else None)
if opaque is not None:
    at = t0 + opaque * 96
    show('E. opaque tensor: rank 2, shape (7, 9) unrelated to nbytes', patch(base, [(at + 7, '<B', 2), (at + 8, '<Q', 7), (at + 16, '<Q', 9)]))
    show('E2. opaque tensor: rank 2, shape (2^62, 2^62) (product overflows u64)',
         patch(base, [(at + 7, '<B', 2), (at + 8, '<Q', 1 << 62), (at + 16, '<Q', 1 << 62)]))
    show('E3. opaque tensor: rank 2, shape (0, 2^63) on a tensor that has bytes', patch(base, [(at + 7, '<B', 2), (at + 8, '<Q', 0), (at + 16, '<Q', 1 << 63)]))

# F. empty container: header page only
page = wex4.HEADER.pack(wex4.MAGIC, 4, 0, 0, 0, 4096, 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, hashlib.sha256(b'').digest())
page += bytes(4064 - len(page))
empty = page + hashlib.sha256(page).digest()
show('F. empty container (4096 bytes, no index)', empty)
# metadata-only container: index = meta bytes only
meta = b'hello'
page = wex4.HEADER.pack(wex4.MAGIC, 4, 0, 0, 0, 8192, 4096, len(meta), 0, 0, 0, 0, 0, 0, 0, len(meta), 0, hashlib.sha256(meta).digest())
page += bytes(4064 - len(page))
show('F2. metadata-only container', page + hashlib.sha256(page).digest() + meta + bytes(4096 - len(meta)))

# G. required VALUE_EXACT bit set although no record is value-exact
show('G. header required bit 0 set, no value-exact record', patch(base, [(8, '<Q', 1)]))

# H. zero-size tensor of a known dtype placed with rank 0 / non-zero shape
z = rd.data_tensors
if z < nt:
    at = t0 + z * 96
    print('zero-size tensor:', rd.tensors[z][:4])
    show('H. zero-size tensor made opaque with shape (5, 5) and no zero dimension', patch(base, [(at + 6, '<B', 0), (at + 7, '<B', 2), (at + 8, '<Q', 5), (at + 16, '<Q', 5)]))

# I. same tensor bytes, two different record encodings -> same content digest, different root
import random
rng = random.Random(1)
data = bytes(rng.choices(range(256), [int(10000 * 0.8 ** abs(i - 120)) + 1 for i in range(256)], k=40000))
roots = []
for chunk in (10, 15):
    p = os.path.join(OUT, f'canon{chunk}.wex4')
    roots.append(wex4.write(p, [wex4.Record([wex4.Tensor('a', 'U8', (40000,), 40000)], 40000, lambda: data, ('bp', 1, 0))],
                            chunk_log2=chunk, encoder=C).hex()[:16])
    print(f'I. chunk_log2 {chunk}:', verdicts(open(p, 'rb').read()), 'root', roots[-1],
          'digest', wex4.check(open(p, 'rb').read(), True, C).hex()[:16])
