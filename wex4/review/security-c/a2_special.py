# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""A2: hand-made structure attacks built field by field (valid hashes everywhere): expert grids, extreme expert
shapes, split tensors, layouts, names, dtypes, blobs, unknown codecs, value-exact rules, an expert "bomb".
expect True/False = what docs/WEX4-SPEC.md says; the three readers must also agree.
    python -I -B a2_special.py"""
import ctypes
import io
import json
import os
import random
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rvlib import *  # noqa: E402,F401,F403

rng = random.Random(5)
log = Log('a2')
notes = {}
ENTRY = struct.Struct('<QIII6H6H3I3II')


def validate3(blob):
    """Index-level verdicts only (wex4_validate in both DLLs, Reader() in Python)."""
    io_, ib = struct.unpack_from('<QQ', blob, 32)
    idx = blob[io_:io_ + al(ib)] if io_ + al(ib) <= len(blob) and ib < (1 << 29) else b''
    out = [lib.lib.wex4_validate(blob[:4096], len(blob), idx, len(idx)) for lib in (C, CREF)]
    try:
        wex4.Reader(io.BytesIO(blob))
        out.append(0)
    except wex4.FormatError:
        out.append(-1)
    return out


def table_check(label, blob):
    """wex4_expert_table against the Python expert_entry of every expert record."""
    rd = wex4.Reader(io.BytesIO(blob), C)
    for lib in (C, CREF):
        table = ctypes.create_string_buffer(rd.layers * rd.experts * 72)
        st = lib.lib.wex4_expert_table(rd.page, rd.file_bytes, rd.index, len(rd.index), table)
        want = bytearray(len(table.raw))
        for r, rec in enumerate(rd.records):
            if rec[4] & 2:
                e = rd.expert_entry(r)
                slot = (rec[6] >> 16) * rd.experts + (rec[6] & 0xffff)
                want[slot * 72:slot * 72 + 72] = ENTRY.pack(
                    e['file_offset'], e['total_bytes'], e['scale_bytes'], e['weight_bytes'],
                    *[d for s in e['scale_shapes'] for d in s], *[d for s in e['weight_shapes'] for d in s],
                    *e['scale_offsets'], *e['weight_offsets'], e['flags'])
        if st != 0 or table.raw != bytes(want):
            print(f'  !! expert table mismatch: {label} status {st}')
            log.bad.append(('EXPERT_TABLE', label, st, st, 0, '', ''))


def expert(shapes, record, owner, data=None, sdt=6, wdt=3, names='abcdef'):
    """6 tensors + a raw EXPERT record of scale shapes [(r, c)] * 3."""
    sizes = [r * c for r, c in shapes] + [r * c * 16 for r, c in shapes]
    total = sum(sizes)
    data = rng.randbytes(total) if data is None else data
    tens, at = [], 0
    for m in range(6):
        r, c = shapes[m % 3]
        tens.append(T(f'L{owner:08x}.{names[m]}', sdt if m < 3 else wdt, [r, c if m < 3 else 16 * c], record, at, sizes[m]))
        at += sizes[m]
    return tens, data


# ---- 1. expert grid not tied to the expert records ----
small = b'0123456789abcdef'
for layers, experts in ((65536, 65536), (1, 1), (65536, 1)):
    blob = build(records=[raw_rec(small, 0, 0, 1)], tensors=[T('x', 2, [16], 0, 0, 16)], layers=layers, experts=experts)
    st = log.tri(f'grid {layers}x{experts} with no expert record (spec 4: "0 when the file has no expert records")', blob,
                 expect=False)
    notes[f'grid_{layers}x{experts}_no_experts'] = dict(verdicts=st, file_bytes=len(blob),
                                                         expert_table_bytes=layers * experts * 72)
    if layers == 65536 and experts == 65536:
        open(os.path.join(FIND, 'grid_65536x65536_no_experts.wex4'), 'wb').write(blob)
tens, data = expert([(4, 2), (8, 1), (4, 2)], 0, 5 << 16 | 7)
blob = build(records=[raw_rec(data, 5 << 16 | 7, 0, 6, flags=2)], tensors=tens, layers=6, experts=8)
log.tri('raw expert L5 E7 in a 6x8 grid', blob, expect=True)
table_check('raw expert L5E7', blob)
blob = build(records=[raw_rec(data, 5 << 16 | 7, 0, 6, flags=2)], tensors=tens, layers=65536, experts=65536)
st = log.tri('one small raw expert, grid 65536x65536 (expert table = 2^32 x 72 B)', blob)
notes['grid_65536x65536_one_expert'] = dict(verdicts=st, file_bytes=len(blob), expert_table_bytes=(1 << 32) * 72)
open(os.path.join(FIND, 'grid_65536x65536_one_expert.wex4'), 'wb').write(blob)
for layers, experts, ok in ((5, 8, False), (6, 7, False), (6, 8, True), (0, 0, False)):
    log.tri(f'expert L5E7 in grid {layers}x{experts}', build(records=[raw_rec(data, 5 << 16 | 7, 0, 6, flags=2)],
                                                              tensors=tens, layers=layers, experts=experts), expect=ok)

# ---- 2. extreme expert shapes (raw slabs) ----
for shapes, ok in (([(65535, 1), (1, 4095), (1, 1)], True), ([(1, 1), (1, 1), (1, 1)], True),
                   ([(65535, 1), (65535, 1), (65535, 1)], True), ([(1, 4095), (1, 4095), (1, 4095)], True)):
    tens, data = expert(shapes, 0, 3)
    blob = build(records=[raw_rec(data, 3, 0, 6, flags=2)], tensors=tens, layers=1, experts=4)
    log.tri(f'extreme expert shapes {shapes}', blob, expect=ok)
    table_check(f'extreme {shapes}', blob)
base_t, base_d = expert([(4, 2), (8, 1), (4, 2)], 0, 3)


def ex_variant(label, edit, ok=False, **kw):
    tens = [dict(t) for t in base_t]
    edit(tens)
    log.tri('expert: ' + label, build(records=[raw_rec(base_d, 3, 0, 6, flags=2)], tensors=tens, layers=1, experts=4, **kw),
            expect=ok)


ex_variant('scale dtype I8', lambda t: t[0].update(dtype=3))
ex_variant('scale dtype U8', lambda t: t[0].update(dtype=2), ok=True)
ex_variant('weight dtype F8_E8M0', lambda t: t[3].update(dtype=6))
ex_variant('weight dtype U8', lambda t: t[5].update(dtype=2), ok=True)
ex_variant('weight dtype opaque 0', lambda t: t[3].update(dtype=0))
ex_variant('scale rank 1', lambda t: t[0].update(shape=[8]))
ex_variant('scale rank 3', lambda t: t[0].update(shape=[4, 2, 1]))
ex_variant('scale shape transposed', lambda t: t[0].update(shape=[2, 4]))
ex_variant('weight shape transposed', lambda t: t[3].update(shape=[32, 4]))
ex_variant('weight cols 8 x scale cols (same bytes, [8, 16])', lambda t: t[3].update(shape=[8, 16]))
ex_variant('w1/w3 scales swapped names only', lambda t: None, ok=True)
ex_variant('scale opaque dtype with shape [65536, 1]', lambda t: t[0].update(dtype=0, shape=[65536, 1]))
tens = base_t[:5]
tens[4] = dict(tens[4], nbytes=tens[4]['nbytes'] + base_t[5]['nbytes'], dtype=0)
log.tri('expert with 5 tensors', build(records=[raw_rec(base_d, 3, 0, 5, flags=2)], tensors=tens, layers=1, experts=4), expect=False)
# WBP1 / unknown codec on an expert, value-exact on raw
comp = bytes(len(base_d))
log.tri('expert with codec WBP1', build(records=[R(wex4.bp_encode(bytes(8192 + 136), 1, 12, 3), 8192 + 136, 6, 3, 0, 6, flags=2)],
                                        tensors=expert([(64, 2), (128, 1), (256, 1)], 0, 3)[0], layers=1, experts=4), expect=False)
log.tri('expert raw with VALUE_EXACT', build(records=[raw_rec(base_d, 3, 0, 6, flags=3)], tensors=base_t, layers=1, experts=4,
                                             required=1), expect=False)
# owners
t2, d2 = expert([(4, 2), (8, 1), (4, 2)], 1, 2)
t2b = [dict(t, record=1) for t in t2]
for o0, o1, ok in ((1, 2, True), (2, 2, False), (2, 1, False), (0, 1 << 16, True), (3, 0x10000 | 3, True), (4, 5, False)):
    ta, _ = expert([(4, 2), (8, 1), (4, 2)], 0, o0, names='abcdef')
    tb, _ = expert([(4, 2), (8, 1), (4, 2)], 1, o1, names='ghijkl')
    blob = build(records=[raw_rec(base_d, o0, 0, 6, flags=2), raw_rec(d2, o1, 6, 6, flags=2)], tensors=ta + tb, layers=2, experts=4)
    log.tri(f'expert owners {o0:#x}, {o1:#x} in a 2x4 grid', blob, expect=ok)
    if ok:
        table_check(f'owners {o0:#x} {o1:#x}', blob)

# ---- 3. split tensors ----
A, B_, C_ = rng.randbytes(5000), rng.randbytes(3000), rng.randbytes(100)


def split(label, nbytes, rcount, parts, ok, shape=None, **kw):
    recs = [raw_rec(p, i, 0, 1 if i == 0 else 0) for i, p in enumerate(parts)]
    for i, edits in kw.get('edit', {}).items():
        recs[i].update(edits)
    tens = [T('big', 2, [nbytes] if shape is None else shape, 0, kw.get('offset', 0), nbytes, rcount=rcount)]
    log.tri('split: ' + label, build(records=recs, tensors=tens + kw.get('more', [])), expect=ok)


split('3 parts exact', 8100, 3, [A, B_, C_], True)
split('2 parts exact', 8000, 2, [A, B_], True)
split('nbytes one short', 8099, 3, [A, B_, C_], False)
split('nbytes one long', 8101, 3, [A, B_, C_], False)
split('record_count 2 but 3 parts', 8100, 2, [A, B_, C_], False)
split('record_count 4 but 3 parts', 8100, 4, [A, B_, C_], False)
split('parts exhausted early (nbytes = first two)', 8000, 3, [A, B_, C_], False)
split('record_count 1 with nbytes > decoded', 8000, 1, [A, B_], False)
split('record_count 0 with data', 5000, 0, [A], False)
split('record_count 2^32-1', 8000, 0xffffffff, [A, B_], False)
split('continuation with owner of first', 8000, 2, [A, B_], False, edit={1: dict(owner=0)})
split('continuation first_tensor 1', 8000, 2, [A, B_], False, edit={1: dict(first=1)})
split('continuation flagged EXPERT', 8000, 2, [A, B_], False, edit={1: dict(flags=2)})
split('first part flagged EXPERT', 8000, 2, [A, B_], False, edit={0: dict(flags=2)})
split('split tensor offset 1', 8000, 2, [A, B_], False, offset=1)
recs = [raw_rec(A, 0, 0, 1), raw_rec(B_, 1, 0, 0), raw_rec(C_, 2, 1, 1)]
log.tri('split then a normal record', build(records=recs, tensors=[T('big', 2, [8000], 0, 0, 8000, rcount=2),
                                                                  T('c', 2, [100], 2, 0, 100)]), expect=True)
recs = [raw_rec(C_, 0, 0, 1), raw_rec(A, 1, 1, 1), raw_rec(B_, 2, 1, 0)]
log.tri('normal record then split', build(records=recs, tensors=[T('c', 2, [100], 0, 0, 100),
                                                                T('big', 0, [], 1, 0, 8000, rcount=2)]), expect=True)
recs = [raw_rec(A, 0, 0, 2), raw_rec(B_, 1, 0, 0)]
log.tri('split tensor shares its first record', build(records=recs, tensors=[T('a', 2, [10], 0, 0, 10), T('big', 0, [], 0, 10, 7990, rcount=2)]),
        expect=False)
recs = [raw_rec(A, 0, 0, 0)]
log.tri('first record is a continuation', build(records=recs, tensors=[]), expect=False)
log.tri('first record is a continuation (first_tensor 2^32-1)', build(records=[raw_rec(A, 0, 0xffffffff, 0)], tensors=[]), expect=False)

# ---- 4. layouts ----
recs = [raw_rec(A, 0, 0, 1), raw_rec(B_, 1, 1, 1), raw_rec(C_, 2, 2, 1)]
tens = [T('a', 2, [5000], 0, 0, 5000), T('b', 2, [3000], 1, 0, 3000), T('c', 2, [100], 2, 0, 100)]
blobs = [dict(kind=16, id=7, data=b'old header'), dict(kind=18, id=0, data=rng.randbytes(5000))]
for name, layout, ok in (
        ('index last', [('B', 0), ('R', 0), ('B', 1), ('R', 1), ('R', 2), 'I'], True),
        ('index in the middle', [('R', 0), 'I', ('R', 1), ('B', 0), ('B', 1), ('R', 2)], True),
        ('blobs after records', [('R', 0), ('R', 1), ('R', 2), ('B', 0), ('B', 1), 'I'], True),
        ('records out of table order', ['I', ('R', 1), ('R', 0), ('R', 2), ('B', 0), ('B', 1)], False),
        ('blobs out of table order', ['I', ('B', 1), ('B', 0), ('R', 0), ('R', 1), ('R', 2)], False)):
    log.tri('layout: ' + name, build(blobs, recs, tens, layout=layout), expect=ok)
info = {}
good = build(blobs, recs, tens, info=info)
for name, edit, ok in (
        ('two records at the same offset', lambda b: struct.pack_into('<Q', b, info['rec0'] + 72, info['offs'][('R', 0)]), False),
        ('record inside the index', lambda b: struct.pack_into('<Q', b, info['rec0'], 4096), False),
        ('blob offset = record offset', lambda b: struct.pack_into('<Q', b, info['blob0'] + 8, info['offs'][('R', 0)]), False),
        ('blob offset 0 (header page)', lambda b: struct.pack_into('<Q', b, info['blob0'] + 8, 0), False),
        ('blob bytes 0', lambda b: struct.pack_into('<Q', b, info['blob0'] + 16, 0), False),
        ('blob bytes 2^30 + 1', lambda b: struct.pack_into('<Q', b, info['blob0'] + 16, (1 << 30) + 1), False),
        ('blob bytes 4096 (same extent, hash differs)', lambda b: struct.pack_into('<Q', b, info['blob0'] + 16, 4096), False),
        ('blob kind 1 with 10 bytes', lambda b: struct.pack_into('<I', b, info['blob0'], 1), False),
        ('blob reserved 1', lambda b: struct.pack_into('<Q', b, info['blob0'] + 56, 1), False),
        ('record reserved 1', lambda b: struct.pack_into('<I', b, info['rec0'] + 68, 1), False),
        ('blob padding not zero', lambda b: b.__setitem__(info['offs'][('B', 0)] + 100, 1), False),
        ('raw record padding not zero', lambda b: b.__setitem__(info['offs'][('R', 0)] + 5000, 1), False),
        ('index padding not zero', lambda b: b.__setitem__(info['index_offset'] + info['index_bytes'], 1), False),
        ('raw record data bit flip', lambda b: b.__setitem__(info['offs'][('R', 1)] + 5, b[info['offs'][('R', 1)] + 5] ^ 1), False),
        ('blob data bit flip', lambda b: b.__setitem__(info['offs'][('B', 1)] + 5, b[info['offs'][('B', 1)] + 5] ^ 1), False)):
    b = bytearray(good)
    edit(b)
    log.tri('extent: ' + name, reseal(b), expect=ok)
log.tri('empty container (header page only)', build(), expect=True)
log.tri('metadata only', build(meta=b'x' * 5000), expect=True)
log.tri('zero-size tensors only', build(tensors=[T('e', 13, [0], 0, 0, 0, rcount=0), T('f', 0, [], 0, 0, 0, rcount=0)]), expect=True)
log.tri('blob only', build(blobs=blobs), expect=True)
log.tri('empty container with max_decoded 1', build(hdr=dict(max_decoded=1)), expect=False)
log.tri('index_offset 8192 with an empty index', pad(build(hdr=dict(index_offset=8192, file_bytes=8192))), expect=False)

# ---- 5. names ----
def named(label, names, ok, order=None):
    recs = [raw_rec(bytes(len(names)), 0, 0, len(names))]
    tens = [T(n, 2, [1], 0, i, 1) for i, n in enumerate(names)]
    log.tri('names: ' + label, build(records=recs, tensors=tens, name_order=order), expect=ok)


for seq, ok in ((b'\xc2\x80', True), (b'\xdf\xbf', True), (b'\xe0\xa0\x80', True), (b'\xed\x9f\xbf', True), (b'\xee\x80\x80', True),
                (b'\xef\xbf\xbf', True), (b'\xf0\x90\x80\x80', True), (b'\xf4\x8f\xbf\xbf', True),
                (b'\xc0\x80', False), (b'\xc1\xbf', False), (b'\xe0\x9f\xbf', False), (b'\xed\xa0\x80', False), (b'\xed\xbf\xbf', False),
                (b'\xf0\x8f\xbf\xbf', False), (b'\xf4\x90\x80\x80', False), (b'\xf5\x80\x80\x80', False), (b'\x80', False),
                (b'\xc2', False), (b'\xe0\xa0', False), (b'\xf0\x90\x80', False), (b'\xc2\x41', False), (b'\xe1\x80\x41', False),
                (b'\xf1\x80\x80\x41', False), (b'\xfe', False), (b'\xff', False), (b'\x00', False), (b'a\x00b', False),
                (b'\xf8\x88\x80\x80\x80', False), (b'\xe0\x80\x80', False), (b'\xf0\x80\x80\x80', False)):
    named(f'{seq.hex()} alone', [seq], ok)
    named(f'{seq.hex()} last of the pool after "a"', [b'a', b'zz' + seq], ok)
    named(f'{seq.hex()} split across two names', [seq[:1], seq[1:] or b'q'], False if seq[0] >= 0x80 else ok and seq[:1] != b'\x00'
          if seq != b'a\x00b' else False)
named('prefix order a, ab', [b'ab', b'a'], True)
named('duplicate names', [b'a', b'a'], False, order=[0, 1])
named('wrong order', [b'a', b'b'], False, order=[1, 0])
named('order repeats one tensor', [b'a', b'b'], False, order=[0, 0])
named('empty name and a', [b'', b'a'], True)
named('two empty names', [b'', b''], False, order=[0, 1])
named('65535-byte name', [b'n' * 65535, b'm'], True)
named('high bytes sort unsigned', [b'\xc3\xa9', b'z', b'\x7f'], True)

# ---- 6. dtypes and sizes ----
def sized(label, dtype, shape, nbytes, ok):
    log.tri(f'dtype: {label}', build(records=[raw_rec(bytes(max(nbytes, 1)), 0, 0, 1)] if nbytes else [raw_rec(b'z', 0, 0, 1)],
                                     tensors=[T('t', dtype, shape, 0, 0, nbytes)] if nbytes else
                                     [T('a', 0, [], 0, 0, 1), T('t', dtype, shape, 0, 0, 0, rcount=0)]), expect=ok)


sized('F4 x 2 = 1 byte', 18, [2], 1, True)
sized('F4 x 3 = 12 bits', 18, [3], 2, False)
sized('F6 x 4 = 3 bytes', 19, [4], 3, True)
sized('F6 x 3 = 18 bits', 20, [3], 3, False)
sized('C64 x 1', 17, [1], 8, True)
sized('BOOL scalar', 1, [], 1, True)
sized('F32 scalar with 0 bytes', 13, [], 0, False)
sized('opaque scalar with 0 bytes', 0, [], 0, True)
sized('dtype 23 (opaque) any size', 23, [1 << 20, 1 << 40], 7, True)
sized('dtype 23 (opaque), shape product overflows u64 (spec 5.4: shape not checked against size)', 23, [1 << 40, 1 << 40], 7, None)
sized('dtype 255 shape product overflow (opaque: not multiplied?)', 255, [1 << 40, 1 << 40], 7, None)
sized('U8 shape [2^31, 2^32, 0] zero-size', 2, [1 << 31, 1 << 32, 0], 0, True)
sized('U8 shape [2^32, 2^32, 0] (product overflows before the 0)', 2, [1 << 32, 1 << 32, 0], 0, False)
sized('U8 shape [2^33, 2^33, 0] (product overflows before the 0)', 2, [1 << 33, 1 << 33, 0], 0, False)
sized('U8 shape [0, 2^33, 2^33]', 2, [0, 1 << 33, 1 << 33], 0, True)
sized('F64 [2^58] elements*64 overflow', 16, [1 << 58], 8, False)
sized('rank 8 of ones', 2, [1] * 8, 1, True)

# ---- 7. codecs, flags, value-exact, model blob ----
src = open(os.path.join(NOUT, 'tiny_nr3_cz.wex4'), 'rb').read()
rd = wex4.Reader(io.BytesIO(src), C)
model, mid = rd.blob(0), rd.blobs[0][1]
er = [r for r, rec in enumerate(rd.records) if rec[4] & 2]


def nr3(r, record, **kw):
    rec = rd.records[r]
    tens = [T(t[0], t[1], list(t[3]), record, t[6], t[7]) for t in rd.tensors[rec[7]:rec[7] + 6]]
    d = R(rd.stored(r), rec[2], 5, rec[6], record * 6, 6, flags=3, blob=1)
    d.update(kw)
    return d, tens


mb = dict(kind=1, id=mid, data=model)
r0, t0 = nr3(er[0], 0)
r1, t1 = nr3(er[1], 1)
log.tri('2 WNR3-CZ experts', build([mb], [r0, r1], t0 + t1, layers=6, experts=8, required=1), expect=True)
table_check('2 WNR3 experts', build([mb], [r0, r1], t0 + t1, layers=6, experts=8, required=1))
for label, kw, ok in (
        ('header lacks the required VALUE_EXACT bit', dict(required=0), False),
        ('model id in the blob entry wrong', dict(blobs=[dict(mb, id=mid ^ 1)]), False),
        ('model blob kind 16', dict(blobs=[dict(mb, kind=16)]), False),
        ('two identical model blobs, record uses the second', dict(blobs=[mb, mb], recs=[dict(r0, blob=2), r1]), True),
        ('record blob 0', dict(recs=[dict(r0, blob=0), r1]), False),
        ('record blob 2 of 1', dict(recs=[dict(r0, blob=2), r1]), False),
        ('index owner differs from the record owner', dict(recs=[dict(r0, owner=0), dict(r1, owner=2)]), False),
        ('CZ record without the VALUE_EXACT flag', dict(recs=[dict(r0, flags=2), r1]), False),
        ('WNR3 without EXPERT', dict(recs=[dict(r0, flags=1), r1]), False),
        ('records swapped on disk (hash moved too)', dict(recs=[dict(r0, stored=r1['stored']), dict(r1, stored=r0['stored'])]), False),
        ('model blob one byte flipped, entry sha fixed', dict(blobs=[dict(mb, data=model[:5000] + bytes([model[5000] ^ 1]) + model[5001:])]), False),
        ('codec 5 record stored raw-sized', dict(recs=[dict(r0, stored=pad(r0['stored'] + bytes(4096))), r1]), False)):
    blob = build(kw.get('blobs', [mb]), kw.get('recs', [r0, r1]), t0 + t1, layers=6, experts=8, required=kw.get('required', 1))
    log.tri('WNR3: ' + label, blob, expect=ok)
for codec in (1, 2, 3, 4, 7, 255, 65535):
    blob = build(records=[R(bytes(4096), 5000, codec, 0, 0, 1)], tensors=[T('t', 2, [5000], 0, 0, 5000)])
    st = log.tri(f'unknown codec {codec}: full check must refuse', blob, expect=False)
    v = validate3(blob)
    notes[f'codec_{codec}_index_level'] = v
    if len(set(x == 0 for x in v)) != 1 or v[0] != 0:
        print(f'  !! unknown codec {codec}: index-level verdicts {v} (spec: the file is still valid)')
        log.bad.append(('VALIDATE', f'unknown codec {codec}', *v, '', ''))
log.tri('raw record + unknown-codec record with a blob index', build(blobs, [R(bytes(4096), 5000, 9, 0, 0, 1, blob=2)],
                                                                     [T('t', 2, [5000], 0, 0, 5000)]), expect=False)

# ---- 8. expert "bomb": ~1 GiB WNR3 expert declared by a 0.7 MB file (index-level only + timing) ----
shapes = [(65535, 900), (1, 1), (1, 1)]
tens, _ = expert(shapes, 0, 0, data=b'')
decoded = 17 * sum(r * c for r, c in shapes)
streams = sum((r * c + 16383) // 16384 + (r * c * 32 + 16383) // 16384 for r, c in shapes)
for stored, ok in ((al(32 + 6 * streams), True), (al(32 + 6 * streams) - 4096, False)):
    body = bytearray(stored)
    struct.pack_into('<IIIIII', body, 0, 0x33524e57, mid, 4 * (streams - 115200 - 2), 4 * 115200 + 8, 16384, 16384)   # plausible header
    blob = build([mb], [R(bytes(body), decoded, 5, 0, 0, 6, flags=2, blob=1)], tens, layers=1, experts=1)
    t = time.time()
    v = validate3(blob)
    notes[f'expert_bomb_stored_{stored}'] = dict(decoded=decoded, file_bytes=len(blob), index_verdicts=v,
                                                 seconds=round(time.time() - t, 3))
    if [x == 0 for x in v] != [ok] * 3:
        print(f'  !! expert bomb stored {stored}: {v}, expected accept={ok}')
        log.bad.append(('VALIDATE', f'expert bomb {stored}', *v, '', ''))
    if ok:
        t = time.time()
        st = [cv(lib, blob, 2)[0] for lib in (C, CREF)]
        notes['expert_bomb_full_check'] = dict(verdicts=st, seconds=round(time.time() - t, 3))
        if st[0] == 0 or st[1] == 0:
            log.bad.append(('BOMB', 'accepted', *st, 0, '', ''))

# ---- 9. validator time on a large index (1,000,000 zero-size tensors, 103 MB of index) ----
n = 1_000_000
tens = [T(b'%07d' % i, 0, [], 0, 0, 0, rcount=0) for i in range(n)]
blob = build(tensors=tens, name_order=list(range(n)))
t = time.time()
io_, ib = struct.unpack_from('<QQ', blob, 32)
st = C.lib.wex4_validate(blob[:4096], len(blob), blob[io_:io_ + al(ib)], al(ib))
notes['large_index'] = dict(tensors=n, index_bytes=ib, status=st, seconds=round(time.time() - t, 3))

res = log.done()
res['notes'] = notes
json.dump(res, open(os.path.join(OUT, 'a2_special.json'), 'w'), indent=1, default=str)
print(json.dumps(notes, indent=1, default=str))
