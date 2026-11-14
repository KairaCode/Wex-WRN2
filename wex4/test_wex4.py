# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""WEX4 prototype tests: round trips, C/Python agreement, hostile files, mutation fuzzing.

    python -I wex4/test_wex4.py WORK_DIR [--fuzz N] [--write-vectors DIR]     (run by wex4/build_and_test.sh)

At the end every file of wex4/vectors/MANIFEST.tsv is checked (conformance vectors); --write-vectors DIR first
writes them anew from the files these tests build (a maintainer step, after a format change).

WORK_DIR holds wex4_tools.dll, wex4_tools_ref.dll, wex_nr3_tools.dll, tiny.wex (WEX1 of fixtures/v41-tiny), tiny_cz.wex (its
WNR3 canonical-zero container), tiny2_cz.wex (the same with 2 experts only, a small fuzz corpus) and tiny.wnm2
(the model of both). Standard library only."""
import concurrent.futures
import ctypes
import hashlib
import io
import json
import os
import random
import shutil
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wex4  # noqa: E402
import wex4_convert as conv  # noqa: E402

WORK = sys.argv[1]
FUZZ = int(sys.argv[sys.argv.index('--fuzz') + 1]) if '--fuzz' in sys.argv else 1000
C = wex4.CLib(os.path.join(WORK, 'wex4_tools.dll'))             # fast WNR3 decoder (AVX2 + OpenMP)
CREF = wex4.CLib(os.path.join(WORK, 'wex4_tools_ref.dll'))     # reference WNR3 decoder
FIXTURE = os.path.join(HERE, '..', 'fixtures', 'v41-tiny', 'model.safetensors')
VECTORS = os.path.join(HERE, 'vectors')
rng = random.Random(20261111)
UPGRADED = {}   # source file name -> its WEX4 form, filled by test_upgrade (vectors)


def bf16(n, sigma=0.05):
    return b''.join(struct.pack('<f', rng.gauss(0, sigma))[2:] for _ in range(n))


def skewed(n):
    return bytes(rng.choices(range(256), [int(10000 * 0.8 ** abs(i - 120)) + (i % 7 == 0) for i in range(256)], k=n))


def rejected(fn, *args):
    try:
        fn(*args)
    except wex4.FormatError:
        return True
    return False


def py_verdict(data):
    try:
        return 0, wex4.check(data, True, C)
    except wex4.FormatError:
        return -1, None


def test_sha():
    for n in list(range(0, 200)) + [4064, 4096, 65537]:
        data = rng.randbytes(n)
        out = ctypes.create_string_buffer(32)
        C.lib.wex4_sha256_bytes(data, n, out)
        assert out.raw == hashlib.sha256(data).digest(), n
    print('sha256: C == hashlib on 203 lengths')


def test_bp():
    cases = [skewed(30000), bf16(9000), rng.randbytes(20000), bytes(5000), b'\x7f' * 70000, skewed(1), skewed(7),
             bf16(40000, 3.0), struct.pack('<2000f', *[rng.gauss(0, 1) for _ in range(2000)])]
    cases += [skewed(n) for n in range(2, 10)]
    count = 0
    for data in cases:
        for planes in (1, 2, 4, 8):
            for chunk_log2 in (10, 12, 15):
                for transform in ((0, 1) if planes > 1 and len(data) % planes == 0 else (0,)):
                    a = wex4.bp_encode(data, planes, chunk_log2, 7, transform)
                    b = C.bp_encode(data, planes, chunk_log2, 7, transform)
                    assert a == b, (len(data), planes, chunk_log2, transform)
                    assert wex4.bp_decode(a, len(data), 7) == data and C.bp_decode(a, len(data), 7) == data
                    assert rejected(wex4.bp_decode, a, len(data), 8) and rejected(C.bp_decode, a, len(data), 8)   # owner
                    count += 1
    sample = skewed(30000)
    print(f'WBP1: {count} encodes byte-identical in C and Python, decoded exactly by both; '
          f'skewed bytes {len(wex4.bp_encode(sample, 1, 12, 0))}/{wex4.align(len(sample))} B')


def sample_records():
    a, b, c, d = skewed(20000), bf16(6000), rng.randbytes(9000), bf16(11000)
    f32 = struct.pack('<6000f', *[rng.gauss(0, 1) for _ in range(6000)])
    big = bf16(26000)   # split in 3 parts of 20000, 20000 and 12000 bytes
    parts = [big[i:i + 20000] for i in range(0, len(big), 20000)]
    T = wex4.Tensor
    records = [
        wex4.Record([T('a.u8', 'U8', (200, 100), 20000), T('b.bf16', 'BF16', (6000,), 12000)], 32000, lambda: a + b),
        wex4.Record([T('c.rand', 'U8', (9000,), 9000)], 9000, lambda: c, ('bp', 1, 0)),          # falls back to raw
        wex4.Record([T('a.skew', 'F8_E4M3', (20000,), 20000)], 20000, lambda: a, ('bp', 1, 0)),
        wex4.Record([T('d.bf16', 'BF16', (11, 1000), 22000), T('scalar', 'F32', (), 4)], 22004,
                    lambda: d + f32[:4], ('bp', 2, 1)),
        wex4.Record([T('big', 'BF16', (26000,), 52000)], 20000, lambda: parts[0], ('bp', 2, 1), parts=3),
        wex4.Record([], 20000, lambda: parts[1], ('bp', 2, 1)),
        wex4.Record([], 12000, lambda: parts[2], ('bp', 2, 0)),
        wex4.Record([T('f32', 'F32', (6000,), 24000), T('été', 'I64', (2,), 16), T('opaque', 99, (3, 3), 5)], 24021,
                    lambda: f32 + bytes(range(21)), ('bp', 4, 1)),
    ]
    empty = [T('empty', 'F32', (0, 7), 0), T('', 'U8', (0,), 0)]
    expect = {'a.u8': a, 'b.bf16': b, 'c.rand': c, 'a.skew': a, 'd.bf16': d, 'scalar': f32[:4], 'big': big,
              'f32': f32, 'été': bytes(range(16)), 'opaque': bytes(range(16, 21)), 'empty': b'', '': b''}
    return records, empty, expect


def test_container():
    records, empty, expect = sample_records()
    paths = [os.path.join(WORK, n) for n in ('t_py.wex4', 't_c.wex4', 't_last.wex4')]
    meta = b'{"note":"free-form bytes, never parsed by the C reader"}'
    roots = [wex4.write(paths[0], records, empty=empty, meta=meta, chunk_log2=10),
             wex4.write(paths[1], records, empty=empty, meta=meta, chunk_log2=10, encoder=C),
             wex4.write(paths[2], records, empty=empty, meta=meta, chunk_log2=10, index_first=False)]
    data = [open(p, 'rb').read() for p in paths]
    assert data[0] == data[1] and roots[0] == roots[1], 'C and Python writers must give the same file'
    assert roots[0] != roots[2]
    digests = set()
    for blob in data:
        rd = wex4.Reader(io.BytesIO(blob))
        for name, value in expect.items():
            assert rd.tensor(name)[2] == value, name
        assert rd.meta == meta
        codecs = [rec[3] for rec in rd.records]
        assert codecs[:3] == [0, 0, 6] and codecs.count(6) >= 4, codecs
        status, digest = C.check(blob, 3, True)
        assert status == 0 and digest == rd.content_digest() and digest == wex4.check(blob, True, C)
        digests.add(digest)
    assert len(digests) == 1, 'content digest must not depend on the layout'
    print(f'container: 12 tensors (multi-tensor, split, zero-size, scalar, opaque dtype) exact; '
          f'C and Python files identical ({len(data[0])} B); index-first and index-last agree')
    return data[0]


def st_write(path, tensors, metadata=None, pad=True):
    header, payload = {}, bytearray()
    if metadata is not None:
        header['__metadata__'] = metadata
    for name, dtype, shape, data in tensors:
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [len(payload), len(payload) + len(data)]}
        payload += data
    raw = json.dumps(header, separators=(',', ':')).encode()
    if pad:
        raw += b' ' * (-len(raw) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(raw)) + raw + payload)


def test_safetensors():
    src = os.path.join(WORK, 'st_src')
    os.makedirs(src, exist_ok=True)
    st_write(os.path.join(src, 'a.safetensors'), [
        ('w.bf16', 'BF16', (300, 200), bf16(60000)), ('bias', 'F32', (64,), struct.pack('<64f', *range(64))),
        ('fp8', 'F8_E4M3', (2000, 40), skewed(80000)), ('none', 'I32', (0,), b''), ('s', 'F64', (), struct.pack('<d', 2.5)),
        ('nib', 'F4', (6,), b'\x12\x34\x56'), ('tiny', 'U8', (3,), b'abc')], {'format': 'pt', 'k': 'v'})
    st_write(os.path.join(src, 'b.safetensors'), [('other.w', 'I16', (5000,), skewed(10000))], pad=False)
    sources = [os.path.join(src, n) for n in ('a.safetensors', 'b.safetensors')]
    for label, clib, part in (('python codec', None, 1 << 20), ('C codec, 16 KiB parts', C, 16 << 10)):
        out = os.path.join(WORK, 'st.wex4')
        conv.pack(out, sources, clib, part_bytes=part)
        back = os.path.join(WORK, 'st_back')
        results = conv.unpack(out, back, clib)
        assert all(ok for _, ok in results) and len(results) == 2
        for path in sources:
            assert open(path, 'rb').read() == open(os.path.join(back, os.path.basename(path)), 'rb').read()
        raw, packed = sum(map(os.path.getsize, sources)), os.path.getsize(out)
        print(f'safetensors round trip ({label}): 2 files rebuilt byte-identical; {raw} B -> {packed} B')
    for bad in (b'\x00' * 7, struct.pack('<Q', 1 << 40) + b'{}', struct.pack('<Q', 2) + b'[]',
                struct.pack('<Q', 17) + b'{"a":1,"a":2}    ', struct.pack('<Q', 2) + b'{}x'):
        path = os.path.join(src, 'bad.bin')
        open(path, 'wb').write(bad)
        assert rejected(conv.st_index, path)
    st_write(os.path.join(src, 'hole.safetensors'), [('x', 'U8', (4,), b'abcd')])
    blob = bytearray(open(os.path.join(src, 'hole.safetensors'), 'rb').read())
    blob[blob.index(b'[0,4]') + 1] = ord('1')   # data_offsets [1,4]: a hole at byte 0 and a wrong size
    open(os.path.join(src, 'hole.safetensors'), 'wb').write(blob)
    assert rejected(conv.st_index, os.path.join(src, 'hole.safetensors'))


def test_experts():
    """The whole tiny V4.1 model (634 tensors, 5 dtypes, 48 experts) as raw, WNR3 and WNR3-CZ expert records."""
    back = os.path.join(WORK, 'tiny_back')
    original = open(FIXTURE, 'rb').read()
    sizes = {}
    for mode, cz in (('raw', False), ('nr3', False), ('nr3', True)):
        nr3 = conv.Nr3(os.path.join(WORK, 'wex_nr3_tools.dll'), os.path.join(WORK, 'tiny.wnm2'), cz) if mode == 'nr3' else None
        out = os.path.join(WORK, f'tiny_{mode}{"_cz" if cz else ""}.wex4')
        conv.pack(out, [FIXTURE], C, mode, nr3)
        data = open(out, 'rb').read()
        status, digest = C.check(data, 3, True)
        assert status == 0 and digest == wex4.check(data, True, C)
        ok = conv.unpack(out, back, C)[0][1]
        rebuilt = open(os.path.join(back, 'model.safetensors'), 'rb').read()
        if cz:   # value-exact: the only differences are FP4 -0 nibbles (0x8) that became +0
            assert not ok and len(rebuilt) == len(original)
            diff = [(a, b) for a, b in zip(original, rebuilt) if a != b]
            assert diff and all((a ^ b) & ~0x88 == 0 and all((a >> s & 15) == 8 and (b >> s & 15) == 0
                                                              for s in (0, 4) if (a ^ b) >> s & 15) for a, b in diff)
        else:
            assert ok and rebuilt == original
        rd = wex4.Reader(io.BytesIO(data), C)
        sizes[(mode, cz)] = (len(data), sum(1 for r in rd.records if r[3] == 5), len(rd.records))
        assert mode == 'nr3' or sizes[(mode, cz)][1] == 0
    print(f'tiny V4.1 model ({len(original)} B safetensors): raw experts {sizes[("raw", False)][0]} B, '
          f'WNR3 {sizes[("nr3", False)][0]} B, WNR3-CZ {sizes[("nr3", True)][0]} B; '
          f'WNR3 records: {sizes[("nr3", False)][1]} without CZ (the others fell back to raw), '
          f'{sizes[("nr3", True)][1]} with CZ, of 48 experts; non-CZ files rebuild byte-identical, CZ differs only in -0 nibbles')
    return open(os.path.join(WORK, 'tiny_nr3_cz.wex4'), 'rb').read()


def test_wnr3_exact():
    """Non-CZ WNR3 records inside WEX4: compressible synthetic experts, byte-exact back to safetensors."""
    source = os.path.join(WORK, 'synth', 'model.safetensors')
    nr3 = conv.Nr3(os.path.join(WORK, 'wex_nr3_tools.dll'), os.path.join(WORK, 'synth.wnm2'), False)
    out, back = os.path.join(WORK, 'synth.wex4'), os.path.join(WORK, 'synth_back')
    conv.pack(out, [source], C, 'nr3', nr3)
    data = open(out, 'rb').read()
    rd = wex4.Reader(io.BytesIO(data), C)
    codecs = [rec[3] for rec in rd.records]
    assert codecs == [5] * 4 and not rd.required, codecs
    assert C.check(data, 3, True) == CREF.check(data, 3, True) and C.check(data, 3, True)[0] == 0
    assert conv.unpack(out, back, C) == [('model.safetensors', True)]
    assert open(os.path.join(back, 'model.safetensors'), 'rb').read() == open(source, 'rb').read()
    print(f'non-CZ WNR3 in WEX4: 4 WNR3 records, {os.path.getsize(source)} B safetensors -> {len(data)} B, '
          f'rebuilt byte-identical through the fast and the reference decoder')


def test_store():
    """The engine-side loader (wex4_store.h) on upgraded containers, and recovery from a torn header page."""
    for name, flags in (('tiny.wex', 0), ('tiny_cz.wex', 32 | 64), ('synth_nr3.wex', 32)):
        before = open(os.path.join(WORK, name), 'rb').read()
        work = os.path.join(WORK, 'store_' + name)
        shutil.copyfile(os.path.join(WORK, name), work)
        conv.upgrade(work)
        _, _, layers, experts, _, table_offset, _, _ = conv.WEX_HEADER.unpack_from(before)
        size = layers * experts * 72
        table, info = ctypes.create_string_buffer(size), (ctypes.c_uint32 * 5)()
        root = open(work, 'rb').read(4096)[4064:]
        present = sum(1 for i in range(layers * experts) if conv.ENTRY.unpack_from(before, table_offset + i * 72)[1])
        assert C.lib.wex4_store_probe(work.encode(), root, table, size, info) == 0
        assert table.raw == before[table_offset:table_offset + size] and info[0] == flags and info[4] == present
        assert (info[1] != 0) == bool(flags) and info[2] % 4096 == 0
        assert C.lib.wex4_store_probe(work.encode(), bytes(32), table, size, info) == -3          # another root pinned
        assert C.lib.wex4_store_probe(work.encode(), root, table, size - 72, info) == -13         # table over budget
        blob = bytearray(open(work, 'rb').read())
        blob[len(before) // 2] ^= 1                                                                # one record bit
        open(work, 'wb').write(blob)
        assert C.lib.wex4_store_probe(work.encode(), root, table, size, info) == -3 and info[4] < present
        blob[len(before) // 2] ^= 1
        blob[0:4096] = os.urandom(4096)                                                            # torn header page
        open(work, 'wb').write(blob)
        conv.downgrade(work)
        assert open(work, 'rb').read() == before
    print('engine loader: legacy table, flags, model and slab size from 3 upgraded containers; pinned root, table '
          'budget and first-read SHA-256 enforced; downgrade recovers a file whose header page is destroyed')


def test_upgrade():
    """A WNR3-CZ container and a WEX1 container become WEX4 in place and back, byte for byte."""
    for name in ('tiny.wex', 'tiny_cz.wex', 'tiny2_cz.wex'):
        src = os.path.join(WORK, name)
        work = os.path.join(WORK, 'up_' + name)
        shutil.copyfile(src, work)
        before = open(src, 'rb').read()
        info = conv.upgrade(work)
        data = UPGRADED[name] = open(work, 'rb').read()
        assert data[4096:len(before)] == before[4096:], 'upgrade must not touch anything but the header page'
        status, digest = C.check(data, 3, True)
        assert status == 0 and digest == wex4.check(data, True, C), status
        rd = wex4.Reader(io.BytesIO(data), C)
        _, _, layers, experts, _, table_offset, _, _ = conv.WEX_HEADER.unpack_from(before)
        table = ctypes.create_string_buffer(layers * experts * 72)
        assert C.lib.wex4_expert_table(rd.page, rd.file_bytes, rd.index, len(rd.index), table) == 0
        assert table.raw == before[table_offset:table_offset + len(table.raw)], 'legacy entry table must be reproduced'
        for r, rec in enumerate(rd.records):
            slab = bytearray(wex1_slabs[rec[6]])
            if name != 'tiny.wex':   # WNR3-CZ decodes to Z(WEX1 slab): every -0 nibble becomes +0
                for i in range(rd.expert_entry(r)['scale_bytes'], len(slab)):
                    b = slab[i]
                    slab[i] = (b & 0x0f if b >> 4 == 8 else b) & (0xf0 if b & 15 == 8 else 0xff)
            assert rd.record(r) == bytes(slab), r
        conv.downgrade(work)
        assert open(work, 'rb').read() == before, 'downgrade must restore the container exactly'
        print(f'upgrade {name}: +{info["added"]} B (index {info["index_bytes"]} B, {info["records"]} records), '
              f'records untouched, legacy table reproduced, every expert decoded exactly, downgrade exact')
    return data


wex1_slabs = {}


def load_wex1_slabs():
    data = open(os.path.join(WORK, 'tiny.wex'), 'rb').read()
    _, _, layers, experts, _, table_offset, _, _ = conv.WEX_HEADER.unpack_from(data)
    for i in range(layers * experts):
        e = conv.ENTRY.unpack_from(data, table_offset + i * 72)
        if e[1]:
            wex1_slabs[(i // experts) << 16 | i % experts] = data[e[0]:e[0] + e[2] + e[3]]


def reseal(blob):
    index_offset, index_bytes = struct.unpack_from('<QQ', blob, 32)
    if index_offset + index_bytes <= len(blob):
        blob[96:128] = hashlib.sha256(blob[index_offset:index_offset + index_bytes]).digest()
    blob[4064:4096] = hashlib.sha256(blob[:4064]).digest()
    return blob


def test_hostile(base):
    """Hand-made attacks. Header and index hashes are recomputed so that each one reaches the validators."""
    rd = wex4.Reader(io.BytesIO(base))
    nb, nr, nt = len(rd.blobs), len(rd.records), len(rd.tensors)
    rec0 = rd.index_offset + nb * 64
    ten0 = rec0 + nr * 72
    order0 = ten0 + nt * 96

    def patch(edits, seal=True):
        blob = bytearray(base)
        for at, fmt, value in edits:
            struct.pack_into(fmt, blob, at, value)
        return bytes(reseal(blob) if seal else blob)
    attacks = {
        'bad magic': patch([(0, '<I', 0x57455833)]),
        'header bit flip, hash not fixed': patch([(60, '<I', 1)], False),
        'major version 5': patch([(4, '<H', 5)]),
        'unknown required feature': patch([(8, '<Q', 2)]),
        'file_bytes larger than the file': patch([(24, '<Q', len(base) + 4096)]),
        'index_bytes huge (overflow)': patch([(40, '<Q', (1 << 64) - 1)]),
        'index_offset past the end': patch([(32, '<Q', len(base))]),
        'index_offset unaligned': patch([(32, '<Q', 4097)]),
        'record_count 2^32-1': patch([(52, '<I', 0xffffffff)]),
        'tensor_count + 1': patch([(56, '<I', nt + 1)]),
        'names_bytes 2^63': patch([(72, '<Q', 1 << 63)]),
        'layers without experts': patch([(60, '<I', 3)]),
        'expert grid with no expert record': patch([(60, '<I', 1), (64, '<I', 1)]),
        'expert grid 65536 x 65536': patch([(60, '<I', 65536), (64, '<I', 65536)]),
        'max_decoded lie': patch([(68, '<I', rd.max_decoded + 1)]),
        'total_decoded lie': patch([(88, '<Q', rd.total_decoded - 1)]),
        'record offset into the header page': patch([(rec0, '<Q', 0)]),
        'record offset past the end': patch([(rec0, '<Q', len(base))]),
        'record offset 2^63 (overflow)': patch([(rec0, '<Q', 1 << 63)]),
        'records overlap': patch([(rec0 + 72, '<Q', rd.records[0][0])]),
        'record stored_bytes 0': patch([(rec0 + 8, '<I', 0)]),
        'record stored_bytes unaligned': patch([(rec0 + 8, '<I', rd.records[0][1] + 1)]),
        'decompression bomb: decoded 2^30 in a 4096-byte WBP1 record': patch([(rec0 + 2 * 72 + 12, '<I', 1 << 30)]),
        'decoded above the cap': patch([(rec0 + 12, '<I', (1 << 30) + 1)]),
        'unknown record flag': patch([(rec0 + 18, '<H', 4)]),
        'value-exact flag on a raw record': patch([(rec0 + 18, '<H', 1)]),
        'raw record with a blob': patch([(rec0 + 20, '<I', 1)]),
        'record owner of another slot': patch([(rec0 + 24, '<I', 3)]),
        'record first_tensor wrong': patch([(rec0 + 28, '<I', 1)]),
        'record tensor_count too large': patch([(rec0 + 32, '<I', nt + 1)]),
        'tensor offset overlap': patch([(ten0 + 96 + 80, '<Q', 100)]),
        'tensor nbytes past the record': patch([(ten0 + 96 + 88, '<Q', 1 << 40)]),
        'tensor nbytes 2^64-1': patch([(ten0 + 88, '<Q', (1 << 64) - 1)]),
        'tensor shape product overflow': patch([(ten0 + 8, '<Q', 1 << 62), (ten0 + 16, '<Q', 1 << 62)]),
        'tensor rank 9': patch([(ten0 + 7, '<B', 9)]),
        'tensor in another record': patch([(ten0 + 72, '<I', 1)]),
        'tensor name offset out of the pool': patch([(ten0, '<I', 1 << 30)]),
        'tensor name length past the pool': patch([(ten0 + 4, '<H', 65535)]),
        'name order: index out of range': patch([(order0, '<I', nt)]),
        'name order: duplicate (not a permutation)': patch([(order0, '<I', struct.unpack_from('<I', base, order0 + 4)[0])]),
        'split tensor claims 100 parts': patch([(ten0 + 6 * 96 + 76, '<I', 100)]),
        'truncated file': base[:-4096],
        'appended bytes (polyglot tail)': base + bytes(4096),
        'only half a header page': base[:2048],
    }
    names = bytearray(base)
    names[order0 + nt * 4] = 0   # NUL in a name
    attacks['NUL byte in a tensor name'] = bytes(reseal(names))
    names[order0 + nt * 4] = 0xff
    attacks['tensor name not UTF-8'] = bytes(reseal(names))
    for label, blob in attacks.items():
        c_status = C.check(blob, 3, True)[0]
        p_status = py_verdict(blob)[0]
        assert c_status != 0 and p_status != 0, f'{label}: C {c_status}, Python {p_status}'
    accepted = {
        'minor version 9': patch([(6, '<H', 9)]),
        'unknown optional feature bits': patch([(16, '<Q', 0xff00)]),
        'bytes in the reserved header area': patch([(1000, '<Q', 0x1122334455667788)]),
    }
    for label, blob in accepted.items():
        assert C.check(blob, 3, True)[0] == 0 and py_verdict(blob)[0] == 0, label
    print(f'hostile files: {len(attacks)} hand-made attacks rejected by C and Python; '
          f'{len(accepted)} forward-compatible variations accepted by both')
    return attacks, accepted


C_STATUS = {0: '-', -1: 'WEX4_E_ARG', -2: 'WEX4_E_MAGIC', -3: 'WEX4_E_HASH', -4: 'WEX4_E_VERSION', -5: 'WEX4_E_FEATURE',
            -6: 'WEX4_E_HEADER', -7: 'WEX4_E_INDEX', -8: 'WEX4_E_BLOB', -9: 'WEX4_E_RECORD', -10: 'WEX4_E_TENSOR',
            -11: 'WEX4_E_NAME', -12: 'WEX4_E_LAYOUT', -13: 'WEX4_E_EXPERT', -14: 'WEX4_E_DATA', -15: 'WEX4_E_UNSUPPORTED'}


def vector_name(label):
    name = ''.join(c if c.isalnum() else '-' for c in label.lower())
    while '--' in name:
        name = name.replace('--', '-')
    return name.strip('-')


def write_vectors(out, generic, hostile):
    """Conformance vectors: the files these tests build, with their verdicts (wex4/vectors/README.md)."""
    attacks, accepted = hostile
    files = {'valid/generic.wex4': generic,
             'valid/generic-index-last.wex4': open(os.path.join(WORK, 't_last.wex4'), 'rb').read(),
             'valid/tiny-raw-upgraded.wex4': UPGRADED['tiny.wex'],
             'valid/tiny2-cz-upgraded.wex4': UPGRADED['tiny2_cz.wex'],
             'legacy/tiny2-cz.wex3': open(os.path.join(WORK, 'tiny2_cz.wex'), 'rb').read()}
    files.update({f'valid/{vector_name(k)}.wex4': v for k, v in accepted.items()})
    files.update({f'invalid/{vector_name(k)}.wex4': v for k, v in attacks.items()})
    review = os.path.join(HERE, 'review', 'security-c', 'findings')
    rows = ['# path\tverdict\tclass\troot\tsha256\tupgrade_of']
    for path in sorted(files) + sorted(f'../review/security-c/findings/{n}' for n in os.listdir(review) if n.endswith('.wex4')):
        data = files.get(path)
        if data is None:
            data = open(os.path.join(out, path), 'rb').read()
        else:
            os.makedirs(os.path.dirname(os.path.join(out, path)), exist_ok=True)
            with open(os.path.join(out, path), 'wb') as f:
                f.write(data)
        status = C.check(data, 3, True)[0]
        assert (status == 0) == (py_verdict(data)[0] == 0), path
        root = data[4064:4096].hex() if status == 0 else '-'
        source = 'legacy/tiny2-cz.wex3' if path == 'valid/tiny2-cz-upgraded.wex4' else '-'
        rows.append(f'{path}\t{"accept" if status == 0 else "reject"}\t{C_STATUS[status]}\t{root}\t'
                    f'{hashlib.sha256(data).hexdigest()}\t{source}')
    with open(os.path.join(out, 'MANIFEST.tsv'), 'w', newline='\n') as f:
        f.write('\n'.join(rows) + '\n')
    print(f'vectors written: {len(rows) - 1} rows in {out}')


def test_vectors(where=VECTORS):
    """Every row of MANIFEST.tsv: file hash, C and Python verdicts, C error class, root; upgrade_of reproduced."""
    rows = [l.split('\t') for l in open(os.path.join(where, 'MANIFEST.tsv'), encoding='utf-8').read().splitlines()
            if l and not l.startswith('#')]
    for path, verdict, cls, root, sha, source in rows:
        data = open(os.path.join(where, path), 'rb').read()
        assert hashlib.sha256(data).hexdigest() == sha, f'{path}: file changed'
        status = C.check(data, 3, True)[0]
        assert C_STATUS.get(status, status) == cls and (py_verdict(data)[0] == 0) == (verdict == 'accept'), path
        assert verdict == 'reject' or data[4064:4096].hex() == root == hashlib.sha256(data[:4064]).hexdigest(), path
        if source != '-':   # the in-place upgrade of the source gives this file byte for byte
            work = os.path.join(WORK, 'vector_upgrade.tmp')
            shutil.copyfile(os.path.join(where, source), work)
            conv.upgrade(work)
            assert open(work, 'rb').read() == data, f'{path}: upgrade of {source} differs'
    print(f'vectors: {len(rows)} files of wex4/vectors/MANIFEST.tsv, {sum(r[1] == "accept" for r in rows)} accepted '
          f'and {sum(r[1] == "reject" for r in rows)} rejected by C and Python as listed, hashes and roots match')


def test_fuzz(corpora):
    t0 = time.time()
    total = [0, 0, 0, 0]
    modes = ((0, 'any byte, hashes checked'), (1, 'compressed record bytes, SHA-256 off'),
             (2, 'header and index, hashes recomputed'))

    def fuzz_run(dll, base, mode, n):
        counts = (ctypes.c_uint64 * 4)()
        assert dll.lib.wex4_fuzz_run(base, len(base), 1000 + mode, n, mode, counts) == 0
        return list(counts)

    # every C fuzz run is independent (own seed, own guard-page buffer) and ctypes drops the GIL during the call:
    # run them all at once, then check them in the usual order, so the output stays the same
    jobs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        for label, base in corpora:
            for mode, _ in modes:
                n = max(100, FUZZ * (25 if mode != 1 else 5) * 150_000 // max(150_000, len(base)))   # same time per corpus
                jobs[label, mode] = (n, pool.submit(fuzz_run, C, base, mode, n), pool.submit(fuzz_run, CREF, base, mode, n))
    for label, base in corpora:
        base_status, base_digest = C.check(base, 3, True)
        assert base_status == 0 and CREF.check(base, 3, True) == (0, base_digest)
        for mode, what in modes:
            n, fast, ref = jobs[label, mode]
            counts, reference = fast.result(), ref.result()
            assert counts == reference, f'{label} mode {mode}: fast and reference decoders disagree'
            rejected_n, same, other, noop = counts
            assert mode == 2 or other == 0, f'{label} mode {mode}: {other} corrupted files accepted with wrong content'
            assert mode != 0 or same == 0, f'{label}: {same} corrupted files accepted'
            print(f'fuzz C [{label}] {what}: {n} mutations, {rejected_n} rejected, {same} accepted with exact content, '
                  f'{other} accepted as a different valid file, {noop} no-ops')
            for i in range(4):
                total[i] += counts[i]
        # differential: the same mutated bytes through the C reader (guard pages) and the Python reader
        agree = accepted = 0
        rd = wex4.Reader(io.BytesIO(base))
        for i in range(max(50, FUZZ * 150_000 // max(150_000, len(base)))):
            blob = bytearray(base)
            sealed = i % 2 == 0
            for _ in range(rng.randint(1, 3)):
                if sealed:
                    at = rng.choice([rng.randrange(0, 128), rd.index_offset + rng.randrange(rd.index_bytes)])
                else:
                    at = rng.randrange(len(blob))
                kind = rng.randrange(4)
                if kind == 0:
                    blob[at] ^= 1 << rng.randrange(8)
                elif kind == 1:
                    blob[at] = rng.randrange(256)
                elif kind == 2 and at + 4 <= len(blob):
                    struct.pack_into('<I', blob, at, rng.choice([0, 1, 0xffffffff, 0x7fffffff, 4096, 0x10000]))
                else:
                    blob[at] = (blob[at] + rng.choice([1, 255])) & 255
            if sealed:
                reseal(blob)
            elif i % 10 == 1:
                blob = blob[:rng.randrange(len(blob))]
            blob = bytes(blob)
            c_status, c_digest = C.check(blob, 3, True) if len(blob) else (-1, None)
            p_status, p_digest = py_verdict(blob)
            assert (c_status == 0) == (p_status == 0), f'{label} #{i}: C {c_status} vs Python {p_status}'
            if c_status == 0:
                assert c_digest == p_digest, f'{label} #{i}: accepted with different content'
                assert sealed or blob == base or c_digest == base_digest
                accepted += 1
            agree += 1
        print(f'fuzz differential [{label}]: {agree} mutated files, C and Python agree on every verdict '
              f'({accepted} accepted, same content digest)')
    print(f'fuzz total: {sum(total)} C mutations (each through the fast and the reference WNR3 decoder, same '
          f'verdicts), 0 silent corruptions; {time.time() - t0:.0f} s')


def main():
    load_wex1_slabs()
    test_sha()
    test_bp()
    generic = test_container()
    test_safetensors()
    tiny = test_experts()
    upgraded = test_upgrade()
    test_wnr3_exact()
    test_store()
    hostile = test_hostile(generic)
    test_fuzz([('generic', generic), ('upgraded WNR3-CZ, 2 experts', upgraded), ('tiny model, WNR3-CZ', tiny)])
    if '--write-vectors' in sys.argv:
        write_vectors(sys.argv[sys.argv.index('--write-vectors') + 1], generic, hostile)
    test_vectors()
    print('WEX4 prototype: all tests passed')


if __name__ == '__main__':
    main()
