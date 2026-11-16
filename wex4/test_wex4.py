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
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wex4  # noqa: E402
import wex4_convert as conv  # noqa: E402

WORK = sys.argv[1]
FUZZ = int(sys.argv[sys.argv.index('--fuzz') + 1]) if '--fuzz' in sys.argv else 1000
C = wex4.CLib(os.path.join(WORK, 'wex4_tools.dll'))             # fast WNR3 decoder (AVX2 or NEON, + OpenMP)
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


def refusal(fn, *args):
    """The message of the FormatError fn raises; '' when it raises none."""
    try:
        fn(*args)
    except wex4.FormatError as error:
        return str(error)
    return ''


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


def test_train_threads():
    """train (a WNM2 model from safetensors, no WEX1 container), pack --threads and unpack --sha-only on the qwen38
    profile, at the expert shapes of GLM-5.3-Flash (hidden 4096, moe_intermediate 2048: gate/up weight [2048, 2048]
    and scale [2048, 128], down weight [4096, 1024] and scale [4096, 64]) and on the small profile fixture."""
    dll = os.path.join(WORK, 'wex_nr3_tools.dll')
    glm = os.path.join(WORK, 'glm', 'model.safetensors')
    tensors = st_tensors(glm)
    shapes = {name.decode().split('.experts.0.')[1]: (dtype, shape) for name, (dtype, shape, _) in tensors.items()
              if b'.experts.0.' in name}
    assert shapes == {'gate_proj.weight': ('U8', (2048, 2048)), 'gate_proj.weight_scale': ('U8', (2048, 128)),
                      'up_proj.weight': ('U8', (2048, 2048)), 'up_proj.weight_scale': ('U8', (2048, 128)),
                      'down_proj.weight': ('U8', (4096, 1024)), 'down_proj.weight_scale': ('U8', (4096, 64))}, shapes
    models = {}
    for threads in (1, 3):
        out = os.path.join(WORK, f'glm_t{threads}.wnm2')
        ident, picks = conv.train(out, [glm], dll, 'qwen38', 80, 7, threads=threads)
        models[threads] = open(out, 'rb').read()
        assert picks == [(0, 0), (0, 1)] and wex4.wnm2_id(models[threads]) == ident
    assert models[1] == models[3], 'train: the model depends on the thread count'
    assert conv.sample_experts([(1, 2), (0, 5), (0, 1), (3, 3)], 2, 9) == conv.sample_experts([(3, 3), (0, 1), (0, 5), (1, 2)], 2, 9)
    assert len(set(conv.sample_experts([(0, e) for e in range(100)], 10, s)[0] for s in range(20))) > 1, 'the seed is used'
    grid = [(L, e) for L in range(3, 45) for e in range(288)]
    spread = conv.sample_experts(grid, 80, 20261116, spread=True)
    assert len(spread) == 80 == len(set(spread)) and {L for L, _ in spread} == set(range(3, 45)), 'spread: every layer'
    assert spread == conv.sample_experts(list(reversed(grid)), 80, 20261116, spread=True)
    assert conv.sample_experts([(1, 0), (1, 1), (2, 0)], 2, 3, spread=True) in ([(1, 0), (2, 0)], [(1, 1), (2, 0)])
    assert rejected(conv.train, os.path.join(WORK, 'none.wnm2'), [glm], dll, 'deepseek')   # no expert under that profile
    nr3 = conv.Nr3(dll, os.path.join(WORK, 'glm_t1.wnm2'), False)
    roots, files = {}, {}
    for threads in (1, 4):
        out = os.path.join(WORK, f'glm_t{threads}.wex4')
        report = {}
        roots[threads] = conv.pack(out, [glm], C, 'nr3', nr3, family='qwen38', report=report, threads=threads)
        files[threads] = open(out, 'rb').read()
        assert report == {'expert_records': 2, 'experts_as_tensors': 0}, report
    assert roots[1] == roots[4] and files[1] == files[4], 'pack --threads changes the file'
    rd = wex4.Reader(io.BytesIO(files[1]), C)
    assert [r[3] for r in rd.records if r[4] & wex4.REC_EXPERT] == [5, 5]
    assert C.check(files[1], 3, True) == CREF.check(files[1], 3, True) and C.check(files[1], 3, True)[0] == 0
    back = os.path.join(WORK, 'glm_back')
    shutil.rmtree(back, ignore_errors=True)
    digests = {}
    assert conv.unpack(os.path.join(WORK, 'glm_t1.wex4'), back, C, sha_only=True, digests=digests) == [('model.safetensors', True)]
    assert os.listdir(back) == [] and digests == {'model.safetensors': conv.file_sha256(glm)}
    small = os.path.join(WORK, 'prof', 'qwen38', 'model.safetensors')
    out = os.path.join(WORK, 'prof_q38_t2.wex4')
    model = os.path.join(WORK, 'prof_q38.wnm2')
    conv.train(model, [small], dll, 'qwen38', 4, 1, threads=2)
    conv.pack(out, [small], C, 'nr3', conv.Nr3(dll, model, False), family='qwen38', threads=2)
    assert conv.unpack(out, os.path.join(WORK, 'prof_q38_back'), C) == [('model.safetensors', True)]
    assert open(os.path.join(WORK, 'prof_q38_back', 'model.safetensors'), 'rb').read() == open(small, 'rb').read()
    print(f'train from safetensors, pack --threads 1/3/4, unpack --sha-only: GLM-5.3 expert shapes, 2 experts, '
          f'{os.path.getsize(glm)} B -> {len(files[1])} B, the same model and the same file for every thread count')


def st_tensors(path):
    """{name: (dtype, shape, bytes)} of a safetensors file."""
    _, start, tensors = conv.st_index(path)
    blob = open(path, 'rb').read()
    return {name: (dtype, shape, blob[start + begin:start + end]) for name, dtype, shape, begin, end in tensors}


def test_profiles():
    """Every naming profile on its synthetic fixture (fixtures/make_synth_mxfp4.py --names): pack with expert
    records and back, and the in-place upgrade of a WEX1 and a WNR3 container and back, byte for byte."""
    names = sorted(os.listdir(os.path.join(WORK, 'prof')))
    assert set(names) == set(conv.PROFILES) == set(conv.NAMES), 'a profile without a fixture, or a fixture without a profile'
    nr3 = conv.Nr3(os.path.join(WORK, 'wex_nr3_tools.dll'), os.path.join(WORK, 'synth.wnm2'), False)
    out, back, work = os.path.join(WORK, 'p.wex4'), os.path.join(WORK, 'pb'), os.path.join(WORK, 'p_up.wex')
    grid = [layer << 16 | expert for layer in range(2) for expert in range(3)]
    reference = [value for _, _, value in st_tensors(os.path.join(WORK, 'prof', 'deepseek', 'model.safetensors')).values()]
    legacy = open(os.path.join(WORK, 'prof.wex'), 'rb').read()   # the WEX1 container c/tools/repack_v4_wex.py made
    table_offset = conv.WEX_HEADER.unpack_from(legacy)[5]
    slabs = [legacy[e[0]:e[0] + e[2] + e[3]] for e in (conv.ENTRY.unpack_from(legacy, table_offset + i * 72) for i in range(6))]
    coded = 0
    for name in names:
        source = os.path.join(WORK, 'prof', name, 'model.safetensors')
        original, expect = open(source, 'rb').read(), st_tensors(source)
        assert [value for _, _, value in expect.values()] == reference, 'the fixtures must differ in names and dtypes only'
        assert all(conv.expert_pattern(name).match(tensor) for tensor in expect), name
        # 1. safetensors -> WEX4 with one EXPERT record per expert (raw, then WNR3) -> safetensors
        for mode in ('raw', 'nr3'):
            report = {}
            conv.pack(out, [source], C, mode, nr3 if mode == 'nr3' else None, family=name, report=report)
            data = open(out, 'rb').read()
            status, digest = C.check(data, 3, True)
            assert status == 0 and digest == wex4.check(data, True, C) and CREF.check(data, 3, True) == (0, digest), name
            rd = wex4.Reader(io.BytesIO(data), C)
            assert (rd.layers, rd.experts) == (2, 3) and [rec[6] for rec in rd.records] == grid, name
            assert all(rec[4] & wex4.REC_EXPERT for rec in rd.records) and report == {'expert_records': 6, 'experts_as_tensors': 0}
            assert all(rec[3] == 0 for rec in rd.records) if mode == 'raw' else any(rec[3] == 5 for rec in rd.records)
            coded += sum(rec[3] == 5 for rec in rd.records)
            assert [rd.record(r) for r in range(6)] == slabs, f'{name}: the slab is not scales w1 w2 w3, weights w1 w2 w3'
            assert conv.unpack(out, back, C) == [('model.safetensors', True)]
            assert open(os.path.join(back, 'model.safetensors'), 'rb').read() == original, name
        # 2. a WEX1 and a WNR3 container of the same experts -> WEX4 in place with this profile's names -> back
        for legacy in ('prof.wex', 'prof_nr3.wex'):
            before = open(os.path.join(WORK, legacy), 'rb').read()
            shutil.copyfile(os.path.join(WORK, legacy), work)
            conv.upgrade(work, family=name)
            data = open(work, 'rb').read()
            assert data[4096:len(before)] == before[4096:]
            status, digest = C.check(data, 3, True)
            assert status == 0 and digest == wex4.check(data, True, C) and CREF.check(data, 3, True) == (0, digest), name
            rd = wex4.Reader(io.BytesIO(data), C)
            assert len(rd.tensors) == len(expect) and [rec[6] for rec in rd.records] == grid, name
            for tensor, value in expect.items():   # the container answers to the checkpoint's own names
                assert rd.tensor(tensor) == value, (name, tensor)
            conv.downgrade(work)
            assert open(work, 'rb').read() == before, name
        # 3. a mixed checkpoint: layer 0 has no experts, layer 1 is MXFP4, layer 2 is float (three BF16 matrices)
        source = os.path.join(WORK, 'mix', name, 'model.safetensors')
        report = {}
        conv.pack(out, [source], C, 'raw', family=name, report=report)
        data = open(out, 'rb').read()
        assert C.check(data, 3, True) == (0, wex4.check(data, True, C))
        rd = wex4.Reader(io.BytesIO(data), C)
        owners = [rec[6] for rec in rd.records if rec[4] & wex4.REC_EXPERT]
        assert owners == [1 << 16, 1 << 16 | 1] and (rd.layers, rd.experts) == (2, 2), name
        assert report == {'expert_records': 2, 'experts_as_tensors': 2}, report
        assert conv.unpack(out, back, C) == [('model.safetensors', True)]
        assert open(os.path.join(back, 'model.safetensors'), 'rb').read() == open(source, 'rb').read(), name
    # another family's names: nothing is taken for an expert, and the file still comes back; a profile that
    # matches the scale tensors only (the same names with another weight suffix) is refused
    source = os.path.join(WORK, 'prof', 'qwen3-moe', 'model.safetensors')
    halves, fixture = 0, st_tensors(source)
    for name in names:
        if name == 'qwen3-moe':
            continue
        if any(conv.expert_pattern(name).match(tensor) for tensor in fixture):
            message = refusal(conv.pack, out, [source], C, 'raw', None, 64 << 20, name)
            assert message.startswith('L0 E0: incomplete expert (3 of 6 tensors)'), (name, message)
            halves += 1
            continue
        report = {}
        conv.pack(out, [source], C, 'raw', family=name, report=report)
        rd = wex4.Reader(open(out, 'rb'), C)
        assert not rd.layers and not any(rec[4] & wex4.REC_EXPERT for rec in rd.records) and not report['expert_records']
        rd.f.close()
    assert conv.unpack(out, back, C) == [('model.safetensors', True)]
    # the command line: the right profile is silent on stderr; a profile that finds no expert warns there;
    # --names without --experts is refused
    def run(*args):
        done = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'wex4_convert.py'), 'pack', out, source,
                               '--dll', os.path.join(WORK, 'wex4_tools.dll'), *args],
                              capture_output=True, text=True)
        return done.returncode, done.stdout, done.stderr
    code, text, errors = run('--experts', 'raw', '--names', 'qwen3-moe')
    assert code == 0 and text.startswith('6 expert records; 0 experts kept') and not errors, (code, text, errors)
    code, text, errors = run('--experts', 'raw')
    assert code == 0 and text.startswith('0 expert records; 0 experts kept') and errors.startswith('warning: '), (code, text, errors)
    code, text, errors = run('--names', 'qwen3-moe')
    assert code != 0 and '--names needs --experts' in errors, (code, text, errors)
    # AMD's export of DeepSeek under DeepSeek's own names: three of the six tensors match, which is refused
    quark = os.path.join(WORK, 'prof', 'deepseek-quark', 'model.safetensors')
    assert refusal(conv.pack, out, [quark], C, 'raw').startswith('L0 E0: incomplete expert (3 of 6 tensors)')
    # a compressed-tensors export under the profile of the Quark export of the same names: the scales match,
    # weight_packed does not, which is refused the same way
    assert halves == 1, halves   # qwen3-moe-ct, above
    pairs = [(ct, base) for ct in names for base in names if conv.PROFILES[ct][2] == 'weight_packed' and
             conv.PROFILES[base][:4] == conv.PROFILES[ct][:2] + ('weight', 'weight_scale')]
    assert len(pairs) == 6, pairs
    for ct, base in pairs:
        message = refusal(conv.pack, out, [os.path.join(WORK, 'prof', ct, 'model.safetensors')], C, 'raw', None, 64 << 20, base)
        assert message.startswith('L0 E0: incomplete expert (3 of 6 tensors)'), (ct, base, message)
    # tensors that only look like experts stay ordinary tensors: a name with something before or after the
    # pattern, an index with a leading zero, a dtype that is not MXFP4
    def slab(prefix, layer, expert, skip=(), dtype='U8', cols=2):
        return [(f'{prefix}model.layers.{layer}.mlp.experts.{expert}.{matrix}.{suffix}', dtype, (4, cols * 16 if k else cols),
                 skewed(4 * (cols * 16 if k else cols) * (1 if dtype == 'U8' else 2)))
                for matrix in ('gate_proj', 'down_proj', 'up_proj') for k, suffix in enumerate(('weight_scale', 'weight'))
                if (matrix, suffix) not in skip]
    source = os.path.join(WORK, 'near.safetensors')
    tensors = slab('', 0, 0) + slab('mtp.', 0, 0) + slab('', '00', 1) + slab('', 0, 3, dtype='I16', cols=1) + \
        slab('', 0, 5, [('up_proj', 'weight_scale')], dtype='BF16') + [(n + '.bias', d, s, v) for n, d, s, v in slab('', 0, 4)]
    second = os.path.join(WORK, 'near2.safetensors')
    st_write(source, tensors[1:])
    st_write(second, tensors[:1])   # one tensor of the good expert is in a second file
    report = {}
    conv.pack(out, [source, second], C, 'raw', family='qwen3-moe', report=report)
    data = open(out, 'rb').read()
    assert C.check(data, 3, True) == (0, wex4.check(data, True, C))
    rd = wex4.Reader(io.BytesIO(data), C)
    assert [rec[6] for rec in rd.records if rec[4] & wex4.REC_EXPERT] == [0] and (rd.layers, rd.experts) == (1, 1)
    assert report == {'expert_records': 1, 'experts_as_tensors': 2}, report
    assert conv.unpack(out, os.path.join(WORK, 'pb2'), C) == [('near.safetensors', True), ('near2.safetensors', True)]
    for file in ('near.safetensors', 'near2.safetensors'):
        assert open(os.path.join(WORK, 'pb2', file), 'rb').read() == open(os.path.join(WORK, file), 'rb').read()
    # a compressed-tensors export with an MXFP8 expert (an F8_E4M3 '.weight' with a U8 '.weight_scale') and a
    # BF16 one (no scale): under the -ct profile only the MXFP4 expert becomes a record, the others stay tensors
    mixed_ct = [(n + '_packed' if n.endswith('.weight') else n, d, s, v) for n, d, s, v in slab('', 0, 0)]
    mixed_ct += [(n, 'F8_E4M3' if n.endswith('.weight') else d, s, v) for n, d, s, v in slab('', 0, 1)]
    mixed_ct += [(n, 'BF16', (4, 32), skewed(256)) for n, d, s, v in slab('', 0, 2) if n.endswith('.weight')]
    ct_source = os.path.join(WORK, 'ct8.safetensors')
    st_write(ct_source, mixed_ct)
    report = {}
    conv.pack(out, [ct_source], C, 'raw', family='qwen3-moe-ct', report=report)
    data = open(out, 'rb').read()
    assert C.check(data, 3, True) == (0, wex4.check(data, True, C))
    rd = wex4.Reader(io.BytesIO(data), C)
    assert [rec[6] for rec in rd.records if rec[4] & wex4.REC_EXPERT] == [0], 'the MXFP4 expert only'
    assert report == {'expert_records': 1, 'experts_as_tensors': 1}, report
    assert conv.unpack(out, os.path.join(WORK, 'pb3'), C) == [('ct8.safetensors', True)]
    assert open(os.path.join(WORK, 'pb3', 'ct8.safetensors'), 'rb').read() == open(ct_source, 'rb').read()
    # an MXFP4 expert that is not a whole slab is refused, and no output file is left: a missing file (the
    # expert above without its second file), a missing tensor, a weight that is not 16 bytes per scale byte
    os.remove(out)
    assert refusal(conv.pack, out, [source], C, 'raw', None, 64 << 20, 'qwen3-moe').startswith('L0 E0: incomplete expert (5 of 6')
    st_write(source, slab('', 0, 0) + slab('', 1, 2, [('up_proj', 'weight_scale')]))
    assert refusal(conv.pack, out, [source], C, 'raw', None, 64 << 20, 'qwen3-moe').startswith('L1 E2: incomplete expert (5 of 6')
    wrong = slab('', 1, 0)
    wrong[1] = (wrong[1][0], 'U8', (4, 30), skewed(120))
    st_write(source, slab('', 0, 0) + wrong)
    assert refusal(conv.pack, out, [source], C, 'nr3', nr3, 64 << 20, 'qwen3-moe').startswith('L1 E0: expert tensor shape')
    assert not os.path.exists(out)
    assert rejected(conv.upgrade, work, True, 'no-such-profile') and rejected(conv.pack, out, [source], C, 'raw', None, 1 << 20, 'x')
    print(f'naming profiles ({", ".join(names)}): on a synthetic fixture each (2 layers x 3 experts, the same expert '
          f'bytes), pack with RAW and with WNR3 expert records ({coded} of {6 * len(names)} coded) and unpack byte-identical; '
          f'upgrade of a WEX1 and a WNR3 container gives the fixture\'s tensors by name, downgrade exact; '
          f'an expert split over two files is one record; a float layer and near-miss names stay ordinary tensors, and '
          f'so do an MXFP8 and a BF16 expert of a compressed-tensors export; '
          f'an MXFP4 expert with a missing tensor or a wrong shape is refused, and so are {len(pairs) + halves} fixtures '
          f'under a profile that matches their scale tensors only (weight against weight_packed); a profile that finds '
          f'no expert warns')


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

    names0 = order0 + nt * 4
    rec = rd.records
    order = struct.unpack_from('<2I', base, order0)

    def patch(edits, seal=True, record=None):
        blob = bytearray(base)
        for at, fmt, value in edits:
            struct.pack_into(fmt, blob, at, value)
        if record is not None:   # the edit is inside a record: its own SHA-256 follows, so the decoder is reached
            entry = rec0 + record * 72
            blob[entry + 36:entry + 68] = hashlib.sha256(blob[rec[record][0]:rec[record][0] + rec[record][1]]).digest()
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
        # hashes, padding and record data (the last three reach the decoders: WEX4_E_DATA)
        'non-zero byte in the index padding': patch([(rd.index_offset + rd.index_bytes, '<B', 1)], False),
        'index bit flip, hash not fixed': patch([(rec0 + 12, '<I', rec[0][2] ^ 1)], False),
        'record bit flip, hash not fixed': patch([(rec[0][0] + 100, '<B', base[rec[0][0] + 100] ^ 1)], False),
        'non-zero byte in the padding of a raw record, record hash fixed': patch([(rec[0][0] + rec[0][2], '<B', 1)], record=0),
        'WBP1 record header damaged, record hash fixed': patch([(rec[2][0] + 5, '<B', base[rec[2][0] + 5] ^ 0x40)], record=2),
        'WBP1 record stream damaged, record hash fixed':
            patch([(rec[2][0] + 9000, '<B', base[rec[2][0] + 9000] ^ 0x40)], record=2),
        # header and record entries
        'file size not a multiple of 4096': bytes(reseal(bytearray(base[:24] + struct.pack('<Q', len(base) + 1) +
                                                                    base[32:] + b'\0'))),
        'index offset 0': patch([(32, '<Q', 0)]),
        'meta_bytes above the cap': patch([(80, '<Q', (1 << 28) + 1)]),
        'non-zero reserved bytes after a record hash': patch([(rec0 + 68, '<I', 1)]),
        'record decoded_bytes 0': patch([(rec0 + 12, '<I', 0)]),
        'unknown codec on a record of raw size': patch([(rec0 + 16, '<H', 7)]),
        'WNR3 record without the expert flag': patch([(rec0 + 2 * 72 + 16, '<H', 5)]),
        'continuation record with a tensor': patch([(rec0 + 5 * 72 + 32, '<I', 1)]),
        'split tensor one part short': patch([(ten0 + 6 * 96 + 76, '<I', 2)]),
        'records out of file order': patch([(rec0, '<Q', rec[1][0]), (rec0 + 72, '<Q', rec[0][0])]),
        # tensor entries and names
        'tensor dtype and size disagree': patch([(ten0 + 6, '<B', 13)]),
        'tensor nbytes 0 for a non-empty shape': patch([(ten0 + 88, '<Q', 0)]),
        'tensor shape beyond its rank': patch([(ten0 + 8 + 16, '<Q', 1)]),
        'zero-size tensor that names a record': patch([(ten0 + 10 * 96 + 72, '<I', 1)]),
        'opaque tensor of zero bytes inside a record': patch([(ten0 + 9 * 96 + 88, '<Q', 0)]),
        'name order not sorted': patch([(order0, '<I', order[1]), (order0 + 4, '<I', order[0])]),
        'tensor names shorter than the name pool': patch([
            (ten0 + 10 * 96 + 4, '<H', 4), (ten0 + 11 * 96, '<I', struct.unpack_from('<I', base, ten0 + 11 * 96)[0] - 1)]),
        'tensor name with a UTF-8 surrogate': patch([(names0, '<B', 0xed), (names0 + 1, '<B', 0xa0), (names0 + 2, '<B', 0x80)]),
        'tensor name with an overlong UTF-8 form': patch([(names0, '<B', 0xc0), (names0 + 1, '<B', 0xae)]),
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


def test_hostile_experts(base):
    """Hand-made attacks on a container with a model blob and WNR3 expert records (the upgraded 2-expert file)."""
    rd = wex4.Reader(io.BytesIO(base), C)
    blob0 = rd.index_offset
    rec0 = blob0 + len(rd.blobs) * 64
    ten0 = rec0 + len(rd.records) * 72
    rec, model = rd.records, rd.blobs[0]
    assert model[0] == wex4.BLOB_WNM2 and len(rd.blobs) == 3 and [r[3] for r in rec] == [5, 5], 'base file changed'

    def patch(edits, seal=True, record=None):
        blob = bytearray(base)
        for at, fmt, value in edits:
            struct.pack_into(fmt, blob, at, value)
        if record is not None:
            entry = rec0 + record * 72
            blob[entry + 36:entry + 68] = hashlib.sha256(blob[rec[record][0]:rec[record][0] + rec[record][1]]).digest()
        return bytes(reseal(blob) if seal else blob)
    attacks = {
        # blob entries and blob bytes
        'WNM2 blob of the wrong size': patch([(blob0 + 16, '<Q', model[3] + 1)]),
        'WNM2 blob with id 0': patch([(blob0 + 4, '<I', 0)]),
        'WNM2 blob id differs from its model': patch([(blob0 + 4, '<I', model[1] ^ 1)]),
        'blob of zero bytes': patch([(blob0 + 64 + 16, '<Q', 0)]),
        'non-zero reserved bytes after a blob hash': patch([(blob0 + 56, '<Q', 1)]),
        'blob bit flip, hash not fixed': patch([(model[2] + 100, '<B', base[model[2] + 100] ^ 1)], False),
        'non-zero byte in the padding of a blob': patch([(model[2] + model[3], '<B', 1)], False),
        # WNR3 record entries
        'WNR3 record without a model blob': patch([(rec0 + 20, '<I', 0)]),
        'WNR3 record whose blob is not a WNM2 model': patch([(rec0 + 20, '<I', 2)]),
        'record blob index past the blob table': patch([(rec0 + 20, '<I', len(rd.blobs) + 1)]),
        'value-exact record without the required feature': patch([(8, '<Q', 0)]),
        'expert record stored as WBP1': patch([(rec0 + 16, '<H', 6)]),
        'raw expert record with a model blob': patch([(rec0 + 16, '<H', 0)]),
        # the expert grid and the six tensors of an expert
        'expert owner layer outside the grid': patch([(rec0 + 72 + 24, '<I', rd.layers << 16)]),
        'expert owner expert outside the grid': patch([(rec0 + 72 + 24, '<I', rd.experts)]),
        'expert grid smaller than the owners': patch([(64, '<I', 1)]),
        'expert owners not increasing': patch([(rec0 + 72 + 24, '<I', 0)]),
        'expert tensor of rank 1': patch([(ten0 + 7, '<B', 1), (ten0 + 8, '<Q', 256), (ten0 + 16, '<Q', 0)]),
        'expert scale tensor with another dtype': patch([(ten0 + 6, '<B', 3)]),
        'expert weight tensor with another dtype': patch([(ten0 + 3 * 96 + 6, '<B', 5)]),
        # record data against the index (they reach the WNR3 decoder: WEX4_E_DATA)
        'canonical-zero WNR3 data without the value-exact flag': patch([(rec0 + 18, '<H', 2)]),
        'WNR3 record owner differs from the owner in its data': patch([(rec0 + 24, '<I', 1), (rec0 + 72 + 24, '<I', 2)]),
        'WNR3 stream damaged, record hash fixed': patch([(rec[0][0] + 6000, '<B', base[rec[0][0] + 6000] ^ 1)], record=0),
    }
    for label, blob in attacks.items():
        c_status = C.check(blob, 3, True)[0]
        assert c_status != 0 and CREF.check(blob, 3, True)[0] == c_status and py_verdict(blob)[0] != 0, \
            f'{label}: C {c_status}, Python {py_verdict(blob)[0]}'
    print(f'hostile expert files: {len(attacks)} hand-made attacks on blobs, WNR3 records and the expert grid '
          f'rejected by C (fast and reference decoder) and Python')
    return attacks


# ---- WNR3 written in Python (docs/WNR3-SPEC.md sections 3 and 4, errata 9 to 12) ----
# The C encoder always writes the anchor of rule g and never a record a decoder must refuse. This one takes the
# chunk pair, the anchors, the flags and the model id as they are given, and computes contexts and stream hashes
# from them, so a record can break one rule and no other. A matrix is (scales, nibbles): one scale byte and 32
# FP4 nibbles per block.
NR3_HALVE = {0: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5}


def nr3_matrix(rng, blocks, kinds):
    """A matrix of `blocks` blocks drawn from `kinds`: 'even' (maximum 7, no magnitude 1 or 3: the ceil form
    halves it), 'odd' (maximum 6 or 7 with a 1 or a 3), 'low' (maximum 0 to 4), 'five' (maximum 5: its matrix
    cannot be in ceil form), 'zero'. Scales 254, 255 and 0 occur; some zeros are -0."""
    scales, nibbles = bytearray(), []
    for _ in range(blocks):
        kind = rng.choice(kinds)
        if kind == 'even':
            mags = rng.choices([0, 2, 4, 5, 6, 7], [60, 22, 9, 5, 2, 2], k=32)
            mags[rng.randrange(32)] = 7
        elif kind == 'odd':
            mags = rng.choices(range(8), [60, 20, 9, 5, 3, 1, 1, 1], k=32)
            mags[rng.randrange(16)] = 6 + rng.randrange(2)
            mags[16 + rng.randrange(16)] = rng.choice([1, 3])
        elif kind == 'low':
            mags = rng.choices(range(5), [60, 24, 9, 5, 2], k=32)
        elif kind == 'five':
            mags = rng.choices(range(6), [60, 22, 9, 5, 2, 2], k=32)
            mags[rng.randrange(32)] = 5
        else:
            mags = [0] * 32
        scales.append(rng.choices([120, 121, 122, 119, 254, 255, 0], [40, 30, 12, 9, 3, 3, 3])[0])
        nibbles += [m | (8 if rng.random() < 0.08 else 0) for m in mags]
    return bytes(scales), nibbles


def nr3_slab(matrices):
    """The decoded slab: scales of w1, w2, w3, then packed weights of w1, w2, w3 (symbol 2i in the low nibble)."""
    return b''.join(bytes(scales) for scales, _ in matrices) + \
        b''.join(bytes(q[k] | q[k + 1] << 4 for k in range(0, len(q), 2)) for _, q in matrices)


def nr3_anchor(scales):
    """Rule g (4.3): the lower value of the densest adjacent pair of scale values, first maximum."""
    h = [0] * 256
    for s in scales:
        h[s] += 1
    return max(range(255), key=lambda v: (h[v] + h[v + 1], -v))


def nr3_coded(matrices, cz):
    """Coded domain (4.2, 4.4): canonical zero, then the ceil form of every matrix that passes the gate.
    Returns (coded matrices, record flags)."""
    coded, flags = [], 8 if cz else 0
    for m, (scales, nibbles) in enumerate(matrices):
        scales = bytearray(scales)
        blocks = [[0 if cz and q == 8 else q for q in nibbles[b:b + 32]] for b in range(0, len(nibbles), 32)]
        tops = [max(q & 7 for q in block) for block in blocks]
        if 5 not in tops:
            flags |= 1 << m
            for b, block in enumerate(blocks):
                if tops[b] == 7 and scales[b] < 255 and not any(q & 7 in (1, 3) for q in block):
                    blocks[b] = [q & 8 | NR3_HALVE[q & 7] for q in block]
                    scales[b] += 1
        coded.append((bytes(scales), [q for block in blocks for q in block]))
    return coded, flags


def nr3_contexts(scales, nibbles, anchor):
    """The context of every weight symbol (4.1 with erratum 9)."""
    out, srel, pm, mx = [], 0, 8, 8
    for i, q in enumerate(nibbles):
        if i % 32 == 0:
            srel, pm, mx = max(-3, min(3, scales[i // 32] - anchor)) + 3, 8, 8
        out.append((srel * 9 + pm) * 9 + mx)
        pm = q & 7
        mx = pm if mx == 8 else max(mx, pm)
    return out


def nr3_model(experts):
    """A WNM2 blob (erratum 12) from the counts of coded experts: [(coded matrices, anchors)]."""
    weight_tables, scale_tables = b'', b''
    for m in range(3):
        counts, scale_counts = [[0] * 16 for _ in range(567)], [0] * 256
        for coded, anchors in experts:
            scales, nibbles = coded[m]
            for s in scales:
                scale_counts[s] += 1
            for q, context in zip(nibbles, nr3_contexts(scales, nibbles, anchors[m])):
                counts[context][q] += 1
        weight_tables += b''.join(struct.pack('<17H', *wex4.quantize(c)) for c in counts)
        scale_tables += struct.pack('<257H', *wex4.quantize(scale_counts))
    return struct.pack('<8I', 0x324d4e57, 1, 14, 567, 3, 0, 0, 0) + weight_tables + scale_tables


def nr3_encode(model, coded, flags, owner, weight_chunk=2048, scale_chunk=512, anchors=None, model_id=None):
    """The padded WNR3 record of coded matrices (3, 4.5, 4.6). anchors None: rule g. model_id None: the model's."""
    anchors = anchors or [nr3_anchor(scales) for scales, _ in coded]
    model_id = model_id or wex4.wnm2_id(model)
    word = anchors[0] | anchors[1] << 8 | anchors[2] << 16 | flags << 24
    streams, payload = [], [0, 0]
    for weights in (0, 1):
        for m, (scales, nibbles) in enumerate(coded):
            if weights:
                cdfs = {}
                symbols, chunk = nibbles, weight_chunk
                tables = [cdfs.setdefault(c, struct.unpack_from('<17H', model, 32 + (m * 567 + c) * 34))
                          for c in nr3_contexts(scales, nibbles, anchors[m])]
            else:
                symbols, chunk = scales, scale_chunk
                tables = [struct.unpack_from('<257H', model, 32 + 3 * 567 * 34 + m * 514)] * len(scales)
            for first in range(0, len(symbols), chunk):
                piece = symbols[first:first + chunk]
                data = bytes(piece[k] | piece[k + 1] << 4 for k in range(0, len(piece), 2)) if weights else bytes(piece)
                state = wex4.RANS_L + (wex4.stream_hash((model_id, owner, len(streams), word), data) & 0x7fffff)
                out = bytearray()
                for j in range(len(piece) - 1, -1, -1):   # last symbol first, as wex4.rans_encode
                    cdf, symbol = tables[first + j], piece[j]
                    start = cdf[symbol]
                    f = cdf[symbol + 1] - start
                    limit = ((wex4.RANS_L >> wex4.PRECISION) << 8) * f
                    while state >= limit:
                        out.append(state & 0xff)
                        state >>= 8
                    state = ((state // f) << wex4.PRECISION) + state % f + start
                out += state.to_bytes(4, 'little')
                out.reverse()
                streams.append(bytes(out))
                payload[weights] += len(out)
    body = struct.pack('<6I3BBI', 0x33524e57, model_id, payload[0], payload[1], weight_chunk, scale_chunk, *anchors,
                       flags, owner) + struct.pack(f'<{len(streams)}H', *map(len, streams)) + b''.join(streams)
    return body + bytes(-len(body) % wex4.PAGE)


def expert_record(name, shapes, slab, expert, coded=None, value_exact=False):
    """An EXPERT record for wex4.write: the six tensors of WEX4-SPEC 5.5 for scale tensors of `shapes`
    (rows, cols). coded: the WNR3 record to store (blob 1 is its model); None: a RAW record."""
    T = wex4.Tensor
    tensors = [T(f'{name}.w{m + 1}.scale', 'F8_E8M0', s, s[0] * s[1]) for m, s in enumerate(shapes)] + \
        [T(f'{name}.w{m + 1}.weight', 'I8', (s[0], 16 * s[1]), 16 * s[0] * s[1]) for m, s in enumerate(shapes)]
    codec = ('pre', wex4.CODEC_WNR3, lambda: coded) if coded else 'raw'
    return wex4.Record(tensors, len(slab), lambda: slab, codec,
                       wex4.REC_EXPERT | (wex4.REC_VALUE_EXACT if value_exact else 0), expert, 1 if coded else 0)


def test_spec_cases():
    """Files for sentences of the two specifications that the files above do not pin (issues 14 to 20 of the
    repository, impl/go-wex4/SPEC-NOTES.md part D). Returns ({label: accepted file}, {label: refused file});
    both become conformance vectors. Its random numbers are its own: the files above keep their bytes."""
    rs = random.Random(20261115)
    T = wex4.Tensor
    path = os.path.join(WORK, 'spec.wex4')
    accepted, attacks = {}, {}

    def written(records, **options):
        wex4.write(path, records, **options)
        return open(path, 'rb').read()

    def patched(base, edits, record=None):
        """base with (offset, format, values...) written into it; record: that record's SHA-256 is recomputed."""
        blob = bytearray(base)
        for at, fmt, *values in edits:
            struct.pack_into(fmt, blob, at, *values)
        rd = wex4.Reader(io.BytesIO(base))
        if record is not None:
            entry = rd.index_offset + len(rd.blobs) * 64 + record * 72
            offset, stored = rd.records[record][:2]
            blob[entry + 36:entry + 68] = hashlib.sha256(blob[offset:offset + stored]).digest()
        return bytes(reseal(blob))

    def tensor_at(base, i):
        rd = wex4.Reader(io.BytesIO(base))
        return rd.index_offset + len(rd.blobs) * 64 + len(rd.records) * 72 + i * 96

    # 1. zero-size tensors (WEX4-SPEC 5.3): nbytes = 0 decides, not the shape; the shape product is taken from
    #    the first dimension to the last and the first multiplication that overflows refuses the tensor
    some = bytes(rs.randrange(256) for _ in range(100))
    for label, dtype, shape in (('opaque tensor of 0 bytes with shape 3 x 3', 99, (3, 3)),
                                ('opaque tensor of 0 bytes of rank 0', 99, ()),
                                ('tensor shape 0 x 2^32 x 2^32', 'F32', (0, 1 << 32, 1 << 32)),
                                ('tensor shape 2^32 x 0 x 2^32', 'F32', (1 << 32, 0, 1 << 32))):
        accepted[label] = written([wex4.Record([T('data', 'U8', (100,), 100)], 100, lambda: some)],
                                  empty=[T('empty', dtype, shape, 0)])
    base = accepted['tensor shape 0 x 2^32 x 2^32']
    attacks['tensor shape 2^32 x 2^32 x 0'] = patched(base, [(tensor_at(base, 1) + 8, '<3Q', 1 << 32, 1 << 32, 0)])

    # 2. WBP1 (WEX4-SPEC 7.2): 4 and 8 planes with and without the rotation, the byte 255 in rANS-coded planes,
    #    sizes that are not a multiple of the plane count, and a record with one page of zeros too many
    small = [int(rs.gauss(0, 40)) for _ in range(5000)]   # negative values: their high bytes are 255
    planes = [('f32.steps', 'F32', 4, 1, struct.pack('<5000f', *(v / 8 for v in small))),
              ('f64.steps', 'F64', 8, 1, struct.pack('<5000d', *(v / 64 for v in small))),
              ('i64.small', 'I64', 8, 0, struct.pack('<5000q', *small)),
              ('i32.small', 'I32', 4, 0, struct.pack('<5000i', *small)),
              ('u8.high', 'U8', 1, 0, bytes(rs.choices([255, 254, 0, 1, 128], [50, 20, 15, 10, 5], k=9000)))]
    records = [wex4.Record([T(name, dtype, (len(data) // p,), len(data))], len(data), lambda data=data: data, ('bp', p, t))
               for name, dtype, p, t, data in planes]
    data = accepted['WBP1 records of 4 and 8 planes'] = written(records, chunk_log2=12)
    assert data == written(records, chunk_log2=12, encoder=C), 'C and Python WBP1 encoders differ'
    rd = wex4.Reader(io.BytesIO(data))
    seen = []
    for r, (name, _, p, t, source) in enumerate(planes):
        stored = rd.stored(r)
        assert rd.records[r][3] == wex4.CODEC_WBP1 and stored[12:14] == bytes([p, 12]) and stored[15] == t, name
        assert rd.tensor(name)[2] == source and C.bp_decode(stored, len(source), r) == source, name
        coded = wex4.rotate(source, p, True) if t else source
        seen.append([q for q in range(p) if stored[14] >> q & 1 and 255 in coded[q::p]])
    assert seen[2] and seen[3] and seen[4], f'no rANS-coded plane with the byte 255: {seen}'

    def ragged(size, p, noisy):
        """Compressible bytes with 255 frequent; plane `noisy` is random bytes (stored raw), None: no such plane."""
        out = bytearray(rs.choices([0, 1, 2, 255, 254, 7], [40, 20, 10, 20, 5, 5], k=size))
        for k in range(size if noisy is None else noisy, size, p):
            out[k] = rs.randrange(256)
        return bytes(out)
    sizes = [('r2', 2, ragged(5001, 2, None)), ('r4', 4, ragged(10001, 4, 1)), ('r8', 8, ragged(20003, 8, 0))]
    records = [wex4.Record([T(name, 'U8', (len(data),), len(data))], len(data), lambda data=data: data, ('bp', p, 0))
               for name, p, data in sizes]
    data = accepted['WBP1 record sizes not a multiple of the planes'] = written(records, chunk_log2=11)
    assert data == written(records, chunk_log2=11, encoder=C)
    rd = wex4.Reader(io.BytesIO(data))
    for r, (name, p, source) in enumerate(sizes):
        assert rd.records[r][3] == wex4.CODEC_WBP1 and rd.stored(r)[12] == p and len(source) % p, name
        assert rd.tensor(name)[2] == source and C.bp_decode(rd.stored(r), len(source), r) == source, name

    class ExtraPage:
        """The reference encoder with one page of zero bytes after the record."""
        @staticmethod
        def bp_encode(raw, planes, chunk_log2, owner, transform=0):
            return wex4.bp_encode(raw, planes, chunk_log2, owner, transform) + bytes(wex4.PAGE)
    source = ragged(10000, 1, None)
    attacks['WBP1 record with a zero page too many'] = written(
        [wex4.Record([T('x', 'U8', (10000,), 10000)], 10000, lambda: source, ('bp', 1, 0))], encoder=ExtraPage)

    # 3. WNR3 records the upgraded vector does not have (its two records are canonical zero, no ceil form,
    #    chunks 2048:512): plain, ceil form, canonical zero with the ceil form, other chunk pairs, anchors
    #    that are not those of rule g. One model, trained on these experts.
    shapes = [(12, 25), (25, 12), (10, 16)]
    blocks = [rows * cols for rows, cols in shapes]
    plain = [nr3_matrix(rs, n, ['even', 'odd', 'low', 'five', 'zero']) for n in blocks]
    ceil = [nr3_matrix(rs, n, ['even', 'even', 'odd', 'low', 'zero']) for n in blocks]
    mixed = [nr3_matrix(rs, n, ['even', 'even', 'odd', 'low', 'zero'] + ['five'] * (m == 1)) for m, n in enumerate(blocks)]
    coded = [nr3_coded(plain, False), nr3_coded(ceil, False), nr3_coded(mixed, True)]
    assert [flags for _, flags in coded] == [0, 7, 13], 'the synthetic experts changed'
    assert any(s == 254 and c == 255 for (before, _), (after, _) in zip(ceil, coded[1][0]) for s, c in zip(before, after)), \
        'no block that the ceil form takes from scale 254 to the coded scale 255'
    rule_g = [nr3_anchor(scales) for scales, _ in coded[1][0]]
    other = [255, 0, rule_g[2] + 2]   # no byte of it is the anchor of rule g
    assert all(a != g for a, g in zip(other, rule_g))
    model = nr3_model([(c, [nr3_anchor(scales) for scales, _ in c]) for c, _ in coded] + [(coded[1][0], other)])
    model_id = wex4.wnm2_id(model)
    stored = [nr3_encode(model, coded[0][0], 0, 0),                              # plain, 2048:512
              nr3_encode(model, coded[1][0], 7, 1, 512, 256),                    # ceil form, the smallest chunks
              nr3_encode(model, coded[2][0], 13, 2, 4096, 1024),                 # canonical zero and ceil form
              nr3_encode(model, coded[1][0], 7, 3, 16384, 16384, other)]         # the expert of record 1 again
    slabs = [nr3_slab(plain), nr3_slab(ceil),
             nr3_slab([(scales, [0 if q == 8 else q for q in nibbles]) for scales, nibbles in mixed]),   # Z(source)
             nr3_slab(ceil)]
    assert slabs[2] != nr3_slab(mixed)
    # the C encoder gives the same bytes wherever it can write the record (it always takes the anchor of rule g)
    open(os.path.join(WORK, 'spec.wnm2'), 'wb').write(model)
    entry = conv.expert_entry(shapes, [(rows, 16 * cols) for rows, cols in shapes])
    for r, (source, cz, chunks) in enumerate(((plain, 0, (2048, 512)), (ceil, 0, (512, 256)), (mixed, 1, (4096, 1024)))):
        lib = conv.Nr3(os.path.join(WORK, 'wex_nr3_tools.dll'), os.path.join(WORK, 'spec.wnm2'), cz).lib
        slab = nr3_slab(source)
        work, out = ctypes.create_string_buffer(len(slab)), ctypes.create_string_buffer(4 * len(slab))
        used, total = ctypes.c_size_t(), ctypes.c_size_t()
        flags = lib.wex_nr3_prepare(entry, slab, cz, work, None)
        assert flags == coded[r][1] and work.raw == nr3_slab(coded[r][0]), f'record {r}: coded domain'
        assert lib.wex_nr3_encode_record(model, entry, work, flags, r, *chunks, out, len(out), ctypes.byref(used),
                                         ctypes.byref(total)) == 0 and out.raw[:total.value] == stored[r], f'record {r}'
    records = [expert_record(f'expert.{r}', shapes, slabs[r], r, stored[r], r == 2) for r in range(4)]
    options = dict(blobs=[(wex4.BLOB_WNM2, model_id, model)], layers=1, experts=4, required=wex4.REQ_VALUE_EXACT)
    data = accepted['WNR3 ceil form, chunk pairs and anchors'] = written(records, **options)
    for lib in (C, CREF):
        rd = wex4.Reader(io.BytesIO(data), lib)
        assert [rec[3] for rec in rd.records] == [5] * 4 and [rd.record(r) for r in range(4)] == slabs
        assert [rd.stored(r)[27] for r in range(4)] == [0, 7, 13, 7]
    # issue 20: records 1 and 3 are different bytes (chunk pair, anchors, every stream) and decode to the same
    # slab, so a decoder takes the anchor and the chunk pair as the record gives them. Changing the anchor byte
    # of a finished record is another thing: its stream hashes include the byte, and the record is refused.
    assert stored[1] != stored[3] and stored[1][16:27] != stored[3][16:27] and slabs[1] == slabs[3]
    at = rd.records[3][0]
    moved = patched(data, [(at + 24, '<B', rule_g[0])], record=3)
    assert C.check(moved, 3, True)[0] == CREF.check(moved, 3, True)[0] == -14 and py_verdict(moved)[0] != 0
    # the chunk fields are outside the stream hashes (WEX4-SPEC 8.2): another legal scale chunk that gives the
    # same streams (every matrix has at most 300 scales) is the same slab
    moved = patched(data, [(at + 20, '<I', 512)], record=3)
    assert C.check(moved, 3, True) == CREF.check(moved, 3, True) == C.check(data, 3, True) and py_verdict(moved)[0] == 0

    # 4. one small expert, and one rule of WNR3-SPEC broken per file; contexts and stream hashes are computed
    #    for the record as it is written, so nothing else is wrong with it
    small_shapes = [(4, 21), (7, 12), (3, 28)]
    source = [nr3_matrix(rs, 84, ['even', 'even', 'odd', 'low', 'zero']) for _ in range(3)]
    good, flags = nr3_coded(source, False)
    zero, zero_flags = nr3_coded(source, True)
    assert flags == 7 and zero_flags == 15
    small_model = nr3_model([(good, [nr3_anchor(scales) for scales, _ in good])])
    small_id = wex4.wnm2_id(small_model)
    options = dict(blobs=[(wex4.BLOB_WNM2, small_id, small_model)], layers=1, experts=1)

    def one(coded_record, value_exact=False):
        data = written([expert_record('expert', small_shapes, nr3_slab(source), 0, coded_record, value_exact)],
                       required=wex4.REQ_VALUE_EXACT if value_exact else 0, **options)
        assert wex4.Reader(io.BytesIO(data)).records[0][3] == wex4.CODEC_WNR3, 'the WNR3 record is not smaller than raw'
        return data
    data = accepted['WNR3 one expert'] = one(nr3_encode(small_model, good, 7, 0))
    assert wex4.Reader(io.BytesIO(data), C).record(0) == nr3_slab(source) and len(data) == 73728
    at = wex4.Reader(io.BytesIO(data)).records[0][0]
    payloads = struct.unpack_from('<2I', data, at + 8)   # bytes of the scale streams, of the weight streams
    used = 32 + 2 * (3 + 6) + sum(payloads)              # 3 scale streams, 6 weight streams (2,688 nibbles per matrix)
    b = next(b for b in range(84) if max(q & 7 for q in good[0][1][32 * b:32 * b + 32]) == 5)   # a halved block
    underflow = [(good[0][0][:b] + b'\0' + good[0][0][b + 1:], good[0][1])] + good[1:]
    negative = [(zero[0][0], [8] + zero[0][1][1:])] + zero[1:]
    # the same coded data without the flag that makes it wrong is a valid record: the streams are right, and
    # what the two files below break is the ceil inverse and the canonical-zero rule
    for matrices, flags in ((underflow, 0), (negative, 7)):
        control = one(nr3_encode(small_model, matrices, flags, 0))
        assert C.check(control, 3, True)[0] == CREF.check(control, 3, True)[0] == 0 and py_verdict(control)[0] == 0
    control = one(nr3_encode(small_model, good, 7, 0, 512, 256))   # the smallest legal chunks
    assert C.check(control, 3, True) == CREF.check(control, 3, True) == C.check(data, 3, True)
    attacks.update({
        'WNR3 ceil-form block of maximum 5 at coded scale 0': one(nr3_encode(small_model, underflow, 7, 0)),
        'WNR3 canonical-zero record with a nibble 8': one(nr3_encode(small_model, negative, 15, 0), True),
        'WNR3 record with a reserved flag bit': one(nr3_encode(small_model, good, 7 | 16, 0)),
        'WNR3 record with a weight chunk of 256': one(nr3_encode(small_model, good, 7, 0, 256, 512)),
        'WNR3 record whose model id is not its blob id': one(nr3_encode(small_model, good, 7, 0, model_id=small_id ^ 1)),
        'WNR3 payload sizes differ from the stream lengths, record hash fixed':
            patched(data, [(at + 8, '<2I', payloads[0] + 1, payloads[1] - 1)], record=0),
        'non-zero byte in the padding of a WNR3 record, record hash fixed': patched(data, [(at + used, '<B', 1)], record=0),
        'WNR3 record with the magic WNR2, record hash fixed': patched(data, [(at, '<I', 0x32524e57)], record=0),
    })
    assert used < 4096 and not any(data[at + used:at + 4096])
    # a WNR3 record with one page of zeros too many: only where the raw record is at least two pages longer
    one_plain = nr3_encode(model, coded[0][0], 0, 0)
    assert len(one_plain) == 8192
    attacks['WNR3 record with a zero page too many'] = written(
        [expert_record('expert', shapes, slabs[0], 0, one_plain + bytes(wex4.PAGE))],
        blobs=[(wex4.BLOB_WNM2, model_id, model)], layers=1, experts=1)

    # 5. WEX4-SPEC 5.5 and 3: the weight tensor of matrix m is [rows, 16 * cols] of scale tensor m, the bound
    #    65535 holds for 16 * cols too, and a blob starts on a 4096-byte boundary
    raw = written([expert_record('expert', small_shapes, nr3_slab(source), 0)], layers=1, experts=1)
    attacks['expert weight tensor with other rows than its scale tensor'] = patched(
        raw, [(tensor_at(raw, 3) + 8, '<2Q', 8, 8 * 21)])   # [4, 336] as [8, 168]: the same bytes
    wide = [(16, 256), (1, 1), (1, 1)]
    raw = written([expert_record('expert', wide, bytes(rs.choices(range(8), k=17 * 4098)), 0)], layers=1, experts=1)
    assert C.check(raw, 3, True)[0] == 0 and py_verdict(raw)[0] == 0
    attacks['expert weight tensor of 65536 columns'] = patched(
        raw, [(tensor_at(raw, 0) + 8, '<2Q', 1, 4096), (tensor_at(raw, 3) + 8, '<2Q', 1, 65536)])
    raw = written([wex4.Record([T('data', 'U8', (100,), 100)], 100, lambda: some)], blobs=[(18, 0, b'opaque bytes')])
    rd = wex4.Reader(io.BytesIO(raw))
    attacks['blob offset not a multiple of 4096'] = patched(raw, [(rd.index_offset + 8, '<Q', rd.blobs[0][2] + 1)])
    # WEX4-SPEC 5.2 and 10, rule 6: a record of a codec the reader does not know may name any blob up to
    # blob_count. Both readers open the file and refuse only to decode that record. No vector: the manifest has no
    # verdict for a file that opens and cannot be verified in full (wex4/vectors/README.md)
    coded = written([wex4.Record([T('x', 'U8', (10000,), 10000)], 10000, lambda: bytes(10000), ('bp', 1, 0))],
                    blobs=[(18, 0, b'opaque bytes')])
    rd = wex4.Reader(io.BytesIO(coded))
    assert rd.records[0][3] == wex4.CODEC_WBP1 and rd.records[0][1] < wex4.align(10000)
    for codec in (7, 3):   # a reserved id, and the WNR1 id of WEX3, which WEX4 does not use
        unknown = patched(coded, [(rd.index_offset + 64 + 16, '<H', codec), (rd.index_offset + 64 + 20, '<I', 1)])
        index_offset, index_bytes = struct.unpack_from('<QQ', unknown, 32)
        index = unknown[index_offset:index_offset + wex4.align(index_bytes)]
        assert C.lib.wex4_validate(unknown[:wex4.PAGE], len(unknown), index, len(index)) == 0, codec
        assert wex4.Reader(io.BytesIO(unknown)).records[0][3:6] == (codec, 0, 1), codec
        assert C.check(unknown, 3, True)[0] == CREF.check(unknown, 3, True)[0] == -15, codec
        assert 'unsupported codec' in refusal(wex4.check, unknown, True, C), codec

    # 6. WEX4-SPEC 5.2: a WNR3 record has at least align(32 + 6 * streams) bytes, the streams counted at the
    #    longest chunk. Three matrices of 4096 x 32 scales have 8 + 256 streams each, 792 in all, so their record
    #    needs 8,192 bytes. It is written with 8,192 (a WNR3 header and zeros: no record of that size can hold
    #    its streams), then cut to 4,096: it is the last extent, and its size, the file size and the hashes are
    #    repaired, so the index is refused for that rule alone
    big = [(4096, 32)] * 3
    head = struct.pack('<6I3BBI', 0x33524e57, small_id, 0, 0, 16384, 16384, 120, 120, 120, 0, 0)
    roomy = written([expert_record('expert', big, bytes(17 * 3 * 4096 * 32), 0, head + bytes(8192 - len(head)))],
                    **options)
    rd = wex4.Reader(io.BytesIO(roomy))
    at, entry = rd.records[0][0], rd.index_offset + len(rd.blobs) * 64
    assert rd.records[0][1:4] == (8192, 17 * 3 * 4096 * 32, wex4.CODEC_WNR3) and at + 8192 == len(roomy)
    cut = bytearray(roomy[:at + 4096])
    struct.pack_into('<Q', cut, 24, len(cut))          # file_bytes
    struct.pack_into('<I', cut, entry + 8, 4096)       # stored_bytes
    cut[entry + 36:entry + 68] = hashlib.sha256(cut[at:]).digest()
    attacks['WNR3 record smaller than its streams need'] = bytes(reseal(cut))
    assert C.check(attacks['WNR3 record smaller than its streams need'], 3, True)[0] == -13

    # 7. WNR3-SPEC erratum 12: every table of a WNM2 blob is strictly increasing. In the weight table of a context
    #    in which no symbol is ever coded (pm = 8 comes only with mx = 8, erratum 9), symbol 7 gets the frequency 0.
    #    The blob's id, the record's model id and its stream hashes are those of the changed blob, and the record
    #    is the one the unchanged tables give, so only the check of the tables refuses the file
    broken = bytearray(small_model)
    at = 32 + ((3 * 9 + 8) * 9 + 0) * 34            # matrix 0, scale context 3, pm 8, mx 0
    struct.pack_into('<H', broken, at + 16, struct.unpack_from('<H', broken, at + 14)[0])   # cdf[8] = cdf[7]
    broken = bytes(broken)
    broken_id = 2166136261
    for byte in broken:
        broken_id = (broken_id ^ byte) * 16777619 & 0xffffffff
    broken_id = broken_id or 1
    assert rejected(wex4.wnm2_id, broken) and broken_id != small_id
    under_broken = nr3_encode(broken, good, 7, 0, model_id=broken_id)
    assert under_broken == nr3_encode(small_model, good, 7, 0, model_id=broken_id), 'the changed table is used'
    attacks['WNM2 blob with a table that is not increasing'] = written(
        [expert_record('expert', small_shapes, nr3_slab(source), 0, under_broken)],
        blobs=[(wex4.BLOB_WNM2, broken_id, broken)], layers=1, experts=1)
    assert C.check(attacks['WNM2 blob with a table that is not increasing'], 3, True)[0] == -8

    digests = set()
    for label, blob in accepted.items():
        status, digest = C.check(blob, 3, True)
        assert status == 0 and CREF.check(blob, 3, True) == (0, digest) and py_verdict(blob) == (0, digest), label
        digests.add(digest)
    assert len(digests) == len(accepted)
    for label, blob in attacks.items():
        status = C.check(blob, 3, True)[0]
        assert status != 0 and CREF.check(blob, 3, True)[0] == status and py_verdict(blob)[0] != 0, \
            f'{label}: C {status}, Python {py_verdict(blob)[0]}'
    data_class = [label for label, blob in attacks.items() if C.check(blob, 3, True)[0] == -14]
    print(f'specification cases: {len(accepted)} files accepted and {len(attacks)} refused by C (fast and reference '
          f'decoder) and Python, {len(data_class)} of them by a record decoder under a valid index; a WNR3 encoder '
          f'in Python gives the bytes of the C encoder on 3 records; two WNR3 records with other chunk pairs and '
          f'anchors decode to one slab; a record of an unknown codec that names a blob opens in both readers')
    return accepted, attacks


C_STATUS ={0: '-', -1: 'WEX4_E_ARG', -2: 'WEX4_E_MAGIC', -3: 'WEX4_E_HASH', -4: 'WEX4_E_VERSION', -5: 'WEX4_E_FEATURE',
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
    rows = ['# path\tverdict\tclass\troot\tsha256\tupgrade_of\tcontent']
    for path in sorted(files) + sorted(f'../review/security-c/findings/{n}' for n in os.listdir(review) if n.endswith('.wex4')):
        data = files.get(path)
        if data is None:
            data = open(os.path.join(out, path), 'rb').read()
        else:
            os.makedirs(os.path.dirname(os.path.join(out, path)), exist_ok=True)
            with open(os.path.join(out, path), 'wb') as f:
                f.write(data)
        status, digest = C.check(data, 3, True)
        python = py_verdict(data)
        assert (status == 0) == (python[0] == 0) and (status != 0 or python[1] == digest), path
        root = data[4064:4096].hex() if status == 0 else '-'
        source = 'legacy/tiny2-cz.wex3' if path == 'valid/tiny2-cz-upgraded.wex4' else '-'
        rows.append(f'{path}\t{"accept" if status == 0 else "reject"}\t{C_STATUS[status]}\t{root}\t'
                    f'{hashlib.sha256(data).hexdigest()}\t{source}\t{digest.hex() if status == 0 else "-"}')
    with open(os.path.join(out, 'MANIFEST.tsv'), 'w', newline='\n') as f:
        f.write('\n'.join(rows) + '\n')
    print(f'vectors written: {len(rows) - 1} rows in {out}')


def test_vectors(where=VECTORS):
    """Every row of MANIFEST.tsv: file hash, C and Python verdicts, C error class, root, content digest (C with the
    fast and with the reference WNR3 decoder, and Python); upgrade_of reproduced. The Python reader decodes WNR3
    records through the C library and RAW and WBP1 records itself: an accepted file without WNR3 records is read
    a second time without the library."""
    rows = [l.split('\t') for l in open(os.path.join(where, 'MANIFEST.tsv'), encoding='utf-8').read().splitlines()
            if l and not l.startswith('#')]
    pure = 0
    for path, verdict, cls, root, sha, source, content in rows:
        data = open(os.path.join(where, path), 'rb').read()
        assert hashlib.sha256(data).hexdigest() == sha, f'{path}: file changed'
        status, digest = C.check(data, 3, True)
        python = py_verdict(data)
        assert C_STATUS.get(status, status) == cls and (python[0] == 0) == (verdict == 'accept'), path
        assert CREF.check(data, 3, True)[0] == status, path
        assert verdict == 'reject' or data[4064:4096].hex() == root == hashlib.sha256(data[:4064]).hexdigest(), path
        assert content == (digest.hex() if verdict == 'accept' else '-'), f'{path}: content digest'
        assert verdict == 'reject' or digest == python[1] == CREF.check(data, 3, True)[1], f'{path}: content digest'
        if verdict == 'accept' and wex4.CODEC_WNR3 not in [rec[3] for rec in wex4.Reader(io.BytesIO(data)).records]:
            assert wex4.check(data, True) == digest, f'{path}: content digest of the Python decoders'
            pure += 1
        if source != '-':   # the in-place upgrade of the source gives this file byte for byte
            work = os.path.join(WORK, 'vector_upgrade.tmp')
            shutil.copyfile(os.path.join(where, source), work)
            conv.upgrade(work)
            assert open(work, 'rb').read() == data, f'{path}: upgrade of {source} differs'
    print(f'vectors: {len(rows)} files of wex4/vectors/MANIFEST.tsv, {sum(r[1] == "accept" for r in rows)} accepted '
          f'and {sum(r[1] == "reject" for r in rows)} rejected by C and Python as listed, hashes, roots and '
          f'content digests match ({pure} accepted files without WNR3 records also decoded without the C library)')


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
    test_profiles()
    test_train_threads()
    test_store()
    hostile = test_hostile(generic)
    expert_attacks = test_hostile_experts(upgraded)
    assert not set(expert_attacks) & set(hostile[0])
    hostile[0].update(expert_attacks)   # one vector per attack, whatever its base file
    spec_accepted, spec_attacks = test_spec_cases()
    assert not set(spec_attacks) & set(hostile[0]) and not set(spec_accepted) & set(hostile[1])
    hostile[0].update(spec_attacks)
    hostile[1].update(spec_accepted)
    test_fuzz([('generic', generic), ('upgraded WNR3-CZ, 2 experts', upgraded), ('tiny model, WNR3-CZ', tiny)])
    if '--write-vectors' in sys.argv:
        write_vectors(sys.argv[sys.argv.index('--write-vectors') + 1], generic, hostile)
    test_vectors()
    print('WEX4 prototype: all tests passed')


if __name__ == '__main__':
    main()
