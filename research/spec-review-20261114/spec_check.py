# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""The checks behind the findings of the spec review of 2026-11-14 (README.md beside this file).

    bash wex4/build_and_test.sh                                   # builds wex4/out/ (the DLLs and the test containers)
    python -I research/spec-review-20261114/spec_check.py         # from the repository root

Standard library only. It holds a WNR3 decoder and a MurmurHash3 written from docs/WNR3-SPEC.md (sections 1 to 4
and errata 9 to 12), which it compares with the C reference decoder, and a few hand-made WEX4 files and WBP1 records
that it gives to both WEX4 readers. It writes only under wex4/out/spec-review/ (git-ignored)."""
import hashlib
import io
import os
import struct
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'wex4'))
import wex4  # noqa: E402

OUT = os.path.join(REPO, 'wex4', 'out')
TMP = os.path.join(OUT, 'spec-review')
os.makedirs(TMP, exist_ok=True)
CREF = wex4.CLib(os.path.join(OUT, 'wex4_tools_ref.dll'))
M32 = 0xffffffff
L = 1 << 23


# ---------- MurmurHash3_x86_32 written from WNR3-SPEC 4.5 (not wex4.stream_hash) ----------
def rotl(x, r):
    return ((x << r) | (x >> (32 - r))) & M32


def h32(key, seed):
    h = seed
    n = len(key)
    for i in range(0, n - n % 4, 4):
        k = int.from_bytes(key[i:i + 4], 'little')
        k = (k * 0xcc9e2d51) & M32
        k = rotl(k, 15)
        k = (k * 0x1b873593) & M32
        h ^= k
        h = rotl(h, 13)
        h = (h * 5 + 0xe6546b64) & M32
    if n % 4:
        t = key[n - n % 4:]
        k = int.from_bytes(t, 'little')
        k = (k * 0xcc9e2d51) & M32
        k = rotl(k, 15)
        k = (k * 0x1b873593) & M32
        h ^= k
    h ^= n
    h ^= h >> 16
    h = (h * 0x85ebca6b) & M32
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & M32
    h ^= h >> 16
    return h


def hash_vectors():
    table = b''
    for i in range(256):
        table += struct.pack('<I', h32(bytes(range(i)), 256 - i))
    assert h32(table, 0) == 0xB0F57EE3, hex(h32(table, 0))
    S = 0x33524E57
    assert h32(b'', 0) == 0 and h32(b'', S) == 0xE7F741F1 and h32(bytes(range(16)), S) == 0x31CE0919
    ident = struct.pack('<4I', 0x299DC586, 0x00270078, 2165, 0x0F787978)
    data = bytes((37 * i + 11) & 0x77 for i in range(1024))
    assert h32(ident + data, S) == 0x126485B2
    ident = struct.pack('<4I', 0x299DC586, 0x00270078, 719, 0x0F787978)
    assert h32(ident + bytes([0x7F, 0x7E, 0x80, 0x7F, 0x7D]), S) == 0x8A4976D8
    print('R0 hash vectors of WNR3-SPEC 4.5: all five reproduce (SMHasher value, 3 function values, 2 stream values)')


# ---------- WNR3 decoder written from the spec text ----------
class Bad(Exception):
    pass


def rans_stream(buf, pos, length, count, cdf_of, state_be=True):
    """count symbols; cdf_of(j, sofar) -> tuple of cumulative frequencies. Returns (symbols, final state)."""
    if length < 4:
        raise Bad('length')
    state = int.from_bytes(buf[pos:pos + 4], 'big' if state_be else 'little')
    if not L <= state < 256 * L:
        raise Bad('initial state')
    ptr, end = pos + 4, pos + length
    out = []
    for j in range(count):
        cdf = cdf_of(j, out)
        slot = state & 16383
        s = 0
        while cdf[s + 1] <= slot:
            s += 1
        state = (cdf[s + 1] - cdf[s]) * (state >> 14) + slot - cdf[s]
        while state < L:
            if ptr == end:
                raise Bad('underrun')
            state = (state << 8) | buf[ptr]
            ptr += 1
        out.append(s)
    if ptr != end:
        raise Bad('not consumed')
    return out, state


def wnr3_decode(rec, scale_shapes, model, expect_owner, expect_cz, mx_literal=False):
    magic, model_id, SP, WP, Cw, Cs = struct.unpack_from('<6I', rec)
    anchor, flags, owner = rec[24:27], rec[27], struct.unpack_from('<I', rec, 28)[0]
    if magic != 0x33524E57 or flags & 0xF0 or owner != expect_owner or bool(flags & 8) != expect_cz:
        raise Bad('header')
    # WNM2: 32-byte header, 3*567 CDFs of 17 u16, 3 CDFs of 257 u16; id = FNV-1a (0 -> 1)
    assert len(model) == 59408 and struct.unpack_from('<8I', model) == (0x324d4e57, 1, 14, 567, 3, 0, 0, 0)
    fnv = 2166136261
    for b in model:
        fnv = ((fnv ^ b) * 16777619) & M32
    if (fnv or 1) != model_id:
        raise Bad('model id')
    wcdf = lambda m, c: struct.unpack_from('<17H', model, 32 + (m * 567 + c) * 34)
    scdf = lambda m: struct.unpack_from('<257H', model, 32 + 3 * 567 * 34 + m * 514)
    scount = [r * c for r, c in scale_shapes]
    count = [32 * s for s in scount]
    for c, lo in ((Cw, 512), (Cs, 256)):
        if c < lo or c > 16384 or c & (c - 1):
            raise Bad('chunk')
    S = sum(-(-s // Cs) for s in scount)
    W = sum(-(-c // Cw) for c in count)
    lengths = struct.unpack_from(f'<{S + W}H', rec, 32)
    used = 32 + 2 * (S + W) + SP + WP
    if sum(lengths[:S]) != SP or sum(lengths[S:]) != WP or wex4.align(used) != len(rec) or any(rec[used:]):
        raise Bad('structure')
    word24 = struct.unpack_from('<I', rec, 24)[0]

    def h23(g, data):
        return h32(struct.pack('<4I', model_id, owner, g, word24) + bytes(data), 0x33524E57) & 0x7FFFFF

    pos, g = 32 + 2 * (S + W), 0
    cs = []
    for m in range(3):
        cdf, sc = scdf(m), []
        for first in range(0, scount[m], Cs):
            n = min(Cs, scount[m] - first)
            syms, state = rans_stream(rec, pos, lengths[g], n, lambda j, o: cdf)
            if state != L + h23(g, syms):
                raise Bad(f'scale stream {g} hash')
            sc += syms
            pos += lengths[g]
            g += 1
        cs.append(sc)
    cq = []
    for m in range(3):
        q = []
        for first in range(0, count[m], Cw):
            n = min(Cw, count[m] - first)
            ctx = {}

            def cdf_of(j, sofar, m=m, first=first, ctx=ctx):
                i = first + j
                if i % 32 == 0:
                    d = cs[m][i // 32] - anchor[m]
                    ctx['srel'] = max(-3, min(3, d)) + 3
                    ctx['pm'] = ctx['mx'] = 8
                else:
                    x = sofar[-1]
                    ctx['pm'] = x & 7
                    if mx_literal:
                        ctx['mx'] = max(ctx['mx'], x & 7)           # the sentence of 4.1 read literally
                    else:
                        ctx['mx'] = x & 7 if ctx['mx'] == 8 else max(ctx['mx'], x & 7)
                return wcdf(m, (ctx['srel'] * 9 + ctx['pm']) * 9 + ctx['mx'])
            syms, state = rans_stream(rec, pos, lengths[g], n, cdf_of)
            if flags & 8 and 8 in syms:
                raise Bad('CZ nibble 8')
            packed = bytes(syms[2 * k] | syms[2 * k + 1] << 4 for k in range(n // 2))
            if state != L + h23(g, packed):
                raise Bad(f'weight stream {g} hash')
            q += syms
            pos += lengths[g]
            g += 1
        cq.append(q)
    D = [0, 2, 4, 5, 6, 7]
    inverted = 0
    for m in range(3):
        if not flags >> m & 1:
            continue
        for b in range(scount[m]):
            blk = cq[m][32 * b:32 * b + 32]
            if max(x & 7 for x in blk) == 5:
                if cs[m][b] == 0:
                    raise Bad('ceil underflow')
                cq[m][32 * b:32 * b + 32] = [(x & 8) | D[x & 7] for x in blk]
                cs[m][b] -= 1
                inverted += 1
    slab = b''.join(bytes(s) for s in cs) + b''.join(
        bytes(q[2 * k] | q[2 * k + 1] << 4 for k in range(len(q) // 2)) for q in cq)
    return slab, flags, inverted, anchor


def wnr3_files():
    total = 0
    for path, limit in ((os.path.join(REPO, 'wex4', 'vectors', 'valid', 'tiny2-cz-upgraded.wex4'), 2),
                        (os.path.join(OUT, 'synth.wex4'), 4), (os.path.join(OUT, 'tiny_nr3_cz.wex4'), 3)):
        with open(path, 'rb') as f:
            rd = wex4.Reader(f, CREF)
            done = literal_fail = inv = 0
            flagset = set()
            for r, rec in enumerate(rd.records):
                if rec[3] != wex4.CODEC_WNR3 or done >= limit:
                    continue
                e = rd.expert_entry(r)
                stored, model = rd.stored(r), rd.blob(rec[5] - 1)
                want = rd.record(r)                      # the C reference decoder
                got, flags, inverted, anchor = wnr3_decode(stored, e['scale_shapes'], model, rec[6], bool(rec[4] & 1))
                assert got == want, (path, r)
                assert all(a <= 254 for a in anchor)
                try:
                    wnr3_decode(stored, e['scale_shapes'], model, rec[6], bool(rec[4] & 1), mx_literal=True)
                except Bad:
                    literal_fail += 1
                done += 1
                inv += inverted
                flagset.add(flags)
            total += done
            print(f'R1 {os.path.basename(path)}: {done} WNR3 records decoded by the decoder written from the spec, '
                  f'byte-identical to the C reference; record flags {sorted(flagset)}, ceil-inverted blocks {inv}; '
                  f'with "mx = max(mx, x & 7)" read literally {literal_fail} of {done} fail')
    return total


# ---------- WEX4 container cases ----------
def c_status(data):
    return CREF.check(data, 3, False)[0]


def py_status(data):
    try:
        wex4.check(data, True, CREF)
        return 0
    except wex4.FormatError as e:
        return str(e)


def reseal(data):
    data = bytearray(data)
    data[4064:4096] = hashlib.sha256(bytes(data[:4064])).digest()
    return bytes(data)


def both_bp(rec, decoded, owner):
    res = []
    for dec in (wex4.bp_decode, CREF.bp_decode):
        try:
            dec(rec, decoded, owner)
            res.append('accept')
        except wex4.FormatError:
            res.append('reject')
    return res


def wbp1_cases():
    owner, word = 7, 2 | 10 << 8
    byte = b'\x41'
    hsh = wex4.stream_hash((wex4.BP_MAGIC, owner, 0, word), byte)
    assert hsh == h32(struct.pack('<4I', wex4.BP_MAGIC, owner, 0, word) + byte, 0x33524E57)
    cdf = struct.pack('<257H', *[i * 64 for i in range(257)])

    def rec(mask):
        body = struct.pack('<IIIBBBB', wex4.BP_MAGIC, owner, 1, 2, 10, mask, 0) + (cdf if mask else b'') + \
            struct.pack('<I', hsh) + byte
        return body + bytes(4096 - len(body))
    print('W1 WBP1, decoded_bytes 1, P 2: mask 0 ->', both_bp(rec(0), 1, owner),
          '; mask bit 1 set on the empty plane 1 ->', both_bp(rec(2), 1, owner), '(Python, C)')
    import random
    rng = random.Random(1)
    data = bytes(rng.choices(range(256), [int(10000 * 0.8 ** abs(i - 120)) + 1 for i in range(256)], k=30000))
    good = wex4.bp_encode(data, 1, 12, owner)
    print('W2 WBP1 record as encoded ->', both_bp(good, len(data), owner), '; the same plus one zero page ->',
          both_bp(good + bytes(4096), len(data), owner))
    # W17: state is big-endian
    at = 16 + 514
    streams = -(-len(data) // 4096)
    first = at + 2 * streams
    be, le = int.from_bytes(good[first:first + 4], 'big'), int.from_bytes(good[first:first + 4], 'little')
    print(f'W17 first stream state of that record: big-endian 0x{be:08x} in [2^23, 2^31): {L <= be < 256 * L}; '
          f'little-endian 0x{le:08x} in range: {L <= le < 256 * L}')


def container_cases():
    T = wex4.Tensor
    p = os.path.join(TMP, 'c.wex4')

    def both(label, **kw):
        records = kw.pop('records', None)
        if records is None:
            records = [wex4.Record([T('a', 'U8', (5,), 5)], 5, lambda: b'hello')]
        try:
            wex4.write(p, records, **kw)
        except wex4.FormatError as e:
            print(label, '-> the Python writer/reader refuses:', e)
            return None
        data = open(p, 'rb').read()
        print(label, '-> C', c_status(data), ', Python', py_status(data))
        return data

    both('W6a opaque tensor, shape (3,), nbytes 0, in the zero-size tail', empty=[T('z', 99, (3,), 0)])
    both('W6b opaque tensor, shape (0, 5), nbytes 5, inside a record',
         records=[wex4.Record([T('o', 99, (0, 5), 5)], 5, lambda: b'hello')])
    both('W6c known dtype U8, shape (0,), nbytes 0, in the zero-size tail', empty=[T('z', 'U8', (0,), 0)])
    both('W7 blob of kind 99 with id 0xdeadbeef, and one of kind 99 with id 0',
         blobs=[(99, 0xdeadbeef, b'side data'), (99, 0, b'more')])
    data = both('W9 empty container (header page only)', records=[])
    bad = bytearray(data)
    struct.pack_into('<I', bad, 68, 5)
    bad = reseal(bad)
    print('W9 the same with max_decoded = 5, resealed -> C', c_status(bad), ', Python', py_status(bad))
    bad = bytearray(data)
    struct.pack_into('<Q', bad, 88, 5)
    bad = reseal(bad)
    print('W9 the same with total_decoded = 5, resealed -> C', c_status(bad), ', Python', py_status(bad))


def digest_case():
    path = os.path.join(REPO, 'wex4', 'vectors', 'valid', 'generic.wex4')
    data = open(path, 'rb').read()
    rd = wex4.Reader(io.BytesIO(data), CREF)
    h = hashlib.sha256()
    for name, dtype, rank, shape, record, count, offset, nbytes in rd.tensors:
        h.update(struct.pack('<H', len(name)) + name + struct.pack('<BB', dtype, rank))
        h.update(struct.pack('<8Q', *(list(shape) + [0] * (8 - rank))) + struct.pack('<Q', nbytes))
        h.update(rd.tensor(name)[2])
    status, cd = CREF.check(data, 3, False)
    assert status == 0
    print('W5 content digest of generic.wex4 from the field widths u16, bytes, u8, u8, 8 x u64, u64, data:',
          'equals the C digest' if h.digest() == cd else 'DIFFERS from the C digest',
          'and the Python digest' if h.digest() == rd.content_digest() else 'and DIFFERS from the Python digest')
    kinds = [b[0] for b in wex4.Reader(io.BytesIO(open(os.path.join(REPO, 'wex4', 'vectors', 'valid',
             'tiny2-cz-upgraded.wex4'), 'rb').read())).blobs]
    ids = [b[1] for b in wex4.Reader(io.BytesIO(open(os.path.join(REPO, 'wex4', 'vectors', 'valid',
           'tiny2-cz-upgraded.wex4'), 'rb').read())).blobs]
    print('W7 blobs of tiny2-cz-upgraded.wex4: kinds', kinds, 'ids', [hex(i) for i in ids])


def ceil_case():
    """A synthetic expert whose matrices 0 and 1 pass the ceil gate and matrix 2 does not, encoded by the C
    encoder (wex_nr3_tools.dll) and decoded by the decoder written from the spec; compared with the source."""
    import ctypes as C
    import random
    lib = C.CDLL(os.path.join(OUT, 'wex_nr3_tools.dll'))
    P, U32, SZ = C.c_void_p, C.c_uint32, C.c_size_t
    lib.wex_nr3_prepare.argtypes = [P, P, C.c_int, P, P]
    lib.wex_nr3_count_expert.argtypes = [P, P, P, P]
    lib.wex_nr3_build_model.argtypes = [P, P, P, P]
    lib.wex_nr3_encode_record.argtypes = [P, P, P, U32, U32, U32, U32, P, SZ, P, P]
    rng = random.Random(7)
    rows, cols = 64, 16
    nblocks = rows * cols
    scales, weights = [], []
    for m in range(3):
        sc, w = bytearray(), bytearray()
        for b in range(nblocks):
            kind = rng.random()
            s = rng.choice([118, 119, 120, 120, 121, 121, 122, 123])
            if kind < 0.6:      # even block, max 7
                mags = [rng.choice([0, 0, 2, 2, 2, 4, 4, 5, 6, 7]) for _ in range(32)]
                mags[rng.randrange(32)] = 7
                if b % 97 == 0:
                    s = 255      # kept as it is
                elif b % 89 == 0:
                    s = 254      # coded scale 255
                elif b % 83 == 0:
                    s = 0
            elif kind < 0.8:    # max 7 with odd codes: not even
                mags = [rng.choice([0, 1, 2, 3, 4, 7]) for _ in range(32)]
                mags[0], mags[1] = 7, 1
            elif kind < 0.9:    # small block, max <= 4
                mags = [rng.choice([0, 0, 1, 2, 3, 4]) for _ in range(32)]
            else:               # max 6
                mags = [rng.choice([0, 1, 2, 3, 6]) for _ in range(32)]
                mags[5] = 6
            if m == 2 and b == 10:   # one max-5 block: the gate of matrix 2 fails
                mags = [rng.choice([0, 1, 2, 5]) for _ in range(32)]
                mags[3] = 5
            q = [a | (8 if rng.random() < 0.5 else 0) for a in mags]
            sc.append(s)
            w += bytes(q[2 * k] | q[2 * k + 1] << 4 for k in range(16))
        scales.append(bytes(sc))
        weights.append(bytes(w))
    slab = b''.join(scales) + b''.join(weights)
    sb, wb = 3 * nblocks, 3 * nblocks * 16
    entry = struct.pack('<QIII6H6H3I3II', 0, wex4.align(len(slab)), sb, wb,
                        rows, cols, rows, cols, rows, cols, rows, cols * 16, rows, cols * 16, rows, cols * 16,
                        0, nblocks, 2 * nblocks, sb, sb + nblocks * 16, sb + 2 * nblocks * 16, 0)
    for cz in (0, 1):
        coded = C.create_string_buffer(len(slab))
        stats = (C.c_uint32 * 16)()
        flags = lib.wex_nr3_prepare(entry, slab, cz, coded, stats)
        assert flags >= 0
        wc, sc_ = (C.c_uint64 * (3 * 567 * 16))(), (C.c_uint64 * (3 * 256))()
        assert lib.wex_nr3_count_expert(entry, coded, wc, sc_) == 0
        blob, mid = C.create_string_buffer(59408), C.c_uint32()
        assert lib.wex_nr3_build_model(wc, sc_, blob, C.byref(mid)) == 0
        cap = 4 * len(slab) + (1 << 20)
        out, used, total = C.create_string_buffer(cap), C.c_size_t(), C.c_size_t()
        owner = 3 << 16 | 5
        rc = lib.wex_nr3_encode_record(blob.raw, entry, coded, flags, owner, 2048, 512, out, cap, C.byref(used), C.byref(total))
        assert rc == 0, rc
        record = out.raw[:total.value]
        got, rflags, inverted, anchor = wnr3_decode(record, [(rows, cols)] * 3, blob.raw, owner, bool(cz))
        expect = slab
        if cz:
            w = bytearray(slab[sb:])
            for i in range(len(w)):
                if w[i] & 15 == 8:
                    w[i] &= 0xf0
                if w[i] >> 4 == 8:
                    w[i] &= 0x0f
            expect = slab[:sb] + bytes(w)
        S = 3 * -(-nblocks // 512)
        Wn = 3 * -(-nblocks * 32 // 2048)
        print(f'R2 synthetic ceil record, cz={cz}: record flags 0x{rflags:02x}, {total.value} B of {wex4.align(len(slab))} raw, '
              f'{S} scale + {Wn} weight streams, C encoder stats transformed/max5/kept255 per matrix '
              f'{[stats[i] for i in range(9)]}; decoder from the spec inverted {inverted} blocks; '
              f'output == {"Z(source)" if cz else "source"}: {got == expect}')
        assert got == expect and rflags & 7 == 3 and inverted == stats[0] + stats[3]
        # a max-5 block at coded scale 0 in a ceil matrix: not constructible through the encoder; not run


hash_vectors()
wnr3_files()
ceil_case()
wbp1_cases()
container_cases()
digest_case()
print('done')
