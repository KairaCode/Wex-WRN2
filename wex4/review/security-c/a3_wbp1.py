# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""A3: WBP1 (codec 6) records made by hand with valid stream hashes.
 1. valid but non-canonical records (forced rANS / raw planes, flat / spiked / adversarial CDFs, 1..9 bytes in 8
    planes, every chunk size): C (guard pages around record and output, out/rv_harness.dll), both product DLLs and
    the pure Python decoder must accept and return the same bytes;
 2. hostile records (bad CDFs, lengths, masks, plane counts, transform, padding, sizes): all must refuse;
 3. exact-fit record (payload ends at the last byte of the last page = end of file = guard page) in a container;
 4. byte mutations of WBP1 records inside a container, record SHA-256 off (flags 2): C fast, C ref, pure Python;
 5. wex4_find against the Python name table; the WNR3 chunk-field malleability shown with the product API.
    python -I -B a3_wbp1.py [mutations]"""
import ctypes
import io
import json
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rvlib import *  # noqa: E402,F401,F403

MUT = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
rng = random.Random(6)
H = ctypes.CDLL(os.path.join(OUT, 'rv_harness.dll'))
H.rv_bp_guard.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
H.rv_find.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
H.rv_find.restype = ctypes.c_int64
problems, stats = [], {}


def guard_bp(rec, decoded, owner):
    out = ctypes.create_string_buffer(max(decoded, 1))
    try:
        st = H.rv_bp_guard(bytes(rec), len(rec), decoded, owner, out)
    except OSError as e:
        return -999, repr(e)
    return st, out.raw[:decoded] if st == 0 else None


def four(label, rec, decoded, owner, expect, data=None):
    """guard harness, fast DLL, ref DLL, pure Python on one record."""
    res = [guard_bp(rec, decoded, owner), c_bp(C, rec, decoded, owner), c_bp(CREF, rec, decoded, owner), py_bp(rec, decoded, owner)]
    acc = [r[0] == 0 for r in res]
    bad = None
    if any(r[0] in (-999, -99, -100) for r in res):
        bad = 'CRASH/EXCEPTION'
    elif len(set(acc)) != 1:
        bad = 'DISAGREE'
    elif acc[0] and len({bytes(r[1]) for r in res}) != 1:
        bad = 'BYTES DIFFER'
    elif acc[0] != expect:
        bad = 'EXPECTED ' + ('accept' if expect else 'refuse')
    elif acc[0] and data is not None and res[0][1] != data:
        bad = 'WRONG DATA'
    if bad:
        name = f'wbp1_{len(problems):03d}.bin'
        open(os.path.join(FIND, name), 'wb').write(rec)
        problems.append((bad, label, [r[0] for r in res], str(res[3][1])[:80] if res[3][0] else '', name, decoded, owner))
        print(f'  !! {bad}: {label}: {[r[0] for r in res]} {res[3][1] if res[3][0] else ""}', flush=True)
    return acc[0]


def skew(n, k=0.7):
    return bytes(rng.choices(range(256), [int(10000 * k ** abs(i - 120)) + (i % 7 == 0) for i in range(256)], k=n))


def cdf_flat(p, counts):
    return [64 * i for i in range(257)]


def cdf_spike0(p, counts):   # symbol 0 takes everything, every used symbol costs 14 bits unless it is 0
    return [0] + [16384 - 255 + i for i in range(256)]


def cdf_anti(p, counts):     # the most frequent symbol gets frequency 1
    top = max(range(256), key=lambda s: counts[s])
    f = [64] * 256
    f[top] = 1
    f[(top + 1) % 256] += 63
    out = [0]
    for x in f:
        out.append(out[-1] + x)
    return out


def cdf_random(p, counts):
    cuts = sorted(rng.sample(range(1, 16384), 255))
    return [0] + cuts + [16384]


# ---- 1. valid, non-canonical ----
n_valid = 0
datas = [skew(30000), bytes(5000), b'\x7f' * 70000, skew(1), skew(7), skew(9), rng.randbytes(3000), skew(33000, 0.3),
         bytes(range(256)) * 20, skew(4096), skew(1024), skew(1025), skew(32768), skew(32769)]
for data in datas:
    for planes in (1, 2, 4, 8):
        for chunk_log2 in (10, 12, 15) if len(data) < 20000 else (10, 15):
            for transform in ((0, 1) if planes > 1 and len(data) % planes == 0 else (0,)):
                for force in (None, 'rans', 'raw', 'alt'):
                    for cdf_fn in (None, cdf_flat, cdf_spike0, cdf_anti, cdf_random) if force in ('rans', 'alt') else (None,):
                        if len(data) > 20000 and cdf_fn in (cdf_spike0, cdf_anti, cdf_random) and planes > 1:
                            continue
                        f = None if force is None else {p: (force if force != 'alt' else ('rans', 'raw')[p & 1]) for p in range(planes)}
                        parts = bp_parts(data, planes, chunk_log2, 9, transform, f, cdf_fn)
                        if any(struct.unpack_from('<H', t, 2 * i)[0] == 65535 for p_, t in enumerate(parts['tables'])
                               if parts['head'][5] >> p_ & 1 for i in range(len(t) // 2)):
                            continue
                        rec = bp_join(parts)
                        four(f'valid n={len(data)} P={planes} c={chunk_log2} t={transform} force={force} cdf={getattr(cdf_fn, "__name__", None)}',
                             rec, len(data), 9, True, data)
                        n_valid += 1
stats['valid_noncanonical_records'] = n_valid
print(f'valid non-canonical WBP1 records: {n_valid}', flush=True)

# ---- 2. hostile ----
base_data = skew(9000)
n_hostile = 0


def hostile(label, edit=None, data=base_data, planes=2, chunk_log2=10, transform=0, force='rans', owner=9, decoded=None,
            raw_edit=None, expect=False, cdf_fn=None):
    global n_hostile
    parts = bp_parts(data, planes, chunk_log2, owner, transform, None if force is None else
                     {p: force for p in range(planes)} if isinstance(force, str) else force, cdf_fn)
    extra = b''
    if edit:
        extra = edit(parts) or b''
    rec = bytearray(bp_join(parts, extra))
    if raw_edit:
        edited = raw_edit(rec)
        rec = rec if edited is None else bytearray(edited)
    four('hostile: ' + label, bytes(rec), len(data) if decoded is None else decoded, owner, expect)
    n_hostile += 1


def set_head(i, v):
    return lambda parts: parts['head'].__setitem__(i, v)


def cdf_edit(fn):
    def edit(parts):
        c = list(struct.unpack('<257H', parts['cdfs'][0]))
        fn(c)
        parts['cdfs'][0] = struct.pack('<257H', *[x & 0xffff for x in c])
    return edit


def table_edit(p, fn):
    def edit(parts):
        t = list(struct.unpack(f'<{len(parts["tables"][p]) // 2}H', parts['tables'][p]))
        fn(t)
        parts['tables'][p] = struct.pack(f'<{len(t)}H', *[x & 0xffff for x in t])
    return edit


hostile('control (valid)', expect=True)
hostile('magic', set_head(0, 0x32504257))
hostile('owner in the record differs', set_head(1, 8))
hostile('decoded + 1 in the record', set_head(2, 9001))
hostile('decoded - 1 (index and record agree, data longer)', decoded=8999, edit=set_head(2, 8999))
hostile('decoded + 2 (index and record agree)', decoded=9002, edit=set_head(2, 9002))
hostile('decoded 0', decoded=0, edit=set_head(2, 0))
for v in (0, 3, 5, 16, 255):
    hostile(f'planes {v}', set_head(3, v))
hostile('planes 4 instead of 2 (mask kept)', set_head(3, 4))
hostile('planes 1 instead of 2', set_head(3, 1))
for v in (0, 9, 16, 31, 32, 255):
    hostile(f'chunk_log2 {v}', set_head(4, v))
for v in (11, 12, 15):
    hostile(f'chunk_log2 {v} instead of 10', set_head(4, v))
hostile('mask bit 2 with 2 planes', set_head(5, 7))
hostile('mask 0xff with 2 planes', set_head(5, 0xff))
hostile('mask 0 (rANS planes read as raw)', set_head(5, 0))
hostile('mask 1 (plane 1 read as raw)', set_head(5, 1))
hostile('transform 2', set_head(6, 2))
hostile('transform 1 claimed, data not rotated', set_head(6, 1))
hostile('transform 1 with planes 1', set_head(6, 1), planes=1)
hostile('transform 1 with decoded % planes != 0', set_head(6, 1), data=skew(9001), planes=2)
hostile('mask bit of an empty plane (decoded 1, 2 planes, mask 3)', set_head(5, 3), data=b'x', force={0: 'rans'})
hostile('decoded 3 in 8 planes, mask 0x0f', set_head(5, 0x0f), data=b'xyz', planes=8, force={0: 'rans', 1: 'rans', 2: 'rans'})
hostile('cdf[0] = 1', cdf_edit(lambda c: c.__setitem__(0, 1)))
hostile('cdf[256] = 16383', cdf_edit(lambda c: c.__setitem__(256, 16383)))
hostile('cdf[256] = 0', cdf_edit(lambda c: c.__setitem__(256, 0)))
hostile('cdf[256] = 65535', cdf_edit(lambda c: c.__setitem__(256, 65535)))
hostile('cdf not increasing (two equal)', cdf_edit(lambda c: c.__setitem__(100, c[99])))
hostile('cdf decreasing pair', cdf_edit(lambda c: (c.__setitem__(100, c[101]), c.__setitem__(101, c[99] + 0))))
hostile('cdf entry 0xffff in the middle', cdf_edit(lambda c: c.__setitem__(128, 0xffff)))
hostile('cdf all zero', cdf_edit(lambda c: [c.__setitem__(i, 0) for i in range(257)]))
hostile('cdf valid but different (symbols change, hash must catch)', cdf_edit(lambda c: (c.__setitem__(120, c[120] + 1))),
        cdf_fn=cdf_flat)
hostile('stream length 3', table_edit(0, lambda t: t.__setitem__(0, 3)))
hostile('stream length 0', table_edit(0, lambda t: t.__setitem__(1, 0)))
hostile('stream length 65535', table_edit(0, lambda t: t.__setitem__(0, 65535)))
hostile('lengths: first -1, second +1 (sum kept)', table_edit(0, lambda t: (t.__setitem__(0, t[0] - 1), t.__setitem__(1, t[1] + 1))))
hostile('lengths: first +1, second -1 (sum kept)', table_edit(0, lambda t: (t.__setitem__(0, t[0] + 1), t.__setitem__(1, t[1] - 1))))
hostile('lengths: last +1 (eats a byte of the next plane)', table_edit(0, lambda t: t.__setitem__(-1, t[-1] + 1)))
hostile('lengths: last plane last +1 (eats padding)', table_edit(1, lambda t: t.__setitem__(-1, t[-1] + 1)))
hostile('lengths: all 4', table_edit(0, lambda t: [t.__setitem__(i, 4) for i in range(len(t))]))
hostile('one stream table entry missing', lambda parts: parts['tables'].__setitem__(0, parts['tables'][0][:-2]))
hostile('one extra stream table entry', lambda parts: parts['tables'].__setitem__(0, parts['tables'][0] + b'\x04\x00'))
hostile('payload byte appended to a stream (length + 1)', lambda parts: (
    parts['payloads'].__setitem__(1, parts['payloads'][1] + b'\x00'),
    parts['tables'].__setitem__(1, parts['tables'][1][:-2] + struct.pack('<H', struct.unpack('<H', parts['tables'][1][-2:])[0] + 1))) and None)
hostile('initial state below L', lambda parts: parts['payloads'].__setitem__(0, b'\x00\x7f\xff\xff' + parts['payloads'][0][4:]))
hostile('initial state 0xffffffff', lambda parts: parts['payloads'].__setitem__(0, b'\xff\xff\xff\xff' + parts['payloads'][0][4:]))
hostile('initial state exactly L (wrong hash)', lambda parts: parts['payloads'].__setitem__(0, b'\x00\x80\x00\x00' + parts['payloads'][0][4:]))
hostile('raw chunk hash wrong', lambda parts: parts['tables'].__setitem__(0, b'\x00' + parts['tables'][0][1:]), force='raw')
hostile('raw chunk byte flipped', lambda parts: parts['payloads'].__setitem__(1, bytes([parts['payloads'][1][0] ^ 1]) + parts['payloads'][1][1:]), force='raw')
hostile('raw chunks swapped', lambda parts: parts['payloads'].__setitem__(0, parts['payloads'][0][1024:2048] + parts['payloads'][0][:1024] + parts['payloads'][0][2048:]), force='raw')
hostile('rANS streams of two planes swapped', lambda parts: (parts['payloads'].reverse(), parts['tables'].reverse(), parts['cdfs'].reverse()) and None,
        data=bytes(a for b in zip(skew(4000), skew(4000)) for a in b))
hostile('padding byte not zero', raw_edit=lambda r: r.__setitem__(len(r) - 1, 1))
hostile('first padding byte not zero', lambda parts: b'\x01')
hostile('extra zero page', raw_edit=lambda r: r + bytes(4096))
hostile('record cut by a page', raw_edit=lambda r: r[:-4096], data=skew(30000))
hostile('record of 0 bytes', raw_edit=lambda r: bytearray())
hostile('record of 16 bytes', raw_edit=lambda r: r[:16])
hostile('record of 4095 bytes', raw_edit=lambda r: r[:4095])
hostile('record of 4097 bytes', raw_edit=lambda r: r[:4096] + b'\0', data=skew(3000))
hostile('header only, decoded 2^30 (bomb, direct call)', raw_edit=lambda r: r[:16] + bytes(4080), decoded=1 << 30,
        edit=set_head(2, 1 << 30))
hostile('all-rANS mask, decoded 2^30, 4096 bytes', raw_edit=lambda r: r[:4096], decoded=1 << 30, edit=set_head(2, 1 << 30), data=skew(3000))
hostile('raw planes, decoded 2^32-1', decoded=0xffffffff, edit=set_head(2, 0xffffffff), force='raw', data=skew(3000))
hostile('P=8 chunk 2^15, decoded 2^32-1, mask 0xff', decoded=0xffffffff, edit=set_head(2, 0xffffffff), planes=8, chunk_log2=15,
        data=skew(4000))
stats['hostile_records'] = n_hostile
print(f'hostile WBP1 records: {n_hostile}', flush=True)

# ---- 3. exact fit: the last stream ends at the last byte of the record ----
log = Log('a3')
exact = None
src = skew(20000, 0.5)
for n in range(9000, 20000):
    rec = C.bp_encode(src[:n], 1, 10, 0)
    if rec[-1] != 0 and len(rec) < al(n):
        exact = (n, rec)
        break
stats['exact_fit'] = None
if exact:
    n, rec = exact
    blob = bp_file(rec, n)
    st = log.tri(f'exact-fit WBP1 record ({n} -> {len(rec)} B, last byte {rec[-1]:#x}) at the end of the file', blob, expect=True,
                 pyclib=None)
    stats['exact_fit'] = dict(decoded=n, stored=len(rec), verdicts=st)
    four('exact-fit direct', rec, n, 0, True, src[:n])
    for i in range(400):   # mutate the tail: lengths of the last streams and the last payload bytes
        b = bytearray(blob)
        at = len(b) - 1 - rng.randrange(64) if i % 2 else len(b) - len(rec) + 16 + 514 + rng.randrange(2 * ((n + 1023) // 1024))
        b[at] = (b[at] + rng.choice([1, 255, 0x80])) & 255
        log.tri(f'exact-fit tail mutation @{at}', bytes(b), flags=2, expect=False, pyclib=None)

# ---- 4. byte mutations of WBP1 records in a container, SHA-256 off ----
bases = []
d1 = skew(6000)
bases.append(('P1 c10', bp_file(wex4.bp_encode(d1, 1, 10, 1), len(d1), owner=1), len(wex4.bp_encode(d1, 1, 10, 1))))
d2 = bytes(a for pair in zip(skew(5000, 0.5), rng.randbytes(5000)) for a in pair)   # plane 0 rANS, plane 1 raw
r2 = wex4.bp_encode(d2, 2, 10, 0, 1)
bases.append(('P2 c10 transform, mixed mask', bp_file(r2, len(d2)), len(r2)))
d4 = struct.pack('<2500f', *[round(rng.gauss(0, 1) * 16) / 16 for _ in range(2500)])
r4 = wex4.bp_encode(d4, 4, 11, 0, 1)
bases.append(('P4 c11 transform f32', bp_file(r4, len(d4)), len(r4)))
for label, base, stored in bases:
    st = log.tri(f'mutation base {label}', base, expect=True, pyclib=None)
    base_digest = cv(C, base, 2)[1]
    rec_at = len(base) - stored
    body = len(base[rec_at:].rstrip(b'\0'))
    for i in range(MUT):
        b = bytearray(base)
        for _ in range(rng.randint(1, 3)):
            zone = rng.randrange(4)
            at = rec_at + (rng.randrange(16) if zone == 0 else rng.randrange(16, min(body, 16 + 1028 + 64)) if zone == 1
                           else rng.randrange(min(body + 8, stored)))
            k = rng.randrange(4)
            b[at] = b[at] ^ (1 << rng.randrange(8)) if k == 0 else rng.randrange(256) if k == 1 else (b[at] + rng.choice([1, 255])) & 255
        if bytes(b) == base:
            continue
        a_, r_, p_ = log.tri(f'{label} mutation #{i}', bytes(b), flags=2, pyclib=None)
        if a_ == 0:
            same = cv(C, bytes(b), 2)[1] == base_digest
            stats.setdefault('mutations_accepted_same_content' if same else 'mutations_accepted_OTHER_content', []).append(f'{label} #{i}')
            if not same:
                open(os.path.join(FIND, f'wbp1_silent_{i}.wex4'), 'wb').write(bytes(b))
res = log.done()

# ---- 5a. wex4_find against the Python table ----
blob = open(os.path.join(NOUT, 'tiny_nr3_cz.wex4'), 'rb').read()
rd = wex4.Reader(io.BytesIO(blob), C)
find_bad = 0
probes = [t[0] for t in rd.tensors] + [t[0][:-1] for t in rd.tensors[:200]] + [t[0] + b'\0' for t in rd.tensors[:200]] + \
    [b'', b'\xff' * 70000, b'\x00', b'zzzzzzzz'] + [rng.randbytes(rng.randrange(1, 40)) for _ in range(500)]
for name in probes:
    got = H.rv_find(rd.page, rd.file_bytes, rd.index, len(rd.index), name, len(name))
    want = rd.by_name.get(name, -1)
    find_bad += got != want
stats['wex4_find'] = dict(probes=len(probes), mismatches=find_bad)
empty = build()
stats['wex4_find_empty_container'] = H.rv_find(empty[:4096], len(empty), None, 0, b'x', 1)

# ---- 5b. WNR3: the chunk-size fields are not covered by the stream hashes ----
r = next(i for i, rec in enumerate(rd.records) if rec[3] == 5)
off = rd.records[r][0]
old = struct.unpack_from('<II', blob, off + 16)
mall = {}
for new_scale_chunk in (256, 1024, 16384):
    b = bytearray(blob)
    struct.pack_into('<I', b, off + 20, new_scale_chunk)
    b = bytes(b)
    mall[new_scale_chunk] = dict(sha_off=[cv(C, b, 2)[0], cv(CREF, b, 2)[0]], same_digest=cv(C, b, 2)[1] == cv(C, blob, 2)[1],
                                 sha_on=[cv(C, b, 3)[0], cv(CREF, b, 3)[0]])
    if new_scale_chunk == 16384:
        open(os.path.join(FIND, 'wnr3_scale_chunk_16384_accepted_without_sha.wex4'), 'wb').write(b)
stats['wnr3_chunk_field_malleability'] = dict(record=r, record_offset=off, weight_chunk_scale_chunk=old, edits=mall)

# ---- 6. wex4_bp_parse with decoded above 2^32 - 8 (direct callers only: wex4_open caps decoded at 2^30) ----
H.rv_bp_parse.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
dec = 0xffffffff
streams7 = ((dec - 7 + 7) // 8 + 32767) // 32768
rec = pad(struct.pack('<IIIBBBB', wex4.BP_MAGIC, 0, dec, 8, 15, 0x80, 0) + struct.pack('<257H', *[64 * i for i in range(257)]) +
          struct.pack('<H', 4) * streams7 + bytes(4 * streams7))
n = (ctypes.c_uint32 * 8)()
st = H.rv_bp_parse(rec, len(rec), dec, 0, n)
open(os.path.join(FIND, 'wbp1_decoded_ffffffff_parse_accepts.bin'), 'wb').write(rec)
stats['bp_parse_decoded_0xffffffff'] = dict(record_bytes=len(rec), c_parse_status=st, c_plane_bytes=list(n), c_plane_sum=sum(n),
                                            declared=dec, true_plane_bytes=[wex4.plane_bytes(dec, 8, p) for p in range(8)],
                                            python=py_bp(rec, dec, 0)[1])

res.update(stats=stats, wbp1_problems=problems)
json.dump(res, open(os.path.join(OUT, 'a3_wbp1.json'), 'w'), indent=1, default=str)
print(json.dumps(stats, indent=1, default=str))
print(f'WBP1 direct problems: {len(problems)}')
