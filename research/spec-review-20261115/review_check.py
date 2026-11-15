# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""The checks behind the findings of the spec review of 2026-11-15 (README.md beside this file).

    bash wex4/build_and_test.sh                                   # builds wex4/out/ (DLLs, models, test containers)
    python -I research/spec-review-20261115/review_check.py       # from the repository root

Standard library only; the G lines also need a C++ compiler (CXX, default g++) and are skipped without one.
It writes only under wex4/out/spec-review-20261115/ (git-ignored). What it checks:
  U  WEX4-SPEC 13.1: an upgrade written from the text against the converter and the vectors; the index sizes
     of the two production grids computed by the converter itself; legacy headers the converter accepts;
     the other refusals of step 1.
  P  WEX4-SPEC 14: a pack written from the text (metadata included, without the json module) against
     `wex4_convert.py pack`; unpack's file-name rule and
     what it does with a value-exact container.
  M  WNR3-SPEC 5: the frequency of nibble 8 in a canonical-zero model and in a plain one.
  G  The pure helper functions of c/wex_cuda.cuh, compiled for the CPU with the CUDA intrinsics they use
     emulated from their documented meaning, against the reference decoder's arithmetic.
  N  Numbers of the informative sections that can be re-derived from sizes, counts and other numbers."""
import hashlib
import io
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'wex4'))
import wex4  # noqa: E402
import wex4_convert as conv  # noqa: E402

OUT = os.path.join(REPO, 'wex4', 'out')
TMP = os.path.join(OUT, 'spec-review-20261115')
VEC = os.path.join(REPO, 'wex4', 'vectors')
PAGE = 4096
os.makedirs(TMP, exist_ok=True)


def rel(path):
    return os.path.relpath(path, REPO).replace(os.sep, '/')


def align(n):
    return (n + PAGE - 1) // PAGE * PAGE


def sha(data):
    return hashlib.sha256(data).digest()


# ---------- shared: a WEX4 index and header written from WEX4-SPEC sections 4 and 5 ----------
def index_and_page(blobs, records, tensors, names, meta, required, optional, layers, experts, file_bytes,
                   index_offset):
    """blobs: (kind, id, offset, bytes, sha); records: 11-tuples of 5.2; tensors: 5.3 rows without the name;
    returns (index, header page)."""
    rows = b''.join(struct.pack('<IIQQ32sQ', *b, 0) for b in blobs)
    rows += b''.join(struct.pack('<QIIHHIIII32sI', *r, 0) for r in records)
    at = 0
    for (dtype, rank, shape, record, count, offset, nbytes), name in zip(tensors, names):
        rows += struct.pack('<IHBB8QIIQQ', at, len(name), dtype, rank, *shape, *[0] * (8 - rank),
                            record, count, offset, nbytes)
        at += len(name)
    order = sorted(range(len(names)), key=lambda i: names[i])
    index = rows + struct.pack(f'<{len(order)}I', *order) + b''.join(names) + meta
    decoded = [r[2] for r in records]
    page = struct.pack('<IHHQQQQQIIIIIIQQQ32s', 0x57455834, 4, 0, required, optional, file_bytes, index_offset,
                       len(index), len(blobs), len(records), len(tensors), layers, experts,
                       max(decoded, default=0), len(b''.join(names)), len(meta), sum(decoded), sha(index))
    page += bytes(4064 - len(page))
    return index, page + sha(page)


# ---------- U1: the in-place upgrade written from WEX4-SPEC 13.1 (not from wex4_convert.py) ----------
def text_upgrade(old):
    """WEX4-SPEC 13.1 with the sentence of 2026-11-15: the area in front of the records becomes blobs in offset
    order (model kind 1 with its WNMD id, entry table kind 17, any other range kind 18, ids 0), the saved header
    page (kind 16) comes after the last record and the index after it; no metadata, no optional bit, required
    bit 0 iff the old header has flag 64; records in table order, six tensors each in slab order."""
    page = old[:PAGE]
    magic, version, layers, experts, _, table_offset, hflags, _ = struct.unpack_from('<IIIIQQII', page)
    n = layers * experts
    entries = [struct.unpack_from('<QIII6H6H3I3II', old, table_offset + 72 * i) for i in range(n)]
    present = [(i, e) for i, e in enumerate(entries) if e[1]]
    first_record = min(e[0] for _, e in present)
    front = [(table_offset, 17, 0, 72 * n)]
    if hflags & 32:
        _, model_bytes, model_offset, model_id, _ = struct.unpack_from('<IIQII', page, 64)
        front.append((model_offset, 1, model_id, model_bytes))
    blobs, at = [], PAGE
    for offset, kind, ident, size in sorted(front) + [(first_record, None, 0, 0)]:
        if offset > at:
            blobs.append((18, 0, at, offset - at, sha(old[at:offset])))
        if kind is not None:
            blobs.append((kind, ident, offset, size, sha(old[offset:offset + size])))
            at = offset + align(size)
    blobs.append((16, 0, len(old), PAGE, sha(page)))
    model = 1 + [b[0] for b in blobs].index(1) if hflags & 32 else 0
    cz = bool(hflags & 64)
    records, tensors, names = [], [], []
    for r, (i, e) in enumerate(present):
        layer, expert = divmod(i, experts)
        codec = e[22]
        records.append((e[0], e[1], e[2] + e[3], codec, 2 | (1 if cz and codec == 5 else 0),
                        model if codec == 5 else 0, layer << 16 | expert, 6 * r, 6, sha(old[e[0]:e[0] + e[1]])))
        shapes, offsets = e[4:16], e[16:22]
        for m in range(6):
            rows, cols = shapes[2 * m:2 * m + 2]
            names.append(f'layers.{layer}.ffn.experts.{expert}.w{m % 3 + 1}.{"weight" if m >= 3 else "scale"}'.encode())
            tensors.append((3 if m >= 3 else 6, 2, (rows, cols), r, 1, offsets[m], rows * cols))
    index_offset = len(old) + PAGE
    file_bytes = index_offset + align(len(index_and_page(blobs, records, tensors, names, b'', 0, 0, 0, 0, 0, 0)[0]))
    index, new_page = index_and_page(blobs, records, tensors, names, b'', 1 if cz else 0, 0, layers, experts,
                                     file_bytes, index_offset)
    return new_page + old[PAGE:] + page + index + bytes(align(len(index)) - len(index))


def u1():
    cases = ((os.path.join(VEC, 'legacy', 'tiny2-cz.wex3'), os.path.join(VEC, 'valid', 'tiny2-cz-upgraded.wex4')),
             (os.path.join(OUT, 'tiny.wex'), os.path.join(VEC, 'valid', 'tiny-raw-upgraded.wex4')))
    for legacy, expected in cases:
        old = open(legacy, 'rb').read()
        mine = text_upgrade(old)
        work = os.path.join(TMP, 'u1.wex')
        shutil.copyfile(legacy, work)
        conv.upgrade(work)
        theirs = open(work, 'rb').read()
        print(f'U1 {rel(legacy)} ({len(old)} B): upgrade written from 13.1 gives {len(mine)} B, '
              f'equal to wex4_convert.py upgrade: {mine == theirs}, equal to {rel(expected)}: '
              f'{mine == open(expected, "rb").read()}')


# ---------- U2: index sizes of the two production grids, computed by wex4_convert.upgrade itself ----------
class SparseFile:
    """A seekable file of `size` bytes that holds `parts` and zeros elsewhere (nothing is stored for the zeros)."""

    def __init__(self, size, parts):
        self.size, self.parts, self.at = size, dict(parts), 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def seek(self, offset, whence=0):
        self.at = self.size + offset if whence == 2 else offset
        return self.at

    def tell(self):
        return self.at

    def read(self, n):
        n = max(0, min(n, self.size - self.at))
        out = bytearray(n)
        for offset, data in self.parts.items():
            lo, hi = max(offset, self.at), min(offset + len(data), self.at + n)
            if lo < hi:
                out[lo - self.at:hi - self.at] = data[lo - offset:hi - offset]
        self.at += n
        return bytes(out)

    def write(self, data):
        self.parts[self.at] = bytes(data)
        self.size = max(self.size, self.at + len(data))
        self.at += len(data)

    def flush(self):
        pass

    def fileno(self):
        raise io.UnsupportedOperation


def legacy_file(magic, version, layers, experts, flags, scale, weight, pages, table_offset, model=None):
    """A WEX1/WEX3 stub: header page, optional model at 4096, entry table, records of the given page counts
    (stored as zeros: upgrade with hash_records=False does not read them)."""
    n = layers * experts
    raw = 3 * (scale[0] * scale[1] + weight[0] * weight[1])
    page = bytearray(PAGE)
    struct.pack_into('<IIIIQQII', page, 0, magic, version, layers, experts, align(raw), table_offset, flags, 0)
    parts = {}
    if model:
        struct.pack_into('<IIQII', page, 64, 0x444d4e57, len(model), PAGE, wex4.wnm2_id(model), 0)
        parts[PAGE] = model
    at = table_offset + align(72 * n)
    table = bytearray()
    s1 = scale[0] * scale[1]
    w1 = weight[0] * weight[1]
    for i in range(n):
        table += struct.pack('<QIII6H6H3I3II', at, pages[i] * PAGE, 3 * s1, 3 * w1, *scale * 3, *weight * 3,
                             0, s1, 2 * s1, 3 * s1, 3 * s1 + w1, 3 * s1 + 2 * w1, 5 if flags & 32 else 0)
        at += pages[i] * PAGE
    parts[0] = bytes(page)
    parts[table_offset] = bytes(table)
    return SparseFile(at, parts)


def u2():
    model = open(os.path.join(OUT, 'tiny.wnm2'), 'rb').read()
    pages = [3920] * 5980 + [3919] * (7680 - 5980)       # 30,103,900 pages of records, as the production file
    f = legacy_file(0x57455833, 3, 40, 192, 1 | 32 | 64, (2048, 180), (2048, 2880), pages, 65536, model)
    old = f.size
    got = conv.upgrade(f, hash_records=False)
    print(f'U2 V4.1 grid 40 x 192, WNR3-CZ, model at 4096, table at 65536, file {old:,} B: index {got["index_bytes"]:,} B, '
          f'added {got["added"]:,} B ({100 * got["added"] / old:.4f} % of the file); WEX4-SPEC 11.2, 13.1, 15.3 say '
          f'6,712,992 and 6,717,440 B')
    f = legacy_file(0x57455831, 1, 48, 512, 0, (800, 64), (800, 1024), [638] * 24576, PAGE)
    old = f.size
    got = conv.upgrade(f, hash_records=False, family='qwen38')
    print(f'U2 Qwen3.8 grid 48 x 512, WEX1 RAW records of 638 pages, table at 4096, file {old:,} B, --names qwen38: '
          f'index {got["index_bytes"]:,} B, added {got["added"]:,} B, file after {got["file_bytes"]:,} B; '
          f'research/qwen38-speed-20261111 says 26,086,592, +26,091,520 and 64,251,047,936 B')


# ---------- U3: a WEX1 header that the legacy validator refuses and the converter upgrades ----------
def u3():
    work = os.path.join(TMP, 'u3.wex')
    shutil.copyfile(os.path.join(OUT, 'tiny.wex'), work)
    with open(work, 'r+b') as f:
        f.seek(32)
        f.write(struct.pack('<I', 2))           # WEX1 header flags 2 (PALETTE3); wex_validate_header wants 0
    before = open(work, 'rb').read()
    try:
        conv.upgrade(work)
        with open(work, 'rb') as f:
            rd = wex4.Reader(f)
            verdict = f'upgraded, the WEX4 file opens ({len(rd.records)} records)'
        conv.downgrade(work)
        verdict += f'; downgrade gives the file back byte for byte: {open(work, "rb").read() == before}'
    except wex4.FormatError as e:
        verdict = f'refused: {e}'
    print(f'U3 {rel(os.path.join(OUT, "tiny.wex"))} with WEX1 header flags 2: {verdict}')
    shutil.copyfile(os.path.join(OUT, 'tiny.wex'), work)
    with open(work, 'r+b') as f:
        f.seek(32)
        f.write(struct.pack('<I', 64))          # WEX1 header flags 64 (canonical zero); wex_validate_header wants 0
    try:
        conv.upgrade(work)
        with open(work, 'rb') as f:
            rd = wex4.Reader(f)
            exact = sum(1 for r in rd.records if r[4] & wex4.REC_VALUE_EXACT)
            verdict = f'upgraded, required = {rd.required}, VALUE_EXACT records = {exact} of {len(rd.records)}'
    except wex4.FormatError as e:
        verdict = f'refused: {e}'
    print(f'U3 {rel(os.path.join(OUT, "tiny.wex"))} with WEX1 header flags 64: {verdict}')


# ---------- U4: the other refusals of step 1 of 13.1 (sentence of 2026-11-15) ----------
def u4():
    raw1 = open(os.path.join(OUT, 'tiny.wex'), 'rb').read()
    raw3 = open(os.path.join(VEC, 'legacy', 'tiny2-cz.wex3'), 'rb').read()

    def entries(old):
        _, _, layers, experts, _, table_offset, _, _ = struct.unpack_from('<IIIIQQII', old)
        return table_offset, [list(conv.ENTRY.unpack_from(old, table_offset + i * conv.ENTRY.size))
                              for i in range(layers * experts)]

    def with_entries(old, change):
        table_offset, rows = entries(old)
        change(rows)
        new = bytearray(old)
        for i, row in enumerate(rows):
            conv.ENTRY.pack_into(new, table_offset + i * conv.ENTRY.size, *row)
        return bytes(new)

    def at(old, offset, fmt, value):
        new = bytearray(old)
        struct.pack_into(fmt, new, offset, value)
        return bytes(new)

    def no_records(rows):
        for row in rows:
            row[:] = [0] * len(row)

    def shift(rows):
        for row in rows:
            if row[1]:
                row[0] += 8

    def codec(value):
        def change(rows):
            next(row for row in rows if row[1])[22] = value
        return change

    table3 = struct.unpack_from('<Q', raw3, 24)[0]
    model3 = struct.unpack_from('<Q', raw3, 72)[0]
    id3 = struct.unpack_from('<I', raw3, 80)[0]
    cases = [
        ('WEX1 without records', with_entries(raw1, no_records)),
        ('WEX1 grid of 65,537 layers', at(raw1, 8, '<I', 65537)),
        ('WEX1 first record 8 bytes past a 4096-byte boundary', with_entries(raw1, shift)),
        ('WEX1 entry with codec 5', with_entries(raw1, codec(5))),
        ('WNR3 entry with codec 3', with_entries(raw3, codec(3))),
        (f'WNR3 model at {model3 + 64} (not on a boundary)', at(raw3, 72, '<Q', model3 + 64)),
        ('WNR3 model at 0 (over the header page)', at(raw3, 72, '<Q', 0)),
        (f'WNR3 model at {table3} (over the entry table)', at(raw3, 72, '<Q', table3)),
        ('WNR3 WNMD magic changed', at(raw3, 64, '<I', 0x444d4e58)),
        ('WNR3 WNMD size 59,409', at(raw3, 68, '<I', 59409)),
        ('WNR3 WNMD id not that of the model', at(raw3, 80, '<I', id3 ^ 1)),
    ]
    work = os.path.join(TMP, 'u4.wex')
    results = []
    for what, data in cases:
        with open(work, 'wb') as f:
            f.write(data)
        try:
            conv.upgrade(work)
            results.append(f'{what}: UPGRADED')
        except wex4.FormatError as e:
            results.append(f'{what}: refused ({e})')
    print('U4 converter upgrade of', '; '.join(results))


# ---------- P1: pack written from WEX4-SPEC 14 (with the sentences of 2026-11-15) ----------
FLOATS = ('F16', 'BF16', 'F32', 'F64')


def st_write(path, tensors, meta=None, indent=None):
    header, at = {}, 0
    for name, dtype, shape, data in tensors:
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [at, at + len(data)]}
        at += len(data)
    if meta:
        header['__metadata__'] = meta
    raw = json.dumps(header, ensure_ascii=False, indent=indent).encode()
    raw += b' ' * (-len(raw) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(raw)) + raw + b''.join(t[3] for t in tensors))


def bf16_bytes(rng, n):
    out = bytearray()
    for _ in range(n):
        out += bytes([rng.randrange(256), rng.choice([0x3c, 0x3c, 0x3d, 0xbc, 0xbd, 0x3e])])
    return bytes(out)


def st_tensors(path):
    data = open(path, 'rb').read()
    n = struct.unpack_from('<Q', data)[0]
    raw = data[8:8 + n]
    doc = json.loads(raw)
    doc.pop('__metadata__', None)
    items = sorted(doc.items(), key=lambda kv: tuple(kv[1]['data_offsets']))
    return raw, [(k.encode(), v['dtype'], tuple(v['shape']), data[8 + n + v['data_offsets'][0]:8 + n + v['data_offsets'][1]])
                 for k, v in items], len(data), hashlib.sha256(data).hexdigest()


SHORT = {0x08: '\\b', 0x09: '\\t', 0x0a: '\\n', 0x0c: '\\f', 0x0d: '\\r'}


def text_string(s):
    """A JSON string as WEX4-SPEC 14 says the metadata writes it: `"` and `\\` as `\\"` and `\\\\`; below U+0020
    the five short forms, the others as \\u00xx with lower-case hex; everything else as it is (UTF-8)."""
    out = []
    for ch in s:
        c = ord(ch)
        if ch in '"\\':
            out.append('\\' + ch)
        elif c < 0x20:
            out.append(SHORT.get(c, f'\\u{c:04x}'))
        else:
            out.append(ch)
    return '"' + ''.join(out) + '"'


def text_meta(sources):
    """The metadata of WEX4-SPEC 14, written from the text (no json module)."""
    rows = [f'{{"name":{text_string(s["name"])},"bytes":{s["bytes"]},"sha256":{text_string(s["sha256"])},'
            f'"header":{text_string(s["header"])}}}' for s in sources]
    return ('{"wex4_meta":1,"sources":[' + ','.join(rows) + ']}').encode('utf-8')


def text_pack(sources, part):
    """WEX4-SPEC 14, no expert records: per source file in the given order, its tensors in offset order; a tensor of
    64 KiB or more is its own record (parts of `part` bytes, the last shorter), WBP1 with P = its element size and
    transform 1 for F16/BF16/F32/F64; a smaller one joins the open group of its element size (sub-byte and opaque
    dtypes count as 1), which becomes a record (transform 0) before it would pass 4 MiB, or at the end of the file,
    groups in the order they were opened; zero-size tensors last. WBP1 at chunk_log2 12, owner = record index, RAW
    when not smaller. Metadata: compact JSON, keys in this order, non-ASCII as UTF-8."""
    plan, empty, meta_sources = [], [], []
    for path in sources:
        raw, tensors, size, digest = st_tensors(path)
        meta_sources.append({'name': os.path.basename(path), 'bytes': size, 'sha256': digest, 'header': raw.decode()})
        groups = {}
        for name, dtype, shape, data in tensors:
            code = wex4.DTYPE_CODE.get(dtype, 0)
            spec = (name, code, shape, len(data))
            if not data:
                empty.append(spec)
                continue
            width = max(1, wex4.BITS[code] // 8) if code else 1
            if len(data) < 65536:
                if width in groups and len(groups[width][1]) + len(data) > 4 << 20:
                    specs, blob = groups.pop(width)
                    plan.append((specs, blob, width, 0, 1))
                specs, blob = groups.get(width, ([], b''))
                groups[width] = (specs + [spec], blob + data)   # an open group keeps its place in the order
                continue
            k = -(-len(data) // part)
            for j in range(k):
                plan.append(([spec] if not j else [], data[j * part:(j + 1) * part], width, int(dtype in FLOATS), k))
        for width, (specs, blob) in groups.items():
            plan.append((specs, blob, width, 0, 1))
    stored_list, records, tensors, names, split_first, t = [], [], [], [], None, 0
    for r, (specs, data, width, transform, parts) in enumerate(plan):
        raw_rec = data + bytes(align(len(data)) - len(data))
        coded = wex4.bp_encode(data, width, 12, r, transform)
        codec, stored = (6, coded) if len(coded) < len(raw_rec) else (0, raw_rec)
        if specs and parts > 1:
            split_first = t
        cursor = 0
        for name, code, shape, nbytes in specs:
            tensors.append((code, len(shape), shape, r, parts, 0 if parts > 1 else cursor, nbytes))
            names.append(name)
            cursor += nbytes
        records.append([0, len(stored), len(data), codec, 0, 0, r, t if specs else split_first, len(specs), sha(stored)])
        stored_list.append(stored)
        t += len(specs)
    for name, code, shape, nbytes in empty:
        tensors.append((code, len(shape), shape, 0, 0, 0, 0))
        names.append(name)
    meta = text_meta(meta_sources)
    index_len = len(index_and_page([], [tuple(r) for r in records], tensors, names, meta, 0, 1, 0, 0, 0, 0)[0])
    at = PAGE + align(index_len)
    for rec, stored in zip(records, stored_list):
        rec[0] = at
        at += len(stored)
    index, page = index_and_page([], [tuple(r) for r in records], tensors, names, meta, 0, 1, 0, 0, at, PAGE)
    return page + index + bytes(align(len(index)) - len(index)) + b''.join(stored_list), plan


def p1():
    rng = random.Random(15)
    a, b = os.path.join(TMP, 'pa.safetensors'), os.path.join(TMP, 'pb.safetensors')
    st_write(a, [('a.small1', 'BF16', (4, 2500), bf16_bytes(rng, 10000)),
                 ('a.codes', 'I8', (30000,), bytes(rng.choice([0, 1, 1, 2, 255]) for _ in range(30000))),
                 ('a.big', 'BF16', (256, 256), bf16_bytes(rng, 65536)),
                 ('a.small2', 'BF16', (12000,), bf16_bytes(rng, 12000)),
                 ('a.split', 'BF16', (600, 256), bf16_bytes(rng, 153600)),
                 ('a.empty', 'F32', (0, 4), b'')],
             {'note': 'café "quoted"'})
    st_write(b, [('b.small', 'BF16', (9000,), bf16_bytes(rng, 9000)),
                 ('b.codes', 'U8', (25000,), bytes(rng.choice([7, 7, 8, 9]) for _ in range(25000)))],
             {'ctl': 'x\x01y\x7f\\z'}, indent='\t')       # header text with tabs, newlines, \u0001, a raw DEL
    work = os.path.join(TMP, 'p1.wex4')
    conv.pack(work, [a, b], part_bytes=128 << 10)
    theirs = open(work, 'rb').read()
    mine, plan = text_pack([a, b], 128 << 10)
    with open(work, 'rb') as f:
        rd = wex4.Reader(f)
        rows = []
        for r, rec in enumerate(rd.records):
            names = [t[0].decode() for t in rd.tensors if t[4] == r and t[5]]
            desc = f'r{r} codec {rec[3]} decoded {rec[2]}'
            if rec[3] == 6:
                planes, log2, mask, transform = rd.stored(r)[12:16]
                desc += f' P {planes} chunk_log2 {log2} transform {transform}'
            rows.append(desc + f' [{",".join(names) or "continuation"}]')
        meta = rd.meta
    compact = meta.startswith(b'{"wex4_meta":1,"sources":[{"name":"pa.safetensors","bytes":')
    utf8 = 'café'.encode() in meta
    escaped = b'\\"note\\"' in meta
    controls = b'\\n\\t' in meta and b'\\\\u0001' in meta and b'\x7f' in meta
    chars = [chr(c) for c in range(0x100)] + ['€', '\U0001f600']
    same = sum(text_string(ch) == json.dumps(ch, ensure_ascii=False) for ch in chars)
    print('P1 wex4_convert.py pack of two synthetic files (part size 128 KiB):', '; '.join(rows))
    print(f'P1 metadata: compact with keys in the stated order: {compact}; non-ASCII kept as UTF-8: {utf8}; '
          f'quotes of the header text escaped: {escaped}; newline and tab of an indented header, a \\u0001 '
          f'and a raw U+007F in it present as the text says: {controls}')
    print(f'P1 escaping written from section 14 against json.dumps(ensure_ascii=False), one character at a time: '
          f'{same} of {len(chars)} (U+0000 to U+00FF, U+20AC, U+1F600) equal')
    print(f'P1 pack written from section 14 gives {len(mine)} B, byte-identical to wex4_convert.py pack: {mine == theirs}')


# ---------- P2, P3: unpack ----------
def p2():
    work = os.path.join(TMP, 'p2.wex4')
    outdir = os.path.join(TMP, 'p2out')
    results = []
    for bad in ('../evil.safetensors', 'sub/x.safetensors', '..', 'p2.wex4', 'twice'):
        sources = [{'name': bad, 'bytes': 8, 'sha256': '0' * 64, 'header': '{}'}]
        if bad == 'twice':
            sources = [dict(sources[0], name='x.safetensors'), dict(sources[0], name='x.safetensors')]
        meta = json.dumps({'wex4_meta': 1, 'sources': sources}, separators=(',', ':')).encode()
        records = [wex4.Record([wex4.Tensor('t', 'U8', (5,), 5)], 5, lambda: b'hello')]
        wex4.write(work, records, meta=meta, optional=wex4.OPT_ST_SOURCES)
        try:
            conv.unpack(work, os.path.dirname(work) if bad == 'p2.wex4' else outdir)
            results.append(f'{bad!r} accepted')
        except wex4.FormatError as e:
            results.append(f'{bad!r} refused ({e})')
    print('P2 unpack with a source name of', '; '.join(results))


def p3():
    clib = wex4.CLib(os.path.join(OUT, 'wex4_tools.dll'))
    nr3 = conv.Nr3(os.path.join(OUT, 'wex_nr3_tools.dll'), os.path.join(OUT, 'tiny.wnm2'), True)
    work = os.path.join(TMP, 'p3.wex4')
    conv.pack(work, [os.path.join(REPO, 'fixtures', 'v41-tiny', 'model.safetensors')], clib, 'nr3', nr3)
    outdir = os.path.join(TMP, 'p3out')
    shutil.rmtree(outdir, ignore_errors=True)
    results = conv.unpack(work, outdir, clib)
    kept = [name for name, ok in results if not ok and os.path.exists(os.path.join(outdir, name))]
    print(f'P3 unpack of a canonical-zero pack of fixtures/v41-tiny: {len(results)} file(s), SHA-256 differs for '
          f'{sum(not ok for _, ok in results)}, kept in place all the same: {len(kept)} (reported "DIFFERS")')


# ---------- M1: nibble 8 in the weight tables of a canonical-zero model ----------
def m1():
    reach = {(pm, mx) for mx in range(8) for pm in range(mx + 1)} | {(8, 8)}
    models = [(rel(os.path.join(OUT, 'tiny.wnm2')), open(os.path.join(OUT, 'tiny.wnm2'), 'rb').read()),
              (rel(os.path.join(OUT, 'synth.wnm2')), open(os.path.join(OUT, 'synth.wnm2'), 'rb').read())]
    with open(os.path.join(VEC, 'valid', 'tiny2-cz-upgraded.wex4'), 'rb') as f:
        models.append(('blob 0 of ' + rel(os.path.join(VEC, 'valid', 'tiny2-cz-upgraded.wex4')), wex4.Reader(f).blob(0)))
    for label, blob in models:
        uniform = one = other = unreachable_used = 0
        for m in range(3):
            for c in range(567):
                cdf = struct.unpack_from('<17H', blob, 32 + (m * 567 + c) * 34)
                freq = [cdf[s + 1] - cdf[s] for s in range(16)]
                pm, mx = c // 9 % 9, c % 9
                if freq == [1024] * 16:
                    uniform += 1
                    continue
                unreachable_used += (pm, mx) not in reach
                if freq[8] == 1:
                    one += 1
                else:
                    other += 1
        print(f'M1 {label} (id 0x{wex4.wnm2_id(blob):08x}): of 1701 weight tables {uniform} uniform (1024 each, no '
              f'counts), {one} with frequency 1 for nibble 8, {other} with more; non-uniform tables of a context that '
              f'cannot occur: {unreachable_used}')


# ---------- G1: helper functions of c/wex_cuda.cuh on the CPU ----------
GPU_PRELUDE = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "wex_neural_rans.h"
#define __device__
#define __forceinline__ inline
#define __global__
struct uint4 { uint32_t x, y, z, w; };
struct Dim3 { uint32_t x, y, z; };
static Dim3 blockIdx, blockDim, threadIdx;
static inline uint32_t max(uint32_t a, uint32_t b) { return a > b ? a : b; }
static inline uint32_t min(uint32_t a, uint32_t b) { return a < b ? a : b; }
static inline int __popc(uint32_t x) { return __builtin_popcount(x); }
static inline uint32_t __vmaxu2(uint32_t a, uint32_t b) {   /* per halfword, unsigned */
    return max(a & 0xffffu, b & 0xffffu) | max(a >> 16, b >> 16) << 16;
}
static inline uint32_t __vminu2(uint32_t a, uint32_t b) {
    return min(a & 0xffffu, b & 0xffffu) | min(a >> 16, b >> 16) << 16;
}
/* stands in for the PTX of wex_cuda_half_sign (prmt.b32 r, x, 0, 0xbb99): bytes 1, 1, 3, 3 of x, sign-replicated */
static inline uint32_t wex_cuda_half_sign(uint32_t x) {
    return (x & 0x8000u ? 0xffffu : 0u) | (x & 0x80000000u ? 0xffff0000u : 0u);
}
'''

GPU_MAIN = r'''
static void ref_find(const unsigned char *cdf, uint32_t symbols, uint32_t slot, uint32_t *s, uint32_t *start, uint32_t *f) {
    uint32_t lo = 0, hi = symbols;   /* the search of wex_nr2_decode_symbol */
    while (lo + 1u < hi) { uint32_t mid = (lo + hi) / 2u; if (wex_nr_u16(cdf + mid * 2u) <= slot) lo = mid; else hi = mid; }
    *s = lo; *start = wex_nr_u16(cdf + lo * 2u); *f = wex_nr_u16(cdf + lo * 2u + 2u) - *start;
}
int main(int argc, char **argv) {
    static unsigned char has8[65536], ok5[65536];
    static uint16_t dbl[65536];
    uint64_t w, bad = 0, n = 0;
    for (w = 0; w < 65536u; w++) {
        uint32_t k, any = 0, valid = 1, d = 0;
        for (k = 0; k < 4u; k++) {
            uint32_t q = (uint32_t)(w >> (4u * k)) & 15u, a = q & 7u;
            any |= q == 8u;
            valid &= a <= 5u;
            d |= ((q & 8u) | (a + (a < 2u ? a : 2u))) << (4u * k);
        }
        has8[w] = (unsigned char)any; ok5[w] = (unsigned char)valid; dbl[w] = (uint16_t)d;
    }
    for (w = 0; w < (UINT64_C(1) << 32); w++)
        bad += (wex_cuda_has_nibble8((uint32_t)w) != 0) != (has8[w & 0xffffu] | has8[w >> 16]);
    printf("G1 wex_cuda_has_nibble8 on all 2^32 words: %llu disagree with \"some nibble is 8\"\n", (unsigned long long)bad);
    bad = 0;
    for (w = 0; w < 65536u; w++) {
        uint32_t h;
        if (!ok5[w]) continue;
        for (h = 0; h < 65536u; h++) {
            if (!ok5[h]) continue;
            n++;
            bad += wex_cuda_ceil_dbl8((uint32_t)w | h << 16) != ((uint32_t)dbl[w] | (uint32_t)dbl[h] << 16);
        }
    }
    printf("G1 wex_cuda_ceil_dbl8 on all %llu words whose 8 magnitudes are <= 5: %llu differ from D(a) = a + min(a, 2), sign kept\n",
           (unsigned long long)n, (unsigned long long)bad);
    for (int i = 1; i < argc; i++) {
        static unsigned char blob[WEX_NR2_MODEL_BYTES];
        static uint32_t rows[3u * WEX_NR2_CONTEXTS * 8u], direct[3u * WEX_NR2_TOTAL];
        static unsigned char symbol_table[3u * WEX_NR2_TOTAL];
        WexNr2Model model;
        uint32_t idx, m, c, slot;
        uint64_t row_bad = 0, scale_bad = 0;
        FILE *f = fopen(argv[i], "rb");
        if (!f || fread(blob, 1, sizeof(blob), f) != sizeof(blob) || wex_nr2_model_open(blob, sizeof(blob), &model)) return 2;
        fclose(f);
        blockDim.x = 256u;
        for (idx = 0; idx < 3u * WEX_NR2_CONTEXTS * 8u; idx++) { blockIdx.x = idx / 256u; threadIdx.x = idx % 256u; wex_cuda_nr2_build_rows(blob, rows); }
        for (idx = 0; idx < 3u * WEX_NR2_TOTAL; idx++) { blockIdx.x = idx / 256u; threadIdx.x = idx % 256u; wex_cuda_nr2_build_scales(blob, direct, symbol_table); }
        for (m = 0; m < 3u; m++)
            for (c = 0; c < WEX_NR2_CONTEXTS; c++) {
                const uint32_t *r = rows + ((size_t)m * WEX_NR2_CONTEXTS + c) * 8u;
                uint4 a = {r[0], r[1], r[2], r[3]}, b = {r[4], r[5], r[6], r[7]};
                for (slot = 0; slot < WEX_NR2_TOTAL; slot++) {
                    uint32_t s, st, fr, s2, st2, fr2;
                    wex_cuda_nr2_row_find(a, b, slot, s, st, fr);
                    ref_find(wex_nr2_weight_cdf(&model, m, c), 16u, slot, &s2, &st2, &fr2);
                    row_bad += s != s2 || st != st2 || fr != fr2;
                }
            }
        for (m = 0; m < 3u; m++)
            for (slot = 0; slot < WEX_NR2_TOTAL; slot++) {
                uint32_t e = direct[m * WEX_NR2_TOTAL + slot], s2, st2, fr2;
                ref_find(wex_nr2_scale_cdf(&model, m), 256u, slot, &s2, &st2, &fr2);
                scale_bad += symbol_table[m * WEX_NR2_TOTAL + slot] != s2 || (e & 0x3fffu) + 1u != fr2 || (e >> 14) != slot - st2;
            }
        printf("G1 model %08x: row lookup (16-way compare) vs the reference search, 3 x 567 tables x 16384 slots: %llu differ; "
               "scale direct table, 3 x 16384 slots: %llu differ\n", model.id, (unsigned long long)row_bad, (unsigned long long)scale_bad);
    }
    return 0;
}
'''


def extract(src, name):
    m = re.search(r'static\s+(?:__global__|__device__\s+__forceinline__)\s[^;{]*?\b' + name + r'\s*\(', src)
    i, depth = src.index('{', m.end()), 0
    for j in range(i, len(src)):
        depth += {'{': 1, '}': -1}.get(src[j], 0)
        if not depth:
            return src[m.start():j + 1]


def g1():
    cxx = os.environ.get('CXX', 'g++')
    if not shutil.which(cxx):
        print(f'G1 skipped: no C++ compiler ({cxx})')
        return
    src = open(os.path.join(REPO, 'c', 'wex_cuda.cuh'), encoding='utf-8').read()
    macro = re.search(r'^#define WEX_CUDA_NR2_ROWS .*$', src, re.M).group(0)
    parts = [macro] + [extract(src, n) for n in ('wex_cuda_nr2_weight_cdf', 'wex_cuda_nr2_scale_cdf',
                                                 'wex_cuda_nr2_build_rows', 'wex_cuda_nr2_build_scales',
                                                 'wex_cuda_nr2_row_find', 'wex_cuda_ceil_dbl8', 'wex_cuda_has_nibble8')]
    cpp, exe = os.path.join(TMP, 'gpu_parts.cpp'), os.path.join(TMP, 'gpu_parts' + ('.exe' if os.name == 'nt' else ''))
    with open(cpp, 'w', encoding='utf-8') as f:
        f.write(GPU_PRELUDE + '\n\n'.join(parts) + GPU_MAIN)
    subprocess.run([cxx, '-O2', '-std=c++17', '-I', os.path.join(REPO, 'c'), '-o', exe, cpp], check=True)
    run = subprocess.run([exe, os.path.join(OUT, 'tiny.wnm2'), os.path.join(OUT, 'synth.wnm2')],
                         capture_output=True, text=True, check=True)
    print(f'G1 {len(parts) - 1} functions and the macro WEX_CUDA_NR2_ROWS taken from c/wex_cuda.cuh as they are, '
          f'built with {cxx} for the CPU')
    sys.stdout.write(run.stdout)


# ---------- N: numbers of the informative sections that can be re-derived ----------
def n_checks():
    raw = 18_800_640
    raw_total = 7680 * raw
    rows = [
        ('WEX4 8.1 / WNR3 0', '+1,352 B per record = +0.007 % of raw', f'{100 * 1352 / raw:.4f} %'),
        ('WEX4 11.2', 'about 20 ms of SHA-256 at 328 MB/s on the 6,712,992 B index', f'{6712992 / 328e6 * 1e3:.1f} ms'),
        ('WEX4 11.2', 'about 3 ms at 2.4 GB/s', f'{6712992 / 2.4e9 * 1e3:.1f} ms'),
        ('WEX4 11.3', 'about 35 MB/s for 505-590 ms per record', f'{raw / 0.590 / 1e6:.1f}-{raw / 0.505 / 1e6:.1f} MB/s'),
        ('WEX4 11.3', 'about 65 minutes for 7,680 records on one thread', f'{7680 * 0.505 / 60:.1f}-{7680 * 0.590 / 60:.1f} min (505-590 ms)'),
        ('WEX4 11.3', '1-2 minutes at 8.8-13.8 ms per record', f'{7680 * 0.0088 / 60:.1f}-{7680 * 0.0138 / 60:.1f} min'),
        ('WEX4 11.3', 'about 0.26 GB/s per thread at 72-90 ms', f'{raw / 0.090 / 1e9:.2f}-{raw / 0.072 / 1e9:.2f} GB/s'),
        ('WEX4 11.3', 'about 1,400 times slower than 0.37 ms on the GPU', f'{0.505 / 0.37e-3:.0f}-{0.590 / 0.37e-3:.0f} x'),
        ('WEX4 12.1', 'the three dtypes sum to 11,130,045,256 B', f'{7086800896 + 3874076672 + 169167688:,} B'),
        ('WEX4 12.1', 'plain byte planes 0.767 of the three, about 2.6 GB saved',
         f'{(0.8187 * 7086800896 + 0.6687 * 3874076672 + 0.8439 * 169167688) / 11130045256:.4f}, '
         f'{11130045256 * (1 - 0.767) / 1e9:.2f} GB'),
        ('WEX4 12.1', 'one BF16 shard 0.6625; one layer shard 0.817', f'{643002368 / 970533624:.4f}; {137338880 / 168090456:.4f}'),
        ('WEX4 12.1', 'F8 context 0.8144 against 0.8187, about 30 MB', f'{(0.8187 - 0.8144) * 7086800896 / 1e6:.1f} MB'),
        ('WEX4 12.2', 'mean record 16,055,413 B = 0.85398 of raw',
         f'{(123306192896 - 618496) / 7680:,.2f} B (file minus header, model and table, over 7,680), '
         f'{(123306192896 - 618496) / 7680 / raw:.5f}'),
        ('WEX4 12.2', '108,266 B = 0.576 % of raw = 5.57 B x 19,440 streams',
         f'{100 * 108266 / raw:.4f} %, {108266 / 19440:.4f} B, streams 3 x (368,640 / 512 + 32 x 368,640 / 2048) = '
         f'{3 * (368640 // 512 + 32 * 368640 // 2048):,}'),
    ]
    for label, pct, stated in ((('length table', 0.130, '188 MB'), ('3-byte state', 0.013, '19 MB'),
                                ('64 B alignment', 0.011, '15.6 MB'), ('WNR4 record', 0.154, 'about 0.22 GB'),
                                ('chunks 4096:1024', 0.288, '416 MB'), ('chunks 8192:2048', 0.432, '624 MB'),
                                ('last 5 nibbles', 0.205, '297 MB'), ('small state', 0.188, '271 MB'))):
        rows.append(('WEX4 12.2', f'{label}: {pct} % of raw = {stated} "on the 123.3 GB container"',
                     f'{pct / 100 * raw_total / 1e6:.1f} MB ({pct}% of the 7,680 raw slabs, {raw_total / 1e9:.2f} GB)'))
    rows += [
        ('WEX4 12.2', 'the three together about 0.22 GB = 0.18 % of the container', f'{0.00154 * raw_total / 123306192896 * 100:.3f} %'),
        ('WEX4 12.2', 'everything at once 0.76 GB = 0.62 % of the container', f'{0.76e9 / 123306192896 * 100:.3f} %'),
        ('WEX4 12.3', 'canonical zero -2.74 % of raw (514,848 B)', f'{100 * 514848 / raw:.3f} %'),
        ('WEX4 12.3', '7,680 records decode to 144.4 GB; the safetensors hold 151.6 GB with the 384 MTP experts',
         f'{raw_total / 1e9:.2f} GB; {(7680 + 384) * raw / 1e9:.2f} GB'),
        ('WEX4 12.4', 'weight tables 57,834 B -> 378 B', f'{3 * 567 * 17 * 2:,} B = 3 x 567 x 17 u16; 378 = 3 x 7 x 9 u16 '
         f'(7 scale classes, 8 magnitudes): {3 * 7 * 9 * 2}'),
        ('WEX4 15.3', '64 records 1.03 GB; about 2 minutes at 1,030 MB/s', f'{64 * 16055413 / 1e9:.3f} GB; {123306192896 / 1.03e9:.0f} s'),
        ('WEX4 16.3', 'about 6 minutes at 0.33 GB/s', f'{123306192896 / 0.328e9 / 60:.1f} min'),
        ('WEX4 11.1', '2.070 -> 2.077 tok/s is +0.32 %', f'{(2.0765 / 2.0705 - 1) * 100:.2f}-{(2.0775 / 2.0695 - 1) * 100:.2f} % '
         'from the rounded means'),
        ('WNR3 0', '16,578,252.8 B = 0.88179; WNR2 16,909,926.4 B = 0.89943; -1.764 %',
         f'{16578252.8 / raw:.6f}; {16909926.4 / raw:.6f}; {(16578252.8 - 16909926.4) / raw * 100:.3f} %'),
        ('WNR3 6', 'WNR2 16,914,680 B 0.8997, 129.9 GB', f'{16914680 / raw:.5f}, {7680 * 16914680 / 1e9:.2f} GB; '
         f'with header, model and table {7680 * 16914680 + 618496:,} B (CHANGELOG 2026-11-07: 129,905,360,896 B)'),
        ('WNR3 6', 'without CZ ~16,579,600 B ~0.8819 ~127.3 GB',
         f'{16578252.8 + 1352:,.1f} B, {(16578252.8 + 1352) / raw:.5f}, {7680 * (16578252.8 + 1352) / 1e9:.2f} GB'),
        ('WNR3 6', 'with CZ ~16,064,800 B ~0.8545 ~123.4 GB',
         f'{16578252.8 + 1352 - 514848:,.1f} B, {(16578252.8 + 1352 - 514848) / raw:.5f}, '
         f'{7680 * (16578252.8 + 1352 - 514848) / 1e9:.2f} GB'),
        ('WNR3 6', 'the full container 123,306,192,896 B is 0.854 of WEX1 (144,389,472,256 B)', f'{123306192896 / 144389472256:.5f}'),
        ('WNR3 5', 'symbol 8 at frequency 1/16384 costs ~0.0001 bit per weight', f'{-__import__("math").log2(1 - 1 / 16384):.6f} bit'),
        ('WNR2-REF 7', 'WNM2 32 + 57,834 + 1,542 = 59,408 B', f'{32 + 3 * 567 * 17 * 2 + 3 * 257 * 2:,} B'),
        ('WNR2-REF 8', 'longest stream 28,676 B, bound 2n + 8 = 32,776', f'{16384 * 14 // 8 + 4:,} B (16,384 symbols of frequency 1, '
         f'14 bits each, plus the 4-byte state); {2 * 16384 + 8:,}'),
        ('WNR2-REF 9', 'about 44 MB/s at 0.29-0.32 s per 12.75 MiB V4 slab', f'{12.75 * 2**20 / 0.32 / 1e6:.1f}-{12.75 * 2**20 / 0.29 / 1e6:.1f} MB/s'),
        ('WNR2-REF 9', '567 x 32 B = 18 KB; a 16384-entry LUT per context 9.3 MB', f'{567 * 32:,} B; {567 * 16384 / 1e6:.2f} MB'),
        ('WNR2-REF 10', 'scale kernel shared memory 90,112 B', f'{16384 * 5 + 64 * 32 * 4:,} B (pre-Ampere build: {16384 + 64 * 32 * 4:,} B)'),
        ('WNR2-REF 10', 'model tables 359,936 B with lookup 1; buckets ~6.97 MB',
         f'{(59408 + 255) // 256 * 256 + (3 * 567 * 32 + 255) // 256 * 256 + 3 * 16384 * 4 + 3 * 16384:,} B; '
         f'{3 * 567 * 1024 * 4 / 1e6:.2f} MB'),
        ('WNR2-REF 10', '8 lanes x 2 x 17.93 MiB = ~287 MiB per thread', f'{raw / 2**20:.2f} MiB, {16 * raw / 2**20:.1f} MiB'),
        ('WNR2-REF 10', 'an 11 MiB V4 record: 12 scale blocks + 192 weight blocks of 32 threads',
         f'{3 * -(-(13369344 // 17 // 3 // 2048) // 32)} + {3 * -(-(32 * (13369344 // 17 // 3) // 4096) // 32)} '
         '(12.75 MiB slab = 17 x 3 x 262,144, chunks 4096:2048)'),
    ]
    for where, stated, computed in rows:
        print(f'N {where}: {stated} | computed: {computed}')


u1()
u2()
u3()
u4()
p1()
p2()
p3()
m1()
g1()
n_checks()
print('done')
