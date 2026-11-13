# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""WEX4 converters (docs/WEX4-SPEC.md, sections 9 and 10). Standard library only.

  pack      safetensors -> WEX4. The source headers are kept verbatim, so unpack rebuilds every
            source file with the same SHA-256 (unless the container is value-exact).
      python -I wex4_convert.py pack OUT.wex4 SRC.safetensors [SRC2 ...] [--dll wex4_tools.dll]
             [--experts raw|nr3 --nr3-dll wex_nr3_tools.dll --model FILE.wnm2 [--cz]] [--part-mib 64]
  unpack    WEX4 -> the source safetensors files, each checked against its recorded SHA-256.
      python -I wex4_convert.py unpack IN.wex4 OUT_DIR [--dll wex4_tools.dll]
  upgrade   WEX1 (raw) or WEX3/WNR3 expert container -> WEX4, in place and non-destructive: the records
            stay where they are, the old header page and table are kept as hashed blobs, the index is appended.
      python -I wex4_convert.py upgrade FILE.wex [--names deepseek|qwen38]
  downgrade undo an upgrade (restores the old header page, truncates the appended part).
      python -I wex4_convert.py downgrade FILE.wex
"""
import ctypes
import hashlib
import io
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wex4  # noqa: E402
from wex4 import need, align, sha256, PAGE  # noqa: E402

ST_MAX_HEADER = 100_000_000
SMALL, GROUP = 64 << 10, 4 << 20      # tensors under SMALL share a record of up to GROUP bytes
ENTRY = struct.Struct('<QIII6H6H3I3II')   # legacy WEX3 entry, 72 B
WEX_HEADER = struct.Struct('<IIIIQQII')
EXPERT_NAME = re.compile(rb'layers\.(0|[1-9]\d*)\.ffn\.experts\.(0|[1-9]\d*)\.w([123])\.(weight|scale)\Z')
BLOB_LEGACY_PAGE, BLOB_LEGACY_TABLE, BLOB_LEGACY_GAP = 16, 17, 18
FLOATS = ('F16', 'BF16', 'F32', 'F64')


def _unique(pairs):
    out = {}
    for key, value in pairs:
        need(key not in out, f'duplicate JSON key {key!r}')
        out[key] = value
    return out


def st_index(path, data_check=True):
    """Strict safetensors header: (header bytes, data start, [(name, dtype, shape, begin, end)] in data order)."""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(8)
        need(len(head) == 8, 'safetensors: no header length')
        n = struct.unpack('<Q', head)[0]
        need(n <= ST_MAX_HEADER and n <= size - 8, 'safetensors: header length')
        raw = f.read(n)
    need(len(raw) == n, 'safetensors: short header')
    try:
        doc = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique)
    except (ValueError, RecursionError) as error:
        raise wex4.FormatError(f'safetensors: header is not UTF-8 JSON ({error})') from None
    need(isinstance(doc, dict), 'safetensors: header is not an object')
    meta = doc.pop('__metadata__', None)
    need(meta is None or (isinstance(meta, dict) and all(isinstance(v, str) for v in meta.values())), 'safetensors: metadata')
    tensors = []
    for name, info in doc.items():
        need(isinstance(info, dict) and set(info) == {'dtype', 'shape', 'data_offsets'}, f'safetensors: {name}')
        dtype, shape, span = info['dtype'], info['shape'], info['data_offsets']
        need(isinstance(dtype, str) and isinstance(shape, list) and len(shape) <= 8 and
             all(type(d) is int and 0 <= d <= wex4.U64_MAX for d in shape), f'safetensors: {name} shape')
        need(isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span) and
             0 <= span[0] <= span[1], f'safetensors: {name} offsets')
        elements = 1
        for d in shape:
            elements *= d
            need(elements <= wex4.U64_MAX, f'safetensors: {name} shape overflow')
        try:
            name = name.encode('utf-8')
        except UnicodeEncodeError:
            raise wex4.FormatError('safetensors: tensor name is not UTF-8') from None
        need(len(name) <= 65535 and b'\0' not in name, 'safetensors: tensor name too long or with NUL')
        bits = wex4.BITS[wex4.DTYPE_CODE[dtype]] if dtype in wex4.DTYPE_CODE else 0
        need(not bits or elements * bits == (span[1] - span[0]) * 8, f'safetensors: {name} size')
        tensors.append((name, dtype, tuple(shape), span[0], span[1]))
    tensors.sort(key=lambda t: (t[3], t[4]))
    at = 0
    for name, _, _, begin, end in tensors:   # contiguous from 0, no holes, no overlaps
        need(begin == at, f'safetensors: hole or overlap before {name!r}')
        at = end
    need(not data_check or 8 + n + at == size, 'safetensors: data does not end at the end of the file')
    return raw, 8 + n, tensors


def file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Nr3:
    """c/tools/wex_nr3_tools.c: the WNR3 encoder used for expert records."""

    def __init__(self, dll, model_path, cz):
        lib = ctypes.CDLL(os.fspath(dll))
        P, U32, SZ = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t
        lib.wex_nr3_prepare.argtypes = [P, P, ctypes.c_int, P, P]
        lib.wex_nr3_encode_record.argtypes = [P, P, P, U32, U32, U32, U32, P, SZ, P, P]
        lib.wex_nr3_verify.argtypes = [P, P, P, U32, U32, P]
        lib.wex_nr3_canonicalize.argtypes = [P, P]
        self.lib, self.cz = lib, cz
        with open(model_path, 'rb') as f:
            self.model = f.read()
        self.model_id = wex4.wnm2_id(self.model)

    def encode(self, entry, slab, layer, expert):
        """The padded WNR3 record of a raw slab, verified against the slab (Z(slab) with cz); None = keep raw."""
        coded = ctypes.create_string_buffer(len(slab))
        flags = self.lib.wex_nr3_prepare(entry, slab, int(self.cz), coded, None)
        need(flags >= 0, 'wex_nr3_prepare failed')
        cap = len(slab) * 4
        out, used, total = ctypes.create_string_buffer(cap), ctypes.c_size_t(), ctypes.c_size_t()
        status = self.lib.wex_nr3_encode_record(self.model, entry, coded, flags, layer << 16 | expert, 2048, 512, out, cap,
                                                ctypes.byref(used), ctypes.byref(total))
        need(status >= 0, 'wex_nr3_encode_record failed')
        if status == 1:
            return None
        record = out.raw[:total.value]
        expected = ctypes.create_string_buffer(slab, len(slab))
        if self.cz:
            self.lib.wex_nr3_canonicalize(entry, expected)
        fields = list(ENTRY.unpack(entry))
        fields[1], fields[-1] = len(record), 5
        need(self.lib.wex_nr3_verify(self.model, ENTRY.pack(*fields), record, layer, expert, expected) == 0,
             f'L{layer} E{expert}: encoded record does not decode to the source')
        return record


def expert_entry(scales, weights):
    """Legacy entry bytes (offset/total/flags 0) of 3 (rows, cols) scale and 3 weight shapes."""
    sizes = [r * c for r, c in scales] + [r * c for r, c in weights]
    offsets = [sum(sizes[:i]) for i in range(6)]
    return ENTRY.pack(0, 0, sum(sizes[:3]), sum(sizes[3:]), *[d for s in scales for d in s], *[d for s in weights for d in s],
                      *offsets, 0)


def pack(out_path, sources, clib=None, experts=None, nr3=None, part_bytes=64 << 20):
    """safetensors files -> one WEX4 file. experts: None (tensors as they come), 'raw' or 'nr3' (expert records)."""
    files, records, empty, meta_sources, groups = [], [], [], [], {}
    seen = set()
    expert_groups = {}

    def reader(path, start, begin, end):
        def read():
            with open(path, 'rb') as f:
                f.seek(start + begin)
                return f.read(end - begin)
        return read

    need(sources and part_bytes > 0, 'pack: no source file or bad part size')
    need(len({os.path.normcase(os.path.basename(p)) for p in sources}) == len(sources), 'two sources have the same file name')
    for path in sources:
        raw, start, tensors = st_index(path)
        meta_sources.append({'name': os.path.basename(path), 'bytes': os.path.getsize(path),
                             'sha256': file_sha256(path), 'header': raw.decode('utf-8')})
        pending = {}   # element width -> (tensors, readers, bytes) of the open small-tensor group

        def flush(width):
            specs, reads, size = pending.pop(width)
            records.append(wex4.Record(specs, size, lambda reads=reads: b''.join(r() for r in reads), ('bp', width, 0)))
        for name, dtype, shape, begin, end in tensors:
            need(name not in seen, f'tensor {name!r} appears twice')
            seen.add(name)
            spec, read, size = wex4.Tensor(name, dtype, shape, end - begin), reader(path, start, begin, end), end - begin
            match = experts and EXPERT_NAME.match(name)
            if match:
                key = (int(match[1]), int(match[2]))
                expert_groups.setdefault(key, {})[(int(match[3]), match[4])] = (spec, read)
                continue
            if not size:
                empty.append(spec)
                continue
            code = wex4.DTYPE_CODE.get(dtype, 0)
            width = max(1, wex4.BITS[code] // 8) if code else 1
            codec = ('bp', width, int(dtype in FLOATS))   # floats: exponent into the top plane
            if size < SMALL:
                if pending.get(width, (0, 0, 0))[2] + size > GROUP:
                    flush(width)
                specs, reads, total = pending.get(width, ([], [], 0))
                pending[width] = (specs + [spec], reads + [read], total + size)
                continue
            parts = (size + part_bytes - 1) // part_bytes
            for k in range(parts):
                lo, hi = k * part_bytes, min(size, (k + 1) * part_bytes)
                records.append(wex4.Record([spec] if not k else [], hi - lo, reader(path, start, begin + lo, begin + hi),
                                           codec, parts=parts))
        for width in list(pending):
            flush(width)
        files.append(path)

    blobs, layers, n_experts, required = [], 0, 0, 0
    if expert_groups:
        if nr3:
            blobs.append((wex4.BLOB_WNM2, nr3.model_id, nr3.model))
            required = wex4.REQ_VALUE_EXACT if nr3.cz else 0
        layers, n_experts = max(k[0] for k in expert_groups) + 1, max(k[1] for k in expert_groups) + 1
        need(layers <= 65536 and n_experts <= 65536, 'expert grid too large')
        for (layer, expert), parts in sorted(expert_groups.items()):
            need(len(parts) == 6, f'L{layer} E{expert}: incomplete expert')
            order = [parts[(m, b'scale')] for m in (1, 2, 3)] + [parts[(m, b'weight')] for m in (1, 2, 3)]
            specs, reads = [s for s, _ in order], [r for _, r in order]
            need(all(len(s.shape) == 2 and all(0 < d <= 65535 for d in s.shape) for s in specs), f'L{layer} E{expert}: expert tensor shape')
            entry = expert_entry([s.shape for s in specs[:3]], [s.shape for s in specs[3:]])

            def slab(reads=reads):
                return b''.join(r() for r in reads)
            codec = 'raw'
            if nr3:
                codec = ('pre', wex4.CODEC_WNR3, lambda entry=entry, slab=slab, layer=layer, expert=expert:
                         nr3.encode(entry, slab(), layer, expert))
            flags = wex4.REC_EXPERT | (wex4.REC_VALUE_EXACT if nr3 and nr3.cz else 0)
            records.append(wex4.Record(specs, sum(s.nbytes for s in specs), slab, codec, flags, layer << 16 | expert,
                                       1 if nr3 else 0))
    meta = json.dumps({'wex4_meta': 1, 'sources': meta_sources}, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return wex4.write(out_path, records, blobs, empty, meta, layers, n_experts, required, wex4.OPT_ST_SOURCES, encoder=clib)


def _header_tensors(path, rd):
    """Names in data order of a file that so far holds only its header; every entry must equal the container's."""
    raw, start, tensors = st_index(path, data_check=False)
    for name, dtype, shape, begin, end in tensors:
        need(name in rd.by_name, f'tensor {name!r} missing from the container')
        t = rd.tensors[rd.by_name[name]]
        need((t[1], tuple(t[3]), t[7]) == (wex4.DTYPE_CODE.get(dtype, 0), shape, end - begin), f'header and container disagree on {name!r}')
    return [t[0] for t in tensors]


def unpack(in_path, out_dir, clib=None):
    """Rebuild every source safetensors file; returns [(name, sha256 matches)]."""
    os.makedirs(out_dir, exist_ok=True)
    results = []
    with open(in_path, 'rb') as f:
        rd = wex4.Reader(f, clib)
        need(rd.optional & wex4.OPT_ST_SOURCES, 'container holds no safetensors sources')
        try:
            doc = json.loads(rd.meta.decode('utf-8'), object_pairs_hook=_unique)
            sources = [(s['name'], s['header'], s['sha256']) for s in doc['sources']]
            need(all(type(x) is str for s in sources for x in s), 'metadata: source fields')
        except (ValueError, RecursionError, KeyError, TypeError) as error:
            raise wex4.FormatError(f'metadata: {error!r}') from None
        names = [os.path.normcase(n) for n, _, _ in sources]
        need(len(set(names)) == len(names) and all(n and n == os.path.basename(n) and n not in ('.', '..') and
                                                   os.path.normcase(os.path.abspath(os.path.join(out_dir, n))) != os.path.normcase(os.path.abspath(in_path)) for n, _, _ in sources),
             'metadata: source file name (unsafe, repeated or the container itself)')
        cache = {}

        def record(r):
            if r not in cache:
                cache.clear()
                cache[r] = rd.record(r)
            return cache[r]
        for name, text, recorded in sources:
            try:
                raw = text.encode('utf-8')
            except UnicodeEncodeError:
                raise wex4.FormatError('metadata: header text') from None
            digest = hashlib.sha256()
            tmp = os.path.join(out_dir, name + '.partial')
            try:
              with open(tmp, 'wb') as out:
                for piece in [struct.pack('<Q', len(raw)), raw]:
                    out.write(piece)
                    digest.update(piece)
                out.flush()
                for key in _header_tensors(tmp, rd):
                    need(key in rd.by_name, f'tensor {key!r} missing from the container')
                    _, _, _, _, r, count, offset, nbytes = rd.tensors[rd.by_name[key]]
                    for k in range(count):
                        data = record(r + k)
                        piece = data[offset:offset + nbytes] if count == 1 else data
                        out.write(piece)
                        digest.update(piece)
              ok = digest.hexdigest() == recorded
              need(ok or rd.required & wex4.REQ_VALUE_EXACT, f'{name}: the rebuilt file does not match its SHA-256')
            except BaseException:
                os.remove(tmp)
                raise
            os.replace(tmp, os.path.join(out_dir, name))
            results.append((name, ok))
    return results


def _sync(f):
    f.flush()
    try:
        os.fsync(f.fileno())
    except (AttributeError, io.UnsupportedOperation):   # an in-memory file (tests, dry runs); real I/O errors propagate
        pass


class _Probe:
    """What wex4.Reader reads to validate: the file size, the header page and the index."""

    def __init__(self, size, parts):
        self.size, self.parts, self.at = size, parts, 0

    def seek(self, offset, whence=0):
        self.at = self.size if whence else offset

    def tell(self):
        return self.at

    def read(self, n):
        return self.parts[self.at][:n]


# tensor names and dtypes of an upgraded expert container, per model family (matrices w1, w2, w3)
NAMES = {
    'deepseek': (lambda L, e, m, w: f'layers.{L}.ffn.experts.{e}.w{m + 1}.{"weight" if w else "scale"}', 'F8_E8M0', 'I8'),
    'qwen38': (lambda L, e, m, w: f'model.language_model.layers.{L}.mlp.experts.{e}.'
                                  f'{("gate_proj", "down_proj", "up_proj")[m]}.weight{"" if w else "_scale"}', 'U8', 'U8'),
}


def upgrade(path, hash_records=True, family='deepseek'):
    """WEX1 / WEX3-WNR3 container -> WEX4 in place: appends the old header page and the index, then rewrites
    the header page. Nothing else is touched; downgrade() restores the file byte for byte.
    path may be an open binary file object (r+b) instead of a path."""
    with (open(path, 'r+b') if isinstance(path, (str, os.PathLike)) else path) as f:
        f.seek(0, os.SEEK_END)
        old_bytes = f.tell()
        f.seek(0)
        page = f.read(PAGE)
        need(len(page) == PAGE, 'file shorter than a header page')
        magic, version, layers, n_experts, _, table_offset, hflags, _ = WEX_HEADER.unpack_from(page)
        need((magic, version) in ((0x57455831, 1), (0x57455833, 3)), 'not a WEX1 or WEX3 container')
        nr3, cz = bool(hflags & 32), bool(hflags & 64)
        need(version == 1 or (nr3 and not hflags & ~(1 | 32 | 64)), 'only raw (WEX1) and WNR3 containers can be upgraded')
        table_bytes = layers * n_experts * ENTRY.size
        need(layers <= 65536 and n_experts <= 65536 and table_bytes <= old_bytes - min(table_offset, old_bytes), 'expert grid or short entry table')
        f.seek(table_offset)
        table = f.read(table_bytes)
        need(len(table) == table_bytes, 'short entry table')
        entries = [ENTRY.unpack_from(table, i * ENTRY.size) for i in range(layers * n_experts)]
        present = sorted((e[0], i) for i, e in enumerate(entries) if e[1])
        need(present and [i for _, i in present] == sorted(i for _, i in present), 'records are not in table order')
        first_record = present[0][0]
        need(first_record >= table_offset + align(table_bytes), 'records before the end of the table')

        # the area before the records: model blob, table, and whatever else as opaque hashed blobs
        front = [(table_offset, align(table_bytes), BLOB_LEGACY_TABLE, 0, table_bytes)]
        if nr3:
            dmagic, model_bytes, model_offset, model_id, _ = struct.unpack_from('<IIQII', page, 64)
            need(dmagic == 0x444d4e57 and model_bytes == wex4.WNM2_BYTES, 'WNMD descriptor')
            front.append((model_offset, align(model_bytes), wex4.BLOB_WNM2, model_id, model_bytes))
        front.sort()
        blobs, at = [], PAGE
        for offset, size, kind, ident, used in front + [(first_record, 0, None, 0, 0)]:
            need(offset >= at and offset % PAGE == 0, 'overlapping front sections')
            if offset > at:
                blobs.append((BLOB_LEGACY_GAP, 0, at, offset - at))
            if kind is not None:
                blobs.append((kind, ident, offset, used))
            at = offset + size
        blob_rows = []
        for kind, ident, offset, size in blobs:
            f.seek(offset)
            data = f.read(size)
            need(not any(f.read(align(size) - size)), 'nonzero padding after a front section')
            need(kind != wex4.BLOB_WNM2 or wex4.wnm2_id(data) == ident, 'model id')
            blob_rows.append(wex4.BLOB.pack(kind, ident, offset, size, sha256(data), 0))
        model_blob = 1 + [b[0] for b in blobs].index(wex4.BLOB_WNM2) if nr3 else 0
        legacy_page_offset = old_bytes
        blob_rows.append(wex4.BLOB.pack(BLOB_LEGACY_PAGE, 0, legacy_page_offset, PAGE, sha256(page), 0))

        record_rows, tensor_rows, names, name_at, at, total, maximum = [], [], [], 0, first_record, 0, 0
        for r, (offset, i) in enumerate(present):
            _, stored, scale_bytes, weight_bytes = entries[i][:4]
            shapes, offsets, codec = entries[i][4:16], entries[i][16:22], entries[i][22]
            need(offset == at and codec in ((0, 5) if nr3 else (0,)), f'entry {i}: unexpected offset or codec')
            layer, expert = divmod(i, n_experts)
            decoded = scale_bytes + weight_bytes
            for m in range(6):
                name = NAMES[family][0](layer, expert, m % 3, m >= 3).encode()
                rows, cols = shapes[2 * m:2 * m + 2]
                tensor_rows.append(wex4.TENSOR.pack(name_at, len(name), wex4.DTYPE_CODE[NAMES[family][2 if m >= 3 else 1]], 2,
                                                    rows, cols, 0, 0, 0, 0, 0, 0, r, 1, offsets[m], rows * cols))
                names.append(name)
                name_at += len(name)
            digest = bytes(32)
            if hash_records:
                f.seek(offset)
                digest = sha256(f.read(stored))
            flags = wex4.REC_EXPERT | (wex4.REC_VALUE_EXACT if cz and codec == 5 else 0)
            record_rows.append(wex4.RECORD.pack(offset, stored, decoded, codec, flags, model_blob if codec == 5 else 0,
                                                layer << 16 | expert, 6 * r, 6, digest, 0))
            at += stored
            total += decoded
            maximum = max(maximum, decoded)
        need(at <= old_bytes, 'a record runs past the end of the file')
        need(at == old_bytes, f'records end at byte {at} but the file has {old_bytes}: an interrupted upgrade leaves '
                              f'appended bytes, truncate the file to {at} bytes and run again')
        order = sorted(range(len(names)), key=lambda k: names[k])
        index = b''.join(blob_rows) + b''.join(record_rows) + b''.join(tensor_rows) + \
            struct.pack(f'<{len(order)}I', *order) + b''.join(names)
        index_offset = legacy_page_offset + PAGE
        file_bytes = index_offset + align(len(index))
        new = wex4.HEADER.pack(wex4.MAGIC, 4, 0, wex4.REQ_VALUE_EXACT if cz else 0, 0, file_bytes, index_offset, len(index),
                               len(blob_rows), len(record_rows), len(names), layers, n_experts, maximum, name_at, 0, total,
                               sha256(index))
        new += bytes(wex4.HEADER_HASHED - len(new))
        new += sha256(new)
        wex4.Reader(_Probe(file_bytes, {0: new, index_offset: index + bytes(align(len(index)) - len(index))}))   # refuse before writing
        f.seek(legacy_page_offset)   # 1. append: the old container is still intact if this is interrupted
        f.write(page + index + bytes(align(len(index)) - len(index)))
        _sync(f)
        f.seek(0)                    # 2. one 4096-byte write switches the file to WEX4
        f.write(new)
        _sync(f)
    return dict(index_bytes=len(index), file_bytes=file_bytes, added=file_bytes - old_bytes, records=len(record_rows))


def _saved_page(f):
    """Offset of the old header page that upgrade appended, found without the WEX4 header: the first page from
    the end that is a WEX1/WEX3 header whose records end exactly there. For a file whose page 0 is damaged."""
    f.seek(0, os.SEEK_END)
    size = f.tell()
    for at in range(size - 2 * PAGE, max(PAGE, size - wex4.MAX_INDEX - 2 * PAGE) - 1, -PAGE):
        f.seek(at)
        page = f.read(PAGE)
        if page[:4] not in (b'1XEW', b'3XEW'):
            continue
        _, _, layers, n_experts, _, table_offset, _, _ = WEX_HEADER.unpack_from(page)
        if not 0 < layers * n_experts <= wex4.MAX_RECORDS or table_offset + layers * n_experts * ENTRY.size > at:
            continue
        f.seek(table_offset)
        table = f.read(layers * n_experts * ENTRY.size)
        ends = [e[0] + e[1] for e in (ENTRY.unpack_from(table, i * ENTRY.size) for i in range(layers * n_experts)) if e[1]]
        if ends and max(ends) == at:
            return at, page
    raise wex4.FormatError('no saved header page found')


def downgrade(path):
    with open(path, 'r+b') as f:
        try:
            rd = wex4.Reader(f)
        except wex4.FormatError:   # header page or index damaged (a torn switch): recover from the records' end
            at, page = _saved_page(f)
            f.seek(0)
            f.write(page)
            _sync(f)
            f.truncate(at)
            return
        legacy = [b for b in rd.blobs if b[0] == BLOB_LEGACY_PAGE]
        need(len(legacy) == 1, 'not an upgraded container')
        f.seek(legacy[0][2])
        page = f.read(PAGE)
        need(sha256(page) == legacy[0][4], 'saved header page is damaged')
        need(legacy[0][2] + PAGE == rd.index_offset and rd.index_offset + align(rd.index_bytes) == rd.file_bytes and
             page[:4] in (b'1XEW', b'3XEW'), 'not the layout that upgrade writes')
        f.seek(0)
        f.write(page)
        f.flush()
        os.fsync(f.fileno())
        f.truncate(legacy[0][2])


def main(argv):
    def option(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    if len(argv) < 3:
        print(__doc__)
        return 2
    clib = wex4.CLib(option('--dll')) if '--dll' in argv else None
    values = {argv[i + 1] for i, a in enumerate(argv[:-1]) if a.startswith('--') and a != '--cz'}
    positional = [a for a in argv[2:] if not a.startswith('--') and a not in values]
    if argv[1] == 'pack':
        mode = option('--experts')
        need(mode in (None, 'raw', 'nr3'), '--experts takes raw or nr3')
        nr3 = Nr3(option('--nr3-dll'), option('--model'), '--cz' in argv) if mode == 'nr3' else None
        root = pack(positional[0], positional[1:], clib, mode, nr3, int(option('--part-mib', '64')) << 20)
        print('root', root.hex())
    elif argv[1] == 'unpack':
        for name, ok in unpack(positional[0], positional[1], clib):
            print(name, 'sha256 identical' if ok else 'DIFFERS (value-exact container)')
    elif argv[1] == 'upgrade':
        print(upgrade(positional[0], family=option('--names', 'deepseek')))
    elif argv[1] == 'downgrade':
        downgrade(positional[0])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
