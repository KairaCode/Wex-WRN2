# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""GGUF <-> WEX4, lossless (docs/WEX4-SPEC.md section 14.1). Standard library only.

    python -I gguf2wex4.py info MODEL.gguf
    python -I gguf2wex4.py pack OUT.wex4 MODEL.gguf [--part-mib 64] [--experts-only]
    python -I gguf2wex4.py unpack IN.wex4 OUT_DIR        # rebuilds every GGUF part, SHA-256 checked
    python -I gguf2wex4.py unpack IN.wex4 --check        # the same, hashed only, nothing written

MODEL may be the first part of a split model (NAME-00001-of-000NN.gguf): every part goes in.

Mapping:
  - the bytes of each part before its tensor data (header, metadata, tensor infos, the padding up to
    the data) are one blob of kind 32, verbatim, id = the part's index;
  - each routed expert (blk.L.ffn_{gate,gate_up,up,down}_exps.weight, ne[2] = experts) is ONE record:
    its slices of those tensors in the order gate, gate_up, up, down, named NAME#E. A layer's experts
    are consecutive RAW records of one size, so expert E starts at first + E * stride;
  - every other tensor is a record of its own (split into parts above --part-mib), named as in the GGUF;
  - dtype: F32 F16 BF16 I8 I16 I32 I64 F64 as the WEX4 codes, any other ggml type t as the opaque
    code 128 + t; shape = ne reversed (outermost first), an expert slice without the expert axis;
  - metadata {"wex4_meta":1,"gguf_sources":[{"name","bytes","sha256","blob","data","alignment"}]},
    optional feature bit 1 (GGUF_SOURCES);
  - --experts-only: the routed-expert records alone (the trunk and every other tensor stay in the GGUF) and
    "experts_only":true in the metadata; unpack refuses such a container, it cannot rebuild the GGUF.
Records are not EXPERT-flagged (section 5.5 reserves that for the MXFP4 slab), so the expert grid of
the header is 0 x 0; the engine finds a layer's experts by name. Pack refuses a part whose bytes
between or after the tensors are not zero: unpack could not rebuild them.
"""
import hashlib
import io
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wex4  # noqa: E402
from wex4 import need  # noqa: E402

BLOB_GGUF_HEADER = 32
OPT_GGUF_SOURCES = 2
# ggml type -> (elements per block, bytes per block); ggml-common.h, the engine's ggml_quants.h
GGML = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
        10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
        16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
        23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2),
        34: (256, 54), 35: (256, 66), 39: (32, 17), 40: (64, 36), 41: (128, 18), 42: (64, 18), 43: (32, 33)}
RAW_DTYPE = {0: 'F32', 1: 'F16', 30: 'BF16', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64'}
EXPERT = re.compile(r'blk\.(0|[1-9]\d*)\.ffn_(gate|gate_up|up|down)_exps\.weight\Z')
KIND_ORDER = ('gate', 'gate_up', 'up', 'down')
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
MAX_HEADER = 1 << 30   # a blob's cap


def dtype_code(ggml_type):
    return wex4.DTYPE_CODE[RAW_DTYPE[ggml_type]] if ggml_type in RAW_DTYPE else 128 + ggml_type


class _In:
    def __init__(self, f, size):
        self.f, self.size, self.pos = f, size, 0

    def read(self, n):
        need(0 <= n <= self.size - self.pos, f'GGUF header truncated at byte {self.pos}')
        data = self.f.read(n)
        need(len(data) == n, 'short read')
        self.pos += n
        return data

    def u32(self):
        return struct.unpack('<I', self.read(4))[0]

    def u64(self):
        return struct.unpack('<Q', self.read(8))[0]

    def string(self):
        n = self.u64()
        need(n <= MAX_HEADER, 'GGUF string too long')
        return self.read(n)


def _skip_value(r, vtype, depth=0):
    need(depth < 4, 'GGUF array nested too deep')
    if vtype == 8:
        r.string()
    elif vtype == 9:
        atype, n = r.u32(), r.u64()
        if atype in SCALAR:
            need(n <= (r.size - r.pos) // SCALAR[atype], 'GGUF array too long')
            r.read(n * SCALAR[atype])
        else:
            need(atype in (8, 9) and n <= r.size - r.pos, 'GGUF array type or length')
            for _ in range(n):
                _skip_value(r, atype, depth + 1)
    else:
        need(vtype in SCALAR, f'GGUF value type {vtype}')
        r.read(SCALAR[vtype])


def parse(f, size):
    """(data_start, alignment, tensors) of one GGUF part; tensors are (name, ne, type, rel_offset, nbytes)
    in header order. Everything is checked against `size` before it is used."""
    r = _In(f, size)
    need(r.read(4) == b'GGUF', 'not a GGUF file')
    version = r.u32()
    need(version in (2, 3), f'GGUF version {version} (2 or 3 supported)')
    n_tensors, n_kv = r.u64(), r.u64()
    need(n_tensors <= 1 << 24 and n_kv <= 1 << 20, 'GGUF header counts')
    alignment = 32
    for _ in range(n_kv):
        key, vtype = r.string(), r.u32()
        if key == b'general.alignment' and vtype == 4:
            alignment = r.u32()
        else:
            _skip_value(r, vtype)
    need(0 < alignment <= 1 << 20 and not alignment & (alignment - 1), f'GGUF alignment {alignment}')
    tensors, names = [], set()
    for _ in range(n_tensors):
        name = r.string().decode('utf-8')
        need(name not in names, f'tensor {name} twice')
        names.add(name)
        n_dims = r.u32()
        need(1 <= n_dims <= 4, f'{name}: {n_dims} dims')
        ne = tuple(r.u64() for _ in range(n_dims))
        ggml_type, offset = r.u32(), r.u64()
        need(ggml_type in GGML, f'{name}: unknown ggml type {ggml_type}')
        block, block_bytes = GGML[ggml_type]
        need(all(0 < d <= 1 << 40 for d in ne) and ne[0] % block == 0, f'{name}: dims {ne}')
        rows = 1
        for d in ne[1:]:
            rows *= d
        tensors.append((name, ne, ggml_type, offset, ne[0] // block * block_bytes * rows))
    data = (r.pos + alignment - 1) // alignment * alignment
    if not tensors:   # a split's first part may be its metadata alone, unpadded
        data = min(data, size)
    need(data <= min(size, MAX_HEADER), 'GGUF header too large')
    end = data
    for name, ne, ggml_type, offset, nbytes in sorted(tensors, key=lambda t: t[3]):
        need(offset % alignment == 0 and data + offset >= end and data + offset + nbytes <= size,
             f'{name}: overlaps another tensor or lies outside the file')
        end = data + offset + nbytes
    return data, alignment, tensors


def split_parts(path):
    m = re.match(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$', os.fspath(path))
    if not m or int(m[2]) != 1:
        return [os.fspath(path)]
    parts = [f'{m[1]}-{i:05d}-of-{m[3]}.gguf' for i in range(1, int(m[3]) + 1)]
    need(all(os.path.isfile(p) for p in parts), 'split model is incomplete')
    return parts


def _zero_gaps(f, data, tensors, size):
    """1 when every byte between and after the tensors is zero (what unpack writes there)."""
    pos = data
    for _, _, _, offset, nbytes in sorted(tensors, key=lambda t: t[3]) + [(None, None, None, size - data, 0)]:
        gap = data + offset - pos
        f.seek(pos)
        while gap > 0:
            chunk = f.read(min(gap, 1 << 24))
            need(len(chunk) and not any(chunk), 'the GGUF has non-zero bytes outside its tensors')
            gap -= len(chunk)
        pos = data + offset + nbytes
    return 1


def scan(model):
    """Parts, their parse, the dense tensors and the expert layers of a GGUF model."""
    parts = split_parts(model)
    need(len({os.path.normcase(os.path.basename(p)) for p in parts}) == len(parts), 'two parts with one name')
    parsed, dense, layers, seen = [], [], {}, set()
    for i, path in enumerate(parts):
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            data, alignment, tensors = parse(f, size)
        parsed.append((path, size, data, alignment, tensors))
        for name, ne, ggml_type, offset, nbytes in tensors:
            need(name not in seen, f'tensor {name} in two parts')   # an expert kind would be overwritten silently
            seen.add(name)
            m = EXPERT.match(name)
            if m and len(ne) == 3:
                kinds = layers.setdefault(int(m[1]), {})
                kinds[m[2]] = (i, data + offset, nbytes, ne, ggml_type, name)
            else:
                dense.append((i, data + offset, nbytes, ne, ggml_type, name))
    for layer, kinds in layers.items():
        counts = {k[3][2] for k in kinds.values()}
        need(len(counts) == 1 and ('gate_up' in kinds) != ('gate' in kinds and 'up' in kinds) and 'down' in kinds,
             f'layer {layer}: expert tensors do not form gate/up/down of one expert count')
    return parsed, dense, layers


def pack(out_path, model, part_bytes=64 << 20, log=sys.stderr, experts_only=False):
    parsed, dense, layers = scan(model)
    need(layers or not experts_only, 'no routed-expert tensor to pack')
    files = {}

    def reader(part, start, n):
        def read():
            if part not in files:
                files[part] = open(parsed[part][0], 'rb')
            files[part].seek(start)
            data = files[part].read(n)
            need(len(data) == n, 'short read from the GGUF')
            return data
        return read

    blobs, sources = [], []
    for i, (path, size, data, alignment, tensors) in enumerate(parsed):
        with open(path, 'rb') as f:
            header = f.read(data)
            _zero_gaps(f, data, tensors, size)
            print(f'[gguf2wex4] {os.path.basename(path)}: hashing {size / 1e9:.1f} GB', file=log, flush=True)
            f.seek(0)
            h = hashlib.sha256()
            for block in iter(lambda: f.read(1 << 20), b''):   # hashlib.file_digest needs Python 3.11
                h.update(block)
            digest = h.hexdigest()
        blobs.append((BLOB_GGUF_HEADER, i, header))
        sources.append({'name': os.path.basename(path), 'bytes': size, 'sha256': digest, 'blob': i,
                        'data': data, 'alignment': alignment})

    records, done = [], set()
    shape = lambda ne: tuple(reversed(ne))   # noqa: E731
    items = sorted([(t[0], t[1], 'dense', t) for t in dense] +
                   [(k[0], k[1], 'expert', layer) for layer, kinds in layers.items() for k in kinds.values()])
    for _, _, what, item in items:
        if what == 'dense':
            if experts_only:   # the trunk and the other tensors stay in the GGUF
                continue
            part, start, nbytes, ne, ggml_type, name = item
            spec = wex4.Tensor(name, dtype_code(ggml_type), shape(ne), nbytes)
            n = (nbytes + part_bytes - 1) // part_bytes
            for k in range(n):
                lo, hi = k * part_bytes, min(nbytes, (k + 1) * part_bytes)
                records.append(wex4.Record([spec] if not k else [], hi - lo, reader(part, start + lo, hi - lo), parts=n))
            continue
        if item in done:
            continue
        done.add(item)
        kinds = [layers[item][k] for k in KIND_ORDER if k in layers[item]]
        n_exp = kinds[0][3][2]
        for e in range(n_exp):
            specs, reads = [], []
            for part, start, nbytes, ne, ggml_type, name in kinds:
                each = nbytes // n_exp
                specs.append(wex4.Tensor(f'{name}#{e}', dtype_code(ggml_type), shape(ne[:2]), each))
                reads.append(reader(part, start + e * each, each))
            records.append(wex4.Record(specs, sum(s.nbytes for s in specs), lambda reads=reads: b''.join(r() for r in reads)))
    print(f'[gguf2wex4] {len(records)} records ({sum(len(layers[k]) for k in layers) and len(layers)} expert layers), writing',
          file=log, flush=True)
    meta = json.dumps({'wex4_meta': 1, 'gguf_sources': sources, **({'experts_only': True} if experts_only else {})},
                      ensure_ascii=False, separators=(',', ':')).encode()
    try:
        return wex4.write(out_path, records, blobs, (), meta, optional=OPT_GGUF_SOURCES)
    finally:
        for f in files.values():
            f.close()


def unpack(in_path, out_dir=None, log=sys.stderr, reader=wex4.Reader):
    """Rebuild every GGUF part (out_dir None: hash only). Returns [(name, sha256)]; raises on a mismatch.
    reader(f) opens the container (wnr4gg.py passes one that also decodes codec-8 records)."""
    results = []
    with open(in_path, 'rb') as f:
        rd = reader(f)
        need(rd.optional & OPT_GGUF_SOURCES, 'container holds no GGUF sources')
        try:
            meta = json.loads(rd.meta.decode('utf-8'))
            sources = [(s['name'], int(s['bytes']), s['sha256'], int(s['blob'])) for s in meta['gguf_sources']]
        except (ValueError, KeyError, TypeError) as error:
            raise wex4.FormatError(f'metadata: {error!r}') from None
        need(not meta.get('experts_only'), 'an experts-only container (pack --experts-only) holds the routed experts '
             'alone, the rest stayed in the GGUF: unpack cannot rebuild the GGUF from it')
        for name, size, recorded, blob in sources:
            need(name and name == os.path.basename(name) and name not in ('.', '..') and 0 <= blob < len(rd.blobs) and
                 rd.blobs[blob][0] == BLOB_GGUF_HEADER, 'metadata: source entry')
        cache = {}

        def record(r):
            if r not in cache:
                cache.clear()
                cache[r] = rd.record(r)
            return cache[r]

        def tensor_bytes(name):
            _, _, _, _, r, count, offset, nbytes = rd.tensors[rd.by_name[name.encode()]]
            if count == 1:
                yield record(r)[offset:offset + nbytes]
            else:
                for k in range(count):
                    yield record(r + k)
        for name, size, recorded, blob in sources:
            header = rd.blob(blob)
            data, _, tensors = parse(io.BytesIO(header), size)
            need(data == len(header), f'{name}: header blob size')
            digest, pos = hashlib.sha256(), 0
            out = open(os.path.join(out_dir, name + '.partial'), 'wb') if out_dir else None
            try:
                def emit(piece):
                    nonlocal pos
                    digest.update(piece)
                    if out:
                        out.write(piece)
                    pos += len(piece)
                emit(header)
                for tname, ne, _, offset, nbytes in sorted(tensors, key=lambda t: t[3]):
                    emit(bytes(data + offset - pos))
                    if tname.encode() in rd.by_name:
                        need(rd.tensors[rd.by_name[tname.encode()]][7] == nbytes, f'{tname}: size differs')
                        pieces = tensor_bytes(tname)
                    else:
                        need(EXPERT.match(tname) and len(ne) == 3, f'{tname} missing from the container')
                        each = nbytes // ne[2]
                        keys = [f'{tname}#{e}' for e in range(ne[2])]
                        need(all(k.encode() in rd.by_name and rd.tensors[rd.by_name[k.encode()]][7] == each for k in keys),
                             f'{tname}: expert slices missing or of another size')
                        pieces = (p for k in keys for p in tensor_bytes(k))
                    for piece in pieces:
                        emit(piece)
                    need(pos == data + offset + nbytes, f'{tname}: rebuilt size')
                emit(bytes(size - pos))
                ok = digest.hexdigest() == recorded
                need(ok, f'{name}: the rebuilt file does not match its SHA-256')
            except BaseException:
                if out:
                    out.close()
                    os.remove(out.name)
                raise
            if out:
                out.close()
                os.replace(out.name, os.path.join(out_dir, name))
            print(f'[gguf2wex4] {name}: sha256 {recorded} identical', file=log, flush=True)
            results.append((name, recorded))
    return results


def info(model):
    parsed, dense, layers = scan(model)
    for path, size, data, alignment, tensors in parsed:
        print(f'{os.path.basename(path)}: {size:,} B, data at {data:,}, alignment {alignment}, {len(tensors)} tensors')
    types = {}
    for t in dense:
        types[t[4]] = types.get(t[4], 0) + t[2]
    print(f'dense: {len(dense)} tensors, {sum(t[2] for t in dense) / 1e9:.2f} GB, by ggml type {types}')
    if layers:
        first = min(layers)
        kinds = layers[first]
        n_exp = next(iter(kinds.values()))[3][2]
        slab = sum(k[2] // n_exp for k in kinds.values())
        print(f'experts: {len(layers)} layers x {n_exp}, layer {first}: ' +
              ', '.join(f'{k} {kinds[k][3]} t{kinds[k][4]}' for k in KIND_ORDER if k in kinds) +
              f', record {slab:,} B -> {wex4.align(slab):,} B stored')
        print(f'expert bytes {sum(k[2] for kinds in layers.values() for k in kinds.values()) / 1e9:.2f} GB')


def main(argv):
    def option(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    values = {argv[i + 1] for i, a in enumerate(argv[:-1]) if a == '--part-mib'}
    pos = [a for a in argv[2:] if not a.startswith('--') and a not in values]
    try:
        if len(argv) >= 3 and argv[1] == 'info':
            info(pos[0])
        elif len(argv) >= 4 and argv[1] == 'pack':
            print('root', pack(pos[0], pos[1], int(option('--part-mib', '64')) << 20,
                               experts_only='--experts-only' in argv).hex())
        elif len(argv) >= 4 and argv[1] == 'unpack':
            if '--check' not in argv:
                os.makedirs(pos[1], exist_ok=True)
            unpack(pos[0], None if '--check' in argv else pos[1])
        else:
            print(__doc__)
            return 2
    except wex4.FormatError as error:
        print('REFUSED:', error)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
