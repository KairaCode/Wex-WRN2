# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""WEX4 prototype on real data: sizes, speeds, exactness. Reads the model and the production container
read-only; writes only under WORK (a scratch folder with a few GB free). CPU only.

    python -I measure_real.py MODEL_DIR CONTAINER.nr3.wex TOOLS_DIR WORK [a] [b] [c] [d] [e]

TOOLS_DIR = wex4/out after wex4/build_and_test.sh (wex4_tools.dll, wex_nr3_tools.dll).
 a  one whole BF16 shard: pack, verify, unpack, SHA-256 of the rebuilt file
 b  the non-expert tensors of one layer shard (F8_E4M3, F8_E8M0, F32, BF16), same steps
 c  4 experts of that shard from safetensors with the production model: records identical to production?
 d  the production container's WEX4 index, built in memory from its real header, model and table
    (the container is only read)
 e  SHA-256 and WBP1 decode throughput
"""
import ctypes
import glob
import hashlib
import io
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import wex4  # noqa: E402
import wex4_convert as conv  # noqa: E402

MODEL, CONTAINER, TOOLS, WORK = sys.argv[1:5]
STEPS = sys.argv[5:] or list('abcde')
C = wex4.CLib(os.path.join(TOOLS, 'wex4_tools.dll'))
SHARDS = sorted(glob.glob(os.path.join(MODEL, '*.safetensors')))
LAYER_SHARD = SHARDS[20]
os.makedirs(WORK, exist_ok=True)
RESULT = {}


def st_subset(src, dst, keep, limit=None):
    """A new safetensors file with the tensors of src whose name passes keep()."""
    raw, start, tensors = conv.st_index(src)
    header, at, picked = {}, 0, []
    for name, dtype, shape, begin, end in tensors:
        if keep(name) and (limit is None or len(picked) < limit):
            header[name.decode()] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [at, at + end - begin]}
            at += end - begin
            picked.append((begin, end))
    blob = json.dumps(header, separators=(',', ':')).encode()
    blob += b' ' * (-len(blob) % 8)
    with open(src, 'rb') as f, open(dst, 'wb') as out:
        out.write(struct.pack('<Q', len(blob)) + blob)
        for begin, end in picked:
            f.seek(start + begin)
            out.write(f.read(end - begin))
    return len(picked)


def round_trip(label, source, **pack_args):
    out, back = os.path.join(WORK, label + '.wex4'), os.path.join(WORK, label + '_back')
    t0 = time.perf_counter()
    conv.pack(out, [source], C, **pack_args)
    t1 = time.perf_counter()
    with open(out, 'rb') as f:
        rd = wex4.Reader(f, C)
        t2 = time.perf_counter()
        rd.content_digest(True)
        t3 = time.perf_counter()
        codecs = {}
        for rec in rd.records:
            c = codecs.setdefault(rec[3], [0, 0, 0])
            c[0] += 1
            c[1] += rec[2]
            c[2] += rec[1]
        index_bytes = rd.index_bytes
    results = conv.unpack(out, back, C)
    t4 = time.perf_counter()
    same = file_sha(os.path.join(back, os.path.basename(source))) == file_sha(source)
    raw, packed = os.path.getsize(source), os.path.getsize(out)
    RESULT[label] = dict(source_bytes=raw, wex4_bytes=packed, ratio=round(packed / raw, 5), index_bytes=index_bytes,
                         records={str(k): dict(count=v[0], decoded=v[1], stored=v[2], ratio=round(v[2] / v[1], 5))
                                  for k, v in codecs.items()},
                         pack_s=round(t1 - t0, 2), open_ms=round((t2 - t1) * 1e3, 2), verify_decode_s=round(t3 - t2, 2),
                         unpack_s=round(t4 - t3, 2), rebuilt_sha256_identical=same,
                         unpack_reported=[ok for _, ok in results])
    print(label, json.dumps(RESULT[label]), flush=True)
    for path in (os.path.join(back, os.path.basename(source)),):
        os.remove(path)
    return out


def file_sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Holes:
    """A file of a given size held in memory as the pieces written to it; everything else reads as zero.
    Lets the upgrade run on the production container's real header, model and table without a 123 GB file."""

    def __init__(self, size):
        self.size, self.at, self.pieces = size, 0, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def seek(self, offset, whence=0):
        self.at = offset if whence == 0 else self.size + offset if whence == 2 else self.at + offset
        return self.at

    def tell(self):
        return self.at

    def flush(self):
        pass

    def write(self, data):
        self.pieces.append((self.at, bytes(data)))
        self.at += len(data)
        self.size = max(self.size, self.at)

    def read(self, n):
        n = max(0, min(n, self.size - self.at))
        out = bytearray(n)
        for start, data in self.pieces:   # later writes win
            lo, hi = max(start, self.at), min(start + len(data), self.at + n)
            if lo < hi:
                out[lo - self.at:hi - self.at] = data[lo - start:hi - start]
        self.at += n
        return bytes(out)


def container_parts():
    with open(CONTAINER, 'rb') as f:
        page = f.read(4096)
        _, _, layers, experts, _, table_offset, flags, _ = conv.WEX_HEADER.unpack_from(page)
        _, model_bytes, model_offset, model_id, _ = struct.unpack_from('<IIQII', page, 64)
        f.seek(model_offset)
        model = f.read(model_bytes)
        f.seek(table_offset)
        table = f.read(layers * experts * 72)
    return page, model, model_offset, table, table_offset, layers, experts


if 'a' in STEPS:
    packed = round_trip('bf16_shard', SHARDS[0])
    if 'e' in STEPS:   # decode throughput of the WBP1 records of that file, from memory
        data = open(packed, 'rb').read()
        rd = wex4.Reader(io.BytesIO(data), C)
        stored = [(r, rd.stored(r)) for r, rec in enumerate(rd.records) if rec[3] == 6][:16]
        t0, total = time.perf_counter(), 0
        for r, blob in stored:
            total += len(C.bp_decode(blob, rd.records[r][2], rd.records[r][6]))
        dt = time.perf_counter() - t0
        RESULT['wbp1_decode'] = dict(records=len(stored), decoded_bytes=total, seconds=round(dt, 3),
                                     MB_per_s=round(total / dt / 1e6, 1), note='1 thread, scalar C, BF16 (1 rANS plane)')
        print('wbp1_decode', RESULT['wbp1_decode'], flush=True)
        del data
    os.remove(packed)

if 'b' in STEPS:
    sample = os.path.join(WORK, 'layer_other.safetensors')
    n = st_subset(LAYER_SHARD, sample, lambda name: not conv.EXPERT_NAME.match(name))
    RESULT['layer_other_tensors'] = n
    os.remove(round_trip('layer_other', sample))
    os.remove(sample)

if 'c' in STEPS:
    page, model, _, table, _, layers, experts = container_parts()
    model_path = os.path.join(WORK, 'prod.wnm2')
    open(model_path, 'wb').write(model)
    sample = os.path.join(WORK, 'experts4.safetensors')
    raw, start, tensors = conv.st_index(LAYER_SHARD)
    keys = sorted({(int(m[1]), int(m[2])) for m in (conv.EXPERT_NAME.match(t[0]) for t in tensors) if m})[:4]
    st_subset(LAYER_SHARD, sample, lambda name: (m := conv.EXPERT_NAME.match(name)) and (int(m[1]), int(m[2])) in keys)
    nr3 = conv.Nr3(os.path.join(TOOLS, 'wex_nr3_tools.dll'), model_path, True)
    out = os.path.join(WORK, 'experts4.wex4')
    t0 = time.perf_counter()
    conv.pack(out, [sample], C, 'nr3', nr3)
    dt = time.perf_counter() - t0
    identical = []
    with open(out, 'rb') as f, open(CONTAINER, 'rb') as prod:
        rd = wex4.Reader(f, C)
        for r, rec in enumerate(rd.records):
            layer, expert = rec[6] >> 16, rec[6] & 0xffff
            entry = conv.ENTRY.unpack_from(table, (layer * experts + expert) * 72)
            prod.seek(entry[0])
            identical.append((f'L{layer}E{expert}', rd.stored(r) == prod.read(entry[1]), rec[1], rec[2]))
    RESULT['experts_from_safetensors'] = dict(
        experts=[i[0] for i in identical], records_identical_to_production=[i[1] for i in identical],
        stored=[i[2] for i in identical], decoded=[i[3] for i in identical], pack_s=round(dt, 2),
        source_bytes=os.path.getsize(sample), wex4_bytes=os.path.getsize(out))
    print('experts', RESULT['experts_from_safetensors'], flush=True)
    for path in (sample, out, model_path):
        os.remove(path)

if 'd' in STEPS:
    page, model, model_offset, table, table_offset, layers, experts = container_parts()
    size = os.path.getsize(CONTAINER)
    stub = Holes(size)   # header page, model and table are real; the records are holes that are never read
    for at, data in ((0, page), (model_offset, model), (table_offset, table)):
        stub.seek(at)
        stub.write(data)
    t0 = time.perf_counter()
    info = conv.upgrade(stub, hash_records=False)
    t1 = time.perf_counter()
    with stub as f:
        t2 = time.perf_counter()
        rd = wex4.Reader(f, C)
        t3 = time.perf_counter()
        runs = 20
        for _ in range(runs):
            assert C.lib.wex4_validate(rd.page, rd.file_bytes, rd.index, len(rd.index)) == 0
        t4 = time.perf_counter()
        legacy = ctypes.create_string_buffer(layers * experts * 72)
        assert C.lib.wex4_expert_table(rd.page, rd.file_bytes, rd.index, len(rd.index), legacy) == 0
        RESULT['production_index'] = dict(
            container_bytes=size, records=info['records'], tensors=len(rd.tensors), index_bytes=info['index_bytes'],
            added_bytes=info['added'], added_fraction=info['added'] / size, build_index_s=round(t1 - t0, 2),
            python_open_ms=round((t3 - t2) * 1e3, 1), c_validate_ms=round((t4 - t3) / runs * 1e3, 2),
            legacy_table_identical=legacy.raw == table,
            note='record SHA-256 fields are zero in this stub: hashing the records means reading the whole container')
    print('production_index', RESULT['production_index'], flush=True)

if 'e' in STEPS:
    buf = os.urandom(256 << 20)
    t0 = time.perf_counter()
    hashlib.sha256(buf).digest()
    t1 = time.perf_counter()
    out = ctypes.create_string_buffer(32)
    C.lib.wex4_sha256_bytes(buf[:64 << 20], 64 << 20, out)
    t2 = time.perf_counter()
    RESULT['sha256_MB_per_s'] = dict(hashlib=round(256 / (t1 - t0)), c_scalar=round(64 / (t2 - t1)))
    with open(CONTAINER, 'rb') as f:   # 64 production records: read + hash, as a full verification would
        _, _, _, table, _, _, _ = container_parts()
        t0, total = time.perf_counter(), 0
        for i in range(0, 7680, 120):
            entry = conv.ENTRY.unpack_from(table, i * 72)
            f.seek(entry[0])
            total += len(hashlib.sha256(f.read(entry[1])).digest()) and entry[1]
        dt = time.perf_counter() - t0
    RESULT['production_record_hash'] = dict(records=64, bytes=total, seconds=round(dt, 2), MB_per_s=round(total / dt / 1e6),
                                            full_container_estimate_s=round(os.path.getsize(CONTAINER) / (total / dt)))
    print('sha', RESULT['sha256_MB_per_s'], RESULT['production_record_hash'], flush=True)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
if os.path.exists(path):   # steps can be rerun one at a time
    RESULT = {**json.load(open(path)), **RESULT}
with open(path, 'w') as f:
    json.dump(RESULT, f, indent=1)
print('done')
