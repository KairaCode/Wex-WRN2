# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Records of the published DeepSeek V4.1 Flash container as a bin of c/tests/test_wex_nr3_gpu.cu.

  python -I make_bin.py CONTAINER wex_nr3_tools.dll OUT.bin L:E,L:E,...

The container is the WEX3 WNR3-CZ file upgraded to WEX4 in place (wex4/wex4_convert.py upgrade): its WEX3 header
page is the blob of kind 16, and the WEX3 entry table, the model and the records are where they were. Each named
record is copied as stored, with its 72-byte WEX3 entry (file offset set to 0), and its expected slab is the
decode of the CPU reference (wex_nr3_decode of c/tools/wex_nr3_tools.c), so the test compares the GPU decoder
with the CPU reference on production records; the source checkpoint is not read. The layout is that of
c/tools/build_nr3.py bins: u32 model bytes, the model, u32 count, then per record the entry, u32 raw bytes, the
expected slab and the record. The container is opened read-only.
"""
import ctypes as C
import hashlib
import json
import os
import struct
import sys

ENTRY = struct.Struct('<QIII6H6H3I3II')


def wex3_page(f):
    f.seek(0)
    page = f.read(4096)
    if page[:4] != b'4XEW':
        return page
    index_offset, _, blob_count = struct.unpack_from('<QQI', page, 32)
    f.seek(index_offset)
    blobs = f.read(64 * blob_count)
    for i in range(blob_count):
        kind, _, offset, size, digest, _ = struct.unpack_from('<IIQQ32sQ', blobs, 64 * i)
        if kind == 16 and size == 4096:
            f.seek(offset)
            page = f.read(4096)
            assert hashlib.sha256(page).digest() == digest, 'the saved WEX3 header page is damaged'
            return page
    raise SystemExit('a WEX4 file without a saved WEX3 header page')


def main():
    path, dll_path, out, picks = sys.argv[1:5]
    dll = C.CDLL(os.path.abspath(dll_path))
    P = C.c_void_p
    dll.wex_nr3_decode.argtypes = [P, P, P, C.c_uint32, C.c_uint32, P]
    pairs = [tuple(map(int, x.split(':'))) for x in picks.split(',')]
    rows = []
    with open(path, 'rb') as f, open(out + '.partial', 'wb') as o:
        page = wex3_page(f)
        magic, version, layers, experts, _, table_offset, flags, _ = struct.unpack_from('<IIIIQQII', page)
        assert magic == 0x57455833 and version == 3 and flags & 32, 'not a WEX3 WNR3 container'
        dmagic, model_bytes, model_offset, model_id, _ = struct.unpack_from('<IIQII', page, 64)
        assert dmagic == 0x444D4E57
        f.seek(model_offset)
        model = f.read(model_bytes)
        f.seek(table_offset)
        table = f.read(layers * experts * ENTRY.size)
        o.write(struct.pack('<I', model_bytes) + model + struct.pack('<I', len(pairs)))
        for layer, expert in pairs:
            entry = table[(layer * experts + expert) * ENTRY.size:][:ENTRY.size]
            e = ENTRY.unpack(entry)
            assert e[-1] == 5, f'L{layer} E{expert}: not a WNR3 record'
            f.seek(e[0])
            record = f.read(e[1])
            raw = e[2] + e[3]
            slab = C.create_string_buffer(raw)
            assert dll.wex_nr3_decode(model, entry, record, layer, expert, slab) == 0, f'L{layer} E{expert}: CPU decode failed'
            o.write(struct.pack('<Q', 0) + entry[8:] + struct.pack('<I', raw) + slab.raw + record)
            rows.append({'layer': layer, 'expert': expert, 'stored': e[1], 'raw': raw, 'flags': record[27],
                         'record_sha256': hashlib.sha256(record).hexdigest(),
                         'slab_sha256': hashlib.sha256(slab.raw).hexdigest()})
            print(f'L{layer:02d} E{expert:03d} stored {e[1]} raw {raw} flags 0x{record[27]:02x}', flush=True)
    os.replace(out + '.partial', out)
    json.dump({'model_id': f'0x{model_id:08x}', 'records': rows}, open(out + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
