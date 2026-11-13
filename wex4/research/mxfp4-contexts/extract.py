# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Step 1: pull 24 real experts out of the production WEX3 container (read-only) into the coded domain.

  python -I extract.py CONTAINER.wex out

For each pick: read the codec-5 record, CPU-decode it with wex_nr3_tools.dll (reference decoder), bring the
slab back to the coded domain with wex_nr3_prepare(cz=1) (canonical zero + ceil form = what the rANS streams
carry) and check that the flags and rule-g anchors match the record header. Writes out/slabs/L##_E###.coded,
out/prod_model.wnm2 and out/manifest.json (entry bytes, record header fields, real payload sizes).
Reads 24 records (~0.39 GiB) plus the header, model and table; writes only under OUT."""
import ctypes as C
import json
import os
import random
import struct
import sys

HEADER = struct.Struct('<IIIIQQII')
ENTRY = struct.Struct('<QIII6H6H3I3II')
LAYERS = [0, 5, 11, 16, 22, 28, 33, 39]
PER_LAYER = 3
SEED = 20261111


def picks():
    return [(L, e) for L in LAYERS for e in sorted(random.Random(SEED + L).sample(range(192), PER_LAYER))]


def main():
    container, out = sys.argv[1], sys.argv[2]
    lib = C.CDLL(os.path.abspath(os.path.join(out, 'wex_nr3_tools.dll')))
    P = C.c_void_p
    lib.wex_nr3_prepare.argtypes = [P, P, C.c_int, P, P]
    lib.wex_nr3_decode.argtypes = [P, P, P, C.c_uint32, C.c_uint32, P]
    os.makedirs(os.path.join(out, 'slabs'), exist_ok=True)
    rows = []
    with open(container, 'rb') as f:
        page = f.read(4096)
        magic, version, layers, experts, _, table_offset, flags, _ = HEADER.unpack_from(page)
        assert magic == 0x57455833 and layers == 40 and experts == 192, 'not the V4.1 WEX3 container'
        wnmd, model_bytes, model_offset, model_id, _ = struct.unpack_from('<IIQII', page, 64)
        assert wnmd == 0x444D4E57 and model_bytes == 59408
        f.seek(model_offset)
        blob = f.read(model_bytes)
        open(os.path.join(out, 'prod_model.wnm2'), 'wb').write(blob)
        f.seek(table_offset)
        table = f.read(layers * experts * ENTRY.size)
        for L, e in picks():
            eb = table[(L * experts + e) * ENTRY.size:][:ENTRY.size]
            ent = ENTRY.unpack(eb)
            assert ent[-1] == 5, 'not a WNR3 record'
            raw = ent[2] + ent[3]
            f.seek(ent[0])
            record = f.read(ent[1])
            slab = C.create_string_buffer(raw)
            assert lib.wex_nr3_decode(blob, eb, record, L, e, slab) == 0, f'decode failed L{L} E{e}'
            coded = C.create_string_buffer(raw)
            stats = (C.c_uint32 * 16)()
            fl = lib.wex_nr3_prepare(eb, slab, 1, coded, stats)
            # the decoded slab is Z(source); preparing it again must give the record's own flags and anchors
            assert fl == record[27] and bytes(stats[9:12]) == record[24:27], 'coded domain mismatch'
            name = f'L{L:02d}_E{e:03d}.coded'
            open(os.path.join(out, 'slabs', name), 'wb').write(coded.raw[:raw])
            h = struct.unpack_from('<8I', record)
            rows.append({'layer': L, 'expert': e, 'file': name, 'entry_hex': eb.hex(), 'raw_bytes': raw,
                         'record_total_bytes': ent[1], 'scale_payload_bytes': h[2], 'weight_payload_bytes': h[3],
                         'weight_chunk': h[4], 'scale_chunk': h[5], 'anchors': list(record[24:27]),
                         'flags': record[27], 'transformed_blocks': [stats[0], stats[3], stats[6]]})
            print(name, ent[1], h[3], fl, flush=True)
    json.dump({'container_bytes': os.path.getsize(container), 'model_id': f'0x{model_id:08x}',
               'header_flags': flags, 'rows': rows}, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
