# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Copy what the WNR4 record experiment needs out of a WEX3 container, or out of one upgraded to WEX4 in place
(the published container is one), opened read-only.

  python -I fetch.py CONTAINER CACHE_DIR records L:E,L:E,...   header page, model blob, entry table, those records
  python -I fetch.py CONTAINER CACHE_DIR scan OUT.json          every record's header + u16 length table only

Nothing is written next to the container. Prints the number of bytes read from it.
"""
import hashlib, json, os, struct, sys
import numpy as np

HDR = struct.Struct('<IIIIQQII')
WNMD = struct.Struct('<IIQII')
ENTRY = struct.Struct('<QIII6H6H3I3II')


def wex3_page(f):
    """The WEX3 header page of the container. The published container is WEX4: `wex4_convert.py upgrade` kept
    its WEX3 header page as a blob of kind 16 near the end of the file and left the WEX3 entry table, the model
    and the records where they were. That saved page (its SHA-256 checked against the index) stands for page 0."""
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
    raise AssertionError('a WEX4 file without a saved WEX3 header page: not an upgraded WEX3 container')


def open_container(path):
    f = open(path, 'rb')
    page = wex3_page(f)
    magic, version, layers, experts, aligned, table_offset, flags, _ = HDR.unpack_from(page, 0)
    assert magic == 0x57455833 and version == 3, 'not a WEX3 container'
    _, model_bytes, model_offset, model_id, _ = WNMD.unpack_from(page, 64)
    f.seek(model_offset); model = f.read(model_bytes)
    f.seek(table_offset); table = f.read(layers * experts * ENTRY.size)
    return f, page, model, table, layers, experts, 4096 + len(model) + len(table)


def lenc_group_bytes(lengths, group):
    """Coded length table: per `group` streams u16 min, u8 width, residuals packed at that width."""
    n = len(lengths)
    pad = (-n) % group
    lo = np.pad(lengths, (0, pad), constant_values=65535).reshape(-1, group).min(1)
    hi = np.pad(lengths, (0, pad), constant_values=0).reshape(-1, group).max(1)
    width = np.ceil(np.log2((hi - lo).astype(np.float64) + 1)).astype(np.int64)
    count = np.full(len(lo), group); count[-1] = group - pad
    return int((3 + (count * width + 7) // 8).sum())


def main():
    path, cache, mode, arg = sys.argv[1:5]
    f, page, model, table, layers, experts, read = open_container(path)
    os.makedirs(cache, exist_ok=True)
    if mode == 'records':
        open(os.path.join(cache, 'header.bin'), 'wb').write(page)
        open(os.path.join(cache, 'model.bin'), 'wb').write(model)
        open(os.path.join(cache, 'entries.bin'), 'wb').write(table)
        for item in arg.split(','):
            layer, expert = map(int, item.split(':'))
            out = os.path.join(cache, f'rec_{layer:02d}_{expert:03d}.bin')
            if os.path.exists(out):
                continue
            e = ENTRY.unpack_from(table, (layer * experts + expert) * ENTRY.size)
            assert e[-1] == 5, 'not a WNR3 record'
            f.seek(e[0]); data = f.read(e[1]); read += len(data)
            open(out, 'wb').write(data)
    else:   # scan: exact padding and length-table figures for the whole container
        rows = []
        for index in range(layers * experts):
            e = ENTRY.unpack_from(table, index * ENTRY.size)
            assert e[-1] == 5
            f.seek(e[0]); head = f.read(32)
            magic, _, sp, wp, wchunk, schunk = struct.unpack_from('<6I', head, 0)
            assert magic == 0x33524E57
            streams = sum((e[4 + 2 * m] * e[5 + 2 * m] + schunk - 1) // schunk for m in range(3)) + \
                      sum((e[10 + 2 * m] * e[11 + 2 * m] * 2 + wchunk - 1) // wchunk for m in range(3))
            lengths = np.frombuffer(f.read(2 * streams), '<u2').astype(np.int64)
            read += 32 + 2 * streams
            used = 32 + 2 * streams + sp + wp
            assert lengths.sum() == sp + wp and (used + 4095) // 4096 * 4096 == e[1]
            rows.append([index // experts, index % experts, e[1], used, streams, wchunk, schunk,
                         lenc_group_bytes(lengths, 32), lenc_group_bytes(lengths, 64)])
        json.dump({'columns': ['layer', 'expert', 'total_bytes', 'used_bytes', 'streams', 'wchunk', 'schunk',
                               'lenc_g32_bytes', 'lenc_g64_bytes'], 'rows': rows}, open(arg, 'w'))
    print('bytes read from the container:', read)


if __name__ == '__main__':
    main()
