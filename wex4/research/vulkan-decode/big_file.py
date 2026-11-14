#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Make a WEX4 file larger than 4 GiB out of a small one, to check that wnr3_vk reads records at 64-bit offsets.

Usage: python -I big_file.py SMALL.wex4 BIG.wex4 [GIB]

BIG.wex4 is SMALL.wex4 with GIB (default 5) blobs of 1 GiB of zero bytes, of a kind no reader knows (0x7a65726f),
placed after the blobs of SMALL and before its records. Every record keeps its bytes and moves up by a little
more than GIB * 2^30. The index and the header are rewritten with their hashes, so the file is a valid WEX4
file; no record byte changes. Needs GIB GiB of disk. Standard library only."""
import hashlib
import struct
import sys

PAGE = 4096
HEADER = struct.Struct('<IHHQQQQQIIIIIIQQQ32s')   # as in wex4/wex4.py
BLOB = struct.Struct('<IIQQ32sQ')
RECORD = struct.Struct('<QIIHHIIII32sI')
GIB = 1 << 30
KIND = 0x7a65726f


def align(n):
    return (n + PAGE - 1) & ~(PAGE - 1)


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit(__doc__)
    extra = int(argv[3]) if len(argv) == 4 else 5
    with open(argv[1], 'rb') as f:
        small = f.read()
    head = list(HEADER.unpack_from(small))
    file_bytes, index_offset, index_bytes, blobs, records = head[5], head[6], head[7], head[8], head[9]
    assert head[0] == 0x57455834 and index_offset == PAGE, 'a WEX4 file with the index first is needed'
    index = small[PAGE:PAGE + index_bytes]
    blob_rows = [list(BLOB.unpack_from(index, i * BLOB.size)) for i in range(blobs)]
    at = blobs * BLOB.size
    record_rows = [list(RECORD.unpack_from(index, at + i * RECORD.size)) for i in range(records)]
    rest = index[at + records * RECORD.size:]
    old_data = PAGE + align(index_bytes)                       # where the blobs of SMALL start
    first_record = min(r[0] for r in record_rows)
    assert all(b[2] < first_record for b in blob_rows), 'blobs before records are needed'
    new_index_bytes = index_bytes + extra * BLOB.size
    new_data = PAGE + align(new_index_bytes)
    zero = hashlib.sha256()
    block = bytes(1 << 24)
    for _ in range(GIB >> 24):
        zero.update(block)
    for b in blob_rows:
        b[2] += new_data - old_data
    big_at = first_record + new_data - old_data
    for i in range(extra):
        blob_rows.append([KIND, 0, big_at + i * GIB, GIB, zero.digest(), 0])
    shift = new_data - old_data + extra * GIB
    for r in record_rows:
        r[0] += shift
    new_index = b''.join(BLOB.pack(*b) for b in blob_rows) + b''.join(RECORD.pack(*r) for r in record_rows) + rest
    assert len(new_index) == new_index_bytes
    head[5], head[7], head[8] = file_bytes + shift, new_index_bytes, blobs + extra
    head[17] = hashlib.sha256(new_index).digest()
    page = HEADER.pack(*head) + small[HEADER.size:4064]
    with open(argv[2], 'wb') as f:
        f.write(page + hashlib.sha256(page).digest())
        f.write(new_index + bytes(align(new_index_bytes) - new_index_bytes))
        f.write(small[old_data:first_record])                  # the blobs of SMALL
        for _ in range(extra * (GIB >> 24)):
            f.write(block)
        f.write(small[first_record:])                          # its records
    print(f'{argv[2]}: {file_bytes + shift} bytes, first record at {min(r[0] for r in record_rows)}')


if __name__ == '__main__':
    main(sys.argv)
