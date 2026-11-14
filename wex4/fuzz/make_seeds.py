# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Seed corpora for the fuzzing harnesses, from the conformance vectors.

    python -I wex4/fuzz/make_seeds.py OUT_DIR

OUT_DIR/file gets every file of wex4/vectors/MANIFEST.tsv (valid, invalid, legacy and the review findings): the
seeds of fuzz_wex4_file and fuzz_wex4_verify. OUT_DIR/wnr3 gets the stored bytes of every WNR3 record of the
accepted vectors: the seeds of fuzz_wnr3_record. OUT_DIR/engine gets small expert containers for the two engine
harnesses (fuzz_store_wex reads it, fuzz_q38_wex reads it and OUT_DIR/file):

  - a WEX1 container made here, 1 layer x 2 experts, raw records of 4096 bytes (hidden 64, intermediate 32);
  - a WEX2 container made here, 1 x 1, its scales stored as tiles of 0 to 3 bits (the smallest such record:
    8192 scale bytes, 131072 weight bytes);
  - the legacy vector tiny2-cz.wex3 (WNR3 records, canonical zero; 6 x 8 with two experts present), and the
    same file with its grid cut to 1 x 2, where every expert is present, as the Qwen3.8 reader requires;
  - the WEX4 form of the WEX1 and of the 1 x 2 WNR3 container (wex4_convert.upgrade).

No seed holds palette (codec 2), WNR (codec 3) or WNR2 (codec 4) records. A seed is named by the first 16 hex digits of the SHA-256 of its bytes, so the same
bytes are never stored twice and the paths stay short (Windows opens no path longer than 260 characters by
default). Standard library only."""
import hashlib
import io
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import wex4  # noqa: E402
import wex4_convert  # noqa: E402

VECTORS = os.path.join(HERE, '..', 'vectors')
PAGE = 4096
HEADER = struct.Struct('<IIIIQQII')        # legacy header: magic, version, layers, experts, slab, table offset, flags, 0
ENTRY = struct.Struct('<QIII6H6H3I3II')    # legacy entry, 72 B


def put(folder, data):
    with open(os.path.join(folder, hashlib.sha256(data).hexdigest()[:16]), 'wb') as f:
        f.write(data)


def pad(data):
    return data + bytes(-len(data) % PAGE)


def legacy(version, experts, rows_cols, tiles):
    """A WEX1 (raw records) or WEX2 (scales as tiles) container of one layer. rows_cols: for w1, w2, w3 the
    rows and the weight bytes per row; a row has one scale per 16 weight bytes."""
    rng = random.Random(20261114 + version)
    scale_shapes = [(r, c // 16) for r, c in rows_cols]
    scale_bytes = sum(r * c for r, c in scale_shapes)
    weight_bytes = sum(r * c for r, c in rows_cols)
    slab = len(pad(bytes(scale_bytes + weight_bytes)))
    scale_offsets, weight_offsets, at = [], [], 0
    for r, c in scale_shapes:
        scale_offsets.append(at)
        at += r * c
    for r, c in rows_cols:
        weight_offsets.append(at)
        at += r * c
    shapes = [x for shape in scale_shapes for x in shape] + [x for shape in rows_cols for x in shape]
    table, records, offset = b'', b'', 2 * PAGE
    for _ in range(experts):
        weights = bytes(rng.randrange(256) for _ in range(weight_bytes))
        if tiles:   # one page: "WCS1", the decoded size, the stream size, then per 256 scales a base, a bit width and the bits
            stream = b''
            for t in range(scale_bytes // 256):
                bits = t % 4
                stream += bytes((118 + t % 5, bits)) + bytes(rng.randrange(256) for _ in range(32 * bits))
            record = pad(struct.pack('<III', 0x31534357, scale_bytes, len(stream)) + stream) + pad(weights)
            flags = 1 << 8 | 1   # one page of scale stream, codec 1
        else:
            record = pad(bytes(118 + rng.randrange(8) for _ in range(scale_bytes)) + weights)
            flags = 0
        table += ENTRY.pack(offset, len(record), scale_bytes, weight_bytes, *shapes, *scale_offsets, *weight_offsets, flags)
        records += record
        offset += len(record)
    magic = 0x57455830 + version
    return pad(HEADER.pack(magic, version, 1, experts, slab, PAGE, 1 if tiles else 0, 0)) + pad(table) + records


def upgraded(folder, data):
    """The WEX4 form of a WEX1 or WNR3 container (in place, on a temporary copy)."""
    path = os.path.join(folder, 'upgrade.tmp')
    with open(path, 'wb') as f:
        f.write(data)
    wex4_convert.upgrade(path, family='qwen38')
    with open(path, 'rb') as f:
        data = f.read()
    os.remove(path)
    return data


def main():
    out = sys.argv[1]
    files, records, engine = os.path.join(out, 'file'), os.path.join(out, 'wnr3'), os.path.join(out, 'engine')
    os.makedirs(files, exist_ok=True)
    os.makedirs(records, exist_ok=True)
    os.makedirs(engine, exist_ok=True)
    wex1 = legacy(1, 2, ((32, 32), (64, 16), (32, 32)), False)
    tiny2 = open(os.path.join(VECTORS, 'legacy', 'tiny2-cz.wex3'), 'rb').read()
    # its two experts are (0, 0) and (0, 1): a 1 x 2 grid has the same table page and the same record owners
    tiny2_1x2 = tiny2[:8] + struct.pack('<II', 1, 2) + tiny2[16:]
    for data in (wex1, legacy(2, 1, ((64, 512), (64, 512), (64, 1024)), True), tiny2, tiny2_1x2,
                 upgraded(engine, wex1), upgraded(engine, tiny2_1x2)):
        put(engine, data)
    rows = [line.split('\t') for line in open(os.path.join(VECTORS, 'MANIFEST.tsv'), encoding='utf-8').read().splitlines()
            if line and not line.startswith('#')]
    for path, verdict in ((row[0], row[1]) for row in rows):
        data = open(os.path.join(VECTORS, path), 'rb').read()
        put(files, data)
        if verdict != 'accept':
            continue
        reader = wex4.Reader(io.BytesIO(data))
        for record in reader.records:
            if record[3] == 5:   # codec 5: WNR3
                put(records, data[record[0]:record[0] + record[1]])
    print(f'seeds: {len(os.listdir(files))} files in {files}, {len(os.listdir(records))} WNR3 records in {records}, '
          f'{len(os.listdir(engine))} expert containers in {engine}')


if __name__ == '__main__':
    main()
