#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Source of testdata/planes.wex4: tensors of 1, 2, 4 and 8 bytes per element for the WBP1 codec.

The WBP1 records of the conformance vectors have 1 or 2 planes, and no rANS-coded plane of theirs holds the
byte 255. This script writes a safetensors file whose tensors give WBP1 records of 4 and 8 planes, with and
without the float rotation (transform 1), and planes in which 255 is a frequent byte (small negative integers).

    python -I make_planes.py OUT.safetensors

prints the SHA-256 of every tensor's bytes: what a reader must return for that name. README.md of this
folder has the command that packs the file into WEX4. Standard library only. Deterministic.
"""
import hashlib
import json
import math
import random
import struct
import sys


def main() -> int:
    rng = random.Random(20261114)
    n = 3000
    walk, x = [], 0.0
    for _ in range(n):
        x += rng.gauss(0, 0.01)
        walk.append(x)
    small = [int(rng.gauss(0, 40)) for _ in range(n)]            # negative values: high bytes are 255
    tensors = [
        ("f64.walk", "F64", struct.pack(f"<{n}d", *walk)),
        ("f32.walk", "F32", struct.pack(f"<{n}f", *walk)),
        ("f32.sine", "F32", struct.pack(f"<{n}f", *(math.sin(k / 50) for k in range(n)))),
        ("f16.walk", "F16", struct.pack(f"<{n}e", *walk)),
        ("i64.small", "I64", struct.pack(f"<{n}q", *small)),
        ("i32.small", "I32", struct.pack(f"<{n}i", *small)),
        ("i16.small", "I16", struct.pack(f"<{n}h", *small)),
        ("u64.count", "U64", struct.pack(f"<{n}Q", *range(n))),
        ("i8.small", "I8", struct.pack(f"<{n}b", *(max(-128, min(127, v // 4)) for v in small))),
        ("u8.high", "U8", bytes(rng.choices([255, 254, 0, 1, 128], [50, 20, 15, 10, 5], k=n))),
        ("bool.sparse", "BOOL", bytes(rng.random() < 0.1 for _ in range(n))),
        # above 64 KiB: a record of its own, so that the float rotation is used with 4 and with 8 planes
        ("f32.big", "F32", struct.pack("<17000f", *(math.sin(k / 300) for k in range(17000)))),
        ("f64.big", "F64", struct.pack("<8200d", *(math.cos(k / 200) * 1e-3 for k in range(8200)))),
    ]
    header, payload = {}, bytearray()
    for name, dtype, data in tensors:
        start = len(payload)
        payload += data
        width = {"64": 8, "32": 4, "16": 2}.get(dtype[1:], 1)
        header[name] = {"dtype": dtype, "shape": [len(data) // width], "data_offsets": [start, len(payload)]}
        print(name, hashlib.sha256(data).hexdigest())
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(sys.argv[1], "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
