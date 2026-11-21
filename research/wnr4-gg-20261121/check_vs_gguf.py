# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Every record of an experts-only WEX4 (gguf2wex4 pack --experts-only) against the GGUF bytes it came from:
the record's stored bytes (RAW) must equal the expert's slices of the GGUF tensors, and the record SHA-256 in the
index must match. Streams one record at a time.
    python -I check_vs_gguf.py WEX4 GGUF_PART1 [WEX4_TOOLS_DIR]   (default: the repository's wex4/)"""
import hashlib
import os
import re
import sys
import time

sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'wex4'))
import gguf2wex4  # noqa: E402
import wex4  # noqa: E402

path, model = sys.argv[1], sys.argv[2]
parsed, dense, layers = gguf2wex4.scan(model)
files = [open(p[0], 'rb') for p in parsed]
NAME = re.compile(r'(blk\.\d+\.ffn_(gate|gate_up|up|down)_exps\.weight)#(\d+)\Z')
slices = {}
for layer, kinds in layers.items():
    for k in kinds.values():
        part, start, nbytes, ne, ggml_type, name = k
        slices[name] = (part, start, nbytes // ne[2])
bad = done = 0
t0 = last = time.time()
with open(path, 'rb') as f:
    rd = wex4.Reader(f)
    for r, rec in enumerate(rd.records):
        offset, stored, decoded, codec, flags, blob, owner, first, count, digest = rec[:10]
        assert codec == wex4.CODEC_RAW, r
        f.seek(offset)
        data = f.read(stored)
        ok = hashlib.sha256(data).digest() == digest and not any(data[decoded:])
        at = 0
        for t in rd.tensors[first:first + count]:
            m = NAME.match(t[0].decode())
            part, start, each = slices[m.group(1)]
            e = int(m.group(3))
            files[part].seek(start + e * each)
            ok &= files[part].read(each) == data[at:at + each] and t[-1] == each
            at += each
        ok &= at == decoded
        bad += not ok
        done += 1
        if time.time() - last > 30:
            last = time.time()
            print(f'{done}/{len(rd.records)} records, {bad} bad, {time.time() - t0:.0f} s', flush=True)
print(f'DONE {done} records, {bad} bad, {time.time() - t0:.0f} s, root {rd.root.hex()}', flush=True)
sys.exit(1 if bad else 0)
