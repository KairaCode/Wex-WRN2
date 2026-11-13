# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""W1 check input: N random WNR2 records of the V4.1 container, paired with the
raw slabs of the same experts read from the WEX1 container, written as a
records .bin for lab\\wex-prod-20261105\\test_decode_batch.cu (u32 model_bytes,
model, u32 count, then per record: 72-byte entry, u32 raw_bytes, raw slab,
record of entry.total_bytes). The test decodes every record on the GPU through
wex_cuda_decode_batch_to_device and compares each byte with the WEX1 slab.

usage: python -I sample_records.py NR2.wex WEX1.wex OUT.bin [count=64] [seed=20261107]"""
import json
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # repo nr2/ (nr2lib.py + wex_nr2_tools.dll)
import nr2lib as L   # noqa: E402

GEOMETRY = slice(2, 22)   # every WexEntry field except offset, total_bytes and flags


def main():
    nr2_path, wex1_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 20261107
    h2, page, _, e2 = L.read_container(nr2_path)
    h1, _, _, e1 = L.read_container(wex1_path)
    assert h2[0] == L.WEX_MAGIC_V3 and h2[6] & L.WEX_HEADER_NEURAL_RANS2, 'not a WNR2 container'
    assert h1[0] == L.WEX_MAGIC_V1 and h2[2:5] == h1[2:5], 'layer/expert geometry differs'
    magic, model_bytes, model_at, model_id, _ = struct.unpack_from('<IIQII', page, 64)
    assert magic == L.WNMD_MAGIC and model_bytes == L.MODEL_BYTES
    with open(nr2_path, 'rb') as f:
        f.seek(model_at)
        blob = f.read(model_bytes)
    assert L.fnv1a(blob) == model_id
    assert len(e1) == len(e2) and all(a[GEOMETRY] == b[GEOMETRY] for a, b in zip(e1, e2))
    wnr2 = [i for i, e in enumerate(e2) if e[L.F_TOTAL] and e[L.F_FLAGS] == L.WEX_ENTRY_NEURAL_RANS2]
    picked = sorted(random.Random(seed).sample(wnr2, count))
    out = bytearray(struct.pack('<I', model_bytes) + blob + struct.pack('<I', count))
    with open(nr2_path, 'rb') as f2, open(wex1_path, 'rb') as f1:
        for i in picked:
            new, old = e2[i], e1[i]
            f1.seek(old[L.F_OFFSET]); raw = f1.read(L.raw_size(old))
            f2.seek(new[L.F_OFFSET]); record = f2.read(new[L.F_TOTAL])
            assert len(raw) == L.raw_size(old) and len(record) == new[L.F_TOTAL]
            out += L.ENTRY.pack(*new) + struct.pack('<I', len(raw)) + raw + record
    out_path.write_bytes(out)
    print(json.dumps({'records': count, 'seed': seed, 'wnr2_records': len(wnr2), 'fallback_raw': len(e2) - len(wnr2),
                      'model_id': f'0x{model_id:08x}', 'picked': picked}))


if __name__ == '__main__':
    main()
