# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""W1 pre-check before the full build: GPU-encode N V4.1 experts of the WEX1
container at WEIGHT:SCALE chunks with an existing WNM2 model (any model is
lossless), verify each with the CPU reference, and write a records .bin for
test_decode_batch.exe (same layout as sample_records.py).

usage: python -I precheck_records.py WEX1.wex MODEL.wnm2 OUT.bin [count=8] [weight_chunk=2048] [scale_chunk=512]"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # repo nr2/ (nr2lib.py + wex_nr2_tools.dll)
import nr2lib as L   # noqa: E402


def main():
    src, model, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    wc = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
    sc = int(sys.argv[6]) if len(sys.argv) > 6 else 512
    blob = model.read_bytes()
    _, _, _, entries = L.read_container(src)
    tools = L.Tools()
    step = len(entries) // count
    out = bytearray(struct.pack('<I', len(blob)) + blob + struct.pack('<I', count))
    with open(src, 'rb') as f:
        for i in range(0, step * count, step):
            e = entries[i]
            f.seek(e[L.F_OFFSET]); raw = f.read(L.raw_size(e))
            rc, rec, written = tools.encode(blob, raw, L.ENTRY.pack(*e), 0, wc, sc, len(raw), gpu=True)
            assert rc == 0, (i, rc, written)
            padded = L.align(len(rec))
            new = list(e); new[L.F_TOTAL], new[L.F_FLAGS] = padded, L.WEX_ENTRY_NEURAL_RANS2
            neb = L.ENTRY.pack(*new)
            rec += bytes(padded - len(rec))
            assert tools.verify(blob, neb, rec, raw) == 0, i
            out += neb + struct.pack('<I', len(raw)) + raw + rec
            print(i, len(raw), padded, round(padded / len(raw), 4), flush=True)
    tools.release()
    out_path.write_bytes(out)


if __name__ == '__main__':
    main()
