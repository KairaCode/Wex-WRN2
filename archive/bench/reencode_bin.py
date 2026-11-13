# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Re-encode a records_*.bin sample file (written by rechunk.py: model, entries,
raw slabs, records) at another weight:scale chunk, from its stored raw slabs,
verifying every record with the lab encoder's CPU reference.

usage: python -I reencode_bin.py IN.bin OUT.bin WEIGHT_CHUNK SCALE_CHUNK"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nr2'))   # repo nr2/
import nr2lib as L   # noqa: E402

SRC, DST, WC, SC = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])


def main():
    b = SRC.read_bytes()
    at = 0
    mb, = struct.unpack_from('<I', b, at); at += 4
    blob = b[at:at + mb]; at += mb
    n, = struct.unpack_from('<I', b, at); at += 4
    tools = L.Tools(L.HERE / 'wex_nr2_tools.dll')
    out = bytearray(struct.pack('<I', mb) + blob + struct.pack('<I', n))
    for _ in range(n):
        e = list(L.ENTRY.unpack_from(b, at)); at += L.ENTRY.size
        rb, = struct.unpack_from('<I', b, at); at += 4
        raw = b[at:at + rb]; at += rb
        at += e[L.F_TOTAL]
        plain = list(e); plain[L.F_FLAGS] = 0; plain[L.F_TOTAL] = L.align(rb)
        rc, rec, _ = tools.encode(blob, raw, L.ENTRY.pack(*plain), 0, WC, SC, L.align(rb) + (1 << 20), gpu=True)
        assert rc == 0, rc
        padded = L.align(len(rec))
        assert padded < L.align(rb)
        e[L.F_TOTAL] = padded
        neb = L.ENTRY.pack(*e)
        rec = rec + bytes(padded - len(rec))
        assert tools.verify(blob, neb, rec, raw) == 0
        out += neb + struct.pack('<I', rb) + raw + rec
    tools.release()
    DST.write_bytes(out)
    print('wrote', DST, n, 'records')


if __name__ == '__main__':
    main()
