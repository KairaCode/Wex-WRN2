"""Bin (test_decode_batch layout) of WNR2 production records (read-only from the WNR2 container) for the
listed experts; expected slab = WEX1 slab. usage: python -I make_wnr2_bin.py WNR2.wex WEX1 OUT.bin L:E,..."""
import struct
import sys
import os  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'nr3'))  # build_nr3.py (KAIRA tools/)
import build_nr3 as B  # noqa: E402

nr2, src, out = sys.argv[1], B.Wex1(sys.argv[2]), sys.argv[3]
pairs = B.parse_pairs(sys.argv[4])
with open(nr2, 'rb') as f:
    page = f.read(B.ALIGN)
    h = B.HEADER.unpack_from(page)
    layers, experts, table_offset = h[2], h[3], h[5]
    magic, mbytes, moff, mid, _ = struct.unpack_from('<IIQII', page, 64)
    assert magic == B.WNMD and mbytes == B.MODEL_BYTES, 'no WNMD descriptor'
    f.seek(moff)
    blob = f.read(mbytes)
    with open(out + '.partial', 'wb') as o:
        o.write(struct.pack('<I', mbytes) + blob + struct.pack('<I', len(pairs)))
        for L, E in pairs:
            f.seek(table_offset + (L * experts + E) * B.ENTRY.size)
            ent = B.ENTRY.unpack(f.read(B.ENTRY.size))
            assert ent[-1] == 4, (L, E, ent[-1])
            f.seek(ent[0])
            rec = f.read(ent[1])
            _, slab = src.slab(L, E)
            o.write(B.entry_bytes(ent, 0, ent[1], 4) + struct.pack('<I', len(slab)) + slab + rec)
            print(f'L{L} E{E} total {ent[1]} model 0x{mid:08x}')
import os  # noqa: E402
os.replace(out + '.partial', out)
