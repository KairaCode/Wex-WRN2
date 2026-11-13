# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Copy the test inputs out of a WEX3/WNR3 container (opened read-only): the WNM2 model blob and a few
records, each with its 72-byte table entry. Output: OUT/model.wnm2 and OUT/L{layer}_E{expert}.rec
(72-B entry, then the record bytes).  python -I extract.py CONTAINER OUT L:E [L:E ...]"""
import struct
import sys

HEADER = struct.Struct('<IIIIQQII')
WNMD = struct.Struct('<IIQII')
ENTRY = struct.Struct('<QIII6H6H3I3II')


def main(container, out, picks):
    with open(container, 'rb') as f:
        page = f.read(4096)
        magic, version, layers, experts, aligned, table, flags, _ = HEADER.unpack_from(page, 0)
        dmagic, mbytes, moff, mid, _ = WNMD.unpack_from(page, 64)
        assert magic == 0x57455833 and dmagic == 0x444D4E57, 'not a WEX3 container with a WNMD descriptor'
        f.seek(moff)
        open(f'{out}/model.wnm2', 'wb').write(f.read(mbytes))
        print(f'layers {layers} experts {experts} aligned {aligned} flags {flags} model {mbytes} B id {mid:#x}')
        for pick in picks:
            layer, expert = (int(x) for x in pick.split(':'))
            f.seek(table + (layer * experts + expert) * ENTRY.size)
            entry = f.read(ENTRY.size)
            e = ENTRY.unpack(entry)
            assert e[-1] == 5, 'entry is not codec 5 (WNR3)'
            f.seek(e[0])
            record = f.read(e[1])
            open(f'{out}/L{layer}_E{expert}.rec', 'wb').write(entry + record)
            raw = e[2] + e[3]
            print(f'L{layer} E{expert}: record {e[1]} B, raw {raw} B, ratio {e[1] / raw:.4f}, '
                  f'scale shapes {e[4:10]}, weight shapes {e[10:16]}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
