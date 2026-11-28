# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Re-stamp a pre-bit WEX4 container to the strict form (docs/WEX4-SPEC.md 7.3.1). Standard library only.

    python -I wex4_restamp.py FILE [--out COPY] [--dry-run]

A pre-bit file was written before required_features bit 2 (REQ_WNR4_EXPERT = 0x4) existed: it holds EXPERT records
of codec 8 (WNR4 kind MX) under a required_features word of 0, no record of it has REC_VALUE_EXACT,
and whether a record is in canonical-zero form is only in the record's own flag byte. Strict readers refuse it.

This rewrites the header page and, when a record is in canonical-zero form, the index: it sets bit 2, sets
REC_VALUE_EXACT on every such record and bit 0 (REQ_VALUE_EXACT) with it, and recomputes the index hash and the
root. No blob or record byte is written or moved; the root changes, so a pinned root has to follow.
Before anything is written the new header page, the new index and every blob go through the strict reader of
wex4.py: a file that is not exactly that shape is REFUSED and left as it is. The records are not hashed here:
run `wex4.py verify` or wex4_verify on the result. A file that is already strict is left alone (exit 0).

In place (default): the index is written first, when it changes, then one 4096-byte write of the header page
switches the file. If that is interrupted the file opens nowhere until this is run again, which finishes it.
Every mirror of a container is re-stamped on its own: the result is the same bytes. --out COPY writes a
re-stamped copy instead and leaves FILE as it is; --dry-run writes nothing and prints what would change.
Exit status: 0 re-stamped or nothing to do, 1 REFUSED, 2 usage.
"""
import os
import shutil
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wex4  # noqa: E402
from wex4 import need, align, sha256, PAGE  # noqa: E402
from wex4_convert import _sync  # noqa: E402

WNR4_MAGIC, KIND_MX, FLAG_CZ, FLAGS_KNOWN = 0x34524e57, 0, 0x08, 0x0f   # the first 16 bytes of a WNR4 record
FLAGS_AT = 18   # the u16 flags of a 72-byte record entry


class _Stamped:
    """The file as it would be: the new header page and index over the bytes on disk (what wex4.Reader reads)."""

    def __init__(self, f, size, parts):
        self.f, self.size, self.parts, self.at = f, size, parts, 0

    def seek(self, offset, whence=0):
        self.at = self.size if whence else offset

    def tell(self):
        return self.at

    def read(self, n):
        if self.at in self.parts:
            return self.parts[self.at][:n]
        self.f.seek(self.at)
        return self.f.read(n)


def plan(f):
    """What a re-stamp of the open file changes: None when it is already strict, else a dict with the new header
    page, the new index (padded) and its offset, the two roots and the counts. Raises FormatError on anything else."""
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)
    page = f.read(PAGE)
    need(len(page) == PAGE, 'file shorter than a header page')
    head = list(wex4.HEADER.unpack_from(page))
    magic, major, required, index_offset, index_bytes, nb, nr = head[0], head[1], head[3], head[6], head[7], head[8], head[9]
    need(magic == wex4.MAGIC, 'not a WEX4 file')
    need(sha256(page[:wex4.HEADER_HASHED]) == page[wex4.HEADER_HASHED:], 'header hash mismatch')
    need(major == 4 and not required & ~wex4.REQUIRED_KNOWN, 'unsupported version or unknown required feature')
    if required & wex4.REQ_WNR4_EXPERT:
        wex4.Reader(f)
        return None
    need(index_bytes <= wex4.MAX_INDEX and index_offset % PAGE == 0 and PAGE <= index_offset <= size and
         align(index_bytes) <= size - index_offset and nb * wex4.BLOB.size + nr * wex4.RECORD.size <= index_bytes,
         'index location or size')
    f.seek(index_offset)
    index = bytearray(f.read(align(index_bytes)))
    rows = [nb * wex4.BLOB.size + r * wex4.RECORD.size for r in range(nr)]
    coded = [(at, wex4.RECORD.unpack_from(index, at)) for at in rows]
    coded = [(at, rec) for at, rec in coded if rec[3] == wex4.CODEC_WNR4 and rec[4] & wex4.REC_EXPERT]
    if sha256(index[:index_bytes]) != head[17]:
        # an interrupted re-stamp: the index is already the new one under the old header page
        for at, _ in coded:
            index[at + FLAGS_AT] &= ~wex4.REC_VALUE_EXACT
        need(sha256(index[:index_bytes]) == head[17], 'index hash mismatch')
        coded = [(at, wex4.RECORD.unpack_from(index, at)) for at, _ in coded]
    if not coded:   # nothing of codec 8 on an expert: strict as it is, or not a file this tool knows
        wex4.Reader(f)
        return None
    need(not required, 'required_features of a pre-bit file is 0 (bit 0 set without bit 2)')
    exact = 0
    for at, (offset, _, _, _, flags, blob, owner, *_) in coded:
        need(flags == wex4.REC_EXPERT, 'an EXPERT record of codec 8 whose flags are not EXPERT alone')
        need(0 < blob <= nb and PAGE <= offset <= size - PAGE,
             'an EXPERT record of codec 8 without a model blob or outside the file')
        f.seek(offset)
        rmagic, model, rowner, kind, rflags = struct.unpack('<IIIBB', f.read(14))
        need(rmagic == WNR4_MAGIC and kind == KIND_MX and not rflags & ~FLAGS_KNOWN and rowner == owner and
             model == wex4.BLOB.unpack_from(index, (blob - 1) * wex4.BLOB.size)[1],
             f'record at byte {offset} is not a WNR4 MX record of its owner and model')
        if rflags & FLAG_CZ:
            index[at + FLAGS_AT] |= wex4.REC_VALUE_EXACT
            exact += 1
    head[3] = required | wex4.REQ_WNR4_EXPERT | (wex4.REQ_VALUE_EXACT if exact else 0)
    head[17] = sha256(index[:index_bytes])
    new = wex4.HEADER.pack(*head) + page[wex4.HEADER.size:wex4.HEADER_HASHED]
    new += sha256(new)
    rd = wex4.Reader(_Stamped(f, size, {0: new, index_offset: bytes(index)}))   # refuse before writing
    need(all(rd.expert_geometry(r)['kind'] == 0 for r, rec in enumerate(rd.records) if rec[4] & wex4.REC_EXPERT),
         'NVFP4 expert geometry: never written without the feature bit')
    for i in range(len(rd.blobs)):
        rd.blob(i)
    return dict(page=new, index=bytes(index), index_offset=index_offset, old_root=page[wex4.HEADER_HASHED:],
                root=rd.root, required=(required, head[3]), coded=len(coded), exact=exact)


def restamp(path, out=None, dry_run=False, log=None):
    """Re-stamp path in place, or into the copy `out`. Returns the new root, or None when nothing had to change."""
    log = log or sys.stdout   # looked up per call, so that a redirected stdout is honoured
    need(not out or os.path.abspath(out) != os.path.abspath(path), '--out names the file itself')
    with open(path, 'rb' if dry_run or out else 'r+b') as f:
        p = plan(f)
        if p is None:
            print(f'{path}: already strict, nothing to do', file=log)
            return None
        print(f'{path}: pre-bit file, {p["coded"]} EXPERT records of codec 8, {p["exact"]} in canonical-zero form; '
              f'required_features {p["required"][0]:#x} -> {p["required"][1]:#x}\n'
              f'root {p["old_root"].hex()} -> {p["root"].hex()}', file=log)
        if dry_run:
            print('dry run: nothing written', file=log)
            return p['root']
        if out:
            tmp = os.fspath(out) + '.partial'
            try:
                with open(tmp, 'w+b') as g:
                    f.seek(0)
                    shutil.copyfileobj(f, g, 8 << 20)
                    g.seek(p['index_offset'])
                    g.write(p['index'])
                    g.seek(0)
                    g.write(p['page'])
                    _sync(g)
                    need(wex4.Reader(g).root == p['root'], 'the copy does not open with the new root')
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
            os.replace(tmp, out)
            print(f'written to {out}', file=log)
            return p['root']
        if p['exact']:              # 1. the index, when a record got VALUE_EXACT (else it is the one on disk)
            f.seek(p['index_offset'])
            f.write(p['index'])
            _sync(f)
        f.seek(0)                   # 2. one 4096-byte write switches the file to the strict form
        f.write(p['page'])
        _sync(f)
        need(wex4.Reader(f).root == p['root'], 'the file does not open with the new root')
    print('re-stamped in place', file=log)
    return p['root']


def main(argv):
    files, out, usage, args = [], None, False, argv[1:]
    while args:
        a = args.pop(0)
        if a == '--out' and args:
            out = args.pop(0)
        elif a.startswith('--') and a != '--dry-run':
            usage = True
        elif a != '--dry-run':
            files.append(a)
    if usage or len(files) != 1:
        print(__doc__)
        return 2
    try:
        restamp(files[0], out, '--dry-run' in argv)
    except (OSError, ValueError) as error:
        print('REFUSED:', error)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
