#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Differential run: hand-made and mutated WEX4 files through the C reader, the Python reader and the Rust
reader (and the Go reader when its binary is given), each used as a black box. Standard library only.

    python -I impl/rust-wex4/differential.py [--count N] [--seed S] [--go WEX4CHECK] WEX4R OUTDIR

Needs wex4/out/wex4_tools.dll, which `bash wex4/build_and_test.sh` builds (under that name on every OS): the C
reader is `CLib.check(data, 3, True)` and the Python reader `wex4.check(data, True, C)`, as wex4/test_wex4.py
calls them.

First the hand-made files of EDGES: one or two fields of a valid vector changed, both hashes recomputed, for
the readings that SPEC-NOTES.md lists (zero-size and split tensors, the order of the overflow check, code
points above U+10FFFF, the empty index, EXPERT flags, unknown codecs, the anchor byte of a WNR3 record, files
shorter than a page). Then N mutated copies of four valid vectors and three files of impl/go-wex4/testdata,
one change each: a bit flipped with every hash left alone; a field of the header page, of a blob, record or
tensor entry or of the name order set to a boundary or random value; a random index byte; two index entries
swapped; one byte of a record or of a blob changed with its SHA-256 recomputed; or two record or tensor fields
changed. Except for the bit flips, the index and header hashes are recomputed so that the change reaches the
validators.

Prints the counts, and the hand-made files whose class differs. Exit status 1 when the Rust reader's verdict
differs from the C reader's on any file, when the C and the Python reader disagree, or when the Rust reader
panicked. A different reject class is counted, not failed: the specifications do not define the classes
(SPEC-NOTES.md, part B).
"""
import argparse, collections, hashlib, os, random, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'wex4'))
import wex4  # noqa: E402  (the Python reference reader)

GENERIC, TINY2 = 'wex4/vectors/valid/generic.wex4', 'wex4/vectors/valid/tiny2-cz-upgraded.wex4'
BASES = [GENERIC, 'wex4/vectors/valid/generic-index-last.wex4', TINY2, 'wex4/vectors/valid/tiny-raw-upgraded.wex4',
         'impl/go-wex4/testdata/ceil-form.wex4', 'impl/go-wex4/testdata/planes.wex4',
         'impl/go-wex4/testdata/ragged.wex4']
C_NAMES = {0: 'accept', -2: 'WEX4_E_MAGIC', -3: 'WEX4_E_HASH', -4: 'WEX4_E_VERSION', -5: 'WEX4_E_FEATURE',
           -6: 'WEX4_E_HEADER', -7: 'WEX4_E_INDEX', -8: 'WEX4_E_BLOB', -9: 'WEX4_E_RECORD', -10: 'WEX4_E_TENSOR',
           -11: 'WEX4_E_NAME', -12: 'WEX4_E_LAYOUT', -13: 'WEX4_E_EXPERT', -14: 'WEX4_E_DATA',
           -15: 'unsupported'}
HEADER_FIELDS = [(4, 2), (6, 2), (8, 8), (16, 8), (24, 8), (32, 8), (40, 8), (48, 4), (52, 4), (56, 4), (60, 4),
                 (64, 4), (68, 4), (72, 8), (80, 8), (88, 8)]
BLOB_FIELDS = [(0, 4), (4, 4), (8, 8), (16, 8), (56, 8)]
RECORD_FIELDS = [(0, 8), (8, 4), (12, 4), (16, 2), (18, 2), (20, 4), (24, 4), (28, 4), (32, 4), (68, 4)]
TENSOR_FIELDS = [(0, 4), (4, 2), (6, 1), (7, 1)] + [(8 + 8 * k, 8) for k in range(8)] + [(72, 4), (76, 4),
                                                                                         (80, 8), (88, 8)]
MODES = ['flip', 'header', 'blob', 'record', 'record', 'tensor', 'tensor', 'index-byte', 'order', 'swap',
         'record-data', 'record-data', 'blob-data', 'two']
# fields by name: (byte offset in the entry, width)
REC = {'stored': (8, 4), 'decoded': (12, 4), 'codec': (16, 2), 'flags': (18, 2), 'owner': (24, 4),
       'first_tensor': (28, 4), 'tensor_count': (32, 4)}
TEN = {'dtype': (6, 1), 'rank': (7, 1), 'record': (72, 4), 'record_count': (76, 4), 'offset': (80, 8),
       'nbytes': (88, 8)}
# the hand-made files: name, base file, edits. An edit is ('r' or 't', entry, field, value), ('shape', tensor,
# dims), ('name', tensor, 5 new bytes) or ('wnr3', record, byte offset in the stored record, value)
EDGES = [
    ('zero-dim-before-an-overflow', GENERIC, [('shape', 10, [0, 1 << 32, 1 << 32])]),
    ('overflow-before-a-zero-dim', GENERIC, [('shape', 10, [1 << 32, 1 << 32, 0])]),
    ('opaque-0-bytes-shape-3x3', GENERIC, [('t', 10, 'dtype', 99), ('shape', 10, [3, 3])]),
    ('opaque-0-bytes-rank-0', GENERIC, [('t', 10, 'dtype', 99), ('shape', 10, [])]),
    ('dtype-0-opaque-0-bytes', GENERIC, [('t', 11, 'dtype', 0), ('shape', 11, [5])]),
    ('f32-0-bytes-shape-3x3', GENERIC, [('shape', 10, [3, 3])]),
    ('rank-8-with-a-zero-dim', GENERIC, [('shape', 10, [1, 2, 3, 4, 5, 6, 7, 0])]),
    ('name-with-u10ffff', GENERIC, [('name', 8, b'\xf4\x8f\xbf\xbfx')]),
    ('name-above-u10ffff', GENERIC, [('name', 8, b'\xf4\x90\x80\x80x')]),
    ('split-tensor-k4-record-7-not-a-part', GENERIC, [('t', 6, 'record_count', 4)]),
    ('tensor-2-k2-record-2-not-a-part', GENERIC, [('t', 2, 'record_count', 2)]),
    ('split-tensor-5-not-alone', GENERIC, [('t', 5, 'record_count', 2)]),
    ('split-tensor-7-past-the-records', GENERIC, [('t', 7, 'record_count', 2)]),
    ('split-tensor-6-k1-orphan-parts', GENERIC, [('t', 6, 'record_count', 1)]),
    ('first-part-larger-than-the-tensor', GENERIC, [('r', 4, 'decoded', 1 << 23)]),
    ('last-part-one-byte-more', GENERIC, [('r', 6, 'decoded', 12001)]),
    ('part-of-another-tensor', GENERIC, [('r', 5, 'first_tensor', 5)]),
    ('expert-flag-on-a-continuation-part', GENERIC, [('r', 6, 'flags', 2)]),
    ('expert-flag-without-a-grid', GENERIC, [('r', 7, 'flags', 2)]),
    ('record-lists-a-zero-size-tensor', GENERIC, [('r', 7, 'tensor_count', 4)]),
    ('record-one-byte-longer-than-its-tensors', GENERIC, [('r', 7, 'decoded', 24022)]),
    ('opaque-tensor-one-byte-past-its-record', GENERIC, [('t', 9, 'nbytes', 6)]),
    ('zero-size-tensor-in-a-record', GENERIC, [('t', 10, 'record_count', 1)]),
    ('data-tensor-with-record-count-0', GENERIC, [('t', 9, 'record_count', 0)]),
    ('zero-size-tensor-at-offset-5', GENERIC, [('t', 11, 'offset', 5)]),
    ('unknown-codec-smaller-than-raw', GENERIC, [('r', 2, 'codec', 7)]),
    ('expert-record-of-an-unknown-codec', TINY2, [('r', 0, 'codec', 7), ('r', 0, 'flags', 2)]),
    ('value-exact-record-of-an-unknown-codec', TINY2, [('r', 0, 'codec', 7)]),
    ('wnr3-anchor-byte-255', TINY2, [('wnr3', 0, 24, 255)]),
    ('wnr3-anchor-byte-0', TINY2, [('wnr3', 0, 24, 0)]),
]


def layout(b):
    index_offset, index_bytes = struct.unpack_from('<QQ', b, 32)
    return (index_offset, index_bytes) + struct.unpack_from('<III', b, 48)


def fix_hashes(b):
    at, n, *_ = layout(b)
    if at + n <= len(b):
        b[96:128] = hashlib.sha256(b[at:at + n]).digest()
    b[4064:4096] = hashlib.sha256(b[:4064]).digest()


def edited(src, edits):
    """src with the edits of an EDGES row, every hash they touch recomputed."""
    b = bytearray(src)
    index_at, _, nb, nr, nt = layout(b)
    rec_at, ten_at = index_at + 64 * nb, index_at + 64 * nb + 72 * nr
    names_at = ten_at + 100 * nt
    for kind, i, *rest in edits:
        if kind in ('r', 't'):
            off, width = (REC if kind == 'r' else TEN)[rest[0]]
            at = (rec_at + 72 * i if kind == 'r' else ten_at + 96 * i) + off
            b[at:at + width] = rest[1].to_bytes(width, 'little')
        elif kind == 'shape':
            dims = rest[0]
            b[ten_at + 96 * i + 7] = len(dims)
            struct.pack_into('<8Q', b, ten_at + 96 * i + 8, *(dims + [0] * (8 - len(dims))))
        elif kind == 'name':
            at = names_at + struct.unpack_from('<I', b, ten_at + 96 * i)[0]
            b[at:at + len(rest[0])] = rest[0]
        elif kind == 'wnr3':
            off, stored = struct.unpack_from('<QI', b, rec_at + 72 * i)
            b[off + rest[0]] = rest[1]
            b[rec_at + 72 * i + 36:rec_at + 72 * i + 68] = hashlib.sha256(b[off:off + stored]).digest()
    fix_hashes(b)
    return bytes(b)


def header_only(index_offset, file_bytes):
    """A file of no blob, record or tensor: its header page, then zero pages up to file_bytes."""
    b = bytearray(file_bytes)
    struct.pack_into('<IHHQQQQQ', b, 0, 0x57455834, 4, 0, 0, 0, file_bytes, index_offset, 0)
    b[96:128] = hashlib.sha256(b'').digest()
    b[4064:4096] = hashlib.sha256(b[:4064]).digest()
    return bytes(b)


def edge_files(sources):
    out = [(name, edited(sources[base], edits)) for name, base, edits in EDGES]
    out += [('header-page-only', header_only(4096, 4096)), ('empty-index-at-8192', header_only(8192, 8192)),
            ('empty-index-and-a-zero-page', header_only(4096, 8192)),
            ('magic-only-4-bytes', sources[GENERIC][:4]), ('first-100-bytes', sources[GENERIC][:100])]
    return out


class Mutator:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def value(self, old, width):
        r, top = self.rng, (1 << (8 * width)) - 1
        k = r.randrange(8 * width)
        return r.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 255, 4095, 4096, 4097, top, top - 1, 1 << k, (1 << k) - 1,
                         old + 1, old - 1, old + 4096, old - 4096, old ^ (1 << k), r.randrange(top + 1)]) & top

    def set_field(self, b, at, width):
        old = int.from_bytes(b[at:at + width], 'little')
        b[at:at + width] = self.value(old, width).to_bytes(width, 'little')

    def mutate(self, src):
        r, b = self.rng, bytearray(src)
        index_at, index_bytes, nb, nr, nt = layout(b)
        rec_at = index_at + 64 * nb
        ten_at = rec_at + 72 * nr
        ord_at = ten_at + 96 * nt
        mode = r.choice(MODES)
        if mode in ('blob', 'blob-data') and nb == 0:
            mode = 'tensor'
        if mode == 'flip':
            b[r.randrange(len(b))] ^= 1 << r.randrange(8)
            return mode, b
        if mode == 'header':
            if r.random() < 0.1:
                b[128 + r.randrange(3936)] = r.randrange(256)
            else:
                self.set_field(b, *r.choice(HEADER_FIELDS))
        elif mode == 'blob':
            off, w = r.choice(BLOB_FIELDS)
            self.set_field(b, index_at + 64 * r.randrange(nb) + off, w)
        elif mode == 'record':
            off, w = r.choice(RECORD_FIELDS)
            self.set_field(b, rec_at + 72 * r.randrange(nr) + off, w)
        elif mode == 'tensor':
            off, w = r.choice(TENSOR_FIELDS)
            self.set_field(b, ten_at + 96 * r.randrange(nt) + off, w)
        elif mode == 'two':
            for _ in range(2):
                off, w = r.choice(RECORD_FIELDS)
                self.set_field(b, rec_at + 72 * r.randrange(nr) + off, w)
                if r.random() < 0.5:
                    off, w = r.choice(TENSOR_FIELDS)
                    self.set_field(b, ten_at + 96 * r.randrange(nt) + off, w)
        elif mode == 'index-byte':
            b[index_at + r.randrange(index_bytes)] = r.randrange(256)
        elif mode == 'order':
            self.set_field(b, ord_at + 4 * r.randrange(nt), 4)
        elif mode == 'swap':
            base, size, n = r.choice([(rec_at, 72, nr), (ten_at, 96, nt), (ord_at, 4, nt)])
            if n > 1:
                i, j = r.sample(range(n), 2)
                a, c = base + size * i, base + size * j
                b[a:a + size], b[c:c + size] = b[c:c + size], b[a:a + size]
        elif mode == 'record-data':
            i = r.randrange(nr)
            off, stored = struct.unpack_from('<QI', b, rec_at + 72 * i)
            k = off + r.randrange(min(stored, 512) if r.random() < 0.5 else stored)
            b[k] = b[k] ^ (1 << r.randrange(8)) if r.random() < 0.7 else r.randrange(256)
            b[rec_at + 72 * i + 36:rec_at + 72 * i + 68] = hashlib.sha256(b[off:off + stored]).digest()
        elif mode == 'blob-data':
            i = r.randrange(nb)
            off, n = struct.unpack_from('<QQ', b, index_at + 64 * i + 8)
            b[off + r.randrange(n + (-n) % 4096)] ^= 1 << r.randrange(8)
            b[index_at + 64 * i + 24:index_at + 64 * i + 56] = hashlib.sha256(b[off:off + n]).digest()
        fix_hashes(b)
        return mode, b


def run_reader(argv, files, parse):
    """Verdicts of a reader binary that prints one line per file, 200 files per call."""
    out = {}
    for i in range(0, len(files), 200):
        done = subprocess.run(argv + files[i:i + 200], capture_output=True, text=True)
        for line in done.stdout.splitlines():
            got = parse(line)
            if got:
                out[got[0]] = got[1:]
    return out


def rust_line(line):
    w = line.split(' ', 3)
    if len(w) < 3:
        return None
    if w[0] == 'FAULT':
        return w[1].rstrip(':'), 'FAULT', line
    return {'accept': (w[1], 'accept', ''), 'reject': (w[1], w[2].rstrip(':'), w[3] if len(w) > 3 else ''),
            'undecodable': (w[1], 'unsupported', '')}.get(w[0])


def go_line(line):
    if ': ' not in line:
        return None
    path, rest = line.split(': ', 1)
    if rest.startswith('reject '):
        return path, rest.split()[1].rstrip(':'), rest
    return path, 'accept' if rest.endswith('accept') else 'unsupported', rest


def reference(c, data):
    """(C verdict or class, Python verdict) of one file."""
    cv = C_NAMES.get(c.check(data, 3, True)[0], 'other')
    try:
        wex4.check(data, True, c)
        return cv, 'accept'
    except wex4.FormatError:
        return cv, 'reject'


def verdict(v):
    return v if v in ('accept', 'unsupported') else 'reject'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('wex4r')
    ap.add_argument('outdir')
    ap.add_argument('--count', type=int, default=7000)
    ap.add_argument('--seed', type=int, default=20261115)
    ap.add_argument('--go', help='the Go reader binary (impl/go-wex4/out/wex4check), compared too')
    ap.add_argument('--list-edges', action='store_true', help='print every hand-made file, not only those whose '
                    'Rust class is not the C class')
    a = ap.parse_args()
    lib = os.path.join(REPO, 'wex4', 'out', 'wex4_tools.dll')
    if not os.path.exists(lib):
        sys.exit('wex4/out/wex4_tools.dll is missing: run bash wex4/build_and_test.sh first')
    c = wex4.CLib(lib)
    os.makedirs(a.outdir, exist_ok=True)
    sources = {p: open(os.path.join(REPO, p), 'rb').read() for p in BASES}
    files, meta = [], {}
    for name, data in edge_files(sources):
        path = os.path.join(a.outdir, 'edge-' + name + '.wex4')
        with open(path, 'wb') as f:
            f.write(data)
        files.append(path)
        meta[path] = ('edge',) + reference(c, data)
    m = Mutator(a.seed)
    for n in range(a.count):
        base = BASES[n % len(BASES)]
        mode, data = m.mutate(sources[base])
        if bytes(data) == sources[base]:
            continue
        path = os.path.join(a.outdir, '%06d.wex4' % n)
        with open(path, 'wb') as f:
            f.write(data)
        files.append(path)
        meta[path] = (mode,) + reference(c, bytes(data))
    rust = run_reader([os.path.abspath(a.wex4r), 'check'], files, rust_line)
    go = run_reader([os.path.abspath(a.go)], files, go_line) if a.go else {}
    s, classes, edges = collections.Counter(), collections.Counter(), []
    for path in files:
        mode, cv, pv = meta[path]
        rv, rwhy = rust.get(path, ('missing', ''))
        gv, gwhy = go.get(path, ('missing', ''))
        kind = 'hand-made' if mode == 'edge' else 'mutated'
        s[kind + ' files'] += 1
        s[kind + ', C ' + verdict(cv)] += 1
        if cv != 'unsupported' and (pv == 'accept') != (cv == 'accept'):
            s['C and Python disagree'] += 1
        s[kind + ', Rust verdict ' + ('as C' if verdict(rv) == verdict(cv) else 'NOT as C')] += 1
        if rv == 'FAULT':
            s['Rust panicked'] += 1
        if a.go:
            s[kind + ', Go verdict ' + ('as C' if verdict(gv) == verdict(cv) else 'NOT as C')] += 1
        if verdict(cv) == 'reject' and verdict(rv) == 'reject':
            s[kind + ', Rust class ' + ('as C' if rv == cv else 'not as C')] += 1
            if rv != cv:
                classes[(mode, cv, rv)] += 1
        if mode == 'edge' and (a.list_edges or rv != cv):
            edges.append(f'{os.path.basename(path)}: C {cv}, Python {pv}, Rust {rv} ({rwhy})'
                         + (f', Go {gv} ({gwhy})' if a.go else ''))
    for k in sorted(s):
        print(f'{k}: {s[k]}')
    for (mode, cv, rv), n in classes.most_common():
        print(f'class: {n} {mode} files, C {cv}, Rust {rv}')
    for e in edges:
        print('hand-made: ' + e)
    bad = s['hand-made, Rust verdict NOT as C'] + s['mutated, Rust verdict NOT as C'] + s['C and Python disagree'] \
        + s['Rust panicked']
    print('differential: ' + ('verdicts as the C reader on every file' if not bad else 'DISAGREEMENT'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
