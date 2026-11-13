# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Field-level differential sweep: every field of the header page and of sampled index rows is set to
boundary values, hashes resealed, and the file is run through the C reader (wex4_check, flags 3) and the
Python reader (wex4.check with the C codec). Reports verdict/digest disagreements, non-FormatError
exceptions, and which changed fields are accepted by both.  python -I sweep.py FILE..."""
import collections, hashlib, io, os, struct, sys
HERE = os.path.dirname(os.path.abspath(__file__))
NEXT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, NEXT)
import wex4
C = wex4.CLib(os.path.join(NEXT, 'out', 'wex4_tools_ref.dll'))

HEADER = [(0, 4, 'magic'), (4, 2, 'major'), (6, 2, 'minor'), (8, 8, 'required'), (16, 8, 'optional'), (24, 8, 'file_bytes'),
          (32, 8, 'index_offset'), (40, 8, 'index_bytes'), (48, 4, 'blob_count'), (52, 4, 'record_count'),
          (56, 4, 'tensor_count'), (60, 4, 'num_layers'), (64, 4, 'num_experts'), (68, 4, 'max_decoded'),
          (72, 8, 'names_bytes'), (80, 8, 'meta_bytes'), (88, 8, 'total_decoded'), (128, 8, 'reserved@128'),
          (4056, 8, 'reserved@4056')]
BLOB = [(0, 4, 'kind'), (4, 4, 'id'), (8, 8, 'offset'), (16, 8, 'bytes'), (56, 8, 'reserved')]
RECORD = [(0, 8, 'offset'), (8, 4, 'stored'), (12, 4, 'decoded'), (16, 2, 'codec'), (18, 2, 'flags'), (20, 4, 'blob'),
          (24, 4, 'owner'), (28, 4, 'first_tensor'), (32, 4, 'tensor_count'), (68, 4, 'reserved')]
TENSOR = [(0, 4, 'name_offset'), (4, 2, 'name_bytes'), (6, 1, 'dtype'), (7, 1, 'rank')] + \
         [(8 + 8 * d, 8, f'shape{d}') for d in range(8)] + \
         [(72, 4, 'record'), (76, 4, 'record_count'), (80, 8, 'offset'), (88, 8, 'nbytes')]


def reseal(b):
    io_, ib = struct.unpack_from('<QQ', b, 32)
    if io_ + ib <= len(b):
        b[96:128] = hashlib.sha256(b[io_:io_ + ib]).digest()
    b[4064:4096] = hashlib.sha256(b[:4064]).digest()


def values(old, size):
    top = (1 << (8 * size)) - 1
    v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 22, 23, 24, 255, 256, 4095, 4096, 4097, 8192, 59408, 65535, 65536, 65537,
         1 << 24, (1 << 24) + 1, 1 << 28, (1 << 28) + 1, 1 << 30, (1 << 30) + 1, 1 << 31, (1 << 32) - 1, 1 << 32, 1 << 62,
         1 << 63, top, top - 1, top >> 1}
    for d in (1, 2, 4, 6, 8, 16, 64, 72, 96, 100, 4096, 8192, 65536):
        v |= {old + d, old - d, old * 2, old // 2}
    for bit in range(8 * size):
        v.add(old ^ (1 << bit))
    return sorted(x for x in v if 0 <= x <= top and x != old)


def py(data):
    try:
        return 0, wex4.check(data, True, C), ''
    except wex4.FormatError as e:
        return -1, None, str(e)
    except Exception as e:   # anything else is a reader bug
        return -2, None, f'{type(e).__name__}: {e}'


def sample(n, extra=()):
    s = set(range(min(n, 5))) | set(range(max(0, n - 3), n)) | {x for x in extra if 0 <= x < n}
    return sorted(s)


def sweep(path):
    base = open(path, 'rb').read()
    rd = wex4.Reader(io.BytesIO(base), C)
    base_digest = rd.content_digest(True)
    assert C.check(base, 3)[0] == 0
    nb, nr, nt = len(rd.blobs), len(rd.records), len(rd.tensors)
    b0 = rd.index_offset
    r0 = b0 + nb * 64
    t0 = r0 + nr * 72
    o0 = t0 + nt * 96
    targets = [(off, size, 'header.' + name) for off, size, name in HEADER]
    for i in sample(nb):
        targets += [(b0 + i * 64 + off, size, f'blob[{i}].{name}') for off, size, name in BLOB]
    # records: ends, one of each (codec, flags, continuation) class
    classes = {}
    for i, rec in enumerate(rd.records):
        classes.setdefault((rec[3], rec[4], rec[8] == 0), i)
        if rec[8] == 0:
            classes.setdefault(('before-cont', i - 1), i - 1)
    for i in sample(nr, classes.values()):
        rec = rd.records[i]
        tag = f'record[{i}:codec{rec[3]},flags{rec[4]}{",cont" if rec[8] == 0 else ""}]'
        targets += [(r0 + i * 72 + off, size, f'{tag}.{name}') for off, size, name in RECORD]
    tens = set()
    for i in classes.values():
        first, count = rd.records[i][7], rd.records[i][8]
        tens |= {first, first + count - 1, first + 1}
    for i in sample(nt, tens | {rd.data_tensors - 1, rd.data_tensors}):
        t = rd.tensors[i]
        tag = f'tensor[{i}:dtype{t[1]},rc{t[5]}]'
        targets += [(t0 + i * 96 + off, size, f'{tag}.{name}') for off, size, name in TENSOR]
    for i in sample(nt):
        targets.append((o0 + i * 4, 4, f'order[{i}]'))
    stats = collections.Counter()
    accepted = collections.defaultdict(list)
    for off, size, name in targets:
        old = int.from_bytes(base[off:off + size], 'little')
        for value in values(old, size):
            data = bytearray(base)
            data[off:off + size] = value.to_bytes(size, 'little')
            reseal(data)
            data = bytes(data)
            cs, cd = C.check(data, 3, True)
            ps, pd, why = py(data)
            stats['mutations'] += 1
            if ps == -2:
                stats['python exception'] += 1
                print(f'  PYTHON EXCEPTION {name} {old} -> {value}: {why} (C status {cs})')
            elif (cs == 0) != (ps == 0):
                stats['VERDICT DIFFERS'] += 1
                print(f'  VERDICT DIFFERS {name} {old} -> {value}: C {cs}, Python {ps} {why}')
            elif cs == 0 and cd != pd:
                stats['DIGEST DIFFERS'] += 1
                print(f'  DIGEST DIFFERS {name} {old} -> {value}')
            elif cs == 0:
                stats['accepted'] += 1
                accepted[name.split('.')[0].split('[')[0] + '.' + name.split('.')[-1] + ('' if cd == base_digest else ' (digest changed)')].append((old, value))
            else:
                stats['rejected'] += 1
    print(os.path.basename(path), dict(stats), f'targets {len(targets)}')
    for name, pairs in sorted(accepted.items()):
        print(f'  accepted by both: {name}: {len(pairs)} values, e.g. {pairs[:4]}')


for p in sys.argv[1:]:
    sweep(p)
