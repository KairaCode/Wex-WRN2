# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""A1: every header / blob / record / tensor / name-order field of two base files set to boundary values and to the
values of its siblings, hashes recomputed; then random 2-3 field combinations. Fast DLL and reference DLL run on
guard pages (wex4_fuzz_check); the Python reader gives the third verdict. Any crash, verdict or digest
disagreement is saved under findings/.   python -I -B a1_fields.py [pairs]"""
import io
import json
import random
import struct
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rvlib import *  # noqa: E402,F401,F403

rng = random.Random(4)
PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000


def base_mixed(info):
    """Blob + WNR3-CZ expert, raw expert, WNR3 expert (last extent, decoded == max_decoded), WBP1, split tensor,
    opaque dtype, zero-size tensors, index in the middle of the file."""
    src = open(os.path.join(NOUT, 'tiny_nr3_cz.wex4'), 'rb').read()
    rd = wex4.Reader(io.BytesIO(src), C)
    model = rd.blob(0)
    ex = [r for r, rec in enumerate(rd.records) if rec[4] & 2][:3]
    a = bytes(rng.choices(range(256), [int(1000 * 0.5 ** abs(i - 120)) + 1 for i in range(256)], k=12000))
    recs, tens = [], []
    recs.append(R(wex4.bp_encode(a[:12000], 2, 10, 0, 1), 12000, 6, 0, 0, 2))
    tens += [T('w.bf16', 10, [4, 1000], 0, 0, 8000), T("w.u8", 2, [4000], 0, 8000, 4000)]
    recs.append(raw_rec(a[:1004], 1, 2, 2))
    tens += [T('opaque', 77, [3, 3], 1, 0, 1000), T('scalar', 13, [], 1, 1000, 4)]
    recs.append(raw_rec(a[:5000], 2, 4, 1))
    tens += [T('big', 2, [14000], 2, 0, 14000, rcount=2)]
    recs.append(R(wex4.bp_encode(a[:9000], 1, 12, 3, 0), 9000, 6, 3, 4, 0))
    for k, r in enumerate(ex):
        rec = rd.records[r]
        first = len(tens)
        for t in rd.tensors[rec[7]:rec[7] + 6]:
            tens.append(T(t[0], t[1], list(t[3]), len(recs), t[6], t[7]))
        if k == 1:
            recs.append(R(pad(rd.record(r)), rec[2], 0, rec[6], first, 6, flags=2))
        else:
            recs.append(R(rd.stored(r), rec[2], 5, rec[6], first, 6, flags=3, blob=1))
    tens += [T('zz.empty', 13, [0, 7], 0, 0, 0, rcount=0), T('', 2, [0], 0, 0, 0, rcount=0)]
    layout = [('R', 0), ('B', 0), ('R', 1), 'I', ('R', 2), ('R', 3), ('R', 4), ('R', 5), ('R', 6)]
    return build([dict(kind=1, id=rd.blobs[0][1], data=model)], recs, tens, meta=b'meta', layers=6, experts=8, required=1,
                 layout=layout, info=info)


def base_generic(info):
    data = open(os.path.join(NOUT, 't_py.wex4'), 'rb').read()
    rd = wex4.Reader(io.BytesIO(data))
    io_, nb, nr, nt = rd.index_offset, len(rd.blobs), len(rd.records), len(rd.tensors)
    info.update(index_offset=io_, blob0=io_, rec0=io_ + 64 * nb, ten0=io_ + 64 * nb + 72 * nr,
                order0=io_ + 64 * nb + 72 * nr + 96 * nt, nb=nb, nr=nr, nt=nt, names0=io_ + 64 * nb + 72 * nr + 100 * nt,
                index_bytes=rd.index_bytes)
    return data


def fields(info):
    """(absolute offset, width, column key)"""
    out = [(o, w, f'h{o}') for o, w in ((4, 2), (6, 2), (8, 8), (16, 8), (24, 8), (32, 8), (40, 8), (48, 4), (52, 4),
                                        (56, 4), (60, 4), (64, 4), (68, 4), (72, 8), (80, 8), (88, 8))]
    for i in range(info['nb']):
        out += [(info['blob0'] + 64 * i + o, w, f'b{o}') for o, w in ((0, 4), (4, 4), (8, 8), (16, 8), (24, 1), (56, 8))]
    for i in range(info['nr']):
        out += [(info['rec0'] + 72 * i + o, w, f'r{o}') for o, w in ((0, 8), (8, 4), (12, 4), (16, 2), (18, 2), (20, 4),
                                                                    (24, 4), (28, 4), (32, 4), (36, 1), (68, 4))]
    for i in range(info['nt']):
        out += [(info['ten0'] + 96 * i + o, w, f't{o}') for o, w in
                [(0, 4), (4, 2), (6, 1), (7, 1)] + [(8 + 8 * d, 8) for d in range(8)] + [(72, 4), (76, 4), (80, 8), (88, 8)]]
        out.append((info['order0'] + 4 * i, 4, 'o'))
    return out


FMT = {1: '<B', 2: '<H', 4: '<I', 8: '<Q'}


def candidates(old, width, siblings):
    m = (1 << 8 * width) - 1
    v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 22, 23, 99, old + 1, old - 1, old + 4096, old - 4096, old * 2, old // 2, old * 8,
         old // 8, old ^ (1 << (8 * width - 1)), m, m - 1, 1 << (8 * width - 1), 4095, 4096, 4097, 8192, 65535, 65536,
         1 << 24, (1 << 24) + 1, 1 << 28, 1 << 30, (1 << 30) + 1, 1 << 31, (1 << 32) - 1, 1 << 32, 1 << 62, 59408}
    return sorted({x & m for x in v | set(siblings)} - {old})


def run(name, make):
    info = {}
    base = make(info)
    log = Log(name)
    st = log.tri('base', base, expect=True)
    assert st == (0, 0, 0), st
    fl = fields(info)
    cols = {}
    for at, w, key in fl:
        cols.setdefault(key, set()).add(struct.unpack_from(FMT[w], base, at)[0])
    cand = {at: candidates(struct.unpack_from(FMT[w], base, at)[0], w, cols[key]) for at, w, key in fl}
    for at, w, key in fl:
        for v in cand[at]:
            blob = bytearray(base)
            struct.pack_into(FMT[w], blob, at, v)
            log.tri(f'{key}@{at}={v:#x}', reseal(blob))
    single = log.n
    for i in range(PAIRS):
        blob = bytearray(base)
        label = []
        for at, w, key in rng.sample(fl, rng.choice([2, 2, 3])):
            v = rng.choice(cand[at])
            struct.pack_into(FMT[w], blob, at, v)
            label.append(f'{key}@{at}={v:#x}')
        log.tri(' '.join(label), reseal(blob))
    # a byte of the name pool / metadata / every tensor name start
    for at in range(info['names0'], info['index_offset'] + info['index_bytes']):
        for v in (0, 0x80, 0xc0, 0xff, base[at] ^ 1, base[at] ^ 0x20):
            blob = bytearray(base)
            blob[at] = v
            log.tri(f'names/meta@{at}={v:#x}', reseal(blob))
    res = log.done()
    res['single_field_files'] = single - 1
    return res


if __name__ == '__main__':
    results = [run('a1_mixed', base_mixed), run('a1_generic', base_generic)]
    json.dump(results, open(os.path.join(OUT, 'a1_fields.json'), 'w'), indent=1)
