# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""WNR4-I4 records (wnr4/wnr4_i4.h) in a WEX4 container of int4-g64 experts. Standard library + ctypes; the codec is
the DLL of wnr4/tools/wnr4_i4_tools.c (--dll).

    python -I wnr4i4.py train RAW.wex4 MODEL.wnm4 --dll D [--every K]    # model from every K-th int4 record
    python -I wnr4i4.py pack RAW.wex4 OUT.wex4 --model MODEL.wnm4 --dll D [--G 32] [--pairs]
    python -I wnr4i4.py verify OUT.wex4 [--source RAW.wex4] --dll D     # every record decoded (and compared)

RAW.wex4 is any WEX4 container whose records are all RAW. An int4-g64 record is one whose tensors are 1 to 3 U8
matrices [rows, cols / 2] (the int4 codes, two per byte) followed by as many F32 matrices [rows, cols / 64] (one
scale per group of 64 codes along the row), with the same rows and cols in the same order, back to back from offset
0, each a whole tensor (WNR4-SPEC section 6.1). --pairs: the codes of a group are packed "pairs" (byte j = codes 2j
and 2j + 1), else "planar" (byte j = codes j and j + 32); the packing changes only the size of the coded record,
never its bytes, and the record carries it (header byte 13), so verify needs no option. --G is the tile depth,
1..4096 (the CUDA decoder of wnr4/wnr4_i4_cuda.cuh takes G <= 128). Draft, experimental:
  - codec 8 = a WNR4 record of any kind (the record header's kind byte tells them apart; this tool writes and reads
    kind 3, I4); blob kind 3 = a WNM4 model of any kind, id = the model id. A codec-8 record names its model in its
    blob field. Owner = the record's owner field (the records are not EXPERT-flagged), in the index and in the
    record header.
  - pack codes an int4-g64 record when the coded record is smaller than raw, and decodes it back before writing it;
    every other record is written RAW. Every blob of the source is copied (same indices), the WNM4 blob comes last;
    the tensor table, the metadata and the header words are those of the source.
Readers that do not know codec 8 open such a file and refuse to decode those records (WEX4-SPEC 5.2, 10 rule 6).
pack and verify stream: a few records in memory at a time.
"""
import contextlib
import ctypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wex4  # noqa: E402
from wex4 import need  # noqa: E402
from wnr4gg import Progress  # noqa: E402

CODEC_WNR4 = 8
BLOB_WNM4 = 3
KIND_I4 = 3
U8, F32 = wex4.DTYPE_CODE['U8'], wex4.DTYPE_CODE['F32']


class Lib:
    """wnr4/tools/wnr4_i4_tools.c through ctypes."""

    def __init__(self, path):
        lib = ctypes.CDLL(os.path.abspath(path))
        need(hasattr(lib, 'wnr4i4_abi'), f'{path}: not a wnr4_i4_tools library')
        P, B, U32, U64 = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint64
        lib.wnr4i4_abi.restype = U32
        need(lib.wnr4i4_abi() == 1, 'wnr4_i4_tools ABI')
        lib.wnr4i4_counts_words.restype = U64
        lib.wnr4i4_model_bytes.restype = U32
        lib.wnr4i4_raw_bytes.restype = U64
        lib.wnr4i4_raw_bytes.argtypes = [U32, P, P, U32]
        lib.wnr4i4_count.restype = ctypes.c_int
        lib.wnr4i4_count.argtypes = [U32, P, P, U32, B, P]
        lib.wnr4i4_build.argtypes = [P, B]
        lib.wnr4i4_model_open.restype = P
        lib.wnr4i4_model_open.argtypes = [B, U64]
        lib.wnr4i4_model_id.restype = U32
        lib.wnr4i4_model_id.argtypes = [P]
        lib.wnr4i4_encode.restype = U64
        lib.wnr4i4_encode.argtypes = [P, U32, P, P, U32, B, U32, U32, B, U64]
        lib.wnr4i4_decode.restype = U32
        lib.wnr4i4_decode.argtypes = [P, U32, P, P, U32, B, U64, U32, B, ctypes.c_int]
        self.lib, self.models = lib, {}

    def model(self, blob):
        """(handle, id) of a WNM4 blob; opened once per process."""
        if blob not in self.models:
            m = self.lib.wnr4i4_model_open(blob, len(blob))
            need(m, 'not a WNM4 kind-3 (I4) model blob')
            self.models[blob] = m, self.lib.wnr4i4_model_id(m)
        return self.models[blob]

    def decode(self, model, geo, pairs, rec, decoded, owner, threads=4):
        n, rows, cols = geo
        # the DLL writes the geometry's bytes: they must be the record's
        need(self.lib.wnr4i4_raw_bytes(n, rows, cols, pairs) == decoded, 'geometry does not tile the record')
        out = ctypes.create_string_buffer(decoded)
        err = self.lib.wnr4i4_decode(model, n, rows, cols, pairs, rec, len(rec), owner, out, threads)
        return err, out.raw


def geometry(rd, r):
    """(nmat, rows, cols) of record r as the DLL takes them, or None when it is not an int4-g64 record."""
    first, count = rd.records[r][7:9]
    ts = rd.tensors[first:first + count]
    if count not in (2, 4, 6) or any(t[5] != 1 or t[2] != 2 for t in ts):
        return None
    n = count // 2
    rows = [t[3][0] for t in ts[:n]]
    cols = [t[3][1] * 2 for t in ts[:n]]
    at = 0
    for k, (_, dtype, _, shape, _, _, offset, nbytes) in enumerate(ts):
        m = k % n
        want = (U8, (rows[m], cols[m] // 2)) if k < n else (F32, (rows[m], cols[m] // 64))
        if (dtype, tuple(shape)) != want or cols[m] % 64 or not rows[m] or offset != at:
            return None
        at += nbytes
    if at != rd.records[r][2]:
        return None
    return n, (ctypes.c_uint32 * n)(*rows), (ctypes.c_uint32 * n)(*cols)


def candidate(rd, r, lib):
    """1 when record r is a RAW int4-g64 record whose geometry the codec takes."""
    geo = geometry(rd, r) if rd.records[r][3] == wex4.CODEC_RAW else None
    return int(bool(geo) and lib.lib.wnr4i4_raw_bytes(geo[0], geo[1], geo[2], 0) == rd.records[r][2])


class Reader(wex4.Reader):
    """wex4.Reader that also decodes codec-8 records of kind I4."""

    def __init__(self, f, lib, threads=4):
        self.lib, self.threads = lib, threads
        super().__init__(f)

    def record(self, r, verify=True):
        offset, stored, decoded, codec, flags, blob, owner, first, count, digest, _ = self.records[r]
        if codec != CODEC_WNR4:
            return super().record(r, verify)
        need(blob and self.blobs[blob - 1][0] == BLOB_WNM4, f'record {r}: codec 8 without a WNM4 blob')
        model, ident = self.lib.model(self.blob(blob - 1, verify))
        need(ident == self.blobs[blob - 1][1], 'WNM4 blob id is not its model id')
        data = self.stored(r)
        need(wex4.sha256(data) == digest, 'record hash mismatch')   # at every read, as WEX4-SPEC 7.3 asks of codec 8
        need(data[12] == KIND_I4, f'record {r}: WNR4 kind {data[12]} (this tool decodes kind 3, I4)')
        geo = geometry(self, r)
        need(geo, f'record {r}: not an int4-g64 record geometry')
        need(data[13] <= 1, f'record {r}: flags {data[13]:#x}')
        err, out = self.lib.decode(model, geo, data[13], data, decoded, owner, self.threads)
        need(err == 0, f'record {r}: WNR4-I4 record rejected (error bits {err})')
        return out


def train(src, out, lib, every=1, log=sys.stderr):
    """Counts every K-th int4-g64 record of RAW container src, writes the WNM4 blob to out; returns its id."""
    need(every >= 1, '--every must be >= 1')
    counts = (ctypes.c_uint64 * lib.lib.wnr4i4_counts_words())()
    with open(src, 'rb') as f:
        rd = wex4.Reader(f)
        need(all(x[3] == wex4.CODEC_RAW for x in rd.records), 'the source must be a RAW container')
        picked = [r for r in range(len(rd.records)) if candidate(rd, r, lib)][::every]
        need(picked, 'no int4-g64 record')
        prog = Progress('train', len(picked), log, tag='wnr4i4')
        for r in picked:
            n, rows, cols = geometry(rd, r)
            data = rd.record(r)
            need(lib.lib.wnr4i4_count(n, rows, cols, 0, data, counts) == 0, f'record {r}: geometry refused')
            prog.step(len(data))
    blob = ctypes.create_string_buffer(lib.lib.wnr4i4_model_bytes())
    lib.lib.wnr4i4_build(counts, blob)
    _, ident = lib.model(blob.raw)
    with open(out + '.partial', 'wb') as f:
        f.write(blob.raw)
    os.replace(out + '.partial', out)
    print(f'[wnr4i4] model {ident:#010x} ({len(blob.raw):,} B) from {len(picked)} records -> {out}', file=log, flush=True)
    return ident


def pack(src, out, model_path, lib, G=32, pairs=0, log=sys.stderr):
    """Writes OUT: src with its int4-g64 records coded as codec 8 where smaller. Returns the root hash."""
    need(1 <= G <= 4096, 'G must be 1..4096')
    pairs = int(bool(pairs))
    with open(model_path, 'rb') as f:
        blob = f.read()
    model, ident = lib.model(blob)
    with open(src, 'rb') as f:
        rd = wex4.Reader(f)
        need(all(x[3] == wex4.CODEC_RAW for x in rd.records), 'the source must be a RAW container')
        blobs = [(b[0], b[1], rd.blob(i)) for i, b in enumerate(rd.blobs)] + [(BLOB_WNM4, ident, blob)]
        prog = Progress('pack', len(rd.records), log, tag='wnr4i4')
        codes = [candidate(rd, r, lib) for r in range(len(rd.records))]
        last, done = [None, None], [0, 0, 0]   # the record read last; coded records, their raw and stored bytes

        def show():
            prog.step(len(last[1]), f'  coded {done[0]}, {done[2] / max(done[1], 1):.4f} of raw' if done[0] else '')

        def read(r):
            last[:] = [r, rd.record(r)]
            if not codes[r]:
                show()
            return last[1]

        def encode(r):
            need(last[0] == r, 'record order')
            data, geo, owner = last[1], geometry(rd, r), rd.records[r][6]
            cap = wex4.align(len(data)) - wex4.PAGE   # the largest record still smaller than raw
            coded = None
            if cap > 0:
                buf = ctypes.create_string_buffer(cap)
                n = lib.lib.wnr4i4_encode(model, geo[0], geo[1], geo[2], pairs, data, owner, G, buf, cap)
                coded = buf.raw[:n] if n else None
            if coded:
                err, back = lib.decode(model, geo, pairs, coded, len(data), owner)
                need(err == 0 and back == data, f'record {r}: the coded record does not decode to its bytes')
                done[0] += 1
                done[1] += wex4.align(len(data))
                done[2] += n
            show()
            return coded

        records = []
        for r, (_, _, decoded, _, flags, _, owner, first, count, _, _) in enumerate(rd.records):
            tensors = [wex4.Tensor(t[0], t[1], t[3], t[7]) for t in rd.tensors[first:first + count]]
            code = codes[r]
            records.append(wex4.Record(tensors, decoded, lambda r=r: read(r),
                                       ('pre', CODEC_WNR4, lambda r=r: encode(r)) if code else 'raw', flags, owner,
                                       len(blobs) if code else 0, rd.tensors[first][5] if count else 1))
        empty = [wex4.Tensor(t[0], t[1], t[3], 0) for t in rd.tensors[rd.data_tensors:]]
        root = wex4.write(out, records, blobs, empty, rd.meta, rd.layers, rd.experts, rd.required, rd.optional)
    print(f'[wnr4i4] {out}: {done[0]} records coded, {done[2]:,} of {done[1]:,} B '
          f'({done[2] / max(done[1], 1):.4f}); root {root.hex()}', file=log, flush=True)
    return root


def verify(path, lib, source=None, log=sys.stderr):
    """Decodes every record of path (every hash checked); with source, also compares the index and every record."""
    with open(path, 'rb') as f, (open(source, 'rb') if source else contextlib.nullcontext()) as g:
        rd = Reader(f, lib)
        src = wex4.Reader(g) if g else None
        if src:
            nb = len(src.blobs)
            need(rd.tensors == src.tensors and rd.meta == src.meta and
                 (rd.required, rd.optional, rd.layers, rd.experts) == (src.required, src.optional, src.layers, src.experts),
                 'tensor table, metadata or header words differ from the source')
            need([b[:2] + b[3:5] for b in rd.blobs[:nb]] == [b[:2] + b[3:5] for b in src.blobs] and
                 all(b[0] == BLOB_WNM4 for b in rd.blobs[nb:]), 'blobs differ from the source')
            need([x[2:3] + x[4:5] + x[6:9] for x in rd.records] == [x[2:3] + x[4:5] + x[6:9] for x in src.records],
                 'record entries differ from the source')
        for i in range(len(rd.blobs)):
            rd.blob(i)
        coded = sum(x[3] == CODEC_WNR4 for x in rd.records)
        prog = Progress('verify', len(rd.records), log, tag='wnr4i4')
        for r in range(len(rd.records)):
            data = rd.record(r)
            need(not src or data == src.record(r), f'record {r} differs from the source')
            prog.step(len(data))
        stored = sum(x[1] for x in rd.records)
    print(f'[wnr4i4] {path}: OK, {len(rd.records)} records ({coded} codec 8) decoded'
          f'{", identical to the source" if src else ""}; stored {stored:,} B of {rd.total_decoded:,} decoded',
          file=log, flush=True)
    return coded


def main(argv):
    flags = ('--dll', '--every', '--model', '--G', '--source')
    values = {i + 1 for i, a in enumerate(argv[:-1]) if a in flags}
    pos = [a for i, a in enumerate(argv[2:], 2) if not a.startswith('--') and i not in values]

    def option(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    want = {'train': 2, 'pack': 2, 'verify': 1}
    if len(argv) < 2 or argv[1] not in want or len(pos) < want[argv[1]] or not option('--dll') or \
            (argv[1] == 'pack' and not option('--model')):
        print(__doc__)
        return 2
    try:
        lib = Lib(option('--dll'))
        if argv[1] == 'train':
            train(pos[0], pos[1], lib, int(option('--every', '1')))
        elif argv[1] == 'pack':
            pack(pos[0], pos[1], option('--model'), lib, int(option('--G', '32')), '--pairs' in argv)
        else:
            verify(pos[0], lib, option('--source'))
    except (OSError, ValueError) as error:   # wex4.FormatError is a ValueError; also a bad path or number
        print('REFUSED:', error)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
