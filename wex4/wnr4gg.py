# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""WNR4-GG records (wnr4/wnr4_gg.h) in a WEX4 container of a GGUF model. Standard library + ctypes; the codec is
the DLL of wnr4/tools/wnr4_gg_tools.c (--dll).

    python -I wnr4gg.py train RAW.wex4 MODEL.wnm4 --dll D [--every K]    # model from every K-th expert record
    python -I wnr4gg.py pack RAW.wex4 OUT.wex4 --model MODEL.wnm4 --dll D [--G 32]
    python -I wnr4gg.py verify OUT.wex4 [--source RAW.wex4] --dll D     # every record decoded (and compared)
    python -I wnr4gg.py unpack OUT.wex4 OUT_DIR|--check --dll D          # the GGUF parts, SHA-256 checked

RAW.wex4 is a container of gguf2wex4.py pack, every record RAW (with --experts-only it holds the routed experts alone;
unpack refuses such a container, verify --source compares with it). --G is the tile depth, 1..4096: G 32 gives about
3.4x the GPU decode throughput of G 128 (RTX 5090, batches of 32) for about 0.07 points of ratio. Draft, experimental:
  - codec 8 = a WNR4 record of any kind (the record header's kind byte tells MX, NV and GG apart; this tool
    writes and reads kind 2, GG); blob kind 3 = a WNM4 model of any kind, id = the model id. A codec-8 record
    names its model in its blob field.
  - the geometry of a GG record is its tensors in table order: ggml type = dtype - 128 (a plain dtype such as F32
    is never a coded type), bytes = nbytes. Owner = the record's index (the records are not EXPERT-flagged), in
    the index and in the record header.
  - pack codes a routed-expert record (tensors blk.L.ffn_{gate,gate_up,up,down}_exps.weight#E) that holds a Q2_0
    or IQ4_NL slice when the coded record is smaller than raw, and decodes it back before writing it; every other
    record is written RAW. Every blob of the source is copied (kind 32 GGUF headers verbatim, same indices), the
    WNM4 blob comes last; the tensor table, the metadata and the header words are those of the source, so the
    tensors, and the GGUF files that unpack rebuilds, are byte for byte the source's.
Readers that do not know codec 8 open such a file and refuse to decode those records (WEX4-SPEC 5.2, 10 rule 6).
pack and verify stream: a few records in memory at a time.
"""
import contextlib
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wex4  # noqa: E402
import gguf2wex4 as g2w  # noqa: E402
from wex4 import need  # noqa: E402

CODEC_WNR4 = 8
BLOB_WNM4 = 3
KIND_GG = 2
CODED = (42, 20)   # ggml Q2_0, IQ4_NL
NOT_GGML = 0xffffffff


class Lib:
    """wnr4/tools/wnr4_gg_tools.c through ctypes."""

    def __init__(self, path):
        lib = ctypes.CDLL(os.path.abspath(path))
        need(hasattr(lib, 'wnr4gg_abi'), f'{path}: not a wnr4_gg_tools library')
        P, B, U32, U64 = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint64
        lib.wnr4gg_abi.restype = U32
        need(lib.wnr4gg_abi() == 1, 'wnr4_gg_tools ABI')
        lib.wnr4gg_counts_words.restype = U64
        lib.wnr4gg_model_bytes.restype = U32
        lib.wnr4gg_count.argtypes = [U32, P, P, B, P]
        lib.wnr4gg_build.restype = ctypes.c_int
        lib.wnr4gg_build.argtypes = [P, B]
        lib.wnr4gg_model_open.restype = P
        lib.wnr4gg_model_open.argtypes = [B, U64]
        lib.wnr4gg_model_id.restype = U32
        lib.wnr4gg_model_id.argtypes = [P]
        lib.wnr4gg_encode.restype = U64
        lib.wnr4gg_encode.argtypes = [P, U32, P, P, B, U32, U32, B, U64]
        lib.wnr4gg_decode.restype = U32
        lib.wnr4gg_decode.argtypes = [P, U32, P, P, B, U64, U32, B, ctypes.c_int]
        self.lib, self.models = lib, {}

    def model(self, blob):
        """(handle, id) of a WNM4 blob; opened once per process."""
        if blob not in self.models:
            m = self.lib.wnr4gg_model_open(blob, len(blob))
            need(m, 'not a WNM4 kind-2 (GG) model blob')
            self.models[blob] = m, self.lib.wnr4gg_model_id(m)
        return self.models[blob]

    def decode(self, model, geo, rec, decoded, owner, threads=4):
        n, types, sizes = geo
        need(sum(sizes) == decoded, 'geometry does not tile the record')   # the DLL writes sum(sizes) bytes
        out = ctypes.create_string_buffer(decoded)
        err = self.lib.wnr4gg_decode(model, n, types, sizes, rec, len(rec), owner, out, threads)
        return err, out.raw


def geometry(rd, r):
    """(nseg, types, sizes) of record r's tensors as the DLL takes them."""
    first, count = rd.records[r][7:9]
    ts = rd.tensors[first:first + count]
    need(1 <= count <= 4 and all(t[5] == 1 for t in ts), f'record {r}: not a WNR4-GG record geometry')
    types = [t[1] - 128 if t[1] >= 128 else NOT_GGML for t in ts]
    return count, (ctypes.c_uint32 * count)(*types), (ctypes.c_uint32 * count)(*[t[7] for t in ts])


def candidate(rd, r):
    """1 when record r is a routed expert of gguf2wex4 with a Q2_0 or IQ4_NL slice."""
    first, count = rd.records[r][7:9]
    ts = rd.tensors[first:first + count]
    if not 1 <= count <= 4 or rd.records[r][3] != wex4.CODEC_RAW:
        return 0
    for name, dtype, _, _, _, t_count, _, nbytes in ts:
        base, _, e = name.decode().rpartition('#')
        if t_count != 1 or not (g2w.EXPERT.match(base) and e.isdigit()):
            return 0
    return any(t[1] - 128 in CODED and t[7] % 18 == 0 for t in ts)


class Reader(wex4.Reader):
    """wex4.Reader that also decodes codec-8 records of kind GG."""

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
        # checked at every read, verify or not: the verbatim slices have no other check (WEX4-SPEC 7.3)
        need(wex4.sha256(data) == digest, 'record hash mismatch')
        need(data[12] == KIND_GG, f'record {r}: WNR4 kind {data[12]} (this tool decodes kind 2, GG)')
        err, out = self.lib.decode(model, geometry(self, r), data, decoded, owner, self.threads)
        need(err == 0, f'record {r}: WNR4-GG record rejected (error bits {err})')
        return out


class Progress:
    def __init__(self, what, total, log, every=5.0):
        self.what, self.total, self.log, self.every = what, total, log, every
        self.n = self.bytes = 0
        self.t0 = self.shown = time.time()

    def step(self, nbytes, extra=''):
        self.n += 1
        self.bytes += nbytes
        now = time.time()
        if now - self.shown >= self.every or self.n == self.total:
            self.shown = now
            print(f'[wnr4gg] {self.what} {self.n}/{self.total} records  {self.bytes / 1e9:.2f} GB  '
                  f'{self.bytes / 1e6 / max(now - self.t0, 1e-6):.0f} MB/s{extra}', file=self.log, flush=True)


def train(src, out, lib, every=1, log=sys.stderr):
    """Counts every K-th routed-expert record of RAW container src, writes the WNM4 blob to out; returns its id."""
    need(every >= 1, '--every must be >= 1')
    counts = (ctypes.c_uint64 * lib.lib.wnr4gg_counts_words())()
    with open(src, 'rb') as f:
        rd = wex4.Reader(f)
        need(all(x[3] == wex4.CODEC_RAW for x in rd.records), 'the source must be a RAW container (gguf2wex4.py pack)')
        picked = [r for r in range(len(rd.records)) if candidate(rd, r)][::every]
        need(picked, 'no routed-expert record with a Q2_0 or IQ4_NL slice')
        prog = Progress('train', len(picked), log)
        for r in picked:
            n, types, sizes = geometry(rd, r)
            data = rd.record(r)
            need(lib.lib.wnr4gg_count(n, types, sizes, data, counts) == 0, f'record {r}: geometry refused')
            prog.step(len(data))
    blob = ctypes.create_string_buffer(lib.lib.wnr4gg_model_bytes())
    need(lib.lib.wnr4gg_build(counts, blob) == 0, 'model build: out of memory')
    _, ident = lib.model(blob.raw)
    with open(out + '.partial', 'wb') as f:
        f.write(blob.raw)
    os.replace(out + '.partial', out)
    print(f'[wnr4gg] model {ident:#010x} ({len(blob.raw):,} B) from {len(picked)} records -> {out}', file=log, flush=True)
    return ident


def pack(src, out, model_path, lib, G=32, log=sys.stderr):
    """Writes OUT: src with its routed-expert records coded as codec 8 where smaller. Returns the root hash."""
    need(1 <= G <= 4096, 'G must be 1..4096')
    with open(model_path, 'rb') as f:
        blob = f.read()
    model, ident = lib.model(blob)
    with open(src, 'rb') as f:
        rd = wex4.Reader(f)
        need(all(x[3] == wex4.CODEC_RAW for x in rd.records), 'the source must be a RAW container (gguf2wex4.py pack)')
        blobs = [(b[0], b[1], rd.blob(i)) for i, b in enumerate(rd.blobs)] + [(BLOB_WNM4, ident, blob)]
        prog = Progress('pack', len(rd.records), log)
        codes = [candidate(rd, r) for r in range(len(rd.records))]
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
                n = lib.lib.wnr4gg_encode(model, geo[0], geo[1], geo[2], data, owner, G, buf, cap)
                coded = buf.raw[:n] if n else None
            if coded:
                err, back = lib.decode(model, geo, coded, len(data), owner)
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
    print(f'[wnr4gg] {out}: {done[0]} records coded, {done[2]:,} of {done[1]:,} B '
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
        prog = Progress('verify', len(rd.records), log)
        for r in range(len(rd.records)):
            data = rd.record(r)
            need(not src or data == src.record(r), f'record {r} differs from the source')
            prog.step(len(data))
        stored = sum(x[1] for x in rd.records)
    print(f'[wnr4gg] {path}: OK, {len(rd.records)} records ({coded} codec 8) decoded'
          f'{", identical to the source" if src else ""}; stored {stored:,} B of {rd.total_decoded:,} decoded',
          file=log, flush=True)
    return coded


def main(argv):
    flags = ('--dll', '--every', '--model', '--G', '--source')
    values = {i + 1 for i, a in enumerate(argv[:-1]) if a in flags}
    pos = [a for i, a in enumerate(argv[2:], 2) if not a.startswith('--') and i not in values]

    def option(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    want = {'train': 2, 'pack': 2, 'verify': 1, 'unpack': 1}
    if len(argv) < 2 or argv[1] not in want or len(pos) < want[argv[1]] or not option('--dll') or \
            (argv[1] == 'pack' and not option('--model')) or (argv[1] == 'unpack' and len(pos) < 2 and '--check' not in argv):
        print(__doc__)
        return 2
    try:
        lib = Lib(option('--dll'))
        if argv[1] == 'train':
            train(pos[0], pos[1], lib, int(option('--every', '1')))
        elif argv[1] == 'pack':
            pack(pos[0], pos[1], option('--model'), lib, int(option('--G', '32')))
        elif argv[1] == 'verify':
            verify(pos[0], lib, option('--source'))
        else:
            out_dir = None if '--check' in argv else pos[1]
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            g2w.unpack(pos[0], out_dir, reader=lambda f: Reader(f, lib))
    except (OSError, ValueError) as error:   # wex4.FormatError is a ValueError; also a bad path or number
        print('REFUSED:', error)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
