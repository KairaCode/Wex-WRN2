# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""WNR4 MXFP4 / NVFP4 EXPERT records (wnr4/wnr4.h) in a WEX4 container. Standard library + ctypes;
the codec is the DLL of wnr4/tools/wnr4_tools.c (--dll).

    python -I wnr4mx.py train RAW.wex4 MODEL.wnm4 --dll D [--pb 1|5] [--cz] [--every K]
    python -I wnr4mx.py pack RAW.wex4 OUT.wex4 --model MODEL.wnm4 --dll D [--G 32] [--cz]
    python -I wnr4mx.py verify OUT.wex4 [--source RAW.wex4] --dll D

RAW.wex4 is a WEX4 container whose records are all RAW with EXPERT-flagged MXFP4 (kind 0) or NVFP4 (kind 1)
records. Packed files set required_features bit 2 (REQ_WNR4_EXPERT = 0x4), plus bit 0 (REQ_VALUE_EXACT = 0x1)
when --cz is used. Every coded record is decoded back and checked before publishing.
"""
import contextlib
import ctypes
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wex4  # noqa: E402
from wex4 import need  # noqa: E402
from wnr4gg import Progress  # noqa: E402

CODEC_WNR4 = wex4.CODEC_WNR4
BLOB_WNM4 = wex4.BLOB_WNM4
KIND_MX, KIND_NV = 0, 1
FLAG_CZ = 0x08

_CZ_TABLE = bytes((0 if (b & 15) == 8 else (b & 15)) | (0 if (b >> 4) == 8 else (b & 0xf0)) for b in range(256))


def canonical_zero(slab, geo):
    """Return a copy of slab with every FP4 -0 nibble (8) in its 3 weight regions mapped to +0 (0)."""
    out = bytearray(slab)
    for m in range(3):
        off = geo['woff'][m]
        n = geo['cells'][m] * 16
        out[off:off + n] = bytes(out[off:off + n]).translate(_CZ_TABLE)
    return bytes(out)


class Lib:
    """wnr4/tools/wnr4_tools.c through ctypes."""

    def __init__(self, path):
        lib = ctypes.CDLL(os.path.abspath(path))
        need(hasattr(lib, 'wnr4t_abi'), f'{path}: not a wnr4_tools library')
        P, B, U32, SZ, I = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_int
        lib.wnr4t_abi.restype = U32
        need(lib.wnr4t_abi() == 1, 'wnr4_tools ABI')
        lib.wnr4t_counts_words.restype = SZ
        lib.wnr4t_counts_words.argtypes = [U32, U32]
        lib.wnr4t_model_bytes.restype = SZ
        lib.wnr4t_model_bytes.argtypes = [U32, U32]
        lib.wnr4t_prepare.restype = U32
        lib.wnr4t_prepare.argtypes = [P, B, B, I, P]
        lib.wnr4t_count.restype = I
        lib.wnr4t_count.argtypes = [P, B, U32, P]
        lib.wnr4t_build.restype = U32
        lib.wnr4t_build.argtypes = [U32, U32, P, B]
        lib.wnr4t_model_open.restype = P
        lib.wnr4t_model_open.argtypes = [B, SZ]
        lib.wnr4t_model_close.argtypes = [P]
        lib.wnr4t_model_id.restype = U32
        lib.wnr4t_model_id.argtypes = [P]
        lib.wnr4t_encode.restype = SZ
        lib.wnr4t_encode.argtypes = [P, P, B, U32, U32, U32, B, SZ]
        lib.wnr4t_decode.restype = U32
        lib.wnr4t_decode.argtypes = [P, P, B, SZ, U32, B, I]
        self.lib, self.models = lib, {}

    def model(self, blob):
        """(handle, id, kind) of a WNM4 kind 0 (MX) or 1 (NV) blob; opened once per process."""
        if blob not in self.models:
            m = self.lib.wnr4t_model_open(blob, len(blob))
            need(m, 'not a WNM4 kind-0 (MX) or kind-1 (NV) model blob')
            ident = self.lib.wnr4t_model_id(m)
            kind = struct.unpack_from('<I', blob, 8)[0]
            self.models[blob] = m, ident, kind
        return self.models[blob]

    def prepare(self, geo_arr, src, cz=False):
        coded = ctypes.create_string_buffer(len(src))
        mapped = ctypes.c_uint64(0)
        flags = self.lib.wnr4t_prepare(geo_arr, src, coded, int(bool(cz)), ctypes.byref(mapped))
        need(flags != 0xffffffff, 'wnr4t_prepare failed')
        return flags, coded.raw, mapped.value

    def decode(self, model, geo_arr, rec, decoded, owner, threads=4):
        out = ctypes.create_string_buffer(decoded)
        err = self.lib.wnr4t_decode(model, geo_arr, rec, len(rec), owner, out, threads)
        return err, out.raw


def geo_array(geo):
    return (ctypes.c_uint32 * 13)(geo['kind'], *geo['scount'], *geo['scols'], *geo['soff'], *geo['woff'])


def geometry(rd, r):
    """(geo_dict, geo_array) for an EXPERT record, or None for non-EXPERT records."""
    if not (rd.records[r][4] & wex4.REC_EXPERT):
        return None
    g = rd.expert_geometry(r)
    return g, geo_array(g)


class Reader(wex4.Reader):
    """wex4.Reader that decodes EXPERT codec-8 records of kind MX (0) or NV (1) using wnr4_tools."""

    def __init__(self, f, lib, threads=4):
        self.lib, self.threads = lib, threads
        super().__init__(f)

    def record(self, r, verify=True):
        offset, stored, decoded, codec, flags, blob, owner, first, count, digest, _ = self.records[r]
        if codec != CODEC_WNR4 or not (flags & wex4.REC_EXPERT):
            return super().record(r, verify)
        need(self.required & wex4.REQ_WNR4_EXPERT, f'record {r}: EXPERT codec 8 requires REQ_WNR4_EXPERT')
        need(blob and self.blobs[blob - 1][0] == BLOB_WNM4, f'record {r}: codec 8 without a WNM4 blob')
        model, ident, mkind = self.lib.model(self.blob(blob - 1, verify))
        need(ident == self.blobs[blob - 1][1], 'WNM4 blob id is not its model id')
        data = self.stored(r)
        need(not verify or wex4.sha256(data) == digest, 'record hash mismatch')
        g, garr = geometry(self, r)
        need(mkind == g['kind'] and data[12] == g['kind'], f'record {r}: WNR4 kind mismatch')
        need(bool(data[13] & FLAG_CZ) == bool(flags & wex4.REC_VALUE_EXACT), f'record {r}: CZ flag mismatch')
        err, out = self.lib.decode(model, garr, data, decoded, owner, self.threads)
        need(err == 0, f'record {r}: WNR4 record rejected (error bits {err})')
        return out


def train(src, out, lib, pb=1, cz=False, every=1, log=sys.stderr):
    """Trains a WNM4 kind-0 (MX) or kind-1 (NV) model on every K-th EXPERT record of RAW container src."""
    need(pb in (1, 5), '--pb must be 1 or 5')
    need(every >= 1, '--every must be >= 1')
    with open(src, 'rb') as f:
        rd = wex4.Reader(f)
        need(all(x[3] == wex4.CODEC_RAW for x in rd.records), 'the source must be a RAW container')
        experts = [r for r in range(len(rd.records)) if rd.records[r][4] & wex4.REC_EXPERT]
        need(experts, 'no EXPERT record')
        kinds = {rd.expert_geometry(r)['kind'] for r in experts}
        need(len(kinds) == 1, 'mixed MXFP4 and NVFP4 EXPERT records')
        kind = next(iter(kinds))
        picked = experts[::every]
        counts = (ctypes.c_uint64 * lib.lib.wnr4t_counts_words(kind, pb))()
        prog = Progress('train', len(picked), log, tag='wnr4mx')
        for r in picked:
            _, garr = geometry(rd, r)
            data = rd.record(r)
            _, coded, _ = lib.prepare(garr, data, cz)
            need(lib.lib.wnr4t_count(garr, coded, pb, counts) == 0, f'record {r}: wnr4t_count failed')
            prog.step(len(data))
    nbytes = lib.lib.wnr4t_model_bytes(kind, pb)
    blob = ctypes.create_string_buffer(nbytes)
    ident = lib.lib.wnr4t_build(kind, pb, counts, blob)
    need(ident != 0, 'wnr4t_build failed')
    lib.model(blob.raw)
    with open(out + '.partial', 'wb') as f:
        f.write(blob.raw)
    os.replace(out + '.partial', out)
    kname = 'NV' if kind else 'MX'
    print(f'[wnr4mx] {kname} pb={pb} model {ident:#010x} ({len(blob.raw):,} B) from {len(picked)} experts -> {out}',
          file=log, flush=True)
    return ident


def pack(src, out, model_path, lib, G=32, cz=False, log=sys.stderr):
    """Writes OUT: src with EXPERT records of the model's kind coded as codec 8 where smaller. Returns root hash."""
    need(1 <= G <= 4096, 'G must be 1..4096')
    with open(model_path, 'rb') as f:
        blob = f.read()
    model, ident, mkind = lib.model(blob)
    with open(src, 'rb') as f:
        rd = wex4.Reader(f)
        need(all(x[3] == wex4.CODEC_RAW for x in rd.records), 'the source must be a RAW container')
        blobs = [(b[0], b[1], rd.blob(i)) for i, b in enumerate(rd.blobs)] + [(BLOB_WNM4, ident, blob)]
        codes = []
        for r in range(len(rd.records)):
            g = rd.expert_geometry(r) if (rd.records[r][4] & wex4.REC_EXPERT) else None
            codes.append(bool(g is not None and g['kind'] == mkind))
        need(any(codes), f'no {"NV" if mkind else "MX"} EXPERT record')
        prog = Progress('pack', len(rd.records), log, tag='wnr4mx')
        last, done = [None, None], [0, 0, 0, 0]   # coded count, raw bytes, stored bytes, cz mapped

        def show():
            prog.step(len(last[1]), f'  coded {done[0]}, {done[2] / max(done[1], 1):.4f} of raw' if done[0] else '')

        def read(r):
            last[:] = [r, rd.record(r)]
            if not codes[r]:
                show()
            return last[1]

        def encode(r):
            need(last[0] == r, 'record order')
            data = last[1]
            g, garr = geometry(rd, r)
            owner = rd.records[r][6]
            cap = wex4.align(len(data)) - wex4.PAGE
            coded_rec = None
            if cap > 0:
                rflags, coded_slab, mapped = lib.prepare(garr, data, cz)
                buf = ctypes.create_string_buffer(cap)
                n = lib.lib.wnr4t_encode(model, garr, coded_slab, rflags, owner, G, buf, cap)
                coded_rec = buf.raw[:n] if n else None
                if coded_rec:
                    expected = canonical_zero(data, g) if cz else data
                    err, back = lib.decode(model, garr, coded_rec, len(data), owner)
                    need(err == 0 and back == expected, f'record {r}: decoded bytes differ from expected')
                    done[0] += 1
                    done[1] += wex4.align(len(data))
                    done[2] += n
                    done[3] += mapped
            show()
            return coded_rec

        records = []
        for r, (_, _, decoded, _, flags, _, owner, first, count, _, _) in enumerate(rd.records):
            tensors = [wex4.Tensor(t[0], t[1], t[3], t[7]) for t in rd.tensors[first:first + count]]
            code = codes[r]
            rec_flags = (flags | wex4.REC_VALUE_EXACT) if (code and cz) else flags
            records.append(wex4.Record(tensors, decoded, lambda r=r: read(r),
                                       ('pre', CODEC_WNR4, lambda r=r: encode(r)) if code else 'raw',
                                       rec_flags, owner, len(blobs) if code else 0,
                                       rd.tensors[first][5] if count else 1))
        empty = [wex4.Tensor(t[0], t[1], t[3], 0) for t in rd.tensors[rd.data_tensors:]]
        required = rd.required | wex4.REQ_WNR4_EXPERT | (wex4.REQ_VALUE_EXACT if cz else 0)
        root = wex4.write(out, records, blobs, empty, rd.meta, rd.layers, rd.experts, required, rd.optional)
    print(f'[wnr4mx] {out}: {done[0]} experts coded, {done[2]:,} of {done[1]:,} B '
          f'({done[2] / max(done[1], 1):.4f}){f", {done[3]:,} -0 mapped" if cz else ""}; root {root.hex()}',
          file=log, flush=True)
    return root


def verify(path, lib, source=None, log=sys.stderr):
    """Decodes every record of path (every hash checked); with source, compares index and every record."""
    with open(path, 'rb') as f, (open(source, 'rb') if source else contextlib.nullcontext()) as g:
        rd = Reader(f, lib)
        src = wex4.Reader(g) if g else None
        if src:
            nb = len(src.blobs)
            need(rd.tensors == src.tensors and rd.meta == src.meta and
                 (rd.optional, rd.layers, rd.experts) == (src.optional, src.layers, src.experts),
                 'tensor table, metadata or header words differ from the source')
            need([b[:2] + b[3:5] for b in rd.blobs[:nb]] == [b[:2] + b[3:5] for b in src.blobs] and
                 all(b[0] == BLOB_WNM4 for b in rd.blobs[nb:]), 'blobs differ from the source')
            need([x[2:3] + (x[4] & ~wex4.REC_VALUE_EXACT,) + x[6:9] for x in rd.records] ==
                 [x[2:3] + (x[4] & ~wex4.REC_VALUE_EXACT,) + x[6:9] for x in src.records],
                 'record entries differ from the source')
        for i in range(len(rd.blobs)):
            rd.blob(i)
        coded = sum(x[3] == CODEC_WNR4 for x in rd.records)
        prog = Progress('verify', len(rd.records), log, tag='wnr4mx')
        for r in range(len(rd.records)):
            data = rd.record(r)
            if src:
                want = src.record(r)
                if rd.records[r][3] == CODEC_WNR4 and (rd.records[r][4] & wex4.REC_VALUE_EXACT):
                    want = canonical_zero(want, rd.expert_geometry(r))
                need(data == want, f'record {r} differs from the source')
            prog.step(len(data))
        stored = sum(x[1] for x in rd.records)
    print(f'[wnr4mx] {path}: OK, {len(rd.records)} records ({coded} codec 8) decoded'
          f'{", matches source" if src else ""}; stored {stored:,} B of {rd.total_decoded:,} decoded',
          file=log, flush=True)
    return coded


def main(argv):
    flags = ('--dll', '--pb', '--every', '--model', '--G', '--source')
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
            train(pos[0], pos[1], lib, int(option('--pb', '1')), '--cz' in argv, int(option('--every', '1')))
        elif argv[1] == 'pack':
            pack(pos[0], pos[1], option('--model'), lib, int(option('--G', '32')), '--cz' in argv)
        else:
            verify(pos[0], lib, option('--source'))
    except (OSError, ValueError) as error:
        print('REFUSED:', error)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
