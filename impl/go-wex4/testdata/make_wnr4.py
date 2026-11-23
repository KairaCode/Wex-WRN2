#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Writes the codec-8 (WNR4) test files of this folder: kind GG (2) and kind I4 (3) records in WEX4 containers,
and changed copies of two of them, from synthetic data with fixed seeds.

    python -I make_wnr4.py --gg-dll GG.dll --i4-dll I4.dll WORK_DIR OUT_DIR

GG.dll and I4.dll are wnr4/tools/wnr4_gg_tools.c and wnr4/tools/wnr4_i4_tools.c built as shared libraries
(README.md of this folder has the commands). WORK_DIR receives the RAW sources (src-*.wex4), the two WNM4 models
and the intermediate packs; OUT_DIR the wnr4-*.wex4 files, wnr4-MANIFEST.tsv and wnr4-records.tsv.

The tools are used as black boxes, through their public calls: wex4/wex4.py (Tensor, Record, write, Reader) writes
the RAW sources, wex4/wnr4gg.py and wex4/wnr4i4.py (Lib, train, pack, verify, Reader) train the models, code the
records and give the verdicts. A pack codes every record with one tile depth G and one packing; to have several
in one file, each source is packed once per (G, packing) and the final file takes each record's stored bytes from
the pack chosen for it, written by wex4.write as the packs write them (the script checks that a file assembled
with every record from one pack is that pack byte for byte). The changed copies are a byte of a record or of the
index changed, then the record's SHA-256, the index hash and the header hash recomputed (as the tests of
wex4/test_wnr4gg.py do), unless the row says the hashes are left alone.

Expected content digests are those of the RAW sources (wex4.Reader.content_digest, the digest of
`python -I wex4/wex4.py verify`); verdicts are those of wnr4gg.py / wnr4i4.py verify. Standard library + ctypes.
Deterministic (the bytes depend on Python's random module with fixed seeds and on the reference encoders).
"""
import ctypes
import hashlib
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "wex4"))
import wex4  # noqa: E402
import wnr4gg  # noqa: E402
import wnr4i4  # noqa: E402

NULL = open(os.devnull, "w")
META = b'{"wex4_meta":1}'
Q2_0, IQ4_NL, IQ2_XXS, Q8_0 = 42, 20, 16, 8
GGML = {Q2_0: (64, 18), IQ4_NL: (32, 18), IQ2_XXS: (256, 66), Q8_0: (32, 34)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- GG sources

def gg_blocks(rng, n, skewed):
    """n 18-byte blocks. skewed: few scale high bytes and few code bytes, as real Q2_0 / IQ4_NL slices;
    else random bytes (coding them is not smaller)."""
    if not skewed:
        return bytes(rng.getrandbits(8) for _ in range(18 * n))
    out = bytearray()
    for _ in range(n):
        out += bytes([rng.getrandbits(8), rng.choice((0x2C, 0x2C, 0x2D, 0x2E, 0x30))])
        out += bytes(rng.choice((0x00, 0x01, 0x10, 0x11, 0x12, 0x21, 0x88)) for _ in range(16))
    return bytes(out)


def gg_slice(rng, layer, kind, expert, t, rows, ne0, skewed=True):
    block, size = GGML[t]
    nbytes = ne0 // block * size * rows
    if t in (Q2_0, IQ4_NL):
        data = gg_blocks(rng, nbytes // 18, skewed)
    else:
        data = bytes(rng.getrandbits(8) for _ in range(nbytes))   # verbatim slice (IQ2_XXS, Q8_0)
    name = f"blk.{layer}.ffn_{kind}_exps.weight#{expert}"
    return wex4.Tensor(name, 128 + t, (rows, ne0), nbytes), data


# record -> (layer, expert, [(kind, ggml type, rows, ne0, skewed)], G of the pack it is taken from)
GG_MIX = [
    (0, 0, [("gate", IQ2_XXS, 2, 256, 1), ("up", Q8_0, 3, 32, 1), ("down", Q2_0, 444, 64, 1)], 32),
    (0, 1, [("gate", IQ2_XXS, 2, 256, 1), ("up", Q8_0, 3, 32, 1), ("down", Q2_0, 444, 64, 1)], 7),
    (1, 0, [("gate_up", IQ4_NL, 387, 32, 1), ("down", Q2_0, 69, 64, 1)], 32),
    (1, 1, [("gate_up", IQ4_NL, 387, 32, 1), ("down", Q2_0, 69, 64, 1)], 1),
    (2, 0, [("down", Q2_0, 460, 64, 1)], 1),
    (3, 0, [("gate", Q2_0, 20, 64, 1), ("gate_up", IQ2_XXS, 1, 256, 1), ("up", IQ4_NL, 200, 32, 1),
            ("down", Q2_0, 240, 64, 1)], 7),
    (4, 0, [("down", IQ4_NL, 460, 32, 0)], 32),
    (4, 1, [("down", IQ4_NL, 460, 32, 0)], 32),
    (None, None, "output_norm.weight", 32),
]
GG_ONE = [GG_MIX[0]]


def gg_source(path, plan, seed):
    rng = random.Random(seed)
    recs, raw = [], []
    for layer, expert, slices, _ in plan:
        if layer is None:   # a tensor that is not a routed expert: never coded
            data = bytes(rng.getrandbits(8) for _ in range(4096))
            ts = [wex4.Tensor(slices, "F32", (1024,), 4096)]
        else:
            ts, parts = [], []
            for kind, t, rows, ne0, skewed in slices:
                ten, d = gg_slice(rng, layer, kind, expert, t, rows, ne0, skewed)
                ts.append(ten)
                parts.append(d)
            data = b"".join(parts)
        raw.append(data)
        recs.append(wex4.Record(ts, len(data), lambda d=data: d))
    wex4.write(path, recs, meta=META)
    return raw


# ---------------------------------------------------------------- I4 sources

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def e4m3(k):
    return k * 2.0 ** -9 if k < 8 else (1 + (k & 7) / 8) * 2.0 ** ((k >> 3) - 7)


def lat(k, base):
    # binary32 multiply then binary32 divide: double arithmetic rounded to binary32 after each step is the
    # correctly rounded binary32 result for one multiply or one divide (53 >= 2 * 24 + 2)
    return f32(f32(e4m3(k) * base) / 7)


# scales put at groups 3, 14, 25, .. of an 'escapes' matrix; in block row 0, whose other scales lie on the lattice:
# smallest subnormal, a larger subnormal, 1e-7, +0 (lat(0, B) = +0: on every lattice); in block row 1:
# NaN, NaN with a payload, +inf, -inf, -0, -0.5
ESCAPES = ([0x00000001, 0x00400000, 0x33D6BF95, 0x00000000],
           [0x7FC00000, 0x7F800001, 0x7F800000, 0xFF800000, 0x80000000, 0xBF000000])


def i4_matrix(rng, rows, cols, style, base_of=None):
    """Codes (planar or pairs packed later) and scale words of one matrix. style: 'lattice' (scales on the lattice
    of one base per 128 x 128 block), 'escapes' (block row 0 on the lattice with ESCAPES at fixed groups, block
    row 1 only negative scales: they all escape, so that block's base is never read), 'subnormal' (a tiny base:
    most lattice values are subnormal), 'noise' (random codes and scale bits)."""
    gc, nbc = cols // 64, (cols // 64 + 1) // 2
    nb = ((rows + 127) // 128) * nbc
    if style == "subnormal":
        bases = [f32(1.375 * 2.0 ** -120), f32(1.0625 * 2.0 ** -118)] * nb
    else:
        bases = [f32(rng.uniform(1e-4, 1e-2)) for _ in range(nb)]
    codes, scales = [], []
    for o in range(rows):
        for g in range(gc):
            b = (o // 128) * nbc + g // 2
            i = o * gc + g
            if style == "noise":
                s = rng.getrandbits(32)
            elif style == "escapes" and o >= 128:
                s = struct.unpack("<I", struct.pack("<f", -rng.uniform(1e-4, 1e-1)))[0]
            else:
                if style == "subnormal":
                    k = 126 if i % 17 == 0 else rng.randrange(0, 127)
                else:
                    k = 126 - rng.randrange(16) - rng.randrange(16)
                s = struct.unpack("<I", struct.pack("<f", lat(k, bases[b])))[0]
            if style == "escapes":
                special = ESCAPES[o >= 128]
                j = (i - 128 * gc * (o >= 128)) // 11
                if i % 11 == 3 and j < len(special):
                    s = special[j]
            scales.append(s)
            v = [rng.randrange(16) if style == "noise" else min(15, max(0, round(rng.gauss(8, 1.2)))) for _ in range(64)]
            codes.append(v)
    return codes, scales


def i4_record(rng, mats, pairs):
    """mats: [(rows, cols, style)]. Bytes of the record (codes of every matrix, then their f32 scales)."""
    cparts, sparts = [], []
    for rows, cols, style in mats:
        codes, scales = i4_matrix(rng, rows, cols, style)
        for v in codes:
            cparts.append(bytes(v[2 * j] | v[2 * j + 1] << 4 if pairs else v[j] | v[j + 32] << 4 for j in range(32)))
        sparts.append(struct.pack(f"<{len(scales)}I", *scales))
    return b"".join(cparts) + b"".join(sparts)


def i4_tensors(prefix, mats):
    names = ("gate_proj", "up_proj", "down_proj")
    ts = [wex4.Tensor(f"{prefix}.{names[m]}.weight", "U8", (r, c // 2), r * c // 2) for m, (r, c, _) in enumerate(mats)]
    return ts + [wex4.Tensor(f"{prefix}.{names[m]}.weight.qs", "F32", (r, c // 64), r * c // 16)
                 for m, (r, c, _) in enumerate(mats)]


# record -> (name prefix, [(rows, cols, style)], (G, pairs) of the pack it is taken from)
I4_MIX = [
    ("layers.0.experts.0", [(129, 64, "lattice"), (128, 64, "lattice"), (127, 128, "lattice")], (32, 0)),
    ("layers.0.experts.1", [(8, 2048, "lattice")], (64, 1)),
    ("layers.0.experts.2", [(4, 2112, "lattice"), (129, 64, "lattice")], (100, 0)),
    ("layers.0.experts.3", [(256, 64, "escapes")], (1, 0)),
    ("layers.0.experts.4", [(130, 128, "subnormal")], (32, 1)),
    ("layers.0.experts.5", [(128, 64, "noise")], (32, 0)),
]
I4_ONE = [("layers.0.experts.0", [(256, 64, "escapes")], (32, 0))]


def i4_source(path, plan, seed):
    rng = random.Random(seed)
    recs, raw = [], []
    for prefix, mats, (_, pairs) in plan:
        data = i4_record(rng, mats, pairs)
        raw.append(data)
        recs.append(wex4.Record(i4_tensors(prefix, mats), len(data), lambda d=data: d))
    wex4.write(path, recs, meta=META)
    return raw


# ---------------------------------------------------------------- assembling and changing files

def assemble(src, packs, choice, model_blob, model_id, out):
    """out: src with record r stored as in packs[choice[r]] (codec 8, or RAW where that pack kept it RAW)."""
    with open(src, "rb") as f:
        rd = wex4.Reader(f)
        stored = {}
        for key, path in packs.items():
            with open(path, "rb") as g:
                pk = wex4.Reader(g)
                for r, rec in enumerate(pk.records):
                    if choice[r] == key:
                        stored[r] = pk.stored(r) if rec[3] == wnr4gg.CODEC_WNR4 else None
        records = []
        for r, (_, _, decoded, _, flags, _, owner, first, count, _, _) in enumerate(rd.records):
            tensors = [wex4.Tensor(t[0], t[1], t[3], t[7]) for t in rd.tensors[first:first + count]]
            data = rd.record(r)
            code = stored[r]
            records.append(wex4.Record(tensors, decoded, lambda d=data: d,
                                       ("pre", wnr4gg.CODEC_WNR4, lambda c=code: c) if code else "raw", flags, owner,
                                       1 if code else 0, rd.tensors[first][5] if count else 1))
        wex4.write(out, records, [(wnr4gg.BLOB_WNM4, model_id, model_blob)], [], rd.meta, rd.layers, rd.experts,
                   rd.required, rd.optional)


def changed(path, out, record=None, at=None, value=None, xor=None, index=None, reseal=True):
    """A copy of path with byte `at` of record `record` set to value (or xor-ed), or index(data, rd, entry) applied;
    reseal: the record's SHA-256, the index hash and the header hash made to match again."""
    with open(path, "rb") as f:
        rd = wex4.Reader(f)
        offset, stored = rd.records[record][:2]
        entry = rd.index_offset + len(rd.blobs) * 64 + record * 72
    data = bytearray(open(path, "rb").read())
    if at is not None:
        for k, v in enumerate(value if value is not None else xor):
            data[offset + at + k] = v if value is not None else data[offset + at + k] ^ v
    if index:
        index(data, rd, entry)
    if reseal:
        data[entry + 36:entry + 68] = wex4.sha256(data[offset:offset + stored])
        index_bytes = struct.unpack_from("<Q", data, 40)[0]
        data[96:128] = wex4.sha256(data[rd.index_offset:rd.index_offset + index_bytes])
        data[4064:4096] = wex4.sha256(data[:4064])
    with open(out, "wb") as f:
        f.write(data)


def layout(path, r):
    """(stored record, ntiles, P, start of the area after the tile offsets) of record r."""
    with open(path, "rb") as f:
        rec = wex4.Reader(f).stored(r)
    nt, p = struct.unpack_from("<II", rec, 20)
    return rec, nt, p, 64 + 4 * (nt + 1)


def oracle(module, lib, path, source=None):
    """('accept', content digest) or ('refuse', message) from the codec-8 tool."""
    try:
        module.verify(path, lib, source, log=NULL)
        with open(path, "rb") as f:
            return "accept", module.Reader(f, lib).content_digest().hex()
    except (OSError, ValueError) as error:
        return "refuse", str(error)


def describe(path):
    """Per record: codec, kind, flags, G, ntiles for a codec-8 record."""
    out = []
    with open(path, "rb") as f:
        rd = wex4.Reader(f)
        for r, rec in enumerate(rd.records):
            if rec[3] == wnr4gg.CODEC_WNR4:
                s = rd.stored(r)
                g, nt = struct.unpack_from("<H", s, 14)[0], struct.unpack_from("<I", s, 20)[0]
                what = f"codec 8 kind {s[12]} flags {s[13]} G {g} ntiles {nt} stored {rec[1]}"
                if s[12] == wnr4i4.KIND_I4:   # the bases: ceil(rows / 128) x ceil(gc / 2) per matrix
                    n = rec[8] // 2
                    nb = sum((t[3][0] + 127) // 128 * ((t[3][1] * 2 // 64 + 1) // 2) for t in rd.tensors[rec[7]:rec[7] + n])
                    bases = struct.unpack_from(f"<{nb}I", s, 64 + 4 * (nt + 1))
                    what += " bases " + ",".join(f"{b:#010x}" for b in bases)
                out.append(what)
            else:
                out.append(f"codec {rec[3]} stored {rec[1]}")
    return out


def main(argv):
    if len(argv) != 7 or argv[1] != "--gg-dll" or argv[3] != "--i4-dll":
        print(__doc__)
        return 2
    gg, i4 = wnr4gg.Lib(argv[2]), wnr4i4.Lib(argv[4])
    work, outd = argv[5], argv[6]
    os.makedirs(work, exist_ok=True)
    os.makedirs(outd, exist_ok=True)
    W, O = (lambda n: os.path.join(work, n)), (lambda n: os.path.join(outd, n))
    rows, recrows = [], []

    # ---- GG
    raw_mix = gg_source(W("src-gg-mix.wex4"), GG_MIX, 20261123)
    raw_one = gg_source(W("src-gg-one.wex4"), GG_ONE, 20261124)
    gg.model_id = wnr4gg.train(W("src-gg-mix.wex4"), W("gg.wnm4"), gg, log=NULL)
    gg_blob = open(W("gg.wnm4"), "rb").read()
    for name, src, plan, raw in (("wnr4-gg-mix.wex4", "src-gg-mix.wex4", GG_MIX, raw_mix),
                                 ("wnr4-gg-one.wex4", "src-gg-one.wex4", GG_ONE, raw_one)):
        packs = {}
        for G in sorted({p[3] for p in plan}):
            packs[G] = W(f"{name}.G{G}")
            wnr4gg.pack(W(src), packs[G], W("gg.wnm4"), gg, G=G, log=NULL)
            assemble(W(src), packs, [G] * len(plan), gg_blob, gg.model_id, W("check.wex4"))
            assert open(W("check.wex4"), "rb").read() == open(packs[G], "rb").read(), "assemble != pack"
        assemble(W(src), packs, [p[3] for p in plan], gg_blob, gg.model_id, O(name))
        for r, (d, what) in enumerate(zip(raw, describe(O(name)))):
            recrows.append((name, r, what, len(d), sha(d)))

    # ---- I4
    raw_mix = i4_source(W("src-i4-mix.wex4"), I4_MIX, 20261125)
    raw_one = i4_source(W("src-i4-one.wex4"), I4_ONE, 20261126)
    i4.model_id = wnr4i4.train(W("src-i4-mix.wex4"), W("i4.wnm4"), i4, log=NULL)
    i4_blob = open(W("i4.wnm4"), "rb").read()
    for name, src, plan, raw in (("wnr4-i4-mix.wex4", "src-i4-mix.wex4", I4_MIX, raw_mix),
                                 ("wnr4-i4-one.wex4", "src-i4-one.wex4", I4_ONE, raw_one)):
        packs = {}
        for G, pairs in sorted({p[2] for p in plan}):
            packs[G, pairs] = W(f"{name}.G{G}.P{pairs}")
            wnr4i4.pack(W(src), packs[G, pairs], W("i4.wnm4"), i4, G=G, pairs=pairs, log=NULL)
            assemble(W(src), packs, [(G, pairs)] * len(plan), i4_blob, i4.model_id, W("check.wex4"))
            assert open(W("check.wex4"), "rb").read() == open(packs[G, pairs], "rb").read(), "assemble != pack"
        assemble(W(src), packs, [p[2] for p in plan], i4_blob, i4.model_id, O(name))
        for r, (d, what) in enumerate(zip(raw, describe(O(name)))):
            recrows.append((name, r, what, len(d), sha(d)))

    # ---- changed copies
    one = O("wnr4-gg-one.wex4")
    rec, nt, p, area = layout(one, 0)
    verb = 132 + 102                       # the IQ2_XXS gate and Q8_0 up slices, stored verbatim
    vpad = area + verb                     # V = 236: 2 zero bytes after them
    tile = area + verb + 2                 # payload: tile 0 (32 initial states, then its words)
    words = tile + 128
    assert rec[vpad:tile] == b"\0\0" and (verb + 2) % 4 == 0, "verbatim area is not where the script expects it"

    def blob_id_plus_1(data, rd, entry):
        at = rd.index_offset + (len(rd.blobs) - 1) * 64 + 4
        struct.pack_into("<I", data, at, struct.unpack_from("<I", data, at)[0] + 1)

    def blob_field_0(data, rd, entry):
        struct.pack_into("<I", data, entry + 20, 0)
    gg_changes = [
        ("wnr4-gg-bad-word.wex4", dict(at=words + 10, xor=[1]), "one byte of tile 0's words changed (bit 0)"),
        ("wnr4-gg-bad-kind0.wex4", dict(at=12, value=[0]), "record header byte 12 (kind) set to 0, MX"),
        ("wnr4-gg-bad-flags.wex4", dict(at=13, value=[1]), "record header byte 13 (flags) set to 1"),
        ("wnr4-gg-bad-reserved.wex4", dict(at=40, value=[1]), "record header byte 40 (of the zero bytes 28..63) set to 1"),
        ("wnr4-gg-bad-ntiles.wex4", dict(at=20, value=list(struct.pack("<I", nt + 1))),
         f"record header bytes 20..23 (ntiles) {nt} -> {nt + 1}"),
        ("wnr4-gg-bad-owner.wex4", dict(at=8, xor=[1]), "record header bytes 8..11 (owner) 0 -> 1"),
        ("wnr4-gg-bad-model.wex4", dict(at=4, xor=[1]),
         "record header bytes 4..7 (model id) xor 1: not the id of the WNM4 blob, which is unchanged"),
        ("wnr4-gg-bad-zero16.wex4", dict(at=16, value=[1]), "record header byte 16 (of the zero bytes 16..19) set to 1"),
        ("wnr4-gg-bad-pad.wex4", dict(at=vpad + 1, value=[1]), "last zero byte after the verbatim slices set to 1"),
        ("wnr4-gg-mut-verbatim.wex4", dict(at=area + 5, xor=[0x40]), "byte 5 of the verbatim gate slice changed"),
        ("wnr4-gg-bad-sha.wex4", dict(at=words + 10, xor=[1], reseal=False),
         "one byte of tile 0's words changed, hashes NOT recomputed (record SHA-256 mismatch)"),
        ("wnr4-gg-bad-modelid.wex4", dict(index=blob_id_plus_1), "index: id of the WNM4 blob entry + 1"),
        ("wnr4-gg-bad-blob0.wex4", dict(index=blob_field_0), "index: blob field of the codec-8 record entry set to 0"),
    ]
    for name, kw, what in gg_changes:
        changed(one, O(name), record=0, **kw)
    one_i4 = O("wnr4-i4-one.wex4")
    rec, nt, p, area = layout(one_i4, 0)
    nbases = 2                              # 256 rows x 1 group: 2 blocks of 128 rows
    bases = struct.unpack_from(f"<{nbases}I", rec, area)
    words_i4 = area + 4 * nbases + 128
    other = 0x3F800000 if bases[1] == 0 else 0
    i4_changes = [
        ("wnr4-i4-bad-word.wex4", dict(at=words_i4 + 10, xor=[1]), "one byte of tile 0's words changed (bit 0)"),
        ("wnr4-i4-bad-base.wex4", dict(at=area, value=list(struct.pack("<I", 0x7F800000))),
         f"base of block 0 {bases[0]:#010x} -> 0x7f800000 (+inf, not a valid base)"),
        ("wnr4-i4-bad-flags.wex4", dict(at=13, value=[1]), "record header byte 13 (packing) 0 -> 1: pairs, while the tile hashes were made with planar packing"),
        ("wnr4-i4-bad-anchor.wex4", dict(at=30, value=[1]), "anchor[1] (bytes 30..31) set to 1 in a record of 1 matrix"),
        ("wnr4-i4-bad-owner.wex4", dict(at=8, xor=[1]), "record header bytes 8..11 (owner) 0 -> 1 (the index says 0)"),
        ("wnr4-i4-mut-base.wex4", dict(at=area + 4, value=list(struct.pack("<I", other))),
         f"base of block 1 {bases[1]:#010x} -> {other:#010x}; every scale of block 1 escapes, so it is never read"),
    ]
    for name, kw, what in i4_changes:
        changed(one_i4, O(name), record=0, **kw)

    # ---- verdicts
    def digest(path):
        with open(path, "rb") as f:
            return wex4.Reader(f).content_digest().hex()
    entries = [("wnr4-gg-mix.wex4", wnr4gg, gg, "src-gg-mix.wex4", "valid: kind GG, 9 records (see wnr4-records.tsv)"),
               ("wnr4-gg-one.wex4", wnr4gg, gg, "src-gg-one.wex4", "valid: kind GG, 1 record; base of the wnr4-gg-* changes")]
    entries += [(n, wnr4gg, gg, None, w) for n, _, w in gg_changes]
    entries += [("wnr4-i4-mix.wex4", wnr4i4, i4, "src-i4-mix.wex4", "valid: kind I4, 6 records (see wnr4-records.tsv)"),
                ("wnr4-i4-one.wex4", wnr4i4, i4, "src-i4-one.wex4", "valid: kind I4, 1 record; base of the wnr4-i4-* changes")]
    entries += [(n, wnr4i4, i4, None, w) for n, _, w in i4_changes]
    for name, module, lib, src, what in entries:
        data = open(O(name), "rb").read()
        verdict, info = oracle(module, lib, O(name))
        if src:
            v2, info2 = oracle(module, lib, O(name), W(src))
            assert (v2, info2) == ("accept", digest(W(src))), (name, v2, info2)
            assert info == info2
            info = f"{info} (= RAW source)"
        else:
            # what each changed copy is for: a bad-* copy must be refused, a mut-* copy accepted (a change of the
            # tools that moved a field would otherwise end in a manifest row that silently says the other thing)
            assert verdict == ("refuse" if "-bad-" in name else "accept"), (name, verdict, info)
        if not src and verdict == "accept":
            base = "wnr4-gg-one.wex4" if name.startswith("wnr4-gg") else "wnr4-i4-one.wex4"
            same = info == oracle(module, lib, O(base))[1]
            assert same == name.startswith("wnr4-i4"), (name, info)   # mut-base: same digest; mut-verbatim: another
            bsrc = W("src-gg-one.wex4" if name.startswith("wnr4-gg") else "src-i4-one.wex4")
            assert oracle(module, lib, O(name), bsrc)[0] == ("accept" if same else "refuse"), name   # verify --source
            info = f"{info} ({'same as' if same else 'differs from'} {base})"
        rows.append((name, len(data), data[4064:4096].hex(), verdict, info, what))
        print(f"{name:28} {len(data):>8,} B  {verdict:6} {info}")
    with open(O("wnr4-MANIFEST.tsv"), "w", newline="\n") as f:
        f.write("# file\tbytes\troot\tverdict\tcontent digest or reason\twhat\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    with open(O("wnr4-records.tsv"), "w", newline="\n") as f:
        f.write("# file\trecord\tstored as\tdecoded bytes\tSHA-256 of the decoded record\n")
        for r in recrows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"GG model id {gg.model_id:#010x} ({len(gg_blob):,} B), I4 model id {i4.model_id:#010x} ({len(i4_blob):,} B)")
    print("I4 one: bases", " ".join(f"{b:#010x}" for b in bases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
