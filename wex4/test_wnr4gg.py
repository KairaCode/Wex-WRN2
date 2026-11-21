# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""wnr4gg tests: a synthetic GGUF with Q2_0 and IQ4_NL experts -> RAW WEX4 (gguf2wex4) -> model -> codec-8 WEX4
-> verify, and back to the same GGUF files; refusals of damaged records. Standard library + ctypes.

    python -I wex4/test_wnr4gg.py [--dll PATH/wnr4_gg_tools.dll]

Without --dll (or WNR4GG_DLL) the test builds wnr4/tools/wnr4_gg_tools.c with $CC (default gcc), $WEX_OPENMP
(default -fopenmp) and $WEX_CFLAGS, into out/wnr4gg-test/."""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wex4  # noqa: E402
import gguf2wex4 as g2w  # noqa: E402
import wnr4gg  # noqa: E402
from test_gguf2wex4 import write_gguf, layer, sha  # noqa: E402

DLL = sys.argv[sys.argv.index('--dll') + 1] if '--dll' in sys.argv else os.environ.get('WNR4GG_DLL')
NULL = open(os.devnull, 'w')
BUILD = os.path.join(HERE, '..', 'out', 'wnr4gg-test')   # git-ignored, overwritten by each run


def lib():
    global DLL
    if not DLL:
        os.makedirs(BUILD, exist_ok=True)
        DLL = os.path.join(BUILD, 'wnr4_gg_tools.dll')
        cc = os.environ.get('CC', 'gcc')
        flags = ['-static', '-shared'] if os.name == 'nt' else ['-shared', '-fPIC']
        env = dict(os.environ, TMP=BUILD, TEMP=BUILD)   # gcc from Git Bash cannot write under C:\WINDOWS
        subprocess.run([cc, '-O2', *os.environ.get('WEX_OPENMP', '-fopenmp').split(),
                        *os.environ.get('WEX_CFLAGS', '').split(), *flags, '-o', DLL,
                        os.path.join(HERE, '..', 'wnr4', 'tools', 'wnr4_gg_tools.c')], check=True, env=env)
    return wnr4gg.Lib(DLL)


def read(path):
    with open(path, 'rb') as f:
        return f.read()


def skew(path, rng, keep_random=()):
    """Rewrites the Q2_0 and IQ4_NL tensors of a GGUF with blocks shaped like real ones (few scale exponents,
    skewed nibbles), except those whose name starts with one of keep_random."""
    with open(path, 'r+b') as f:
        data, _, tensors = g2w.parse(f, os.path.getsize(path))
        for name, _, t, off, n in tensors:
            if t in wnr4gg.CODED and not name.startswith(keep_random):
                blocks = bytearray()
                for _ in range(n // 18):
                    blocks += bytes([rng.getrandbits(8), rng.choice((0x2c, 0x2c, 0x2d, 0x2e, 0x30))])
                    blocks += bytes(rng.choice((0x00, 0x01, 0x10, 0x11, 0x12, 0x21, 0x88)) for _ in range(16))
                f.seek(data + off)
                f.write(blocks)


class Wnr4Gg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = lib()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.model = os.path.join(self.d, 'tiny.gguf')
        tensors = [('token_embd.weight', (64, 40), 42), ('output_norm.weight', (64,), 0)]
        tensors += layer(0, 4, t=16, td=42)                # IQ2_XXS gate/up stored verbatim, Q2_0 down coded
        tensors += layer(1, 4, merged=True, t=20, td=42)   # IQ4_NL gate_up and Q2_0 down: both subtypes
        tensors += layer(2, 4, t=16, td=8)                 # no coded slice: RAW
        tensors += layer(3, 4, t=16, td=20)                # random IQ4_NL down: coding is not smaller, RAW
        # 69 Q2_0 down blocks per expert, not a multiple of 32: at G 2 a full tile and a tile of 5 blocks (lanes
        # 5..31 idle), at G 32 one tile whose last row is partial
        tensors += [('blk.4.ffn_gate_up_exps.weight', (256, 64, 4), 20), ('blk.4.ffn_down_exps.weight', (64, 69, 4), 42)]
        write_gguf(self.model, tensors)
        skew(self.model, random.Random(20261121), keep_random=('blk.3.',))
        self.raw = os.path.join(self.d, 'raw.wex4')
        g2w.pack(self.raw, self.model, log=NULL)
        self.wnm4 = os.path.join(self.d, 'm.wnm4')
        self.out = os.path.join(self.d, 'gg.wex4')
        wnr4gg.train(self.raw, self.wnm4, self.lib, every=2, log=NULL)
        self.root = wnr4gg.pack(self.raw, self.out, self.wnm4, self.lib, G=2, log=NULL)   # several tiles per coded slice
        self.coded = 12   # layers 0, 1 and 4, 4 experts each

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        with open(self.raw, 'rb') as f, open(self.out, 'rb') as g:
            src, rd = wex4.Reader(f), wex4.Reader(g)   # a reader without codec 8 opens the file
            self.assertEqual(rd.root, self.root)
            nb = len(src.blobs)
            self.assertEqual(rd.blobs[:nb], [b[:2] + (rd.blobs[i][2],) + b[3:] for i, b in enumerate(src.blobs)])
            self.assertEqual(rd.blobs[nb][:2], (wnr4gg.BLOB_WNM4, self.lib.model(read(self.wnm4))[1]))
            want = {}
            for r, rec in enumerate(rd.records):
                name = rd.tensors[rec[7]][0].decode() if rec[8] else ''
                want[r] = name.startswith(('blk.0.ffn_', 'blk.1.ffn_', 'blk.4.ffn_')) and '#' in name
                self.assertEqual((rec[3], rec[5]), (wnr4gg.CODEC_WNR4, nb + 1) if want[r] else (0, 0), name)
                if want[r]:
                    self.assertLess(rec[1], src.records[r][1])
                    with self.assertRaisesRegex(wex4.FormatError, 'unsupported codec 8'):
                        rd.record(r)
            self.assertEqual(sum(want.values()), self.coded)
            self.assertEqual(rd.tensors, src.tensors)
            digest = src.content_digest()
        with open(self.out, 'rb') as g:
            self.assertEqual(wnr4gg.Reader(g, self.lib).content_digest(), digest)
        self.assertEqual(wnr4gg.verify(self.out, self.lib, self.raw, log=NULL), self.coded)
        self.assertEqual(wnr4gg.verify(self.out, self.lib, log=NULL), self.coded)
        back = os.path.join(self.d, 'back')
        os.makedirs(back)
        g2w.unpack(self.out, back, log=NULL, reader=lambda f: wnr4gg.Reader(f, self.lib))
        self.assertEqual(sha(os.path.join(back, 'tiny.gguf')), sha(self.model))
        # the command line, and a pack with the default G gives the same tensors
        out2 = os.path.join(self.d, 'gg2.wex4')
        dll = ['--dll', DLL]
        self.assertEqual(wnr4gg.main(['', 'pack', self.raw, out2, '--model', self.wnm4] + dll), 0)
        self.assertEqual(wnr4gg.main(['', 'verify', out2, '--source', self.raw] + dll), 0)
        self.assertEqual(wnr4gg.main(['', 'unpack', out2, '--check'] + dll), 0)
        with open(out2, 'rb') as g:   # the default tile depth G is 32 (record header bytes 14..15)
            rd = wex4.Reader(g)
            coded = [r for r, rec in enumerate(rd.records) if rec[3] == wnr4gg.CODEC_WNR4]
            self.assertEqual(len(coded), self.coded)
            self.assertEqual({struct.unpack_from('<H', rd.stored(r), 14)[0] for r in coded}, {32})

    def damaged(self, at_in_record=None, reseal=True, index=None):
        """A copy of the output with one byte of the first codec-8 record flipped, or index(data, rd, entry) applied
        to the index (entry: that record's index entry); reseal: the record's SHA-256, the index hash and the
        header hash made to match again."""
        with open(self.out, 'rb') as f:
            rd = wex4.Reader(f)
            r = next(r for r, rec in enumerate(rd.records) if rec[3] == wnr4gg.CODEC_WNR4)
            offset, stored = rd.records[r][:2]
            entry = rd.index_offset + len(rd.blobs) * 64 + r * 72
        data = bytearray(read(self.out))
        if at_in_record is not None:
            data[offset + at_in_record] ^= 1
        if index:
            index(data, rd, entry)
        if reseal:
            data[entry + 36:entry + 68] = wex4.sha256(data[offset:offset + stored])
            index_bytes = struct.unpack_from('<Q', data, 40)[0]
            data[96:128] = wex4.sha256(data[rd.index_offset:rd.index_offset + index_bytes])
            data[4064:4096] = wex4.sha256(data[:4064])
        path = os.path.join(self.d, 'bad.wex4')
        with open(path, 'wb') as f:
            f.write(data)
        return path, r

    def test_damaged_record_refused(self):
        with open(self.out, 'rb') as f:
            rd = wnr4gg.Reader(f, self.lib)
            r = next(r for r, rec in enumerate(rd.records) if rec[3] == wnr4gg.CODEC_WNR4)
            rec, good = rd.stored(r), rd.record(r)
            model = self.lib.model(rd.blob(rd.records[r][5] - 1))[0]
            geo = wnr4gg.geometry(rd, r)
            self.assertEqual(self.lib.decode(model, geo, rec, len(good), r), (0, good))
            self.assertNotEqual(self.lib.decode(model, geo, rec, len(good), r + 1)[0], 0)   # served for another slot
        verbatim = 64 + 4 * (struct.unpack_from('<I', rec, 20)[0] + 1)   # the IQ2_XXS gate and up slices, 2 x 2112 B
        tile = verbatim + 2 * 2112   # the first tile: 32 lane states, then its words
        cases = ((tile + 200, False, 'record hash'), (verbatim + 5, False, 'record hash'), (tile + 200, True, 'rejected'),
                 (tile + 4, True, 'rejected'), (12, True, 'WNR4 kind 3'), (8, True, 'rejected'), (verbatim + 5, True, None))
        for at, reseal, msg in cases:   # reseal: the record's SHA-256 (and the index and header hashes) made to match
            path, r = self.damaged(at, reseal)
            with open(path, 'rb') as f:
                rd = wnr4gg.Reader(f, self.lib)
                if msg is None:   # a verbatim slice is covered by the record SHA-256 only, not by a tile hash
                    self.assertNotEqual(rd.record(r), good)
                else:
                    with self.assertRaisesRegex(wex4.FormatError, msg):
                        rd.record(r)
                    if not reseal:   # a codec-8 record's SHA-256 is checked even with verify off
                        with self.assertRaisesRegex(wex4.FormatError, 'record hash'):
                            rd.record(r, verify=False)
            self.assertEqual(wnr4gg.main(['', 'verify', path, '--source', self.raw, '--dll', DLL]), 1)

    def test_damaged_index_refused(self):
        """Index fields a reader without codec 8 accepts (WEX4-SPEC 10 rule 6) but the codec-8 decoder refuses."""
        def model_id_plus_1(data, rd, entry):   # the WNM4 blob entry's id (field at +4)
            at = rd.index_offset + (len(rd.blobs) - 1) * 64 + 4
            struct.pack_into('<I', data, at, struct.unpack_from('<I', data, at)[0] + 1)

        def blob_field(value):   # the record entry's blob field (+20): 1 = the kind-32 GGUF header blob, 0 = none
            return lambda data, rd, entry: struct.pack_into('<I', data, entry + 20, value)
        for patch, msg in ((model_id_plus_1, 'WNM4 blob id is not its model id'),
                           (blob_field(1), 'without a WNM4 blob'), (blob_field(0), 'without a WNM4 blob')):
            path, r = self.damaged(index=patch)
            with open(path, 'rb') as f:
                wex4.Reader(f)   # still a valid WEX4 file
                with self.assertRaisesRegex(wex4.FormatError, msg):
                    wnr4gg.Reader(f, self.lib).record(r)
            self.assertEqual(wnr4gg.main(['', 'verify', path, '--dll', DLL]), 1)

    def test_refusals(self):
        with self.assertRaises(wex4.FormatError):
            self.lib.model(b'WNM4' + bytes(100))
        with self.assertRaisesRegex(wex4.FormatError, 'must be a RAW container'):   # a codec-8 container is not a source
            wnr4gg.pack(self.out, os.path.join(self.d, 'x.wex4'), self.wnm4, self.lib, log=NULL)
        self.assertFalse(os.path.exists(os.path.join(self.d, 'x.wex4')))
        with self.assertRaisesRegex(wex4.FormatError, 'must be a RAW container'):   # nor a training source
            wnr4gg.train(self.out, os.path.join(self.d, 'x.wnm4'), self.lib, log=NULL)
        with self.assertRaises(wex4.FormatError):
            wnr4gg.pack(self.raw, os.path.join(self.d, 'x.wex4'), self.wnm4, self.lib, G=0, log=NULL)
        plain = os.path.join(self.d, 'plain.gguf')   # no Q2_0 or IQ4_NL expert slice: nothing to train on
        write_gguf(plain, layer(0, 2, td=8))
        g2w.pack(os.path.join(self.d, 'plain.wex4'), plain, log=NULL)
        with self.assertRaisesRegex(wex4.FormatError, 'no routed-expert record'):
            wnr4gg.train(os.path.join(self.d, 'plain.wex4'), self.wnm4, self.lib, log=NULL)
        with open(self.out, 'rb') as f:   # a geometry that does not tile the record: refused before the DLL writes
            rd = wnr4gg.Reader(f, self.lib)
            r = next(r for r, rec in enumerate(rd.records) if rec[3] == wnr4gg.CODEC_WNR4)
            model = self.lib.model(rd.blob(rd.records[r][5] - 1))[0]
            with self.assertRaisesRegex(wex4.FormatError, 'does not tile'):
                self.lib.decode(model, wnr4gg.geometry(rd, r), rd.stored(r), rd.records[r][2] - 18, r)
        # the command line: a wrong --dll path, a number that is not one, a missing output folder
        x = os.path.join(self.d, 'x.wex4')
        for argv in (['verify', self.out, '--dll', os.path.join(self.d, 'none.dll')],
                     ['pack', self.raw, x, '--model', self.wnm4, '--G', 'big', '--dll', DLL],
                     ['train', self.raw, os.path.join(self.d, 'no', 'm.wnm4'), '--every', '2', '--dll', DLL]):
            self.assertEqual(wnr4gg.main([''] + argv), 1, argv)
        self.assertFalse(os.path.exists(x))

    def test_experts_only(self):
        """An experts-only container (gguf2wex4 pack --experts-only): train, pack and verify --source work on it,
        unpack refuses it."""
        raw, model, out = (os.path.join(self.d, n) for n in ('eo.wex4', 'eo.wnm4', 'eo-gg.wex4'))
        g2w.pack(raw, self.model, log=NULL, experts_only=True)
        wnr4gg.train(raw, model, self.lib, every=2, log=NULL)
        wnr4gg.pack(raw, out, model, self.lib, log=NULL)
        self.assertEqual(wnr4gg.verify(out, self.lib, raw, log=NULL), self.coded)
        dll = ['--dll', DLL]
        self.assertEqual(wnr4gg.main(['', 'verify', out, '--source', raw] + dll), 0)
        with self.assertRaisesRegex(wex4.FormatError, 'differ'):   # the full RAW container has other tables
            wnr4gg.verify(out, self.lib, self.raw, log=NULL)
        with open(out, 'rb') as f:
            rd = wnr4gg.Reader(f, self.lib)
            self.assertEqual(len(rd.records), 20)   # 5 expert layers x 4, no other record
            self.assertTrue(json.loads(rd.meta)['experts_only'])
            self.assertEqual({struct.unpack_from('<H', rd.stored(r), 14)[0] for r, x in enumerate(rd.records)
                              if x[3] == wnr4gg.CODEC_WNR4}, {32})   # pack's default G
        with self.assertRaisesRegex(wex4.FormatError, 'experts-only container'):
            g2w.unpack(out, None, log=NULL, reader=lambda f: wnr4gg.Reader(f, self.lib))
        self.assertEqual(wnr4gg.main(['', 'unpack', out, '--check'] + dll), 1)


if __name__ == '__main__':
    argv = [sys.argv[0]] + [a for i, a in enumerate(sys.argv[1:], 1) if a != '--dll' and sys.argv[i - 1] != '--dll']
    unittest.main(argv=argv)
