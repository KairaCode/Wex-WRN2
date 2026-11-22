# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""wnr4i4 tests: a synthetic RAW WEX4 container of int4-g64 expert records -> model -> codec-8 WEX4 -> verify;
refusals of damaged records. Standard library + ctypes.

    python -I wex4/test_wnr4i4.py [--dll PATH/wnr4_i4_tools.dll]

Without --dll (or WNR4I4_DLL) the test builds wnr4/tools/wnr4_i4_tools.c with $CC (default gcc), $WEX_OPENMP
(default -fopenmp) and $WEX_CFLAGS, into out/wnr4i4-test/."""
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
import wnr4i4  # noqa: E402

DLL = sys.argv[sys.argv.index('--dll') + 1] if '--dll' in sys.argv else os.environ.get('WNR4I4_DLL')
NULL = open(os.devnull, 'w')
BUILD = os.path.join(HERE, '..', 'out', 'wnr4i4-test')   # git-ignored, overwritten by each run


def lib():
    global DLL
    if not DLL:
        os.makedirs(BUILD, exist_ok=True)
        DLL = os.path.join(BUILD, 'wnr4_i4_tools.dll')
        cc = os.environ.get('CC', 'gcc')
        flags = ['-static', '-shared'] if os.name == 'nt' else ['-shared', '-fPIC']
        env = dict(os.environ, TMP=BUILD, TEMP=BUILD)   # gcc from Git Bash cannot write under C:\WINDOWS
        subprocess.run([cc, '-O2', *os.environ.get('WEX_OPENMP', '-fopenmp').split(),
                        *os.environ.get('WEX_CFLAGS', '').split(), *flags, '-o', DLL,
                        os.path.join(HERE, '..', 'wnr4', 'tools', 'wnr4_i4_tools.c')], check=True, env=env)
    return wnr4i4.Lib(DLL)


def read(path):
    with open(path, 'rb') as f:
        return f.read()


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def int4_record(rng, rows, cols, pairs=False, noise=False):
    """The bytes of an int4-g64 record: the codes of every matrix, then their f32 scales. The scales lie on the
    lattice f32(f32(E4M3 * B) / 7) of one base B per block of 128 rows x 128 inputs, as a converter of an FP8
    source writes them; the codes lean to 8 (int4 0). noise: random codes and scales (not smaller when coded)."""
    codes, scales = [], []
    for r, c in zip(rows, cols):
        gc, nbc = c // 64, (c // 64 + 1) // 2
        bases = [f32(rng.uniform(1e-4, 1e-2)) for _ in range(((r + 127) // 128) * nbc)]
        for o in range(r):
            for g in range(gc):
                k = 126 - rng.randrange(16) - rng.randrange(16)
                e4m3 = (1 + (k & 7) / 8) * 2.0 ** ((k >> 3) - 7)
                s = f32(f32(e4m3 * bases[(o // 128) * nbc + g // 2]) / 7)
                scales.append(struct.pack('<I', rng.getrandbits(32)) if noise else struct.pack('<f', s))
                v = [rng.randrange(16) if noise else min(15, max(0, round(rng.gauss(8, 1.2)))) for _ in range(64)]
                codes.append(bytes(v[2 * j] | v[2 * j + 1] << 4 if pairs else v[j] | v[j + 32] << 4 for j in range(32)))
    return b''.join(codes) + b''.join(scales)


def int4_tensors(prefix, rows, cols):
    names = ('gate_proj', 'up_proj', 'down_proj')
    ts = [wex4.Tensor(f'{prefix}.{names[m]}.weight', 'U8', (r, c // 2), r * c // 2) for m, (r, c) in enumerate(zip(rows, cols))]
    return ts + [wex4.Tensor(f'{prefix}.{names[m]}.weight.qs', 'F32', (r, c // 64), r * c // 16)
                 for m, (r, c) in enumerate(zip(rows, cols))]


class Wnr4I4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = lib()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        rng = random.Random(20261122)
        q38 = ((130, 130, 256), (256, 256, 128))   # rows, cols: a partial 128-row block, 4 and 2 groups per row
        recs = []
        for e in range(4):
            data = int4_record(rng, *q38)
            recs.append(wex4.Record(int4_tensors(f'layers.0.experts.{e}', *q38), len(data), lambda d=data: d))
        two = ((40, 3), (2048, 64))   # two matrices, 32 and 1 groups per row
        data = int4_record(rng, *two)
        recs.append(wex4.Record(int4_tensors('layers.1.experts.0', *two), len(data), lambda d=data: d))
        data = int4_record(rng, *q38, noise=True)   # random codes and scales: coding is not smaller, RAW
        recs.append(wex4.Record(int4_tensors('layers.1.experts.1', *q38), len(data), lambda d=data: d))
        data = bytes(rng.getrandbits(8) for _ in range(9000))   # not an int4 record
        recs.append(wex4.Record([wex4.Tensor('norm.weight', 'F32', (2250,), 9000)], len(data), lambda d=data: d))
        self.raw = os.path.join(self.d, 'raw.wex4')
        wex4.write(self.raw, recs, meta=b'{"wex4_meta":1}')
        self.wnm4 = os.path.join(self.d, 'm.wnm4')
        self.out = os.path.join(self.d, 'i4.wex4')
        wnr4i4.train(self.raw, self.wnm4, self.lib, every=1, log=NULL)
        self.root = wnr4i4.pack(self.raw, self.out, self.wnm4, self.lib, G=2, log=NULL)   # several tiles per matrix
        self.coded = 5

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def coded_records(rd):
        return [r for r, rec in enumerate(rd.records) if rec[3] == wnr4i4.CODEC_WNR4]

    def test_round_trip(self):
        with open(self.raw, 'rb') as f, open(self.out, 'rb') as g:
            src, rd = wex4.Reader(f), wex4.Reader(g)   # a reader without codec 8 opens the file
            self.assertEqual(rd.root, self.root)
            self.assertEqual(rd.blobs[0][:2], (wnr4i4.BLOB_WNM4, self.lib.model(read(self.wnm4))[1]))
            self.assertEqual(self.coded_records(rd), [0, 1, 2, 3, 4])
            for r in self.coded_records(rd):
                self.assertLess(rd.records[r][1], src.records[r][1])
                self.assertEqual(rd.records[r][5], 1)
                with self.assertRaisesRegex(wex4.FormatError, 'unsupported codec 8'):
                    rd.record(r)
            self.assertEqual(rd.tensors, src.tensors)
            digest = src.content_digest()
        with open(self.out, 'rb') as g:
            self.assertEqual(wnr4i4.Reader(g, self.lib).content_digest(), digest)
        self.assertEqual(wnr4i4.verify(self.out, self.lib, self.raw, log=NULL), self.coded)
        self.assertEqual(wnr4i4.verify(self.out, self.lib, log=NULL), self.coded)
        # the command line, the default G (32) and pairs packing: other record bytes, the same tensors
        out2 = os.path.join(self.d, 'i4b.wex4')
        dll = ['--dll', DLL]
        self.assertEqual(wnr4i4.main(['', 'pack', self.raw, out2, '--model', self.wnm4, '--pairs'] + dll), 0)
        self.assertEqual(wnr4i4.main(['', 'verify', out2, '--source', self.raw] + dll), 0)
        with open(out2, 'rb') as g:
            rd = wex4.Reader(g)
            coded = self.coded_records(rd)
            self.assertEqual(len(coded), self.coded)
            self.assertEqual({struct.unpack_from('<H', rd.stored(r), 14)[0] for r in coded}, {32})
            self.assertEqual({rd.stored(r)[13] for r in coded}, {1})

    def damaged(self, at_in_record=None, reseal=True, index=None):
        """A copy of the output with one byte of the first codec-8 record flipped, or index(data, rd, entry) applied
        to the index; reseal: the record's SHA-256, the index hash and the header hash made to match again."""
        with open(self.out, 'rb') as f:
            rd = wex4.Reader(f)
            r = self.coded_records(rd)[0]
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
            rd = wnr4i4.Reader(f, self.lib)
            r = self.coded_records(rd)[0]
            rec, good = rd.stored(r), rd.record(r)
            model = self.lib.model(rd.blob(rd.records[r][5] - 1))[0]
            geo = wnr4i4.geometry(rd, r)
            self.assertEqual(self.lib.decode(model, geo, 0, rec, len(good), r), (0, good))
            self.assertNotEqual(self.lib.decode(model, geo, 0, rec, len(good), r + 1)[0], 0)   # served for another slot
            self.assertNotEqual(self.lib.decode(model, geo, 1, rec, len(good), r)[0], 0)       # the other packing
        bases = 64 + 4 * (struct.unpack_from('<I', rec, 20)[0] + 1)
        tile = bases + 4 * (4 + 4 + 2)   # blocks of 128 rows x 2 groups: 2 x 2, 2 x 2 and 2 x 1
        cases = ((tile + 200, False, 'record hash'), (tile + 200, True, 'rejected'), (tile + 4, True, 'rejected'),
                 (bases + 1, True, 'rejected'), (12, True, 'WNR4 kind 2'), (13, True, 'rejected'), (8, True, 'rejected'),
                 (28, True, 'rejected'))
        for at, reseal, msg in cases:
            path, r = self.damaged(at, reseal)
            with open(path, 'rb') as f:
                rd = wnr4i4.Reader(f, self.lib)
                with self.assertRaisesRegex(wex4.FormatError, msg):
                    rd.record(r)
                if not reseal:   # a codec-8 record's SHA-256 is checked even with verify off
                    with self.assertRaisesRegex(wex4.FormatError, 'record hash'):
                        rd.record(r, verify=False)
            self.assertEqual(wnr4i4.main(['', 'verify', path, '--source', self.raw, '--dll', DLL]), 1)

    def test_damaged_index_refused(self):
        """Index fields a reader without codec 8 accepts (WEX4-SPEC 10 rule 6) but the codec-8 decoder refuses."""
        def model_id_plus_1(data, rd, entry):   # the WNM4 blob entry's id (field at +4)
            at = rd.index_offset + 4
            struct.pack_into('<I', data, at, struct.unpack_from('<I', data, at)[0] + 1)

        def blob_none(data, rd, entry):   # the record entry's blob field (+20)
            struct.pack_into('<I', data, entry + 20, 0)
        for patch, msg in ((model_id_plus_1, 'WNM4 blob id is not its model id'), (blob_none, 'without a WNM4 blob')):
            path, r = self.damaged(index=patch)
            with open(path, 'rb') as f:
                wex4.Reader(f)   # still a valid WEX4 file
                with self.assertRaisesRegex(wex4.FormatError, msg):
                    wnr4i4.Reader(f, self.lib).record(r)
            self.assertEqual(wnr4i4.main(['', 'verify', path, '--dll', DLL]), 1)

    def test_refusals(self):
        with self.assertRaises(wex4.FormatError):
            self.lib.model(b'WNM4' + bytes(100))
        with self.assertRaisesRegex(wex4.FormatError, 'must be a RAW container'):
            wnr4i4.pack(self.out, os.path.join(self.d, 'x.wex4'), self.wnm4, self.lib, log=NULL)
        self.assertFalse(os.path.exists(os.path.join(self.d, 'x.wex4')))
        with self.assertRaisesRegex(wex4.FormatError, 'must be a RAW container'):
            wnr4i4.train(self.out, os.path.join(self.d, 'x.wnm4'), self.lib, log=NULL)
        with self.assertRaises(wex4.FormatError):
            wnr4i4.pack(self.raw, os.path.join(self.d, 'x.wex4'), self.wnm4, self.lib, G=0, log=NULL)
        plain = os.path.join(self.d, 'plain.wex4')   # U8 codes with F32 scales of another shape: not int4-g64
        data = bytes(4096 + 1024)
        wex4.write(plain, [wex4.Record([wex4.Tensor('a', 'U8', (64, 64), 4096), wex4.Tensor('a.qs', 'F32', (64, 4), 1024)],
                                       len(data), lambda: data)])
        with self.assertRaisesRegex(wex4.FormatError, 'no int4-g64 record'):
            wnr4i4.train(plain, self.wnm4, self.lib, log=NULL)
        with open(self.out, 'rb') as f:   # a geometry that does not tile the record: refused before the DLL writes
            rd = wnr4i4.Reader(f, self.lib)
            r = self.coded_records(rd)[0]
            model = self.lib.model(rd.blob(rd.records[r][5] - 1))[0]
            with self.assertRaisesRegex(wex4.FormatError, 'does not tile'):
                self.lib.decode(model, wnr4i4.geometry(rd, r), 0, rd.stored(r), rd.records[r][2] - 36, r)
        x = os.path.join(self.d, 'x.wex4')
        for argv in (['verify', self.out, '--dll', os.path.join(self.d, 'none.dll')],
                     ['pack', self.raw, x, '--model', self.wnm4, '--G', 'big', '--dll', DLL],
                     ['train', self.raw, os.path.join(self.d, 'no', 'm.wnm4'), '--every', '2', '--dll', DLL]):
            self.assertEqual(wnr4i4.main([''] + argv), 1, argv)
        self.assertFalse(os.path.exists(x))


if __name__ == '__main__':
    argv = [sys.argv[0]] + [a for i, a in enumerate(sys.argv[1:], 1) if a != '--dll' and sys.argv[i - 1] != '--dll']
    unittest.main(argv=argv)
