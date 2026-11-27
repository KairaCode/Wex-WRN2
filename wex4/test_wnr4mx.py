# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""wnr4mx tests: synthetic RAW WEX4 containers of MXFP4 and NVFP4 EXPERT records -> WNM4 model ->
codec-8 WEX4 -> verify (pb=1/5, plain and --cz, RAW fallback, G=32/64, corruption refusals).

    python -I wex4/test_wnr4mx.py [--dll PATH/wnr4_tools.dll]

Without --dll (or WNR4MX_DLL) builds wnr4/tools/wnr4_tools.c into out/wnr4mx-test/."""
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
import wnr4mx  # noqa: E402

DLL = sys.argv[sys.argv.index('--dll') + 1] if '--dll' in sys.argv else os.environ.get('WNR4MX_DLL')
NULL = open(os.devnull, 'w')
BUILD = os.path.join(HERE, '..', 'out', 'wnr4mx-test')


def lib():
    global DLL
    if not DLL:
        os.makedirs(BUILD, exist_ok=True)
        DLL = os.path.join(BUILD, 'wnr4_tools.dll')
        cc = os.environ.get('CC', 'gcc')
        flags = ['-static', '-shared'] if os.name == 'nt' else ['-shared', '-fPIC']
        env = dict(os.environ, TMP=BUILD, TEMP=BUILD)
        subprocess.run([cc, '-O2', *os.environ.get('WEX_OPENMP', '-fopenmp').split(),
                        *os.environ.get('WEX_CFLAGS', '').split(), *flags, '-o', DLL,
                        os.path.join(HERE, '..', 'wnr4', 'tools', 'wnr4_tools.c')], check=True, env=env)
    return wnr4mx.Lib(DLL)


def read(path):
    with open(path, 'rb') as f:
        return f.read()


def fp4_slab(rng, kind, shapes, ceil_friendly=False, noise=False):
    """Build a synthetic MXFP4 (kind 0) or NVFP4 (kind 1) 3-matrix expert slab."""
    wmult = 8 if kind else 16
    B = 16 if kind else 32
    scales, weights = [], []
    for rows, scols in shapes:
        sc = rows * scols
        if noise:
            scales.append(bytes(rng.getrandbits(8) for _ in range(sc)))
            weights.append(bytes(rng.getrandbits(8) for _ in range(sc * wmult)))
            continue
        if kind == 0:
            scales.append(bytes(120 + rng.randrange(-1, 2) for _ in range(sc)))
        else:
            s = bytearray(sc)
            for r in range(rows):
                base = 56 + (r % 5) * 2
                for c in range(scols):
                    s[r * scols + c] = base + (c & 1)
            scales.append(bytes(s))
        w = bytearray(sc * wmult)
        for blk in range(sc):
            even_blk = ceil_friendly and kind == 0 and (blk % 3 == 0)
            for j in range(B):
                if j == B - 1 and ceil_friendly and kind == 0:
                    a = 7
                elif even_blk:
                    a = (0, 2, 4, 6)[rng.randrange(4)]
                else:
                    r = rng.randrange(100)
                    a = 0 if r < 45 else 1 if r < 72 else 2 if r < 86 else 3 if r < 94 else 4 if r < 98 else 7
                nib = a | (8 if rng.getrandbits(1) else 0)
                idx = blk * B + j
                if idx & 1:
                    w[idx >> 1] |= nib << 4
                else:
                    w[idx >> 1] = nib
        weights.append(bytes(w))
    return b''.join(scales) + b''.join(weights)


def expert_tensors(layer, expert, kind, shapes):
    s_dt = 'F8_E4M3' if kind else 'F8_E8M0'
    wmult = 8 if kind else 16
    prefix = f'layers.{layer}.experts.{expert}'
    ts = [wex4.Tensor(f'{prefix}.w{m + 1}.scale', s_dt, (r, c), r * c) for m, (r, c) in enumerate(shapes)]
    ts += [wex4.Tensor(f'{prefix}.w{m + 1}.weight', 'U8', (r, c * wmult), r * c * wmult)
           for m, (r, c) in enumerate(shapes)]
    return ts


def build_raw_container(path, kind, shapes, seed, ceil_friendly=False):
    rng = random.Random(seed)
    recs = []
    # 3 compressible experts + 1 incompressible noise expert (tests RAW fallback) + 1 non-expert tensor
    for e in range(4):
        data = fp4_slab(rng, kind, shapes, ceil_friendly=ceil_friendly, noise=(e == 3))
        recs.append(wex4.Record(expert_tensors(0, e, kind, shapes), len(data), lambda d=data: d,
                                flags=wex4.REC_EXPERT, owner=e))
    norm = bytes(rng.getrandbits(8) for _ in range(4096))
    recs.append(wex4.Record([wex4.Tensor('norm.weight', 'F32', (1024,), 4096)], 4096, lambda d=norm: d))
    req = wex4.REQ_WNR4_EXPERT if kind else 0
    wex4.write(path, recs, meta=b'{"kind":"nv"}' if kind else b'{"kind":"mx"}', layers=1, experts=4, required=req)


class Wnr4Mx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = lib()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _run_roundtrip(self, kind, shapes, pb, cz, G, seed, ceil_friendly=False):
        tag = f'k{kind}_pb{pb}_cz{int(cz)}_G{G}'
        raw = os.path.join(self.d, f'{tag}.raw.wex4')
        mdl = os.path.join(self.d, f'{tag}.wnm4')
        out = os.path.join(self.d, f'{tag}.wex4')
        build_raw_container(raw, kind, shapes, seed, ceil_friendly=ceil_friendly)
        ident = wnr4mx.train(raw, mdl, self.lib, pb=pb, cz=cz, log=NULL)
        self.assertNotEqual(ident, 0)
        root = wnr4mx.pack(raw, out, mdl, self.lib, G=G, cz=cz, log=NULL)
        with open(out, 'rb') as f:
            rd = wex4.Reader(f)   # wex4.Reader opens valid REQ_WNR4_EXPERT containers without DLL
            self.assertEqual(rd.root, root)
            want_req = wex4.REQ_WNR4_EXPERT | (wex4.REQ_VALUE_EXACT if cz else 0)
            self.assertEqual(rd.required, want_req)
            self.assertEqual(rd.blobs[0][:2], (wex4.BLOB_WNM4, ident))
            coded = [r for r, rec in enumerate(rd.records) if rec[3] == wex4.CODEC_WNR4]
            self.assertEqual(coded, [0, 1, 2])               # expert 3 fell back to RAW
            self.assertEqual(rd.records[3][3], wex4.CODEC_RAW)
            self.assertEqual(rd.records[3][4], wex4.REC_EXPERT)   # no REC_VALUE_EXACT on RAW fallback
            for r in coded:
                self.assertEqual(bool(rd.records[r][4] & wex4.REC_VALUE_EXACT), cz)
                self.assertEqual(struct.unpack_from('<H', rd.stored(r), 14)[0], G)
                self.assertEqual(bool(rd.stored(r)[13] & wnr4mx.FLAG_CZ), cz)
                if ceil_friendly and kind == 0:
                    self.assertEqual(rd.stored(r)[13] & 7, 7)
                if kind == 1:
                    self.assertEqual(rd.stored(r)[13] & 7, 0)
        self.assertEqual(wnr4mx.verify(out, self.lib, raw, log=NULL), 3)
        self.assertEqual(wnr4mx.verify(out, self.lib, log=NULL), 3)
        return raw, mdl, out

    def test_mx_variants(self):
        shapes = ((16, 12), (16, 12), (12, 16))
        self._run_roundtrip(0, shapes, pb=1, cz=False, G=32, seed=20261101)
        self._run_roundtrip(0, shapes, pb=5, cz=False, G=64, seed=20261102)
        self._run_roundtrip(0, shapes, pb=1, cz=True, G=32, seed=20261103, ceil_friendly=True)
        # CLI check
        raw = os.path.join(self.d, 'cli.raw.wex4')
        mdl = os.path.join(self.d, 'cli.wnm4')
        out = os.path.join(self.d, 'cli.wex4')
        build_raw_container(raw, 0, shapes, 20261104, ceil_friendly=True)
        self.assertEqual(wnr4mx.main(['', 'train', raw, mdl, '--pb', '1', '--cz', '--dll', DLL]), 0)
        self.assertEqual(wnr4mx.main(['', 'pack', raw, out, '--model', mdl, '--G', '32', '--cz', '--dll', DLL]), 0)
        self.assertEqual(wnr4mx.main(['', 'verify', out, '--source', raw, '--dll', DLL]), 0)

    def test_nv_variants(self):
        # scols 36 >= 33 (exercises row-above scale context) and scols 24 < 33 (no row-above context), even scount
        shapes = ((20, 36), (20, 36), (12, 24))
        self._run_roundtrip(1, shapes, pb=1, cz=False, G=32, seed=20261110)
        self._run_roundtrip(1, shapes, pb=5, cz=True, G=32, seed=20261111)

    def test_corruption_refusals(self):
        shapes = ((16, 12), (16, 12), (12, 16))
        raw, mdl, out = self._run_roundtrip(0, shapes, pb=1, cz=True, G=32, seed=20261120, ceil_friendly=True)
        base = read(out)

        def mutate(fn, reseal=True):
            data = bytearray(base)
            with open(out, 'rb') as f:
                rd = wex4.Reader(f)
                r0_off, r0_len = rd.records[0][:2]
                entry = rd.index_offset + len(rd.blobs) * 64
                index_bytes = rd.index_bytes
            fn(data, rd, r0_off, r0_len, entry)
            if reseal:
                data[entry + 36:entry + 68] = wex4.sha256(data[r0_off:r0_off + r0_len])
                data[96:128] = wex4.sha256(data[rd.index_offset:rd.index_offset + index_bytes])
                data[4064:4096] = wex4.sha256(data[:4064])
            p = os.path.join(self.d, 'bad.wex4')
            with open(p, 'wb') as f:
                f.write(data)
            return p

        # 1. Bit flip in coded record without resealing -> SHA-256 mismatch
        p = mutate(lambda d, rd, off, n, e: d.__setitem__(off + 100, d[off + 100] ^ 1), reseal=False)
        self.assertEqual(wnr4mx.main(['', 'verify', p, '--dll', DLL]), 1)
        # 2. Bit flip in coded record WITH resealing -> codec hash / struct refusal
        p = mutate(lambda d, rd, off, n, e: d.__setitem__(off + 100, d[off + 100] ^ 1), reseal=True)
        self.assertEqual(wnr4mx.main(['', 'verify', p, '--dll', DLL]), 1)
        # 3. Owner mismatch in record header
        p = mutate(lambda d, rd, off, n, e: d.__setitem__(off + 8, d[off + 8] ^ 1), reseal=True)
        self.assertEqual(wnr4mx.main(['', 'verify', p, '--dll', DLL]), 1)
        # 4. CZ flag mismatch between record header and index
        p = mutate(lambda d, rd, off, n, e: d.__setitem__(off + 13, d[off + 13] ^ wnr4mx.FLAG_CZ), reseal=True)
        self.assertEqual(wnr4mx.main(['', 'verify', p, '--dll', DLL]), 1)
        # 5. Corrupt WNM4 blob id in index
        p = mutate(lambda d, rd, off, n, e: struct.pack_into('<I', d, rd.index_offset + 4,
                                                            struct.unpack_from('<I', d, rd.index_offset + 4)[0] ^ 1),
                   reseal=True)
        self.assertEqual(wnr4mx.main(['', 'verify', p, '--dll', DLL]), 1)
        # 6. CLI bad args / non-RAW source
        self.assertEqual(wnr4mx.main(['', 'pack', out, os.path.join(self.d, 'x.wex4'), '--model', mdl, '--dll', DLL]), 1)
        self.assertEqual(wnr4mx.main(['', 'train', raw, os.path.join(self.d, 'x.wnm4'), '--pb', '3', '--dll', DLL]), 1)


if __name__ == '__main__':
    argv = [sys.argv[0]] + [a for i, a in enumerate(sys.argv[1:], 1) if a != '--dll' and sys.argv[i - 1] != '--dll']
    unittest.main(argv=argv)
