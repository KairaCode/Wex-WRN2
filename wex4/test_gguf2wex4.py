# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""gguf2wex4 tests: byte-exact round trips of synthetic GGUF files, the engine's expert-record contract,
refusals. Standard library only.

    python -I wex4/test_gguf2wex4.py [--verify PATH/wex4_verify.exe]

--verify (or WEX4_VERIFY) also runs the C checker on every container."""
import hashlib
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

VERIFY = sys.argv[sys.argv.index('--verify') + 1] if '--verify' in sys.argv else os.environ.get('WEX4_VERIFY')
rng = random.Random(20261120)
NULL = open(os.devnull, 'w')


def s(text):
    b = text.encode()
    return struct.pack('<Q', len(b)) + b


def nbytes(ne, t):
    block, size = g2w.GGML[t]
    rows = 1
    for d in ne[1:]:
        rows *= d
    return ne[0] // block * size * rows


def write_gguf(path, tensors, alignment=32, tail=0, kv_extra=b'', n_kv_extra=0, gap_byte=0):
    """tensors: [(name, ne, ggml type)]; data random. Returns {name: bytes}."""
    kv = s('general.architecture') + struct.pack('<I', 8) + s('tiny')
    kv += s('tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, 3) + s('a') + s('b') + s('é')
    kv += s('nested') + struct.pack('<IIQ', 9, 9, 2) + struct.pack('<IQ', 4, 2) + bytes(8) + struct.pack('<IQ', 8, 1) + s('x')
    n_kv = 3 + n_kv_extra
    if alignment != 32:
        kv += s('general.alignment') + struct.pack('<II', 4, alignment)
        n_kv += 1
    kv += kv_extra
    infos, payload, off, data = b'', {}, 0, []
    for name, ne, t in tensors:
        n = nbytes(ne, t)
        infos += s(name) + struct.pack('<I', len(ne)) + struct.pack(f'<{len(ne)}Q', *ne) + struct.pack('<IQ', t, off)
        payload[name] = bytes(rng.getrandbits(8) for _ in range(n))
        data.append(payload[name] + bytes([gap_byte]) * (-n % alignment))
        off += n + (-n % alignment)
    head = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), n_kv) + kv + infos
    if tensors:
        head += bytes(-len(head) % alignment)
    body = b''.join(data)
    if tensors:   # what llama.cpp writes: no padding after the last tensor
        body = body[:len(body) - (-nbytes(tensors[-1][1], tensors[-1][2]) % alignment)]
    with open(path, 'wb') as f:
        f.write(head + body + bytes(tail))
    return payload


def layer(L, n_exp, merged=False, t=16, td=20):
    gate = [(f'blk.{L}.ffn_gate_up_exps.weight', (256, 64, n_exp), t)] if merged else \
        [(f'blk.{L}.ffn_gate_exps.weight', (256, 32, n_exp), t), (f'blk.{L}.ffn_up_exps.weight', (256, 32, n_exp), t)]
    return gate + [(f'blk.{L}.ffn_down_exps.weight', (64, 256, n_exp), td), (f'blk.{L}.attn_norm.weight', (256,), 0)]


def sha(path):
    with open(path, 'rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1 << 20), b''):   # hashlib.file_digest needs Python 3.11
            h.update(block)
        return h.hexdigest()


class GgufToWex4(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def round_trip(self, model, part_bytes=64 << 20):
        out = os.path.join(self.d, 'm.wex4')
        root = g2w.pack(out, model, part_bytes, log=NULL)
        with open(out, 'rb') as f:
            rd = wex4.Reader(f)
            self.assertEqual(rd.root, root)
            rd.content_digest(True)   # every hash, every record
        if VERIFY:
            p = subprocess.run([VERIFY, out], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(p.returncode, 0, p.stdout.decode(errors='replace'))
        back = os.path.join(self.d, 'back')
        os.makedirs(back, exist_ok=True)
        g2w.unpack(out, None, log=NULL)
        g2w.unpack(out, back, log=NULL)
        for p in g2w.split_parts(model):
            self.assertEqual(sha(os.path.join(back, os.path.basename(p))), sha(p), p)
        return out

    def test_single_file(self):
        model = os.path.join(self.d, 'tiny.gguf')
        tensors = [('token_embd.weight', (64, 300), 8), ('output_norm.weight', (64,), 0), ('big.weight', (256, 100), 1)]
        for L in range(3):
            tensors += layer(L, 4, td=8 if L % 2 else 20)
        src = write_gguf(model, tensors)
        out = self.round_trip(model, part_bytes=8192)   # big.weight (51,200 B) split in 7 parts
        with open(out, 'rb') as f:
            rd = wex4.Reader(f)
            # the engine's contract: a layer's experts are consecutive RAW records of one stride, gate|up|down
            for L in range(3):
                first = rd.tensors[rd.by_name[f'blk.{L}.ffn_gate_exps.weight#0'.encode()]][4]
                stride = rd.records[first][1]
                for e in range(4):
                    r = first + e
                    off, stored, decoded, codec, flags = rd.records[r][:5]
                    self.assertEqual((off, stored, codec, flags), (rd.records[first][0] + e * stride, stride, 0, 0))
                    names = [rd.tensors[i][0] for i in range(rd.records[r][7], rd.records[r][7] + rd.records[r][8])]
                    self.assertEqual(names, [f'blk.{L}.ffn_{k}_exps.weight#{e}'.encode() for k in ('gate', 'up', 'down')])
                    each = [len(src[f'blk.{L}.ffn_{k}_exps.weight']) // 4 for k in ('gate', 'up', 'down')]
                    want = b''.join(src[f'blk.{L}.ffn_{k}_exps.weight'][e * n:(e + 1) * n] for k, n in zip(('gate', 'up', 'down'), each))
                    self.assertEqual(rd.record(r), want)
            dt, shape, data = rd.tensor('output_norm.weight')
            self.assertEqual((dt, tuple(shape), data), ('F32', (64,), src['output_norm.weight']))
            self.assertEqual(rd.tensors[rd.by_name[b'token_embd.weight']][1], 128 + 8)
            self.assertEqual(rd.tensor('big.weight')[2], src['big.weight'])

    def test_split_model(self):
        """part 1 metadata only (unpadded), a layer's experts across two parts, other alignment, zero tail."""
        stem = os.path.join(self.d, 'm')
        write_gguf(f'{stem}-00001-of-00003.gguf', [])
        write_gguf(f'{stem}-00002-of-00003.gguf', [('token_embd.weight', (64, 10), 0)] + layer(0, 3)[:2], alignment=64)
        write_gguf(f'{stem}-00003-of-00003.gguf', layer(0, 3)[2:] + layer(1, 3, merged=True), tail=100)
        self.round_trip(f'{stem}-00001-of-00003.gguf')

    def test_nonzero_padding_refused(self):
        model = os.path.join(self.d, 'bad.gguf')
        write_gguf(model, [('a.weight', (3,), 0), ('b.weight', (3,), 0)], gap_byte=7)
        with self.assertRaises(wex4.FormatError):
            g2w.pack(os.path.join(self.d, 'x.wex4'), model, log=NULL)
        self.assertFalse(os.path.exists(os.path.join(self.d, 'x.wex4')))

    def test_incomplete_experts_refused(self):
        model = os.path.join(self.d, 'bad.gguf')
        write_gguf(model, layer(0, 2)[1:])   # no gate
        with self.assertRaises(wex4.FormatError):
            g2w.pack(os.path.join(self.d, 'x.wex4'), model, log=NULL)

    def test_tensor_in_two_parts_refused(self):
        stem = os.path.join(self.d, 'm')
        for k in (1, 2):
            write_gguf(f'{stem}-0000{k}-of-00002.gguf', layer(0, 2)[:3])
        with self.assertRaises(wex4.FormatError):
            g2w.pack(os.path.join(self.d, 'x.wex4'), f'{stem}-00001-of-00002.gguf', log=NULL)

    def test_experts_only(self):
        """pack --experts-only: the expert records of a full pack, nothing else; unpack refuses the container."""
        model = os.path.join(self.d, 'tiny.gguf')
        write_gguf(model, [('token_embd.weight', (64, 30), 8)] + layer(0, 3) + layer(1, 2, merged=True, td=8))
        full, eo = os.path.join(self.d, 'full.wex4'), os.path.join(self.d, 'eo.wex4')
        g2w.pack(full, model, log=NULL)
        self.assertEqual(g2w.main(['', 'pack', eo, model, '--experts-only']), 0)
        with open(full, 'rb') as f, open(eo, 'rb') as g:
            src, rd = wex4.Reader(f), wex4.Reader(g)
            rd.content_digest(True)
            names = [t[0].decode() for t in rd.tensors]
            self.assertEqual(len(rd.records), 5)
            self.assertTrue(all(g2w.EXPERT.match(n.rpartition('#')[0]) for n in names), names)
            self.assertEqual(json.loads(rd.meta), dict(json.loads(src.meta), experts_only=True))
            self.assertEqual([b[:2] + b[3:] for b in rd.blobs], [b[:2] + b[3:] for b in src.blobs])
            for r, rec in enumerate(rd.records):
                t = rd.tensors[rec[7]]
                self.assertEqual(rd.record(r), src.record(src.tensors[src.by_name[t[0]]][4]), t[0])
        if VERIFY:
            p = subprocess.run([VERIFY, eo], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(p.returncode, 0, p.stdout.decode(errors='replace'))
        back = os.path.join(self.d, 'back')
        os.makedirs(back)
        for out_dir in (None, back):
            with self.assertRaisesRegex(wex4.FormatError, 'experts-only container'):
                g2w.unpack(eo, out_dir, log=NULL)
        self.assertEqual(os.listdir(back), [])
        self.assertEqual(g2w.main(['', 'unpack', eo, '--check']), 1)
        plain = os.path.join(self.d, 'plain.gguf')   # no routed expert: nothing to pack
        write_gguf(plain, [('token_embd.weight', (64, 30), 8)])
        with self.assertRaisesRegex(wex4.FormatError, 'no routed-expert tensor'):
            g2w.pack(os.path.join(self.d, 'x.wex4'), plain, log=NULL, experts_only=True)
        self.assertFalse(os.path.exists(os.path.join(self.d, 'x.wex4')))

    def test_tampered_record_refused(self):
        model = os.path.join(self.d, 'tiny.gguf')
        write_gguf(model, layer(0, 2))
        out = self.round_trip(model)
        with open(out, 'rb') as f:
            rd = wex4.Reader(f)
            at = rd.records[-1][0] + 5
        with open(out, 'r+b') as f:
            f.seek(at)
            b = f.read(1)
            f.seek(at)
            f.write(bytes([b[0] ^ 1]))
        with self.assertRaises(wex4.FormatError):
            g2w.unpack(out, None, log=NULL)

    def test_tensor_name_not_utf8_refused(self):
        """a tensor name that is not UTF-8 is a FormatError (REFUSED, exit 1), not a UnicodeDecodeError traceback"""
        model = os.path.join(self.d, 'bad.gguf')
        write_gguf(model, [('zz.weight', (3,), 0)])
        with open(model, 'rb') as f:
            data = f.read()
        with open(model, 'wb') as f:
            f.write(data.replace(b'zz.weight', b'\xff\xfe.weight', 1))
        with open(model, 'rb') as f, self.assertRaisesRegex(wex4.FormatError, 'not UTF-8'):
            g2w.parse(f, os.path.getsize(model))
        self.assertEqual(g2w.main(['', 'info', model]), 1)

    def test_source_size_of_another_type_refused(self):
        """unpack takes each GGUF's size from the container's metadata: a size that is not a JSON integer >= 0
        (inf, 1e3, '32544', True, -1) is refused before anything is built"""
        model = os.path.join(self.d, 'tiny.gguf')
        write_gguf(model, layer(0, 2))
        dumps = g2w.json.dumps
        for size in (float('inf'), 1e3, '32544', True, -1):
            out = os.path.join(self.d, 'm.wex4')

            def patched(obj, *a, **k):
                obj['gguf_sources'][0]['bytes'] = size
                return dumps(obj, *a, **k)
            g2w.json.dumps = patched
            try:
                g2w.pack(out, model, log=NULL)
            finally:
                g2w.json.dumps = dumps
            with open(out, 'rb') as f:
                wex4.Reader(f)   # the container itself is valid: only the metadata lies
            with self.assertRaises(wex4.FormatError, msg=repr(size)):
                g2w.unpack(out, None, log=NULL)
            self.assertEqual(g2w.main(['', 'unpack', out, '--check']), 1, repr(size))
            os.remove(out)


if __name__ == '__main__':
    argv = [sys.argv[0]] + [a for i, a in enumerate(sys.argv[1:], 1) if a != '--verify' and sys.argv[i - 1] != '--verify']
    unittest.main(argv=argv)
