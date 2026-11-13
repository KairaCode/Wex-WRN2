# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Review probes: grouping/split round trips, Reader exceptions on resealed mutations, CLI. python -I t3.py"""
import os, sys, json, struct, io, hashlib, shutil, random, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1 import run, st, st_raw, p, verdicts, OUT, NEXT, C, wex4, conv, exp, pack_then_read

rnd = random.Random(7)
print('--- L. round trips: groups that flush at 4 MiB, split parts of odd size, zero-size, boundary sizes')
tensors = []
for i in range(150):   # ~60 KiB tensors of 4 widths: each width group passes 4 MiB at least once? (150*60k/4 ~ 2.2 MiB) -> add more below
    dt, w = [('U8', 1), ('BF16', 2), ('F32', 4), ('F64', 8)][i % 4]
    n = (60000 // w) - (i % 3)
    tensors.append((f't{i:03d}', dt, (n,), rnd.randbytes(n * w // 2) + bytes(n * w - n * w // 2)))
for i in range(80):
    tensors.append((f'u{i:03d}', 'U8', (65535,), rnd.randbytes(65535)))          # just under SMALL: 80 * 64K > 4 MiB
tensors += [('edge64k', 'I16', (32768,), rnd.randbytes(65536)), ('zero', 'F32', (0, 7), b''), ('scalar', 'F64', (), bytes(8)),
            ('big', 'F32', (50001,), rnd.randbytes(200004)), ('nib', 'F4', (2, 3), b'\x12\x34\x56'),
            ('c64', 'C64', (9000,), rnd.randbytes(72000)), ('opaque', 'WHAT', (3,), b'xyz')]
rnd.shuffle(tensors)
s = st(p('l.safetensors'), tensors, {'k': 'v'})
for part in (70001, 65536, 1 << 20):
    out = p('l.wex4')
    conv.pack(out, [s], C, part_bytes=part)
    back = p('l_back'); shutil.rmtree(back, ignore_errors=True)
    res = conv.unpack(out, back, C)
    same = open(s, 'rb').read() == open(os.path.join(back, 'l.safetensors'), 'rb').read()
    with open(out, 'rb') as f:
        rd = wex4.Reader(f, C)
        n_rec, big = len(rd.records), max(r[2] for r in rd.records)
    print(f'    part {part}: {res}, byte-identical {same}, readers {verdicts(out)}, {n_rec} records, largest decoded {big}')

print('--- M. expert name regex: "$" also matches before a trailing newline')
pack_out = pack_then_read('m_newline', st(p('m.st'), exp(0, 0, extra=[('layers.0.ffn.experts.0.w2.scale\n', 'F8_E8M0', (4, 2), b'\x7f' * 8)])),
                          experts='raw')
if pack_out:
    with open(pack_out, 'rb') as f:
        rd = wex4.Reader(f)
        print('    tensors stored:', len(rd.tensors), 'of 7 in the source;',
              [n for n in (b'layers.0.ffn.experts.0.w2.scale', b'layers.0.ffn.experts.0.w2.scale\n') if n not in rd.by_name], 'missing')

print('--- N. Python Reader on header/index mutations with the hashes recomputed (exceptions other than FormatError, C disagreement)')
for name in ('t_py.wex4', 'up_tiny2_cz.wex'):
    base = open(os.path.join(NEXT, 'out', name), 'rb').read()
    if base[:4] != b'4XEW':
        tmp = p('n_' + name); open(tmp, 'wb').write(base); conv.upgrade(tmp); base = open(tmp, 'rb').read()
    index_offset, index_bytes = struct.unpack_from('<QQ', base, 32)
    odd = disagree = accepted = 0
    N = 4000
    for it in range(N):
        blob = bytearray(base)
        for _ in range(rnd.choice((1, 1, 2, 3))):
            if rnd.random() < 0.3:
                at = rnd.randrange(0, 128)
            else:
                at = index_offset + rnd.randrange(index_bytes)
            blob[at] = rnd.choice((0, 1, 0xff, blob[at] ^ (1 << rnd.randrange(8)), rnd.randrange(256)))
        io_, ib = struct.unpack_from('<QQ', blob, 32)
        if io_ + ib <= len(blob):
            blob[96:128] = hashlib.sha256(blob[io_:io_ + ib]).digest()
        blob[4064:4096] = hashlib.sha256(blob[:4064]).digest()
        try:
            digest = wex4.check(bytes(blob), True, C)
            py = 0
        except wex4.FormatError:
            py = 1
        except BaseException as e:
            odd += 1
            if odd <= 3:
                print('    UNCAUGHT', type(e).__name__, str(e)[:100])
            continue
        status, cd = C.check(bytes(blob), 3)
        accepted += not py
        if (status != 0) != bool(py) or (not py and cd != digest):
            disagree += 1
            if disagree <= 3:
                print('    DISAGREE python', 'refuses' if py else 'accepts', 'C status', status)
    print(f'    {name}: {N} mutations, {accepted} accepted by both, {odd} non-FormatError exceptions, {disagree} disagreements')

print('--- O. command line')
exe = [sys.executable, '-I', os.path.join(NEXT, 'wex4_convert.py')]
for args in (['pack', p('o.wex4')], ['unpack', p('l.wex4')], ['pack', p('o.wex4'), s, '--part-mib', '0'],
             ['pack', p('o.wex4'), s, '--dll'], ['upgrade', p('a1.st')], ['pack', p('o.wex4'), p('a3.st')]):
    r = subprocess.run(exe + args, capture_output=True, text=True)
    last = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or [''])[-1]
    print(f'    {" ".join(os.path.basename(a) for a in args)} -> exit {r.returncode}: {last[:110]}')
