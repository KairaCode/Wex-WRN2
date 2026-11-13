# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Design-review probes for WEX4. Writes only under ./out. Run: python -I probe.py"""
import ctypes, hashlib, io, os, random, shutil, struct, sys, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
NEXT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(HERE, 'out')
sys.path.insert(0, NEXT)
import wex4, wex4_convert as conv
C = wex4.CLib(os.path.join(NEXT, 'out', 'wex4_tools_ref.dll'))
P = lambda n: os.path.join(OUT, n)

def reseal(b):
    io_, ib = struct.unpack_from('<QQ', b, 32)
    b[96:128] = hashlib.sha256(b[io_:io_ + ib]).digest()
    b[4064:4096] = hashlib.sha256(b[:4064]).digest()
    return b

def both(data, flags=3):
    try:
        wex4.check(bytes(data), verify=bool(flags & 1), clib=C); py = 'accept'
    except wex4.FormatError as e:
        py = f'refuse({e})'
    st, dg = C.check(bytes(data), flags)
    return st, py, dg.hex()[:16]

rnd = random.Random(1)
A = bytes(rnd.choice(b'aaaaaaabbbc') for _ in range(200000))
recs = lambda d=A: [wex4.Record([wex4.Tensor('t', 'U8', (len(d),), len(d))], len(d), lambda: d, ('bp', 1, 0)),
                    wex4.Record([wex4.Tensor('raw', 'U8', (5000,), 5000)], 5000, lambda: bytes(range(250)) * 20, 'raw')]

print('== 1 same content, two layouts: roots')
r1 = wex4.write(P('a_first.wex4'), recs()); r2 = wex4.write(P('a_last.wex4'), recs(), index_first=False)
d1 = both(open(P('a_first.wex4'), 'rb').read()); d2 = both(open(P('a_last.wex4'), 'rb').read())
print(' roots equal:', r1 == r2, ' content digests equal:', d1[2] == d2[2], d1, d2)
b = bytearray(open(P('a_first.wex4'), 'rb').read()); b[200:216] = b'%PDF-1.7 hidden!'; b[6] = 9; reseal(b)
print(' reserved-area payload + minor 9:', both(b), ' root changed:', bytes(b[4064:4096]) != r1)

print('== 2 ZIP polyglot through the free metadata (index-last layout)')
z = io.BytesIO()
with zipfile.ZipFile(z, 'w') as zf:
    zf.writestr('payload.txt', 'hello from inside a fully verified WEX4 file')
zb = bytearray(z.getvalue())
base_index = 2 * 72 + 2 * 100 + len(b't') + len(b'raw')
pad = wex4.align(base_index + len(zb)) - (base_index + len(zb))
struct.pack_into('<H', zb, len(zb) - 2, pad)            # EOCD comment length = the index zero padding
wex4.write(P('poly.wex4'), recs(), meta=bytes(zb), index_first=False)
data = open(P('poly.wex4'), 'rb').read()
print(' WEX4 verdict (C status, Python):', both(data)[:2])
with zipfile.ZipFile(P('poly.wex4')) as zf:
    print(' zipfile.testzip():', zf.testzip(), ' names:', zf.namelist(), ' content:', zf.read('payload.txt')[:20])

print('== 3 downgrade on a file that was never upgraded but has a kind-16 blob')
fake_page = b'\x00' * 4096
wex4.write(P('k16.wex4'), recs(), blobs=[(16, 0, fake_page)])
before = os.path.getsize(P('k16.wex4')); print(' verifies before:', both(open(P('k16.wex4'), 'rb').read())[:2], 'size', before)
conv.downgrade(P('k16.wex4'))
print(' after downgrade: size', os.path.getsize(P('k16.wex4')), ' first 8 bytes', open(P('k16.wex4'), 'rb').read(8).hex())

print('== 4 forged / damaged record bytes, header+index untouched (root unchanged)')
wex4.write(P('f.wex4'), recs()); f = bytearray(open(P('f.wex4'), 'rb').read())
rd = wex4.Reader(io.BytesIO(bytes(f))); off0, st0 = rd.records[0][:2]; off1 = rd.records[1][0]
orig = both(f, 2)
forged = None
for seed in range(200):
    Bd = bytearray(A); random.Random(seed).shuffle(Bd)
    rec = wex4.bp_encode(bytes(Bd), 1, 12, 0)
    if len(rec) == st0:
        forged = rec; break
g = bytearray(f); g[off0:off0 + st0] = forged
print(' WBP1 record replaced by attacker-encoded different data:')
print('   decode, SHA-256 off (flags=2):', both(g, 2), ' original digest', orig[2])
print('   decode, SHA-256 on  (flags=3):', both(g, 3)[:2])
h = bytearray(f); h[off1 + 100] ^= 1
print(' RAW record, one bit flipped: SHA off', both(h, 2), ' SHA on', both(h, 3)[:2])

print('== 5 unknown codec id 7: expert record vs plain record')
t = None
for cand in ('tiny_nr3.wex4', 'tiny_nr3_cz.wex4', 'st.wex4', 't_py.wex4'):
    tt = bytearray(open(os.path.join(NEXT, 'out', cand), 'rb').read()); rdd = wex4.Reader(io.BytesIO(bytes(tt)))
    cs = [r[3] for r in rdd.records]; print(' ', cand, {c: cs.count(c) for c in set(cs)})
    if 5 in cs and 6 in cs and t is None:
        t, rd = tt, rdd
rec0 = rd.index_offset + len(rd.blobs) * 64
ie = next(i for i, r in enumerate(rd.records) if r[3] == 5); ip = next(i for i, r in enumerate(rd.records) if r[3] == 6)
for label, i in (('EXPERT WNR3 record -> codec 7', ie), ('plain WBP1 record -> codec 7', ip)):
    m = bytearray(t); struct.pack_into('<H', m, rec0 + i * 72 + 16, 7); reseal(m)
    st = C.lib.wex4_check(bytes(m), len(m), 0, None)
    try:
        wex4.Reader(io.BytesIO(bytes(m))); py = 'opens'
    except wex4.FormatError as e:
        py = f'refused at open ({e})'
    print(f' {label}: C open status {st}, Python {py}')

print('== 6 upgrade seals whatever is on disk; no self-check before the switch')
for name, what in (('tiny.wex', 'record byte flipped before upgrade'), ('tiny.wex', 'entry-table shape damaged before upgrade')):
    w = P('up_' + what.split()[0] + '.wex'); shutil.copyfile(os.path.join(NEXT, 'out', name), w)
    with open(w, 'r+b') as fh:
        page = fh.read(4096); table_offset = struct.unpack_from('<Q', page, 24)[0]
        fh.seek(table_offset); e0 = conv.ENTRY.unpack(fh.read(72))
        if what.startswith('record'):
            fh.seek(e0[0] + e0[2] + 10); c = fh.read(1); fh.seek(-1, 1); fh.write(bytes([c[0] ^ 0x10]))
        else:
            fh.seek(table_offset + 20); fh.write(struct.pack('<H', e0[4] + 1))   # scale_shapes[0][0] += 1
    try:
        info = conv.upgrade(w); res = 'upgrade OK'
    except Exception as e:
        res = f'upgrade refused: {e}'
    d = open(w, 'rb').read(); v = both(d) if d[:4] == b'4XEW' else ('still legacy',)
    try:
        conv.downgrade(w); dg = 'downgrade OK'
    except Exception as e:
        dg = f'downgrade refused: {type(e).__name__}: {e}'
    print(f' {what}: {res}; full verify after: {v[:2]}; then {dg}')

print('== 7 arithmetic: effective caps under index <= 2^28')
print(' max tensors, 1 per record, no names:', (1 << 28) // 172, ' with 40-byte names:', (1 << 28) // 212, ' spec says 2^24 =', 1 << 24)
