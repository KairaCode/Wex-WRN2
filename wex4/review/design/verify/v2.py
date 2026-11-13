# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
import hashlib, io, os, shutil, struct, sys
HERE = os.path.dirname(os.path.abspath(__file__)); NEXT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, NEXT)
import wex4, wex4_convert as conv
C = wex4.CLib(os.path.join(NEXT, 'out', 'wex4_tools_ref.dll'))
P = lambda n: os.path.join(HERE, 'out', n)
def reseal(b):
    o, n = struct.unpack_from('<QQ', b, 32)
    b[96:128] = hashlib.sha256(b[o:o+n]).digest(); b[4064:4096] = hashlib.sha256(b[:4064]).digest(); return b
def both(data, flags=3):
    try: wex4.check(bytes(data), verify=bool(flags & 1), clib=C); py = 'accept'
    except wex4.FormatError as e: py = f'refuse({e})'
    return C.check(bytes(data), flags)[0], py
# D7: expert codec 7 with VALUE_EXACT cleared (reaches the expert-codec rule itself)
t = bytearray(open(os.path.join(NEXT, 'out', 'tiny_nr3_cz.wex4'), 'rb').read()); rd = wex4.Reader(io.BytesIO(bytes(t)))
rec0 = rd.index_offset + len(rd.blobs) * 64
ie = next(i for i, r in enumerate(rd.records) if r[3] == 5)
m = bytearray(t); struct.pack_into('<HH', m, rec0 + ie*72 + 16, 7, 2); reseal(m)
print('D7 expert codec 7, flags=EXPERT only:', both(m, 0))
# D6/D8: required bit 0 set on a file with no value-exact record
r = bytearray(open(os.path.join(NEXT, 'out', 'tiny_raw.wex4'), 'rb').read()); r[8] |= 1; reseal(r)
print('D6/D8 required VALUE_EXACT with only RAW experts:', both(r, 3))
# D8(1): CZ container, clear VALUE_EXACT on ONE WNR3 record in the index only -> open ok? decode of that record?
m = bytearray(t); struct.pack_into('<H', m, rec0 + ie*72 + 18, 2); reseal(m)
print('D8 one WNR3 record without VALUE_EXACT under required bit: open', both(m, 0), 'decode', both(m, 2))
# D3: fsync failing is swallowed
w = P('fs.wex'); shutil.copyfile(os.path.join(NEXT, 'out', 'tiny_cz.wex'), w)
real = os.fsync
def bad(fd): raise OSError(5, 'simulated I/O error')
os.fsync = bad
try: print('D3 upgrade with failing fsync:', 'completed', conv.upgrade(w)['records'])
except Exception as e: print('D3 upgrade aborted', e)
os.fsync = real
# D3: torn header page after upgrade
d = bytearray(open(w, 'rb').read()); d[2048:4096] = bytes(2048); open(w, 'wb').write(d)
for f in (conv.downgrade, conv.upgrade):
    try: f(w); print(' torn page:', f.__name__, 'OK')
    except Exception as e: print(' torn page:', f.__name__, 'refused:', e)
rdlen = len(d); print(' legacy magic still at saved page:', any(d[i:i+4] in (b'3XEW', b'1XEW') for i in range(rdlen-4096*8, rdlen, 4096)))
