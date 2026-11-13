# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Review probes: unpack with hostile metadata, upgrade / downgrade. python -I t2.py (writes only under ./out)"""
import os, sys, json, struct, io, hashlib, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1 import run, st, st_raw, p, verdicts, OUT, NEXT, C, wex4, conv

FIX = os.path.join(NEXT, 'out')


def container(label, meta, data=b'abcd', name=b'x', optional=1):
    out = p(label + '.wex4')
    rec = wex4.Record([wex4.Tensor(name, 'U8', (len(data),), len(data))], len(data), lambda: data)
    wex4.write(out, [rec], meta=meta if isinstance(meta, bytes) else json.dumps(meta).encode(), optional=optional)
    return out


def unpack(label, meta, **k):
    out = container(label, meta, **k)
    back = p(label + '_back')
    shutil.rmtree(back, ignore_errors=True)
    r = run(label + ' unpack', conv.unpack, out, back)
    print('    out dir now holds:', sorted(os.listdir(back)) if os.path.isdir(back) else None)
    return back, r


def src(name, header, sha=None, payload=b'abcd'):
    raw = header if isinstance(header, str) else json.dumps(header, separators=(',', ':'))
    full = struct.pack('<Q', len(raw.encode())) + raw.encode() + payload
    return {'name': name, 'bytes': len(full), 'sha256': sha or hashlib.sha256(full).hexdigest(), 'header': raw}


GOOD = {'x': {'dtype': 'U8', 'shape': [4], 'data_offsets': [0, 4]}}

print('--- E. unpack: hostile "sources" metadata')
unpack('e0_good', {'wex4_meta': 1, 'sources': [src('ok.safetensors', GOOD)]})
unpack('e1_not_json', b'\xff\xfe not json')
unpack('e2_bad_json', b'{')
unpack('e3_list', [])
unpack('e4_no_sources', {'wex4_meta': 1})
unpack('e5_no_header', {'sources': [{'name': 'a', 'sha256': ''}]})
unpack('e6_name_int', {'sources': [dict(src('a', GOOD), name=5)]})
unpack('e7_offsets_str', {'sources': [src('a', {'x': {'dtype': 'U8', 'shape': [4], 'data_offsets': 'zz'}})]})
unpack('e8_header_not_object', {'sources': [src('a', '[1,2]')]})
unpack('e9_deep', b'{"sources":' + b'[' * 100000 + b']' * 100000 + b'}')
unpack('e10_dotdot', {'sources': [src('..', GOOD)]})
unpack('e11_empty_name', {'sources': [src('sub/', GOOD)]})
unpack('e12_traversal', {'sources': [src('..\\..\\..\\evil.safetensors', GOOD)]})
print('--- F. unpack: header that does not describe the container, with a matching recorded sha256')
lie = {'x': {'dtype': 'F32', 'shape': [1000, 1000], 'data_offsets': [0, 4000000]}}
back, r = unpack('f1_lying_header', {'sources': [src('lie.safetensors', lie)]})
if r:
    run('    st_index on the rebuilt file', conv.st_index, os.path.join(back, 'lie.safetensors'))
back, r = unpack('f2_dup_tensor_twice', {'sources': [src('twice.safetensors', {'x': GOOD['x'], '__metadata__': {}}, payload=b'abcd'),
                                                     src('twice.safetensors', GOOD, payload=b'abcd')]})
print('--- G. unpack: sha256 mismatch on a non value-exact container')
back, r = unpack('g1_wrong_sha', {'sources': [src('bad.safetensors', GOOD, sha='00' * 32)]})
print('    bad.safetensors left under its final name:', os.path.exists(os.path.join(back, 'bad.safetensors')))
print('--- H. unpack: source name = the container itself, unpacked next to it')
os.makedirs(p('h'), exist_ok=True)
c = container('h_self', {'sources': [src('victim.wex4', GOOD)]})
victim = os.path.join(p('h'), 'victim.wex4')
shutil.copyfile(c, victim)
before = hashlib.sha256(open(victim, 'rb').read()).hexdigest()
run('h unpack into own folder', conv.unpack, victim, p('h'))
print('    container unchanged:', hashlib.sha256(open(victim, 'rb').read()).hexdigest() == before, sorted(os.listdir(p('h'))))
os.makedirs(p('h2'), exist_ok=True)
open(os.path.join(p('h2'), 'notes.txt'), 'w').write('user file')
c = container('h2', {'sources': [src('notes.txt', GOOD)]})
run('h2 unpack over an existing unrelated file', conv.unpack, c, p('h2'))
print('    notes.txt now:', open(os.path.join(p('h2'), 'notes.txt'), 'rb').read()[:30])

print('--- I. upgrade / downgrade')
open(p('i_empty.wex'), 'wb').close()
run('upgrade empty file', conv.upgrade, p('i_empty.wex'))
shutil.copyfile(os.path.join(FIX, 't_py.wex4'), p('i_w4.wex'))
run('upgrade a WEX4 file', conv.upgrade, p('i_w4.wex'))
run('downgrade a packed WEX4 file', conv.downgrade, p('i_w4.wex'))
shutil.copyfile(os.path.join(FIX, 'tiny_cz.wex'), p('i_twice.wex'))
orig = open(p('i_twice.wex'), 'rb').read()
run('upgrade', conv.upgrade, p('i_twice.wex'))
run('upgrade again', conv.upgrade, p('i_twice.wex'))
# damaged saved page
blob = bytearray(open(p('i_twice.wex'), 'rb').read())
blob[len(orig) + 100] ^= 1
open(p('i_dam.wex'), 'wb').write(blob)
run('downgrade, saved page damaged', conv.downgrade, p('i_dam.wex'))
print('    file untouched:', open(p('i_dam.wex'), 'rb').read() == bytes(blob))
run('downgrade', conv.downgrade, p('i_twice.wex'))
print('    byte-identical to the original:', open(p('i_twice.wex'), 'rb').read() == orig)
huge = bytearray(orig[:4096]); struct.pack_into('<II', huge, 8, 0xffffffff, 0xffffffff)
open(p('i_huge.wex'), 'wb').write(huge + orig[4096:])
run('upgrade, 2^32-1 x 2^32-1 grid', conv.upgrade, p('i_huge.wex'))


class Torn(io.BytesIO):
    """Fails at the write that starts at offset 0 (step 2), like a crash between the two writes."""
    def write(self, b):
        if self.tell() == 0 and self.getbuffer().nbytes > 4096:
            raise OSError('power cut')
        return super().write(b)
    def close(self):
        self.saved = self.getvalue()
        super().close()


for name in ('tiny.wex', 'tiny_cz.wex'):
    data = open(os.path.join(FIX, name), 'rb').read()
    f = Torn(data)
    run(f'{name}: upgrade interrupted before the header write', conv.upgrade, f)
    torn = f.saved
    print(f'    old bytes intact: {torn[:len(data)] == data}; appended {len(torn) - len(data)} B')
    open(p('i_torn_' + name), 'wb').write(torn)
    run('    run again', conv.upgrade, p('i_torn_' + name))
    open(p('i_torn2_' + name), 'wb').write(torn[:len(data) + 5000])
    run('    run again after a torn append (not a page multiple)', conv.upgrade, p('i_torn2_' + name))

print('--- J. upgrade of a legacy-valid WEX1 file that the WEX4 rules refuse (1 layer x 65537 experts)')
layers, experts = 1, 65537
table = bytearray(layers * experts * 72)
first = 4096 + wex4.align(len(table))
slab = bytes(range(16 + 16)) * 0 + bytes(3 * 1 * 1) + bytes([0x11]) * (3 * 16)      # 3 scales 1x1, 3 weights 1x16
struct.pack_into('<QIII6H6H3I3II', table, 0, first, 4096, 3, 48, 1, 1, 1, 1, 1, 1, 1, 16, 1, 16, 1, 16, 0, 1, 2, 3, 19, 35, 0)
head = struct.pack('<IIIIQQII', 0x57455831, 1, layers, experts, 4096, 4096, 0, 0)
legacy = head + bytes(4096 - len(head)) + table + bytes(wex4.align(len(table)) - len(table)) + slab + bytes(4096 - len(slab))
open(p('j_wide.wex'), 'wb').write(legacy)
run('upgrade', conv.upgrade, p('j_wide.wex'))
print('    readers after upgrade:', verdicts(p('j_wide.wex')))
run('downgrade', conv.downgrade, p('j_wide.wex'))
now = open(p('j_wide.wex'), 'rb').read()
print('    file is the original again:', now == legacy, '; magic now', now[:4])

print('--- K. downgrade of a crafted WEX4 file with a kind-16 blob')
rec = wex4.Record([wex4.Tensor(b'w', 'U8', (8192,), 8192)], 8192, lambda: b'\x55' * 8192)
wex4.write(p('k.wex4'), [rec], blobs=[(16, 0, b'JUNK' + bytes(4092))])
size = os.path.getsize(p('k.wex4'))
run('downgrade', conv.downgrade, p('k.wex4'))
print(f'    size {size} -> {os.path.getsize(p("k.wex4"))}; first bytes {open(p("k.wex4"), "rb").read(4)}')
