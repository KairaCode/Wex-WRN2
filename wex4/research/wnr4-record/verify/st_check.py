# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""(1) reference-decoded slab == Z(safetensors source) for two experts, independent of the codec;
(2) chunk-size variants through the UNCHANGED WNR3 encoder/decoder (DLL): sizes vs the prototype's, decode check."""
import ctypes as C, glob, json, os, struct, sys
import numpy as np
model_dir, cache = sys.argv[1:3]
ENT = struct.Struct('<QIII6H6H3I3II')
dll = C.CDLL(os.path.abspath('out/wex_nr3_tools.dll'))
dll.wex_nr3_encode_record.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32,
                                      C.c_char_p, C.c_size_t, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
model = open(os.path.join(cache, 'model.bin'), 'rb').read()
table = open(os.path.join(cache, 'entries.bin'), 'rb').read()
want = [(39, 120), (16, 177)]
names = {f'layers.{l}.ffn.experts.{e}.{m}.{k}': (l, e, m, k) for l, e in want for m in ('w1', 'w2', 'w3') for k in ('scale', 'weight')}
found, read = {}, 0
for shard in sorted(glob.glob(os.path.join(model_dir, '*.safetensors'))):
    with open(shard, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n)); read += 8 + n
        for key in names.keys() & hdr.keys():
            a, b = hdr[key]['data_offsets']
            f.seek(8 + n + a); found[key] = (f.read(b - a), hdr[key]['dtype'], hdr[key]['shape']); read += b - a
assert len(found) == len(names), sorted(names.keys() - found.keys())
print('safetensors bytes read:', read)
slabs = {}
for l, e in want:
    entry = table[(l * 192 + e) * 72:][:72]; en = ENT.unpack(entry)
    raw = en[2] + en[3]
    src = bytearray(raw)
    for i, m in enumerate(('w1', 'w2', 'w3')):
        s = found[f'layers.{l}.ffn.experts.{e}.{m}.scale'][0]; w = found[f'layers.{l}.ffn.experts.{e}.{m}.weight'][0]
        src[en[16 + i]:en[16 + i] + len(s)] = s; src[en[19 + i]:en[19 + i] + len(w)] = w
        assert len(s) == en[4 + 2 * i] * en[5 + 2 * i] and len(w) == en[10 + 2 * i] * en[11 + 2 * i]
    a = np.frombuffer(bytes(src), np.uint8).copy()
    wts = a[en[2]:]
    neg0 = int(((wts & 15) == 8).sum() + ((wts >> 4) == 8).sum())
    wts[(wts & 15) == 8] &= 0xF0; wts[(wts >> 4) == 8] &= 0x0F       # canonical zero, my own
    rec = open(os.path.join(cache, f'rec_{l:02d}_{e:03d}.bin'), 'rb').read()
    slab = C.create_string_buffer(raw)
    assert dll.wex_nr3_decode(model, entry, rec, l, e, slab) == 0
    same = slab.raw == a.tobytes()
    print(f'L{l} E{e}: raw {raw} B, -0 nibbles in source {neg0}, reference decode == Z(safetensors): {same}; == untouched source: {slab.raw == bytes(src)}')
    slabs[(l, e)] = (entry, rec, slab)
# (2) chunk variants via the production codec, one expert
l, e = 16, 177
entry, rec, slab = slabs[(l, e)]
raw = len(slab)
coded, stats = C.create_string_buffer(raw), (C.c_uint32 * 16)()
flags = dll.wex_nr3_prepare(entry, slab, 1, coded, stats)
proto = {(x['layer'], x['expert']): x for x in json.load(open('out/proto_verify.json'))['experts']}[(l, e)]
cfgn = [c['name'] for c in json.load(open('out/proto_verify.json'))['configs']]
for name, wc, sc in (('base_2048_512', 2048, 512), ('w4096', 4096, 512), ('w8192', 8192, 512), ('w16384', 16384, 512),
                     ('s1024', 2048, 1024), ('s2048', 2048, 2048), ('w4096_s1024', 4096, 1024), ('w8192_s2048', 8192, 2048)):
    out, used, total = C.create_string_buffer(raw * 2), C.c_size_t(), C.c_size_t()
    assert dll.wex_nr3_encode_record(model, entry, coded, flags, l << 16 | e, wc, sc, out, raw * 2, C.byref(used), C.byref(total)) == 0
    sized = entry[:8] + struct.pack('<I', total.value) + entry[12:]
    ok = dll.wex_nr3_verify(model, sized, out, l, e, slab) == 0
    pu = proto['cfg'][cfgn.index(name)][1]
    print(f'{name:14s} WNR3 codec used {used.value}  prototype used {pu}  diff {pu - used.value:+d}  reference decode ok {ok}  saved vs production {len(rec) and proto["cfg"][0][1] - used.value}')
