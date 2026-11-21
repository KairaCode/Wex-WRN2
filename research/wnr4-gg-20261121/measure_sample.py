# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""WNR4-GG on sampled GGUF expert records (wnr4/tests/sample_gg_experts.py writes SAMPLE_DIR): train on the even
experts, code the odd ones, decode them back on 8 threads, compare. Stored/raw against the raw record aligned to 4096 B.
    python -I measure_sample.py SAMPLE_DIR wnr4_gg_tools.dll [G]   (G default 128)"""
import ctypes as C
import json
import os
import sys
import time

S, DLL = sys.argv[1], sys.argv[2]
G = int(sys.argv[3]) if len(sys.argv) > 3 else 128
lib = C.CDLL(os.path.abspath(DLL))
P = C.c_void_p
lib.wnr4gg_counts_words.restype = C.c_uint64
lib.wnr4gg_model_bytes.restype = C.c_uint32
lib.wnr4gg_count.argtypes = [C.c_uint32, P, P, C.c_char_p, P]
lib.wnr4gg_build.restype = C.c_int
lib.wnr4gg_build.argtypes = [P, C.c_char_p]
lib.wnr4gg_model_open.restype = P
lib.wnr4gg_model_open.argtypes = [C.c_char_p, C.c_uint64]
lib.wnr4gg_model_id.argtypes = [P]
lib.wnr4gg_model_id.restype = C.c_uint32
lib.wnr4gg_encode.restype = C.c_uint64
lib.wnr4gg_encode.argtypes = [P, C.c_uint32, P, P, C.c_char_p, C.c_uint32, C.c_uint32, C.c_char_p, C.c_uint64]
lib.wnr4gg_decode.restype = C.c_uint32
lib.wnr4gg_decode.argtypes = [P, C.c_uint32, P, P, C.c_char_p, C.c_uint64, C.c_uint32, C.c_char_p, C.c_int]

man = json.load(open(os.path.join(S, 'manifest.json')))
recs = []
for e in man['experts']:
    data = open(os.path.join(S, e['file']), 'rb').read()
    types = (C.c_uint32 * 4)(*[s['type'] for s in e['segments']])
    sizes = (C.c_uint32 * 4)(*[s['bytes'] for s in e['segments']])
    recs.append((e, data, types, sizes, len(e['segments'])))
counts = (C.c_uint64 * lib.wnr4gg_counts_words())()
for e, data, types, sizes, n in recs:
    if e['expert'] % 2 == 0:
        assert lib.wnr4gg_count(n, types, sizes, data, counts) == 0
blob = C.create_string_buffer(lib.wnr4gg_model_bytes())
assert lib.wnr4gg_build(counts, blob) == 0, 'model build'
model = lib.wnr4gg_model_open(blob.raw, len(blob.raw))
assert model, 'model open'
align = lambda x: (x + 4095) // 4096 * 4096
by_type, tot_raw, tot_st, fails, t_enc, t_dec = {}, 0, 0, 0, 0.0, 0.0
for e, data, types, sizes, n in recs:
    if e['expert'] % 2 == 0:
        continue
    cap = len(data) * 2 + (1 << 20)
    out = C.create_string_buffer(cap)
    owner = e['layer'] << 16 | e['expert']
    t0 = time.time(); m = lib.wnr4gg_encode(model, n, types, sizes, data, owner, G, out, cap); t_enc += time.time() - t0
    assert m, 'encode'
    back = C.create_string_buffer(len(data))
    t0 = time.time(); err = lib.wnr4gg_decode(model, n, types, sizes, out.raw[:m], m, owner, back, 8); t_dec += time.time() - t0
    fails += err != 0 or back.raw != data
    tot_raw += align(len(data)); tot_st += m
    key = '+'.join(str(s['type']) for s in e['segments'])
    a = by_type.setdefault(key, [0, 0, 0])
    a[0] += align(len(data)); a[1] += m; a[2] += 1
print(f'G={G} records {sum(v[2] for v in by_type.values())} fails {fails} stored/raw {tot_st / tot_raw:.4f}  '
      f'encode {t_enc:.1f}s decode {t_dec:.1f}s')
for k, (r, s, n) in sorted(by_type.items()):
    print(f'  types {k}: {n} records, stored/raw {s / r:.4f}')
print('model id', hex(lib.wnr4gg_model_id(model)), 'blob bytes', len(blob.raw))
