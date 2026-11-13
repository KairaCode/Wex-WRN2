# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Decode EVERY record of a WNR2 sample container through the CUDA helper (decode_to_host) and
compare it with the WEX1 raw record; --zero compares against WEX1 with the -0 code mapped to +0.
usage: check_all.py helper.dll v1.wex v3-nr2.wex [--zero] [--json out.json]"""
import ctypes, json, struct, sys, time
from pathlib import Path
import numpy as np

args = sys.argv[1:]
if len([a for a in args if not a.startswith('--')]) < 3 or '-h' in args or '--help' in args:
    sys.exit(__doc__)
zero ='--zero' in args
out = args[args.index('--json') + 1] if '--json' in args else None
helper, v1, v3 = args[:3]
HEADER = struct.Struct('<IIIIQQII')
lib = ctypes.CDLL(helper)
lib.wex_cuda_helper_features.restype = ctypes.c_uint
lib.wex_cuda_register_model.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
lib.wex_cuda_decode_to_host.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]

def manifest(path):
    f = open(path, 'rb'); page = f.read(4096)
    magic, version, layers, experts, aligned, table, flags, _ = HEADER.unpack_from(page)
    f.seek(table); raw = f.read(layers * experts * 72)
    return f, page, aligned, flags, [raw[i * 72:(i + 1) * 72] for i in range(layers * experts)]

f1, _, aligned1, _, e1 = manifest(v1)
f3, page, aligned3, flags3, e3 = manifest(v3)
assert aligned1 == aligned3 and flags3 & 16
d = page[64:88]
assert struct.unpack_from('<I', d)[0] == 0x444D4E57
size, offset, model_id = struct.unpack_from('<IQI', d, 4)
f3.seek(offset); blob = f3.read(size)
assert lib.wex_cuda_helper_features() & 4
assert lib.wex_cuda_register_model(blob, size) == 0
slab = ctypes.create_string_buffer(aligned3)
nibble_zero = np.arange(256, dtype=np.uint8)
lo, hi = nibble_zero & 15, nibble_zero >> 4
nibble_zero = (np.where(lo == 8, 0, lo) | (np.where(hi == 8, 0, hi) << 4)).astype(np.uint8)
stats = dict(records=0, exact=0, mismatch=0, failed=0, raw_fallback=0, seconds_decode=0.0)
t0 = time.time()
for i, (a, b) in enumerate(zip(e1, e3)):
    off3, total3, sb, wb = struct.unpack_from('<QIII', b)
    flags = struct.unpack_from('<I', b, 68)[0]
    if not total3: continue
    off1, total1 = struct.unpack_from('<QI', a)
    f1.seek(off1); ref = f1.read(sb + wb)
    if zero:
        r = np.frombuffer(ref, np.uint8).copy(); r[sb:] = nibble_zero[r[sb:]]; ref = r.tobytes()
    f3.seek(off3); rec = f3.read(total3)
    stats['records'] += 1
    if flags == 0:   # stored raw
        stats['raw_fallback'] += 1
        stats['exact' if rec[:sb + wb] == ref else 'mismatch'] += 1
        continue
    raw = sb + wb; aligned_raw = (raw + 4095) // 4096 * 4096
    ctypes.memset(slab, 0, aligned3)
    ctypes.memmove(ctypes.addressof(slab) + aligned_raw - total3, rec, total3)
    t = time.perf_counter(); status = lib.wex_cuda_decode_to_host(b, slab, aligned3); stats['seconds_decode'] += time.perf_counter() - t
    if status: stats['failed'] += 1; print('failed', i, status, flush=True); continue
    ok = ctypes.string_at(slab, raw) == ref
    stats['exact' if ok else 'mismatch'] += 1
    if not ok: print('MISMATCH', i, flush=True)
stats['seconds_total'] = round(time.time() - t0, 1)
stats['decode_to_host_mean_ms'] = round(stats['seconds_decode'] / max(1, stats['records'] - stats['raw_fallback']) * 1e3, 3)
stats['model_id'] = f'{model_id:08x}'
print(json.dumps(stats), flush=True)
if out: Path(out).write_text(json.dumps(stats, indent=2))
