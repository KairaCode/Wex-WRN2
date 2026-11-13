# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Read-only copy of chosen records (+ header page, model, entry table) into out/cache for the verification rerun."""
import json, os, struct, sys
path, cache, theirs = sys.argv[1:4]
rows = json.load(open(theirs))['rows']
big = max(rows, key=lambda r: r[3]); small = min(rows, key=lambda r: r[3]); worst_lenc = max(rows, key=lambda r: r[7])
fresh = [(l, (53 * l + 97) % 192) for l in (0, 3, 5, 8, 11, 13, 16, 19, 21, 24, 27, 29, 32, 35, 37, 39)]
their = {(l, (37 * l + 11) % 192) for l in range(40)} | {(39, 120)}
assert not (set(fresh) & their)
overlap = [(0, 11), (20, 175), (39, 110), (39, 120)]
assert all(o in their for o in overlap)
extreme = [(big[0], big[1]), (small[0], small[1]), (worst_lenc[0], worst_lenc[1])]
picks = []
for p in fresh + overlap + extreme:
    if p not in picks: picks.append(p)
f = open(path, 'rb')
page = f.read(4096)
_, _, L, E, _, toff, _, _ = struct.unpack_from('<IIIIQQII', page, 0)
_, mb, moff, _, _ = struct.unpack_from('<IIQII', page, 64)
f.seek(moff); model = f.read(mb); f.seek(toff); table = f.read(L * E * 72)
os.makedirs(cache, exist_ok=True)
open(os.path.join(cache, 'header.bin'), 'wb').write(page)
open(os.path.join(cache, 'model.bin'), 'wb').write(model)
open(os.path.join(cache, 'entries.bin'), 'wb').write(table)
read = 4096 + len(model) + len(table)
for l, e in picks:
    off, total = struct.unpack_from('<QI', table, (l * E + e) * 72)
    out = os.path.join(cache, f'rec_{l:02d}_{e:03d}.bin')
    if not os.path.exists(out):
        f.seek(off); d = f.read(total); read += len(d); open(out, 'wb').write(d)
print('extreme (largest, smallest, worst lenc):', extreme, 'fresh in their sample:', [p for p in extreme if p in their])
print('bytes read:', read)
print(' '.join(f'{l}:{e}' for l, e in picks))
open(os.path.join(cache, 'picks.txt'), 'w').write(' '.join(f'{l}:{e}' for l, e in picks))
