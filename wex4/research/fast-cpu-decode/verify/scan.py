# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Read-only: check the other experiment's .rec copies against the production container, scan the entry table,
and (optionally) extract held-out records into verify/out."""
import struct, sys, os, hashlib, random
HEADER = struct.Struct('<IIIIQQII'); WNMD = struct.Struct('<IIQII'); ENTRY = struct.Struct('<QIII6H6H3I3II')
C = r'U:\kaira_wex\v41-reap\v4_flash_experts.nr3.wex'
SRC = sys.argv[1]; OUT = sys.argv[2]
with open(C, 'rb') as f:
    page = f.read(4096)
    magic, version, layers, experts, aligned, table, flags, _ = HEADER.unpack_from(page, 0)
    dmagic, mbytes, moff, mid, _ = WNMD.unpack_from(page, 64)
    print('container', os.path.getsize(C), 'layers', layers, 'experts', experts, 'aligned', aligned, 'model', mbytes, hex(mid))
    f.seek(moff); model = f.read(mbytes)
    print('model.wnm2 equal to experiment copy:', model == open(SRC + '/model.wnm2', 'rb').read())
    f.seek(table); tab = f.read(layers * experts * ENTRY.size)
    ents = [ENTRY.unpack_from(tab, i * ENTRY.size) for i in range(layers * experts)]
    codecs = {}
    for e in ents: codecs[e[-1]] = codecs.get(e[-1], 0) + 1
    print('codecs', codecs)
    raws = {e[2] + e[3] for e in ents}; print('raw sizes', raws)
    shapes = {(e[4:10], e[10:16]) for e in ents}; print('distinct shape sets', len(shapes))
    tot = [e[1] for e in ents]
    raw = 18800640
    print('record bytes: min %d (%.4f) max %d (%.4f) mean %.1f (%.4f) sum %d' % (min(tot), min(tot)/raw, max(tot), max(tot)/raw, sum(tot)/len(tot), sum(tot)/len(tot)/raw, sum(tot)))
    for name in sorted(os.listdir(SRC)):
        if not name.endswith('.rec'): continue
        l, e = name[:-4].split('_'); l = int(l[1:]); e = int(e[1:])
        i = l * experts + e
        data = open(SRC + '/' + name, 'rb').read()
        f.seek(ents[i][0]); rec = f.read(ents[i][1])
        print(name, 'entry equal', data[:72] == tab[i*72:(i+1)*72], 'record equal', data[72:] == rec, 'ratio %.4f' % (ents[i][1]/raw))
    # record header flags of every record (32 bytes each): flags byte 27, chunks
    fl = {}; ch = {}
    for i, e in enumerate(ents):
        f.seek(e[0]); h = f.read(32)
        fl[h[27]] = fl.get(h[27], 0) + 1
        k = struct.unpack_from('<II', h, 16); ch[k] = ch.get(k, 0) + 1
        ents[i] = e + (h[27],)
    print('record flag bytes', {hex(k): v for k, v in fl.items()}, 'chunks', ch)
    order = sorted(range(len(ents)), key=lambda i: ents[i][1])
    insample = {(0,0),(7,100),(13,5),(20,191),(33,64),(39,120)}
    random.seed(20261111 ^ 0x5eed)
    picks = [order[0], order[-1], order[len(order)//2]]
    odd = [i for i, e in enumerate(ents) if e[-1] != 0x0f]
    print('records with flags != 0x0f:', len(odd), [(i // experts, i % experts, hex(ents[i][-1])) for i in odd[:10]])
    picks += odd[:2]
    while len(picks) < 8:
        i = random.randrange(len(ents))
        if (i // experts, i % experts) not in insample and i not in picks: picks.append(i)
    for i in picks:
        l, e = divmod(i, experts)
        f.seek(ents[i][0]); rec = f.read(ents[i][1])
        open(f'{OUT}/H_L{l}_E{e}.rec', 'wb').write(tab[i*72:(i+1)*72] + rec)
        print('held-out', f'H_L{l}_E{e}.rec', ents[i][1], '%.4f' % (ents[i][1]/raw), 'flags', hex(ents[i][-1]))
