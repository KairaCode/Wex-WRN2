# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
import sys; sys.dont_write_bytecode = True
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from v import *
b, rd = load('tiny_raw.wex4')
r0 = rd.index_offset + len(rd.blobs) * 64
print('tiny_raw grid', rd.layers, rd.experts, 'blobs', [x[:2] for x in rd.blobs], 'recs', [(i, r[3], r[4], hex(r[6])) for i, r in enumerate(rd.records)][:10])
show('base', b)
cand = [i for i, r in enumerate(rd.records) if r[4] & 2 and r[6] == i]
print('expert records with owner == index:', cand)
if cand:
    # clearing EXPERT on the LAST such record keeps the others increasing
    i = cand[0]
    show(f'S6b EXPERT flag 2->0 on record {i}', patch(b, [(r0 + i * 72 + 18, '<H', 0)]))
b3, rd3 = load('tiny_nr3_cz.wex4')
print('nr3 blobs', [x[:2] for x in rd3.blobs])
