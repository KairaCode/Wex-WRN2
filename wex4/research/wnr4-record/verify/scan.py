# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Independent re-scan of every record header + u16 length table (read-only), compared with ../out/container_scan.json.

  python -I scan.py CONTAINER ../out/container_scan.json out/scan_verify.json
"""
import json, os, struct, sys
import numpy as np

path, theirs_path, out_path = sys.argv[1:4]
size = os.path.getsize(path)
f = open(path, 'rb')
page = f.read(4096)
magic, version, L, E, aligned, table_off, hflags, _ = struct.unpack_from('<IIIIQQII', page, 0)
dm, model_bytes, model_off, model_id, _ = struct.unpack_from('<IIQII', page, 64)
f.seek(table_off)
table = f.read(L * E * 72)
read = 4096 + len(table)
ENT = struct.Struct('<QIII6H6H3I3II')
BITLEN = np.array([int(x).bit_length() for x in range(65536)], np.int64)   # integer, no float log2

tot = used_sum = pad = lenc32 = lenc32_permatrix = 0
al = {512: 0, 64: 0, 4: 0, 1: 0}
mism = 0
theirs = json.load(open(theirs_path))['rows']
chunks, flagset, minlen, maxlen, max_off_end = set(), set(), 1 << 30, 0, 0
lenc_list = []
for i in range(L * E):
    e = ENT.unpack_from(table, i * 72)
    off, total = e[0], e[1]
    sshape, wshape, codec = e[4:10], e[10:16], e[-1]
    assert codec == 5
    f.seek(off)
    head = f.read(32)
    mg, mid, sp, wp, wch, sch, w24, owner = struct.unpack('<8I', head)
    assert mg == 0x33524E57 and mid == model_id and owner == ((i // E) << 16 | (i % E))
    parts = [(sshape[2 * m] * sshape[2 * m + 1] + sch - 1) // sch for m in range(3)] + \
            [(wshape[2 * m] * wshape[2 * m + 1] * 2 + wch - 1) // wch for m in range(3)]
    n = sum(parts)
    ln = np.frombuffer(f.read(2 * n), '<u2').astype(np.int64)
    read += 32 + 2 * n
    assert int(ln.sum()) == sp + wp
    u = 32 + 2 * n + sp + wp
    assert (u + 4095) // 4096 * 4096 == total and off % 4096 == 0
    chunks.add((wch, sch)); flagset.add(w24 >> 24)
    minlen = min(minlen, int(ln.min())); maxlen = max(maxlen, int(ln.max()))
    max_off_end = max(max_off_end, off + total)
    # coded table, groups of 32 over the whole table in stream order
    g = 0
    for a in range(0, n, 32):
        blk = ln[a:a + 32]
        g += 3 + (len(blk) * int(BITLEN[int(blk.max() - blk.min())]) + 7) // 8
    lenc_list.append(g)
    lenc32 += g
    tot += total; used_sum += u; pad += total - u
    for a in al:
        al[a] += total - (u + a - 1) // a * a
    t = theirs[i]
    if t[2] != total or t[3] != u or t[4] != n or t[7] != g:
        mism += 1
res = {
    'file_size': size, 'records': L * E, 'bytes_read': read, 'header_flags': hflags,
    'record_bytes_sum': tot, 'used_bytes_sum': used_sum, 'padding_sum': pad,
    'mean_total': tot / (L * E), 'mean_used': used_sum / (L * E), 'mean_used_of_raw': used_sum / (L * E) / 18800640,
    'mean_total_of_raw': tot / (L * E) / 18800640,
    'saved_by_alignment': al, 'chunks': sorted(chunks), 'record_flags': sorted(flagset),
    'min_stream_len': minlen, 'max_stream_len': maxlen, 'last_record_end': max_off_end,
    'lenc_g32_sum': lenc32, 'lenc_g32_mean': lenc32 / (L * E), 'lenc_g32_min': min(lenc_list), 'lenc_g32_max': max(lenc_list),
    'lenc_g32_saved_sum_vs_u16': 2 * 19440 * L * E - lenc32,
    'lenc_g32_saved_sum_net_of_8B_header': 2 * 19440 * L * E - lenc32 - 8 * L * E,
    'rows_mismatching_their_scan': mism,
}
json.dump(res, open(out_path, 'w'), indent=1)
print(json.dumps(res, indent=1))
