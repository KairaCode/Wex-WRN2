"""Read back a built WNR3 container from disk and check every record without re-reading WEX1.

usage: python -I readback.py TOOLS.dll CONTAINER.wex REPORT.json
For each table entry: codec 5 -> wex_nr3_decode the record as stored on disk, sha256 of the decoded slab must equal
the builder's sha256_expected (the canonical-zero WEX1 slab it verified against); codec 0 -> counted, not checked.
Also hashes the whole file (sha256) for the .verified marker. Prints one summary line; exit 1 on any mismatch."""
import concurrent.futures as cf
import ctypes as C
import hashlib
import json
import os
import struct
import sys
import threading

# build_nr3.py lives in nr3/ of this repo; NR3_TOOLS_DIR overrides
sys.path.insert(0, os.environ.get('NR3_TOOLS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'nr3')))
import build_nr3 as B  # noqa: E402

tools = B.Tools(sys.argv[1])
tools.lib.wex_nr3_decode.argtypes = [C.c_void_p] * 3 + [C.c_uint32, C.c_uint32, C.c_void_p]
path, rows = sys.argv[2], json.load(open(sys.argv[3]))['rows']
want = {(r['layer'], r['expert']): r for r in rows}

with open(path, 'rb') as f:
    page = f.read(B.ALIGN)
    magic, ver, layers, experts, maximum, table_offset, flags, _ = B.HEADER.unpack_from(page, 0)
    assert magic == B.WEX_MAGIC_V3 and ver == 3 and flags == 97, (hex(magic), ver, flags)
    f.seek(B.ALIGN)
    blob = f.read(B.MODEL_BYTES)
    f.seek(table_offset)
    table = [B.ENTRY.unpack(f.read(B.ENTRY.size)) for _ in range(layers * experts)]

local = threading.local()


def check(i):
    L, E = divmod(i, experts)
    e = table[i]
    if e[-1] != B.CODEC_NR3:
        return 'raw', None
    if not hasattr(local, 'f'):
        local.f = open(path, 'rb')
    local.f.seek(e[0])
    rec = local.f.read(e[1])
    r = want[(L, E)]
    out = C.create_string_buffer(r['raw'])
    if tools.lib.wex_nr3_decode(blob, B.entry_bytes(e, 0, e[1], e[-1]), rec, L, E, out) != 0:
        return 'bad', f'L{L} E{E} decode rejected'
    if hashlib.sha256(out.raw).hexdigest() != r['sha256_expected']:
        return 'bad', f'L{L} E{E} sha256 differs'
    return 'ok', None


with cf.ThreadPoolExecutor(8) as pool:
    res = list(pool.map(check, range(layers * experts)))
bad = [m for s, m in res if s == 'bad']
h = hashlib.sha256()
with open(path, 'rb') as f:
    while chunk := f.read(64 << 20):
        h.update(chunk)
print(json.dumps({'entries': len(res), 'ok': sum(s == 'ok' for s, _ in res), 'raw': sum(s == 'raw' for s, _ in res),
                  'bad': len(bad), 'first_bad': bad[:5], 'model_id_header': hex(struct.unpack_from('<I', page, 80)[0]),
                  'sha256': h.hexdigest()}))
sys.exit(1 if bad or len(res) != len(rows) else 0)
