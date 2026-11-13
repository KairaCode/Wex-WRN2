"""WEX4 index of the Qwen3.8 MXFP4 expert container, built in memory from its real header page and entry
table (the container is only read): index size, validation time, legacy table reproduced.
    python -I dryrun.py CONTAINER.wex TOOLS_DIR"""
import ctypes, json, os, struct, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'next'))
import wex4, wex4_convert as conv


class Holes:
    def __init__(self, size): self.size, self.at, self.pieces = size, 0, []
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def seek(self, offset, whence=0):
        self.at = offset if whence == 0 else self.size + offset
        return self.at
    def tell(self): return self.at
    def flush(self): pass
    def write(self, data):
        self.pieces.append((self.at, bytes(data))); self.at += len(data); self.size = max(self.size, self.at)
    def read(self, n):
        n = max(0, min(n, self.size - self.at)); out = bytearray(n)
        for start, data in self.pieces:
            lo, hi = max(start, self.at), min(start + len(data), self.at + n)
            if lo < hi: out[lo - self.at:hi - self.at] = data[lo - start:hi - start]
        self.at += n
        return bytes(out)


path, tools = sys.argv[1:3]
C = wex4.CLib(os.path.join(tools, 'wex4_tools.dll'))
size = os.path.getsize(path)
with open(path, 'rb') as f:
    page = f.read(4096)
    _, version, layers, experts, _, table_offset, flags, _ = conv.WEX_HEADER.unpack_from(page)
    f.seek(table_offset)
    table = f.read(layers * experts * 72)
stub = Holes(size)
for at, data in ((0, page), (table_offset, table)):
    stub.seek(at); stub.write(data)
info = conv.upgrade(stub, hash_records=False, family='qwen38')
rd = wex4.Reader(stub, C)
t0 = time.perf_counter()
for _ in range(10):
    assert C.lib.wex4_validate(rd.page, rd.file_bytes, rd.index, len(rd.index)) == 0
ms = (time.perf_counter() - t0) / 10 * 1e3
legacy = ctypes.create_string_buffer(len(table))
assert C.lib.wex4_expert_table(rd.page, rd.file_bytes, rd.index, len(rd.index), legacy) == 0
print(json.dumps(dict(container_bytes=size, wex_version=version, layers=layers, experts=experts, records=info['records'],
                      tensors=len(rd.tensors), index_bytes=info['index_bytes'], added_bytes=info['added'],
                      added_fraction=info['added'] / size, c_validate_ms=round(ms, 1),
                      legacy_table_identical=legacy.raw == table, first_tensor=rd.tensors[0][0].decode()), indent=1))
