# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""ctypes bindings of wex_nr2_tools.dll and the WEX wire structs (lab only)."""
import ctypes
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
DLL_PATH = HERE / 'wex_nr2_tools.dll'

HEADER = struct.Struct('<IIIIQQII')            # WexHeader, 40 bytes
ENTRY = struct.Struct('<QIII6H6H3I3II')         # WexEntry, 72 bytes
assert HEADER.size == 40 and ENTRY.size == 72
WEX_ALIGN = 4096
WEX_MAGIC_V1, WEX_MAGIC_V3 = 0x57455831, 0x57455833
WEX_HEADER_SCALE_TILES, WEX_HEADER_DEDUP, WEX_HEADER_NEURAL_RANS2 = 1, 4, 16
WEX_ENTRY_NEURAL_RANS2 = 4
WNR2_MAGIC, WNMD_MAGIC = 0x32524e57, 0x444d4e57
MODEL_BYTES = 59408
WEIGHT_BINS, SCALE_BINS = 3 * 567 * 16, 3 * 256
F_OFFSET, F_TOTAL, F_SCALE, F_WEIGHT, F_FLAGS = 0, 1, 2, 3, 22


def align(n, a=WEX_ALIGN):
    return (n + a - 1) // a * a


def fnv1a(data):
    h = 2166136261
    for b in data:
        h = ((h ^ b) * 16777619) & 0xffffffff
    return h or 1


def ptr(buf):
    """Address of a bytes/bytearray/numpy buffer for ctypes (no copy for writable buffers)."""
    if hasattr(buf, 'ctypes'):
        return ctypes.c_void_p(buf.ctypes.data)
    if isinstance(buf, bytes):
        return ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p)
    return ctypes.c_void_p(ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf)))


class Tools:
    def __init__(self, path=DLL_PATH):
        self.lib = lib = ctypes.CDLL(str(path))
        V, I, U32, SZ = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_size_t
        proto = {
            'wex_nr2_tools_abi': (ctypes.c_uint, []),
            'wex_nr2_count_gpu': (I, [V, V, I, V, V]),
            'wex_nr2_count_cpu': (I, [V, V, I, V, V]),
            'wex_nr2_build_model': (I, [V, V, V]),
            'wex_nr2_model_id': (I, [V, V]),
            'wex_nr2_encode_gpu': (I, [V, V, V, I, U32, U32, V, SZ, V]),
            'wex_nr2_encode_cpu': (I, [V, V, V, I, U32, U32, V, SZ, V]),
            'wex_nr2_verify': (I, [V, V, V, SZ, V]),
            'wex_nr2_verify_slab': (I, [V, V, V, V]),
            'wex_nr2_check_container': (I, [V, V, SZ, V, SZ, ctypes.c_uint64, V]),
            'wex_nr2_canonical_zero': (I, [V, V, V]),
            'wex_nr2_release': (None, []),
        }
        for name, (res, args) in proto.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        assert lib.wex_nr2_tools_abi() == 1

    def count(self, raw, entry_bytes, cz, wc, sc, gpu=True):
        fn = self.lib.wex_nr2_count_gpu if gpu else self.lib.wex_nr2_count_cpu
        return fn(ptr(raw), ptr(entry_bytes), int(cz), ptr(wc), ptr(sc))

    def build_model(self, wc, sc):
        blob = bytearray(MODEL_BYTES)
        assert self.lib.wex_nr2_build_model(ptr(wc), ptr(sc), ptr(blob)) == 0
        return bytes(blob)

    def model_id(self, blob):
        out = ctypes.c_uint32()
        rc = self.lib.wex_nr2_model_id(ptr(blob), ctypes.byref(out))
        return out.value if rc == 0 else None

    def encode(self, blob, raw, entry_bytes, cz, weight_chunk, scale_chunk, capacity, gpu=True, out=None):
        """(rc, record bytes or None, written). On a capacity failure the GPU
        encoder reports the size it needs in written."""
        fn = self.lib.wex_nr2_encode_gpu if gpu else self.lib.wex_nr2_encode_cpu
        if out is None:
            out = bytearray(capacity)
        written = ctypes.c_size_t()
        rc = fn(ptr(blob), ptr(raw), ptr(entry_bytes), int(cz), weight_chunk, scale_chunk,
                ptr(out), capacity, ctypes.byref(written))
        return rc, (bytes(memoryview(out)[:written.value]) if rc == 0 else None), written.value

    def verify(self, blob, entry_bytes, record, expected_raw):
        return self.lib.wex_nr2_verify(ptr(blob), ptr(entry_bytes), ptr(record), len(record), ptr(expected_raw))

    def verify_slab(self, blob, entry_bytes, record, expected_raw):
        return self.lib.wex_nr2_verify_slab(ptr(blob), ptr(entry_bytes), ptr(record), ptr(expected_raw))

    def canonical(self, entry_bytes, raw):
        """Canonicalized copy (-0 -> +0) and the number of mapped nibbles."""
        copy = bytearray(raw)
        mapped = ctypes.c_uint64()
        assert self.lib.wex_nr2_canonical_zero(ptr(entry_bytes), ptr(copy), ctypes.byref(mapped)) == 0
        return bytes(copy), mapped.value

    def check_container(self, header_page, pre_table, table, file_bytes):
        out = ctypes.c_uint32()
        rc = self.lib.wex_nr2_check_container(ptr(header_page), ptr(pre_table), len(pre_table),
                                              ptr(table), len(table), file_bytes, ctypes.byref(out))
        return rc, out.value

    def release(self):
        self.lib.wex_nr2_release()


def read_container(path):
    """Header tuple, header page, raw table bytes and the entry tuples of a WEX file."""
    with open(path, 'rb') as f:
        page = f.read(WEX_ALIGN)
        header = HEADER.unpack_from(page)
        count = header[2] * header[3]
        f.seek(header[5])
        table = f.read(count * ENTRY.size)
    entries = [ENTRY.unpack_from(table, i * ENTRY.size) for i in range(count)]
    return header, page, table, entries


def raw_size(entry):
    return entry[F_SCALE] + entry[F_WEIGHT]


def layout_streams(entry, weight_chunk, scale_chunk):
    scale_counts = [entry[4 + 2 * m] * entry[5 + 2 * m] for m in range(3)]
    symbol_counts = [entry[10 + 2 * m] * entry[11 + 2 * m] * 2 for m in range(3)]
    ss = sum(-(-c // scale_chunk) for c in scale_counts)
    ws = sum(-(-c // weight_chunk) for c in symbol_counts)
    return scale_counts, symbol_counts, ss, ws
