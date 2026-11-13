# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""GPU neural entropy encoder and independent lossless Python reference."""
import ctypes
from pathlib import Path
import struct

MAGIC = 0x31524e57
CHUNK_SYMBOLS = 1024
CDF_BYTES = 17 * 17 * 2
PREFIX = struct.Struct("<8I")


def validate_model(raw):
    if len(raw) < 512 + CDF_BYTES or (len(raw) - 512) % CDF_BYTES:
        raise ValueError("invalid compiled neural CDF size")
    classes = (len(raw) - 512) // CDF_BYTES
    if classes > 256:
        raise ValueError("too many neural scale classes")
    mapping = struct.unpack_from("<256H", raw)
    if any(c != 65535 and c >= classes for c in mapping):
        raise ValueError("invalid neural scale mapping")
    for offset in range(512, len(raw), 34):
        cdf = struct.unpack_from("<17H", raw, offset)
        if cdf[0] or cdf[-1] != 4096 or any(a >= b for a, b in zip(cdf, cdf[1:])):
            raise ValueError("invalid neural integer CDF")
    return classes, mapping


class Encoder:
    def __init__(self, model_path, dll_path, chunk_symbols=512):
        self.model = Path(model_path).read_bytes()
        self.classes, self.mapping = validate_model(self.model)
        self.scale_only = all(len({self.model[512 + cls * CDF_BYTES + p * 34:512 + cls * CDF_BYTES + (p + 1) * 34]
                                   for p in range(17)}) == 1 for cls in range(self.classes))
        if chunk_symbols not in (256, 512, 1024):
            raise ValueError("neural chunk size must be 256, 512 or 1024 symbols")
        self.chunk_symbols = chunk_symbols
        self.lib = ctypes.CDLL(str(Path(dll_path).resolve()))
        self.fn = self.lib.wex_nr_encode_gpu
        self.fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                            ctypes.POINTER(ctypes.c_uint32)]
        self.fn.restype = ctypes.c_int
        self.verify = self.lib.wex_nr_verify_cpu
        self.verify.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                                ctypes.c_void_p, ctypes.c_uint32]
        self.verify.restype = ctypes.c_int

    def encode(self, scales, weights, scale_stream):
        if len(weights) != len(scales) * 16 or not scales:
            raise ValueError("invalid neural scale/weight size")
        if any(self.mapping[s] == 65535 for s in set(scales)):
            raise ValueError("neural model has no CDF for an expert scale")
        streams = (len(weights) * 2 + self.chunk_symbols - 1) // self.chunk_symbols
        buffer = ctypes.create_string_buffer(len(weights) * 4 + (streams + 1) * 4 + 16)
        written = ctypes.c_uint32()
        status = self.fn(scales, len(scales), weights, len(weights), self.model, self.classes,
                         self.chunk_symbols, buffer, len(buffer), ctypes.byref(written))
        if status:
            raise ValueError(f"CUDA neural entropy encoder failed ({status})")
        framed = buffer.raw[:written.value]
        offset_bytes = (streams + 1) * 4
        offsets, payload = framed[:offset_bytes], framed[offset_bytes:]
        header = PREFIX.pack(MAGIC, len(scale_stream), len(payload), self.classes, streams, self.chunk_symbols, int(self.scale_only), 0)
        record = header + self.model + offsets + scale_stream + payload
        aligned = record + bytes((-len(record)) % 4096)
        if self.verify(aligned, len(aligned), scales, len(scales), weights, len(weights)):
            raise ValueError("neural CUDA output failed independent CPU byte-for-byte verification")
        return record


def parse_record(record, weight_bytes):
    if len(record) < 32:
        raise ValueError("truncated neural record")
    magic, scale_n, payload_n, classes, streams, chunk, reserved0, reserved1 = PREFIX.unpack_from(record)
    if (magic != MAGIC or not 1 <= classes <= 256 or chunk not in (256, 512, 1024) or reserved0 > 1 or reserved1 or
            not scale_n or streams != (weight_bytes * 2 + chunk - 1) // chunk):
        raise ValueError("invalid neural record prefix")
    model_end = 32 + 512 + classes * CDF_BYTES
    metadata_end = model_end + (streams + 1) * 4
    disk = metadata_end + scale_n + payload_n
    if disk > len(record) or (disk + 4095) // 4096 * 4096 != len(record):
        raise ValueError("invalid neural record bounds")
    model = record[32:model_end]
    validate_model(model)
    if reserved0 and any(len({model[512 + c * CDF_BYTES + p * 34:512 + c * CDF_BYTES + (p + 1) * 34]
                              for p in range(17)}) != 1 for c in range(classes)):
        raise ValueError("scale-only neural model has context-dependent CDFs")
    offsets = struct.unpack_from(f"<{streams + 1}I", record, model_end)
    if offsets[0] or offsets[-1] != payload_n or any(b - a < 4 for a, b in zip(offsets, offsets[1:])):
        raise ValueError("invalid neural rANS offsets")
    return model, offsets, record[metadata_end:metadata_end + scale_n], record[metadata_end + scale_n:disk], chunk


def decode_weights(model, offsets, payload, scales, weight_bytes, chunk_symbols=512):
    """Slow independent Python correctness oracle, never needed by runtime."""
    classes, mapping = validate_model(model)
    cdfs = [struct.unpack_from("<17H", model, offset) for offset in range(512, len(model), 34)]
    output = bytearray(weight_bytes)
    for stream, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        state = int.from_bytes(payload[begin:begin + 4], "big"); cursor = begin + 4
        if not 1 << 23 <= state < 1 << 31:
            raise ValueError("invalid neural rANS state")
        previous = 16
        first = stream * chunk_symbols
        for position in range(first, min(first + chunk_symbols, weight_bytes * 2)):
            if not position % 32:
                previous = 16
            cls = mapping[scales[position // 32]]
            if cls >= classes:
                raise ValueError("missing neural scale class")
            cdf = cdfs[cls * 17 + previous]
            slot = state & 4095
            symbol = next(s for s in range(16) if slot < cdf[s + 1])
            state = (cdf[symbol + 1] - cdf[symbol]) * (state >> 12) + slot - cdf[symbol]
            while state < 1 << 23:
                if cursor >= end:
                    raise ValueError("truncated neural rANS payload")
                state = state << 8 | payload[cursor]; cursor += 1
            output[position // 2] |= symbol << (position % 2 * 4)
            previous = symbol
        if cursor != end or state != 1 << 23:
            raise ValueError("invalid neural rANS terminal state")
    return bytes(output)
