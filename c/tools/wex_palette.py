# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Experimental three-bit palettes over native E2M1 symbols, conditioned on UE8M0.

The packing convention is shared with KAIRA's TurboQuant-QJL. Palettes are
discrete optimal 1-D eight-level quantizers, not QJL/Hadamard or an E8 decoder.
The original scale bytes and native FP4 kernel layout are preserved. This is
additional, lossy quantization unless every used symbol is in its palette.
"""
from collections import Counter
import math
import struct

MAGIC = 0x33504357
PREFIX = struct.Struct("<IIII")
TABLE_BYTES = 3 * 256 * 4
PREFIX_BYTES = PREFIX.size + TABLE_BYTES
LEVELS = (0., .5, 1., 1.5, 2., 3., 4., 6., 0., -.5, -1., -1.5, -2., -3., -4., -6.)
ORDER = (15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7)


def fit_palette(histogram):
    """Globally optimal squared-error medoids on the 15 distinct FP4 levels.

    Dynamic programming partitions the ordered alphabet into eight intervals;
    the best representable centroid in each interval is its discrete medoid.
    Stable ties make output reproducible. Signed zeros share one numeric level.
    """
    counts = [histogram[s] + (histogram[8] if s == 0 else 0) for s in ORDER]
    costs, centers = {}, {}
    for lo in range(15):
        for hi in range(lo + 1, 16):
            cost, center = min((sum(counts[j] * (LEVELS[ORDER[j]] - LEVELS[ORDER[c]]) ** 2
                                          for j in range(lo, hi)), c) for c in range(lo, hi))
            costs[lo, hi], centers[lo, hi] = cost, center
    dp = {(0, 0): 0.}
    splits = {}
    for k in range(1, 9):
        for end in range(k, 16):
            value, start = min((dp[k - 1, begin] + costs[begin, end], begin)
                               for begin in range(k - 1, end) if (k - 1, begin) in dp)
            dp[k, end], splits[k, end] = value, start
    palette, end = [], 15
    for k in range(8, 0, -1):
        begin = splits[k, end]
        palette.append(ORDER[centers[begin, end]])
        end = begin
    palette.reverse()
    mapping = bytes(min(range(8), key=lambda c: ((LEVELS[s] - LEVELS[palette[c]]) ** 2, c))
                    for s in range(16))
    return bytes(palette), mapping


def pack_matrix(scales, weights):
    if len(weights) != len(scales) * 16:
        raise ValueError("palette3 scale/weight group mismatch")
    buckets = {}
    for group, scale in enumerate(scales):
        if scale == 255:
            raise ValueError("palette3 does not quantize non-finite UE8M0 scales")
        buckets.setdefault(scale, bytearray()).extend(weights[group * 16:group * 16 + 16])
    tables, maps = bytearray(1024), {}
    error = energy = 0.
    maximum_scale = max(buckets, default=0)
    for scale, raw in buckets.items():
        hist = [0] * 16
        for value, count in Counter(raw).items():
            hist[value & 15] += count
            hist[value >> 4] += count
        palette, mapping = fit_palette(hist)
        for code in range(4):
            tables[scale * 4 + code] = palette[2 * code] | (palette[2 * code + 1] << 4)
        # An entire byte maps to six packed bits; four bytes form a 24-bit word.
        maps[scale] = tuple(mapping[b & 15] | (mapping[b >> 4] << 3) for b in range(256))
        factor = math.ldexp(1., 2 * (scale - maximum_scale))
        error += factor * sum(hist[s] * (LEVELS[s] - LEVELS[palette[mapping[s]]]) ** 2 for s in range(16))
        energy += factor * sum(hist[s] * LEVELS[s] ** 2 for s in range(16))
    packed = bytearray(len(weights) // 4 * 3)
    cursor = 0
    for group, scale in enumerate(scales):
        lookup = maps[scale]
        start = group * 16
        for pos in range(start, start + 16, 4):
            word = (lookup[weights[pos]] | (lookup[weights[pos + 1]] << 6) |
                    (lookup[weights[pos + 2]] << 12) | (lookup[weights[pos + 3]] << 18))
            packed[cursor:cursor + 3] = word.to_bytes(3, "little")
            cursor += 3
    return bytes(tables), bytes(packed), math.sqrt(error / energy) if energy else 0.


def unpack_matrix(scales, tables, packed):
    if len(tables) != 1024 or len(packed) != len(scales) * 12:
        raise ValueError("invalid palette3 matrix size")
    decoded = bytearray(len(scales) * 16)
    for group, scale in enumerate(scales):
        palette = [(tables[scale * 4 + code // 2] >> (4 * (code & 1))) & 15 for code in range(8)]
        for block in range(4):
            start = group * 12 + block * 3
            word = int.from_bytes(packed[start:start + 3], "little")
            for pair in range(4):
                decoded[group * 16 + block * 4 + pair] = (palette[(word >> (pair * 6)) & 7] |
                                                        palette[(word >> (pair * 6 + 3)) & 7] << 4)
    return bytes(decoded)
