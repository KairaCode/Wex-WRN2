#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Source of testdata/ceil-form.wex4: two synthetic MXFP4 experts whose WNR3 records are in ceil form.

The conformance vectors hold no WNR3 record with a CEIL bit (their experts have blocks of maximum 5, so the
encoder's gate keeps every matrix plain). This script writes experts made of blocks the ceil form changes
(maximum 7, no magnitude 1 or 3), blocks it leaves alone, blocks at scale 254 and 255, and all-zero blocks;
the second matrix of the second expert has blocks of maximum 5, so that one matrix stays plain.

    python -I make_ceil_form.py OUT_DIR

writes OUT_DIR/config.json and OUT_DIR/model.safetensors, and prints the SHA-256 of each expert's slab
(scales of w1, w2, w3, then packed weights of w1, w2, w3): what a WNR3 record without canonical zero must
decode to. README.md of this folder has the commands that turn OUT_DIR into the WEX4 file.
Standard library only. Deterministic.
"""
import hashlib
import json
import random
import struct
import sys
from pathlib import Path

H, I = 128, 64  # hidden size, expert inner size, in FP4 elements
SHAPES = {"w1": (I, H), "w2": (H, I), "w3": (I, H)}  # rows, columns


def block(rng, kind):
    """32 nibbles and the block's scale."""
    scale = rng.choices(range(116, 128), [1, 2, 3, 5, 8, 12, 16, 20, 16, 10, 5, 2])[0]
    if kind == "even":       # maximum 7, magnitudes in {0, 2, 4, 5, 6, 7}: halved by the ceil form
        mags = rng.choices([0, 2, 4, 5, 6, 7], [30, 26, 18, 10, 6, 3], k=32)
        mags[rng.randrange(32)] = 7
        scale = rng.choices([scale, 254, 255, 0], [94, 2, 2, 2])[0]   # 255 is not transformed
    elif kind == "odd":      # maximum 6 or 7 with a 1 or a 3: left alone
        mags = rng.choices(range(8), [34, 24, 14, 10, 6, 5, 4, 3], k=32)
        mags[rng.randrange(16)] = rng.choice([6, 7])
        mags[16 + rng.randrange(16)] = rng.choice([1, 3])
    elif kind == "low":      # maximum 0 to 4: legal, left alone
        mags = rng.choices(range(5), [40, 28, 16, 10, 6], k=32)
    elif kind == "five":     # maximum 5: its matrix cannot be in ceil form
        mags = rng.choices(range(6), [36, 26, 16, 10, 6, 6], k=32)
        mags[rng.randrange(32)] = 5
    else:                    # all zero, of either sign
        mags = [0] * 32
    return [m | (8 if rng.random() < 0.45 else 0) for m in mags], scale


def matrix(rng, rows, cols, with_five):
    kinds = ["even", "odd", "low", "zero"] + (["five"] if with_five else [])
    odds = [62, 18, 15, 5] + ([4] if with_five else [])
    weights, scales = bytearray(), bytearray()
    for _ in range(rows * cols // 32):
        q, s = block(rng, rng.choices(kinds, odds)[0])
        weights += bytes(q[2 * k] | q[2 * k + 1] << 4 for k in range(16))
        scales.append(s)
    return bytes(weights), bytes(scales)


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "num_hidden_layers": 1, "n_routed_experts": 2, "hidden_size": H, "moe_intermediate_size": I}))
    rng = random.Random(20261114)
    header, payload = {}, bytearray()

    def add(name, dtype, shape, data):
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}

    for expert in range(2):
        parts = {}
        for name, (rows, cols) in SHAPES.items():
            parts[name] = matrix(rng, rows, cols, with_five=(expert == 1 and name == "w2"))
            base = f"layers.0.ffn.experts.{expert}.{name}"
            add(base + ".weight", "I8", (rows, cols // 2), parts[name][0])
            add(base + ".scale", "F8_E8M0", (rows, cols // 32), parts[name][1])
        slab = b"".join(parts[n][1] for n in SHAPES) + b"".join(parts[n][0] for n in SHAPES)
        print(f"expert {expert}: slab of {len(slab)} bytes, sha256 {hashlib.sha256(slab).hexdigest()}")
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    (out / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
