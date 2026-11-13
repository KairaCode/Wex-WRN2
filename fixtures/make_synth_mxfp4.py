#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Synthetic DeepSeek MXFP4 expert checkpoint (experts only, no runnable model).

hidden 512 x moe_inter 256, so every WEX1 record is a whole number of 4096-byte
pages (208,896 B = 51 pages): the shape build_nr2_full.py needs. FP4 codes and UE8M0 scales
are skewed so an entropy coder has something to gain. Deterministic.

Use it for the store-level checks that need no forward pass: the WNR2 pipeline
(build_nr2_full, check_container, sample_records + test_decode_batch, check_all)
and c/tests/test_wex_planned.c (1 layer x 2 experts).

    python -I fixtures/make_synth_mxfp4.py OUT_DIR [--layers 2] [--experts 8]

Standard library only.
"""
import argparse
import json
import random
import struct
from pathlib import Path

H, I = 512, 256


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps({
        "num_hidden_layers": a.layers, "n_routed_experts": a.experts,
        "hidden_size": H, "moe_intermediate_size": I}))
    rng = random.Random(a.seed)
    codes = list(range(16))
    weights = [40, 22, 12, 6, 3, 2, 1, 1, 30, 18, 10, 5, 3, 2, 1, 1]

    def fp4_bytes(n):
        lo = rng.choices(codes, weights, k=n)
        hi = rng.choices(codes, weights, k=n)
        return bytes(x | (y << 4) for x, y in zip(lo, hi))

    def scale_bytes(n):
        return bytes(rng.choices(range(114, 128), [1, 1, 2, 3, 5, 8, 12, 16, 20, 16, 10, 5, 2, 1], k=n))

    shapes = {"w1": (I, H), "w2": (H, I), "w3": (I, H)}  # (out, in) in FP4 elements
    header, payload = {}, bytearray()

    def add(name, dtype, shape, data):
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}

    for layer in range(a.layers):
        for expert in range(a.experts):
            for name, (rows, cols) in shapes.items():
                base = f"layers.{layer}.ffn.experts.{expert}.{name}"
                add(base + ".weight", "I8", (rows, cols // 2), fp4_bytes(rows * cols // 2))
                add(base + ".scale", "F8_E8M0", (rows, cols // 32), scale_bytes(rows * cols // 32))
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    (a.out / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    record = sum(r * c // 2 + r * c // 32 for r, c in shapes.values())
    assert record % 4096 == 0, record
    print(json.dumps({"dir": str(a.out), "records": a.layers * a.experts, "record_bytes": record}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
