#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Synthetic MXFP4 expert checkpoint (experts only, no runnable model), DeepSeek names by default.

hidden 512 x moe_inter 256, so every WEX1 record is a whole number of 4096-byte
pages (208,896 B = 51 pages): the shape build_nr2_full.py needs. FP4 codes and UE8M0 scales
are skewed so an entropy coder has something to gain. Deterministic.

Use it for the store-level checks that need no forward pass: the WNR2 pipeline
(build_nr2_full, check_container, sample_records + test_decode_batch, check_all)
and c/tests/test_wex_planned.c (1 layer x 2 experts).

    python -I fixtures/make_synth_mxfp4.py OUT_DIR [--layers 2] [--experts 8]
                                           [--names PROFILE] [--first-layer K] [--float-layer N]
    python -I fixtures/make_synth_mxfp4.py --list-names

--names writes the same expert bytes under the tensor names and dtypes of another model family (NAMES below;
the sources of the names are in wex4/README.md). The table is written here by hand, apart from the one in
wex4/wex4_convert.py, so that the tests compare two tables. --first-layer K leaves layers 0..K-1 without
routed experts (dense first layers, as first_k_dense_replace in DeepSeek V3, Kimi K2 and GLM-4). --float-layer N
writes the experts of layer N as float checkpoints do: three BF16 weight matrices and no scale tensor.

Standard library only.
"""
import argparse
import json
import random
import struct
from pathlib import Path

H, I = 512, 256

W = ("w1", "w2", "w3")
PROJ = ("gate_proj", "down_proj", "up_proj")   # the slab order w1, w2, w3
# profile -> (expert prefix, matrix names, weight suffix, scale suffix, weight dtype, scale dtype)
NAMES = {
    "deepseek": ("layers.{layer}.ffn.experts.{expert}.", W, ".weight", ".scale", "I8", "F8_E8M0"),
    "deepseek-quark": ("layers.{layer}.ffn.experts.{expert}.", W, ".weight", ".weight_scale", "U8", "U8"),
    "qwen38": ("model.language_model.layers.{layer}.mlp.experts.{expert}.", PROJ, ".weight", ".weight_scale", "U8", "U8"),
    "qwen3-moe": ("model.layers.{layer}.mlp.experts.{expert}.", PROJ, ".weight", ".weight_scale", "U8", "U8"),
    "kimi-k26": ("language_model.model.layers.{layer}.mlp.experts.{expert}.", PROJ, ".weight", ".weight_scale", "U8", "U8"),
    "mixtral": ("model.layers.{layer}.block_sparse_moe.experts.{expert}.", W, ".weight", ".weight_scale", "U8", "U8"),
    "minimax-m3": ("language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.", W, ".weight", ".weight_scale",
                   "U8", "U8"),
    "mistral-large4": ("layers.{layer}.experts.{expert}.", W, ".weight", ".weight_scale", "U8", "U8"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--names", choices=sorted(NAMES), default="deepseek")
    ap.add_argument("--first-layer", type=int, default=0)
    ap.add_argument("--float-layer", type=int, default=-1)
    ap.add_argument("--list-names", action="store_true")
    a = ap.parse_args()
    if a.list_names:
        print(" ".join(NAMES))
        return 0
    if a.out is None:
        ap.error("OUT_DIR is required")
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

    def bf16_bytes(n):
        return b"".join(struct.pack("<f", rng.gauss(0, 0.05))[2:] for _ in range(n))

    shapes = ((I, H), (H, I), (I, H))  # w1, w2, w3: (out, in) in FP4 elements
    prefix, matrices, weight_suffix, scale_suffix, weight_dtype, scale_dtype = NAMES[a.names]
    header, payload = {}, bytearray()

    def add(name, dtype, shape, data):
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}

    records = 0
    for layer in range(a.first_layer, a.layers):
        for expert in range(a.experts):
            for name, (rows, cols) in zip(matrices, shapes):
                base = prefix.format(layer=layer, expert=expert) + name
                if layer == a.float_layer:
                    add(base + weight_suffix, "BF16", (rows, cols), bf16_bytes(rows * cols))
                    continue
                add(base + weight_suffix, weight_dtype, (rows, cols // 2), fp4_bytes(rows * cols // 2))
                add(base + scale_suffix, scale_dtype, (rows, cols // 32), scale_bytes(rows * cols // 32))
            records += layer != a.float_layer
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    (a.out / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    record = sum(r * c // 2 + r * c // 32 for r, c in shapes)
    assert record % 4096 == 0, record
    print(json.dumps({"dir": str(a.out), "records": records, "record_bytes": record}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
