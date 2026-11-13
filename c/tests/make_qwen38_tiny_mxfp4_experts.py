#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Quantize the tiny Qwen3.8 FP8 fixture's routed experts to MXFP4 (deterministic).

Test input for tests/test_qwen36_tier_mxfp4_gpu.c: e4m3 * 128x128 block scale ->
f32 -> OCP MXFP4 (ue8m0 = floor(log2(amax)) - 2 + 127 per 32 inputs, e2m1
round-to-nearest-even on the grid {0,.5,1,1.5,2,3,4,6}, low nibble = even column).
Output: int32 [L, E, D, I], then per layer, per expert: gate|up|down nibbles,
gate|up|down ue8m0 bytes.  usage: make_qwen38_tiny_mxfp4_experts.py FIXTURE OUT
"""
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def mxfp4(w):
    """[O, K] f32 -> (packed [O, ceil(K/2)] u8, scales [O, ceil(K/32)] u8)."""
    O, K = w.shape
    ng = (K + 31) // 32
    wp = torch.zeros(O, ng * 32)
    wp[:, :K] = w
    g = wp.view(O, ng, 32)
    amax = g.abs().amax(-1)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(1e-30))) - 2, torch.zeros_like(amax))
    e = e.clamp(-127, 127)
    v = g / torch.exp2(e).unsqueeze(-1)
    idx = (v.abs().unsqueeze(-1) - GRID).abs().argmin(-1)        # nearest magnitude (ties -> lower)
    nib = (idx | ((v < 0).long() << 3)).view(O, ng * 32)[:, :K]
    if K % 2:
        nib = torch.cat([nib, torch.zeros(O, 1, dtype=nib.dtype)], 1)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    return packed.numpy().tobytes(), (e + 127).to(torch.uint8).numpy().tobytes()


def main():
    fix, out = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = json.loads((fix / "config.json").read_text())
    idx = json.loads((fix / "model.safetensors.index.json").read_text())["weight_map"]
    L, E, D, I = cfg["num_hidden_layers"], cfg["num_experts"], cfg["hidden_size"], cfg["moe_intermediate_size"]

    def f32(name):
        with safe_open(str(fix / idx[name]), "pt") as f:
            w = f.get_tensor(name).float()
            s = f.get_tensor(name + "_scale_inv").float()
        O, K = w.shape
        return w * s.repeat_interleave(128, 0)[:O].repeat_interleave(128, 1)[:, :K]

    with open(out, "wb") as f:
        f.write(struct.pack("<4i", L, E, D, I))
        for l in range(L):
            for e in range(E):
                p = f"model.layers.{l}.mlp.experts.{e}."
                q = [mxfp4(f32(p + n + ".weight")) for n in ("gate_proj", "up_proj", "down_proj")]
                for w, _ in q:
                    f.write(w)
                for _, s in q:
                    f.write(s)
    print(f"wrote {out}: L={L} E={E} D={D} I={I}")


if __name__ == "__main__":
    main()
