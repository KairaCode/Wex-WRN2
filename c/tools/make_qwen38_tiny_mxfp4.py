#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""The tiny Qwen3.8 fixture with MXFP4 routed experts (qwen38_wex.h's test model).

Starts from the tiny FP8 fixture (make qwen38-tiny-fp8-generate) and gives it new routed
experts, deterministic from --seed, quantized to MXFP4 the way Quark stores them (OCP:
one E8M0 exponent per 32 inputs, floor(log2 amax) - 2, E2M1 values rounded to nearest
even, no -0). moe_intermediate_size becomes --inter (32): a WEX1 record holds whole groups
of 32 only, and the fixture's 8 cannot. --inter 512 makes records of 7 pages that WNR3
compresses below their raw size (at 32 every record is one page: WNR3 keeps them raw).
Writes into --out:

  * the snapshot, its experts rewritten as F8_E4M3 bytes (weight_scale_inv 1.0) holding
    exactly the MXFP4 values -- e2m1 times a power of two is exact in e4m3 -- so the
    engine's FP8 path computes the same real weights as its MXFP4 path;
  * quark/model-mxfp4.safetensors: the experts as Quark names them (U8 nibbles
    [rows, cols/2], low nibble = even column; U8 weight_scale [rows, cols/32]);
  * experts-mxfp4.wex (+ .manifest.json): tools/repack_q38_mxfp4_wex.py on those shards;
  * mxfp4_ref.f32: every expert's dequantized values in float32, layer by layer, expert by
    expert, gate, up, down, row-major: the Python reference the engine's decode must match
    to the bit (tests/test_qwen38_mxfp4.c).

The gain makes the experts move the logits (tests/test_qwen38_mxfp4.py checks that the
engine's tokens differ from the source fixture's). Needs numpy, torch, safetensors.

  python tools/make_qwen38_tiny_mxfp4.py --src ./qwen38_tiny_fp8 --out ./qwen38_tiny_mxfp4
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from safetensors.torch import load_file, save_file

E2M1_MAG = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float64)
INTER = 32
SEED = 20261111


def quantize_mxfp4(w):
    """[rows, cols] float64 -> (U8 nibbles [rows, cols/2], U8 scales [rows, cols/32], float32 values)."""
    rows, cols = w.shape
    g = w.reshape(rows, cols // 32, 32)
    amax = np.abs(g).max(-1)
    e = np.where(amax > 0, np.floor(np.log2(np.where(amax > 0, amax, 1))) - 2, -127).astype(np.int64)
    e = np.clip(e, -127, 127)
    x = g / np.exp2(e)[..., None]
    mag = np.minimum(np.abs(x), 6.0)
    # nearest e2m1 magnitude, ties to the even code (the one with mantissa bit 0)
    d = np.abs(mag[..., None] - E2M1_MAG)
    best = d.min(-1, keepdims=True)
    tie = (d == best)
    code = np.where(tie & (np.arange(8) % 2 == 0), np.arange(8), 99).min(-1)
    code = np.where(code == 99, np.argmax(tie, -1), code)
    nib = np.where((x < 0) & (code > 0), code | 8, code).astype(np.uint8).reshape(rows, cols)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    scales = (e + 127).astype(np.uint8)
    values = (np.where(nib & 8, -1.0, 1.0) * E2M1_MAG[nib & 7]).reshape(rows, cols // 32, 32)
    values = (values * np.exp2(e)[..., None]).reshape(rows, cols).astype(np.float32)
    return packed, scales, values


def e4m3_bytes(values):
    """Exact e4m3 bytes of float32 values (each must be representable)."""
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    lookup = {}
    for byte in range(256):
        v = float(table[byte])
        if np.isfinite(v) and (v, np.signbit(v)) not in lookup:
            lookup[(v, bool(np.signbit(v)))] = byte
    flat = values.ravel()
    out = np.empty(flat.shape, np.uint8)
    for i, v in enumerate(flat):
        key = (float(v), bool(np.signbit(v)))
        if key not in lookup:
            raise ValueError(f"{v!r} is not exact in e4m3: pick a smaller --gain")
        out[i] = lookup[key]
    return out.reshape(values.shape)


def build(src, out, seed=SEED, gain=3.0, inter=INTER):
    src, out = Path(src), Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("logits", "ref*.json", "experts-*", "quark"))
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    text = cfg.get("text_config", cfg)
    layers, experts, hidden = text["num_hidden_layers"], text["num_experts"], text["hidden_size"]
    text["moe_intermediate_size"] = inter
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    index_path = out / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = {name: load_file(str(out / name)) for name in sorted(set(index["weight_map"].values()))}
    prefix = next(k.split("layers.")[0] + "layers." for k in index["weight_map"] if ".mlp.experts." in k)
    tensors = {k: t for s in shards.values() for k, t in s.items()}
    spread = np.std(np.concatenate([(t.float() * tensors[k + "_scale_inv"].float()).numpy().ravel()
                                    for k, t in tensors.items() if ".mlp.experts." in k and k.endswith(".weight")]))
    rng = np.random.default_rng(seed)
    shapes = {"gate_proj": (inter, hidden), "up_proj": (inter, hidden), "down_proj": (hidden, inter)}
    quark, ref = {}, []
    for layer in range(layers):
        for expert in range(experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                name = f"{prefix}{layer}.mlp.experts.{expert}.{proj}.weight"
                packed, scales, values = quantize_mxfp4(rng.normal(0, spread * gain, shapes[proj]))
                quark[name] = torch.from_numpy(packed)
                quark[name + "_scale"] = torch.from_numpy(scales)
                ref.append(values)
                shard = index["weight_map"][name]
                assert shards[shard][name].dtype == torch.float8_e4m3fn, name
                shards[shard][name] = torch.from_numpy(e4m3_bytes(values)).view(torch.float8_e4m3fn)
                rows, cols = shapes[proj]   # one 1.0 per 128x128 block
                shards[shard][name + "_scale_inv"] = torch.ones(-(-rows // 128), -(-cols // 128), dtype=torch.bfloat16)
    for name, tensors in shards.items():
        save_file(tensors, str(out / name), metadata={"format": "pt"})
    index["metadata"]["total_size"] = sum(t.numel() * t.element_size() for s in shards.values() for t in s.values())
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    (out / "quark").mkdir()
    save_file(quark, str(out / "quark" / "model-mxfp4.safetensors"), metadata={"format": "pt"})
    np.concatenate([v.ravel() for v in ref]).astype("<f4").tofile(out / "mxfp4_ref.f32")
    spec = importlib.util.spec_from_file_location("q38_repack", Path(__file__).with_name("repack_q38_mxfp4_wex.py"))
    repack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repack)
    manifest = repack.repack(out / "quark", out, out / "experts-mxfp4.wex", min_cos=0.999999)
    print(f"MXFP4 fixture {out}: {layers} layers x {experts} experts, inter {inter}, gain {gain}; "
          f"FP8 check min cos {min(v['min_cos'] for v in manifest['fp8_check'].values())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="the tiny FP8 fixture (qwen38_tiny_fp8)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gain", type=float, default=3.0, help="the experts' spread over the source experts'")
    parser.add_argument("--inter", type=int, default=INTER, help="moe_intermediate_size, a multiple of 32")
    args = parser.parse_args()
    build(args.src, args.out, args.seed, args.gain, args.inter)


if __name__ == "__main__":
    main()
