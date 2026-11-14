#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Copy a tiny DeepSeek V4/V4.1 MXFP4 checkpoint with a wider MoE intermediate size.

Why: a WNR2 build needs WEX1 records that are whole 4096-byte pages, i.e.
hidden * moe_inter a multiple of 131072. The engine fixtures (v41-tiny: 128 x 64,
colibri deepseek_v4_tiny: 128 x 128) are not, so no WNR2 container can be made
for a model the engine can actually run. Widening moe_inter to 1024 fixes that
and keeps every other tensor byte-identical.

What changes: every routed and shared expert matrix (ffn.experts.N.w1/w2/w3 and
ffn.shared_experts.w1/w2/w3, backbone and mtp stages), and moe_intermediate_size /
moe_inter_dim in config.json. New bytes are drawn from the original tensor's own
byte distribution (seeded, deterministic); w2 scales drop by ~sqrt(k) so the
wider sum keeps its magnitude. The output is a different model: tokens differ
from the source and ref.json is not copied. It is only for safetensors-vs-WEX
token identity.

    python -I fixtures/widen_experts.py --src fixtures/v41-tiny --out out/v41-tiny-w1024 --moe-inter 1024

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import struct
from pathlib import Path

EXPERT = re.compile(r"\.ffn\.(experts\.\d+|shared_experts)\.(w[123])\.(weight|scale)$")
MOE_KEYS = ("moe_intermediate_size", "moe_inter_dim")
SKIP = {"ref.json", "ref_vendor.json", ".coli_usage"}


def read_safetensors(path: Path):
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    return header, raw[8 + n:]


def write_safetensors(path: Path, meta, tensors):
    """tensors: list of (name, dtype, shape, bytes) in file order."""
    header, payload = {}, bytearray()
    if meta is not None:
        header["__metadata__"] = meta
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + payload)


def moe_values(cfg: dict) -> set:
    found = set()
    for scope in (cfg, cfg.get("text_config") or {}):
        found |= {scope[k] for k in MOE_KEYS if k in scope}
    return found


def widen(src: Path, out: Path, moe_inter: int, seed: int) -> dict:
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    old = moe_values(cfg)
    if len(old) != 1:
        raise SystemExit(f"config.json: expected one moe intermediate size, found {sorted(old)}")
    (old_inter,) = old
    if moe_inter % old_inter:
        raise SystemExit(f"--moe-inter {moe_inter} must be a multiple of {old_inter}")
    k = moe_inter // old_inter
    w2_shift = round(math.log2(k) / 2)   # E8M0 exponent: the w2 sum has k times more terms
    shards = sorted(src.glob("*.safetensors"))
    if not shards or list(src.glob("*.safetensors.index.json")):
        raise SystemExit("need *.safetensors without an index json")   # shortcut: single-file fixtures only
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name not in SKIP and f.suffix != ".wex":
            shutil.copyfile(f, out / f.name)
    for scope in (cfg, cfg.get("text_config") or {}):
        for key in MOE_KEYS:
            if key in scope:
                scope[key] = moe_inter
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    changed = 0
    for shard in shards:
        header, payload = read_safetensors(shard)
        meta = header.pop("__metadata__", None)
        tensors = []
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = info["data_offsets"]
            data, shape = payload[a:b], list(info["shape"])
            m = EXPERT.search(name)
            if m:
                assert len(shape) == 2 and len(data) == shape[0] * shape[1], (name, shape, len(data))
                axis = 1 if m.group(2) == "w2" else 0          # the moe_inter axis
                shape[axis] *= k
                rng = random.Random(f"{seed}:{name}")
                data = bytes(rng.choices(data, k=shape[0] * shape[1]))
                if m.group(2) == "w2" and m.group(3) == "scale" and w2_shift:
                    data = bytes(max(1, v - w2_shift) for v in data)
                changed += 1
            tensors.append((name, info["dtype"], shape, data))
        write_safetensors(out / shard.name, meta, tensors)
    if not changed:
        raise SystemExit("no expert tensors found")
    return {"out": str(out), "moe_inter": moe_inter, "factor": k, "tensors_widened": changed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True, help="tiny checkpoint dir (config.json + *.safetensors)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--moe-inter", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args()
    print(json.dumps(widen(args.src, args.out, args.moe_inter, args.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
