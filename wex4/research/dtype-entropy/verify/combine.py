#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Kind-level comparison: the experiment's sample (../results.json) vs the verification sample (out/verify_*.json),
and whole-dtype estimates recomputed with the verification data. No model access."""
import json, re, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
T = json.loads((HERE.parent / "results.json").read_text())
V = {s: json.loads((HERE / "out" / f"verify_{s}.json").read_text()) for s in ("f8", "bf16", "f32", "tail")}
coarse = lambda n: re.sub(r"\d+", "#", n)
MAP = {"F8_E4M3": {"wbp1": "o0_byte (WBP1)", "ctx_block_scale": "ctx block scale (E8M0 value)"},
       "BF16": {"wbp1": "o0 byte planes (WBP1)", "exp8_coded_rest_raw": "exp8 o0 + sign,mant7 raw",
                "hi_o0_lo_given_hi4": "hi o0 + lo|hi low 4 bits"},
       "F32": {"wbp1": "o0 byte planes (WBP1)", "signexp9_coded_mant23_raw": "signexp9 o0 + mant23 raw"}}
ROWS, TAIL0 = 129280, 126155
out = {}

for dtype, stage in (("F8_E4M3", "f8"), ("BF16", "bf16"), ("F32", "f32")):
    pop = T["population"][dtype]
    total = sum(pop.values())
    print(f"\n== {dtype}: {total:,} B")
    for mine_key, their_key in MAP[dtype].items():
        theirs = defaultdict(lambda: [0, 0])
        for t in T["tensors"]:
            if t["dtype"] == dtype and t["tensor_bytes"] >= 1 << 20:
                m = t["models"][their_key]
                k = theirs[coarse(t["name"])]
                k[0] += m["payload"] + m["table"]; k[1] += t["raw_bytes"]
        mean_t = sum(c for c, _ in theirs.values()) / sum(r for _, r in theirs.values())
        mine = defaultdict(lambda: [0, 0])
        for t in V[stage]["tensors"]:
            if t["name"] in ("embed.weight", "head.weight") and t["rows"][0] >= TAIL0:
                continue                                    # the tail is taken from the 250-row probe below
            k = mine[coarse(t["name"])]
            k[0] += t["models"][mine_key]["bytes"]; k[1] += t["raw_bytes"]
        ratio_m = {k: c / r for k, (c, r) in mine.items()}
        if dtype == "BF16":                                 # embed/head: body from 31 stratified slices + exact tail rows
            for name in ("embed.weight", "head.weight"):
                tail = [t for t in V["tail"]["tensors"] if t["name"] == name]
                tr = sum(t["models"][mine_key]["bytes"] for t in tail) / sum(t["raw_bytes"] for t in tail)
                ratio_m[name] = (ratio_m[name] * TAIL0 + tr * (ROWS - TAIL0)) / ROWS
        print(f"-- {mine_key}  (theirs: '{their_key}')")
        print(f"   {'kind':42} {'pop B':>14} {'theirs':>8} {'mine':>8} {'diff':>8}")
        est_t = est_m = 0.0
        for k, b in sorted(pop.items(), key=lambda kv: -kv[1]):
            rt = theirs[k][0] / theirs[k][1] if k in theirs else mean_t
            rm = ratio_m.get(k, rt)                         # kinds I did not measure keep their number
            est_t += b * rt; est_m += b * rm
            if k in ratio_m:
                print(f"   {k:42} {b:>14,} {rt:8.5f} {rm:8.5f} {rm - rt:+8.5f}{'' if k in theirs else '  (theirs = sample mean, kind not read)'}")
        cov = sum(pop[k] for k in ratio_m) / total
        print(f"   whole dtype: theirs {est_t / total:.5f} ({est_t / 1e6:,.1f} MB)  with my kinds substituted {est_m / total:.5f} "
              f"({est_m / 1e6:,.1f} MB)  diff {(est_m - est_t) / 1e6:+.1f} MB   [my kinds cover {100 * cov:.1f} % of bytes]")
        out[f"{dtype} {mine_key}"] = {"theirs_ratio": est_t / total, "mine_ratio": est_m / total,
                                      "theirs_bytes": round(est_t), "mine_bytes": round(est_m), "my_kinds_cover": cov}

w = [out[f"{d} wbp1"] for d in MAP]
tot = sum(sum(T["population"][d].values()) for d in MAP)
print(f"\nWBP1 total of the three dtypes ({tot:,} B): theirs {sum(x['theirs_bytes'] for x in w) / 1e6:,.1f} MB "
      f"({sum(x['theirs_bytes'] for x in w) / tot:.4f}), with my kinds {sum(x['mine_bytes'] for x in w) / 1e6:,.1f} MB "
      f"({sum(x['mine_bytes'] for x in w) / tot:.4f})")
ho = V["f8"]["held_out_10_vs_30"]
print("F8 held-out table (layer 10 <-> layer 30, no table charged):",
      f"{sum(h['held_out_bytes'] for h in ho) / sum(h['raw_bytes'] for h in ho):.5f}",
      "| per-record tables on the same 16 tensors:",
      f"{sum(t['models']['wbp1']['bytes'] for t in V['f8']['tensors'] if t['name'].startswith('layers')) / sum(h['raw_bytes'] for h in ho):.5f}")
print("F8 scales: distinct per record", sorted(set(t["models"]["scale_distinct"] for t in V["f8"]["tensors"])),
      "| o0 bits/scale (block-weighted)",
      round(sum(t["models"]["scale_o0_bits"] * t["elements"] for t in V["f8"]["tensors"]) / sum(t["elements"] for t in V["f8"]["tensors"]), 4))
print("bytes read by the verification stages:", sum(v["read_bytes"] for v in V.values()) + json.loads((HERE / "out" / "verify_repro.json").read_text())["read_bytes"])
(HERE / "out" / "combine.json").write_text(json.dumps(out, indent=1))
