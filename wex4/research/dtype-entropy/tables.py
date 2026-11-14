#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Print the Markdown tables and facts used in RESULTS.md from results.json (python -I tables.py)."""
import json
from pathlib import Path

RAW_BITS = {"F8_E4M3": 8, "BF16": 16, "F32": 32}
BIG = 1 << 20
r = json.loads((Path(__file__).parent / "results.json").read_text())
esc = lambda s: s.replace("|", "&#124;")

for group, rows in r["summary"].items():
    dtype = group.split()[0]
    first = next(iter(rows.values()))
    print(f"\n### {group} ({first['tensors']} records, {first['sample_raw_bytes']:,} B sampled, raw = {RAW_BITS[dtype]} bits/element)\n")
    print("| model | ideal bits/el | coded bits/el | ratio (sample) | ratio (whole dtype, extrapolated) "
          "| table B/record | break-even elements | rANS symbols/el | decode class |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, m in rows.items():
        be = m.get("break_even_elements_vs_wbp1", "-")
        be = "never" if be is None else "-" if be == "-" else f"{be:,}"
        cls = ("bound only" if m["cls"].startswith("bound") else "order-0" if m["cls"].startswith("order-0") else
               "context (serial)" if "serial" in m["cls"] else "context (side)")
        est = f"{m['ratio_dtype_extrapolated']:.5f}" if "ratio_dtype_extrapolated" in m else "-"
        print(f"| {esc(name)} | {m['ideal_bits_per_element']:.4f} | {m['coded_bits_per_element']:.4f} "
              f"| {m['ratio_sample']:.5f} | {est} | {m['table_bytes'] / m['tensors']:,.0f} | {be} "
              f"| {m['coded_symbols_per_element']:g} | {cls} |")

print("\n### Shared table vs per-record table (order-0 byte planes, ratio vs raw)\n")
print("| dtype | kind | tensors | raw B | per-record tables | one record for the kind | shared table, held-out |")
print("|---|---|---|---|---|---|---|")
for s in r["shared"]:
    f = lambda k: f"{s[k] / s['raw_bytes']:.5f}"
    print(f"| {s['dtype']} | {s['kind']} | {s['tensors']} | {s['raw_bytes']:,} | {f('wbp1_per_record')} "
          f"| {f('wbp1_one_record_for_kind')} | {f('wbp1_shared_table_held_out')} |")

print("\n### Per-tensor WBP1 ratio and best realisable model (>= 1 MiB)\n")
print("| dtype | tensor | elements | WBP1 ratio | best model | its ratio |")
print("|---|---|---|---|---|---|")
for t in r["tensors"]:
    if t["tensor_bytes"] < BIG:
        continue
    ms = {k: (v["payload"] + v["table"]) / t["raw_bytes"] for k, v in t["models"].items() if not k.startswith("bound")}
    best = min(ms, key=ms.get)
    rows = f"[{t['rows'][0]}:{t['rows'][1]}]" if t["rows"] else ""
    print(f"| {t['dtype']} | {t['name']}{rows} | {t['elements']:,} | {next(iter(ms.values())):.5f} "
          f"| {esc(best)} | {ms[best]:.5f} |")

print("\n### Facts\n")
pop = r["population"]
for d, k in pop.items():
    print(f"- population {d}: {sum(k.values()):,} B in {len(k)} kinds")
f8 = [t for t in r["tensors"] if t["dtype"] == "F8_E4M3"]
nb = lambda t: t["elements"] // 1024
blocks = sum(nb(t) for t in f8)
print(f"- F8_E4M3: {len(f8)} sampled records, {blocks:,} blocks of 32x32; records with a .scale tensor:",
      sum(t["facts"]["has_scale"] for t in f8))
print("- F8_E4M3 kinds with a matching F8_E8M0 .scale kind in the whole model:",
      sum(1 for k in pop["F8_E4M3"] if k[:-6] + "scale" in pop["F8_E8M0"]), "of", len(pop["F8_E4M3"]))
print("- distinct scale values per record: min", min(t["facts"]["scale_distinct"] for t in f8),
      "max", max(t["facts"]["scale_distinct"] for t in f8),
      "| E8M0 code range", min(t["facts"]["scale_min"] for t in f8), "..", max(t["facts"]["scale_max"] for t in f8))
for k in ("scale_o0_bits", "scale_bits_given_left", "block_max_exp_bits"):
    print(f"- {k}: {sum(t['facts'][k] * nb(t) for t in f8) / blocks:.4f} bits per block (block-weighted mean)")
print("- F8 -0 codes:", sum(t["facts"]["neg_zero"] for t in f8), "| +0 codes:", sum(t["facts"]["pos_zero"] for t in f8),
      "| NaN codes:", sum(t["facts"]["nan_codes"] for t in f8), "| elements:", sum(t["elements"] for t in f8))
bf = [t for t in r["tensors"] if t["dtype"] == "BF16" and t["tensor_bytes"] >= BIG]
el = sum(t["elements"] for t in bf)
for k in ("sign_bits", "exp_bits", "mant_bits"):
    print(f"- BF16 {k}: {sum(t['facts'][k] * t['elements'] for t in bf) / el:.4f} (element-weighted, >= 1 MiB records)")
f3 = [t for t in r["tensors"] if t["dtype"] == "F32" and t["tensor_bytes"] >= BIG]
print("- F32 plane bits b0..b3 (mean over >= 1 MiB records):",
      [round(sum(t["facts"]["plane_bits_b0_b3"][p] for t in f3) / len(f3), 4) for p in range(4)],
      "| f16-exact fraction max:", max(t["facts"]["f16_exact_frac"] for t in f3),
      "| distinct/elements min:", round(min(t["facts"]["distinct_values"] / t["elements"] for t in f3), 4))
print("- equality checks (first 4 MiB of each):", json.dumps(r["equality_checks"]))
print("- bytes read by the run:", f"{r['read_bytes']:,}")
