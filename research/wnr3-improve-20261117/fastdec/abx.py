#!/usr/bin/env python3
"""Interleaved A/B across binaries: abx.py ROUNDS THREADS PASSES LABEL=BIN:VARIANT [...] -- REC...
Each round runs every (binary, variant) once, in an order rotated by one each round; one run =
`BIN ab MODEL PASSES THREADS VARIANT REC...` (PASSES timed passes over the records after one warm-up pass, the
run's median ms/record is taken). Prints per row: median, min, max over rounds, and the median over rounds of the
per-round ratio to the first row, plus how many rounds the row beat the first row."""
import subprocess, sys, statistics, os, re

MODEL = os.environ.get("MODEL", "<TMP>/wnr3-data/recs/model.bin")
rounds, threads, passes = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
args = sys.argv[4:]
sep = args.index("--")
rows = []
for a in args[:sep]:
    label, rest = a.split("=", 1)
    binary, variant = rest.rsplit(":", 1)
    rows.append((label, binary, variant))
recs = args[sep + 1:]
times = [[] for _ in rows]
for r in range(rounds):
    order = [(i + r) % len(rows) for i in range(len(rows))]
    for i in order:
        label, binary, variant = rows[i]
        out = subprocess.run([binary, "ab", MODEL, str(passes), str(threads), variant] + recs,
                             capture_output=True, text=True, check=True).stdout
        line = [l for l in out.splitlines() if l.startswith("| k")][0]
        times[i].append(float(line.split("|")[2]))
print(f"abx: {len(recs)} records, {rounds} rounds x {passes} passes, {threads} threads; ms/record")
print("| variant | binary | median | min | max | median ratio to row 1 | rounds faster than row 1 |")
print("|---|---|---:|---:|---:|---:|---:|")
for i, (label, binary, variant) in enumerate(rows):
    ratios = [t / b for t, b in zip(times[i], times[0])]
    faster = sum(t < b for t, b in zip(times[i], times[0]))
    print(f"| {label} | {os.path.basename(binary)}:{variant} | {statistics.median(times[i]):.2f} | {min(times[i]):.2f} | "
          f"{max(times[i]):.2f} | {statistics.median(ratios):.3f} | {faster}/{rounds} |")
