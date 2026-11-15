#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Median and spread over several runs of `synth.sh speed` with one compiler on one kind of machine.

    python -I speed_runs.py LOG...

Each LOG holds the output of one or more `bash synth.sh speed RUNS` calls; a log of a GitHub Actions job
works too (the text before each line is skipped). A call gives one median per row of each of its two tables
(t_simd and t_scalar). Printed: the lines that describe the conditions (machine, CPU, compiler, records),
each with the number of calls that printed it; then a Markdown table with, per row, the number of calls,
the median of their medians, the lowest and the highest median, the spread (highest - lowest) / median, the
MB/s of the median and the scalar kernel's median of the same build divided by this one. Last, for kernel 1
with and without the fused hash: the number of interleaved streams with the lowest median, its difference
from 4 streams (the default of WEX_FAST_LANES) and whether the two ranges of medians overlap.
Exit 2 if no table was found.
"""
import re
import statistics
import sys

BUILD = re.compile(r"== (t_\w+) speed (\d+)")
ROW = re.compile(r"\| ([^|]+?) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|")
SIZE = re.compile(r"(\d+) records, (\d+) decoded bytes each")
INFO = re.compile(r"((?:machine|compiler): .*|(?:Model name|Vendor ID|CPU part):\s.*|"
                  r"machdep\.cpu\.brand_string: .*|hw\.model: .*|\d+ records, \d+ decoded bytes each, \d+ threads?, \d+ runs)")
K1 = re.compile(r"^k1: (?:\S+ search, )?(\d+) streams?(?: interleaved)?(, fused)?$")


def read(paths):
    data, info, order, size = {}, {}, [], None
    build = None
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\r\n")
                m = BUILD.search(line)
                if m:
                    build = m.group(1)
                    continue
                m = INFO.search(line)
                if m:
                    text = re.sub(r"\s+", " ", m.group(1).strip())
                    if text not in info:
                        order.append(text)
                    info[text] = info.get(text, 0) + 1
                    s = SIZE.search(line)
                    if s:
                        size = int(s.group(2))
                    continue
                m = ROW.search(line)
                if m and build:
                    rows = data.setdefault(build, {})
                    rows.setdefault(m.group(1), []).append(float(m.group(2)))
    return data, info, order, size


def main(argv):
    if not argv:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    data, info, order, size = read(argv)
    if not data:
        print("no speed table found", file=sys.stderr)
        return 2
    for text in order:
        print(f"{text}   [{info[text]}x]")
    print()
    print("| build | decoder | calls | median of the medians, ms/record | lowest | highest | spread | MB/s | "
          "k0 / this |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    summary = {}
    for build, rows in data.items():
        k0 = next((statistics.median(v) for name, v in rows.items() if name.startswith("k0:")), None)
        for name, values in rows.items():
            mid = statistics.median(values)
            summary[(build, name)] = (mid, min(values), max(values))
            rate = f"{size / mid / 1000.0:.1f}" if size else "-"
            ratio = f"{k0 / mid:.2f}" if k0 else "-"
            print(f"| {build} | {name} | {len(values)} | {mid:.3f} | {min(values):.3f} | {max(values):.3f} | "
                  f"{100.0 * (max(values) - min(values)) / mid:.1f}% | {rate} | {ratio} |")
    for fused in (False, True):
        lanes = {}
        for (build, name), value in summary.items():
            m = K1.match(name)
            if build == "t_simd" and m and bool(m.group(2)) == fused:
                lanes[int(m.group(1))] = value
        if 4 not in lanes:
            continue
        best = min(lanes, key=lambda n: lanes[n][0])
        mid, low, high = lanes[best]
        mid4, low4, high4 = lanes[4]
        kind = "fused" if fused else "not fused"
        if best == 4:
            second = min((n for n in lanes if n != 4), key=lambda n: lanes[n][0], default=None)
            line = f"kernel 1, {kind}: fastest 4 streams, {mid4:.3f} ms/record"
            if second is not None:
                line += (f"; next {second} streams, {lanes[second][0]:.3f} ms ({100.0 * (lanes[second][0] / mid4 - 1):+.1f}%), "
                         f"ranges {'overlap' if lanes[second][1] <= high4 else 'apart'}")
        else:
            line = (f"kernel 1, {kind}: fastest {best} streams, {mid:.3f} ms/record against {mid4:.3f} for 4 streams "
                    f"({100.0 * (mid / mid4 - 1):+.1f}%); ranges {'overlap' if low4 <= high else 'apart'} "
                    f"({best}: {low:.3f} to {high:.3f}, 4: {low4:.3f} to {high4:.3f})")
        print()
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
