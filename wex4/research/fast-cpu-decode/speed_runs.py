#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Median and spread over several runs of `synth.sh speed` with one compiler on one kind of machine.

    python -I speed_runs.py [--compiler TEXT] LOG...

Each LOG holds the output of one or more `bash synth.sh speed RUNS` calls; the log of a GitHub Actions job
works too (the text before each line is skipped, and so are the lines that this script prints). A call
starts at its `machine:` line. With --compiler only the calls whose first `compiler:` line contains TEXT
count (an arm-speed job of the CI workflow runs two compilers on macOS). A call gives one median per row of
each of its two tables (t_simd and t_scalar). Printed: the lines that describe the conditions (machine, CPU,
compiler, records), each with the number of calls that printed it; then a Markdown table with, per row, the
number of calls, the median of their medians, the lowest and the highest median, the spread (highest -
lowest) / median, the MB/s of the median and the scalar kernel's median of the same build divided by this
one. Then, for kernel 1 with and without the fused hash, one row per number of interleaved streams: the
same median and range, the median over the calls of this row's median divided by the 4-stream median of the
same call, and in how many calls this row was faster than 4 streams (4 is the default of WEX_FAST_LANES).
Last, per kind: the number of streams with the lowest median, its difference from 4 streams and whether the
two ranges of medians overlap. Exit 2 if no table was found.
"""
import re
import statistics
import sys

BUILD = re.compile(r"== (t_\w+) speed (\d+)")
# a row of the table of test_fast_decode.c `speed`: exactly five cells, the last four numbers
ROW = re.compile(r"\| ([^|]+?) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|\s*$")
SIZE = re.compile(r"(\d+) records, (\d+) decoded bytes each")
MACHINE = re.compile(r"machine: ")
COMPILER = re.compile(r"compiler: (.*)")
INFO = re.compile(r"((?:machine|compiler): .*|(?:Model name|Vendor ID|CPU part):\s.*|"
                  r"machdep\.cpu\.brand_string: .*|hw\.model: .*|\d+ records, \d+ decoded bytes each, \d+ threads?, \d+ runs)")
OURS = re.compile(r"\s\[\d+x\]$")   # a condition line as this script prints it
K1 = re.compile(r"^k1: (?:\S+ search, )?(\d+) streams?(?: interleaved)?(, fused)?$")


def read(paths, compiler):
    calls, call = [], None
    for path in paths:
        call = None
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if OURS.search(line):
                    continue
                if MACHINE.search(line):
                    call = {"info": [], "compiler": None, "build": None, "rows": {}, "size": None}
                    calls.append(call)
                if call is None:
                    continue
                m = BUILD.search(line)
                if m:
                    call["build"] = m.group(1)
                    continue
                m = INFO.search(line)
                if m:
                    text = re.sub(r"\s+", " ", m.group(1).strip())
                    c = COMPILER.match(text)
                    if c and call["compiler"] is not None:
                        continue   # a later compiler line (synth.sh check or lanes) is not this call's
                    if c:
                        call["compiler"] = c.group(1)
                    if text not in call["info"]:
                        call["info"].append(text)
                    s = SIZE.search(line)
                    if s:
                        call["size"] = int(s.group(2))
                    continue
                m = ROW.search(line)
                if m and call["build"]:
                    call["rows"][(call["build"], m.group(1))] = float(m.group(2))
    return [c for c in calls if c["rows"] and (compiler is None or compiler in (c["compiler"] or ""))]


def lanes_of(name):
    m = K1.match(name)
    return (int(m.group(1)), bool(m.group(2))) if m else None


def main(argv):
    compiler = None
    if argv[:1] == ["--compiler"] and len(argv) > 1:
        compiler, argv = argv[1], argv[2:]
    if not argv or argv[0].startswith("-"):
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    calls = read(argv, compiler)
    if not calls:
        print("no speed table found", file=sys.stderr)
        return 2
    order, count = [], {}
    for c in calls:
        for text in c["info"]:
            if text not in count:
                order.append(text)
            count[text] = count.get(text, 0) + 1
    for text in order:
        print(f"{text}   [{count[text]}x]")
    print()
    size = next((c["size"] for c in calls if c["size"]), None)
    data = {}
    for c in calls:
        for key, value in c["rows"].items():
            data.setdefault(key, []).append(value)
    print("| build | decoder | calls | median of the medians, ms/record | lowest | highest | spread | MB/s | "
          "k0 / this |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    summary = {}
    for (build, name), values in data.items():
        k0 = next((statistics.median(v) for (b, n), v in data.items() if b == build and n.startswith("k0:")), None)
        mid = statistics.median(values)
        summary[(build, name)] = (mid, min(values), max(values))
        rate = f"{size / mid / 1000.0:.1f}" if size else "-"
        ratio = f"{k0 / mid:.2f}" if k0 else "-"
        print(f"| {build} | {name} | {len(values)} | {mid:.3f} | {min(values):.3f} | {max(values):.3f} | "
              f"{100.0 * (max(values) - min(values)) / mid:.1f}% | {rate} | {ratio} |")
    lines = []
    for fused in (False, True):
        lanes = {}
        for (build, name), value in summary.items():
            k = lanes_of(name)
            if build == "t_simd" and k and k[1] == fused:
                lanes[k[0]] = (name, value)
        if 4 not in lanes:
            continue
        kind = "fused" if fused else "not fused"
        if not lines:
            print()
            print("| kernel 1 | streams | calls | median of the medians, ms/record | lowest | highest | "
                  "median of (this / 4 streams) per call | calls faster than 4 streams |")
            print("|---|---:|---:|---:|---:|---:|---:|---:|")
        four = lanes[4][0]
        for n in sorted(lanes):
            name, (mid, low, high) = lanes[n]
            pairs = [c["rows"][("t_simd", name)] / c["rows"][("t_simd", four)] for c in calls
                     if ("t_simd", name) in c["rows"] and ("t_simd", four) in c["rows"]]
            faster = sum(p < 1.0 for p in pairs)
            print(f"| {kind} | {n} | {len(pairs)} | {mid:.3f} | {low:.3f} | {high:.3f} | "
                  f"{statistics.median(pairs):.3f} | {'-' if n == 4 else f'{faster} of {len(pairs)}'} |")
        best = min(lanes, key=lambda n: lanes[n][1][0])
        mid, low, high = lanes[best][1]
        mid4, low4, high4 = lanes[4][1]
        if best == 4:
            second = min((n for n in lanes if n != 4), key=lambda n: lanes[n][1][0], default=None)
            line = f"kernel 1, {kind}: fastest 4 streams, {mid4:.3f} ms/record"
            if second is not None:
                low2 = lanes[second][1][1]
                line += (f"; next {second} streams, {lanes[second][1][0]:.3f} ms "
                         f"({100.0 * (lanes[second][1][0] / mid4 - 1):+.1f}%), ranges {'overlap' if low2 <= high4 else 'apart'}")
        else:
            line = (f"kernel 1, {kind}: fastest {best} streams, {mid:.3f} ms/record against {mid4:.3f} for 4 streams "
                    f"({100.0 * (mid / mid4 - 1):+.1f}%); ranges {'overlap' if low4 <= high else 'apart'} "
                    f"({best}: {low:.3f} to {high:.3f}, 4: {low4:.3f} to {high4:.3f})")
        lines.append(line)
    for line in lines:
        print()
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
