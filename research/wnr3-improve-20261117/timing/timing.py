#!/usr/bin/env python3
"""One-session interleaved timing of every decoder variant of the fastdec and fastfmt lanes.

timing.py ROUNDS PASSES MODE...   MODE = 1 | 4 | 4p (4 threads, OMP_WAIT_POLICY=passive) | r4 (4 records at once, 1 thread each)

Each round: the non-baseline variants in an order rotated by one per round, and the baseline A (unchanged
wex_fast_decode.h, kernel 1 x 4 fused, AVX2 flags) run before every one of them and once at the end:
A X1 A X2 A X3 ... A. One run = one process (bin/tdrv time PASSES THREADS VARIANT RECS): 1 warm-up pass plus
PASSES timed passes over the records; the run's value is the median pass (wall ms/record; process CPU ms/record).
Speed-up of X in a round = mean(A before X, A after X) / X. Reported: median and min..max over rounds.
Raw samples go to logs/raw_MODE.json."""
import json, os, statistics, subprocess, sys, time

HERE = "<TMP>/wnr3-research/timing"
FF = "<TMP>/wnr3-research/fastfmt/out"
RECS = "00_007 05_080 10_153 15_034 20_107 25_180 30_061 35_134 39_120".split()
RAW = 18800640
BASE = ("A: wex_fast_decode unchanged, k1 x4 fused", "orig_a2", "w3:1,4,1")
VARS = [
    ("WNR3 reference wex_nr2_decode", "orig_a2", "ref"),
    ("wex_fast_decode unchanged, k1 x4 fused, AVX-512 flags", "orig_a5", "w3:1,4,1"),
    ("fastdec kernel 3 (AVX2) x4 fused", "new_a2", "w3:3,4,1"),
    ("fastdec kernel 4 (AVX-512) x3 fused", "new_a5", "w3:4,3,1"),
    ("WNX MAG 2048, AVX2 kernel G=1", "orig_a2", f"wnx:{FF}/mag_2048:1:1"),
    ("WNX MAG 2048, AVX2 kernel G=2", "orig_a2", f"wnx:{FF}/mag_2048:1:2"),
    ("WNX MAG 2048, AVX-512 kernel G=1", "orig_a5", f"wnx:{FF}/mag_2048:2:1"),
    ("WNX MAG 2048, AVX-512 kernel G=2", "orig_a5", f"wnx:{FF}/mag_2048:2:2"),
    ("WNX MAG 2048, AVX-512 G=2, fastfmt's own flags", "ffold_a5", f"wnx:{FF}/mag_2048:2:2"),
    ("WNX MAGT 2048, AVX2 kernel G=1", "orig_a2", f"wnx:{FF}/magt_2048:1:1"),
    ("WNX MAGT 2048, AVX-512 kernel G=2", "orig_a5", f"wnx:{FF}/magt_2048:2:2"),
    ("WNX JOINT 2048, AVX2 kernel G=1", "orig_a2", f"wnx:{FF}/joint_2048:1:1"),
    ("WNX JOINT 2048, AVX-512 kernel G=2", "orig_a5", f"wnx:{FF}/joint_2048:2:2"),
]

if os.environ.get("FLAGSET"):   # supplementary flag-sensitivity run
    VARS = [
        ("wex_fast_decode unchanged k1x4, -march=native", "orig_nat", "w3:1,4,1"),
        ("fastdec kernel 4 x3, x86-64-v4+vbmi2 (main table)", "new_a5", "w3:4,3,1"),
        ("fastdec kernel 4 x3, haswell+AVX-512 flags", "new_h5", "w3:4,3,1"),
        ("fastdec kernel 4 x3, -march=native", "new_nat", "w3:4,3,1"),
        ("WNX MAG AVX-512 G=2, x86-64-v4+vbmi2 (main table)", "orig_a5", f"wnx:{FF}/mag_2048:2:2"),
        ("WNX MAG AVX-512 G=2, fastfmt flags (haswell+f/bw/vl/dq)", "ffold_a5", f"wnx:{FF}/mag_2048:2:2"),
        ("WNX MAG AVX-512 G=2, haswell+AVX-512 flags incl. cd/vbmi2", "orig_h5", f"wnx:{FF}/mag_2048:2:2"),
        ("WNX MAG AVX-512 G=2, -march=native", "orig_nat", f"wnx:{FF}/mag_2048:2:2"),
    ]

def run(binary, variant, mode, passes):
    env = dict(os.environ)
    thr = {"1": "1", "4": "4", "4p": "4", "r4": "-4"}[mode]
    if mode == "4p": env["OMP_WAIT_POLICY"] = "passive"
    out = subprocess.run([f"{HERE}/bin/{binary}", "time", str(passes), thr, variant] + RECS,
                         capture_output=True, text=True, check=True, env=env).stdout
    p = [tuple(map(float, l.split()[1:3])) for l in out.splitlines() if l.startswith("PASS")]
    assert len(p) == passes, out
    return statistics.median(x[0] for x in p), statistics.median(x[1] for x in p)

def load():
    return open("/proc/loadavg").read().split()[:3]

def main():
    rounds, passes, modes = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    for mode in modes:
        vs = [v for v in VARS if not (v[2] == "ref" and mode != "1")]
        res = {v[0]: {"wall": [], "cpu": [], "ratio": [], "cratio": []} for v in vs}
        base = {"wall": [], "cpu": []}
        loads = [("start", load())]
        t0 = time.time()
        for r in range(rounds):
            order = [vs[(i + r) % len(vs)] for i in range(len(vs))]
            prev = run(BASE[1], BASE[2], mode, passes); base["wall"].append(prev[0]); base["cpu"].append(prev[1])
            for v in order:
                x = run(v[1], v[2], mode, passes)
                nxt = run(BASE[1], BASE[2], mode, passes); base["wall"].append(nxt[0]); base["cpu"].append(nxt[1])
                d = res[v[0]]
                d["wall"].append(x[0]); d["cpu"].append(x[1])
                d["ratio"].append((prev[0] + nxt[0]) / 2 / x[0]); d["cratio"].append((prev[1] + nxt[1]) / 2 / x[1])
                prev = nxt
            loads.append((f"after round {r + 1}", load()))
            print(f"mode {mode} round {r + 1}/{rounds} done, load {load()}, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
        json.dump({"mode": mode, "recs": RECS, "passes": passes, "rounds": rounds, "base": base, "res": res, "loads": loads},
                  open(f"{HERE}/logs/raw_{mode}{os.environ.get('FLAGSET', '')}.json", "w"), indent=1)
        title = {"1": "1 thread per record", "4": "4 threads per record (OpenMP default wait policy)",
                 "4p": "4 threads per record, OMP_WAIT_POLICY=passive", "r4": "4 records at once, 1 thread each (throughput; ms = wall / records)"}[mode]
        print(f"\n## {title}: {len(RECS)} records ({' '.join(RECS)}), {rounds} rounds, {passes} timed passes per run (median pass)")
        print(f"load average: {loads[0][1]} at start, {loads[-1][1]} at end; per round: {[l[1][0] for l in loads[1:]]}")
        print("| variant | binary:variant | wall ms/record median (min..max) | MB/s median (min..max) | speed-up vs A, paired, median (min..max) | CPU ms/record median | CPU-time speed-up median (min..max) |")
        print("|---|---|---:|---:|---:|---:|---:|")
        def row(name, bv, w, c, ra=None, cr=None):
            med = statistics.median(w)
            s = f"| {name} | {bv} | {med:.2f} ({min(w):.2f}..{max(w):.2f}) | {RAW / med / 1e3:.0f} ({RAW / max(w) / 1e3:.0f}..{RAW / min(w) / 1e3:.0f}) | "
            s += "1 |" if ra is None else f"{statistics.median(ra):.2f}x ({min(ra):.2f}..{max(ra):.2f}) |"
            s += f" {statistics.median(c):.2f} | " + ("1 |" if cr is None else f"{statistics.median(cr):.2f}x ({min(cr):.2f}..{max(cr):.2f}) |")
            print(s)
        row(BASE[0], f"{BASE[1]}:{BASE[2]}", base["wall"], base["cpu"])
        for v in vs:
            d = res[v[0]]
            row(v[0], f"{v[1]}:{v[2].replace(FF + '/', '')}", d["wall"], d["cpu"], d["ratio"], d["cratio"])
        sys.stdout.flush()

main()
