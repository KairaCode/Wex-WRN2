# One-session re-timing of the fastdec and fastfmt speed claims

Everything ran in one session, 05:18 to 05:50, on the shared 4-vCPU Xeon @ 2.1 GHz. gcc reports `-march=native` as sapphirerapids.
- Load average was 1.22 before the main run. During the 1-thread rounds it was 1.00-1.04, which is my own single process, so no other agent was running. During the 4-thread rounds it rose to 3.4-3.6, which is my own 4 threads. It was 2.74 after the main run and 3.48 after the flag run (logs/session.txt). The machine was effectively idle.
- Compiler: gcc 13.3.0, a single driver source `tdrv.c` for every variant. The AVX2 set is `-O3 -march=haswell -fopenmp`. The AVX-512 set is `-O3 -march=x86-64-v4 -mavx512vbmi2 -fopenmp`. `build.sh` builds them, and `build_flags.sh` builds the flag-sensitivity builds.
- Records: 9 TEST records, the same for every variant: 00_007 05_080 10_153 15_034 20_107 25_180 30_061 35_134 39_120. These are fastfmt's 8 timing records plus the L39 E120 outlier, which fastdec also used. MB/s = 18,800,640 decoded B / ms per record.
- Method (`timing.py`): the baseline A is the unchanged Wex-WRN2 `wex_fast_decode.h`, kernel 1 x 4 fused, AVX2 flags. Each round runs A X1 A X2 A ... A, with the order of the X variants rotated each round. One run is a separate process: 1 warm-up pass, then 3 timed passes over the 9 records, and the run's value is the median pass, wall clock. The speed-up of X in a round is the mean of the A runs around it divided by X. Tables report the median and min..max over rounds. The main run had 7 rounds per mode; the flag run had 9 rounds. Raw samples are in logs/raw_*.json and the full tables in logs/timing_tables.md and logs/timing_flags.md. Those tables also give process-CPU time, which agrees with wall within 2%.
- Equality: decode into a buffer poisoned with 0xa5, then memcmp the whole 18.8 MB slab with Z(source), at 1 and 4 threads.
  - First, Z(source) itself was checked on all 121 records with the reference `wex_nr2_decode` on the stored records: 0 mismatches.
  - WNX: I re-encoded all 121 records with fastfmt's `ff` and TRAIN model. The 80 TRAIN records are in-sample, but they are used only for equality, not for size. All 121 round trips are equal. My 41 TEST re-encodes are byte-identical to fastfmt/out for mag, magt and joint. Timing used fastfmt/out.
  - Negative control: a build that flips one output bit was caught on every record (bin/negctl_a5).
  - Result: 29/29 variant equality runs had 0 mismatches (logs/eq_summary.txt, logs/eq_flags_summary.txt).

## Main table (main run, 7 rounds; wall clock; median (min..max))

| variant | ISA | 1 thr ms/record | 1 thr MB/s | speed-up vs wex_fast_decode, 1 thr | 4 thr ms/record | 4 thr MB/s | speed-up, 4 thr | equal 121/121 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| WNR3 reference wex_nr2_decode | scalar | 898.8 (855.0..927.6) | 21 (20..22) | 0.11x (0.10..0.13) | single-threaded | – | – | yes (defines Z) |
| **wex_fast_decode unchanged (k1 x4 fused), AVX2 flags = A** | AVX2 | 103.0 (89.2..115.2) | 183 (163..211) | 1 | 25.7 (22.9..29.2) | 732 (644..820) | 1 | yes, 1 and 4 thr |
| wex_fast_decode unchanged, AVX-512 flags | AVX2 code | 101.6 (87.3..107.8) | 185 (174..215) | 0.98x (0.91..1.26) | 25.8 (24.4..27.0) | 729 (696..770) | 1.04x (0.94..1.13) | yes |
| fastdec kernel 3 x4 (same format) | AVX2 | 55.8 (49.0..61.5) | 337 (306..384) | 1.91x (1.72..2.13) | 15.3 (14.0..15.8) | 1229 (1192..1348) | 1.71x (1.52..1.95) | yes |
| fastdec kernel 4 x3 (same format) | AVX-512 F/BW/VL/CD/DQ + VBMI2 | 42.8 (40.9..50.8) | 439 (370..460) | 2.40x (1.97..2.61) | 12.6 (11.8..13.3) | 1493 (1414..1595) | 2.09x (1.96..2.17) | yes |
| WNX MAG 2048, AVX2 kernel G=1 (new format) | AVX2 | 30.7 (26.2..32.1) | 613 (585..718) | 3.28x (3.01..3.93) | 7.67 (7.18..8.31) | 2452 (2262..2617) | 3.32x (3.06..3.72) | yes |
| WNX MAG 2048, AVX2 G=2 | AVX2 | 28.8 (27.3..34.3) | 653 (548..689) | 3.48x (2.70..3.90) | 7.89 (7.33..9.92) | 2383 (1895..2567) | 3.19x (2.50..3.47) | yes |
| WNX MAG 2048, AVX-512 G=1 | AVX-512 F/BW/VL | 30.5 (23.8..31.5) | 616 (597..790) | 3.51x (2.97..4.30) | 6.73 (6.09..7.14) | 2794 (2632..3089) | 3.88x (3.59..4.23) | yes |
| WNX MAG 2048, AVX-512 G=2 | AVX-512 F/BW/VL | 22.2 (18.4..23.9) | 846 (787..1021) | 4.54x (4.35..5.87) | 5.74 (5.15..6.26) | 3275 (3001..3653) | 4.46x (4.05..4.87) | yes |
| WNX MAG 2048, AVX-512 G=2, fastfmt's flags (haswell+f/bw/vl/dq) | AVX-512 | 18.7 (17.4..22.3) | 1005 (844..1083) | 5.41x (4.65..5.91) | 5.17 (4.78..6.24) | 3639 (3013..3934) | 5.13x (4.37..5.26) | yes |
| WNX MAGT 2048, AVX2 G=1 | AVX2 | 30.5 (26.6..31.8) | 617 (592..707) | 3.37x (3.13..3.89) | 7.48 (6.91..7.70) | 2513 (2443..2720) | 3.58x (3.30..3.97) | yes |
| WNX MAGT 2048, AVX-512 G=2 | AVX-512 F/BW/VL | 22.6 (18.1..23.9) | 831 (788..1040) | 4.42x (4.27..5.64) | 6.06 (4.91..6.77) | 3100 (2779..3830) | 4.13x (3.65..5.24) | yes |
| WNX JOINT 2048, AVX2 G=1 | AVX2 | 35.2 (29.5..38.8) | 534 (485..638) | 2.98x (2.69..3.36) | 8.37 (7.87..9.61) | 2247 (1956..2389) | 3.05x (2.64..3.27) | yes |
| WNX JOINT 2048, AVX-512 G=2 | AVX-512 F/BW/VL | 24.1 (19.4..25.1) | 779 (749..967) | 4.40x (3.59..5.34) | 5.75 (5.09..6.25) | 3268 (3006..3697) | 4.63x (4.01..5.29) | yes |

Other modes from the same session (logs/timing_tables.md):
- 4 threads with OMP_WAIT_POLICY=passive: A 26.4 ms; k3 1.81x; k4 2.07x; MAG AVX2 G1 3.52x; MAG AVX-512 G2 4.92x; MAGT 3.52x / 4.21x; JOINT 3.16x / 4.66x.
- 4 records at once, 1 thread each: A 35.0 ms/record (wall / records); k3 1.90x; k4 2.36x; MAG 3.49x / 5.14x; MAGT 3.33x / 4.64x; JOINT 3.07x / 4.40x.

## Flag sensitivity (second run, 9 rounds, logs/timing_flags.md)
- In this run, the same binaries as the main table gave k4 2.19x and WNX MAG AVX-512 G=2 4.50x at 1 thread (main run: 2.40x and 4.54x). So between runs, a variant's median speed-up moves by up to about 10%.
- k4 with other flags: haswell+AVX-512 2.08x, -march=native 2.16x.
- WNX MAG AVX-512 G=2 with other flags: fastfmt's flags 4.98x, haswell+AVX-512 including cd/vbmi2 5.33x, native 5.02x.
- At 4 threads, all of these land at 2.11-2.19x (k4) and 4.59-4.93x (WNX).
- The unchanged header built with -march=native: 0.98x / 1.00x.
- So the WNX AVX-512 kernel depends on -march/-mtune by about ±10% (4.5-5.4x). Kernel 4 depends on it by about ±5%.

## Verdicts on the lanes' claims
- **fastdec kernel 3** (claimed 1.8-2.0x at 1 thread): confirmed, 1.91x. At 4 threads it is 1.71x default / 1.81x passive; the lane's ~1.55x came from the overloaded machine.
- **fastdec kernel 4** (claimed 2.2-2.3x): confirmed, 2.19-2.40x across two runs. At 4 threads 2.07-2.19x; the lane claimed ~1.6x.
- **WNX speed-ups vs the unchanged decoder:**

  | variant | claimed | measured here |
  |---|---|---|
  | MAG AVX2 | 3.49-3.54x | 3.28x (G1), 3.48x (G2) |
  | MAGT AVX2 | 3.39-3.40x | 3.37x |
  | MAG AVX-512 | 4.57-4.87x | 4.50-5.41x, depending on flags |
  | MAGT AVX-512 | 4.20-4.82x | 4.42x |
  | JOINT | 3.06x / 4.19x | 2.98x / 4.40x |

  All confirmed within the run-to-run and flag spread.
- **Not reproduced:**
  - "AVX2 G=2 is 3-10% slower": here G=2 is 6% faster at 1 thread and 4% slower at 4 threads, which is noise-level. Either G is fine.
  - "with the default spinning OMP policy 4 threads give only 1.9-2.3x": on the quiet machine the default and passive policies are the same within noise. The lane's figure came from oversubscription.
- **Thread scaling, measured for the first time on an idle machine:**
  - A: 4.0x from 1 to 4 threads (103.0 to 25.7 ms).
  - k3: 3.65x.
  - k4: 3.4x.
  - WNX MAG: 4.0x (AVX2 G1), 3.9x (AVX-512 G2).
- **Fairer comparison for the format decision:** the new format should be compared with the best decoder for today's format, not with the unchanged one.
  - WNX MAG AVX-512 G=2 against kernel 4: 42.8/22.2 = 1.9x at 1 thread, 12.6/5.74 = 2.2x at 4 threads.
  - WNX MAG AVX2 against kernel 3: 55.8/30.7 = 1.8x at 1 thread, 15.3/7.67 = 2.0x at 4 threads.
  - Ratios of medians, not paired.

## What transfers (inferred, not measured)
Everything above was measured on this Xeon only.
- **i7-7700K** (AVX2, no AVX-512): only the AVX2 rows apply. Kernel 3 and the WNX AVX2 scale kernel use vpgatherdd, which is slow on Kaby Lake and slower again under the Downfall/GDS microcode. The AVX2 ratios there are unknown.
- **Ryzen 7950X3D** (Zen 4): it has F/BW/VL/DQ/CD/VBMI2, so every row can run on it. But it executes 512-bit operations as two halves and its gathers differ, so the AVX-512 ratios need measuring there.
- **GPU**: nothing here says anything about GPU speed.

## Rerun
`./build.sh && ./build_flags.sh && ./eq_all.sh && python3 timing.py 7 3 1 4 4p r4 > logs/timing_tables.md && FLAGSET=_flags python3 timing.py 9 3 1 4 > logs/timing_flags.md`
`eq_all.sh` needs about 2 GB of free disk at a time; it deletes its re-encodes after use. Spot re-checks of the lanes' mutation tests passed: fastfmt bench mutate 200 cases, fastdec test_z mutate 300 cases, 0 mismatches (logs/mutate_*_recheck.txt).
