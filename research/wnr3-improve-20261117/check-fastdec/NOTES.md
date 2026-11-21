# check-fastdec: reproduction of the 'fastdec' lane (faster decoder, same format)

All files here; the lane's directory \<TMP>/wnr3-research/fastdec was only read.

## What was rerun
- `build.sh` (copied) -> out/test_z, test_h, test_orig, ... built clean with gcc 13.
- extra baselines: out/test_orig_z (unchanged header, -march=x86-64-v4 -mavx512vbmi2), out/test_orig_l6 (-DWEX_FAST_LANES=6).
- `wex_fast_decode.orig.h` == repo `wex4/research/fast-cpu-decode/wex_fast_decode.h` (cmp identical);
  `patch -p4` of `wex_fast_decode.patch` onto a copy of the repo dir gives exactly the lane's `wex_fast_decode.h`.
- Independent correctness check (own code, `check_ref.c`): production CPU path `wex_nr3_decode`
  (libwex_nr3_tools.so -> wex_decode_nr2_record, owner check, full 18,800,640-B slab) vs the patched header's
  default entry point `wex_fast_decode` (tables built per call), full-slab memcmp + FNV-1a.
  logs/check_ref_z_default_121.txt (kernel 4 x 3): 121/121 equal, bad 0.
  logs/check_ref_h_default_121.txt (kernel 3 x 4): 121/121 equal, bad 0.
  Checker negative control: one flipped byte in a copy of rec_20_073 -> ref -1, fast -1, counted bad.
- Harness equal, AVX-512 build, all 29 variants x {1,2} threads + default: logs/equal_z_121.txt (see final count there).
- mutate, new seed 31337, 1000 cases, AVX-512 build: 29,000 fast decodes, 0 mismatches; 946,960 weight streams in
  kernel 4 groups, 72,576 scale streams in the AVX-512 scale kernel (logs/mutate_z_1000_seed31337.txt).
- negcontrol, the three stream-hash controls (k4 weights, k4 scales, k3 weights), 1500 cases: 3/3 caught
  (logs/negcontrol_hash_1500.txt).
- Timing (short, machine load average ~11, interleaved, 1 thread, thread CPU time, same 3 records):
  logs/time_check_cpu1.txt (5 rounds) and logs/time_check_cpu1_vs_k1x6.txt (9 rounds).

## Timing results (1 thread CPU time, median ms/record, median per-round ratio)
| | run A (5 rounds, row1 = orig k1x4) | run B (9 rounds, row1 = orig k1x6) |
|---|---|---|
| orig k1x4 (shipped default, -march=haswell) | 109.3, 1.000 | 108.9, 1.137 |
| orig k1x6 | 85.6, 0.855 (5/5) | 93.6, 1.000 |
| orig built -march=x86-64-v4, k1x6 | 88.4, 0.805 | 87.3, 0.965 |
| new k3 x 4 (AVX2 build) | 61.2, 0.568 (5/5) | 56.5, 0.577 (9/9) |
| new k4 x 3 (AVX-512 build) | 52.0, 0.484 (5/5) | 48.2, 0.511 (9/9) |

So vs the shipped default the gain reproduces (k4 ~2.1x, k3 ~1.75x, all rounds), a bit smaller than the lane's
2.2-2.3x / 1.8-2.0x. Against the best setting of today's decoder ON THIS CPU (k1 x 6 streams, a compile-time option
already in the header) k4 is ~1.95x and k3 ~1.73x. The lane's headline used k1x4, which is the repo's best on the
production CPUs, so the comparison is fair for the shipped default but inflates the ratio on this VM by ~10-15%.

## Not rerun
4-thread wall-clock A/B, 1-thread wall-clock A/B, ablation breakdown, scale-kernel -4% A/B, full 12-control negcontrol,
UBSan mutate, AVX2 all-variant equal on 12 records. (Timings are left to the joint timing phase.)

Final: logs/equal_z_121.txt = 121/121 records, 29 variants x {1,2} threads + default, 0 mismatches, 0 FAIL.
