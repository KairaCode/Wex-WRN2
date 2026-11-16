# Fast CPU decode of WNR3 records (no format change)

Experiment: a bit-exact CPU decoder for today's WNR3 records that is much faster than the reference
`wex_nr2_decode` (`c/wex_neural_rans.h`). Nothing on disk changes: record bytes, model blob and container are
untouched, so the size effect is exactly 0 bytes.

## Result in one table

Best of 5, decoded output 18,800,640 B per record, MB = 10^6 bytes. "8 threads" for the reference means 8
records decoded at once (it has no threads of its own).

| machine | decoder | 1 thread ms/record | MB/s | 8 threads ms/record | MB/s |
|---|---|---:|---:|---:|---:|
| desktop, Ryzen 9 7950X3D | reference | 505.2 | 37.2 | 75.9 (8 records at once) | 247.7 |
| desktop | `wex_fast_decode` configuration (kernel 1, 4 streams, fused) | 71.6 | 262.5 | 13.8 | 1361.5 |
| laptop, i7-7700HQ | reference | 1077.2 | 17.5 | 233.5 (8 records at once) | 80.5 |
| laptop | same configuration | 129.2 | 145.5 | 38.0 | 494.6 |

Speed-up over the reference on one thread: 7.1x (desktop), 8.3x (laptop). With 8 threads against the
single-threaded reference: 36.6x (desktop), 28.3x (laptop); against 8 reference decodes at once: 5.5x and 6.1x.
For scale: the GPU batch decode figure quoted in the repo docs is about 0.37 ms per record (not measured
here), i.e. the fast CPU path with 8 desktop threads is about 37x slower than that.

These numbers are x86 (AVX2). Kernel 1 also has a NEON version for 64-bit ARM, added on 2026-11-14 and
measured on synthetic records only, on a Neoverse-N2 and on Apple M1 virtual machines: see "NEON kernel
(AArch64)" below. Kernel 2 has an SVE version for ARM
CPUs with SVE, added on 2026-11-15; on the one SVE CPU it ran on it is slower than NEON kernel 1 and is not
the default: see "SVE kernel 2 (AArch64)".

## Files

| file | purpose |
|---|---|
| `wex_fast_decode.h` | the decoder (C11 + GNU builtins, includes `wex_format.h`) |
| `test_fast_decode.c` | harness: `equal`, `mutate`, `bench`, `speed` |
| `build.sh` | x86: builds the harness three ways into `out/` (AVX2 + OpenMP, scalar fallback, UBSan trap build) |
| `synth.sh` | any CPU, no real model: builds the harness with the SIMD kernels of the CPU and with the scalar kernel into `out/synth/` and runs it on the 4 synthetic records of `wex4/out/synth_nr3.wex` (`check`, run by `scripts/check_all.sh`, or `speed`); `lanes` builds it once for each `-DWEX_FAST_LANES`; `WEX4_MARCH=-march=armv8-a+sve` adds the SVE kernel 2 |
| `speed_runs.py` | median and spread over several `synth.sh speed` calls, from their output or from CI job logs |
| `arm-speed-20261115.txt` | the `synth.sh speed` output of the hand-started `arm-speed` CI jobs of 2026-11-15 (section "Streams per CPU, Apple silicon, Apple clang") |
| `negative_control.sh` | removes one check at a time from a copy of the header and expects the mutation test to notice; exit 2 if it cannot read the model or a record; `WEX4_MARCH` as in `synth.sh` |
| `extract.py` | copies the model blob and chosen records out of a container opened read-only |
| `verify/run.sh` | the verifier's reruns on real records: `equal`, `bench` and `mutate` of the harness, the three steps of `verify/vh.c` (its own timer, full-size re-encodes and boundary mutations, stream tails cut against a guard page; Windows only, skipped elsewhere) and `mutate` with the stream hash check removed from both decoders. Records: the `out/H_*.rec` of `verify/scan.py`, or `VERIFY_MODEL` and `VERIFY_RECORDS` (WNR3-CZ records as `extract.py` writes them: `vh.c reenc` encodes with the canonical-zero flag). `CC` and `WEX_OPENMP` as in `synth.sh`; `-march=haswell` on x86-64 only, `WEX4_MARCH` overrides it. Last line: `verify run.sh passed` |
| `out/` | build products, data copies, logs (not in git) |

API (same view / layout / model / output arguments as `wex_nr2_decode`):

```c
int  wex_fast_decode(v, layout, model, scales_out, weights_out);            /* tables built per call */
void wex_fast_model_init(WexFastModel *tables, const WexNr2Model *model);   /* once per model */
int  wex_fast_decode_with(tables, v, layout, model, scales_out, weights_out, &options);
```

`WexFastModel` is 223,944 B (3 x 16384-byte scale tables + 1365 weight contexts x 128 B). Building it took
11 us on the desktop and 19 us on the laptop (mean of 100 calls); `wex_fast_decode` adds a 224 KiB `malloc`
to that on every call. Measured end to end the per-call variant is 73.7 / 14.0 ms per record against
71.6 / 13.8 with cached tables (desktop, 1 / 8 threads), a difference inside the run-to-run noise.

## How to rerun

```sh
python -I extract.py CONTAINER.wex out 0:0 7:100 13:5 20:191 33:64 39:120   # reads header, model, table rows, 6 records
sh build.sh
R="out/L0_E0.rec out/L7_E100.rec out/L13_E5.rec out/L20_E191.rec out/L33_E64.rec out/L39_E120.rec"
out/test_fast.exe   equal  out/model.wnm2 $R
out/test_fast.exe   mutate out/model.wnm2 36000 20261111 $R     # about 6 minutes
out/test_ubsan.exe  mutate out/model.wnm2 8000 777 $R
out/test_scalar.exe mutate out/model.wnm2 36000 4242 $R
out/test_fast.exe   bench  out/model.wnm2 5 $R
sh negative_control.sh out/model.wnm2 out/L0_E0.rec out/L7_E100.rec
```

## What was measured, on which data

Data: the production V4.1 REAP-192 container, read-only: header page, WNM2 model (59,408 B, id 0x27e8a1e0)
and 6 of its 7,680 records (L0 E0, L7 E100, L13 E5, L20 E191, L33 E64, L39 E120; 16.0 to 17.4 MB each, 0.852
to 0.923 of raw, all flags 0x0f = ceil form on the three matrices + canonical zero, 2,160 scale + 17,280
weight streams). About 98 MB read in total. Everything below is in-sample to these 6 records; no other
record was decoded.

### Byte equality

`equal`: reference decode, then every variant (19 option sets x {1, 8} threads) and the default entry point
must return 0 and give identical bytes; outputs are filled with 0xa5 before each fast decode.

- desktop, AVX2 build: 6 records, 0 mismatches
- desktop, scalar fallback build: 6 records, 0 mismatches
- desktop, UBSan trap build: 1 record (L7 E100), 0 mismatches, no trap
- laptop, AVX2 and scalar builds: 1 record (L13 E5), 0 mismatches

### Mutation test

The reference needs about 0.5 s per full record, so the mutation test uses mini-records: random cuts of the 6
decoded records (1 to 2,500 blocks per matrix), re-encoded with the reference encoder and the production
model, weight chunk 512..16384, scale chunk 256..16384, random owner. Two classes:

- constructed records with correct stream hashes, so the decoders run to the end: plain cuts, uniform noise,
  constant data, records without the CZ flag carrying -0 nibbles (must accept); CZ flag with a -0 nibble
  (must reject); a coded block of maximum 5 with scale 0 in a ceil matrix (must reject: ceil underflow); ceil
  flags that do not match the data; random anchor bytes;
- byte mutations of valid records: single and multiple bit flips, byte changes, header bits, length bits,
  moved stream boundaries, truncated / extended last stream, swapped neighbouring streams, anchors / flags /
  owner, initial states, runs of 0x00 / 0xff / random, stray bytes appended to one stream, last byte of a
  stream dropped when the next stream starts with the same byte (the case that separates "needs a byte at
  the end" from "reads ahead"), flips in the last 80 bytes.

Each record that passes `wex_nr3_parse` is copied without its zero padding flush against a PAGE_NOACCESS page
and decoded by the reference, by `wex_fast_decode` and by all 19 variants (threads 1 or 3); status must match
and, when both accept, the bytes. Output buffers sit flush against a guard page (alternating upper / lower),
the model blob too.

| build | records at the decoders | byte-mutated (all rejected by both) | constructed (accept / reject / either) | fast decodes | mismatches |
|---|---:|---:|---:|---:|---:|
| desktop AVX2, seed 20261111 | 36,000 | 25,400 | 8,984 / 1,083 / 533 | 684,000 | 0 |
| desktop scalar fallback, seed 4242 | 36,000 | 25,377 | 8,970 / 1,115 / 538 | 36,000 | 0 |
| desktop UBSan trap build, seed 777 | 8,000 | 5,603 | 2,047 / 230 / 120 | 152,000 | 0 |
| laptop AVX2, seed 31337 | 3,000 | 2,081 (older count: includes mutations that changed nothing) | 790 / 81 / 48 | 57,000 | 0 |

In the 36,000-record AVX2 run 3,360 more mutated records were stopped by the parser (shared code, not a test
of this decoder) and 2,633 mutations that changed no byte were dropped. No access violation and no UBSan trap
in any run. The toolchain accepts `-fsanitize=undefined -fsanitize-trap=all`.

Limits of this test, stated plainly: every byte-mutated record was rejected by both decoders, almost always
through the per-stream hash, which hides most other differences. The checks that the hash can hide are
covered by the constructed class and by `negative_control.sh`: with one check removed from the header
(canonical-zero check, ceil underflow check, end-of-stream pointer check in kernel 1 weights, in kernel 2
weights and in the scale kernel, stream hash) the mutation test reports a mismatch within 6,000 cases, 6 of 6.
Before the two end-of-stream mutations were added, a decoder without the end-of-stream check passed 6,000
cases unnoticed. The 32-symbol bound check and the slack rule are memory-safety rules; they are only
observable through the guard pages (no fault seen), not through the status.

## Speed, every variant

Desktop: Ryzen 9 7950X3D (16 cores / 32 threads), gcc 16.2 (MSYS2 UCRT64), `-O3 -march=haswell -fopenmp`,
6 records, best of 5, time per record = mean over the 6. The machine was not idle (it serves production), and
an earlier run of the same code gave 90.4 / 14.6 ms for the default configuration and 520 ms for the
reference, so single numbers carry roughly +-20%.

| decoder | 1 thread: ms/record | MB/s | 8 threads: ms/record | MB/s |
|---|---:|---:|---:|---:|
| reference wex_nr2_decode | 505.20 | 37.2 | 75.89 (8 records at once) | 247.7 |
| k0: reference loop per stream | 495.81 | 37.9 | 77.09 | 243.9 |
| k1: AVX2 search, 1 stream | 204.78 | 91.8 | 29.05 | 647.1 |
| k1: 2 streams interleaved | 124.62 | 150.9 | 19.78 | 950.3 |
| k1: 3 streams interleaved | 100.46 | 187.2 | 17.92 | 1049.3 |
| k1: 4 streams interleaved | 91.14 | 206.3 | 18.27 | 1029.1 |
| k1: 6 streams interleaved | 110.71 | 169.8 | 22.04 | 853.0 |
| k1: 8 streams interleaved | 114.94 | 163.6 | 21.94 | 857.0 |
| k1: 1 stream, fused | 187.34 | 100.4 | 25.97 | 723.8 |
| k1: 2 streams, fused | 104.07 | 180.7 | 15.64 | 1202.4 |
| k1: 3 streams, fused | 80.39 | 233.9 | 14.07 | 1335.9 |
| k1: 4 streams, fused | 71.62 | 262.5 | 13.81 | 1361.5 |
| k1: 6 streams, fused | 90.63 | 207.5 | 17.44 | 1078.0 |
| k1: 8 streams, fused | 91.60 | 205.2 | 17.76 | 1058.7 |
| k2: gather 8 streams | 139.84 | 134.4 | 20.15 | 933.0 |
| k2: gather 16 streams | 133.16 | 141.2 | 18.08 | 1039.8 |
| k2: gather 24 streams | 129.62 | 145.0 | 18.65 | 1007.9 |
| k2: gather 8 streams, fused | 119.55 | 157.3 | 16.61 | 1132.0 |
| k2: gather 16 streams, fused | 103.97 | 180.8 | 15.60 | 1205.1 |
| k2: gather 24 streams, fused | 102.82 | 182.9 | 15.80 | 1189.9 |
| wex_fast_decode_opt (k1, 4 streams, fused; tables built per call) | 73.68 | 255.2 | 14.00 | 1343.1 |

Laptop: i7-7700HQ (4 cores / 8 threads, so "8 threads" is two per core), gcc 16.2 (MSYS2 UCRT64), same flags,
1 record (L13 E5), best of 5.

| decoder | 1 thread: ms/record | MB/s | 8 threads: ms/record | MB/s |
|---|---:|---:|---:|---:|
| reference wex_nr2_decode | 1077.16 | 17.5 | 233.52 (8 records at once) | 80.5 |
| k0: reference loop per stream | 1215.24 | 15.5 | 257.14 | 73.1 |
| k1: AVX2 search, 1 stream | 379.93 | 49.5 | 68.14 | 275.9 |
| k1: 2 streams interleaved | 242.92 | 77.4 | 47.00 | 400.0 |
| k1: 3 streams interleaved | 185.76 | 101.2 | 47.35 | 397.0 |
| k1: 4 streams interleaved | 167.09 | 112.5 | 51.00 | 368.7 |
| k1: 6 streams interleaved | 250.49 | 75.1 | 57.75 | 325.6 |
| k1: 8 streams interleaved | 231.82 | 81.1 | 58.15 | 323.3 |
| k1: 1 stream, fused | 320.97 | 58.6 | 54.64 | 344.1 |
| k1: 2 streams, fused | 176.86 | 106.3 | 34.77 | 540.6 |
| k1: 3 streams, fused | 141.02 | 133.3 | 35.80 | 525.2 |
| k1: 4 streams, fused | 129.22 | 145.5 | 38.01 | 494.6 |
| k1: 6 streams, fused | 200.69 | 93.7 | 47.68 | 394.3 |
| k1: 8 streams, fused | 196.07 | 95.9 | 47.98 | 391.8 |
| k2: gather 8 streams | 231.96 | 81.1 | 43.70 | 430.2 |
| k2: gather 16 streams | 227.76 | 82.5 | 45.63 | 412.0 |
| k2: gather 24 streams | 216.78 | 86.7 | 44.82 | 419.5 |
| k2: gather 8 streams, fused | 182.39 | 103.1 | 32.25 | 582.9 |
| k2: gather 16 streams, fused | 157.44 | 119.4 | 32.93 | 570.9 |
| k2: gather 24 streams, fused | 164.49 | 114.3 | 32.35 | 581.1 |
| wex_fast_decode_opt (k1, 4 streams, fused; tables built per call) | 137.63 | 136.6 | 38.26 | 491.4 |

## What each idea gave (desktop numbers above)

| idea | measured effect |
|---|---|
| (a) OpenMP across streams, reference inner loop (k0) | 495.8 -> 77.1 ms with 8 threads (6.4x); on its own it only matches 8 reference decodes at once (75.9 ms) |
| (b) branchless search and renormalisation, per-context tables (k1, 1 stream) | 505.2 -> 204.8 ms on one thread (2.5x) |
| (c) interleaving streams in one loop | 204.8 -> 124.6 (2) -> 100.5 (3) -> 91.1 ms (4); slower again at 6 (110.7) and 8 (114.9) |
| (d) AVX2 gathers over 8 / 16 / 24 streams (k2) | 139.8 / 133.2 / 129.6 ms unfused, 119.6 / 104.0 / 102.8 ms fused: slower than k1 x4 on the desktop at 1 and 8 threads; on the laptop with 8 threads it is the fastest variant (32.3 ms against 38.0 for k1 x4 and 34.8 for k1 x2) |
| (e) hash + ceil inverse inside the loop | k1 x4: 91.1 -> 71.6 ms (-21%) on one thread, 18.3 -> 13.8 ms (-24%) with 8 |

How (b) is built, because the first attempt was much slower. The first kernel 1 searched the model blob
directly (unaligned 34-byte CDFs, context computed as (srel\*9 + previous)\*9 + maximum, 2-byte
renormalisation with two compares): 135 ms at 3 streams fused, measured on 2 records only and no longer
reproducible from the final source. The final kernel replaces the context arithmetic by a `next[]` table
inside each 128-byte context entry, loads start / frequency from the same entry, derives the renormalisation
shift from `lzcnt`, and packs 16 symbols with `pext`. One stream alone runs at about 5.3 ns per symbol, four
interleaved at about 2.0 ns (whole-record time, fused, divided by the 35.4 M weight symbols of a record).

Why 4 streams and not 8: only the measurement is certain. A plausible reading (not verified with performance
counters) is that the per-symbol dependency chain is long, so interleaving helps until the kernel's live
state no longer fits the 15 general registers and the out-of-order window; the compiler's code for 6 and 8
streams spills state to the stack. On ARM the fastest number differs by CPU (6 on a Neoverse-N2, 8 on Apple
M1 virtual machines, synthetic records): "Streams per CPU, Apple silicon, Apple clang" below.
`-DWEX_FAST_LANES` sets the number that `wex_fast_decode` uses; the default stays 4.

Thread scaling on the desktop is 5.2x from 1 to 8 threads for the default configuration; 16 and 32 threads
were not measured.

## NEON kernel (AArch64)

Added on 2026-11-14. Kernel 1 is the same loop on x86 and on 64-bit ARM: same tables, same scale kernel,
same hash and ceil inverse, same bound checks and slack rule. Three steps use other instructions:

| step | AVX2 build | NEON build |
|---|---|---|
| symbol search over the 16 `upper[]` values of a context | one 256-bit signed compare, `movemask`, `tzcnt` | two 128-bit signed compares (`vcgtq_s16`), one byte kept of each 16-bit lane (`vuzp1q_u8`), 4 bits kept of each byte (`vshrn_n_u16`), trailing zeros of the 64-bit mask |
| renormalisation shift | `lzcnt` | `clz` |
| 16 symbols into 8 bytes | two `pext` | `vsraq_n_u16` and `vmovn_u16` |

Kernel 2 has no NEON version: NEON has no gather load. A NEON build that is asked for kernel 2 runs kernel 1
with `WEX_FAST_LANES` streams (4 by default). A build for SVE has an SVE kernel 2 ("SVE kernel 2 (AArch64)" below). `-DWEX_FAST_SCALAR`
builds kernel 0 alone, as on x86.

### What was run on ARM

Everything below ran on 2026-11-14 in the `linux-arm` job of `.github/workflows/ci.yml`: GitHub runner
`ubuntu-24.04-arm`, Neoverse-N2, 4 logical CPUs, gcc 13.3.0, `-O3 -fopenmp`. Data: the 4 records of
`wex4/out/synth_nr3.wex`, which `wex4/build_and_test.sh` writes from `fixtures/make_synth_mxfp4.py`
(208,896 decoded bytes each, 24 scale and 192 weight streams, flags 0x00, model trained on the same 4
records). No real record was decoded on ARM.

The numbers come from three runs of the CI workflow on the branch of this change (GitHub Actions run
ids; the logs are kept by GitHub for a limited time):

| run | what it gives here |
|---|---|
| 43618200630 | the "earlier run" of the speed notes. Its `linux-arm` job ended red: a temporary step called `negative_control.sh` with a relative model path, which the script could not read then |
| 43618569181 | the speed table, the `bench` figure, the negative controls and the clang build |
| 43619432196 | the first run without the two temporary steps (negative controls, clang) |

The decoder, the harness and the scripts they call were the same in the three runs; the runs differ in
the temporary steps and in documentation. The correctness results of the first two bullets were the same
in all three.

- `synth.sh check`, NEON build: `equal` on 4 records x 13 variants x {1, 8} threads + default, 0
  mismatches; `mutate` with 600 mini-records (205 accepted and 395 rejected by the reference), 7,800 fast
  decodes, 0 mismatches. The scalar build of the same run: 0 mismatches. The job fails if the build is
  not the NEON one.
- `wex4/build_and_test.sh`, where `wex4_tools.dll` holds the fast decoder and `wex4_tools_ref.dll` the
  reference: all tests passed with the NEON kernel and again with `WEX4_MARCH=-DWEX_FAST_SCALAR`
  (112,399 C mutations each time, same verdicts from both decoders).
- `negative_control.sh` on the synthetic records caught 5 of 5 broken decoders (the kernel 2 control
  does not apply). The harness built with clang 18.1.3 without OpenMP gave 0 mismatches in `equal` and
  in `mutate` with 3,000 mini-records (39,000 fast decodes). Both in run 43618569181, as temporary
  steps. They are steps of the job since then: the negative controls as above, the clang build with
  `equal` and `mutate` on 600 mini-records. The job fails if a broken decoder is not caught, if the
  clang build is not the NEON one or if it gives a mismatch.
- From this change on the harness has guard pages off Windows too (`mmap` + `mprotect`). No fault in any
  of these runs.

Source records without the CZ flag, such as the synthetic ones, can hold -0 nibbles; `mutate` now sets
them to +0 before it cuts mini-records (the production records have none, so nothing changes for them).

### Speed on the CI runner

`bash synth.sh speed 21`: one thread, one run is one pass over the 4 records after a pass that is not
counted, the table gives the median and the best of 21 runs. Same runner, compiler and flags as above; a
shared virtual machine, load unknown.

| decoder | median ms/record | MB/s | best ms/record | scalar kernel median / this median |
|---|---:|---:|---:|---:|
| reference wex_nr2_decode | 5.633 | 37.1 | 5.621 | 0.98 |
| k0: reference loop per stream (the scalar kernel) | 5.540 | 37.7 | 5.482 | 1.00 |
| k1: NEON search, 1 stream | 3.723 | 56.1 | 3.719 | 1.49 |
| k1: 2 streams interleaved | 2.111 | 99.0 | 2.110 | 2.62 |
| k1: 3 streams interleaved | 1.684 | 124.1 | 1.682 | 3.29 |
| k1: 4 streams interleaved | 1.403 | 148.9 | 1.400 | 3.95 |
| k1: 6 streams interleaved | 1.396 | 149.7 | 1.394 | 3.97 |
| k1: 8 streams interleaved | 1.859 | 112.4 | 1.857 | 2.98 |
| k1: 1 stream, fused | 3.678 | 56.8 | 3.676 | 1.51 |
| k1: 2 streams, fused | 2.050 | 101.9 | 2.047 | 2.70 |
| k1: 3 streams, fused | 1.673 | 124.9 | 1.671 | 3.31 |
| k1: 4 streams, fused (the `wex_fast_decode` configuration) | 1.406 | 148.6 | 1.404 | 3.94 |
| k1: 6 streams, fused | 1.392 | 150.0 | 1.391 | 3.98 |
| k1: 8 streams, fused | 1.826 | 114.4 | 1.824 | 3.03 |

- NEON kernel 1 against the scalar kernel, default configuration: 5.54 -> 1.41 ms per record, 3.9x. The
  table is run 43618569181. The two other runs of the job the same day gave 5.542 -> 1.408 ms (run
  43618200630, earlier) and 5.544 -> 1.406 ms (run 43619432196, later). The scalar build of the same
  source measured 5.475, 5.457 and 5.476 ms for kernel 0 in these three runs, in the same order.
- 4 and 6 streams are equal within 1%; the default stays at 4 streams, as on x86. Eighteen more calls on
  six VMs and the Apple M1 VMs of macOS: "Streams per CPU, Apple silicon, Apple clang" below.
- Fusing the hash into the loop changes nothing measurable here (1.403 against 1.406 ms), unlike on the
  x86 desktop with real records (-21%).
- `bench` in the same job (best of 15): the default configuration with 8 threads on the 4 CPUs took 0.47
  ms per record against 1.41 ms on one thread (0.48 and 0.47 ms in the two other runs). The records are small (216 streams), so this says little
  about thread scaling on full records.
- The harness built with clang 18.1.3 (no OpenMP), run 43618569181 only: 5.97 -> 1.48 ms, 4.0x. The
  clang step of the job as committed does not measure speed.
- For scale, the same script on the x86 desktop (AVX2, gcc 16.2, one run, machine not idle): 6.87 -> 1.16
  ms, 5.9x, where real records gave 7.1x. Whether the synthetic records understate the gain on ARM too
  was not measured.

Not checked on ARM: real records; the speed on any CPU other than the Neoverse-N2 of the Linux runner and
the Apple M1 virtual machine of the macOS runner (subsection "Streams per CPU, Apple silicon, Apple clang"
below; no Apple silicon outside a VM, no M2 or later, no Ampere, no Cortex-A); big-endian AArch64 (it builds the scalar kernel, never run); 32-bit ARM
(scalar kernel, never run).

Under ASan and UBSan (added on 2026-11-15, no report): the `linux-arm-sanitizers` CI job runs
`scripts/check_all.sh` and the fuzz replay with the sanitizer flags of `linux-sanitizers` on
ubuntu-24.04-arm (gcc 13.3), so the NEON build runs under both in `wex4_tools.dll`, in this harness
(`synth.sh check`) and in `fuzz_wnr3_record`; `macos-clang` does the same on macOS with Apple clang. The
Fuzz workflow runs `fuzz_wnr3_record` for 60 s on ARM64 Linux and on macOS, where it compares the NEON
kernel with the reference decoder under libFuzzer. Not run on ARM: ThreadSanitizer and MemorySanitizer.

macOS: the `macos` CI job (macos-15 on Apple silicon, Homebrew GCC 15.3.0 on 2026-11-14) builds the NEON
kernel and fails if it built another one. It runs `scripts/check_all.sh`, so the WEX4 tests and
`synth.sh check` compare that build with the reference decoder there, and `wex4/fuzz/run.sh --replay`. The
`macos-clang` job does the same with Apple clang (17.0.0 on 2026-11-15) and the `libomp` of Homebrew
(`WEX_OPENMP`), and fails unless the build is NEON with OpenMP. Neither runs `synth.sh speed`, the
negative controls or the WEX4 tests with the scalar kernel forced: those are steps of `linux-arm` only.
The `arm-speed` jobs, which only a run started by hand has, run `synth.sh lanes` and `synth.sh speed` on
macos-15 with Homebrew GCC and with Apple clang (below).

### Streams per CPU, Apple silicon, Apple clang

Added on 2026-11-15, for three questions that the list above left open: the speed of the NEON kernel on
Apple silicon, a build with Apple clang, and the number of streams interleaved by kernel 1 that is fastest
on each CPU.

What changed in the code:

- `-DWEX_FAST_LANES=N`, N one of 1, 2, 3, 4, 6, 8 (default 4), sets the streams of kernel 1 in
  `wex_fast_decode`, on NEON and on AVX2; any other N stops the build with `#error`. An explicit `lanes` of
  1, 2, 3, 4, 6 or 8 in `WexFastOptions` is kept; only another value becomes N, and so does
  a request for kernel 2 on NEON (before, it got 4). The six loops are compiled
  in every build, so N changes the speed of the default entry point and nothing else.
- `synth.sh lanes` builds the harness once per N and compares the default entry point with the reference
  decoder: `equal` on the 4 records (1 and 8 threads), `mutate` with 200 mini-records. It checks from the
  `equal` line that the build took N, and that N = 5 does not compile. The `linux` (AVX2) and `linux-arm`
  (NEON) CI jobs run it on every push; it also passed on Windows x86-64 with MSYS2 gcc 16.2 (AVX2 build).
- `synth.sh` takes `WEX_OPENMP` in place of `-fopenmp`: that is how Apple clang, which has no OpenMP
  runtime of its own, gets the `libomp` of Homebrew. Set and empty, it builds without OpenMP (the
  harness then runs one thread only). `speed_runs.py` gives median and spread over several
  `synth.sh speed` calls, also from whole job logs (`--compiler` keeps the calls of one compiler).

Where it ran: the `arm-speed` jobs of `.github/workflows/ci.yml` (input `arm_speed` of a run started by
hand), in two runs on 2026-11-15 with 3 jobs per runner image each, so 6 virtual machines of each kind.

| | ubuntu-24.04-arm | macos-15 |
|---|---|---|
| runner image | ubuntu24-arm64 20261111.182.1 | macos15 20261104.0514.1 |
| CPU (`lscpu`; `sysctl machdep.cpu.brand_string hw.model`) | Neoverse-N2, 4 logical CPUs | Apple M1 (Virtual), VirtualMac2,1, 3 logical CPUs, 7 GiB |
| compilers and flags | gcc 13.3.0 (Ubuntu), `-O3 -fopenmp` | Homebrew GCC 15.3.0, `-O3 -fopenmp`; Apple clang 17.0.0 (clang-1700.0.13.5), `-O3 -Xclang -fopenmp`, the include and lib folders of Homebrew libomp 23.1.5, `-lomp` |
| `synth.sh speed 21` calls | 3 per VM, 18 in all | 3 per compiler per VM, the two compilers taking turns: 18 per compiler |

The runs are GitHub Actions run ids 43717734347 (the first commit of this change, before it was rebased
onto `main`) and 43740953159 (the next commit). Both built the same harness with the same `synth.sh
speed`; the decoder differs by the line that keeps an explicit `lanes = 4`, which does nothing in these
builds (default N). The output of all 54 calls is in `arm-speed-20261115.txt`;
`python -I speed_runs.py --compiler TEXT arm-speed-20261115.txt` (TEXT `gcc (Ubuntu`, `Homebrew GCC` or
`Apple clang`) prints the full tables that the ones below are taken from. Data: the 4 records of
`wex4/out/synth_nr3.wex` (synthetic, 208,896 decoded bytes each, 24 scale and 192 weight streams, flags
0x00), one thread. One call gives, per row, the median of 21 passes over the 4 records.

Byte identity: in all 12 jobs `synth.sh lanes` passed with each compiler of the job (6 builds per compiler):
for every N, `equal` and `mutate` (200 mini-records, 100 accepted and 100 rejected by the reference) gave 0
mismatches, and N = 5 did not compile. The jobs of the second run built with `-Werror` (no compiler gave a
warning; the logs of the first run show none either) and checked that every build was the NEON kernel with
OpenMP and that `clang` was Apple clang.

Kernel 1 with the fused hash (the `wex_fast_decode` configuration), ms per record: the median of the 18 call
medians, in brackets the lowest and the highest call median.

| streams | Neoverse-N2, gcc 13.3 | Apple M1 VM, Homebrew GCC 15.3 | Apple M1 VM, Apple clang 17 |
|---:|---:|---:|---:|
| 1 | 3.712 (3.671 to 3.752) | 5.766 (5.167 to 6.495) | 5.925 (4.960 to 8.759) |
| 2 | 2.054 (2.047 to 2.065) | 3.032 (2.690 to 3.859) | 3.143 (2.566 to 4.027) |
| 3 | 1.678 (1.673 to 1.687) | 2.250 (1.982 to 2.765) | 2.487 (1.931 to 3.154) |
| 4, the default | 1.408 (1.406 to 1.416) | 1.732 (1.509 to 2.406) | 1.808 (1.463 to 2.994) |
| 6 | 1.397 (1.393 to 1.403) | 1.607 (1.412 to 1.982) | 1.827 (1.423 to 2.409) |
| 8 | 1.830 (1.827 to 1.835) | 1.546 (1.383 to 2.570) | 1.514 (1.305 to 2.799) |
| scalar kernel (`-DWEX_FAST_SCALAR` build) | 5.474 (5.431 to 5.499) | 8.368 (7.211 to 10.043) | 9.067 (8.097 to 15.662) |

Each call against itself: the time of 6 or 8 streams divided by the time of 4 streams in the same call,
median over the 18 calls, and in how many calls the row was faster than 4 streams.

| kernel 1 | streams | Neoverse-N2, gcc | Apple M1 VM, Homebrew GCC | Apple M1 VM, Apple clang |
|---|---:|---:|---:|---:|
| fused | 6 | 0.991, 18 of 18 | 0.936, 16 of 18 | 0.983, 11 of 18 |
| fused | 8 | 1.300, 0 of 18 | 0.905, 14 of 18 | 0.865, 13 of 18 |
| not fused | 6 | 0.994, 18 of 18 | 0.956, 15 of 18 | 0.951, 14 of 18 |
| not fused | 8 | 1.330, 0 of 18 | 0.908, 15 of 18 | 0.877, 15 of 18 |

1, 2 and 3 streams were slower than 4 in every call on the Neoverse-N2 and in all but at most 2 of the 18
calls on the M1 VMs.

- Apple silicon: on the M1 VMs the default configuration took 1.73 ms per record with Homebrew GCC and
  1.81 ms with Apple clang, against 8.37 and 9.07 ms for the scalar kernel of the same compiler: 4.8 and
  5.0 times faster. On the Neoverse-N2: 5.47 -> 1.41 ms, 3.9x, as on 2026-11-14. The calls of one row on
  macOS lay 23% to 99% of the median apart (on the Neoverse-N2 at most 2.8%), so one macOS number alone
  says little.
- Apple clang against Homebrew GCC on the same VMs: every row of the two compilers lies inside the range
  of the other (4 streams fused 1.808 against 1.732 ms, 8 streams fused 1.514 against 1.546 ms).
- The fastest number of streams is not the same on the two ARM CPUs. Neoverse-N2: 6 streams, 1.397
  against 1.408 ms for 4 (0.8% less, faster in 18 of 18 calls, the two ranges apart); 8 streams take 30%
  more. Apple M1 VMs: 8 streams, 1.546 against 1.732 ms with Homebrew GCC (10.7% less) and 1.514 against
  1.808 ms with Apple clang (16.2% less), faster in 14 and 13 of 18 calls, the ranges of 4 and 8 streams
  overlapping. On both x86 machines with real records (section "Speed, every variant", best of 5, one
  thread, fused) the fastest is 4 streams; 6 and 8 streams took 27% and 28% more on the desktop, 55% and
  52% more on the laptop.
- **The default stays at 4 streams.** No value is the fastest on every CPU measured: 6 on the
  Neoverse-N2, 8 on the M1 VMs, 4 on the two x86 machines. 6 streams (fused) beat 4 in 18 of 18 calls on
  the Neoverse-N2, by 0.8%, and in 16 and 11 of 18 calls on the M1 VMs, by less than the spread between
  calls (with Apple clang its median is even 1% above that of 4 streams); on both x86 machines it was
  slower. So a separate NEON default of 6 was not adopted either. A build for one CPU can set
  `-DWEX_FAST_LANES` itself; for 8 on Apple silicon the evidence is these VMs and synthetic records only.

Limits: shared virtual machines, load unknown; the macOS runner is an M1 under virtualization with 3
logical CPUs, not a whole Mac, and no M2, M3 or M4 was measured. Synthetic records only, as no real record
is in CI (on x86 the synthetic records gave 5.9x where real records gave 7.1x; on ARM this was not
measured). One thread only. Within a call the rows run one after the other in a fixed order (1, 2, 3, 4,
6, 8 streams, then the same fused), so a change of the VM's speed during a call is not separated from the
row; the ratio within each call and the 6 VMs per image are the only guard against it. In these runs Apple clang
built only this harness; the `macos-clang` job, merged later, builds `scripts/check_all.sh`, `wex4_tools` and
the fuzz replay with it, at the default of 4 streams. A value
of `WEX_FAST_LANES` other than 4 was compared with the reference decoder by `synth.sh lanes` only, not by
the WEX4 tests and not on real records.

## SVE kernel 2 (AArch64)

Added on 2026-11-15. Kernel 2 has an SVE version. It is the AVX2 kernel 2 step for step, on SVE vectors:
per symbol 4 gathers of 16-bit CDF values (`svld1uh_gather_u32index_u32`) binary-search the 16 entries of
the context in the model blob, one more gather fetches the next 2 bytes of each stream
(`svld1uh_gather_u32offset_u32`), then the same renormalisation, canonical-zero flags, fused hash and
ceil inverse, and the same end-of-stream checks at each block start and at the end. The gathers read 16
bits where the AVX2 ones read 32 bits and mask. The vector length is read from the CPU when the code runs
(`svcntw()`), never fixed when it is compiled: a vector holds 4 streams at 128 bits and 64 at 2048 bits, and
`lanes` = 1, 2 or 3 vectors per loop, as on x86.

How it is selected:

- Only when compiling for SVE. The kernel exists when the compiler defines `__ARM_FEATURE_SVE`:
  `-march=armv8-a+sve`, or `-mcpu=native` on a CPU with SVE (gcc 13.3.0 and clang 18.1.3 on the runner
  below define it there). clang 23.1.5 for an AArch64 target also selects it with `-march=armv8-a+sve2`,
  `-mcpu=neoverse-n2` and `-mcpu=a64fx` (compiled, not run). Then `WEXF_SVE` is 1 and `WEXF_KERNELS` is
  "NEON+SVE". Without such a flag the build is the NEON one, on any CPU, with or without SVE.
- Nothing is chosen at run time. A binary built for SVE needs a CPU with SVE, because the compiler may put
  SVE instructions anywhere in it, not only in kernel 2. There is no `getauxval` test and no fallback.
- NEON kernel 1 (4 streams, fused) stays the default of `wex_fast_decode` in both builds. Kernel 2 runs only
  when `WexFastOptions.kernel` is 2. Only `test_fast_decode.c` asks for it: `wex4_tools.dll` and the fuzzing
  harness call the default.
- Tasks: in an SVE build `wexf_section` cuts tasks of a multiple of the least common multiple of 48 and
  6 x `svcntw()` streams, so that full groups of 1, 2 or 3 vectors fit. That is 48 streams at 128 and 256
  bits, as in every other build, 96 at 512 bits and 384 at 2048 bits.
- A build without SVE compiles to the same machine code as before this change. Compared with `objdump -d`
  between objects built from the old and from the new header: `wex4/wex4_tools.c` with `-DWEX4_FAST_NR3`,
  and `test_fast_decode.c` with its SIMD kernels and with `-DWEX_FAST_SCALAR`. On the runner below with gcc
  13.3.0 (`-O3 -fopenmp`) and clang 18.1.3 (`-O3`), in a temporary step of run 43715133271; on x86-64
  Windows with gcc 16.2.0 (`-march=haswell`, `-DWEX_FAST_SCALAR`, no flag) and clang 23.1.5
  (`-march=haswell`, and the NEON and scalar builds for an AArch64 target as assembly text). A first
  version that declared the task unit as a variable in every build gave other gcc 13 code for
  `wex4_tools.c` on AArch64 (the decoder's results did not change; clang and the x86 builds were the same
  code); the unit is now a variable in the SVE build only.

### The CI runner has SVE

A temporary step of run 43700995179 (`linux-arm` job, 2026-11-15) printed: `lscpu` model name
Neoverse-N2, 4 CPUs; the `Features` of `/proc/cpuinfo` include `sve`, `sve2`, `sveaes`, `svebitperm`,
`svesha3`, `svesm4`, `svei8mm` and `svebf16`; `getauxval(AT_HWCAP)` has `HWCAP_SVE` and
`getauxval(AT_HWCAP2)` has `HWCAP2_SVE2`; `prctl(PR_SVE_GET_VL)` gave 16 bytes and
`/proc/sys/abi/sve_default_vector_length` 16; a program built with `-march=armv8-a+sve` by gcc and by
clang printed `svcntb()` = 16. So the vectors are 128 bits long. gcc 13.3.0 turns `-mcpu=native` into
`-mcpu=neoverse-n2+crc+sve2-sm4+sve2-aes+sve2-sha3+norng+nossbs` and defines `__ARM_FEATURE_SVE` and
`__ARM_FEATURE_SVE2` with it; clang 18.1.3 defines both with `-mcpu=native` too. The `linux-arm` job now
prints the `sve` and `sve2` flags and the harness prints the vector length.

### What was run

Three runs of the CI workflow on the branch of this change, 2026-11-15: 43713913185, 43715133271 and
43716011418. The SVE kernel, the harness and the scripts were the same in the three; they differ in
temporary steps and in the code of builds without SVE (the task unit above). The correctness results were
the same in all three. A fourth run, 43717886891, had the decoder and the harness as committed and no
temporary step: the same correctness results, and in its speed step 1.741 ms per record for kernel 2 with 2
vectors fused against 1.413 ms for the default. Runner, compiler and data as in the NEON section:
`ubuntu-24.04-arm`, Neoverse-N2, gcc 13.3.0, the 4 synthetic records (64 weight streams of 2,048 symbols per
matrix).

- `WEX4_MARCH=-march=armv8-a+sve bash synth.sh check`: `equal` on 4 records x 19 variants x {1, 8}
  threads + default, 0 mismatches; `mutate` with 600 mini-records (205 accepted and 395 rejected by the
  reference), 11,400 fast decodes, 0 mismatches. The scalar build of the same step: 0 mismatches. The step
  fails if the build is not the "NEON+SVE" one.
- `WEX4_MARCH=-march=armv8-a+sve bash wex4/build_and_test.sh`: all WEX4 tests passed (112,399 C
  mutations, same verdicts from both decoders), and `wex4/fuzz/run.sh --replay` with
  `WEX_CFLAGS=-march=armv8-a+sve` passed. Both run kernel 1 only, built with the SVE flag.
- `negative_control.sh` with `WEX4_MARCH=-march=armv8-a+sve`: 8 of 8 broken decoders caught, the 5 of the
  NEON build and 3 in the SVE kernel 2 (canonical-zero check removed, end-of-stream check removed, stream
  hash not checked in the fused variants).
- clang 18.1.3, `-march=armv8-a+sve`, no OpenMP, `-Werror`: `equal` 0 mismatches, `mutate` with 600
  mini-records (11,400 fast decodes) 0 mismatches.
- Other vector lengths, under emulation: QEMU 8.2.2 in user mode (`qemu-aarch64 -cpu
  max,sve-default-vector-length=N`, N in bytes) ran the gcc build of the harness with 256-, 512- and
  2048-bit vectors: `equal` 0 mismatches and `mutate` with 200 mini-records (3,800 fast decodes) 0
  mismatches at each length. This checks bytes only; QEMU says nothing about speed.
- Does kernel 2 run at all? In run 43713913185 a temporary step built a copy of the header whose SVE
  kernel 2 returns -1 at once and ran `equal`. Every kernel 2 variant then reported mismatches at 128 bits
  (hardware) and at 256 and 512 bits (QEMU); at 1024 bits the variants with 1 and 2 vectors, at 2048 bits
  the one with 1 vector. A group holds streams of one matrix with the same count, so with 64 streams per
  matrix the wider groups do not form, and those variants ran kernel 1 alone on these records.
- Which groups the checks reach, per group size (a step of the `linux-arm` job since run 43769703428).
  Three copies of the header return -1 where the SVE kernel 2 takes a group of 1, 2 or 3 vectors; a copy
  is reached when `equal` (4 records) or `mutate` (200 mini-records, seed 20261115) reports a mismatch.
  At 128 bits (hardware) and at 256 and 512 bits (QEMU) both reach all three group sizes; `equal` gives
  16 mismatching decodes each time (2 variants x {1, 8} threads x 4 records). At 2048 bits `equal` reaches
  the 1-vector groups only (16 mismatching decodes, 0 for 2 and 3 vectors), and `mutate` the 1- and
  2-vector groups but not the 3-vector ones: a group of 3 vectors is 192 streams of one matrix, and a
  mini-record has at most 157 weight streams per matrix (2,499 blocks of 32 symbols, chunks of 512 or
  more). So the 3-vector code at 2048 bits never ran, and the 2-vector code at 2048 bits ran in `mutate`
  only. The step fails if a group size that was reached here is no longer reached.

All of these are steps of the `linux-arm` job now, except the test of 1024 bits in the "Does kernel 2 run at
all?" bullet. The probe of the runner, the comparison of machine code and that test were temporary steps.

### Speed on the CI runner

`WEX4_MARCH=-march=armv8-a+sve bash synth.sh speed 21`: one thread, median and best of 21 runs, as in the
NEON section; the table is run 43716011418. The last column divides by the median of the default
configuration of the same build (kernel 1, 4 streams, fused).

| decoder | median ms/record | MB/s | best ms/record | scalar kernel median / this median | this median / default median |
|---|---:|---:|---:|---:|---:|
| reference wex_nr2_decode | 5.504 | 38.0 | 5.457 | 1.00 | 3.90 |
| k0: reference loop per stream (the scalar kernel) | 5.524 | 37.8 | 5.479 | 1.00 | 3.91 |
| k1: 4 streams, fused (the default) | 1.412 | 147.9 | 1.410 | 3.91 | 1.00 |
| k1: 6 streams interleaved | 1.397 | 149.5 | 1.396 | 3.95 | 0.99 |
| k2: SVE gather 1 vector (4 streams) | 2.971 | 70.3 | 2.970 | 1.86 | 2.10 |
| k2: SVE gather 2 vectors | 1.798 | 116.2 | 1.797 | 3.07 | 1.27 |
| k2: SVE gather 3 vectors | 1.976 | 105.7 | 1.974 | 2.80 | 1.40 |
| k2: SVE gather 1 vector, fused | 2.923 | 71.5 | 2.916 | 1.89 | 2.07 |
| k2: SVE gather 2 vectors, fused | 1.741 | 120.0 | 1.739 | 3.17 | 1.23 |
| k2: SVE gather 3 vectors, fused | 1.956 | 106.8 | 1.954 | 2.82 | 1.39 |

- The fastest kernel 2 variant, 2 vectors fused, takes 1.74 ms per record against 1.41 ms for the default
  kernel 1: 1.23 times slower. The three runs gave 1.742, 1.740 and 1.741 ms against 1.413, 1.411 and
  1.412 ms. Against the scalar kernel it is 3.2 times faster.
- With 8 threads on the 4 CPUs (`bench`, best of 15, the same step): 0.58 ms per record for kernel 2 with 2
  vectors fused against 0.47 to 0.48 ms for the default, in all three runs.
- The NEON build of the same runs (the step "Speed of the NEON kernel"): 1.407, 1.406 and 1.405 ms for the
  default. In run 43713913185 the SVE flag moved the kernel 1 rows by at most 3.2% either way against the
  NEON build (the same C, compiled with SVE allowed); the default by 0.4%.
- Why kernel 2 is slower here was not measured (no performance counters). A plausible reading: at 128 bits
  a vector holds 4 streams, no more than kernel 1 interleaves, and every symbol costs 5 gathers, 4 of them
  in a chain (the binary search), where kernel 1 needs a few loads and compares per stream. 3 vectors are
  slower than 2; whether the code for 3 runs out of vector registers was not checked.

Kept as an option, not as the default. This is how the folder treats its other slower variants: the AVX2
kernel 2 is slower than kernel 1 on the desktop and stays (it is the fastest variant on the laptop with 8
threads), and so do the kernel 1 variants with 1, 2, 3, 6 and 8 streams; each is a tested choice of
`WexFastOptions` with its speed stated. The SVE kernel 2 gives the reference's bytes and verdicts in every
check above; on 128-bit SVE it is not the faster choice. Whether longer vectors make it faster (256 bits on
Neoverse V1, 512 bits on A64FX) is not known: no such CPU was available.

Not checked for the SVE kernel: real records; any CPU other than the Neoverse-N2 of the runner; the speed
at any vector length but 128 bits; sanitizers (the sanitizer jobs are x86-64); macOS (the `macos` job
builds NEON only); the libFuzzer harnesses (they call the default entry point); the groups of 3 vectors at
2048 bits (no record of the checks forms one). `mutate` ran 600 mini-records per build on the hardware and
200 per vector length under QEMU.

## Semantics and safety notes

- Same accept / reject as the reference for WNR3 views produced by `wex_nr3_parse`; views with another magic
  are passed to `wex_nr2_decode` unchanged. On failure the partially written output can differ from the
  reference's partial output (both are unspecified).
- The fast kernels read up to 4 bytes ahead and notice a stream that runs past its end at the next 32-symbol
  boundary. Streams whose end is closer than 104 bytes to the end of the weight payload use the reference
  loop with its per-byte check, so no byte at or beyond that end is read. One check is added that the
  reference does not have: a stream length that would cross the end of the weight payload returns -1 (cannot
  happen on a parsed record).
- The model must be validated (`wex_nr2_model_open`), as the reference requires. With an unvalidated blob
  the fast kernels stay inside their buffers by construction (bounded symbol, bounded shift), but this was
  not tested and the result would not match the reference.
- `wex_fast_decode` allocates its tables with `malloc`; if that fails it falls back to kernel 0 (same
  result, slower). Not exercised by a test.
- Compile-time selection: AVX2 kernels need `__AVX2__`, `__BMI__`, `__BMI2__`, `__LZCNT__` (`-march=haswell`);
  the NEON kernel 1 needs `__aarch64__`, `__ARM_NEON` and a little-endian target (no flag); the SVE
  kernel 2 needs that and `__ARM_FEATURE_SVE` (`-march=armv8-a+sve`, or `-mcpu=native` on a CPU with SVE);
  otherwise, or with `-DWEX_FAST_SCALAR`, only kernel 0 is compiled (portable C, still parallel across
  streams). `WEXF_SIMD` is 1, 2 or 0, `WEXF_SVE` is 1 or 0, and `WEXF_KERNELS` is "AVX2", "NEON",
  "NEON+SVE" or "scalar". GNU extensions used: `always_inline`, `__builtin_bswap32`, on ARM
  `__builtin_clz` and `__builtin_ctzll`. The SIMD paths assume little-endian.
  `-DWEX_FAST_LANES=N` (1, 2, 3, 4, 6 or 8, default 4) sets the streams of kernel 1 in `wex_fast_decode`;
  any other N stops the build.
- Parallel region inside an existing OpenMP parallel region runs serially unless nesting is enabled.

## Not checked

- Only 6 of 7,680 records were decoded at full size (0.08%); the decoded output was compared with the
  reference decoder, not with the safetensors source.
- Mini-records in the mutation test come from the same 6 records plus synthetic data; no full-size mutated
  record was decoded by the reference.
- Real records were decoded only by builds of gcc 16.2 on Windows (x86) until 2026-11-16. That day
  `verify/run.sh` ran to its end on 8 other real WNR3-CZ records (DeepSeek V4.1 Flash, layers 10 and 30,
  experts 0 to 3, a model trained on those 8) with three AVX2 builds on one x86-64 desktop: gcc 16.2, and
  clang 23.1.5 without OpenMP (MSYS2 UCRT64, Windows 10), and gcc 15.2 (Ubuntu under WSL2, without the three
  `vh.c` steps). `equal`, `mutate`, `mutate` with the stream hash check removed and, on Windows, `reenc` and
  `tailcut` gave 0 mismatches in each. Still no real record on ARM. Elsewhere the harness runs on
  synthetic records only, in CI: gcc 13.3 on x86-64 Linux and on ARM64 Linux (on both also under ASan and UBSan);
  clang 18 on x86-64 Linux (the `linux-tsan` job, `CC=clang`: the AVX2 and the scalar build) and on ARM64
  Linux (the NEON and the SVE build without OpenMP, see the NEON and the SVE sections); Homebrew GCC 15 and
  Apple clang 17 on macOS arm64 (both builds of Apple clang also under ASan and UBSan; the speed with both
  only in the `arm-speed` jobs that a run started by hand has); the gcc SVE build under
  QEMU at three vector lengths. No MSVC, no big-endian, no SVE build with Apple clang or under a sanitizer.
- WNR2 records (delegated to the reference), the `malloc` failure path and unvalidated models.
- Thread counts other than 1, 3 (mutation test) and 8; no run pinned to physical cores; the desktop was busy.
- Laptop: one record and 3,000 mutation cases only.
- Extrapolation, not a measurement: 7,680 records x 13.8 ms is about 106 s of wall time for the whole
  container with 8 desktop threads, against about 65 minutes for the single-threaded reference.
