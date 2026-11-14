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
measured on synthetic records only: see "NEON kernel (AArch64)" below.

## Files

| file | purpose |
|---|---|
| `wex_fast_decode.h` | the decoder (C11 + GNU builtins, includes `wex_format.h`) |
| `test_fast_decode.c` | harness: `equal`, `mutate`, `bench`, `speed` |
| `build.sh` | x86: builds the harness three ways into `out/` (AVX2 + OpenMP, scalar fallback, UBSan trap build) |
| `synth.sh` | any CPU, no real model: builds the harness with the SIMD kernels of the CPU and with the scalar kernel into `out/synth/` and runs it on the 4 synthetic records of `wex4/out/synth_nr3.wex` (`check`, run by `scripts/check_all.sh`, or `speed`) |
| `negative_control.sh` | removes one check at a time from a copy of the header and expects the mutation test to notice |
| `extract.py` | copies the model blob and chosen records out of a container opened read-only |
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
streams spills state to the stack.

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

Kernel 2 is not ported: NEON has no gather load. A NEON build that is asked for kernel 2 runs kernel 1
with 4 streams. `-DWEX_FAST_SCALAR` builds kernel 0 alone, as on x86.

### What was run on ARM

Everything below ran on 2026-11-14 in the `linux-arm` job of `.github/workflows/ci.yml`: GitHub runner
`ubuntu-24.04-arm`, Neoverse-N2, 4 logical CPUs, gcc 13.3.0, `-O3 -fopenmp`. Data: the 4 records of
`wex4/out/synth_nr3.wex`, which `wex4/build_and_test.sh` writes from `fixtures/make_synth_mxfp4.py`
(208,896 decoded bytes each, 24 scale and 192 weight streams, flags 0x00, model trained on the same 4
records). No real record was decoded on ARM.

- `synth.sh check`, NEON build: `equal` on 4 records x 13 variants x {1, 8} threads + default, 0
  mismatches; `mutate` with 600 mini-records (205 accepted and 395 rejected by the reference), 7,800 fast
  decodes, 0 mismatches. The scalar build of the same run: 0 mismatches. The job fails if the build is
  not the NEON one.
- `wex4/build_and_test.sh`, where `wex4_tools.dll` holds the fast decoder and `wex4_tools_ref.dll` the
  reference: all tests passed with the NEON kernel and again with `WEX4_MARCH=-DWEX_FAST_SCALAR`
  (112,399 C mutations each time, same verdicts from both decoders).
- Once, not part of the job as committed: `negative_control.sh` on the synthetic records caught 5 of 5
  broken decoders (the kernel 2 control does not apply); the harness built with clang 18.1.3 without
  OpenMP gave 0 mismatches in `equal` and in `mutate` with 3,000 mini-records (39,000 fast decodes).
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

- NEON kernel 1 against the scalar kernel, default configuration: 5.54 -> 1.41 ms per record, 3.9x. An
  earlier run of the job the same day gave 5.542 -> 1.408 ms. The scalar build of the same source
  measured 5.48 and 5.46 ms for kernel 0 in the two runs.
- 4 and 6 streams are equal within 1%; the default stays at 4 streams, as on x86.
- Fusing the hash into the loop changes nothing measurable here (1.403 against 1.406 ms), unlike on the
  x86 desktop with real records (-21%).
- `bench` in the same job (best of 15): the default configuration with 8 threads on the 4 CPUs took 0.47
  ms per record against 1.41 ms on one thread. The records are small (216 streams), so this says little
  about thread scaling on full records.
- The harness built with clang 18.1.3 (no OpenMP), one run: 5.97 -> 1.48 ms, 4.0x.
- For scale, the same script on the x86 desktop (AVX2, gcc 16.2, one run, machine not idle): 6.87 -> 1.16
  ms, 5.9x, where real records gave 7.1x. Whether the synthetic records understate the gain on ARM too
  was not measured.

Not checked on ARM: real records; any CPU other than the Neoverse-N2 of the runner (no Apple silicon, no
Ampere, no Cortex-A); macOS and Apple clang; big-endian AArch64 (it builds the scalar kernel, never run);
32-bit ARM (scalar kernel, never run); UBSan and ASan (the sanitizer job is x86-64).

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
  the NEON kernel 1 needs `__aarch64__`, `__ARM_NEON` and a little-endian target (no flag);
  otherwise, or with `-DWEX_FAST_SCALAR`, only kernel 0 is compiled (portable C, still parallel across
  streams). `WEXF_SIMD` is 1, 2 or 0 and `WEXF_KERNELS` is "AVX2", "NEON" or "scalar". GNU extensions
  used: `always_inline`, `__builtin_bswap32`, on ARM `__builtin_clz` and `__builtin_ctzll`. The SIMD
  paths assume little-endian.
- Parallel region inside an existing OpenMP parallel region runs serially unless nesting is enabled.

## Not checked

- Only 6 of 7,680 records were decoded at full size (0.08%); the decoded output was compared with the
  reference decoder, not with the safetensors source.
- Mini-records in the mutation test come from the same 6 records plus synthetic data; no full-size mutated
  record was decoded by the reference.
- Real records were decoded only by builds of gcc 16.2 on Windows (x86). On Linux the harness runs on
  synthetic records only, in CI: gcc 13.3 on x86-64 (also under ASan and UBSan) and on ARM64. No MSVC, no
  big-endian; clang once, on ARM (see the NEON section).
- WNR2 records (delegated to the reference), the `malloc` failure path and unvalidated models.
- Thread counts other than 1, 3 (mutation test) and 8; no run pinned to physical cores; the desktop was busy.
- Laptop: one record and 3,000 mutation cases only.
- Extrapolation, not a measurement: 7,680 records x 13.8 ms is about 106 s of wall time for the whole
  container with 8 desktop threads, against about 65 minutes for the single-threaded reference.
