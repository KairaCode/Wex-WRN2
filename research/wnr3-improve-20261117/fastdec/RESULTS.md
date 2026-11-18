# Faster WNR3 CPU decode, same format (lane: fastdec)

Records on disk do not change, so the size effect is 0 bytes. Only the CPU decoder changes. Everything below was
measured on this machine: a shared 4-vCPU Xeon VM (Emerald Rapids, family 6 model 207, AVX2 + AVX-512
F/BW/VL/CD/VBMI/VBMI2/IFMA, 2.1 GHz nominal; a dependent `imul` chain runs at 1.24 core cycles per TSC tick,
so the core clock is about 2.6 GHz; `gbench.c`). Data: the real production records in `/tmp/wnr3-data/recs`
(WNR3-CZ, flags 0x0f, 35,389,440 weight symbols + 1,105,920 scale symbols per record). gcc 13.3, `-O3`.
No perf counters exist in this VM (`perf_event_open` fails), so the breakdown comes from ablation builds and
timing.

## Result

Two new kernels in `wex_fast_decode.h` (patch: `wex_fast_decode.patch`). They decode many streams at once with
SIMD gathers from a two-level table per context, so no per-symbol search is needed:

- **kernel 3 (AVX2)**: 4 vectors x 8 streams. It is the default of `wex_fast_decode` in an AVX2 build
  (`-march=haswell`, the i7-7700K's level).
- **kernel 4 (AVX-512 F/BW/VL/CD/VBMI2)**: 3 vectors x 16 streams, plus an AVX-512 scale kernel (2 x 16 streams,
  stream hash computed in the loop). It is the default when compiled with `-march=x86-64-v4 -mavx512vbmi2`
  (Zen 4 and Ice Lake or later have these instructions).

Headline: interleaved A/B, 1 thread, 3 records (L0 E7, L20 E73, L39 E120), 9 rounds, per-thread CPU time
(`logs/time_run1.txt`, second run in `logs/time_run2.txt`):

| decoder | binary | median ms/record | min..max | per-round ratio to the unchanged decoder (median) | rounds faster |
|---|---|---:|---:|---:|---:|
| unchanged `wex_fast_decode.h`, k1 x 4 fused (`-march=haswell`) | out/test_orig | 92.5 | 89.0..125.5 | 1.000 | - |
| new AVX2 default, kernel 3 x 4 vectors (`-march=haswell`) | out/test_h | 48.4 | 46.6..58.6 | **0.543 (1.84x)** | 9/9 |
| new AVX-512 default, kernel 4 x 3 vectors (`-march=x86-64-v4 -mavx512vbmi2`) | out/test_z | 48.9 | 39.0..52.0 | **0.464 (2.16x)** | 9/9 |
| k1 x 4 inside the AVX-512 build (control) | out/test_z | 102.7 | 85.8..123.9 | 0.957 | 5/9 |

Second run, same commands (`logs/time_run2.txt`): orig 100.5 ms, AVX2 49.3 ms (ratio 0.502), AVX-512 45.1 ms
(ratio 0.426), 9/9 rounds each. Wall-clock instead of CPU time: ratios 0.585 / 0.418 (run 1) and 0.488 / 0.398
(run 2), 9/9 rounds each.
In the single-binary sweep of the same log (every row in one AVX-512 binary, ratio to k1 x 4): kernel 4 with 48
streams 0.521, with 64 streams 0.486; kernel 3 with 32 streams 0.562. Overall: **about 1.8x (AVX2) and 2.0 to 2.2x
(AVX-512) faster than today's fast decoder on one thread** on this CPU. Byte-identical output and identical
accept/reject on everything tested (section "Correctness").

4 threads, wall clock: the gain holds, but thread scaling itself could not be measured. The VM's load average was
5 to 13 on 4 vCPUs (other agents), so 4 threads gave at best 1.6x over 1 thread for any decoder. Run 1: orig 109.5,
AVX2 70.7, AVX-512 65.3 ms/record (ratios 0.655 / 0.621). Run 2: 67.6 / 38.6 / 38.6 ms (ratios 0.651 / 0.638). All
9/9 rounds. Kernels 3 and 4 use kernel 1's OpenMP loop. Tasks are 192 streams instead of 48, so a record has 90
weight tasks instead of 360. Inferred, not measured: they should scale like kernel 1 on an idle machine. Run 2 says
**about 1.55x at 4 threads under load**.

## Table of variants (1 thread, ratio to k1 x 4 fused in the same binary unless noted)

| variant | what | ratio (lower = faster) | correctness evidence | kept? |
|---|---|---:|---|---|
| k1 x 4 fused | today's default | 1.000 (92-103 ms/record) | the reference harness | baseline |
| k1 x 6 fused | today's code, `-DWEX_FAST_LANES=6` | 0.92 (7/9 rounds; 0.86-0.91 in earlier runs) | existing | on this CPU only; the repo measured 4 streams best on the 7950X3D and the 7700HQ |
| k2 x 24 fused | today's AVX2 gather kernel | 1.34 | existing | no |
| (all rows of this table: `logs/time_run1.txt` / `time_run2.txt` for the final kernels, research rows from the session's abx runs on 2-3 records, 3-7 rounds) | | | | |
| research k3: scalar bucket table (u64 entry per 128-slot bucket: symbol, start, frequency of up to 2 symbols; 1 load replaces the AVX2 search) | `wex_fast_decode_research.h` kernel 3 | **1.4 to 2.9 (slower)** with 3-8 streams, at bucket sizes 2^6..2^10 | equal on 2 records | **no** (negative result, below) |
| research k4: AVX-512, 16 lanes, same u64 bucket table, 2 x vpgatherdq + 1 byte gather per 2 symbols | research kernel 4 | 0.56-0.70 (2^7 buckets); 0.51-0.56 (2^5 buckets) | equal on 2 records | superseded by the two-level table |
| research k4 + exact sub-tables for 3+-symbol buckets | research | same as above (0.56 at 2^5; 2^6/2^7 not better) | equal on 1 record per setting | superseded |
| **kernel 3: AVX2 two-level, 4 x 8 streams** | final | **0.50-0.54** (AVX2 build against the unchanged binary); 0.56-0.63 when compiled inside the AVX-512 build | equal 121 (default-only) + 12 records (all variants) + 1 in the z build x 121; mutate 3,000 (AVX2 build) + 4,000 + 800 UBSan (z build); 5 negative controls | **yes, AVX2 default** |
| kernel 3 with 1 / 2 / 3 vectors | final | 1.20-1.26 / 0.75-0.81 / 0.63-0.67 | same | options |
| **kernel 4: AVX-512 two-level, 3 x 16 streams + AVX-512 scale kernel** | final | **0.43-0.46** (against the unchanged binary); 0.50-0.52 against k1 x 4 in the same binary | equal 121 records (all variants); mutate 4,000 + 800 UBSan; 7 negative controls | **yes, AVX-512 default** |
| kernel 4 with 1 / 2 / 4 vectors | final | 0.82 / 0.57-0.58 / 0.49-0.53 | same | options (4 vectors is about as fast as 3) |
| AVX-512 scale kernel on top of kernel 4 | final | 0.957 against kernel 4 without it (8/9 rounds) | same | yes |
| two-level table size: 2^5 / 2^6 / 2^7 slots per bucket | research k5 | 0.416 / 0.444 / 0.466 of k1 x 4 | equal on 1 record each | 2^5 |

## Where the time goes (today's kernel 1)

Ablation builds (`wex_fast_decode.abl.h`): one step of the loop is replaced by a cheap stand-in that keeps the data
flow. The output is wrong on purpose and the checks are off. The numbers are the time saved when a step is removed,
interleaved A/B, 2 records, CPU time (`logs/ablation_k1x1.txt`, `logs/ablation_k1x4.txt`). Each record also carries
about 2.9 ms of scale decode.

| step removed | 1 stream (latency-bound), ms/record | share | 4 streams (today's default) | share |
|---|---:|---:|---:|---:|
| none (full kernel) | 302.9 = **8.4 ns = about 22 core cycles per symbol** | - | 105.0 = **2.9 ns = about 7.5 cycles per symbol** | - |
| symbol search (vmovd + vpbroadcastw + vpcmpgtw + vpmovmskb + tzcnt), replaced by a shift | 204.9 | **-34%** | 75.7 | **-28%** |
| renormalisation (lzcnt, unaligned 4-byte load, bswap, 64-bit shift) | 236.6 | **-22%** | 77.0 | **-26%** |
| next-context load (`next[t]` from the 128-byte entry) | 297.6 | -2% | 89.5 | -12% |
| fused MurmurHash3 | 320.6 | (noise) | 97.8 | -7% |
| nibble packing (pext), store, canonical-zero test | 290.3 | -4% | 94.0 | (noise, 3/7 rounds) |

Reading: one stream is a dependency chain of about 22 cycles per symbol. Its links are x -> slot -> broadcast /
compare / movemask / tzcnt (about 10 cycles) -> start/freq loads (5) -> imul (3) -> add -> lzcnt / shift (about 4)
-> x. Four interleaved streams bring this to about 7.5 cycles per symbol and per stream, which is close to the
throughput limit: about 30 uops per symbol at 4 to 5 uops per cycle. With 8 streams the compiler spills the
per-stream state (x, read pointer, context pointer, srel base), and 8 is slower again. The last bytes of headroom
in kernel 1 come from shorter chains and fewer uops per symbol. Kernels 3 and 4 get both by moving many streams
into vector lanes.

Kernel 4 (`logs/ablation_k4x3.txt`, same method, 3 x 16 streams; noisy, load average 13): the ta gather (bucket
entry) -36%, the tc gather (start/frequency) -18%, the byte gather -16% (one gather per 2 symbols), the 16 scalar
output stores per 8 symbols -8%, the vector hash about 1%. With all three gathers removed, 55% of the time is left
(ALU work: about 33 vector uops per 16 symbols, plus the scale section). So kernel 4 is bound by gathers about as
much as by ALU work. The ta table (1 KB per context, 777 reachable contexts) does not fit in L1, and most ta
gathers hit L2. On this VM, 16 qword gathers took 7 TSC ticks from L1 and 49 from a 2 MB table (`gbench.c`).

Measured per-symbol cost of today's decoder: about 2.9 ns (1 thread, 4 streams, this VM). Of that, 28% is search,
26% renormalisation, 12% next-context load and 7% hash; the rest is overlap that the ablations cannot attribute.
The new kernels: about 1.4 ns per symbol (kernel 4) and about 1.5 ns (kernel 3).

## How kernels 3 and 4 work

Tables (built by `wex_fast_model_init`, read-only, shared between threads):

- `ta`: one u16 per (context, 32-slot bucket). The context is (matrix, srel, maximum, previous), and only the 777
  reachable ones are filled (previous <= maximum, plus block start). Entry = `b | sym_lo << 8`. `sym_lo` is the
  symbol of the bucket's first slot. `b` is the offset in the bucket where `sym_lo + 1` starts (32 if the bucket
  holds one symbol). Then symbol = `sym_lo + (slot mod 32 >= b)`, with no search. A bucket with 3 or more symbols
  (811 of 398,000 in the production model, 0.38% of decoded symbols, 5.9% of 16-lane steps) has bit 15 set and
  points to an exact sub-table of 32 entries. Those lanes do one masked gather.
- `tc`: one u32 per (context, symbol) = start | frequency << 16.
- Per symbol and vector: gather ta, compare, gather tc, multiply-add, then renormalise. Every second symbol, one
  gather reads 4 bytes per stream: the state stays in [2^23, 2^31) for any input bytes, so one symbol takes at most
  2 bytes. The shift in bits is `lzcnt(x << 1) & 24`, followed by VPSHLDVD (kernel 4), or two compares (kernel 3).
  The next context is `base(srel) + 8 * max + previous`, computed in registers.
- Checks are kernel 1's, in the same places: end pointer at every block start (memory safety, as in kernel 1) and
  at stream end, canonical zero (`has8` on the packed words), fused stream hash, fused ceil inverse on blocks whose
  maximum is 5 (including the scale-0 underflow), and the WEXF_SLACK rule for streams near the payload end.
- Cost: `WexFastModel` grows from 223,944 B to 1,840,144 B. `wex_fast_model_init` takes about 170 us instead of
  13 us (loaded VM). Production should cache the tables (`wex_fast_decode_with`). The per-call `wex_fast_decode`
  adds this to a decode of about 45 ms, under 0.5%.

## Correctness (every kept variant)

All logs are in `logs/`. The harness is the repo's `test_fast_decode.c`. My changes to it touch inputs and coverage
only (README), not the checks.

- `equal`, AVX-512 build: **121 of 121 records, 29 variants (k0, k1 x all, k2 x all, k3 x 1..4 vectors fused + 1
  unfused, k4 x 1..4 fused + 1 unfused) x {1, 2} threads + the default entry point (kernel 4 x 3), 0 mismatches**
  (`logs/equal_z_121.txt`).
- `equal`, AVX2 build: 121 of 121 records with k0 + the default entry point (kernel 3 x 4), 0 mismatches
  (`equal_h_default_121.txt`). 12 records (one every ~3 layers + L39 E120) with all 24 variants, 0 mismatches
  (`equal_h_12.txt`).
- `mutate` (mini-records cut from 6 real records, re-encoded with the reference encoder, plus byte mutations; every
  variant against the reference, status and bytes; guard pages around record, outputs and model):
  - AVX-512 build, seed 20261117: 4,000 records at the decoders (1,088 accepted, 2,912 rejected), 116,000 fast
    decodes, **0 mismatches**. 3.60 M weight streams decoded inside kernel 4 groups, 3.24 M inside kernel 3 groups,
    289,920 scale streams in the AVX-512 scale kernel.
  - AVX2 build, seed 777001: 3,000 records, 72,000 fast decodes, 0 mismatches, 3.21 M weight streams in kernel 3
    groups.
  - UBSan trap build (AVX-512), seed 4242: 800 records, 23,200 fast decodes, 0 mismatches, no trap, no fault.
  - Harness changes made for coverage: kinds 18/19 cut 8,192..9,000 blocks per matrix (scale chunk 256, weight
    chunk 512), so that 32-stream scale groups and 48-stream weight groups occur. Two mutation cases were added:
    bit flips inside the scale payload, and a flip of the last byte of a scale stream. Without them the mutation
    test never reached the AVX-512 scale kernel with a corrupted stream (see the negative controls).
- Negative controls (`negcontrol.py`: remove one check from one function of a copy of the header, build with
  TEST_DEFAULT_ONLY, expect MISMATCH within 6,000 cases): **10 of 12 caught, and the 2 misses are the expected ones**
  (`logs/negcontrol_final.txt`, final harness). Earlier runs (`negcontrol_run1.txt`, `negcontrol_scales.txt`) are kept because they show why the
  harness had to change.
  - Kernel 4 weights: end-of-stream pointer, final state / stream hash, canonical zero and ceil underflow: all
    caught.
  - Kernel 4 scales: end-of-stream pointer and stream hash: both caught with the final harness. The stream hash was
    NOT caught before the "last byte of a scale stream" mutation was added.
  - Kernel 3 weights: the same four controls as kernel 4 weights, all caught.
  - The block-start end-pointer checks (k3, k4) are not caught, as expected. They only keep reads inside the 104-byte
    slack, and the end-of-stream check gives the same verdict. The repo's own negative_control.sh does not test
    kernel 1's equivalent check either (RESULTS of fast-cpu-decode: "memory-safety rules").

- Builds: gcc 13 `-Werror -Wall -Wextra` clean for `-march=haswell`, `-march=x86-64-v4` (kernel 4 off, kernel 3
  default) and `-march=x86-64-v4 -mavx512vbmi2`. clang 18 builds without warnings, and a clang AVX-512 build passed
  `equal` on L39 E120 with all 29 variants. The scalar build (`-DWEX_FAST_SCALAR`) passed `equal` on L0 E7.

## What did not work (negative results)

- **Replacing the AVX2 search by a table load in the scalar interleaved kernel (research k3) made it 1.4 to 2.9x
  slower**, measured at every bucket size 2^6..2^10 and with 3 to 8 streams. Kernel 1's `upper[]` load depends only
  on the context, so it issues early, and only the compare sits on the x chain. A bucket load is indexed by the slot,
  so the load latency lands on the critical chain. Field extraction and selects add about 8 ALU ops. The compiler
  turned the 2-way select into a branch and spilled the state until I rewrote it with masks, and the gain was still
  negative. The latency floor of a scalar rANS chain is close to what kernel 1 already does.
- A u64 bucket entry (symbol, start and frequency of 2 symbols in one word; research k4) needs 2 x 8-element qword
  gathers per 16 symbols and a 1 to 5 MB table. Splitting it into u16 bucket + u32 (context, symbol) was faster
  (0.42 against 0.51-0.56 of k1 x 4), even though it adds a dependent gather.
- Exact sub-tables instead of a scalar fallback for 3+-symbol buckets did not change the speed at 2^5 buckets, where
  only 2.9% of vector steps are affected. They are kept because they make the kernel branch-light and simple.
- Larger buckets (2^6, 2^7) shrink the table but make the 3+-symbol case 2x / 4x more frequent: not faster.
- 8 streams in kernel 1 stay slower than 4 to 6 (state spills), as the repo found.
- 4-thread timing was impossible on this oversubscribed VM (above).
- Not done: an AVX2 scale kernel for kernel 3 (expected about -3%, since the scale section is about 6% of kernel
  3's time); a Zen 4 or Skylake measurement (no such machine here).

## What transfers to the production decode machines

- **Ryzen 9 7950X3D (Zen 4, AVX-512 incl. VBMI2/CD)**: kernel 4 compiles and runs there (same ISA level as here),
  but its speed is not measured. Zen 4 executes 512-bit ops as two 256-bit halves, and its gathers have different
  throughput from Intel's. Kernel 3 is the fallback there too. Expect a gain, but measure: the harness `ab` command
  and `abx.py` work on Windows/MSYS2 with gcc as on Linux (CPU time needs `clock_gettime`; use wall time).
- **i7-7700K (Kaby Lake, AVX2, no AVX-512)**: kernel 3 is the candidate. Not measured: Skylake-family gathers
  (`vpgatherdd ymm`) are slower than this Emerald Rapids'. Kaby Lake is also affected by Downfall (GDS), and with the
  GDS microcode update gathers get much slower. On the 7700HQ laptop the repo measured the old gather kernel k2 at
  1.2 to 1.4x slower than k1 x 4 on one thread (here: 1.34x slower). Kernel 3 does 2.5 gathers per 8 symbols instead
  of k2's 5, and 1 dependent gather level instead of 4. Inferred: probably faster than k1 there, possibly not by
  1.8x, and possibly not at all with GDS mitigation. Run `abx.py` with out/test_orig:10 against out/test_h:22 on
  that machine before switching the default; `-DWEX_FAST_LANES` and the options still select kernel 1.
- ARM (NEON/SVE): unchanged (kernels 3/4 are x86-only; kernel 1 stays the default there).
- GPU: the two-level table idea (u16 per bucket: one compare instead of a 4-step binary search, then one table
  read) maps directly to a CUDA / Vulkan thread-per-stream decoder. That is an inference: the GPU decoders were not
  examined in this lane.

## Rerun

```sh
cd /tmp/wnr3-research/fastdec
./build.sh                                   # out/test_z, out/test_h, out/test_orig, scalar, UBSan
./run_equal2.sh                              # (uses logs/equal_z_121.txt to skip done records; empty it to redo)
./run_mutate.sh                              # 4,000 + 3,000 + 800 UBSan mutations (logs/mutate_*.txt)
python3 negcontrol.py 6000 99 /tmp/wnr3-data/recs/rec_00_007.bin /tmp/wnr3-data/recs/rec_20_073.bin /tmp/wnr3-data/recs/rec_39_120.bin
./run_time.sh                                # timing tables (logs/time_run*.txt)
```
