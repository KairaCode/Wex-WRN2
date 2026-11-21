# A record coding designed for decode speed (WNX prototype), against WNR3-CZ + `wex_fast_decode`

Lane: decode-speed format. Everything here was run on the shared 4-core Xeon (Sapphire Rapids class, 2.1 GHz,
AVX2 + AVX-512) of this session, CPU only, on the 121 real production records of `<TMP>/wnr3-data` with the
suggested split (models trained on the 80 TRAIN records, every size and speed figure on the 41 TEST records or
a subset of them). Scripts, raw logs and the rerun line are in this directory (`README.md`).

## Result in one table

Sizes: mean over the 41 TEST records (40 + L39 E120), real records (header, offset table, every stream, zero
padding to 4096 B, as the stored records), each encoded with the TRAIN model and decoded by the WNX reference
decoder to exactly the bytes the production WNR3 CPU reference decodes the stored record to (Z(source)): 41/41
in every row. Baseline = the stored WNR3-CZ records of the same 41 experts (mean 16,080,996 B = 0.85534 of
raw; the whole 121-record sample: 0.85436).

Speed: decoded MB/s (10^6 B of slab output per second) and ns per weight on 1 thread, process CPU time,
median over the rounds of 8 TEST records; the speed-up is the median of paired ratios, each WNX run timed right
next to a `wex_fast_decode` run (k1, 4 streams, fused hash, the production fast CPU decoder) on the same
records, same build flags. Two runs: run 1 (`logs/time.txt`, 15 rounds, load average 7-13 on the 4 cores) and
run 2 (`logs/time2.txt`, 11 rounds, load 2.3-6.9); the table gives run 1 / run 2. The machine was shared with
other agents throughout: absolute numbers carry that load, the paired ratios are the figure to use.

| record coding | streams / record (lanes) | mean record | of raw | vs WNR3-CZ, % of raw (min..max) | CPU decode, AVX2 build | AVX-512 build, AVX-512 kernel | GPU structure |
|---|---:|---:|---:|---:|---|---|---|
| **WNR3-CZ, stored** (`wex_fast_decode`) | 19,440 (19,440) | 16,080,996 | 0.85534 | 0 | 177 / 175 MB/s, 3.00 / 3.03 ns/w (1.00x) | 184 / 186 MB/s, 2.89 / 2.85 ns/w (1.00x) | 1 thread per stream |
| C1 **MAG**, 32-lane interleaved rANS, Bw = Bs = 2048 | 558 (17,856) | 16,029,946 | 0.85263 | **-0.272** (-1.02..-0.24) | 598 / 592 MB/s, 0.89 / 0.90 ns/w (**3.54x / 3.49x**) | 786 / 1002 MB/s, 0.68 / 0.53 ns/w (**4.57x / 4.87x**) | 1 warp per stream |
| C1t **MAGT** (= MAG + last signs in the lane start states), 2048:2048 | 558 (17,856) | 15,997,378 | 0.85090 | **-0.445** (-1.20..-0.41) | 601 / 581 MB/s, 0.88 / 0.91 ns/w (**3.39x / 3.40x**) | 785 / 891 MB/s, 0.68 / 0.60 ns/w (**4.20x / 4.82x**) | same |
| C1t MAGT, Bs = 512 (shorter scale streams for single-record GPU latency) | 609 (19,488) | 16,002,872 | 0.85119 | -0.416 | (as MAGT) | (as MAGT) | same |
| C1t MAGT, Bw = Bs = 4096 (half the lanes) | 279 (8,928) | 15,984,690 | 0.85022 | -0.512 (-1.26..-0.48) | (same kernel) | (same kernel) | half the warps, 2x serial depth |
| C2 **JOINT**, 15-symbol alphabet, no sign region, 2048:2048 | 558 (17,856) | 16,029,446 | 0.85260 | -0.274 (-1.05..-0.24) | 591 MB/s, 0.90 ns/w (3.06x, run 1 only) | 771 MB/s, 0.69 ns/w (4.19x, run 1 only) | same |
| C1 MAG, Bw = Bs = 1024 (twice the lanes) | 1,116 (35,712) | 16,087,789 | 0.85570 | +0.036 | (same kernel) | (same kernel) | twice the warps, half the depth |
| C1 MAG, Bw = Bs = 4096 | 279 (8,928) | 16,000,974 | 0.85109 | -0.426 | (same kernel) | (same kernel) | |

Without the outlier L39 E120 (which gains about 1% of raw) the deltas are -0.253 (MAG 2048), -0.426 (MAGT 2048),
-0.494 (MAGT 4096), -0.255 (JOINT 2048) % of raw (`logs/enc_*.txt`).

In plain words: **the same output from 0.27-0.51% fewer bytes, decoded 3.1-3.5x faster than today's fast CPU
decoder with AVX2 and 4.2-4.9x with AVX-512, on one thread, and 3.1-4.3x with 4 threads**, with a GPU weight kernel that has no context chain and no
per-lane table lookups. The size gain does not come from modelling (the model is slightly worse, +0.018% of raw)
but from spending the per-stream overhead on 32 lanes of one stream instead of 32 streams.

## The format (what was prototyped)

Same coded domain as WNR3-CZ: canonical zero, ceil form per matrix, rule-g anchors, srel contexts from the coded
scales, T^-1 after decoding; the builder's `wex_nr3_prepare` step is unchanged. Only the entropy coding and
the record skeleton differ (`wnx.h` has the full definition in its header comment).

- **Stream = 32 rANS lanes sharing one word stream.** 32-bit states, L = 2^16, 16-bit renormalisation (at most
  one word per lane per step, for any input), 12-bit weight CDFs (14-bit scale CDFs). In each step the lanes
  decode in lane order and every lane whose state fell below L takes the next 16-bit word. Layout of a stream:
  32 x u32 initial states, then the words, then (MAG) the sign bits as u32 words stored backwards from the
  stream end. The stream is valid only if the word pointer ends exactly where the sign words begin.
- **Weights: one step = one block.** Lane k decodes block position 2k (k < 16) or 2(k - 16) + 1, so all 32
  symbols of a step share one context, `srel` of the block's coded scale (7 contexts per matrix), known before
  the step. A weight stream is Bw blocks (2048 = 64 K nibbles, i.e. each lane carries 2048 symbols like a WNR3
  stream does). Output byte i of the block = lane i | lane i+16 << 4.
- **C1 MAG:** 8-symbol magnitude alphabet; the sign of every non-zero magnitude is a raw bit (an exact coin,
  `mxfp4-contexts`); magnitude 0 has no sign, so -0 cannot be represented and the CZ check disappears.
  **C2 JOINT:** 15 symbols (every nibble except 8), no sign region.
- **C1t MAGT:** lanes 2..31 start (encoder side) at L + 16 raw bits instead of L + hash bits: the first 480 sign
  bits of the stream's last 24 blocks; the rest of those blocks' sign bits follow the region's. The decoder
  holds those 24 blocks' magnitudes and gets their signs from the final states. This turns 30 x 16 bits of
  start state per stream, which every rANS lane pays anyway, into payload: -0.17% of raw.
- **Scales:** 32 lanes, lane k = scale 32t + k of the stream, order-0 per matrix, Bs steps per stream.
- **Integrity.** 16 MurmurHash3 chains per stream (chain j = lanes 2j and 2j+1, a 32-bit word every 4 steps
  of coded nibbles, or 2 steps of coded scale bytes), seeded with {model id, owner, stream index, anchors and
  flags} like WNR3; G = MurmurHash3 of the 16 chain hashes. Lane 0 starts at L + (G & 0xffff), lane 1 at
  L + (G >> 16): **a 32-bit check per stream (WNR3: 23 bits per stream)**. In MAG/JOINT lanes 2..31 also start at
  L + 16 bits of their chain hash (480 more check bits); in MAGT those bits carry data, the unused pool bits
  must be 0, and lanes 2..31 must end in [L, 2L). Plus: exact meet of the word and sign pointers, unused sign
  bits 0, ceil underflow. The hashes are vectorised (one 512-bit Murmur step per 4 symbols per lane).
- **Record:** 64-byte header (magic, model id, flags, owner, anchors, Bw, Bs, stream count, payload bytes),
  u32 end offset per stream (4 B x 558 = 2.2 KB; WNR3 has 2 B x 19,440 = 38.9 KB of u16 lengths), payload,
  zero padding to 4096. Model blob: 1,952 B (MAG) / 2,246 B (JOINT): 3 x 7 weight CDFs + 3 scale CDFs.

Where the bytes go (MAG 2048, from the ideal code lengths of `logs/eval_test.txt`): weights 3.57662 bit/w +
scales 147.8 KB ideal = 15,971,843 B with header and offsets; the real records are 16,027,826 B unpadded, so
the rANS skeleton costs 56 KB per record = 100 B per stream = 3.1 B per lane (of which ~2 B is the 16-bit
start-state slot every lane has, ~1 B the 32-bit flush, ~2 B per stream the last sign word). WNR3 spends 108 KB
per record on the same thing (5.57 B x 19,440 streams, `wnr4-record`). MAGT moves 60 B per stream of sign bits
into the start-state slots: 23.6 KB per record of overhead left.

## Size evidence (held-out, real round trip)

| measurement | how | result |
|---|---|---|
| model cost of the simpler contexts | ideal -log2 p, TRAIN counts, TEST data (`ff eval`, `logs/eval_test.txt`) | WNR3's 567 contexts at 14 bit 3.57587 bit/w; 7 srel contexts, 8 magnitudes + raw sign, 12 bit: 3.57662 (+0.018% of raw); JOINT 12 bit 3.57668; per-position tables (7 x 32 contexts) 3.57665: no gain. Bound only. |
| scale CDF precision | same | 12 bit 155.9 KB/record, 14 bit 147.8 KB: **keep 14 bit for scales** (12 bit would cost +0.043% of raw) |
| every row of the table | `run_sizes.sh` -> `logs/enc_*.txt` | 41/41 records decode with the WNX reference decoder to Z(source) byte for byte, in all 7 configurations |
| encoder determinism | re-encoding L06 E133 with the final `ff` | byte-identical to the stored `out/mag_2048` and `out/joint_2048` records |

The size gain is a skeleton effect, so it follows the number of lanes: at Bw = 1024 (35,712 lanes, 1.8x the
19,440 single-lane streams of WNR3) the gain is gone (+0.036%).

## Speed evidence

`bench.c`, builds `-O3 -march=haswell -fopenmp` (AVX2: the WNX AVX2 kernels and `wex_fast_decode` in its AVX2
form) and the same plus `-mavx512f -mavx512bw -mavx512vl -mavx512dq` (adds the WNX AVX-512 kernels;
`wex_fast_decode` has no AVX-512 kernel, it is the same AVX2 code). 8 TEST records (L00, L05, ..., L35), 15
rounds; in every round each WNX decoder is timed directly before or after (alternating) a `wex_fast_decode`
pass over the same 8 records, and the ratio of that pair is one sample. Wall clock and process CPU time
(`CLOCK_PROCESS_CPUTIME_ID`) are both in `logs/time.txt`; on this oversubscribed machine CPU time is the cleaner
per-core figure. G = number of streams decoded in one loop.

1 thread, CPU time, run 1 / run 2 (wall-clock ratios agree within the noise):

| build | decoder | ms / record | MB/s | ns / weight | speed-up, median of paired ratios (min..max) |
|---|---|---:|---:|---:|---|
| AVX2 | WNR3 `wex_fast_decode` | 106.1 / 107.3 | 177 / 175 | 3.00 / 3.03 | 1 |
| AVX2 | MAG AVX2 G=1 | 31.4 / 31.8 | 598 / 592 | 0.89 / 0.90 | 3.54x (2.66..3.90) / 3.49x (2.87..3.94) |
| AVX2 | MAGT AVX2 G=1 | 31.3 / 32.3 | 601 / 581 | 0.88 / 0.91 | 3.39x (3.13..4.39) / 3.40x (2.66..4.24) |
| AVX2 | JOINT AVX2 G=1 | 31.8 / - | 591 / - | 0.90 / - | 3.06x (2.76..3.59) / - |
| AVX-512 | WNR3 `wex_fast_decode` | 102.3 / 100.9 | 184 / 186 | 2.89 / 2.85 | 1 |
| AVX-512 | MAG AVX-512 G=2 | 23.9 / 18.8 | 786 / 1002 | 0.68 / 0.53 | 4.57x (3.51..5.80) / 4.87x (3.19..5.46) |
| AVX-512 | MAGT AVX-512 G=2 | 23.9 / 21.1 | 785 / 891 | 0.68 / 0.60 | 4.20x (3.16..5.18) / 4.82x (3.95..5.37) |
| AVX-512 | JOINT AVX-512 G=2 | 24.4 / - | 771 / - | 0.69 / - | 4.19x (3.26..4.64) / - |
| AVX-512 | MAG, AVX2 kernel in this build, G=1 | 34.2 / 28.6 | 550 / 657 | 0.97 / 0.81 | 3.11x / 3.34x |

4 threads per record (run 2, `OMP_WAIT_POLICY=passive`, wall clock, paired): AVX2 build: `wex_fast_decode`
56.3 ms per record (334 MB/s), MAG AVX2 G=1 17.1 ms (1,099 MB/s, **3.24x**), MAGT 17.7 ms (3.08x); AVX-512 build:
`wex_fast_decode` 46.3 ms (406 MB/s), MAG AVX-512 G=2 11.8 ms (1,594 MB/s, **3.96x**), MAGT 11.9 ms (4.31x). With
the default (spinning) OpenMP wait policy the same 4-thread runs gave only 1.9-2.3x and absolute times close to
1 thread for both decoders: with more runnable processes than cores, threads that spin at the region barriers
steal the time of the ones that work, and the 3-4x shorter WNX decode loses proportionally more. 4 records at
once with 1 thread each (throughput mode, CPU-time paired ratio): AVX2 build MAG 3.26x (run 1) / 3.22x (run 2),
AVX-512 build MAG 4.55x / 4.33x. A best-of-7 probe on one record (`logs/scal.txt`, load 6.9) shows both decoders
scale while a vCPU is free (3 threads: `wex_fast_decode` 107.7 -> 48.3 ms, WNX AVX-512 21.5 -> 8.5 ms) and both
lose at 4 threads (60.5 and 20.5 ms) when the 4th vCPU is taken by other work. A clean 4-core scaling curve needs
an idle machine.

Byte identity of the fast decoders (`run_equal.sh` -> `logs/equal_*.txt`): every WNX kernel (AVX2 and AVX-512,
G = 1 and 2, 1 and 4 threads) and `wex_fast_decode` reproduce Z(source) on all 41 TEST records in all 7
configurations; the output buffer is filled with 0xa5 before each decode. Mutation test (`bench mutate`,
`logs/mutate_*.txt`): 1,200 damaged records (bit flips anywhere in a stream, in the lane states, in the sign
words, random bytes, 4 flips, swapped equal-length streams) - the reference decoder rejected 1,198 and accepted
2 whose output was Z(source) (a mutation that changed nothing), and every fast kernel gave the same verdict and,
when accepting, the same bytes. Not a negative control: no check was removed to see the test notice it.

What a step costs, counted from the code (`wnx_fast.h`): about 70 AVX-512 uops or 150 AVX2 uops per 32
symbols, no table lookup per symbol (7 compares against the block's thresholds, 2 in-register permutes for
start/frequency), one load per 8 or 16 lanes for the words, one pdep for the signs. At 2 (AVX-512) or 3 (AVX2)
vector uops per cycle that is 35-50 cycles per step, which is about what was measured (0.68-0.89 ns per
symbol at 2.1 GHz under load), so the kernels are throughput-bound and close to what this design allows; the
remaining levers are fewer uops, not more interleaving (G=2 helps AVX-512 by 5-20%, not AVX2: 16 ymm registers).

### What transfers to the production machines (inferred, not measured)

- **i7-7700K (AVX2, no AVX-512)**: the AVX2 build is the relevant one. Kaby Lake has the same three 256-bit
  vector ports, 2-uop `vpmulld` and fast `pdep`/`pext` as the core measured here, and both decoders are AVX2,
  so the 3.1-3.5x ratio should roughly hold; `fast-cpu-decode` measured `wex_fast_decode` at 129 ms per record
  on an i7-7700HQ (same core), so WNX would be about 37-42 ms per record per thread there (estimate).
- **Ryzen 9 7950X3D (AVX-512, Zen 4)**: both kernels run; Zen 4 executes 512-bit ops as two 256-bit halves with
  more vector pipes, fast `vpermd`, `vpexpandd` (register form) and `pdep`. Expect between the AVX2 and the
  AVX-512 ratio, i.e. about 3-4.5x over the 71.6 ms of `wex_fast_decode` there: roughly 16-24 ms per record per
  thread (estimate). Zen 2 and older (microcoded `pdep`) would need a pdep-free sign path (a 256-entry table per
  byte of the mask does it).
- 4-thread scaling: measured only on this loaded VM (above). The WNX decoder parallelises over 558 streams (270
  pairs of weight streams) in two OpenMP regions like `wex_fast_decode`; nothing in the format limits it, but at
  Bw = 4096 there are only 135 pairs, still enough for 16 threads.

## GPU kernel (from reading `c/wex_cuda.cuh`; no GPU was available, nothing here was run)

Today (`wex_cuda_nr2_weights<1>`): one thread per weight stream (17,280 per record, blocks of 32 threads, one
matrix per block), 2048 symbols per thread. Shared memory per block: the matrix's 567 CDF rows (18 KB) and a
64-word cp.async ring per lane (8 KB). Per symbol and lane: two ring reads, a 16-way SIMD-in-word CDF compare
(`wex_cuda_nr2_row_find`: 8 subtractions + 8 `prmt` + masks + `popc` + 14 `vmax2/vmin2`), a shared-memory row
load whose address depends on the previous symbol (the context chain), the rANS update, funnel-shift
renormalisation of 0..2 bytes; every 32 symbols 4 Murmur blocks, the CZ test and the ceil SWAR. The 32 lanes of
a warp decode 32 different streams with 32 different contexts.

WNX (one warp per stream, lane = block position; 540 warps per record at Bw = 2048, the same number of warps as
today's 540 blocks, and the same 2048-step serial depth):
- context: one scale byte per step, warp-uniform; the 7 thresholds of (matrix, srel) are uniform (constant or
  shared memory broadcast, 28 B); symbol = 7 compares of the lane's slot (or 4 `vcmpgeu2` on packed pairs);
- start / frequency: `__shfl_sync` from the lane that holds entry `sym` of the 8-entry table (1 shuffle), or an
  8-entry shared table (no bank conflicts);
- renormalisation: `ballot(x < L)`, `popc(mask & lanemask_lt)` gives the lane's word index; the words of a step
  are contiguous (at most 64 B), so one coalesced 128-B load serves 1-2 steps, kept in registers and handed out
  with a shuffle; the pointer advance `popc(mask)` is uniform;
- signs: `ballot(sym != 0)`, `popc(nz & lanemask_lt)` = the lane's bit in a warp-uniform 64-bit sign buffer;
- hash: each lane pair folds one Murmur block every 4 steps (one shuffle to pair the lanes), G at the stream end;
- ceil: `ballot(mag == 5) && !ballot(mag > 5)` is uniform; D is 2 ALU ops per lane; one lane decrements the
  scale; output: `shfl_down(nib, 16)` makes byte k in lane k (k < 16), 16 consecutive bytes stored by 16 lanes
  (or packed to 4 words);
- MAGT: each lane keeps its last 24 nibbles in 3 registers until the final states give the pool.
So per symbol about 20-30 ALU instructions and 3-5 warp collectives, against about 40-60 instructions plus two
dependent shared loads today; no 18 KB CDF table and no per-lane ring (shared memory per block falls from about
26 KB to well under 1 KB, so occupancy is set by registers); every global read is coalesced. The dependency
chain per step now runs through `ballot`/`shfl` (tens of cycles), so single-record latency should be in the
same range as today rather than far below it; batch throughput should improve with the lower instruction count.
Scales: a separate kernel as today (kernel A), one warp per scale stream with the same 16-K-entry direct table
in shared memory; Bs = 512 gives 69 warps per record and today's 512-step depth (costs +0.029% of raw against
Bs = 2048). The Vulkan prototype maps the same way onto subgroup ballot / shuffle (subgroup size 32 on NVIDIA;
on a 64-wide AMD wave two streams per wave).

## Negative results and limits

- **4 threads per record need a passive OpenMP wait policy on a loaded machine**: with the default spinning policy
  the 4-thread ratios fell to 1.9-2.3x and neither decoder beat its own 1-thread time; with `OMP_WAIT_POLICY=passive`
  the ratios were 3.1-4.3x. An idle-machine scaling curve was not obtained.
- **More lanes cost bytes, fewer save them**: the gain is entirely the per-stream skeleton (3.1 B per lane,
  WNR3 5.6 B per stream). Bw = 1024 is +0.036%, 4096 is -0.43% (MAG) but halves the GPU warps and doubles the
  serial depth (the single-record latency trade the owner already measured for WNR2 chunk sizes).
- **Context richness is not worth it here**: position-in-block tables (free in this layout, lane = position)
  gain nothing (3.57665 vs 3.57662 bit/w); srel-only costs +0.018% against the 567 contexts (bound, held-out).
- **JOINT vs MAG**: same size within 0.003%, same speed within noise; MAG is the one that admits the MAGT trick
  (-0.17%), JOINT needs no sign region. A JOINT-tail variant (raw nibbles in the start states) would save less
  (a raw nibble costs 4 bits against 3.58 coded).
- **12-bit scale CDFs**: +0.043% of raw; scales stay at 14 bit (the 16-bit renormalisation allows up to 16).
- **AVX2 interleaving of 2 streams (G=2)** does not help (register pressure); on AVX-512 it gives 5-20%.
- **tANS/FSE was not built**: with 8 symbols and per-block contexts the rANS step is 7 compares and 2
  in-register permutes; a tANS step needs a per-lane table gather and a variable-width bit read per lane, which
  is a gather plus a 32-lane prefix sum on AVX2/AVX-512 and on a GPU a 7 x 4096-entry table per matrix that does
  not fit shared memory. It cannot be cheaper than the measured rANS step on either target.
- **Not checked**: the baseline records were coded with the production model (trained on 80 other experts), the WNX
  records with a model trained on TRAIN; both are out-of-sample on TEST, but not the same training set; no GPU or Vulkan run; no idle-machine timing; only 41 TEST records (8 for timing); the model
  is trained on 80 records (production: 80 other ones); the mutation test has no negative controls; the fast
  kernels need 192 readable bytes after a stream (records padded to 4096 give that; otherwise the last stream
  takes the reference path, as `wex_fast_decode` does with its slack rule); no encoder speed figures (the
  prototype encoder plus reference decode took about 1 s per record on one thread); MAGT puts no hash bits in
  lanes 2..31 (the 32-bit G per stream is still stronger than WNR3's 23 bits, but the extra lane checks of
  MAG/JOINT are gone).

## Recommendation

If a WNR4 / codec 7 is made, make it **this skeleton: 32-lane interleaved rANS per stream, 16-bit words, L = 2^16,
one block per step with the srel context, magnitude alphabet + raw signs (C1), with the tail signs in the lane
start states (C1t MAGT)**, Bw = 2048 (keeps today's warp count and serial depth) and Bs = 512..2048. Measured
on held-out data: -0.42..-0.45% of raw smaller than the stored WNR3-CZ records (about 0.6 GB on the 123 GB
container, extrapolated from 41 records), 3.4x faster CPU decode than `wex_fast_decode` on AVX2 and 4.2-4.8x with
AVX-512 on one thread (3.1x and 4.3x with 4 threads), a 32-bit hash per stream instead of 23 bits, and a GPU kernel without
context chain, CDF tables in shared memory or per-lane rings. Bw = 4096 adds another -0.07% for batch-only
decoding. The coded domain (CZ, ceil form, rule g) and the builder's prepare step stay as they are. Before
committing: (1) write the CUDA warp kernel and measure batch-16 and single-record latency on the RTX 3070
against today's 0.37 ms/record; (2) rerun `run_time.sh` on an idle i7-7700K and 7950X3D; (3) a mutation run
with negative controls like `fast-cpu-decode/negative_control.sh`.
