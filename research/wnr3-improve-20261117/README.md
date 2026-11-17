# How much better can WNR3 get? Floor, candidate changes, and a WNR4 decision (2026-11-17)

**Answer.**
- **Size.** The best adaptive code found is **0.69% of raw** below the stored WNR3-CZ records. That is a bound,
  not a coder. 0.585 points of it are framing: u16 lengths, per-stream flush and 23-bit hash, padding. Only
  **0.11-0.13% is left in the model**.
- **What a WNR4 can get.** Real, held-out, round-trip-verified profiles get **-0.39..-0.42% of raw** with
  per-stream integrity kept or made stronger. That is about 0.56-0.61 GB over the 7,680 records. They get
  -0.21% with today's decoder structure kept, and up to -0.51% if the owner accepts record-level integrity
  and longer chunks.
- **What is not worth it.** Model-only changes (classes, sibling context, scale alphabet, K table sets,
  precision, retraining) total **≤ 0.06-0.07%** and do not justify a codec by themselves.
- **Speed without a format change.** A new SIMD decoder for today's format decodes **1.9-2.4x faster** than
  the shipped `wex_fast_decode` on one core of this Xeon. Its output is byte-identical on all 121 records.
- **Speed with a new format.** The WNX skeleton (32 interleaved rANS lanes per stream) decodes another
  **~1.8-1.9x** faster than that improved decoder. That is 3.4x (AVX2) or 4.4x (AVX-512) against the shipped
  decoder.
- **No GPU numbers.** The production GPU path was not measured: there is no GPU here. Every GPU statement
  below is read from the code.

**Recommendation in two lines.** Adopt the decoder-only patch now, after an A/B on the two production CPUs.
Build WNR4 only on the WNX skeleton (MAGT, Bw = 2048, Bs = 512), and only after a CUDA warp kernel for it
matches today's 0.37 ms/record on the RTX 3070. If it does not, the fallback is profile C (-0.215%) or no WNR4.

All numbers below were measured on this machine, CPU only: a shared 4-vCPU Xeon (Sapphire/Emerald Rapids
class, 2.1 GHz, AVX2 + AVX-512), on the 121 sample records of `/tmp/wnr3-data` (plus 37 fetched
anchor-shifted records where stated). Where a skeptic's reproduction differs from a lane's number, the
**skeptic's number is used** and the lane's is given in brackets.

## Data, baseline, conventions

- Raw record = 18,800,640 B. 0.01% of raw = 1,880 B per record = 14.4 MB over 7,680 records (raw container
  144.39 GB; the container file is 123.31 GB = 0.8540 of raw).
- Stored baseline, recomputed from file sizes by four skeptics independently:
  - all 121 sample records **0.85436** of raw;
  - the 41 TEST records 0.85534;
  - the 40 typical TEST records (one per layer, L39 E120 apart) 0.85364.
  - The container's entry table gives 0.85398 for all 7,680. The brief's 0.8546 is the 24-record figure of
    mxfp4-contexts.
- Split (data README): TRAIN = experts (37l+11)%192 and (37l+101)%192 (80 records). TEST = (53l+7)%192 (40)
  plus L39 E120. The model lane also used a split B.
- "Held-out" = no table, threshold or alphabet was trained on the scored record. Per-record tables stored
  inside the record count as two-part codes, and their bytes are counted.
- "Real" = actual rANS encode, then decode compared byte for byte (or by SHA-256) with Z(source). Z(source) is
  what the production reference decodes the stored record to. Every harness first re-encoded the stored records
  byte-identically with the production encoder, so all lanes work in the exact WNR3-CZ coded domain.
- **Format effect vs "vs stored".** A model retrained on TRAIN beats the production model by 835 B/record
  (0.004%) on typical records, and by 0.0255% on the 41 TEST records because of L39. The WNX and overhead
  lanes therefore compare against **WNR3 re-encoded with the same TRAIN model** where possible. The model and
  floor lanes compare against the production model.
- Container figures (MB, GB) are **extrapolations** = % of raw × 144.39 GB, unless marked exact.
- Status codes in the tables:
  - **R** = reproduced by the skeptic (rerun identical and/or independent re-derivation).
  - **R≠** = reproduced with a different number; the skeptic's number is shown.
  - **NC** = not checked.
  - **E** = estimate or inference, not a measurement.

## 1. The floor: how far WNR3-CZ is from the best code found

Lane `floor`, check `check-floor`. 40 typical TEST records. The floor is an ideal adaptive code length from a
PAQ-style context mixer (`cm.c`, 13 weight and 10 scale context models, learning only from the record's own
past, optionally primed on one TRAIN record). The production static model is allowed per group where it is
better. 41 B/record is added for coder overhead and a selector.

| | of raw | bit/weight | status |
|---|---:|---:|---|
| stored WNR3-CZ | 0.85364 | 3.5883 | R |
| ideal cost under the production WNM2 (= stored minus framing) | 0.84779 | 3.5743 | R (independent: 0.5854% framing) |
| best adaptive code found (bound) | **0.84674** [lane 0.84673] | 3.5714 | R≠ (rerun primes 12 records, not 14; negligible) |

The 0.69% gap (per record 0.60..1.61%) splits as follows (all measured; the model rows are bounds):

| part | % of raw | container (E) | what it is |
|---|---:|---:|---|
| u16 length table | 0.207 | 299 MB | 2 B × 19,440 streams |
| weight-stream rANS start/flush | 0.328 | 474 MB | 3.57 B/stream, of which 2.875 B is the 23-bit hash slot |
| scale-stream start/flush | 0.041 | 59 MB | |
| 4096-B padding | 0.009 | 13 MB | |
| **framing total** | **0.585** | **845 MB** | |
| weight model (static 567 contexts vs best adaptive) | 0.069 | 100 MB | layers 0-7 0.157, 8-38 0.028, L39 E154 0.658 |
| scale model | 0.037 | 53 MB | |
| learning cost left in the bound (priming gains 0.020 more) | ~0.02 | ~29 MB | measured on 13-14 records |

Reading:
- **Integrity is the largest single item.** Today it costs 2.875 B × 19,440 streams = 55.9 KB per record =
  0.297% of raw. Lengths are next.
- **The model is close to done.** An adaptive CM restricted to WNR3's own contexts is within -0.06..+0.04% of
  the static model, so adaptivity alone buys nothing.
- **Mutual information (in-sample, plug-in with Miller-Madow correction, all 121 records).** The only
  cross-tensor relation with information is w1↔w3 at the same [i,j]: 0.0147 bit/w on w3 in layers 0-7 and
  0.0011 in layers 8-39 (skeptic spot-check on 7 records matches to 1e-5).
- **Relations that carry nothing.** Other experts of the same layer (≤ 6e-5 bit/w), the same expert in the
  adjacent layer (≤ 3e-6, 2 pairs), and sign against anything including the sibling's sign (≤ 1e-4). The
  skeptic did not rerun these negative MI results (NC), but they agree with mxfp4-contexts.
- **L39 E120.** Stored 0.923 of raw, floor 0.820. Its scales are coded by an absolute global table that does
  not fit its anchors (121,122,121). Its w2 also has per-neuron column structure. The scale part is fixed by any
  anchor-relative or per-record scale table (rows S4, S5, M3 below).

## 2. Every candidate, one table

Effects are held-out real bytes per typical record (40 TEST records) unless marked. The "speed" column refers
to section 3. "WNR4" = needs a new codec id.

### 2a. No format change (decoder work)

| # | candidate | size | decode speed (idle machine, timing rerun) | integrity | format | decoder cost | status |
|---|---|---|---|---|---|---|---|
| D1 | **kernel 4** (AVX-512 F/BW/VL/CD/VBMI2, 3 × 16 streams, two-level table, AVX-512 scale kernel): `fastdec/wex_fast_decode.patch` | 0 | 1 thread **2.19-2.40x** vs the shipped decoder (42.8 vs 103.0 ms/record); 4 threads 2.07-2.19x (12.6 vs 25.7 ms) | same checks; 121/121 equal; 0 mismatches in the mutation tests; hash negative controls caught | none | tables 1.84 MB (today 224 KB), init 0.17-0.8 ms: production must cache them (`wex_fast_decode_with`) | R≠: under load the skeptic got 2.07x, and **1.95x against `k1 × 6`**, today's best setting on this CPU |
| D2 | **kernel 3** (AVX2, 4 × 8 streams, same tables) | 0 | 1 thread **1.91x** (55.8 ms); 4 threads 1.71-1.81x | same | none | same | R≠: skeptic 1.73-1.76x under load. **Gathers: unknown on the i7-7700K** (Kaby Lake, GDS microcode) |
| D3 | today's kernel 1 with 6 streams (`-DWEX_FAST_LANES=6`, an existing option) | 0 | 1.09-1.17x on this CPU | same | none | none | R≠ (lane 0.92, skeptic 0.855-0.88 ratio). The repo found 4 streams best on the 7950X3D and 7700HQ: CPU-specific |

The per-symbol breakdown of today's kernel 1 comes from ablation builds and is an estimate (NC): search 28-34%,
renormalisation 22-26%, next-context load 2-12%, fused hash ~7%. One stream is a ~22-cycle dependency chain per
symbol; 4 streams give ~7.5 cycles. A scalar table lookup instead of the AVX2 search was **1.4-2.9x slower**
(negative result).

### 2b. New skeleton, WNR3-style (lane `overhead`; against WNR3 re-encoded with the same TRAIN model)

| # | candidate | B/record | % of raw | container (E) | speed | integrity | format | decoder cost | status |
|---|---|---:|---:|---:|---|---|---|---|---|
| S1 | coded length table (group-32, wnr4-record) | -24,460 | -0.130 | 188 MB | none measured (host offsets 8.5 → 36.5 µs, prior work) | unchanged | WNR4 | host only | R (independent -0.1301%) |
| S2 | minimal 3/4-B start state | -2,406 | -0.013 | 19 MB | none | unchanged | WNR4 | trivial | R (independent 2,423 B) |
| S3 | 64-B instead of 4096-B alignment | -1,630 | -0.009 | 13-16 MB (16 MB exact from wnr4-record) | none | – | WNR4 / container | none | R |
| S4 | per-record order-0 scale table, separate scale streams | -5,036 (floor lane: -4,982 over 120 records) | -0.027 (median -0.019) | 39 MB | none | unchanged | WNR4 | ~25-60 B table parsed per record; GPU table build per record not measured | R (two lanes, independent entropy bound -0.02652%). **Median is noise-level; the value is the tail: L39 E120 -4.75% of its raw** |
| S5 | scales **inline** in the weight streams, per-record 16-symbol anchor-relative table | -17,041 | -0.091 | 131 MB | E: GPU kernel A and its 88 KB shared table disappear; kernel B decodes +3.1% symbols | unchanged | WNR4 | kernel B and fast-CPU change, not written | R (mostly deterministic: 2,160 streams × 5.6 B removed) |
| S6 | **profile C = S1 + S2 + S5** (+ S3) | **-40,471** | **-0.215** | **310 MB** | reference-style decoder: no detectable change (±15%); fast CPU / GPU not measured | 23-bit stream hash unchanged; audit 0/61,280 wrong | WNR4 | moderate | R (independent estimate -0.217%) [lane "vs stored" -0.229% includes retrain and alignment] |
| S7 | tail 5 nibbles in x0 (3-bit stream hash) + 32-bit record digest, on top of C = **profile E** | -38,591 more; E total **-79,088** | -0.205; E **-0.421** | 608 MB | digest = 1 XOR per stream (CPU), warp XOR + atomicXor (GPU, E) | record-level. P(false accept) 2^-32 for one corrupted stream (today 2^-23), but 2^-(3n+29) for n ≥ 2 streams (today 2^-23n). Per-stream localisation lost; the whole record must be buffered fail-closed | WNR4 | GPU commit-kernel change, not written | R for size and audit (0/18,000 wrong; the no-digest control fails on ~1/64 of swaps). **"Stronger overall" not reproduced: only for single-stream corruption** |
| S8 | tail 2 + digest (15-bit stream hash), profile D | -55,911 (D) | -0.297 | 429 MB | as S7 | as S7, 15-bit stream check | WNR4 | as S7 | R |
| S9 | chunks 4096:1024 (alone / profile H = E + 4096:1024) | -54,110 / -95,471 | -0.288 / -0.508 | 416 / 734 MB | **single-record GPU latency up** (earlier 5090/V4: 0.264 → 0.437 ms); batch unchanged | as base | WNR4 build parameter | none | R (size). Latency NC (prior measurement) |
| S10 | 16-bit renormalisation, L = 2^16, 23-bit hash kept | **+8,590 (worse)** | +0.046 | – | – | unchanged | – | – | R (identical rerun only). The earlier -0.045% came only from dropping 7 hash bits |

S1-S9 are partly cumulative as listed (C, D, E, H are measured combinations). They are **not additive to the
WNX rows** below: both attack the same per-stream overhead.

### 2c. New skeleton built for decode speed: WNX (lane `fastfmt`)

Each stream is 32 rANS lanes sharing one 16-bit word stream (32-bit state, L = 2^16). One step decodes one
block, and all lanes share that block's context (srel only, 7 per matrix, 12-bit weight CDFs, 14-bit scale
CDFs). Weights are coded as 8 magnitudes plus raw sign bits. Each stream has a 32-bit MurmurHash3 check. CZ,
ceil form and rule g are unchanged. MAGT also puts the last 480 sign bits of each stream into the start states
of lanes 2..31. Sizes are padded real records (4096 B), **like-for-like against WNR3-CZ re-encoded with the same
TRAIN model** (skeptic). The lane's figures against the stored records are in brackets.

| # | variant | % of raw (per-record range) | container (E) | CPU speed vs shipped decoder, 1 / 4 threads (idle rerun) | vs best same-format decoder (D1/D2), 1 thread | GPU structure (E) | integrity | status |
|---|---|---:|---:|---|---|---|---|---|
| W1 | **MAGT, Bw = Bs = 2048** | **-0.419** (-0.436..-0.349) [-0.445] | 605 MB | AVX2 3.37x / 3.58x; AVX-512 G=2 4.42x / 4.13x | ~1.8x (AVX2) / ~1.9x (AVX-512), ratio of medians | 1 warp per stream; 540 weight warps and 2048-step depth as today; no CDF table in shared memory | **32-bit hash per stream** (today 23) | R≠ |
| W2 | **MAGT, Bs = 512** (today's scale depth) | **-0.390** [-0.416] | 563 MB | as W1 | as W1 | 69 scale warps, 512-step depth | as W1 | R≠ |
| W3 | MAGT, Bw = Bs = 4096 | -0.487 [-0.512] | 703 MB | as W1 | as W1 | half the warps, 2x serial depth | as W1 | R≠ |
| W4 | MAG 2048 (no tail trick) / JOINT 2048 (15-symbol alphabet) | -0.246 / -0.249 | 355 / 360 MB | MAG AVX2 3.28-3.48x, AVX-512 4.54x; JOINT 2.98x / 4.40x | | | 32-bit + 480 lane-check bits | R≠ |
| W5 | MAG, Bw = Bs = 1024 | **+0.062 (worse)** | – | | | 2x warps | | R≠ |

- **Where the gain comes from.** It is skeleton, not model. The skeleton costs ~3.1 B per lane against 5.6 B
  per WNR3 stream. The srel-only model costs +0.018% (bound; independently reproduced as 3.57662 bit/w).
- **Rejected inside WNX.**
  - Per-position contexts gain 0.
  - 12-bit scale CDFs cost +0.043%.
  - JOINT is no better than MAG and cannot use the tail trick.
  - tANS was not built: it needs a per-lane gather and does not fit GPU shared memory (argued, E).
- **Equality.** All 41 TEST records decode to Z(source) in all 7 configurations, with every kernel. For timing,
  the timing lane re-encoded all 121 records, and they all decode to Z(source) too.
- **Mutation tests.** 1,200 cases in the lane plus 300 + 200 in reruns. Every fast kernel agrees with the
  reference. The skeptic added a negative control (reference checks removed), and the test caught it.

### 2d. Model changes (lanes `floor` and `model`; against the production model unless noted)

| # | candidate | B/record | % of raw | container (E) | speed | integrity | format | decoder cost | status |
|---|---|---:|---:|---:|---|---|---|---|---|
| M1 | w1 → w3 sibling context (srel3 × mx × clamp(s1-s3) × a1, 2,520 contexts, 2 table sets split at layer 8) | -5,941 | **-0.0316** (layers 0-7 -0.121, layers 8-39 -0.009) | 46 MB | E: w3 stream j must wait for w1 stream j, which halves the independent GPU chains. Not measured | unchanged | WNR4 | 171 KB of tables | R (independent -0.03162%; a 2-fold CV inside TRAIN picks the same design). **Expect -0.02..-0.035%**: global tables give -0.0198%, 40 training records give -0.024% |
| M2 | C16: 16 stream classes from the sum of srel over the stream's decoded scales (no selector bits) | -7,927 | -0.042 | 61 MB (64.2 stratified) | scalar prototype: no change. **AVX2 (2.8 MB of tables) and GPU (tables do not fit shared memory → regroup streams by class) not measured** | unchanged | WNR4 | high on GPU (E) | R |
| M3 | sesc: anchor-relative 10-value scale alphabet + escape, 14-bit | -3,836 typical; -88 KB mean on the 38 shifted records | -0.020 | 32.7 MB (stratified) | none | unchanged | WNR4 (model blob) | trivial | R. **Makes 16 of 37 shifted records bigger** (up to +91 KB); the mean comes from a few (121,122,121) records. Overlaps S4/S5 |
| M4 | sesc + z8 (symbol 8 at frequency 0) | **-4,224** | -0.0225 | ~36 MB (E for shifted) | none | requires CZ on every record | WNR4 (model blob) | trivial | **measured by the skeptic only** (the lane estimated it) |
| M5 | C16 + z8 + sesc | **-11,317** split A / -13,363 split B (medians -7,858 / -9,498; without L39 -9,366 / -11,595) | -0.060 / -0.071 (without L39 -0.050 / -0.062) | **92.4 MB = 0.064% of raw** [lane "0.075%" was of the 123.3 GB file] | as M2 | as M4 | WNR4 | as M2 | R (independent numpy bound within 6 B/record). Bootstrap 95% CI of the split-A mean: -16.0..-8.4 KB |
| M6 | encoder-chosen K = 2 / 4 / 8 table sets (selector bits counted) | -5,168 / -6,574 / -5,717 | -0.028 / -0.035 / -0.030 | 40-50 MB | host can sort streams per set | unchanged | WNR4 | moderate | R for K4 only. Beaten by M2 |
| M7 | 15- / 16-bit CDFs | -2,621 / -3,480 | -0.014 / -0.019 | | breaks the CUDA scale direct table in sm_86 shared memory (P15); AVX2 compare needs changes at P16 (E, read from code) | unchanged | WNR4 | | R for P16. Noise-level; most of it is scale-table waste that M3 removes at 14 bits |
| M8 | retrain on 80 TRAIN records | -828 (median **+692**) | -0.004 | | | | model only | | R. Noise |
| M9 | per-record models (in-sample ceiling, tables not counted) | -7,071 | -0.038 | | | | | | R. Net loss once the tables are stored |
| M10 | cross-expert, cross-layer, sign modelling | ~0 (MI ≤ 1e-4 bit/w) | ~0 | | | | | | MI spot-checked only (NC on all 121) |
| M11 | global MLP over a wide neighbourhood | w1 -0.0001 bit/w | ~0 | | | | | | NC; the lane calls it weak evidence |

How M1-M6 combine with each other and with WNX was **not measured**. They are different features, and the
floor says their sum cannot exceed ~0.11-0.13%. WNX's warp-uniform design has room for M2 (a stream is a warp,
so a class per stream is uniform) but not for M1 (a per-lane context). Both are untested inferences.

## 3. Decode speed, one table

Timing lane: one session on an effectively **idle** machine (load 1.0 during 1-thread runs). One driver for
every variant. 9 TEST records including L39 E120. Interleaved A/B, 7 rounds, wall clock, median (min..max).
All 29 equality runs were byte-identical on 121/121 records. AVX2 flags `-O3 -march=haswell`; AVX-512 flags
`-O3 -march=x86-64-v4 -mavx512vbmi2`.

| decoder | format | ISA | 1 thread ms/record | speed-up | 4 threads ms/record | speed-up |
|---|---|---|---:|---:|---:|---:|
| reference `wex_nr2_decode` | WNR3 | scalar | 898.8 | 0.11x | – | – |
| shipped `wex_fast_decode` (k1 × 4 fused) | WNR3 | AVX2 | 103.0 (89.2..115.2) | 1 | 25.7 | 1 |
| D2 kernel 3 × 4 | WNR3 | AVX2 | 55.8 | 1.91x (1.72..2.13) | 15.3 | 1.71x |
| D1 kernel 4 × 3 | WNR3 | AVX-512 | 42.8 | 2.40x (1.97..2.61); flag run 2.08-2.19x | 12.6 | 2.09x |
| WNX MAGT 2048, AVX2 G=1 | WNX | AVX2 | 30.5 | 3.37x | 7.48 | 3.58x |
| WNX MAG 2048, AVX2 G=1 | WNX | AVX2 | 30.7 | 3.28x | 7.67 | 3.32x |
| WNX MAGT 2048, AVX-512 G=2 | WNX | AVX-512 | 22.6 | 4.42x | 6.06 | 4.13x |
| WNX MAG 2048, AVX-512 G=2 | WNX | AVX-512 | 22.2 | 4.54x (flags: 4.50-5.41x) | 5.74 | 4.46x |

- **Run-to-run spread.** A variant's median ratio moves by up to ~10% between runs on the same binary. The WNX
  AVX-512 kernel moves by about ±10% with `-march`/`-mtune`.
- **Thread scaling, 1 → 4 threads (idle).** Shipped decoder 4.0x, k3 3.65x, k4 3.4x, WNX 3.9-4.0x.
- **Lane claims the rerun did not reproduce.** "AVX2 G=2 is 3-10% slower" and "the spinning OpenMP policy gives
  only 1.9-2.3x" were both effects of the loaded machine.
- **Transfer to production machines (E, not measured).**
  - i7-7700K: only the AVX2 rows apply. D2 and the WNX AVX2 scale kernel use gathers, which are slow on Kaby
    Lake and slower under the GDS microcode. An A/B there is required before making D2 the default.
  - 7950X3D (Zen 4): every row runs, but 512-bit ops execute as two halves, so the AVX-512 ratios must be
    measured there.
- **GPU: nothing measured.** WNX's warp kernel is a design read against `c/wex_cuda.cuh`: per symbol ~20-30 ALU
  ops and 3-5 warp collectives, against ~40-60 ops plus 2 dependent shared loads today; shared memory per
  block drops from ~26 KB to < 1 KB. Its dependency chain runs through ballot/shfl, so single-record latency may
  not improve.

## 4. Recommendation

### Decoder work that needs no format change (do first)
1. **Adopt `fastdec/wex_fast_decode.patch`.** It touches only the header in
   `wex4/research/fast-cpu-decode/`; the brief's `c/wex_fast_decode.h` does not exist.
   - Make kernel 4 the default in AVX-512 builds **after** an interleaved A/B on the 7950X3D (`fastdec/abx.py`,
     `out/test_orig:10` vs `out/test_z:26`).
   - On the i7-7700K, make kernel 3 the default **only if** an A/B there shows a gain.
   - Cache the 1.84 MB tables through `wex_fast_decode_with`.
   - Expected: ~2x per core on this Xeon (measured); unknown on the production CPUs.
2. Cheap follow-ups, estimated: an AVX2 scale kernel (~3%), and the same two-level table idea in the
   CUDA/Vulkan decoders (an inference; the GPU decoders were not examined).

### WNR4: build it only if all of the following hold
- 0.39-0.42% of raw (~0.56-0.61 GB on the container) is worth a new codec id, new encoders and new GPU kernels.
- A CUDA warp kernel for WNX, written and timed on the RTX 3070, is **no slower than today's 0.37 ms/record in
  batch-16** and acceptable for single-record latency. The Vulkan prototype needs the same check (subgroup
  ballot/shuffle).
- The A/B on the 7700K and 7950X3D confirms the CPU ratios. CPU decode is then ~1.8-1.9x faster than the
  improved decoder for today's format, and ~3.4-4.4x faster than the shipped one.

**Recommended WNR4 (codec 7) profile, if built.**

| element | choice | evidence |
|---|---|---|
| stream skeleton | 32 interleaved rANS lanes per stream, one shared 16-bit word stream, 32-bit state, L = 2^16 | W1/W2, measured |
| weights | one step = one block; srel-only context (7 per matrix); 8 magnitudes + raw sign bits; 12-bit CDFs | +0.018% model cost (bound), measured |
| tail | MAGT: last 480 sign bits in the start states of lanes 2..31 | worth -0.17 points of W1's -0.42%, measured |
| chunking | Bw = 2048 (today's warp count and depth); Bs = 512 (today's scale depth). Bs = 2048 adds -0.03%; Bw = 4096 adds -0.07% for batch-only use | W1-W3 |
| scales | 14-bit CDFs (12-bit costs +0.043%) | measured |
| integrity | 32-bit MurmurHash3 per stream, seeded as WNR3; exact pointer meet; ceil-underflow check | stronger than today's 23 bits, at lower total cost |
| coded domain | CZ (mandatory), ceil form and rule g exactly as WNR3-CZ; builder's prepare step unchanged | all round trips == Z(source) |
| expected size | **-0.39% of raw** like-for-like (Bs = 512), about 563 MB on the container (extrapolated from 41 records) | R≠ |

Optional additions. None was measured on WNX; each must be round-tripped on it first.
- A per-record anchor-relative scale table (S4/M3 idea). Estimated -0.02..-0.03%. It removes the L39 E120-type
  tails.
- C16 stream classes (M2). Up to ~-0.04% on the 567-context model; the gain on WNX's 7-context model is unknown.
- 64-B alignment. -0.009%, container level.

**Fallback if the GPU gate fails: profile C** on today's skeleton (S6): -0.215% (~310 MB), integrity unchanged,
GPU kernel A removed. Its kernel B and fast-CPU changes are still to be written and timed. Add profile E's
tail5 + record digest (-0.421%) only as an explicit integrity-policy decision: record-level verdicts,
whole-record fail-closed buffering, and weaker than today when two or more streams are corrupted (both rates are
negligible for random faults). Add M1/M2 only after their GPU cost is measured; together they are worth roughly
0.05-0.07%. If neither the GPU gate nor profile C's 0.2% justifies a codec, **do not build WNR4**: keep WNR3-CZ
and ship the decoder patch.

### Not worth doing (measured)
- 16-bit renormalisation in the WNR3 skeleton (+0.046%).
- More lanes (Bw = 1024: +0.06%).
- Richer or per-position contexts.
- Cross-expert, cross-layer and sign modelling (~0).
- 15/16-bit CDFs (≤ 0.019%; they conflict with the GPU scale table).
- Optimal quantiser (0 on weights, worse on scales).
- Retraining or more training data (-0.004%, median record bigger).
- Per-layer or per-record models (net loss).
- Encoder-chosen K sets (beaten by classes).
- A scalar table lookup in kernel 1 (1.4-2.9x slower).
- JOINT alphabet.

## 5. What was not checked

- **No GPU.** No CUDA or Vulkan kernel was written or run for any candidate. Every GPU cost or benefit here is
  read from `c/wex_cuda.cuh`:
  - WNX warp kernel, inline scales, record digest;
  - the w1-before-w3 dependency, C16 class regrouping, per-record scale tables;
  - single-record latency of longer chunks (only earlier 5090 figures exist).
- **Shared machine.** The lanes timed at load 2-13; their absolute times are not usable. The final timing rerun
  ran on an idle machine, but on one Xeon only (Sapphire/Emerald Rapids class). Nothing was timed on the
  i7-7700K or the 7950X3D. Ratios moved by up to ~10% between runs.
- **Sample.**
  - 121 of 7,680 records (3 per layer + L39 E120), plus 37 fetched anchor-shifted records in the model lane.
  - The 38 anchor-shifted records are fully enumerated from all 7,680 headers.
  - All container figures are extrapolations. Typical-record means are sensitive to one layer-39 record per
    split (~2 KB of the M5 mean). No MTP experts.
- **Integrity.**
  - The full 61,280-trial audit with header-bit flips was not rerun by the skeptic; a reduced 18,000-trial
    audit was. 0 events bound the false-accept rate only at ~5e-5 per corruption; 2^-32 is analysis.
  - WNX's mutation tests cover streams only, not header/offset fields.
  - The record-digest fail-closed path is not implemented in any fast decoder.
- **Not rerun by any skeptic.**
  - Lanes: mi2.py on all 121 records, mi.py, adjacent.py, nn.py; P15, C32, K2, K8, C8×K sets, T40 retrains,
    the bound-only screens; the fastdec ablation breakdown and its -4% scale-kernel A/B; the lane's AVX2
    all-variant equality on 12 records.
  - The timing rerun superseded the lanes' 4-thread numbers.
- **Combinations.** Overhead-lane profiles and WNX are not additive. Model items with WNX, sibling with C16,
  and per-record scale tables with WNX were not encoded together.
- **Not in a real WNR4 parser.** All round trips use prototype codecs (`ovh_proto`, `mlib.c`, `wnx.h`, `rt.c`).
  The escape path for out-of-window scales was only exercised with injected values: no real record uses it.
- **Encoder speed** was not measured (the WNX prototype encoder took ~1 s per record).

## Files (this folder, 3.7 MB, no record data, no binaries)

Each subfolder holds the scripts and small raw results of one lane or check. Each has a `RERUN.md` with its
rerun command; the lanes' own `RESULTS.md`/`README.md` and the checkers' `CHECK.md`/`NOTES.md` are included.
`collect.py` re-creates this layout from `/tmp/wnr3-research`. Generated data (coded slabs, encoded records,
table sets, fetched headers/records) is excluded and is rebuilt by the first steps of each rerun.

| folder | backs | key raw files |
|---|---|---|
| `floor/`, `check-floor/` | section 1, S4, M1, M10-M11 | out/floor.json, decomp.jsonl, cm_*_test.jsonl, rt.jsonl, rt_scale.jsonl, mi2.jsonl; check: out/sib_check.json, out/indep.jsonl |
| `overhead/`, `check-overhead/` | S1-S10, audit | out/report.txt, final_table.txt, audit_report.txt, explain_outlier.txt; check: indep/est.json, myaudit/summary.txt |
| `model/`, `check-model/` | M2-M9 | out/real_*.json, results.md; check: out/real_P14_z8_sesc.json, indep_A.json, indep_B.json |
| `fastfmt/`, `check-fastfmt/` | W1-W5 | logs/enc_*.txt, eval_test.txt, equal_*.txt, mutate_*.txt; check: logs/vs_trainwnr3.txt (like-for-like sizes), wnr3_train.txt, indep_bound.txt, mutate_negcontrol.txt |
| `fastdec/`, `check-fastdec/` | D1-D3 | wex_fast_decode.patch, logs/equal_*.txt, mutate_*.txt, negcontrol_final.txt, ablation_*.txt; check: logs/check_ref_*_121.txt, time_check_cpu1*.txt |
| `timing/` | section 3 | logs/timing_tables.md, timing_flags.md, raw_*.json, eq_summary.txt, session.txt |
