# WNR3 static model: training, quantisation, precision, table sets, free contexts (2026-11-17)

**Short answer.** The static model is close to its floor. Retraining (more or other data), a better
quantiser, and 15/16-bit CDFs each give at most 0.014% of raw on typical records, which is noise level.
Per-stream table sets give 0.02-0.04%. Three changes are worth putting in a WNR4 model. Together they make
every typical record smaller: **-11.3 KB/record (-0.060% of raw)** held-out on split A, **-13.4 KB (-0.071%)**
on split B. The 38 records of the container whose anchors are not 120 shrink by 175 KB on average.
Stratified extrapolation to the whole container: **about -92 MB (0.075% of the 123.3 GB container)**. All of
this was measured with real rANS encodes. Each record was then decoded and checked against Z(source) by
SHA-256. That is far less than the skeleton items of wnr4-record (coded length table alone -0.130%), so
model work alone does not justify a WNR4.

The three changes:
1. **Selector-free stream classes (C16).** Each weight stream uses one of 16 table sets. The decoder picks
   the set from the sum of srel over the stream's 64 already-decoded scales, so no selector bits are stored.
   Gain: -7.1 KB/rec against a retrained single model.
2. **Anchor-relative scale alphabet with an escape (sesc).** The scale symbol is `s - anchor[m]`. Only the
   10 values seen in training get a symbol (-3..+6), plus an escape followed by 8 raw bits, at 14-bit CDFs.
   Gain: -3.0 KB/rec on typical records and -88 KB on anchor-shifted ones.
3. **Symbol 8 gets frequency 0 (z8).** Under canonical zero, -0 cannot occur. Gain: -0.4 KB, free.

## Data, method, checks

- Data: the 121 sample records of `<TMP>/wnr3-data` (README there). Split A, as in the README:
  TRAIN = experts (37l+11)%192 and (37l+101)%192 (80 records), TEST = (53l+7)%192 (40) + L39 E120.
  Split B, a robustness check: TRAIN = (37l+11) and (53l+7), TEST = (37l+101) + L39 E120.
- **Additional data fetched (read only, HTTP range, into this directory):**
  - the 32-byte header of all 7,680 records (`headers.bin`, `fetch_headers.py`);
  - the 37 records other than L39 E120 whose rule-g anchors are not (120,120,120) (`extra/`,
    `fetch_shifted.py`, 586 MB).

  Why it was needed: 7,642 records have anchors (120,120,120). The 38 others carry every gain of
  anchor-relative scale coding, so the sample's single outlier could not stand for them. Those 37 are never
  used for training.
- `prep.py` takes each record through the following steps:
  1. reference decode with `libwex_nr3_tools.so`, giving Z(source);
  2. `wex_nr3_prepare(cz=1)`; flags and anchors equal the record header for 158/158 records;
  3. re-encode with the production model: **byte-identical to the stored record, 158/158**;
  4. training counts, plus the SHA-256 of Z(source).
- `mlib.c` is a prototype codec with WNR3's skeleton: 32-B header, u16 lengths, MurmurHash3 x0 = L + h23,
  4-B flush, 32-bit state, L = 2^23, byte renormalisation. It has switches for:
  - CDF precision P;
  - K selected weight sets, with ceil(log2 K) selector bits per weight stream packed after the length
    table and counted in the record size;
  - C selector-free classes;
  - scale symbol mode: absolute / anchor-relative / anchor-relative with a reduced alphabet and escape;
  - frequency-0 symbols.

  With P = 14, K = C = 1 and the WNM2 tables it **reproduces the stored records byte for byte**
  (`harness.py`). Its decoder checks every stream hash, the CZ rule and the exact end of each stream, then
  undoes the ceil form. **Every size in the table comes from a record that decoded to the SHA-256 of
  Z(source)**: 41/41 TEST records for every variant, plus 37/37 shifted records where shown, plus 41/41 on
  split B. `q_rule` (the WNR3 quantiser generalised to 2^P) reproduces `wex_nr3_build_model` exactly.
- The escape path is never taken by real data: no record has a scale outside the 10 TRAIN values.
  `esc_test.py` therefore checks it synthetically. In 2 records, 999 scales of non-max-5 blocks were set to
  anchor+40, anchor-30, 255 and 0. Encode and decode reproduce the modified slab exactly (`esc_test.log`).
- Baseline, recomputed: the 121-record sample is 0.85436 of raw (padded; 0.85426 unpadded). The 41 TEST
  records are 0.85525 unpadded. The whole container's entry table gives 0.85398. Real rANS start/flush is
  3.569 B per stream (3.565..3.575) on top of the ideal code length under the production model, which
  agrees with wnr4-record. Ideal and real deltas agree within about 10 B/record (for example P16: ideal
  -3,474 B, real -3,480 B).

## Results (real encodes, held-out, split A)

How to read the table:
- Delta = unpadded record bytes minus the stored production record's unpadded bytes.
- "normal" = the 40 TEST records with anchors 120. Of the 7,680 records, 7,642 are like them.
- "shifted" = the 38 records with other anchors (37 fetched + L39 E120). Not every variant was run on them.
- Container = 7,642 x normal mean + 38 x shifted mean + extra table bytes, extrapolated. Where shifted
  records were not run: 7,680 x normal mean.
- "vs P14" = paired difference against a single model retrained on TRAIN with today's rule, which isolates
  the structural change from the retraining.
- Source: `results_table.md` / `results.json` from `report.py`; per record in `out/real_*.json`.

| variant | normal: mean B/rec (% raw) | median | min..max | sem | vs P14 | L39 E120 | shifted mean (38) | container | extra container table B |
|---|---|---|---|---|---|---|---|---|---|
| **C16 + z8 + sesc** (P14) | **-11,317 (-0.060%)** | -7,858 | -87,424..-3,174 | 2,077 | -10,489 | -940,364 | **-174,790** | **-92.4 MB** | 763,980 |
| C8 x K2 selected | -8,508 (-0.045%) | -5,101 | -87,246..+517 | 2,171 | -7,680 | -331,483 | - | -64.6 MB* | 765,450 |
| C32 (23 classes) | -8,279 (-0.044%) | -4,810 | | 2,147 | -7,451 | -309,308 | - | -62.5 MB* | 1,122,660 |
| C16 | -7,927 (-0.042%) | -4,472 | -84,055..+219 | 2,077 | -7,099 | -277,133 | -116,329 | -64.2 MB | 765,450 |
| C8 | -7,343 (-0.039%) | -4,072 | | 2,000 | -6,514 | -247,267 | - | -56.0 MB* | 357,210 |
| *in-sample bound: own WNM2 per record (tables not counted)* | -7,071 (-0.038%) | -2,260 | | 3,170 | -6,243 | -1,385,126 | - | - | - |
| C8 x K4 selected | -6,928 (-0.037%) | -3,330 | | 2,255 | -6,100 | -400,773 | - | -51.6 MB* | 1,581,930 |
| K4 selected (2 bit/stream) | -6,574 (-0.035%) | -3,370 | -84,343..+2,120 | 2,136 | -5,746 | -306,649 | - | -50.3 MB* | 153,090 |
| K8 selected (3 bit/stream) | -5,717 (-0.030%) | -2,146 | | 2,228 | -4,888 | -422,889 | - | -43.5 MB* | 357,210 |
| K2 selected (1 bit/stream) | -5,168 (-0.028%) | -2,534 | | 1,970 | -4,339 | -220,880 | - | -39.6 MB* | 51,030 |
| sesc (P14) | -3,836 (-0.020%) | -2,324 | -49,225..-375 | 1,184 | -3,008 | -820,043 | -88,427 | -32.7 MB | -1,470 |
| P16 | -3,480 (-0.019%) | -2,081 | | 1,117 | -2,652 | -134,344 | - | -26.7 MB* | 0 |
| P16 + srel (full 256 relative) | -3,481 | -2,082 | | 1,117 | -2,652 | -815,992 | -83,122 | -29.8 MB | 0 |
| P15 | -2,621 (-0.014%) | -1,192 | | 1,146 | -1,792 | -146,226 | - | -20.1 MB* | 0 |
| z8 (P14) | -1,217 (-0.006%) | +294 | | 1,185 | -388 | -157,680 | - | -9.3 MB* | 0 |
| retrain, 80 TRAIN records, WNR3 rule (P14 / library) | -828 (-0.004%) | +692 | -46,241..+2,629 | 1,185 | 0 | -157,241 | -30,464 | -7.5 MB | 0 |
| retrain, 40 records (T40b) | -820 | +752 | | 1,275 | +8 | -219,802 | - | | 0 |
| retrain, 40 records (T40a) | +725 (+0.004%) | +772 | | 503 | +1,554 | +48,518 | - | | 0 |
| production model | 0 | 0 | | | +828 | 0 | 0 | 0 | 0 |

\* normal records only (7,680 x mean). Every variant was round-trip verified (41/41; 37/37 extra where shown).

Split B, the same procedure with the other split (`*_splitB`):
- C16 + z8 + sesc: **-13,363 B (-0.071%)**, median -9,498, sem 2,520;
- sesc: -3,580 (-0.019%);
- retrained single model: -558 (-0.003%), median +402.

The ranking and the size of the effects hold.

Note on the spread: the layer-39 TEST record L39 E154 gains 46-87 KB in every variant. It is 1 of 40,
and layer 39 is 1/40 of the container, so the mean is the right container estimator. The median is the
typical record.

## What each question gave

**(a) More training data: nothing.**
- Retraining on the 80 TRAIN records with today's rule gives -828 B mean. The median is **+692 B**: most
  typical records get slightly bigger.
- 40 records give +725 or -820 B depending on which 40.
- The production model (trained on 80 other experts) is as good as any retrain for typical records. Only
  the outliers are sensitive to the training set: L39 E120 moves from +49 KB to -220 KB depending on the
  40 records chosen.

**(b) Optimal frequency rounding: nothing.**
- The quantiser that exactly minimises the expected code length for the training counts (`q_opt`: greedy
  plus exchange, a global optimum for this separable concave problem) gives the same weight cost as the
  WNR3 rule to 6 decimals (3.575867 bit/w, ideal, TEST).
- On scales it is *worse* held-out: 1.0748 against 1.0690 bit/scale. The rule's "+1 then proportional"
  acts as smoothing for rare scale values.
- Only structural zeros help: z8 = -388 B (-0.002%).
- Bound only (`screen.py`, `screen.log`). The opt quantiser was not encoded, since there was nothing to
  encode for.

**(c) Higher CDF precision: noise level, and it conflicts with the GPU scale table.**
- Limits: the encoder's `((L >> P) << 8) * f` needs P <= 23. At P = 16 the decode product stays below 2^31,
  and real and ideal sizes agree (no state-size penalty).
- Gains: P15 -1.8 KB and P16 -2.7 KB against P14. Of P16's gain, about 2.3 KB is scales: the ~246 scale
  values never seen in training each hold 1/16384 of the scale table at P14. Weights gain only about
  0.35 KB.
- Decoder cost (read from the code, not measured):
  - The fast CPU decoder's AVX2 kernel compares u16 CDFs with a signed int16 compare (sentinel 0x7fff).
    That works up to P15; P16 needs an unsigned compare.
  - The CUDA scale kernel keeps a `[2^P]` u32 + u8 direct table per matrix in shared memory: 80 KB at P14,
    160 KB at P15. That exceeds the 99 KB per block of sm_86 (RTX 3070), so it would fall back to the
    global-memory path that `wex_cuda.cuh` keeps for sm < 80.
- The scale part of the gain is got more cheaply at P14 by sesc. **Recommendation: stay at 14 bits.**

**(d) K table sets with a per-stream selector chosen by the encoder: -0.03% at best, beaten by selector-free
classes.**
- Training: Lloyd iterations on the TRAIN streams (`ksets.py`), with the gain before selector logged in
  `ksets.log`. In training the sets save 0.039 / 0.048 / 0.069% of raw for K = 2 / 4 / 8.
- Cost: the selector takes 1 / 2 / 3 bits per weight stream (2,160 / 4,320 / 6,480 B per record).
- Held-out real gain against a retrained single model: -4.3 / -5.7 / -4.9 KB. K = 8 is already worse
  than K = 4.
- The encoder tries all K for every stream (K x the encode time).
- Extra container bytes: 57.8 KB per set.

**(e) Contexts the decoder has for free:**
- **Stream class from the decoded scales (C)**:
  - Feature: f = sum of srel over the stream's 64 blocks, a row-level scale level.
  - Classes: TRAIN quantiles, one 567-context set per class (`classsets.py` gives bounds,
    `ksets2.py` builds the tables).
  - Held-out real gain: C8 -6.5 KB, C16 -7.1 KB, C32 -7.5 KB, with no selector bits.
  - Adding an encoder-selected set inside each class adds at most -1.2 KB (C8 x K2), which is noise; C8 x K4
    is worse.
  - Other stream features (count of blocks off the anchor or above it) do slightly worse than the sum.
    The srel spread does nothing.
- **Anchor-relative scales**:
  - All 7,642 typical records have anchors (120,120,120), so for them a relative scale symbol is the same
    as an absolute one (P14_srel = P14).
  - The 38 shifted records are the largest scale payloads of the container: L39 E120 has 1.08 MB of
    scales against a median of 134 KB.
  - For those records the relative symbol cuts 85-88 KB on average and 817 KB on L39 E120.
  - The reduced alphabet (10 values + escape) adds the -3.0 KB on typical records described under (c).
- **Previous scale as a context** (prev ±1/±3/±7, relative): -0.002..-0.003 bit/scale, about -300 B
  (bound, `scale_ctx.log`). Noise, as in the WNR3 drafts.
- Weight-context features were not redone; mxfp4-contexts covers them.

**Ceiling of per-record adaptivity (in-sample bound).**
- Measured by coding each TEST record with a WNM2 model trained on that record itself (real encode,
  `lib_self`), without counting its tables.
- Typical records: -7.1 KB mean (-0.038%), median -2.3 KB.
- A per-record table set would cost far more than that to store: 3 x 567 x 15 frequencies, tens of KB.
- The held-out selector-free C16 classes already beat this in-sample per-record bound. Within-record
  variation (from stream to stream) matters more than variation from record to record.
- The exception is L39 E120: -1.39 MB in-sample, against -0.94 MB for C16 + z8 + sesc.

## Decoder cost of the recommended changes

| change | CPU (fast decoder) | GPU (CUDA kernels in `wex_cuda.cuh`) | measured? |
|---|---|---|---|
| sesc | scale LUT stays 2^14 entries; one add of the anchor per scale; an escape branch never taken on real data | scale tables shrink (11 live symbols); same structure | scalar prototype only |
| z8 | none (the CZ check becomes redundant: symbol 8 has no slots) | none | yes, no cost |
| C16 classes | per stream: sum 64 srel values before the stream starts (64 byte loads). Kernel 1 needs a table base per stream instead of per matrix. Tables 16 x 175 KB = 2.8 MB, against 256 KB L2 on the i7-7700K and 1 MB on the 7950X3D | each weight block (32 lanes = 32 streams of one matrix) keeps its matrix's 567 rows (18 KB) in shared memory. 16 classes = 290 KB, which does not fit, so it needs a class-sorted stream order (class known only after the scale kernel: a small device pass) or table reads through L1/L2 | **not measured** on AVX2 or GPU |
| K selected sets (not recommended) | as classes, plus selector unpacking | host knows the selectors, so it can sort streams per set before launch | not measured |

Timing that was measured (`timing.py`, `timing.json`):
- Setup: the scalar prototype decoder (binary search, u32 tables, one thread, -O2), interleaved A/B over
  7 rounds x 3 TEST records on the shared Xeon.
- Results: P14 3,569 ms/record median; C16 3,577 (x1.002); K4 3,611 (x1.012); P16 3,541 (x0.992); sesc
  3,510 (x0.984). Spread 2.3-3.8 s from machine load.
- So the per-symbol work does not change. This says nothing about the cache behaviour of the AVX2 kernel
  or about the GPU shared-memory layout. Those are the costs to measure before adopting C16.
- What transfers to the production machines: the size results transfer to any machine (format). The
  footprint concern is strongest on the i7-7700K (256 KB L2, AVX2) and weakest on the 7950X3D (1 MB L2,
  96 MB L3). C8 halves the footprint for about 0.6 KB/record less gain.

## Negative results

- More training data (80 vs 40 vs production) is noise: -0.004%, with the median record *bigger*.
- Optimal frequency rounding gains nothing on weights and is worse on scales held-out.
- 15/16-bit CDFs give -0.010 / -0.014%, mostly scale-table waste that sesc removes at 14 bits; it also
  breaks the GPU scale table.
- Encoder-selected table sets (K = 2/4/8) give -0.023..-0.031% after selector cost and are beaten by
  selector-free classes. Selection on top of classes is noise.
- A previous-scale context for scales is about -0.002% (bound).
- Per-record models: the in-sample ceiling before table cost is -0.038%. The table cost makes it a loss.

## Recommendation for the model of a WNR4

1. Model changes alone do not justify a WNR4. The best combination is about 0.06-0.07% of raw on typical
   records, about 92 MB on the container (extrapolated).
2. If WNR4 is built for other reasons, its model ("WNM3") should have:
   - the WNR3 weight contexts (567 per matrix; or 63, per mxfp4-contexts, which costs nothing);
   - **16 (or 8) selector-free stream classes** with thresholds stored in the blob;
   - **frequency 0 for symbol 8**;
   - an **anchor-relative scale alphabet** with an escape;
   - 14-bit CDFs and the WNR3 quantisation rule, trained on about 80 records as today.

   Expected: -11.3 / -13.4 KB/record on the two splits, -175 KB on the 38 anchor-shifted records. If the
   GPU work for classes is not wanted, sesc + z8 alone is free in the decoder:
   - sesc measured: -3.8 KB/record typical against production (-3.0 KB against the retrain), -88 KB on
     shifted records, -32.7 MB on the container;
   - z8 adds about -0.4 KB more. This is estimated as additive; the pair was not encoded alone.
3. Do not spend effort on precision, quantiser, training-set size, encoder-selected sets or per-layer or
   per-record models.

## Not checked

- No AVX2 or GPU implementation of classes or sesc; their decode cost comes from reading the code.
- Only one Lloyd start per K; the class thresholds are TRAIN quantiles and were not optimised.
- Classes were tried only with the 567-context model at 14 bits; not combined with the 63- or 7-context
  models of mxfp4-contexts.
- No corruption audit of the prototype records. They keep WNR3's 23-bit hash per stream, and a wrong class
  or escape changes the decoded bytes, which the hash covers.
- Typical-record figures rest on 40 + 40 held-out records (two splits) and the shifted-record figures on
  38 records. Container figures are extrapolations.

## Files

- Data preparation and checks:
  - `prep.py`: decode, coded slab, counts, re-encode check;
  - `recount.py`: relative scale counts, count cross-check;
  - `harness.py`: prototype == stored records, rule == library.
- `fetch_headers.py`, `fetch_shifted.py` and `headers.bin`: the extra data.
- `mlib.c` / `mlib.so`: the prototype codec. `common.py`: quantisers, models, ctypes glue.
- Bound screening: `screen.py`, `scale_ctx.py`, `scale_alpha.py`, `classsets.py`.
- Table training: `ksets.py` (selected sets) and `ksets2.py` (classes x selected sets).
- `real.py`: real encode + decode + SHA-256 check, one JSON per variant in `out/`.
- `report.py`: writes `results_table.md` / `results.json`.
- `esc_test.py`, `timing.py`.
- `run_all.sh`: everything in order. `coded/` (2.8 GB) holds the coded slabs; it can be regenerated by
  `prep.py`.
