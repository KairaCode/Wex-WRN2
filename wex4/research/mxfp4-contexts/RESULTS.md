# MXFP4 weight contexts: can a cheap static model beat WNR3-CZ? (2026-11-11)

**Answer: no.** No static-table context model within the limits (16 symbols, <= ~20,000 contexts per matrix,
independent streams) gains more than **0.05% of raw** held-out. The best combination of everything that helped
gets -0.048..-0.051% of raw (about -9.3 KB per 18.8 MB record).

**What the data says instead:** in the coded domain (canonical zero + ceil form) the nibbles are, to within
0.015% of raw, independent draws given the block's `srel`. Today's 567 contexts can shrink to 63
(`srel` x running max: +0.0003% of raw) or to 7 (`srel` only: +0.015%), and the sign is an exact coin
(1.00001 bit per non-zero weight). That does not save bytes, but it removes the per-symbol context
dependency, which is a decode-speed lever: on one CPU thread a 7-context slot-table decoder ran 3.5-4.5x
faster than the reference algorithm at +0.02% of raw (real rANS bytes).

All numbers below are measured on this PC, CPU only. Raw data: `results.json`.

## Data and method

- 24 real experts of the production container (V4.1 REAP-192, WEX3, codec 5 = WNR3-CZ): layers
  0, 5, 11, 16, 22, 28, 33, 39, three experts each (seeded sample). 0.39 GiB read.
- Each record is decoded with the reference decoder (`wex_nr3_decode`) and brought back to the coded domain
  with `wex_nr3_prepare(cz=1)`; the flags and rule-g anchors match the record header for all 24.
- Cost = sum of -log2(f / 16384) with CDFs quantised by the `wex_nr2_quantize` rule, in bits per weight.
  35,389,440 weights and 17,280 weight streams per expert. The rANS flush is a constant added to every
  variant alike, so it is left out of the deltas.
- Held-out splits. A: train on experts 0,1 of each layer (16), test on expert 2 (8). B: train on 1,2, test on
  0. C (harder, no layer shared): train on 5 layers (15 experts), test on layers 5, 22, 39 (9 experts).
- Delta = (variant - baseline) / 4.25 bit, in % of raw. 0.01% of raw = 1,880 B per record = 14.4 MB over
  the 7,680 records (the container-level figure is an extrapolation from the test experts).

## Baseline check

| Check | Result |
|---|---|
| Context walk vs `wex_nr3_count_expert`, 24 experts | identical counts |
| Quantiser vs `wex_nr3_build_model` | identical CDFs |
| Mean record of the 24 | 16,067,755 B = 0.85464 of raw |
| Real weight payload, 24 records | 3.59204 bit/w |
| Ideal cost under the production model | 3.57809 bit/w |
| Difference = rANS flush | 3.570 B per stream (3.568-3.579 per record) = 0.01395 bit/w = 0.328% of raw |
| Ideal + 3.5 B x 17,280 streams vs real, per record | max error 0.0085% |
| Own model (16 experts) held-out + flush vs real, same test experts | A 3.58974 vs 3.58972 (+0.0006%), B 3.59081 vs 3.59074 (+0.002%), C 3.59681 vs 3.59585 (+0.027%) |
| Order-0 entropy of the coded nibbles | 3.6627 bit/w |

Baseline held-out (ideal, no flush): **A 3.57579, B 3.57686, C 3.58286 bit/w**. The global model costs
3.5695-3.5874 bit/w depending on the layer (layers 16 and 39 are the expensive ones).

## Results, held-out, 14-bit CDFs

Model bytes = 3 matrices x contexts x 34 B (today: 57,834 B of weight CDFs). Negative = smaller.

| # | Variant | Contexts / matrix | Model bytes | A bit/w | B bit/w | Delta A | Delta B | Delta C | Decode cost class |
|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline (srel, previous magnitude, running max) | 567 | 57,834 | 3.57579 | 3.57686 | 0 | 0 | 0 | today |
| 1 | + nibble above, 9 values | 5,103 | 520,506 | 3.57587 | 3.57696 | +0.0019 | +0.0022 | +0.0076 | **cross-stream** (forces a stream order) |
| 1 | + nibble above, 3 classes | 1,701 | 173,502 | 3.57581 | 3.57689 | +0.0004 | +0.0006 | +0.0024 | **cross-stream** |
| 1 | + sign of the nibble above | 1,701 | 173,502 | 3.57581 | 3.57688 | +0.0004 | +0.0004 | +0.0015 | **cross-stream** |
| 2 | + magnitude two back | 5,103 | 520,506 | 3.57583 | 3.57691 | +0.0008 | +0.0010 | +0.0058 | extra in-stream state |
| 2 | + sum of last three magnitudes | 4,536 | 462,672 | 3.57581 | 3.57688 | +0.0004 | +0.0003 | +0.0037 | extra in-stream state |
| 3 | + position class 0 / 1-7 / 8-31 | 1,701 | 173,502 | 3.57568 | 3.57675 | -0.0027 | -0.0027 | -0.0006 | extra in-stream state |
| 3 | + exact position (32) | 18,144 | 1,850,688 | 3.57517 | 3.57626 | -0.0145 | -0.0142 | +0.0070 | extra in-stream state, 1.85 MB table |
| 4 | + sign of previous nibble | 1,701 | 173,502 | 3.57580 | 3.57688 | +0.0002 | +0.0003 | +0.0010 | extra in-stream state |
| 4 | sign plane: 8 magnitudes + raw sign bit | 567 | 30,618 | 3.57569 | 3.57676 | -0.0023 | -0.0025 | -0.0017 | 8-symbol search + 1 raw bit per non-zero |
| 4 | sign plane, sign modelled (context x magnitude) | 567 | 54,432 | 3.57570 | 3.57677 | -0.0022 | -0.0023 | +0.0001 | extra binary symbol |
| 4 | sign modelled given previous sign / sign above | 567 | 102,060 | 3.57571 | 3.57677 | -0.0020 | -0.0022 | +0.0008 | extra binary symbol (above: cross-stream) |
| 5 | + scale of next block (delta, 5 values) | 2,835 | 289,170 | 3.57567 | 3.57668 | -0.0028 | -0.0043 | -0.0038 | per-block base from decoded scales |
| 5 | + scale of previous block | 2,835 | 289,170 | 3.57567 | 3.57668 | -0.0028 | -0.0042 | -0.0028 | per-block base |
| 5 | + scale of block above / below | 2,835 | 289,170 | 3.57575 | 3.57691 | -0.0009 | +0.0010 | +0.0072 / +0.0098 | per-block base |
| 5 | + scale minus mean of 4 neighbours | 2,835 | 289,170 | 3.57566 | 3.57677 | -0.0031 | -0.0022 | -0.0043 | per-block base |
| 5 | + srel against the row's own anchor | 3,969 | 404,838 | 3.57549 | 3.57629 | -0.0070 | -0.0134 | -0.0405 | per-block base + anchor per row |
| 5 | + srel against the column's anchor | 3,969 | 404,838 | 3.57576 | 3.57685 | -0.0007 | -0.0004 | +0.0367 | per-block base + anchor per column |
| 5 | srel clamp +-5 instead of +-3 | 891 | 90,882 | 3.57579 | 3.57687 | 0.0000 | 0.0000 | +0.0002 | today |
| 6 | zeros so far instead of running max | 504 | 51,408 | 3.57640 | 3.57752 | +0.0144 | +0.0155 | +0.0030 | today |
| 6 | zeros so far (8) and running max | 4,536 | 462,672 | 3.57562 | 3.57668 | -0.0041 | -0.0043 | +0.0020 | extra in-stream state |
| 6 | count of magnitudes >= 4 (4) and running max | 2,268 | 231,336 | 3.57565 | 3.57672 | -0.0033 | -0.0034 | -0.0029 | extra in-stream state |
| 7 | two-pass, side = block max (3 values, 0.033 bit/w) | 1,701 | 173,502 | 3.58002 | 3.58103 | +0.0996 | +0.0981 | +0.1067 | two-pass, extra symbol stream |
| 7 | two-pass, side = exact zero count (0.089 bit/w), remaining zeros tracked | 9,072 | 925,344 | 3.57821 | 3.57930 | +0.0569 | +0.0573 | +0.0713 | two-pass |
| 7 | two-pass, side = zero-count bucket (0.074 bit/w), not tracked | 3,402 | 347,004 | 3.62635 | 3.62734 | +1.1896 | +1.1878 | +1.1991 | two-pass |
| 8 | K=2 table sets, best per stream (1 bit) | 567 x 2 | 115,668 | 3.57437 | 3.57584 | -0.0335 | -0.0242 | not run | today + set index per stream |
| 8 | K=3 (1.58 bit) | 567 x 3 | 173,502 | 3.57403 | 3.57532 | -0.0415 | -0.0363 | not run | same |
| 8 | K=4 (2 bit) | 567 x 4 | 231,336 | 3.57391 | 3.57523 | -0.0443 | -0.0386 | not run | same |
| 9 | one model per layer | 567 | 2,376,320 for 40 layers | 3.57496 | 3.57711 | -0.0196 | +0.0058 | n/a | today + model switch per layer |
| 9 | one model per 2 adjacent sampled layers | 567 | 1,188,160 if 20 groups | 3.57529 | 3.57677 | -0.0117 | -0.0023 | n/a | same |
| 9 | one model per 4 sampled layers | 567 | 594,080 if 10 groups | 3.57546 | 3.57698 | -0.0077 | +0.0028 | n/a | same |
| 10 | + max class of the previous block in the stream | 2,268 | 231,336 | 3.57578 | 3.57686 | -0.0002 | 0.0000 | +0.0006 | extra in-stream state |
| 10 | previous magnitude not reset at block starts | 567 | 57,834 | 3.57580 | 3.57687 | +0.0001 | +0.0001 | 0.0000 | today |
| 10 | + row level (row anchor - matrix anchor, 5 values) | 2,835 | 289,170 | 3.57549 | 3.57629 | -0.0070 | -0.0134 | -0.0403 | per-block base + anchor per row |
| 10 | + row level + position class | 8,505 | 867,510 | 3.57539 | 3.57619 | -0.0095 | -0.0160 | -0.0295 | same + extra in-stream state |
| 10 | row level + position class, previous magnitude dropped | 945 | 96,390 | 3.57536 | 3.57615 | -0.0101 | -0.0168 | -0.0494 | same |
| **best** | the 945-context model with K=2 per stream | 945 x 2 | 192,780 | 3.57402 | 3.57530 | -0.0416 | -0.0367 | not run | row anchors + set index per stream |
| **best** | the 945-context model with K=4 per stream | 945 x 4 | 385,560 | 3.57363 | 3.57481 | **-0.0508** | **-0.0482** | not run | same |

Over the context limit, shown only as a bound: row anchor + exact position (127,008 contexts) -0.017 / -0.024
on A / B and **+0.28** on C; four features together (238,140 contexts) -0.009 / -0.016 and +0.09 on C. Large
tables do not transfer to unseen layers.

### Ablation: what today's three features are worth

| Contexts kept | Contexts / matrix | A bit/w | B bit/w | Delta A | Delta B | Delta C |
|---|---|---|---|---|---|---|
| srel, previous magnitude, running max (today) | 567 | 3.57579 | 3.57686 | 0 | 0 | 0 |
| srel, running max | 63 | 3.57580 | 3.57689 | +0.0002 | +0.0005 | -0.0015 |
| srel, running max class (none / <5 / 5 / >=6) | 28 | 3.57591 | 3.57701 | +0.0028 | +0.0035 | +0.0014 |
| srel, previous magnitude | 63 | 3.57639 | 3.57752 | +0.0140 | +0.0153 | +0.0007 |
| **srel only** | **7** | 3.57641 | 3.57755 | **+0.0146** | **+0.0161** | -0.0012 |
| srel only, K=4 table sets per stream | 7 x 4 | 3.57482 | 3.57612 | -0.0228 | -0.0174 | not run |
| previous magnitude, running max (no srel) | 81 | 3.62584 | 3.62737 | +1.1776 | +1.1884 | +1.0506 |
| running max only | 9 | 3.62889 | 3.63049 | +1.2493 | +1.2617 | +1.1141 |

`srel` carries the whole model. The previous magnitude adds nothing once the running max is there.

### CDF precision (table-driven decoders want small slot tables)

Held-out bit/w, split A / B. "Joint" = today's 16-symbol alphabet; "mag + sign" = 8 magnitudes plus one raw
bit per non-zero weight.

| Model | 14 bit | 12 bit | 10 bit |
|---|---|---|---|
| today's 567 contexts, joint | 3.57579 / 3.57686 | 3.57614 / 3.57726 | 3.57811 / 3.57954 |
| 63 contexts (srel, running max), mag + sign | 3.57571 / 3.57678 | 3.57574 / 3.57684 | 3.57600 / 3.57718 |
| 7 contexts (srel), mag + sign | 3.57631 / 3.57744 | 3.57633 / 3.57749 | 3.57661 / 3.57782 |

So 7 contexts x 8 magnitudes at 12 bit costs +0.013 / +0.015% of raw against today, and at 10 bit
+0.019 / +0.023%. The 63-context magnitude model at 12 bit equals today's cost (-0.001 / -0.0005%).

## Decode cost on the CPU (`decbench.c`)

One thread, AMD desktop (family 25, model 97), machine not idle, best of 5 runs, two experts (L16 E066 and
L39 E018). Models trained on the slab itself. Every decoder reproduces the coded weights exactly. The stream
hash check of WNR3 is not included in any of them.

| Decoder | ns / weight | ms / record | Real coded bytes (L16 E066) |
|---|---|---|---|
| A: today's model, binary search (reference algorithm) | 13.3-15.6 | 470-551 | 15,898,628 |
| A: same streams, AVX2 16-way compare instead of the search | 6.1-6.7 | 216-237 | same |
| F: 7 contexts, 8 magnitudes, 12-bit slot table, raw sign bits, byte renormalisation | 2.95-3.78 | 104-134 | 15,910,121 (+0.061% of raw) |
| Fw: F with 16-bit renormalisation and L = 2^16 | 3.07-4.00 | 108-142 | 15,901,579 (+0.016% of raw) |

- On the second expert the size deltas are +0.069% (F) and +0.024% (Fw).
- Fw is 8.5 KB per record smaller than F only because its flush is 0.49 B per stream shorter; that saving
  (0.045% of raw) is independent of the context model and as large as the best context gain above.
- F and Fw include the cost of byte-aligning each stream's sign bits (expected 3.5 bits per stream, about
  7.6 KB per record; not measured separately).
- Decoding 4 streams interleaved in one scalar loop gave no consistent gain in this plain C (0.65-1.1x).
- The AVX2 search needs no format change: 2.1-2.3x over the reference algorithm on today's records.

## Ranking

1. Table-set choice per stream (K=2..4): -0.024..-0.044% of raw. The only item above 0.02%.
2. Row level from the decoded scales: -0.007..-0.013% (A, B), -0.040% across layers (C).
3. Exact position in block: -0.014% with a 1.85 MB table, lost across layers (C +0.007%).
4. Per-layer models: -0.020% on one split, +0.006% on the other: noise.
5. Zero count next to the running max, next/previous block scale, position class: -0.003..-0.004% each.
6. Nibble above, two magnitudes back, previous sign, sign plane with any context, srel +-5, previous block
   max, no block reset: within +-0.003%. The sign costs 1.00001-1.00002 bit per non-zero weight.
7. Two-pass side symbols: worse by 0.06-1.2%. Zero count instead of the running max: worse by 0.015%.

## Recommendation

- **Do not build a new codec for a richer context model.** The combined best is -0.05% of raw: about
  9.3 KB per record, about 72 MB on the 123.3 GB container (extrapolated). It needs 4 table sets, a selector
  per stream and a row-anchor pass, and the rANS flush alone (0.33% of raw) is six times larger.
- **If WEX4 gets a new weight codec, make it simpler, not richer:**
  - smallest step: drop the previous magnitude (63 contexts, same bytes);
  - speed step: `srel`-only context (fixed for the 32 symbols of a block, known before the block), 8-symbol
    magnitude alphabet, raw sign bit, 12-bit CDFs. Cost +0.013..+0.015% of raw in the model, +0.016..+0.024%
    in real rANS bytes with the shorter flush, 3.5-4.5x faster than the reference algorithm on one CPU thread.
    The weight CDFs shrink from 57,834 B to 378 B (3 x 7 x 9 u16).
  - optional: K=4 `srel`-only table sets per stream recover the cost and a little more (-0.017..-0.023% vs
    today) at the same speed class, with 2 bits per stream.
- Without any format change, the AVX2 compare search halves the CPU decode time of today's records.

## Not checked

- Only 24 of 7,680 experts (8 of 40 layers); 8 or 9 held-out experts per split. No MTP experts.
- Context variants are ideal costs with quantised CDFs, not encoded streams (the baseline shows ideal + flush
  = real within 0.009%). Only models A, F, Fw were actually encoded and decoded, on two experts, with models
  trained on the same slab.
- No GPU run (not allowed here): the effect of a per-block context on the GPU decoder is not measured.
- CPU timings: one desktop, one thread, not idle; the laptop was not measured; the decoders are plain C
  without the stream hash, ceil-form inverse or record parsing.
- The K-set search is one Lloyd run from one starting split; a better search could move it by a few 0.001%.
- Model F as a format (where the sign bits sit, how the stream hash covers them) is not designed or verified.
- `run_all.sh` lists the commands that were run one by one; the script itself was only syntax-checked.

## Files

- `extract.py`: 24 production records -> coded slabs under `out/` (container opened read-only).
- `ctxlib.c`: context walk for 34 features (counts, per-stream costs, per-stream table-set counts, side symbols).
- `ctxeval.py`: stages `validate`, `contexts`, `combo`, `kmix`, `layers`, `precision`, `decbench` -> `results.json`.
- `decbench.c`: real rANS encode + decode timing of models A, F, Fw.
- `run_all.sh`: the full sequence. `out/` holds build products and data copies and is not in git.
