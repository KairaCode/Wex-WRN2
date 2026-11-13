# Non-expert tensors: how compressible, and with which static model

Scope: the tensors of the production model (DeepSeek V4.1 Flash REAP-192 MXFP4,
48 safetensors shards) that are **not** routed-expert weights or their scales.
CPU only, numpy only, nothing written outside this folder.

| dtype | bytes in the model | tensors | role |
|---|---|---|---|
| F8_E4M3 | 7,086,800,896 | 353 | attention projections, shared experts, MTP; every one has a `.scale` |
| BF16 | 3,874,076,672 | 516 | embed and head (68 % of the bytes), vision tower, aligner, gates, norms |
| F32 | 169,167,688 | 387 | `hc_*_fn` tables [24, 20480] (99.9 %), tiny biases |
| F8_E8M0 | 6,920,704 | 353 | the scales of the F8_E4M3 tensors, one per 32x32 block |

## Answer in five lines

1. **WBP1 as drafted (byte planes, order-0, per-record table)**: F8_E4M3 0.820 of raw,
   BF16 0.672, F32 0.845. Extrapolated to the three dtypes: 11,130,045,256 B ->
   about 8,556 MB (0.769), i.e. about 2,574 MB saved.
2. **No richer static model is worth more than about 0.5 to 1 point.** Best realisable
   gains over WBP1, whole dtype, extrapolated: F8_E4M3 -35 MB (0.8151), BF16 -38 MB
   (0.6618), F32 -0.4 MB. Together about 74 MB on 11.1 GB.
3. **BF16 should be split by field, not by byte.** The exponent's low bit sits in the
   low byte, so byte planes leave 0.15 bit/element on the table. Coding only the
   8-bit exponent and storing sign + 7 mantissa bits raw gives 0.6649 (better than
   WBP1's 0.6717) with **one** rANS symbol per element instead of two.
4. **Order-1 / previous-symbol contexts do not pay** on any dtype (gain <= 0.02 bit per
   element, tables of 8 to 130 KB, and they add a serial dependency to the decoder).
   Per-row and per-column CDFs lose badly to their table cost.
5. **Per-record tables are right for every tensor >= 1 MiB** (a held-out shared table
   is 0.1 to 0.2 point worse and saves 514 B). Tensors < 1 MiB total 4.8 MB; nothing
   done to them matters.

## Method

`measure.py` reads a fixed sample (1,278,339,912 B per run, cap enforced in the script):
whole tensors from early / middle / late layers of every large kind, five 32 MB row
slices of `embed` and of `head`, one slice of two other large tensors, and every tensor
below 1 MiB. Costs per model:

- **ideal bits/el**: plain (conditional) entropy, no overhead, in-sample.
- **coded bits/el**: bits against the CDF quantised to 14 bits (present symbols >= 1),
  plus 6 B per 4096 symbols per coded stream, plus raw side information.
- **table**: 2 B per CDF entry, K + 1 entries per context that occurs, per record.
- **ratio** = (payload + table) / raw. A stream that does not shrink is stored raw.
- **break-even elements**: record size at which the model's extra table is repaid by
  its payload gain over WBP1 (sample average); `0` = cheaper at any size, `never` =
  no payload gain.
- **ratio (whole dtype, extrapolated)**: each coarse kind (digits wildcarded) weighted
  by its real size in the model; kinds not sampled get the sample mean. This is an
  extrapolation over layers, not a measurement.
- **decode class**: `order-0` = one table lookup per symbol; `context (side)` = the
  table is selected by side information or by an already decoded plane, streams stay
  independent; `context (serial)` = the table is selected by the previous symbol of
  the same stream.
- **rANS symbols/el**: entropy-decoded symbols per element (raw planes do not count).

All per-record models are in-sample by construction and pay their table. The
shared-table column further down is held-out (leave-one-out over the tensors of a kind).

## F8_E4M3 >=1MiB tensors (30 records, 533,200,896 B sampled, raw = 8 bits/element)

| model | ideal bits/el | coded bits/el | ratio (sample) | ratio (whole dtype, extrapolated) | table B/record | break-even elements | rANS symbols/el | decode class |
|---|---|---|---|---|---|---|---|---|
| o0_byte (WBP1) | 6.5477 | 6.5622 | 0.82030 | 0.82001 | 514 | - | 1 | order-0 |
| sign_raw + mag7_o0 | 6.5477 | 6.5602 | 0.82004 | 0.81975 | 258 | 0 | 1 | order-0 |
| exp4_o0 + sign,mant3 raw | 6.6095 | 6.6213 | 0.82766 | 0.82739 | 34 | never | 1 | order-0 |
| sign_raw + exp4_o0 + mant3&#124;exp | 6.5477 | 6.5712 | 0.82142 | 0.82112 | 322 | never | 2 | context (side) |
| o1 prev byte in row | 6.5395 | 6.5537 | 0.82656 | 0.82579 | 130,556 | 122,985,324 | 1 | context (serial) |
| ctx prev exp in row (16) | 6.5420 | 6.5565 | 0.82003 | 0.81976 | 8,224 | 10,891,597 | 1 | context (serial) |
| ctx exp above (16) | 6.5429 | 6.5574 | 0.82013 | 0.81985 | 8,224 | 12,841,552 | 1 | context (serial) |
| ctx prev exp x exp above (256) | 6.5360 | 6.5504 | 0.82605 | 0.82532 | 128,843 | 87,103,417 | 1 | context (serial) |
| per-row CDF | 6.4739 | 6.9867 | 0.93529 | 0.93636 | 1,100,919 | never | 0.678 | context (side) |
| per-column CDF | 6.4988 | 6.8985 | 0.91069 | 0.90657 | 859,682 | never | 0.747 | context (side) |
| ctx row class x column class (8x8) | 6.5226 | 6.5371 | 0.81899 | 0.81871 | 32,896 | 10,327,486 | 1 | context (side) |
| ctx block scale (E8M0 value) | 6.5177 | 6.5323 | 0.81665 | 0.81640 | 2,039 | 408,161 | 1 | context (side) |
| ctx block max exponent (16) | 6.5399 | 6.5543 | 0.81935 | 0.81908 | 1,028 | 523,260 | 1 | context (side) |
| ctx block class (16, by mean magnitude) | 6.5052 | 6.5199 | 0.81545 | 0.81527 | 8,224 | 1,459,626 | 1 | context (side) |
| ctx block class (64, by mean magnitude) | 6.5039 | 6.5185 | 0.81666 | 0.81638 | 32,896 | 5,925,703 | 1 | context (side) |
| ctx block class x block scale | 6.5019 | 6.5166 | 0.81532 | 0.81509 | 13,124 | 2,214,991 | 1 | context (side) |

Whole-dtype extrapolation covers 95.0 % of the F8_E4M3 bytes by sampled kinds (the
MTP copies of `wo_a`, `wo_b`, `wq_b` were not read). Per-record WBP1 ratio ranges
from 0.8117 to 0.8389 over the 30 records.

Reading the table:

- Splitting into sign / exponent / mantissa cannot beat the byte model (same
  information, smaller tables): `sign raw + mag7` is equal within 0.03 point with a
  128-symbol table; `exp4 + raw sign and mantissa` costs +0.74 point (+52 MB on the
  dtype) and buys a 16-symbol alphabet, the same size as the expert nibble coder.
- Sign is incompressible (1.000 bit). `-0` exists (2,631 codes in 533 M elements, as
  many as `+0`), no NaN codes: canonical zero would gain nothing here.
- The useful contexts are per 32x32 block, all side-information contexts.

### Scales (question 3)

- Every F8_E4M3 tensor has an F8_E8M0 `.scale` (353 and 353; 14 of 14 kinds; all 30
  sampled records), one scale per 32x32 block of the weight matrix.
- The scales are nearly constant: 2 to 8 distinct values per record, codes 114..121;
  order-0 entropy 0.68 bit per scale, 0.51 bit given the scale to the left. The
  6.9 MB of scales would shrink to roughly 0.6 MB of payload plus 0.2 MB of
  per-record tables (extrapolated from 520,704 blocks).
- Conditioning the weight byte on its block's scale helps a little: 0.8164 against
  0.8200 (-25.6 MB on the dtype, extrapolated), table about 2 KB per record, no side
  information because the scale is already there. Break-even about 0.4 M elements.
- A 4-bit block class sent as side information (16 quantiles of the block's mean
  magnitude) does slightly better alone (0.8153) and combined with the scale reaches
  0.8151 (-34.9 MB). 64 classes are worse than 16 once the table is paid.
- The block maximum exponent is almost determined (0.47 bit per block) and is a weak
  context (0.8191).

## BF16 >=1MiB tensors (37 records, 660,156,416 B sampled, raw = 16 bits/element)

| model | ideal bits/el | coded bits/el | ratio (sample) | ratio (whole dtype, extrapolated) | table B/record | break-even elements | rANS symbols/el | decode class |
|---|---|---|---|---|---|---|---|---|
| o0 byte planes (WBP1) | 10.6995 | 10.7279 | 0.67055 | 0.67165 | 1,028 | - | 2 | order-0 |
| bound: joint 16-bit order-0 | 10.5414 | 10.5414 | 0.65982 | 0.66067 | 17,545 | 708,410 | 0 | bound only |
| hi&#124;prev hi in row + lo o0 | 10.6784 | 10.7065 | 0.66998 | 0.67131 | 14,670 | 5,094,624 | 2 | context (serial) |
| hi o0 + lo&#124;hi low 2 bits | 10.5653 | 10.5934 | 0.66223 | 0.66276 | 2,570 | 91,746 | 2 | context (side) |
| hi o0 + lo&#124;hi low 4 bits | 10.5416 | 10.5697 | 0.66104 | 0.66175 | 7,613 | 333,075 | 2 | context (side) |
| hi o0 + lo&#124;hi (256 ctx) | 10.5414 | 10.5698 | 0.66138 | 0.66218 | 13,822 | 647,253 | 1.998 | context (side) |
| signexp9 o0 + mant7 o0 | 10.5869 | 10.6129 | 0.66338 | 0.66403 | 1,284 | 17,808 | 2 | order-0 |
| signexp9 o0 + mant7&#124;exp low 3 bits | 10.5491 | 10.5751 | 0.66112 | 0.66181 | 3,090 | 107,967 | 2 | context (side) |
| sign raw + exp8 o0 + mant7 o0 | 10.5870 | 10.6121 | 0.66330 | 0.66394 | 772 | 0 | 2 | order-0 |
| exp8 o0 + sign,mant7 raw | 10.6149 | 10.6275 | 0.66425 | 0.66489 | 514 | 0 | 1 | order-0 |
| signexp9&#124;prev signexp9 + mant7 o0 | 10.5615 | 10.5866 | 0.66463 | 0.66599 | 52,889 | 2,937,042 | 2 | context (serial) |
| per-row CDF (both planes) | 10.4901 | 11.3610 | 0.76992 | 0.76255 | 1,068,036 | never | 0.872 | context (side) |
| per-column CDF (both planes) | 10.5698 | 10.9517 | 0.75012 | 0.76345 | 1,171,086 | never | 0.951 | context (side) |
| signexp9&#124;row class x column class (8x8) + mant7 o0 | 10.5164 | 10.5411 | 0.66251 | 0.66338 | 65,922 | 2,779,259 | 2 | context (side) |

Field entropies, element-weighted: sign 1.000 bit, exponent 2.615 bits, mantissa
6.972 of 7 bits. The joint 16-bit order-0 entropy (10.541 bits, 0.6607 extrapolated)
is the floor for any memoryless model; `hi + lo | hi low 4 bits` is within 0.11 point
of it. Per-record WBP1 ratio ranges from 0.6646 to 0.6914; the worst record is the
last slice of `embed` (rows 126155..129280), where the row-class x column-class
context recovers 2.8 points - rows there have very different magnitudes.

`embed` and `head` are 68 % of the BF16 bytes and were sampled with five 32 MB slices
each (12 % of each tensor). They are not tied: the first 4 MiB differ.

## F32 >=1MiB tensors (28 records, 55,050,240 B sampled, raw = 32 bits/element)

| model | ideal bits/el | coded bits/el | ratio (sample) | ratio (whole dtype, extrapolated) | table B/record | break-even elements | rANS symbols/el | decode class |
|---|---|---|---|---|---|---|---|---|
| o0 byte planes (WBP1) | 26.9853 | 27.0138 | 0.84470 | 0.84495 | 1,028 | - | 2 | order-0 |
| signexp9 o0 + mant 7/8/8 o0 | 26.9109 | 26.9366 | 0.84242 | 0.84265 | 1,284 | 26,534 | 2 | order-0 |
| signexp9 o0 + mant23 raw | 26.9406 | 26.9531 | 0.84281 | 0.84304 | 1,026 | 0 | 1 | order-0 |
| b3 o0 + b2&#124;b3 low 4 bits + b1 o0 + b0 o0 | 26.8806 | 26.9563 | 0.84450 | 0.84476 | 4,167 | 436,883 | 1.536 | context (side) |
| b3&#124;prev b3 in row + b2,b1,b0 o0 | 26.8769 | 26.9052 | 0.84869 | 0.84856 | 15,548 | 1,069,134 | 2 | context (serial) |
| b3,b2 o0 + b1&#124;b2 high 4 bits + b0 o0 | 26.9797 | 27.0138 | 0.84470 | 0.84495 | 1,028 | 0 | 2 | context (side) |
| per-row CDF (4 planes) | 26.8099 | 26.8781 | 0.84621 | 0.84603 | 12,336 | 666,437 | 1 | context (side) |

Only the top byte compresses (plane entropies b0..b3: 8.000, 8.000, 7.971, 3.015
bits). WBP1's raw fallback already stores b0 and b1 raw, so it decodes two symbols
per element; `signexp9 + raw 23-bit mantissa` is 0.2 point better with one symbol per
element and a 512-symbol table. The values are genuine float32: at most 0.014 % are
exactly representable in 16 bits, at least 99.7 % of the values in a record are
distinct, and neighbouring layers are not duplicates.

## Tensors below 1 MiB, and shared tables (held-out)

| group | records | raw B | per-record table | one record per kind | shared table, held-out |
|---|---|---|---|---|---|
| BF16 < 1 MiB (17 kinds with >= 2 tensors) | 322 | 4,599,296 | 0.6347 | 0.6367 | 0.6390 |
| F32 < 1 MiB (14 kinds) | 301 | 84,808 | 1.0000 | 0.7820 | 0.7540 |
| BF16 >= 1 MiB (8 kinds, whole tensors) | 23 | 254,672,896 | 0.6682 | 0.6684 | 0.6701 |
| F32 >= 1 MiB (4 kinds) | 28 | 55,050,240 | 0.8447 | 0.8467 | 0.8490 |
| F8_E4M3 >= 1 MiB (9 kinds) | 26 | 496,500,736 | 0.8200 | 0.8204 | 0.8209 |

Order-0 byte planes in all three columns; "shared table, held-out" trains the table on
the other tensors of the same kind and charges no table. For tensors >= 1 MiB the
per-record table always wins. For the small ones the picture is mixed per kind (norm
weights of 10 KB compress to 0.34..0.39 with their own table but 0.50..0.53 with a
shared one; 1 KB norms need a shared table to compress at all), and the total at
stake is under 2 MB. `tables.py` prints the per-kind rows.

## What was not done

- **No decoder was run.** Decode cost is given as a class and a symbol count, not as
  a measured time. No rANS encoder was run either: sizes are cross-entropy against
  the quantised CDF plus the stated framing, not round-tripped bytes.
- The whole-dtype ratios are extrapolations from 533 MB of 7.09 GB (F8_E4M3),
  665 MB of 3.87 GB (BF16) and 55 MB of 169 MB (F32); three layers per large kind.
- Adaptive models, mixing, two-pass parametric models and anything needing more than
  one table select per symbol were not tried.
- Whether these tensors are read from disk on the per-token path was not checked; if
  they are loaded once, their bytes do not cost NVMe time per token.
- The sample was read twice during development (one full run before the last two
  models and the extra embed/head slices were added), about 2.5 GB read in total.

## Files

- `measure.py` - sampling plan, models, cost accounting; `--selftest` checks the
  entropy code on synthetic data. Writes `results.json`.
- `tables.py` - prints every table above (and the per-tensor and per-kind ones) from
  `results.json`.
- `results.json` - per-record costs for every model (records >= 1 MiB), WBP1 only for
  the small ones, per-kind shared-table results, summary, population by kind.
