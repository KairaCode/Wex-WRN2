# WNR4 record skeleton: bytes saved over WNR3-CZ (2026-11-11)

Question: how many bytes does a "WNR4" expert record save over the production WNR3-CZ record with changes
that keep the decoder a static table-lookup rANS (same 567 contexts, same model, same independent streams)?

Short answer: **at most 0.58% of raw (108 KB of a 16.05 MB record)**, because that is everything a production
record spends outside the model's own code length. The changes that touch neither the per-stream 23-bit check
nor the stream count save **0.143% of raw per record, about 0.22 GB of the 123.3 GB container**.

## What was measured, on what

- Data: production container `v4_flash_experts.nr3.wex` (V4.1 Flash REAP-192 MXFP4, 7,680 WNR3-CZ records),
  opened read-only, with its own WNM2 model. Raw record 18,800,640 B.
- **Sample: 41 records**, one per layer (expert = (37 * layer + 11) mod 192) plus L39 E120 (the outlier: its
  scale payload is 1,079,648 B against ~135,000 B elsewhere). Deltas are against each expert's own production
  record, so nothing is in-sample or held-out here: it is the production model on production data. It is a
  sample of 41 of 7,680, though: every "container" figure not marked exact is sample mean x 7,680.
- **Whole container, exact**: the 32-B header and u16 length table of all 7,680 records were read (299 MB), so
  the padding figures and the coded-length-table figure at 2048:512 are exact sums, not extrapolations.
- Harness check: for 41/41 records, reference decode (`wex_nr3_decode`) -> `wex_nr3_prepare(cz=1)` ->
  `wex_nr3_encode_record` with the container's model reproduces the stored record **byte for byte**, and the
  prototype's own encoder in its WNR3 profile does too.
- Each candidate is a real record: encoded, decoded by the prototype decoder, and compared with the reference
  slab Z(source). **Round trip exact for 41/41 records in all 26 candidate profiles.**
- Smoke test only (not a corruption audit): one flipped payload bit per record was rejected 41/41 in the
  three `lenc_sd*` profiles at 2048:512.
- Total read from the container: 1.52 GB. CPU only, 8 threads. No GPU code was built or run; the GPU column
  is a reading of `c/wex_cuda.cuh` (`wex_cuda_nr2_offsets`, `wex_cuda_nr2_open`, `wex_cuda_nr2_renorm`,
  `wex_cuda_nr2_scales`, `wex_cuda_nr2_weights`).

Where a production record's bytes go (sample mean): model code length 3.577 bit/weight and 1.070 bit/scale;
on top of that **5.57 B per stream**: 2.00 B of u16 length and 3.57 B of rANS start/flush (the 23-bit hash
slot is 2.875 B of it). 19,440 streams x 5.57 B = 108,266 B = 0.576% of raw. Every row below takes a slice
of those 108 KB; none of them can exceed it.

## Results

Saved bytes per record are unpadded, mean (min..max) over the 41 records; candidate records carry a 40-B
header (8 B more than WNR3, already subtracted). Container saving = mean x 7,680 (extrapolated) unless
marked exact.

| # | change | saved B/record | % of raw | container | per-stream check | GPU kernel change (judged from the source) |
|---|---|---|---|---|---|---|
| 1 | **Coded length table** (`lenc`): streams in g order, groups of 32: u16 min, u8 width w, residuals packed at w bits (14,396 B instead of 38,880 B) | 24,399 (20,022..25,054) | 0.130 | **188.0 MB exact** (24,484 B/record over all 7,680) | unchanged (23-bit hash) | none: the host builds the same u32 offsets table (`wex_cuda_nr2_offsets`); host cost 8.5 -> 36.5 us/record with a naive unpack |
| 2a | **Minimal start state** (`sd`): the 4-B state is stored in 3 B when it is < 2^24 (12.5% of streams) | 2,418 (2,320..2,501) | 0.013 | 18.6 MB | unchanged (23-bit hash) | 3 lines in `wex_cuda_nr2_open`: take 3 bytes, take the 4th if state < L; minimum stream length 3 |
| 2b | Tail in the start state (`tailN_hashM`): the last N weight nibbles of each stream raw in the low bits of x0, hash shortened to 23 - 4N bits. N = 1 / 2 / 3 / 4 / 5 | 7,684 / 15,433 / 23,165 / 30,877 / 38,628 (N = 5: 38,447..39,578) | 0.041 / 0.082 / 0.123 / 0.164 / 0.205 | 59 / 119 / 178 / 237 / 297 MB | 19 / 15 / 11 / 7 / 3 hash bits, plus a 32-bit record digest (XOR of the stream hashes) | weight kernel: the last block decodes 32 - N symbols and takes N from the final state (before the hash, CZ test, block max and ceil inverse); window check instead of equality; warp XOR, one `atomicXor`, digest compare |
| 2c | Same plus 2 raw scales per scale stream (`tail5_stail2`, the drafts' TAIL) | 39,207 (38,925..43,719) | 0.208 | 301 MB | 3 bits (weights), 7 bits (scales), plus digest | as 2b, in both kernels |
| 2d | Small initial state x0 = 2^k + k-bit hash, k = 16 / 8 / 0 | 16,985 / 34,928 / 35,278 | 0.090 / 0.186 / 0.188 | 130 / 268 / 271 MB | 16 / 8 / 0 hash bits | **not free**: `wex_cuda_nr2_renorm` must stop at the stream end, a clamp on every symbol of the hot loop |
| 2e | x0 = 2^20 + 5 tail nibbles, no hash (`small_k20_tail5`) | 46,092 (45,854..47,123) | 0.245 | 354 MB | none; record digest only | 2b and 2d together |
| 3a | Weight chunk 4096 / 8192 / 16384 (scale 512) | 48,103 / 72,159 / 84,193 | 0.256 / 0.384 / 0.448 | 369 / 554 / 647 MB | unchanged | none (chunks are already per record in WNR3, no new codec needed); weight streams 17,280 -> 8,640 / 4,320 / 2,160 |
| 3b | Scale chunk 1024 / 2048 (weight 2048) | 6,006 / 9,013 | 0.032 / 0.048 | 46 / 69 MB | unchanged | none; scale streams 2,160 -> 1,080 / 540 |
| 3c | 4096:1024 / 8192:2048 | 54,119 / 81,187 | 0.288 / 0.432 | 416 / 624 MB | unchanged | none; streams 19,440 -> 9,720 / 4,860 |
| 4 | Record alignment 512 B / 64 B / none instead of 4096 B (mean padding today 2,058 B) | 1,802 / 2,027 / 2,058 | 0.010 / 0.011 / 0.011 | **13.84 / 15.57 / 15.81 MB exact** | n/a | none for 512 and 64 (the kernels view the record as u32 words, so it must stay 4-byte aligned: "none" is out); the store must read the enclosing sectors |
| 1+2a | **`lenc_sd`, 2048:512** | **26,825 (22,428..27,458)** | **0.143** | **206 MB** | unchanged | host offsets builder, 3 lines in open |
| | `lenc_sd` + tail2 (hash 15) / tail5 (hash 3) | 42,276 / 65,478 | 0.225 / 0.348 | 325 / 503 MB | 15 / 3 bits, plus digest | as 2b |
| | `lenc_sd` at 4096:1024 / 8192:2048 | 66,680 / 87,182 | 0.355 / 0.464 | 512 / 670 MB | unchanged | as 1+2a |
| | `lenc_sd` + tail5 at 4096:1024 / 8192:2048 | 86,010 / 96,845 | 0.458 / 0.515 | 661 / 744 MB | 3 bits, plus digest | as 2b |

Other things measured:

- **Length-table alternatives** (76 records, 2048:512, B/record): u16 38,880; one fixed width per matrix
  15,788; groups of 16 / 32 / 64 / 128 with min + width 15,761 / 14,499 / 14,259 / 14,525; order-0 entropy of
  (length - matrix median), the bound for any static per-matrix code, 11,471. The drafts' scale-predicted
  LENC measured 13,119 B (13,366 B with CZ) and needs the decoded scales before the weight offsets exist,
  that is a GPU-side offset pass or a CPU scale decode; the group scheme needs neither and is 1.1 KB behind.
- **Shorter flush with the full 23-bit hash**: 2a is all there is. With byte renormalisation the encoder's
  final state lies in [2^23, 2^31) and needs 3 or 4 bytes; the hash slot is 23 of the stored bits by
  definition. Everything larger (2b-2e) spends hash bits.
- **Per-layer model instead of one container model** (the drafts' table sets, tried per layer): 7 layers,
  model trained on 4 experts of the layer, real WNR3-CZ records of 2 other experts of that layer, all
  verified: 3,197 B saved on average (0.017% of raw), range -13,623..+50,479 B over 14 records; a pooled
  retrain on the same 28 experts saves 1,229 B. Nothing there.
- **Sign plane under canonical zero**: 11.72% of the coded nibbles have magnitude 0 and carry no sign under
  CZ. A raw sign plane stores a bit for each of them: +518 KB/record (+2.76% of raw), computed from the
  measured share, not encoded. It gives back the whole CZ gain; it is incompatible with CZ.
- CPU reference decode (scalar): 0.55 s/record with 2 threads busy, 0.73 s mean with 8.

## What the numbers say about decode speed

- Nothing in this table makes the GPU decode faster. Rows 1, 2a and 4 leave both kernels' per-symbol loops
  untouched; 2b adds per-stream work only; 2d adds per-symbol work.
- Chunk size is the only row that trades bytes against parallelism, and that trade was already measured on
  the 5090 (V4 WNR2 records, `docs/WNR2-REFERENCE.md`, bench_nr2_rechunk and test_decode_batch; not repeated
  here and not on V4.1): single record 0.264 ms at weight chunk 2048 against 0.437 ms at 4096, 8 concurrent
  0.132 against 0.137 ms; scale chunk 512 against 2048 at weight 2048: 0.200 against 0.261 ms single; batch
  of 16 records 0.382 ms (2048:512) against 0.352 ms (4096:2048). Larger chunks cost single-record latency
  and nothing in large batches. Production moved to 2048:512 on those numbers.
- The u32 offsets table sent to the GPU with every record is 4 B per stream: 77.8 KB at 19,440 streams,
  38.9 KB at 9,720, 19.4 KB at 4,860 (computed).
- Alignment below 4096 B saves file bytes only. An unbuffered read needs a sector-aligned offset and length,
  so the store would read the enclosing sectors and decode from an offset inside the buffer; the bytes read
  per record do not go down. Today `expert_store_wex.c` requires a 4096-aligned `file_offset` for that path.

## Recommendation

1. **These changes do not justify a new record codec on their own.** The full skeleton overhead is 0.576%
   of raw; the part reachable without touching integrity or stream count is 0.143% plus padding.
2. **If codec 6 is created for another reason, include rows 1, 2a and 64-B alignment**: `lenc_sd` at
   2048:512 with records aligned to 64 B. Expected saving: 206 MB (extrapolated; 188.0 MB of it exact) +
   15.6 MB (exact) = **about 0.22 GB, 0.18% of the 123.3 GB container**. Decoder cost: a different host
   offsets builder, three lines in the stream open, sector-rounded reads.
3. **Chunk size is a build parameter, not a codec feature**: 4096:1024 adds 0.21% of raw on top of point 2
   (0.53 GB in total with `lenc_sd` and 64-B alignment), 8192:2048 0.32% (0.69 GB). Decide it from the batch
   sizes the engine really issues; the V4 measurements say single-record decodes get 30-65% slower.
4. **Tail in the start state only together with a record digest that the GPU path verifies**: +0.205% of
   raw (0.52 GB in total with `lenc_sd` and 64-B alignment), for a per-stream check of 3 bits instead of 23
   and new kernel code in the last block of every stream. Its integrity side was not audited here.
5. Not worth doing: small initial state (per-symbol cost), per-layer models (no gain), sign plane (costs
   2.76% under CZ), alignment below 64 B.

Everything together (8192:2048, `lenc_sd`, tail5, 64-B alignment): **0.76 GB, 0.62% of the container**, with
the latency and integrity costs of points 3 and 4.

## Not checked

- No GPU build or run: kernel changes are read from the source, their cost is not measured.
- Chunk-size speed figures are the earlier V4 / WNR2 measurements, not V4.1 / WNR3.
- No corruption audit of the shortened-hash profiles (one bit flip per record only).
- The model was not retrained; contexts and CDF precision were not varied (other pieces of this research).
- `run.sh` was assembled from the commands that produced these numbers; it was not run end to end as one script.

## Files

- `wnr4_proto.c`: encoder/decoder copies with the four switches, harness check, measurements.
- `fetch.py`: read-only copy of the sample records; exact scan of every record's header and length table.
- `layer_model.py`: per-layer model test through the encoder DLL.
- `report.py`: reduces `out/*.json` to `results.json` and prints the candidate table.
- `results.json`: all figures above. `run.sh CONTAINER`: build and rerun everything (reads ~1.5 GB).
