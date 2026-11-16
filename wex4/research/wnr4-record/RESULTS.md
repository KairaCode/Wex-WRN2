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
- `run.sh` was assembled from the commands that produced these numbers; it was first run end to end as one
  script on 2026-11-16 (next section).
  Since 2026-11-16 it and `verify/run_verify.sh` read `CC` and `WEX_OPENMP` and build their library
  position-independent outside Windows (`run_verify.sh` takes `python` as its default `PY`). That day both built
  with gcc 16.2 (MSYS2 UCRT64, Windows 10) and gcc 15.2 (Ubuntu under WSL2); clang 23.1.5 of MSYS2 stops at
  `omp.h` (it has no OpenMP runtime). Before the production container was on that machine, the programs they
  build ran by hand on 8 experts of layers 10 and 30 as WNR3-CZ records (model trained on the 8), with both
  builds: `wnr4_proto` `8 experts, 0 failures`, and `verify/extra.c` on one record rejected every single-bit
  flip of 1,220 per profile, with no wrong output accepted. Under WSL2 both scripts stopped at `import numpy`
  (no numpy there).

## Rerun on the published container (2026-11-16)

Both scripts ran to their end, once each, on the container published at
<https://huggingface.co/Kaira666/Kaira-WEX4-Experts> (`v41-reap/v4_flash_experts.nr3.wex`): the WEX3 WNR3-CZ
file above upgraded to WEX4 in place, 123,312,910,336 B, root `f0f78a84...3beeb1`, which `wex4_verify` had
checked whole on the same machine (every blob and record matches its SHA-256). Machine: Intel Core i7-7700K
(4 cores, 8 threads, AVX2), 32 GB of RAM, Windows 10, the container on a SATA SSD (Crucial BX500); MSYS2 UCRT64
gcc 16.2.0, Python 3.14.10, numpy 2.5.4; CPU load 19% to 33% in the readings before and after each script
(other work on the desktop). Commands: `PY=python3 bash run.sh CONTAINER` and
`PY=python3 ST_EXPERTS=10:0,10:1,10:2,10:3,30:0,30:1,30:2,30:3 bash verify/run_verify.sh CONTAINER DIR`.

What had to change for that (no change of what is measured): `fetch.py`, `verify/scan.py` and
`verify/fetch_v.py` read the WEX3 header page that the upgrade kept (blob of kind 16, its SHA-256 checked
against the index) when page 0 is a WEX4 header; the WEX3 entry table, the model and the records are where
they were. `st_check.py` takes the experts of its first part as an argument (`ST_EXPERTS` of `run_verify.sh`,
default the two of before), and `run_verify.sh` fetches their records with `fetch.py`.

- **`run.sh`**: exit 0 in 355 s, last line of `report.py` with `"records": 76`. 41 of 41 records `harness ok`
  (the stored record reproduced byte for byte). `results.json` came out identical to the committed one except
  three timings, which are of another machine and load: `reference_decode_s_mean` 0.727 -> 1.083 s (8 threads
  busy) and the host offsets table 8.5 -> 14.0 us (u16 prefix sum) and 36.5 -> 71.5 us (`lenc` unpack). Every
  byte count, saving and share is the same. The rerun's file is `results-rerun-20261116.json`;
  `results.json` is left as it was. `fetch.py` read 659,916,816 B (records) and 299,460,624 B (the
  scan) of the container.
- **`verify/run_verify.sh`**: exit 0 in 330 s. `scan.py` read every record header and length table of the
  7,680 records: 0 rows differ from `run.sh`'s scan. `wnr4_proto` on the 23 records of `fetch_v.py` (16 fresh,
  4 overlapping, 3 extreme): `23 experts, 0 failures`; `compare.py`: the 4 overlapping records have identical
  sizes, and every profile's mean saving on the 16 fresh records is within 0.4% of the 41-record figure (for
  example `lenc_sd` 26,800 against 26,825 B). `extra.exe` on L16 E177: reference decode 0.761 s per record on one
  thread (best of 3), offsets table 7.00 / 32.20 us, and in each profile every single-bit flip of the header
  (320), the length section (300), the payload (300) and the stream starts (300) rejected, 0 accepted with a
  wrong output. The chunk variants of `st_check.py` part 2 through the unchanged WNR3 DLL: all 8 decode back to
  the slab, and the prototype's sizes are those of the WNR3 codec plus its 8-byte longer header.
- **`st_check.py` part 1 did not check what it is for.** The only safetensors on the machine were the expert
  shards 13 and 33 of 48 of `deepseek-ai/DeepSeek-V4.1-Flash` (layers 10 and 30, experts 0 to 3), which has
  384 routed experts per layer (its `config.json`). The container was made from a REAP-192 checkpoint, which
  keeps 192 of them under new numbers. For all 8, the container's record of the same layer and number decoded
  to other bytes than Z(source) (`False` on every line). One more check by hand (a short script, not in the
  repository): all 192 records of layers 10 and 30 decoded with the CPU reference and their SHA-256 compared
  with those of Z(source) of the 8 experts, whole slab and scale part apart: no match, not even of the scales.
  So these shards (commit `314113f8` of that repository) are not the source of the container's records; the
  REAP-192 checkpoint it was made from (`DeepSeek-V4.1-Flash-1017-REAP192-MXFP4` in `run_verify.sh`) was not on
  the machine, and the comparison with it is still open. Nothing here points at the codec: every record
  decoded, and every record the two harness runs re-encoded from its decode (41 and 23) came out byte for
  byte as stored.

## Files

- `wnr4_proto.c`: encoder/decoder copies with the four switches, harness check, measurements.
- `fetch.py`: read-only copy of the sample records; exact scan of every record's header and length table.
- `layer_model.py`: per-layer model test through the encoder DLL.
- `report.py`: reduces `out/*.json` to `results.json` and prints the candidate table.
- `results.json`: all figures above. `run.sh CONTAINER`: build and rerun everything (reads ~1.5 GB).
- `results-rerun-20261116.json`: `results.json` as the rerun of 2026-11-16 wrote it.
- `verify/`: the verification rerun (`run_verify.sh`), its scripts and `extra.c`.
