# NVFP4 expert weights under a WNR3-style coder (2026-11-17)

## Question

How well would a lossless coder built like WNR3 compress **NVFP4** routed-expert weights (E2M1 nibbles, one FP8 E4M3 scale per 16 values, FP32 per-tensor scalars), compared with what WNR3 gets on **MXFP4** (one E8M0 scale per 32 values) of the same base model? The answer decides whether WEX should get an NVFP4 codec; two engines that read NVFP4 checkpoints, Basalt and TensorFold, are the users it would serve.

## Recommendation

**An NVFP4 variant of WNR3 pays only modestly, and less than WNR3 pays on MXFP4.** On the two real NVFP4 exports measured (Qwen3.8-Flash-Next and GLM-5.3-Flash, NVIDIA modelopt), the best prototype configuration stores 0.9158 and 0.9148 of raw value-exact (canonical zero), 0.9336 and 0.9300 byte-exact, against 0.8810 and 0.8892 for WNR3 on the MXFP4 exports of the same models. The DeepSeek V4.1 Flash NVFP4 export is a special case (power-of-two scales, every two adjacent scales equal) and stores 0.8284.

Projected on the whole checkpoints (mean of the 16 sampled experts x the routed experts of the checkpoint; an estimate, not a full encode):

| model | NVFP4 experts raw | NVFP4 coded (prototype, value-exact) | saved | today's WEX4 option for these bytes (WBP1, 1 plane) | MXFP4 export coded by WNR3 |
|---|---:|---:|---:|---:|---:|
| DeepSeek V4.1 Flash (15,360 experts) | 305.77 GB | 253.29 GB (0.8284) | **52.5 GB** | 279.4 GB (0.9137) | 246.40 GB (CZ) / 254.32 GB (byte-exact) |
| Qwen3.8-Flash-Next (24,576 experts) | 67.95 GB | 62.23 GB (0.9158) | **5.7 GB** | 66.5 GB (0.9791) | 56.54 GB |
| GLM-5.3-Flash (12,096 experts) | 171.23 GB | 156.65 GB (0.9148) | **14.6 GB** | 167.2 GB (0.9767) | 143.79 GB |

GB = 10^9 bytes. Byte-exact (no canonical zero) the savings are 51.7, 4.5 and 12.0 GB.

Conditions under which it pays: an engine that must read NVFP4 (Blackwell NVFP4 kernels, TensorFold's NVFP4 path) and storage or read bandwidth that values 8 to 9% of the expert bytes (17% for the DeepSeek export). Where an engine can run the MXFP4 export of the same model instead, MXFP4 + WNR3 is smaller in absolute bytes on all three models (by 6.9 GB on DeepSeek value-exact, 5.7 GB on Qwen, 12.9 GB on GLM). Decode cost per byte is the same as WNR3's in the prototype (the WNR3 rANS, one more scale symbol per 16 nibbles): 22.5 to 27.0 MB/s on one thread, the speed of the WNR3 reference decoder (25.1 to 25.9 MB/s on the same machine); no fast NVFP4 decoder was written, so the 200 MB/s of the fast WNR3 decoder is not shown for NVFP4.

What a spec change would need (none is made here):

1. **A new record codec id and record magic** (WEX4 gives id 6 to WBP1, so 7; WNR3 records cannot carry NVFP4: `count_m = 32 * scount_m` is a WNR3 parse rule, erratum 11).
2. **Weight contexts**: blocks of 16 (pm and mx reset every 16 nibbles), the scale class from the whole E4M3 code, `clamp(code - anchor, -K, K)` with K = 6 to 24 (13 to 49 classes, 1,053 to 3,969 contexts per matrix; K = 6 gave 0.9158 on Qwen and 0.9149 on GLM, K = 12 0.9159 and 0.9148). The WNR3 class on the exponent field alone (7 classes, the WNM2 context count) costs 0.58 points more on Qwen and 0.34 on GLM, but is 0.95 points better on DeepSeek, whose scales are all powers of two; a coder would choose the class per matrix or per record. A new model blob is needed either way (WNM2 fixes 567 contexts and order-0 scale tables).
3. **Scale model**: order 1 on the previous scale of the row (257 tables of 256 symbols per matrix, 396 KB of tables for three matrices) saves 0.6 points on Qwen, 1.1 on DeepSeek, nothing on GLM against WNR3's order 0; with the parity of the position (513 tables) DeepSeek gains 0.24 more.
4. **Canonical zero** unchanged (Qwen and GLM NVFP4 hold -0 in 3.96% and 3.42% of nibbles; it is worth 1.8 and 1.5 points); value-exactness would need its own proof for each NVFP4 kernel that reads the records (not checked here). It cannot apply to NVFP4_NZ5 checkpoints, where code 8 means +5.
5. **No ceil form**: it changed the ratio by -0.03 to +0.01 points; on DeepSeek its gate fails on every matrix (blocks with maximum magnitude 3 exist).
6. **The WEX4 expert record** (WEX4-SPEC 5.5) admits only U8/F8_E8M0 scales `[rows, cols]` and U8/I8 weights `[rows, 16 * cols]`. NVFP4 needs F8_E4M3 scales `[rows, cols]`, weights `[rows, 8 * cols]` bytes and two F32 scalars per matrix (`weight_scale_2`, `input_scale`): either a second expert layout with 12 tensors (or 9, `input_scale` left as an ordinary tensor), selected by the scale dtype, or the scalars kept as ordinary tensors outside the record. Both WEX4 readers, `wex4_expert_entry()` and the legacy `WexEntry` (which has no field for the scalars) would change. TensorFold's csf0_v1 / NZ5 layout (a base per row and a 4-bit offset per 32) is a third layout and is not covered by this proposal.

## Setup

- Machine: Intel Core i7-7700K (4 cores, 8 threads), 32 GB RAM, Windows 10, MSYS2 UCRT64: gcc 16.2.0, Python 3.14.10, numpy 2.5.4, lz4 (Python) 4.4.5, zstd of Python 3.14 (`compression.zstd`). Other work on the machine was not controlled.
- This repository at `3e6cf26` (origin/main of 2026-11-17), DLLs of `wex4/build_and_test.sh` (built by `bash scripts/check_all.sh`, which ended with `all checks passed` before the run).
- Data: HTTP range requests only, `https://huggingface.co/<repo>/resolve/<commit>/<file>`: the 8-byte header length and the JSON header of each shard, then the byte range of each chosen tensor (adjacent tensors merged into one request). 1,839,700,920 B received in 631 requests (indexes and headers included), 0 retries. The samples were deleted after the run; `results/samples.tsv` gives every tensor with its shard, absolute byte range and SHA-256, so they can be fetched again and checked.
- Command: `bash research/nvfp4-wnr3-20261117/run.sh <DATA> <WORK> [<GLM container>]` runs every step below in order (the run itself was made step by step with the same commands; its logs are in `results/`).

### Checkpoints (commit = the `sha` of the Hugging Face API on 2026-11-17)

| name here | repository | commit | format | sampled file SHA-256 (fetched tensors, rewritten as one safetensors file) |
|---|---|---|---|---|
| ds-nvfp4 | nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 | 3ae9c0d3fbbea5370bd06e679d22044db15883f1 | NVFP4 (modelopt) | 9dd9a32c...6460b |
| ds-mxfp4 | deepseek-ai/DeepSeek-V4.1-Flash | 314113f8a7dadb388215396ba5fbee2c610a04d6 | MXFP4, I8 weights + F8_E8M0 scales (the original checkpoint) | 68ed960e...1c6 |
| ds-quark | amd/DeepSeek-V4.1-Flash-Quark-MXFP4 | 8eea1b23c275a839a1cf3a4cfffcbda6ad49ef14 | MXFP4, U8 + U8 | ecfbdd3a...2386 |
| qwen-nvfp4 | nvidia/Qwen3.8-Flash-Next-NVFP4 | fc694b54fb0174e0913e6adf86691ef85a4ead47 | NVFP4 (modelopt) | ea52967e...37e2 |
| qwen-mxfp4 | amd/Qwen3.8-Flash-Next-Quark-MXFP4 | b71e28afcd4b43add751159862235035613a58ab | MXFP4, U8 + U8 | b5c05d38...021f |
| glm-nvfp4 | nvidia/GLM-5.3-Flash-NVFP4 | 1d00738646cbb0c6d0a9eb403dac5f366f1473a7 | NVFP4 (modelopt) | 9cf27851...4c369 |
| glm-mxfp4 | amd/GLM-5.3-Flash-Quark-MXFP4 | b5688f25491202978c19c4d036eef579f61bbe07 | MXFP4, U8 + U8 | 8ec7ace7...eaf3e |
| glm-nz5 | Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4-NZ5 | 451ca95509e14f7d1a23d81bbe4591cba5a1ade7 | NVFP4_NZ5, csf0_v1 scales (TensorFold) | c15438e5...1a6c8 |

The full SHA-256 of each file is in `results/fetch.log`; per tensor, `results/samples.tsv` (columns: checkpoint, repository, commit, shard, tensor, layer, expert, first byte, last byte, bytes, SHA-256) and `results/fetch-log.json` (every request). glm-nz5 is not in the brief: it was added because it is the published NZ5 checkpoint that TensorFold's README names (at commit 1c900ea0 there; the newer commit of the API was used).

### Samples

`fetch.py`, `random.Random(20261117)` per model: the MoE layers present in every export of the model (layers past `num_hidden_layers`, the MTP layers, left out) are cut into 8 runs of consecutive layers, one layer is drawn per run and two experts per layer; every export of a model gets the same 16 (layer, expert) pairs, all six tensors (weight and scale of gate, down, up) and, for NVFP4, `weight_scale_2` and `input_scale`.

| model | MoE layers | (layer, expert) |
|---|---|---|
| DeepSeek V4.1 Flash | 0..39, 384 experts | (1,12) (1,119) (6,110) (6,175) (12,110) (12,176) (18,5) (18,31) (22,145) (22,150) (28,60) (28,361) (34,204) (34,375) (38,65) (38,88) |
| Qwen3.8-Flash-Next | 0..47, 512 experts | (1,24) (1,239) (7,220) (7,351) (14,221) (14,352) (21,11) (21,62) (29,300) (29,362) (32,121) (32,493) (40,408) (40,482) (43,131) (43,308) |
| GLM-5.3-Flash | 3..44, 288 experts | (4,12) (4,119) (9,110) (9,175) (15,110) (15,176) (22,5) (22,31) (26,145) (26,150) (32,60) (32,260) (38,88) (38,241) (41,154) (41,275) |

### Steps

1. `analyze.py`: raw sizes; nibble order; entropies pooled per matrix over the 16 experts (WNR3 trains one model per matrix); writes the slabs (`.nvs` for `nvwnr.c`, `.slab` for the baselines). `extra.py`: the DeepSeek byte comparisons, position and parity contexts, and the NZ5 checkpoint (`extra.py nz5`).
2. `wnr3_real.py pack`: the repository encoder on the MXFP4 samples, `wex4/wex4_convert.py pack --experts nr3` with WNM2 models trained by `c/tools/build_nr3.py`'s `train()` (via `c/tools/wex_nr3_tools.c`): on all 16 experts (in-sample) and on two folds (fold A = experts of the 1st, 3rd, 5th and 7th chosen layer, fold B = the others; cross-fold = each expert coded with the other fold's model), with and without `--cz`; every record decoded by the reference DLL and compared with the source slab (or Z(source)). `wnr3_real.py container`: the same 16 GLM experts read from the published GLM-5.3-Flash WNR3 container (`GLM-5.3-Flash-Quark-MXFP4.nr3.wex4`, content root `db3fb6c2...49e9d`) with `wex4/wex4.py`'s `Reader`.
3. `nvwnr.c` (research code, not a format): the WNR2/WNR3 rANS (32-bit state, L = 2^23, 14-bit tables, the quantizer of `wex_nr2_quantize`), WNR3's record overheads (32-byte header, u16 length per stream, 4-byte flush, chunks of 2048 nibbles and 512 scales, padding to 4096), no stream hashes. Options choose the block (16 or 32, from the file), the scale class of the nibble contexts, the scale model, canonical zero and ceil form (`nvwnr.c` header). `run_proto.py` runs it cross-fold with the folds of step 2 (and, for five NVFP4 configurations, in-sample) and checks every record's round trip byte for byte; decode time = one timed decode per record after one untimed, one thread.
4. `baselines.py`: zstd level 19, lz4 (default and level 12), WBP1 (`wex4_bp_encode_record`, 1 and 2 planes, chunk 2^15) on the same slab bytes (for NVFP4: scales, weights and the 24 bytes of scalars); every output decompressed and compared; decode = median of 3 passes on one thread.
5. `research/real-experts-20261115/bench_real.c wnr3 5`: the repository's reference and fast WNR3 decoders on the in-sample records of step 2, for the speed reference.
6. `summarize.py` writes `results/tables.md` (all tables below come from it).

## Results

All ratios are stored / raw bytes of the same 16 experts. One run of each step.

### Layout of the NVFP4 exports

- Shapes: weights U8 `[rows, cols/2]` (the same tensor shape as MXFP4), scales F8_E4M3 `[rows, cols/16]`, `weight_scale_2` and `input_scale` F32 scalars per matrix (24 bytes per expert). 4.5000 bits per weight against 4.2500 for MXFP4.
- **Nibble order: the same as MXFP4**, element 2i in the low nibble of byte i. Dequantized against the MXFP4 export of the same expert (64 rows of each matrix of the first sampled expert), the low-first reading correlates 0.9878 to 0.9906 with the MXFP4 values and the high-first reading -0.0086 to 0.0016 (`results/estimates.json`, `nibble_order`). The NZ5 checkpoint also reads low-first (0.9696 to 0.9729 against -0.0021 to 0.0004, unscaled values). No E4M3 scale had its sign bit set, none was 0 or NaN.
- **DeepSeek V4.1 NVFP4 is not an ordinary NVFP4 quantization**: all 35,389,440 sampled scales have mantissa 0 (6 codes, 0x50 to 0x78, powers of two) and every pair of adjacent scales of a row (blocks 2k and 2k + 1) is equal (17,694,720 of 17,694,720); `weight_scale_2` is a power of two (2^-13 to 2^-11). The values are not the original's: 79.80% of the dequantized values equal the original checkpoint's and 75.68% of the nibbles equal the Quark export's (`results/extra.json`). It contains no -0.
- The Quark MXFP4 export of DeepSeek is Z(original), byte for byte, in 16 of 16 experts (every -0 nibble of the original made +0, scales identical). The AMD Quark exports of Qwen and GLM contain no -0 either.

### Raw bytes and nibble properties (`results/tables.md` sections 1 and 2)

| checkpoint | weights B | scales B | scalars B | -0 nibbles | blocks whose max magnitude is 6 | scale codes used |
|---|---:|---:|---:|---:|---:|---:|
| ds-nvfp4 | 283,115,520 | 35,389,440 | 384 | 0 | 0.4074 | 6 |
| ds-mxfp4 | 283,115,520 | 17,694,720 | 0 | 5.83% | 0.5211 | 6 |
| ds-quark | 283,115,520 | 17,694,720 | 0 | 0 | 0.5211 | 6 |
| qwen-nvfp4 | 39,321,600 | 4,915,200 | 384 | 3.96% | 0.6031 | 51 |
| qwen-mxfp4 | 39,321,600 | 2,457,600 | 0 | 0 | 0.4152 | 7 |
| glm-nvfp4 | 201,326,592 | 25,165,824 | 384 | 3.42% | 1.0000 | 31 |
| glm-mxfp4 | 201,326,592 | 12,582,912 | 0 | 0 | 0.5601 | 4 |
| glm-nz5 | 201,326,592 | 6,422,528 (csf0: 131,072 bases + 6,291,456 offset bytes) | 384 | code 8 is +5: 1.96% | | |

Every GLM NVFP4 block of 16 holds a ±6: the scale is the block maximum over 6, so the scale carries what the nibble distribution would otherwise show.

### Entropy estimates (in-sample, no model or chunk cost; `results/tables.md` section 3)

Bits per nibble with the WNR3 contexts (scale class, pm, mx), and bits per scale:

| checkpoint | nibbles, order 0 | nibbles, WNR3 contexts | same after CZ | full scale code as class, CZ | scales, order 0 | scales, order 1 (left) | estimated stored / raw, WNR3-like, CZ | best estimate, CZ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ds-nvfp4 | 3.7787 | 3.6419 | 3.6419 | 3.6605 | 1.3016 | 0.8419 (0.5546 with parity) | 0.8274 | 0.8251 |
| ds-mxfp4 | 3.8942 | 3.7474 | 3.6307 | 3.6307 | 1.0004 | 0.9988 | 0.8616 | 0.8616 |
| qwen-nvfp4 | 3.9446 | 3.9206 | 3.8414 | 3.8422 | 4.5278 | 4.0992 | 0.9165 | 0.9108 |
| qwen-mxfp4 | 3.7760 | 3.6848 | 3.6848 | 3.6848 | 0.9636 | 0.9605 | 0.8741 | 0.8741 |
| glm-nvfp4 | 3.9680 | 3.9461 | 3.8777 | 3.8643 | 3.5395 | 3.4964 | 0.9109 | 0.9073 |
| glm-mxfp4 | 3.7860 | 3.7308 | 3.7308 | 3.7308 | 0.7060 | 0.7055 | 0.8830 | 0.8830 |
| glm-nz5 | 3.8937 | 3.8607 (class = csf0 offset) | not applicable | | 3.4572 (offset) | 3.3115 | | 0.9605 |

Why NVFP4 compresses less: its scales cost 3.5 to 4.1 bits per 16 weights (0.22 to 0.26 bits per weight) against 0.7 to 1.0 bits per 32 for MXFP4 (0.02 to 0.03), and its nibbles cost 3.84 to 3.88 bits against 3.68 to 3.73, because a scale fitted to each block of 16 leaves the nibbles spread more evenly over the grid. The scale class helps NVFP4 nibbles little: on GLM, 3.9580 bits with no class, 3.9461 with the exponent class, 3.9300 with the code class (no CZ). Other contexts tried: the position in the block while no ±6 has been seen saves 0.0003 to 0.0085 bits per nibble on NVFP4 (at most 0.2%); a delta along the row and the scale above are not better than the previous scale. These estimates are in-sample; on DeepSeek NVFP4 the cross-fold prototype is 1.45 points above the order-0 estimate (0.8419 against 0.8274), mostly in the scale stream (1.94 bits per scale coded cross-fold, 1.38 in-sample, 1.30 estimated): its scale distribution moves with depth, and folds made of whole layers see it (in-sample prototype rows below).

### WNR3 on the MXFP4 samples (`results/wnr3-*.json`, `results/glm-container.json`)

| checkpoint | byte-exact, cross-fold | byte-exact, in-sample | CZ, cross-fold | CZ, in-sample |
|---|---:|---:|---:|---:|
| ds-mxfp4 (profile deepseek) | 0.8807 | 0.8804 | **0.8533** | 0.8529 |
| ds-quark (deepseek-quark) | 0.8533 | 0.8529 | 0.8533 | 0.8529 |
| qwen-mxfp4 (qwen38) | **0.8810** | 0.8810 | 0.8810 | 0.8810 |
| glm-mxfp4 (qwen38) | **0.8892** | 0.8891 | 0.8892 | 0.8891 |
| glm-mxfp4, the 16 records of the published container (its own model) | 0.8891 | | | |

Every record decoded to the source slab (or Z(source) under `--cz`). These agree with the measured ratios of the brief: 0.854 on DeepSeek V4.1 REAP-192 with CZ, 0.878 on a Qwen3.8 layer, 0.8895 on the 12,096 GLM records. The container's 16 records (WNR3 flags 0x07: ceil form on all three matrices, no CZ bit) decode to the HF source bytes in 16 of 16; record sizes match the in-sample pack record for record (both are padded to 4096 B, which hides small model differences).

### Prototype (`nvwnr.c`), cross-fold (`results/proto.json`, `results/tables.md` section 5)

First its check against WNR3 on MXFP4 (the same streams, contexts, anchors and ceil form, no stream hashes):

| checkpoint | prototype | WNR3 encoder, cross-fold | difference |
|---|---:|---:|---:|
| ds-mxfp4, byte-exact | 0.8806 | 0.8807 | -0.0001 |
| ds-mxfp4, CZ | 0.8532 | 0.8533 | -0.0001 |
| qwen-mxfp4 | 0.8810 | 0.8810 | 0.0000 |
| glm-mxfp4 | 0.8891 | 0.8892 | -0.0001 |

NVFP4 (stored / raw; 16 of 16 round trips byte-identical in every row; decode MB/s on one thread):

| configuration | ds-nvfp4 | qwen-nvfp4 | glm-nvfp4 | MB/s |
|---|---:|---:|---:|---:|
| `nv-exp7-o0`: closest WNR3 adaptation (exponent class, scales order 0), byte-exact | 0.8419 | 0.9455 | 0.9335 | 24.7 to 26.0 |
| `nv-exp7-o0-cz`: the same with canonical zero | 0.8419 | 0.9278 | 0.9183 | 25.0 to 26.5 |
| `nv-exp7-o1-cz`: + scales order 1 | 0.8308 | 0.9216 | 0.9182 | 25.0 to 26.5 |
| `nv-exp7-o1p-cz`: + parity of the scale position | **0.8284** | 0.9215 | 0.9182 | 24.9 to 26.1 |
| `nv-code6-o1-cz`: class = clamp(code - anchor, -6, 6) | 0.8403 | **0.9158** | 0.9149 | 25.3 to 26.9 |
| `nv-code12-o1-cz`: K = 12 | 0.8403 | 0.9159 | **0.9148** | 24.7 to 26.9 |
| `nv-code24-o1-cz`: K = 24 | 0.8403 | 0.9161 | 0.9148 | 22.5 to 27.0 |
| `nv-exp7-o1-cz-ceil`: + ceil form (scale step one octave, 8 codes) | 0.8308 | 0.9213 | 0.9183 | 24.0 to 26.8 |
| `nv-exp7-o1`: byte-exact, scales order 1 | 0.8308 | 0.9391 | 0.9334 | 24.9 to 26.7 |
| `nv-code12-o1`: byte-exact | 0.8403 | 0.9336 | 0.9300 | 23.4 to 26.9 |

The same with the model trained on all 16 experts and tested on them (in-sample; an optimistic bound, where cross-fold by whole layers is a pessimistic one):

| configuration, in-sample | ds-nvfp4 | qwen-nvfp4 | glm-nvfp4 |
|---|---:|---:|---:|
| `nv-exp7-o0` | 0.8338 | 0.9412 | 0.9325 |
| `nv-exp7-o0-cz` | 0.8338 | 0.9237 | 0.9173 |
| `nv-exp7-o1p-cz` | **0.8234** | 0.9177 | 0.9167 |
| `nv-code6-o1-cz` | 0.8313 | 0.9130 | 0.9131 |
| `nv-code12-o1-cz` | 0.8313 | **0.9125** | **0.9131** |

So the value-exact NVFP4 ratio lies between 0.8234 and 0.8284 on DeepSeek, 0.9125 and 0.9158 on Qwen, 0.9131 and 0.9148 on GLM; for MXFP4 the two ends of WNR3 differ by at most 0.0004 (table above).

The prototype's MXFP4 rows decode at 24.4 to 25.5 MB/s. The repository's decoders on the same machine (`results/speed-*.txt`, 5 runs per row, MXFP4 records of step 2): reference `wex_nr2_decode` 25.1 to 25.9 MB/s on one thread, the fast CPU decoder 198.7 to 203.7 MB/s on one thread and 843.6 to 879.7 MB/s with 8 records at once. The prototype is a reference-style decoder (one symbol at a time, binary search in the tables); NVFP4 and MXFP4 decode at the same speed in it.

### Baselines on the same bytes (`results/baselines.log`)

| checkpoint | zstd -19 | lz4 default | lz4 -12 | WBP1 1 plane | WBP1 2 planes | best WNR3 / prototype |
|---|---:|---:|---:|---:|---:|---:|
| ds-nvfp4 | 0.8493 | 0.9289 | 0.9026 | 0.9137 | 0.9137 | 0.8284 |
| ds-mxfp4 | 0.9263 | 0.9702 | 0.9521 | 0.9595 | 0.9595 | 0.8533 (CZ) |
| ds-quark | 0.8990 | 0.9702 | 0.9521 | 0.9333 | 0.9333 | 0.8533 |
| qwen-nvfp4 | 0.9454 | 0.9989 | 0.9711 | 0.9791 | 0.9792 | 0.9158 |
| qwen-mxfp4 | 0.9012 | 0.9693 | 0.9523 | 0.9355 | 0.9359 | 0.8810 |
| glm-nvfp4 | 0.9363 | 0.9863 | 0.9578 | 0.9767 | 0.9768 | 0.9148 |
| glm-mxfp4 | 0.9011 | 0.9640 | 0.9501 | 0.9359 | 0.9359 | 0.8892 |

One-thread decode: zstd 722 to 1,622 MB/s, lz4 2,171 to 2,493 MB/s, WBP1 172 to 178 MB/s. zstd does well on DeepSeek NVFP4 (0.8493) because its duplicated scales are long matches.

### Canonical zero and NZ5

- -0 exists in the NVIDIA NVFP4 exports of Qwen (3.96% of nibbles) and GLM (3.42%), not in DeepSeek's; among the MXFP4 checkpoints only the original DeepSeek has it (5.83%). Value-exact results (CZ) and byte-exact results are kept apart in every table. CZ is worth 1.77 points on Qwen NVFP4 and 1.52 on GLM NVFP4 (`nv-exp7-o0` against `nv-exp7-o0-cz`), 2.74 on the original DeepSeek MXFP4 (WNR3).
- **NZ5** (TensorFold's `NVFP4_NZ5`, code 0b1000 read as +5): in the GLM NZ5 checkpoint code 8 is 1.96% of the nibbles. It removes canonical zero as an option (code 8 carries a value) and moves the waste of -0 into a real level: the nibbles cost 3.8607 bits with WNR3-style contexts (3.8603 when the context ranks +5 between 4 and 6), close to GLM NVFP4 after CZ (3.8777). Its csf0 scales are already small (4.1276 bits per weight raw, against 4.5 for NVFP4) and their 4-bit offsets cost 3.31 to 3.46 bits; the estimated stored / raw is 0.9605, so a coder would save about 6.2 GB of its 157.06 GB of experts (estimate, no coder was run on it). Not measured: the value change NZ5 makes (it is a different quantization, not a recoding).

## Limits

- 16 experts per checkpoint, one draw; whole-checkpoint numbers are projections (per-expert mean x routed experts) from them, using the cross-fold ratios. Models were trained on 8 experts of 4 layers (cross-fold); a production model trained on experts of every layer would land between the cross-fold and the in-sample rows (on MXFP4 they differ by at most 0.0004, on NVFP4 by up to 0.0081, DeepSeek order 0).
- The prototype has no stream hashes (WNR3's cost +0.0072% of raw), no owner and no model blob format; its model tables (up to 3,969 weight contexts and 257 scale tables per matrix) are not counted in the record sizes, as WNR3 does not count its blob.
- Decode speed: one timed decode per record after one untimed (`RUNS = 1` in `run_proto.py`), one thread, other work on the machine not controlled. No fast (vectorised) NVFP4 decoder, no GPU decoder.
- Not measured: a coder that recognises DeepSeek NVFP4's paired power-of-two scales and codes the export as blocks of 32; value-exactness of CZ under any NVFP4 kernel; NVFP4 exports with `input_scale` missing or with other modelopt versions; the Basalt and TensorFold engines themselves; TensorFold's other csf0 checkpoints; any coder on the NZ5 checkpoint (entropy only).
- The Hugging Face commits are those of 2026-11-17; the NZ5 repository has a newer commit than the one TensorFold's README names.

## Raw files

`results/`: `tables.md` (all tables, from `summarize.py`); `estimates.json`, `analyze.log` (step 1); `extra.json`, `extra.log`, `nz5.json`, `nz5.log`; `wnr3-<checkpoint>.json` (models, record sizes and slab SHA-256 per expert), `glm-container.json`, `wnr3.log` (step 2); `proto.json` (every record), `proto.log`, `proto2.log`, `proto3.log` (step 3: the first pass, three configurations added after it, the in-sample runs); `baselines.log` (step 4); `speed-<checkpoint>.txt` (step 5); `fetch.log`, `fetch-nz5.log`, `fetch-log.json`, `samples.tsv` (the samples).

Scripts: `fetch.py`, `analyze.py`, `extra.py`, `wnr3_real.py`, `nvwnr.c`, `run_proto.py`, `baselines.py`, `summarize.py`, `run.sh`. The weights stay under the licences of their checkpoints and are not in the repository.
