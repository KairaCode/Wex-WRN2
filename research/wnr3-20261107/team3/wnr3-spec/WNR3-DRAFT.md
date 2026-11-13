# WNR3 draft specification (DRAFT 0.1, 2026-11-07, team3 / wnr3-spec)

**Status:** paper design. Nothing here is implemented in C/CUDA. It extends WNR2 as described in `D:\Desktop\WNR2-REFERENCE.md` sections 4-7 and 9-11; the KAIRA sources (`wex_format.h`, `wex_neural_rans.h`, `wex_cuda.cuh`, `wex_cuda_bridge.cu`) are NOT on this machine (`<HOME>` does not exist), so no source line was checked. Every rule that depends on today's code says so and is listed in section 12 as a check to do.

**Evidence tags:** [M] measured on the real V4.1 sample (160 experts, `<RESEARCH-PC>\v41-sample`), either by the earlier team (`measured-v41.md`, `<RESEARCH-PC>\v41-results\*.json`) or here (`lenres.json`, `wnr3_ref.out.json`, both in this folder); [R] read in the WNR2 reference; [S] from a source opened on the web; [I] inferred or arithmetic. No GPU number in this file is measured: the RTX 3070 was in use by another agent and no CUDA toolkit is installed.

**Files in this folder:** `WNR3-DRAFT.md` (this file); `wnr3_ref.py` (executable reference for H32, the stream hash, the canonical context index and the coded length table, with self-checks; output `wnr3_ref.out.json`); `lenres.py` (length-residual measurement; output `lenres.json`, log `lenres.log`).

---

## 0. Summary of decisions

| id | feature | record flag | needs in the model | where it lives | expected effect (V4.1, per record) | decoder cost [I] |
|---|---|---|---|---|---|---|
| a | block-max side symbol `bm7` + context x2 + block half x2 | BM7 (bit 0) | layout 1 (2268 full contexts, 910 rows, canonical index) + `bm7` tables | one binary rANS symbol at the START of every 32-weight block, inside the weight stream | -0.21% of weight bits [M, held-out] | +1 rANS step per 32 symbols (<= +3.1% steps); table 18,144 -> 29,120 B shared memory |
| b | K weight-table sets, one chosen per matrix | (none: `set_id[3]`, K = 1 means off) | K sets (1..16) | `set_id[m]` u8 in the record header | with (a): -0.34% (K=2) / -0.35% (K=4) of weight bits [M] | none per symbol; set rows copied from L2 at block start |
| c | canonical zero (FP4 code 8 -> 0) | CZ (bit 1) + header flag 64 + entry bit 9 | (optional CZ_TRAINED tag) | encoder-side mapping; decoder rejects symbol 8 | with (a)+(b): -3.38% of weight bits [M]; VALUE-exact only | one nibble test per output word |
| d | coded stream-length table | LENC (bit 2) | LENPRED tables | length section: per 32-stream group a u32 base offset + a width byte + bit-packed zigzag residuals vs a prediction from the decoded scales | 38,880 -> ~15,825 B = -0.123% of the raw record [M residuals, I layout] | per lane: ~64 scale reads + one warp scan in the kernel prologue; NO extra kernel; removes the 77,764-B host offsets upload |
| e | per-stream integrity | HASH (bit 3) | none | rANS initial state x0 = L + h23; decoder checks final state == L + h23(decoded) | +0.0697 B/stream mean = +1,354 B (+0.0072% of raw) [I, matches 0.067 B M] | ~24 integer ops per 32 symbols off the critical chain + one compare |

Other decisions:
- **Container**: stay on WEX3 version 3, 4096 alignment, 72-B entries; add header flag `NEURAL_RANS3 = 32`, header flag `CANONICAL_ZERO = 64`, entry codec `5 = NEURAL_RANS3`, descriptor `WND3` at header byte 64, model blob `WNM3`. Old code fails closed twice (unknown header flag, unknown codec). Section 2.
- **Context compaction is structural, not data-driven**: the 910 reachable rows of layout 1 follow from the rule itself (pm <= mx; bm7 = 0 forbids magnitude 7; block start only in half 0). The model still STORES the index (context -> row), but v1 requires it to equal the canonical one, so the GPU computes the row with 3-4 integer ops and no extra dependent shared-memory load. Checked here [M]: 0 rows used by the 160-expert counts lie outside the 910; the used rows per matrix are 589 / 803 / 623; 100 of the 910 are never used. Section 3.3.
- **Hash function**: `H32` = MurmurHash3_x86_32 over a 16-B identity + the stream's decoded bytes, seed 0x33524E57, truncated to the low 23 bits. Chosen over CRC32C because NVIDIA GPUs have no CRC instruction (a table CRC would put more dependent loads and 1-4 KB of tables into the same shared memory the CDF rows need) [I]. Implementation checked against the SMHasher verification value 0xB0F57EE3 [S, M]. Section 5.6.
- **Record identity** (owner layer/expert) is stored in the record header AND mixed into every stream hash; the caller passes the expected identity, which needs new helper entry points (feature bit 16). Section 8.
- **Expected V4.1 average record** [I from M deltas]: (a)+(b)+(d)+(e) ~16,834,470 B = 0.8954 of raw (-0.47% vs WNR2's 16,914,680; capacity vs WEX1 +11.68% instead of +11.15%); with (c) also ~16,327,660 B = 0.8685 of raw (+15.15%). Section 9.

---

## 1. Scope, terms, constants

- Scope: one record = one expert = 3 matrices m = 0 gate w1, 1 down w2, 2 up w3 (unchanged matrix order, unchanged DECODED slab layout: scales of w1, w2, w3, then packed FP4 weights of w1, w2, w3) [R §5].
- Not in scope: new scale contexts (left+upper gave -0.5% of scale bits = 0.004% of the record [M]); cross-expert coding (no headroom [M]); per-expert models (net +0.07% [M]).
- rANS is exactly WNR2's [R §6-7]: 32-bit state, `L = 2^23` (`WEX_NR_L`), precision `P = 14` bits (total 16384), byte renormalisation (`x = x<<8 | byte` while `x < L`), 4-B big-endian initial state in `[L, 256L)`, every symbol frequency >= 1, encoder `x_max = ((L >> 14) << 8) * freq`. Decoder step: `slot = x & 16383`; symbol s with `cum[s] <= slot < cum[s+1]`; `x = freq[s]*(x >> 14) + slot - cum[s]`; renormalise.
- `align4096(n)` = n rounded up to a multiple of 4096; `pad4(n)` = n rounded up to a multiple of 4. All integers little-endian unless stated (the rANS initial state stays big-endian, as in WNR2).
- Block = 32 consecutive FP4 weights of a row with one UE8M0 scale. `count_m` = weights of matrix m = `weight_shapes[m][0] * weight_shapes[m][1] * 2`; `scount_m` = scales of m. `count_m % 32 == 0` is already enforced (scale cols * 16 == weight byte cols, [R §5]); so every weight stream holds whole blocks and block b of matrix m uses scale b (row-major).
- Chunks: `Cw` (weight chunk, power of 2, 512..16384) and `Cs` (scale chunk, power of 2, 256..16384), stored per record as in WNR2 [R §6]. Streams of m: `Ws_m = ceil(count_m / Cw)`, `Ss_m = ceil(scount_m / Cs)`; `S = sum Ss_m`, `W = sum Ws_m`. V4.1 at 2048:512: Ws_m = 5,760, Ss_m = 720, S = 2,160, W = 17,280, 19,440 streams [R §2].
- **Global stream index g**: scale streams of m0, m1, m2 get g = 0..S-1, weight streams of m0, m1, m2 get g = S..S+W-1. The scale payload is the concatenation of the scale streams in g order, the weight payload the concatenation of the weight streams in g order (same as WNR2).
- **Groups** (feature d): within each matrix and each kind, streams are cut into groups of 32 consecutive streams; the last group of a matrix may be partial. `ngS = sum ceil(Ss_m/32)`, `ngW = sum ceil(Ws_m/32)`. V4.1: ngS = 3*23 = 69, ngW = 3*180 = 540. A weight group is exactly the 32 streams one warp of today's weight kernel decodes ("one thread per stream, 32 threads/block, each block serves one matrix" [R §10]).

Constants (new names in the style of `wex_format.h`; values are this draft's proposal):

| name | value | meaning |
|---|---|---|
| WEX_HEADER_NEURAL_RANS3 | 32 | header flag: WNR3 records + a WNM3 model before the table |
| WEX_HEADER_CANONICAL_ZERO | 64 | header flag: some records may be value-exact (canonical zero) |
| WEX_CODEC_NEURAL_RANS3 | 5 | entry codec |
| WEX_NR3_RECORD_MAGIC | 0x33524E57 | 'WNR3' |
| WEX_NR3_MODEL_MAGIC | 0x334D4E57 | 'WNM3' |
| WEX_NR3_DESCRIPTOR_MAGIC | 0x33444E57 | 'WND3' |
| WEX_NR3_DESCRIPTOR_OFFSET | 64 | header byte offset of WND3 |
| WEX_NR3_HEADER_BYTES | 64 | record header size |
| WEX_NR3_MAX_MODEL_BYTES | 1,048,576 | upper bound for a WNM3 blob |
| WEX_NR3_MAX_SETS | 16 | K <= 16 |
| WEX_NR3_GROUP | 32 | streams per length group |
| WEX_NR3_HASH_SEED | 0x33524E57 | H32 seed (= record magic) |
| WEX_NR3_HASH_MASK | 0x7FFFFF | 23 bits |
| WEX_CUDA_FEATURE_NR3 | 16 | helper feature bit |

---

## 2. Container (WEX3 version 3)

### 2.1 Header page
- `WexHeader` is unchanged (40 B: magic WEX3 0x57455833, version 3, num_layers, num_experts, aligned_expert_bytes, table_offset, flags, reserved = 0), followed by the 16-B build UUID at bytes 40..55 [R §5].
- Bytes 56..63 are zero. The **WND3 descriptor** occupies bytes 64..119 (56 B):

| off | type | field | rule |
|---|---|---|---|
| 64 | u32 | magic | 0x33444E57 'WND3' |
| 68 | u32 | model_bytes | == WNM3 `total_bytes`; 64 < bytes <= WEX_NR3_MAX_MODEL_BYTES |
| 72 | u64 | model_offset | == 4096 (the builder's choice; readers accept any 4096-aligned value >= 4096 with model end <= table_offset) |
| 80 | u32 | model_id | FNV-1a 32 of the blob (section 3.7), != 0 |
| 84 | u32 | model_summary | `layout | K << 8 | features << 16`, must equal the blob's fields (lets the engine refuse a model it cannot run before reading it) |
| 88 | u8[32] | model_sha256 | SHA-256 of the blob, checked once at open |

- Bytes 120..4095 of the header page are zero (checked).
- The model lies in `[model_offset, model_offset + model_bytes)`, zero-padded to `model_offset + align4096(model_bytes)`, which equals `table_offset` for the builder's layout. Examples (section 3.6): K = 4, layout 1 -> 130,088 B -> table_offset 135,168 (WNR2: 65,536).
- Then the 4096-aligned entry table, then records contiguous to EOF, each padded to 4096 (unchanged rules of `wex_validate_table` [R §5]).

### 2.2 Header flags
- New: `NEURAL_RANS3 = 32`, `CANONICAL_ZERO = 64`. Accepted v3 flag set for new readers: 1|2|4|8|16|32|64.
- 32 requires version 3; in this draft 32 excludes 8 and 16 (one model kind per container; the descriptor slot at byte 64 is shared). 64 requires 32.
- The WNR3 builder writes 37 (SCALE_TILES|DEDUP|NEURAL_RANS3, mirroring WNR2's 21 [R §5]), or 101 when it ran with canonical zero.
- Meaning of 64: "records with the CZ flag may exist; their decoded weights equal the source in VALUE, not in bytes". A container with flag 64 and no CZ record is valid. Readers that require byte-exactness refuse a flag-64 container (engine policy in section 8.3).

### 2.3 Entry codec 5
- `WexEntry` (72 B) is unchanged. `flags` bits 0-7 = 5. For codec 5, **bits 8-11 mirror the record flags** (bit 8 BM7, bit 9 CZ, bit 10 LENC, bit 11 HASH) and bits 12-31 are 0. (WEX2 uses bits 8-31 for scale_pages [R §5]; codec 5 never has scale pages. See section 12 for the check this needs.)
- `wex_validate_entry` for codec 5 requires:
  1. header version 3 with flag 32;
  2. bit 9 set => header flag 64;
  3. bit 8 set <=> WND3 layout == 1 (layout 1 models REQUIRE BM7 records; layout 0 models forbid them);
  4. bit 10 set => WND3 features has LENPRED;
  5. `total_bytes` 4096-aligned, `>= align4096(64 + 4 * streams_at(16384, 16384))` (every stream is >= 4 B) and `< aligned_raw` (a record that is not smaller than raw is stored raw, flags 0, as today);
  6. the decoded geometry passes `wex_entry_raw_sizes` as for codec 4.
- Record placement is unchanged: `wex_record_read_offset = aligned_raw - total_bytes`; the record is read into the END of the decoded slab [R §5].

### 2.4 DEDUP and record identity
- Each WNR3 record stores its **owner** (layer, expert). Aliases (byte-identical 72-B entries sharing file_offset, only with DEDUP [R §5]) all decode the owner's record.
- Owner rule: the owner of a physical record is the entry with the smallest `layer * num_experts + expert` among the entries that reference it. `wex_validate_table` computes, for every entry, its **bind** = (owner_layer, owner_expert); for a non-aliased entry bind = its own (layer, expert). Decoders receive the bind from the caller and require `record.owner == bind`.
- V4.1 has no duplicates (0 duplicate matrices among 160 experts [M]), so in practice bind = own index.

### 2.5 Why not a new version or magic
- WEX3 v4 or a WEX4 magic would also fail closed, but the 40-B header, the 72-B entry, the table rules, the mirror rule (byte-identical header page + table [R §11]) and the record placement do not change. A flag + codec keeps all of them and costs nothing.
- Old readers fail closed: the v3 validator accepts only flags 1|2|4|8|16 [R §5] and refuses flag 32; even a reader that ignored unknown header bits refuses codec 5 in `wex_validate_entry`; `repack_v4_wex.py validate_container` allows v3 flags only up to 15 [R §8]; an engine without helper feature 16 refuses the container (section 8.2). The first point must be confirmed in `wex_format.h` (section 12).

---

## 3. WNM3 model blob

### 3.1 Header (64 B)

| off | type | field | rule |
|---|---|---|---|
| 0 | u32 | magic | 0x334D4E57 'WNM3' |
| 4 | u16 | version | 1 |
| 6 | u16 | precision | 14 |
| 8 | u16 | layout | 0 BASE (WNR2 rule), 1 BM7H (feature a) |
| 10 | u16 | sets K | 1..16 |
| 12 | u16 | full_contexts | 567 (layout 0) or 2268 (layout 1) |
| 14 | u16 | rows | 259 (layout 0) or 910 (layout 1) |
| 16 | u16 | matrices | 3 |
| 18 | u16 | scale_symbols | 256 |
| 20 | u32 | features | bit 0 LENPRED (length predictor present), bit 1 CZ_TRAINED (informational: trained on canonicalised data); other bits 0 |
| 24 | u32 | index_off | == 64 |
| 28 | u32 | wsets_off | == index_off + pad4(2 * full_contexts) |
| 32 | u32 | bm7_off | layout 1: == wsets_off + pad4(K * rows * 34); layout 0: 0 |
| 36 | u32 | scales_off | next section start |
| 40 | u32 | lenpred_off | LENPRED: == scales_off + 1544; else 0 |
| 44 | u32 | total_bytes | == size of the blob |
| 48 | u32 | index_kind | 1 = canonical structural index (only value accepted by v1 readers) |
| 52 | u32[3] | reserved | 0 |

### 3.2 Sections (in this order, contiguous, each zero-padded to a multiple of 4)
1. **index**: `u16 index[full_contexts]`, full context -> row, 0xFFFF = unreachable. Must equal the canonical index (3.3).
2. **wsets**: K sets, each `rows` CDFs of 17 u16 (`cum[0] = 0 < cum[1] < ... < cum[16] = 16384`), so every frequency is >= 1. Set k, row r at `wsets_off + (k * rows + r) * 34`.
3. **bm7** (layout 1 only): `u16 f0[3][7]` = frequency of "block max != 7" for matrix TYPE m and srel s, `1 <= f0 <= 16383`; the binary CDF is `[0, f0, 16384]` (symbol 0 = max != 7, symbol 1 = max == 7). 42 B, padded to 44. Per matrix type (not per set) because that is what was measured: the side cost 0.0155 bit/w [M] used `bm2_side[7,2]` per matrix type.
4. **scales**: `3 x 257 u16` cumulative CDFs, one per matrix type, exactly as WNM2 (1,542 B, padded to 1,544). Scale coding is unchanged: order-0, 1.066 bit/scale [M].
5. **lenpred** (LENPRED only): `u32 lp_w[K][7]` (expected coded bits of one 32-weight block of set k with srel s, INCLUDING the bm7 symbol when layout 1, in 1/256 bit; each <= 33 * 14 * 256 = 118,272); `i32 lp_wbias[K]` (per-stream bias in 1/256 bit, |bias| <= 2^24); `u32 lp_s[3]` (expected bits per scale symbol of matrix type m, 1/256 bit, <= 3,584). 32K + 12 B.

### 3.3 Canonical context index (normative)
Full context of a weight symbol (p = position in its block, 0..31, half = p >> 4; srel, pm, mx as in WNR2, section 5.3):
- layout 0: `full = (srel*9 + pm)*9 + mx` (WNR2's 567 contexts).
- layout 1: `full = ((srel*9 + pm)*9 + mx)*4 + bm7*2 + half`.

Reachable full contexts: the block start (pm = mx = 8, only at p = 0 so only half 0), or 0 <= pm <= mx <= 7 with, in layout 1, mx <= 6 when bm7 = 0 (a block whose max is not 7 contains no magnitude 7). Rows (`tri(pm, mx) = mx*(mx+1)/2 + pm`):
- layout 0: `row = srel*37 + (start ? 36 : tri(pm, mx))` -> 7 x 37 = **259 rows**.
- layout 1: per srel 130 rows in four groups (bm7, half): (0,0) offset 0, 29 rows; (0,1) offset 29, 28 rows; (1,0) offset 57, 37 rows; (1,1) offset 94, 36 rows. `row = srel*130 + goff[bm7][half] + (start ? ntri[bm7] : tri(pm, mx))`, `ntri = {28, 36}` -> **910 rows**.
- `index[full] = row` for reachable contexts, 0xFFFF otherwise. SHA-256 of the canonical index as u16 LE: layout 0 `e0aebac7d1b9b257737b92d1065bde74243f7f8f617a1619b077f6e6639d0738`, layout 1 `0634ddaa1af56ab246e8bb4dc9cef736cfa502657c2183e57c7db2f772da7e41` [M, `wnr3_ref.py`].
- Checked by `wnr3_ref.py` [M]: the index is a bijection onto 0..rows-1, and the closed form stays inside the table for EVERY pm, mx in 0..7 (max row 909 of 910 in layout 1, 257 of 259 in layout 0), including the invalid case bm7 = 0 with magnitude 7. So a GPU decoder may skip the per-symbol reachability test and check bm7 consistency once per block (5.3).
- Against the data [M, `lenres.json`]: all rows used by the 160-expert `c_bm2h` counts are inside the canonical 910 (0 outside); used rows per matrix type 589 / 803 / 623; 100 canonical rows are never used by any matrix. Unused rows cost no bits (they are never selected); they cost 100 x 32 B = 3.2 KB of shared memory per block versus a data-driven 810-row compaction, which would need a per-symbol index lookup (section 7.3). (810 = rows used by ANY matrix type, the relevant number because the K sets are pooled across matrix types; the "803 rows, 25 KB" quoted before is the largest single matrix type; neither is a structural bound, so data from another model could use rows outside them.)
- Side effect: layout 0 (feature a off) needs only 259 rows = 8,288 B of shared rows versus WNR2's 567 x 32 = 18,144 B, because WNR2 stores unreachable rows (pm > mx, half-empty start rows).

### 3.4 Validation (`wex_nr3_model_open`)
Reject unless: magic, version 1, precision 14, layout in {0,1} with the matching full_contexts/rows, 1 <= K <= 16, matrices 3, scale_symbols 256, no unknown feature bits, index_kind 1, every offset equal to its formula, total_bytes equal to the formula and to the buffer size, all padding and reserved bytes zero, index == canonical, every CDF strictly increasing from 0 to 16384, every f0 in [1, 16383], lp bounds (3.2).

### 3.5 Training (reference trainer, deterministic)
- Counts use exactly the decode contexts: per (expert, matrix) unit, `w[rows][16]` (rows = canonical rows), `bm7[7][2]` per matrix type, `scale[256]` per type.
- Sets: K-means on units, initialised by layer (`assign = layer*K // num_layers`), cost = sum counts x (-log2 quantized freq), at most 15 iterations, ties to the lower k; then `wex_nr2_quantize` per row (1 + floor(count*(16384-16)/total), largest remainders, ties to the lower symbol; empty rows near-uniform) [R §7]. This is the procedure behind the measured K = 2 / 4 numbers [M, `mxcombo_an.py`]. The trainer may fit sets on a deterministic subsample of experts; the encoder assigns every matrix.
- lenpred: `lp_w[k][s]` = round(256 x mean coded bits of a block of set k and srel s, including the bm7 symbol); `lp_s[m]` = round(256 x mean bits per scale); `lp_wbias[k]` chosen so the median weight-length residual is 0 on the training set.
- Memory note [I]: per-unit counts for all 7,680 x 3 units at 910 x 16 int32 are 1.34 GB; fine on 32 GB, or train on a subsample.

### 3.6 Sizes
`total = 64 + pad4(2*FC) + pad4(K*rows*34) + (layout1 ? 44 : 0) + 1544 + (LENPRED ? 32K + 12 : 0)` [I, computed]:

| layout | K | bytes (with LENPRED) | table_offset |
|---|---|---|---|
| 0 | 1 | 11,596 | 16,384 |
| 1 | 1 | 37,172 | 45,056 |
| 1 | 2 | 68,144 | 73,728 |
| 1 | 4 | 130,088 | 135,168 |
| 1 | 8 | 253,976 | 262,144 |
| 1 | 16 | 501,752 | 507,904 |

(WNM2: 59,408 B, table_offset 65,536 [R §5].) The blob is stored once per container, so its size is irrelevant for capacity.

### 3.7 Model id and registry
- `model_id` = FNV-1a 32 over the whole blob (offset basis 2166136261, prime 16777619; 0 mapped to 1), the same function as WNM2 [R §7]. Every record repeats it. The descriptor also holds the SHA-256 (checked once at open).
- Registries (engine `coli_wex_register_model`, helper `wex_cuda_register_model`) keep WNM2 and WNM3 blobs in the same 16-entry table, keyed by id, and remember the kind (by magic). A codec 4 record must name a WNM2 id, a codec 5 record a WNM3 id; the same id with different bytes stays a collision (-1) [R §9-10].

---

## 4. WNR3 record

### 4.1 Layout
`[0, 64)` header; `[64, 64 + LS)` length section; scale payload (SP bytes); weight payload (WP bytes); zero padding. `total_bytes == align4096(64 + LS + SP + WP)` and the padding is all zero (canonical, checked).

### 4.2 Header (64 B)

| off | type | field | rule |
|---|---|---|---|
| 0 | u32 | magic | 0x33524E57 'WNR3' |
| 4 | u32 | model_id | registered WNM3 id == WND3 model_id |
| 8 | u32 | flags | bit 0 BM7, bit 1 CZ, bit 2 LENC, bit 3 HASH; bits 4-31 = 0; must equal entry flags bits 8-11 |
| 12 | u32 | LS | length section bytes (4.3) |
| 16 | u32 | SP | scale payload bytes |
| 20 | u32 | WP | weight payload bytes |
| 24 | u32 | weight_chunk Cw | power of 2, 512..16384 |
| 28 | u32 | scale_chunk Cs | power of 2, 256..16384 |
| 32 | u8[3] | median[m] | lower median of matrix m's scales (`wex_nr2_median`) |
| 35 | u8[3] | set_id[m] | < K |
| 38 | u16 | owner_layer | == bind layer |
| 40 | u16 | owner_expert | == bind expert |
| 42 | u16 | reserved | 0 |
| 44 | u32 | cz_count | CZ: number of FP4 code-8 nibbles mapped to 0 (>= 1); not CZ: 0 |
| 48 | u32[3] | reserved | 0 |
| 60 | u32 | header_check | `H32(bytes 0..59, seed 0x33524E57)` |

Flag constraints: BM7 <=> model layout 1; LENC => model LENPRED; CZ => header flag 64 and cz_count >= 1. HASH has no model dependency. The header check is always present (cheap, catches header damage before any length or payload is trusted).

### 4.3 Length section
**LENC = 0** (WNR2 table): u16 lengths of the S scale streams then of the W weight streams in g order, zero-padded to a multiple of 4. `LS == pad4(2*(S+W))`.

**LENC = 1**: two parts, scale part then weight part, each laid out as:
```
u32 base_off[ng]        byte offset of the group's first stream, relative to the start of that part's payload
u8  width[ng]           bits per packed residual in the group, 0..17
    zero pad to a multiple of 4
u32 words[sum(width)]   group g's 32 residuals in width[g] words (32*w bits), LSB-first:
                        residual i occupies bits [i*w, i*w + w) of the group's little-endian bit string
```
- `ng` = ngS for the scale part, ngW for the weight part (section 1). A part's size is `4*ng + pad4(ng) + 4*sum(width)`, so it delimits itself; `LS` must equal the sum of both parts exactly.
- Residual of stream j: `z = zigzag(len_j - pred_j)` with `zigzag(r) = 2r` for r >= 0, `-2r - 1` otherwise; `width[g]` = bit length of the group's largest z (0 if all are 0). Values of the missing streams of a partial last group are 0.
- Decoders check: `width <= 17`; `base_off[first group of the part] == 0`; for every group `base_off[g] + sum(len in g) == base_off[g+1]`, and for the part's last group `== SP` (scale part) or `== WP` (weight part); every length in `[4, min(65535, 2n + 8)]` where n is the stream's output symbol count; unused residual slots are 0.
- Prediction (normative, integer, identical on CPU and GPU):
  - scale stream with n symbols, matrix m: `pred = 4 + ((n * lp_s[m] + 1024) >> 11)` (u64).
  - weight stream j of matrix m, set k = set_id[m], blocks b of the stream: `T = sum_b lp_w[k][srel_b] + lp_wbias[k]` (i64), `pred = 4 + ((max(0, T) + 1024) >> 11)`, where `srel_b = clamp(scale_b - median_m, -3, 3) + 3` from the DECODED scales.
  - `len = pred + unzigzag(z)`.
- Why base offsets per group: each warp computes its 32 lengths, does one exclusive warp scan and adds `base_off[g]`; no lane depends on another warp, so the offsets need no global prefix sum, no extra kernel and no host work. The stored base offsets cost 4 B per 32 streams (2,436 B on V4.1) and make every group self-checking (section 7.4).

### 4.4 Size bounds
- Stream length >= 4 and <= 2n + 8 (WNR2's bound 2n+8 [R §8]). With BM7 a weight stream of n symbols also carries n/32 binary symbols; the worst case 14 bits per symbol gives `ceil(14*(n + n/32)/8) + 4 = 1.805n + 4 < 2n + 8` [I]. For n = 16384: 29,572 <= 32,776 <= 65,535, so lengths still fit u16.

---

## 5. Codec

### 5.1 Scale streams
Scale stream g (matrix m) holds scales `[j*Cs, min((j+1)*Cs, scount_m))` of m in row-major order, coded order-0 with CDF `scales[m]`. Unchanged from WNR2 except the initial/final state rule (5.6).

### 5.2 Weight streams: symbol order
Weight stream j of matrix m holds weights `[j*Cw, min((j+1)*Cw, count_m))` of m, row-major, symbol 2i = low nibble of byte i [R §7]. In decode order, for every block of the stream:
1. if BM7: one binary symbol `bm7` with CDF `[0, f0[m][srel], 16384]`;
2. the block's 32 weight symbols, each with the CDF row `set[set_id[m]].rows[index[full]]`.

So the side information lives in the weight stream, immediately before the weights it describes; the decoder has it before the block's first weight, at the cost of one extra rANS step per 32 weights. The encoder computes `bm7 = (max(x & 7 over the block) == 7)` and encodes the whole symbol sequence backwards as in WNR2.

### 5.3 Weight contexts
At block start: `srel = clamp(scale - median_m, -3, 3) + 3`, `pm = 8`, `mx = 8`, p = 0. After decoding symbol x: `pm = x & 7`; `mx = (mx == 8) ? (x & 7) : max(mx, x & 7)`; p += 1; half = p >> 4. Full context and row as in 3.3. Per-record decoder checks:
- CPU reference: `index[full] == 0xFFFF` -> invalid (can only happen with bm7 = 0 and a decoded magnitude 7).
- every decoder: at block end, BM7 => `(mx == 7) == bm7`, else invalid. A valid encoder cannot produce a mismatch, so this is a free integrity check (and it is what makes the GPU's no-test closed form safe).
- CZ => no decoded weight symbol equals 8, else invalid.

### 5.4 Canonical zero (feature c)
- Encoder (`--canonical-zero`): every FP4 code 8 (-0) becomes 0 (+0) before counting and encoding; `cz_count` = number of mapped nibbles; the record gets CZ only if `cz_count >= 1` (a record without -0 is byte-exact and is written without CZ). Contexts are unaffected (8 & 7 == 0) [R §7].
- Decoded weights are the canonicalised source: equal in value, not in bytes. Every CZ record therefore needs the record flag, the entry bit 9 and the header flag 64; checkers compare it against the canonicalised source and check `cz_count` against the source's count of code 8.
- The 16-symbol alphabet is kept; a CZ-trained model gives symbol 8 the minimum frequency 1/16384, which costs log2(16384/16383) = 0.00009 bit per weight [I].
- Measured: +0 and -0 are 5.8255% and 5.8263% of weights; canonical zero with (a)+(b) gives 3.6544 bit/w held-out = -3.38% [M]. Production still needs the logprob check of the decision rules [R §1, §7].

### 5.5 Length prediction (feature d)
As in 4.3. The weight predictor uses srel only: bm7 is inside the weight streams and is not known when the offsets are needed. Measured on 40 held-out experts (model and predictor trained on the other 80) [M, `lenres.json`]:

| part | u16 today | entropy of residuals | held-out order-0 table (no flush, no escape cost) | grouped bit-packing G = 32 (width byte included) | + base offsets | WNR3 part |
|---|---|---|---|---|---|---|
| weight lengths (feature-a model) | 34,560 | 10,336 | 10,435 | 12,260 | 2,160 | 14,420 |
| scale lengths | 4,320 | 950 | 979 | 1,125 | 276 (+3 pad) | 1,404 |
| total | 38,880 | 11,286 | 11,414 | 13,385 | | **15,825** |

- Saving 23,056 B = **0.123% of the raw record** (the earlier estimate with ideal entropy coding was 0.148% [M]). G = 16 / 32 / 64 give 12,371 / 12,260 / 12,484 B for the weight part, so 32 is also the best group size.
- Residuals (weights, feature-a model): mean +0.06..+0.48 B, std 6.1..9.2 B, range -28..+418; 73-79% within +-7 B. Scales: std 3.1-4.5 B, range -14..+112 [M]. Stream bytes in this measurement are `round(bits/8) + 4` estimates (same convention as `mxstreams.py`), not real rANS outputs.

### 5.6 Integrity hash (feature e)
- **H32** = MurmurHash3_x86_32 (Appleby, public domain [S]): `c1 = 0xcc9e2d51`, `c2 = 0x1b873593`; per 4-byte little-endian block `k *= c1; k = rotl(k,15); k *= c2; h ^= k; h = rotl(h,13); h = h*5 + 0xe6546b64`; tail of 1-3 bytes `k = tail[0] | tail[1]<<8 | tail[2]<<16` mixed the same way without the h rotation; then `h ^= length_in_bytes`; `fmix32: h ^= h>>16; h *= 0x85ebca6b; h ^= h>>13; h *= 0xc2b2ae35; h ^= h>>16`. `wnr3_ref.py` reproduces the SMHasher verification value 0xB0F57EE3 listed for Murmur3A [S, M].
- **Stream hash input**: 16-B identity, then the stream's decoded output bytes exactly as they land in the slab (weight stream of n symbols: n/2 bytes, low nibble = even symbol; scale stream: n bytes). The bm7 symbols are not in the output; they are implied by it through the block-end check (5.3).
  ```
  ident[0] = model_id
  ident[1] = owner_layer << 16 | owner_expert
  ident[2] = g                                   (global stream index, section 1)
  ident[3] = (flags & 0xFF) | m << 8 | set_id[m] << 16 | median[m] << 24
  h23 = H32(ident || decoded_bytes, seed = 0x33524E57) & 0x7FFFFF
  ```
- **Rule**: HASH = 1 -> the encoder starts the stream at `x0 = L + h23` (in `[L, 2L)`, inside the legal `[L, 256L)`), and the decoder requires, after consuming exactly the stream's bytes, `final state == L + h23(decoded)`. HASH = 0 -> `x0 = L`, `final == L` (WNR2 rule). The initial-state check `[L, 256L)` is unchanged.
- **Cost**: x0 uniform in [L, 2L) costs E[log2(1 + u)] = (2 ln 2 - 1)/ln 2 = 0.557 bit = 0.0697 B per stream [I], i.e. 1,354 B per V4.1 record (0.0072% of raw); measured 0.067 B/stream on 300 real streams [M]. A 24-bit hash (x0 in [L, 3L)) would cost 0.935 bit/stream for a 2x smaller miss rate; not worth it [I].
- **What it adds** [M, 40 trials each, with CRC-32 instead of H32]: random single-bit flips were already caught 40/40 by today's rule; a VALID stream decoded at the wrong index was caught 0/40 by today's rule and 40/40 by the hash rule. With identity in the hash, a stream or record decoded under the wrong (layer, expert, index, model, set, median, flags) passes with probability 2^-23 per stream; a whole misplaced V4.1 record would need 19,440 independent collisions [I]. The detection rates must be re-measured with H32 (test plan).
- **Test vectors** [M, `wnr3_ref.out.json`]: `H32("", 0) = 0x00000000`; `H32("", 0x33524E57) = 0xE7F741F1`; `H32(bytes 00..0F, 0x33524E57) = 0x31CE0919`; weight stream (model 0xAB4261BD, layer 7, expert 191, g = 2165, flags 0x0F, m = 1, set 3, median 127, 1024 bytes `data[i] = (37i + 11) & 0x77`): h23 = 0x0C48D3, x0 = 0x008C48D3; scale stream (same record, g = 719, m = 0, set 0, median 127, bytes 7F 7E 80 7F 7D): h23 = 0x3C0999, x0 = 0x00BC0999.
- **Why not CRC32C**: x86 has a CRC32C instruction but NVIDIA GPUs have none, so the GPU would need table lookups (1-4 KB in the shared memory the CDF rows need) or a bitwise loop [I]. H32 is IMAD/shift/rotate only. Neither function keeps a "detects every burst" guarantee here: a payload error desynchronises the rANS decoder and changes the decoded bytes pseudo-randomly from that point on, and the check is truncated to 23 bits, so both give ~2^-23 [I].
- **Header check**: `header_check` (4.2) uses the same H32. Together with the stream hashes and the length checks, every byte of the record is covered by at least one check: header (header_check), length section (group continuity + stream decoding), payloads (stream structure + hash), padding (zero test) [I].

### 5.7 Reference encoder order
1. Canonicalise (CZ mode), compute medians, srel, bm7 per block.
2. Choose `set_id[m]` = argmin over k of the estimated cost (counts of m x -log2 of set k's quantized frequencies), ties to the lower k (the measured selection [M]); an exact "encode all K, keep the smallest" is allowed but costs K encodes.
3. For each stream in g order: forward pass for contexts and the H32 of its output; encode backwards from x0 (L or L + h23); flush the 4-B state.
4. Lengths: u16 table, or (LENC) predictions from the scales being encoded, residuals, groups, base offsets.
5. Header (header_check last), padding. If `total_bytes >= aligned_raw`, store raw (flags 0).
6. Verify with the CPU reference decoder before writing (as the W1 build does [R §8]).

---

## 6. Decoder order of operations (normative)

**Open (once per container)**
1. Header: magic WEX3, version 3, flags subset of 1|2|4|8|16|32|64, 32 without 8/16, 64 only with 32, bytes 56..63 and 120..4095 zero.
2. WND3: magic, bounds, model region inside [4096, table_offset), zero gap; read the blob; SHA-256 == descriptor; FNV-1a == model_id; `wex_nr3_model_open` (3.4); model_summary == blob fields.
3. Register the model (engine and helper registries).
4. Table: `wex_validate_table` as today, plus per entry: codec 5 rules (2.3) and the bind (2.4).
5. Policy: a flag-64 container is refused unless the canonical-zero opt-in is set (8.3).

**Per record** (CPU reference `wex_nr3_decode`, GPU helper; any failure -> status -1, destination untouched)
1. Entry rules (2.3); read `total_bytes` into the slab end.
2. Header: magic; `header_check`; model_id registered as WNM3 and == descriptor id; flags == entry bits 8-11; flag constraints (4.2); chunks; set_id < K; owner == bind; reserved == 0; cz_count rule; `64 + LS + SP + WP <= total_bytes` and `total_bytes == align4096(64 + LS + SP + WP)`; padding zero.
3. Geometry from the entry: count_m, scount_m, Ws_m, Ss_m, S, W, groups.
4. Length section, scale part: LENC = 0 -> u16 lengths (bounds; sums == SP and WP); LENC = 1 -> widths, part sizes, LS equality; scale lengths with the constant predictor; bounds; group continuity against SP.
5. Scale streams: decode; final-state rule (L, or L + h23).
6. LENC = 1: weight lengths from the decoded scales and lenpred; bounds; group continuity against WP.
7. Weight streams: bm7 symbol per block (BM7), contexts, block-end bm7 check, CZ check, final-state rule.
8. Commit the 6 regions only if no error (unchanged commit semantics [R §10]).

The only new dependency is step 6 on step 5 (weight offsets need decoded scales). WNR2 already runs the weight kernel after the scale kernel on the same stream because the weight contexts need the scales [R §10], so this adds no new synchronisation.

---

## 7. GPU helper and kernels

### 7.1 Scale kernel (`wex_cuda_nr3_scales`)
- As `wex_cuda_nr2_scales` (one thread per stream, 32 threads per block, direct tables in shared memory [R §10]) plus: an H32 state per lane (4 identity words, one word per 4 scales, tail, `h ^= 16 + n`, fmix32) and the final-state rule.
- LENC = 1: blocks are laid out per matrix (block = one 32-stream group of one matrix); each lane computes its own length with the constant predictor and its residual, one exclusive warp scan gives `offset = base_off[g] + prefix`, the last lane checks group continuity. The host no longer prefix-sums or uploads offsets for this record.

### 7.2 Weight kernel (`wex_cuda_nr3_weights`)
Changes per lane / block:
1. **Shared CDF rows**: the record's set `set_id[m]`, `rows x 32 B` (16 u16 starts per row, PRMT 16-way compare as today): layout 1 910 x 32 = **29,120 B** (WNR2: 18,144 B); layout 0 259 x 32 = 8,288 B. Copied from the device table (K sets) at block start: 29 KB from L2 per block [I: negligible next to a ~2048-step chain].
2. **bm7 / half state**: `bm7` is one register bit per lane, decoded at block start with `f0[m][srel]` (7 u16 per matrix type, in shared memory or registers); `half` comes from the existing in-block position counter (`p >> 4`), so no stored state. The group base `srel*130 + goff[bm7][half]` is recomputed at p = 0 and p = 16; per symbol `row = gbase + tri(pm, mx)` (2-3 integer ops, comparable to WNR2's `(srel*9 + pm)*9 + mx`), no index lookup and no extra dependent shared load.
3. **Block-start binary step**: one rANS step with a 2-way compare; +1 step per 32 symbols (<= +3.1% chain steps) [I].
4. **Block-end checks**: bm7 consistency (1 compare); CZ: "any nibble == 8" on each output word, `t = w ^ 0x88888888; ((t - 0x11111111) & ~t & 0x88888888) != 0` (4 ops per 8 symbols).
5. **Hash**: 4 H32 blocks per 32 symbols on the uint4 the kernel already stores (~24 integer ops per 32 symbols, independent of the rANS dependency chain), finalisation per stream, `final == L + h23` instead of `final == L`.
6. **LENC = 1 prologue**: the lane reads its stream's scales (Cw/32 = 64 bytes on V4.1, 4 x 16-B loads; the same scales the decode reads anyway), sums `lp_w[k][srel]`, adds its unpacked residual (base_off, width, the group's word offset = sum of earlier widths by a strided warp reduction over <= ngW bytes, 1-2 words, funnel shift), then one warp exclusive scan and the group-continuity check. About 300 instructions once per stream versus ~2048 x ~25 for the decode [I].
7. Errors set the record's error word as today; commit unchanged.

### 7.3 Occupancy and decode-cost risk
- Shared memory per 32-thread block [I, assuming 100 KB of shared memory per SM and 99 KB per block as on sm_86; to confirm on sm_86 and sm_120 with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, which also sees register limits]:

| kernel geometry | rows | bytes per block (rows + 64-word ring per lane) | blocks / SM | warps / SM |
|---|---|---|---|---|
| WNR2, 32 threads | 567 | 18,144 + 8,192 = 26,336 | 3 | 3 |
| WNR3 layout 1, 32 threads | 910 | 29,120 + 8,192 = 37,312 | 2 | 2 |
| WNR3 layout 1, 64 threads | 910 | 29,120 + 16,384 = 45,504 | 2 | 4 |
| WNR3 layout 1, 256 threads | 910 | 29,120 + 65,536 = 94,656 | 1 | 8 |
| WNR3 layout 0, 32 threads | 259 | 8,288 + 8,192 = 16,480 | 6 | 6 |

- **Main risk**: launched like WNR2 (32 threads per block), layout 1 drops resident warps per SM from 3 to 2. The kernel is latency-bound at ~200 cycles/symbol/lane [R §10], so throughput in a saturated batch would fall by up to a third. Wave arithmetic for one V4.1 record (540 weight blocks) on a 5090 (170 SMs): 510 resident slots for WNR2 and 340 for WNR3, both 2 waves; on a 3070 (46 SMs): 4 vs 6 waves [I].
- **Mitigation (recommended)**: launch 64 threads per block (2 warps share one table, each warp = one length group): 4 warps/SM, more than WNR2's 3. Larger blocks (up to 256 threads, 8 warps/SM) are possible; a smaller ring (32 words per lane) frees more. Kernel-only time per record must be measured at 1 and 8 concurrent records (bench_nr2_kernel / bench_nr2_rechunk style [R §12]).
- **Per-symbol cost** [I]: +1 binary step per 32 symbols (<= +3.1%), the hash and CZ test off the critical chain (~0-2%), the closed-form row with no extra load (~0). Expected total +3-6% at equal occupancy. A data-driven (non-canonical) index would add one dependent shared-memory load per symbol on the chain (~+10-15% [I]), which is why v1 forbids it.
- **K sets**: the device table grows to K x 29 KB (116 KB at K = 4); L2-resident on both GPUs [I].
- **Lengths**: no extra kernel launch (WDDM costs ~8.5 us per kernel launched via streams [R §12, wddm_graph]); the host stops uploading `(S+W+1) x 4 = 77,764 B` of offsets per V4.1 record.
- **Gate (proposal)**: WNR3 is acceptable if kernel-only time per record at 1 and 8 concurrent records is <= WNR2 + 5% on the same records and GPU. If layout 1 misses it even with 64-thread blocks, ship layout 0 (keeps b, c, d, e; -0.26/-0.28% from K sets on the base context [M]) and keep (a) for later.

### 7.4 What the record's checks catch on the GPU
Group continuity (each warp), bounds, width, final-state rule (with hash), bm7 consistency, CZ symbol test, header check (host), padding (host). Any one sets the error word; the commit kernel then writes nothing [R §10].

---

## 8. Engine, helper API, migration

### 8.1 Helper (`kaira_wex_cuda.dll`)
- ABI stays 1. New feature bit **16 NR3**: `wex_cuda_register_model` accepts WNM3 blobs; new entry points take the expected bind:
  ```c
  int wex_cuda_decode_to_host3(const WexEntry*, uint32_t bind, unsigned char *slab, size_t cap);
  int wex_cuda_decode_to_device3(const WexEntry*, uint32_t bind, const unsigned char *record, int device,
                                 unsigned char *const *dst_scale, unsigned char *const *dst_weight);
  int wex_cuda_decode_batch_to_device3(int n, const WexEntry *const*, const uint32_t *binds,
                                       const unsigned char *const*, int device,
                                       unsigned char *const *dst_scale, unsigned char *const *dst_weight, int *status);
  ```
  `bind = owner_layer << 16 | owner_expert`. Status semantics as today: 0 written, -1 invalid (untouched), 1 unavailable [R §10].
- The old entry points return 1 for codec 5 (they cannot check identity), so a caller never decodes WNR3 without a bind. Codecs 3 and 4 are unchanged.
- Device tables per (model, device): blob + K x rows x 32 B rows + scale direct tables (as WNR2: 3 x 16384 x 5 B) + bm7 + lenpred, ~0.5 MB at K = 4 [I].

### 8.2 Engine (deepseek_v4.c, wex_decode.c, expert_store_wex.c)
- Container open: header flag 32 -> read WND3, register WNM3, require registered id == descriptor id; if the helper lacks feature 16, print "[v4_wex] WNR3 container needs kaira_wex_cuda.dll with WNR3 support (helper feature 16) next to the engine" and refuse the container (same pattern as the WNR2 guard [R §11]). Give `expert_store_wex.c` the same guard (today it has none for WNR2 [R §11]).
- `coli_wex_decode_record` and `coli_wex_cuda_decode_*` gain the bind; the store computes binds at table load (2.4). Store-side `coli_v4_expert_store_wex_entry` accepts codec 5 only with feature 16.
- RAM cache slots are sized to the largest codec 3/4/5 record (unchanged rule [R §11]). Mirrors: unchanged (header page + table byte-identical).
- Restore the caller's CUDA device after helper calls (existing W2 note [R §3]); the new entry points should do it themselves.

### 8.3 Canonical-zero policy
- New env `COLI_WEX_ALLOW_CANONICAL_ZERO` (default 0). With 0, a flag-64 container is refused with "[v4_wex] value-exact (canonical-zero) container refused; set COLI_WEX_ALLOW_CANONICAL_ZERO=1 after the logprob check". The build report and `check_container` print the CZ record count and the total `cz_count`.

### 8.4 Compatibility matrix

| reader \ container | WEX1 / WEX2 | WEX3 + WNR2 (flag 16, codec 4) | WEX3 + WNR3 (flag 32, codec 5) |
|---|---|---|---|
| today's engine + helper (features 15) | OK | OK | refused at header (flag 32) or entry (codec 5); falls back to safetensors |
| new engine, helper without bit 16 | OK | OK | refused (guard) |
| new engine + helper with bit 16 | OK | OK | OK (CZ containers only with opt-in) |
| repack_v4_wex.py | OK | refused (flags > 15) [R §8] | refused |

- WNR2 containers stay valid; nothing converts them. A WNR3 container is a new build from the WEX1 source (same procedure and checks as the W1 build: retrain, encode, CPU verify every record, every 43rd GPU/CPU byte compare, readback, sha256 [R §8]).
- Disk: a full non-CZ V4.1 WNR3 container is ~129.3 GB [I]; at 17:11 on 2026-11-07 no drive had that free next to the existing files (E: 105.9 GB, U: 122.6 GB, D: 61.4 GB [R §2]), so a full build needs an explicit decision (e.g. replacing the WNR2 on E:). All tests below run on 64-record sample files.
- Tools: `wex_nr2_tools.dll` -> add NR3 exports (count, build model, encode, verify, check container) or a separate `wex_nr3_tools.dll`; `nr3lib.py` wire structs (record header `'<IIIIIIII3B3BHHHI3II'` = 64 B, WND3 `'<IIQII32s'` = 56 B); `check_container.c` learns codec 5 and reads CZ from the records; CPU reference in a new `wex_neural_rans3.h` so `wex_neural_rans.h` (WNR2) stays untouched.
- KAIRA-only rule applies: none of this goes upstream [R §1]. After changes, update KAIRA src first, then re-sync Wex-WRN2 [R header].

---

## 9. Expected sizes (V4.1, average record) [I from M]

Base: WNR2 measured average 16,914,680 B (header 32, lengths 38,880, scales 147,386.4, weights 16,726,327.8, padding 2,053.9) [R §12]. Deltas applied to the weight payload: -0.35% for (a)+(b) K = 4, -3.38% with (c) [M held-out, % of weight bits including flush]; lengths 15,825 B (5.5); hash +1,354 B; header 64 B; padding and scales unchanged.

| | WNR2 | WNR3 a+b+d+e | WNR3 a+b+c+d+e |
|---|---|---|---|
| header | 32 | 64 | 64 |
| length section | 38,880 | 15,825 | 15,825 |
| scales | 147,386 | 147,386 | 147,386 |
| weights | 16,726,328 | 16,667,786 | 16,160,978 |
| hash state cost | 0 | 1,354 | 1,354 |
| padding | 2,054 | 2,054 | 2,054 |
| **record** | **16,914,680** | **~16,834,470** | **~16,327,660** |
| of raw 18,800,640 | 0.8997 | 0.8954 | 0.8685 |
| vs WNR2 | | -0.47% | -3.47% |
| records per fixed tier vs WEX1 | +11.15% | +11.68% | +15.15% |
| container (7,680 records) | 129.9 GB | ~129.3 GB | ~125.4 GB |

Caveats: the context and K deltas come from ideal-cost estimates of 14-bit quantized tables on 160 experts (in-sample error of that method vs production: 3.7821 vs 3.781 bit/w [M]); the length numbers use estimated stream lengths; the CZ column assumes the length section does not change size. Each switch must be measured on real encoded records (test T9).

---

## 10. Test plan (mirrors the WNR2 tests [R §11])

**T1 codec round trips** (`test_wex_format.c test_nr3_codec`): 3 chunk pairs (512:256, 2048:512, 4096:2048) x all 16 combinations of BM7 / CZ / LENC / HASH (layout 1 for BM7, layout 0 otherwise) x K in {1, 2, 4}; synthetic matrices with: all-zero blocks (max 0, bm7 = 0), maxima below 6, maxima exactly 6 and 7, all -0 (CZ), uniform random nibbles (raw fallback expected), a scale count not a multiple of 4 (H32 tail), stream counts not multiples of 32 (partial groups), a matrix with a single stream, streams near the 2n+8 bound. CPU decode byte-exact (or canonical-exact for CZ).

**T2 corruption, each rejected with the slab untouched** (WNR2 has 14 cases; WNR3 >= 30): header byte flips (header_check); each flag bit; flags != entry bits; set_id = K and set_id changed to another valid set (header check recomputed: must fail on the streams); owner mismatch; model id of another registered model; CZ without header flag 64; CZ with cz_count 0; width 18; LS off by 4; nonzero unused residual; base_off[0] != 0; base_off off by one (continuity); a length of 3 and of 2n+9; flips in a scale payload, a weight payload, a residual word; initial state < L and >= 256L; truncated record; nonzero padding; crafted record with symbol 8 under CZ; crafted record with a wrong bm7 that is otherwise a valid stream; crafted stream with a valid x0 = L + h23 computed for another g.

**T3 misplaced data**: record of expert A decoded with bind B (header check passes) -> -1; same record with owner patched to B and header_check recomputed -> -1 from the stream hashes (proves the hash binds identity independently of the header); two equal-length streams swapped inside a matrix -> -1; DEDUP alias decoded with the owner's bind -> OK, with its own bind -> -1; re-measure the 40-trial table of `measured-v41.md` with H32 (single-bit flips, misplaced streams, HASH = 0 vs 1).

**T4 canonical zero**: CZ output == canonicalised source and differs from the source exactly at the code-8 nibbles; cz_count == source count; non-CZ records byte-exact; engine refuses a flag-64 container without the opt-in and opens it with it; `check_container` compares CZ records against the canonicalised source.

**T5 model**: WNM3 open rejects bad magic/version/precision/layout/K = 0 or 17/offsets/total/padding/index (one entry changed)/non-increasing CDF/f0 = 0 or 16384/unknown feature bits/index_kind 2; FNV id and SHA-256 mismatch at container open; registry: WNM2 and WNM3 in one process, same-id-different-bytes = -1.

**T6 container** (`test_expert_store_wex.c test_nr3_container`): model registered at open; exact decode; foreign model refused; codec 5 entries invalid without flag 32; flags 16|32 refused; entry bit 9 without header 64 refused; a WNR3 container opened by today's validator code is refused (keep a copy of the current validator in the test).

**T7 GPU** (`test_nr3_cuda`, `test_decode_batch` style): 30 records exact via host and device with buffers from another runtime; batch sizes 1, 2, 3, 4, 6, 8, 10, 16 byte-equal to the CPU reference; a corrupted record inside a batch -> -1 and untouched while the others are 0; the 70 malformed records of T2 rejected by both decoders with the same status; old entry points return 1 for codec 5; compute-sanitizer 0 errors; kernel at 32 and 64 threads per block gives identical output.

**T8 fuzzing**: >= 300 random mutations per feature combination (bit flips, byte runs, truncation, random header fields with a recomputed header_check, random length sections) -> 0 accepted; differential CPU vs GPU on every case (same status; identical output when valid).

**T9 sizes and speed**: on the 64-record V4.1 sample (`sample_records.py`, seed 20261107) encode with each switch alone and all together; report bytes per part against section 9; kernel-only ms/record at 1 and 8 concurrent records for WNR2 vs WNR3 (32 and 64 threads/block) on the same GPU, idle, against the gate in 7.3.

**T10 build checks** (full or sample build): all records CPU-verified; every 43rd GPU/CPU byte compare; slab readback; whole-file sha256; per-record rehash; CZ build: total cz_count == source count of code 8.

**T11 inference**: non-CZ WNR3 is byte-exact, so a short token-identity run (W3-style settings) is enough; CZ needs the logprob comparison before any production use [R §1].

---

## 11. Open choices

1. **Index**: stored-but-canonical (this draft) vs not storing it at all (4.5 KB saved once per container) vs a free index (index_kind 2, data-driven ~810 rows, one more dependent load per symbol). Recommendation: keep canonical in v1.
2. **bm7 placement**: in the weight stream (this draft) vs in the scale streams (would let the length predictor use bm7 but doubles the scale streams' steps, ~512 extra steps per 512-scale stream on the critical path before the weight kernel [I]) vs a raw bit plane (1 bit/block = 0.031 bit/w vs 0.0155 coded: +0.41% of weight bits, eats the gain [I]).
3. **bm7 tables**: per matrix type (measured) vs per set (unmeasured).
4. **Length coding**: base offsets per 32-stream group (this draft, 15.8 KB, no extra kernel) vs a global scan (13.4 KB, +1-2 kernels: ~8.5 us each via WDDM streams) vs rANS-coded residuals (~11.4 KB + flushes and escapes, sequential per residual stream) vs base offsets per 64 streams (saves ~1.2 KB, ties the format to 64-thread blocks).
5. **Scale lengths**: coded (this draft, 1.4 KB) vs kept u16 (4.3 KB, host offsets as today).
6. **Hash width**: 23 bits (x0 in [L, 2L), 0.557 bit/stream) vs 24 bits (0.935 bit/stream). Hash identity fields: whether flags/set/median belong in ident[3] (they are already covered by header_check).
7. **K**: 4 (measured -0.35% with a) vs 2 (-0.34%); cap 16 vs 8.
8. **Containers**: one model kind per container (this draft) vs mixed WNR2/WNR3 containers (needs two descriptors).
9. **Entry bits 8-11** mirroring the record flags (lets the engine count CZ records without reading them) vs bits 8-31 = 0.
10. **Kernel geometry**: 64 threads per block (recommended) vs larger; ring size.
11. **CZ alphabet**: 16 symbols (this draft; 0.00009 bit/w of waste) vs 15.

---

## 12. Checks that need the KAIRA sources (not done here)
1. `wex_validate_header` (v3) really rejects unknown flag bits (the reference only says "any of flags 1|2|4|8|16" [R §5]).
2. No code path reads `entry.flags >> 8` as scale_pages for codecs other than 1 (needed for 2.3).
3. The WNR2 weight kernel's real shared memory, register use and blocks/SM (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) on sm_86 and sm_120; the 2-vs-3 blocks/SM risk in 7.3 assumes 100 KB per SM and the 64-word ring.
4. Whether the scale kernel's blocks are per matrix (needed for 7.1 with LENC).
5. `wex_decode.c` registry: room to store the model kind next to the id.

---

## 13. Sources and evidence
- `D:\Desktop\WNR2-REFERENCE.md` sections 1-14 (read) [R].
- `...\research\wnr2-brief.md`, `...\research\measured-v41.md` (read); measurement scripts read in the temporary folder `<TMP>\wnr2\` (`mxcombo.py`, `mxcombo_an.py`, `mxstreams.py`, `rans_check.py`, `mxanalyze.py`, `mxcount.py`); results `<RESEARCH-PC>\v41-results\combo.json`, `streams.json`.
- Measured here: `<RESEARCH-PC>\team3\wnr3-spec\lenres.json` (structural rows; length residuals on 40 held-out experts: model and predictor trained on the 80 split-0 experts, evaluated on 40 of the 80 split-1 experts, run 2026-11-07, one process, ~3 min); `wnr3_ref.out.json` (H32 verification, vectors, index checks, length-part round trip).
- [S] MurmurHash3 source, https://raw.githubusercontent.com/aappleby/smhasher/master/src/MurmurHash3.cpp (constants, structure, public-domain statement); SMHasher verification table, https://raw.githubusercontent.com/aappleby/smhasher/master/src/main.cpp (Murmur3A 0xB0F57EE3).
