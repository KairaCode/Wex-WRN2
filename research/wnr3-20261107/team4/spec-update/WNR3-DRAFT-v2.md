# WNR3 draft specification (DRAFT 0.2, 2026-11-07, team4 / spec-update)

**Revises** team 3's DRAFT 0.1 (`<RESEARCH-PC>\team3\wnr3-spec\WNR3-DRAFT.md`, unmodified; byte copy kept next to this file as `WNR3-DRAFT.md`, sha256 086C4B56...844BED0). Sections that DRAFT 0.1 got right are kept, sometimes verbatim; everything that changed is listed in section 0.2.

**Status:** paper design. Nothing here is implemented in KAIRA C/CUDA. Two independent NumPy reference codecs (team 3 `ref-impl`, team 4 `ceil-ref`) and one CuPy/NVRTC GPU prototype (team 2 `gpu-proto`, extended by team 4 `ceil-gpu`) implement the parts marked [M] below. The KAIRA sources (`wex_format.h`, `wex_neural_rans.h`, `wex_cuda.cuh`, `wex_cuda_bridge.cu`, engine files) are NOT on this PC: every rule that depends on them is marked **OPEN-K** and collected in section 13.

**Evidence tags.** [M] measured on real data (source file given); [R] read in `D:\Desktop\WNR2-REFERENCE.md` (line given as REF:n); [C] read in code on this PC (path:line); [S] external source; [I] inferred or arithmetic (never a measurement).

**Source abbreviations** (all paths on this PC):

| tag | what | where |
|---|---|---|
| CR | real rANS records, held-out: models from the 80 even-indexed sample experts, test on 40 odd-indexed experts (one per layer, team 3's protocol), 18 configs x 40 records, all decoded from bytes alone | `<RESEARCH-PC>\team4\ceil-ref\results\heldout_summary.json`, `heldout_summary_g0.json` (constant anchor), `test_ceil.json` (fallback tests); code `<RESEARCH-PC>\team4\ceil-ref\wnr_common.py` |
| CG | GPU prototype, RTX 3070 sm_86, NVRTC, 8 experts, in-sample model | `<RESEARCH-PC>\team4\ceil-gpu\out\bench_final.json`, `best_final.txt`, `ratios_final.json`; `...\ceil-gpu\enc\encode_report*.json`; kernel `...\ceil-gpu\src\wnr2dec.cu` |
| RF | refutation study, 236 records (160 sample + 70 extra routed + 6 MTP) | `<RESEARCH-PC>\team4\ceil-refuter\out\` (`refute_an.json`, `anchor_an.json`, `edge_tests2.json`, `flip_ceil_big.json`, `signsym.json`, `e2e_real.json`) |
| CO | origin / generality study, 9 checkpoints | `<RESEARCH-PC>\team4\ceil-origin\` (`summary.json`, `upstream_v41_1017.json`, `other_*.json`, `sim_double.json`, `transfer_ds.json`) |
| MV | earlier measurements on the 160-expert sample | `...\research\measured-v41.md` (same folder as `wnr2-brief.md`) |
| RIC | consolidated research report (team summaries) | `...\research\WNR-RICERCA-2026-11-07.md` |
| T2A | team 2 corruption audit | `<RESEARCH-PC>\team2\audit\corrupt_audit.json`, `symperm_audit.json` |
| T2Z | team 2 canonical-zero kernel tests | `<RESEARCH-PC>\team2\zero-safety\out\run_full4_quick160.json`, `gpu_all160.json`, `mma_signedzero.json`; `<RESEARCH-PC>\team2\zero-refuter\out\zero_probe.json`, `gpu_probe.log`; kernel source map `<RESEARCH-PC>\team2\zero-safety\scripts\zs_kern.c:4-19` |
| T2G | team 2 GPU prototype table | `<RESEARCH-PC>\team2\gpu-proto\out\final_table.txt`, `signent.json` |
| T3S | team 3 expert statistics (srel anchor) | `<RESEARCH-PC>\team3\expert-stats\out\anchor_clusters.json` |
| ANS | team 2 start-state experiments | `<RESEARCH-PC>\team2\ans-prior-art\flushx.json` |

All record sizes are for one V4.1 record: raw 18,800,640 B = 35,389,440 weights + 1,105,920 scales, 17,280 weight streams (2048 weights) + 2,160 scale streams (512 scales) [R REF:7 brief, CR]. "% of raw" always means percent of 18,800,640 B.

---

## 0. Summary

### 0.1 Decisions

| id | feature | record flag | decision | measured effect per V4.1 record (CR, 40 held-out experts, real rANS bytes) | decoder cost |
|---|---|---|---|---|---|
| f | **ceil-form** (even blocks halved, scale + 1; inverse in the decoder) | CEIL (bit 4) + `ceil_mask` | **ADOPT** (new, first-class) | f alone vs WNR2: -331,162 B = **-1.761% of raw** (CI -1.948..-1.632), ratio 0.89943 -> 0.88182 [M CR] | nibble LUT on blocks with coded max 5: x0.995-0.999 (free) [M CG]; deferred scale write: +0.3% (8 records) .. +1.7% (1 matrix) median own-cost [M CG]; 0 B shared memory, 0-7 registers [M CG] |
| g | **fixed srel anchor** (constant per set and matrix type, stored in the record's anchor byte) | none | **ADOPT** as rule g0 (constant) | f+g0 vs f: -6,028 B (CI -16,967..-459) [M CR]; f+g0 = 0.88152 (-1.791%) [M CR g0]. The densest-pair rule of DRAFT 0.1 is a wash on ceil-form (-853 B, CI -16,952..+15,062) [M CR] | none (decoder unchanged, reads the byte) |
| d | coded stream-length table | LENC (bit 2) | **KEEP** | -25,791 B (-0.137%) with the ref-impl layout [M CR fgd vs fg]; about -23.3 KB (-0.124%) with this spec's GPU layout (+2,436 B group base offsets) [I] | as DRAFT 0.1 (warp scan, no extra kernel) |
| t | **raw tail seeding** in the rANS start state + 32-bit **record digest** | TAIL (bit 5) + DIGEST (bit 7) | **ADOPT as default** (resolves the 23-bit conflict, section 5.8) | tail: -40,199 B (-0.214%) vs no start-state use [M CR fgt vs fg]; tail instead of hash: -41,548 B (**-0.221%**) [M CR fgdt vs fgde]; digest: 0 B (reserved header word) | tail: extract 5 nibbles / 2 bytes from the final state; digest: per-lane H32 (same work as e) + one atomicXor per warp + one compare in the commit kernel [I, not measured] |
| e | per-stream 23-bit hash in x0 | HASH (bit 3) | **KEEP as alternative** (mutually exclusive with TAIL) | +1,352 B (+0.0072%) [M CR fge vs fg] | as DRAFT 0.1 |
| c | canonical zero (code 8 -> 0, value-exact) | CZ (bit 1) + header flag 64 | **KEEP, separate decision** (needs the 5090 bit test + per-step logits memcmp, section 5.6) | -514,848 B (**-2.738%**) on top of f+g [M CR fgc vs fg]; f+g0+c+d+e = 0.85282 (-4.661%) [M CR g0] | one nibble test per output word |
| s | sign bypass (8-symbol magnitude CDFs + raw sign plane) | SIGNPLANE (bit 6), model layout 2 | **OPTIONAL decode-speed layout** (not the v1 default) | bytes +1.03e-6 bit/w [M T2G signent.json]; CFD vs CF +120 B/record [M CG]; kernel best-vs-best -6.5..-11.4% vs A259, -4.3..-9.1% vs CF5 [M CG] | smaller table (4,160 vs 8,304 B smem) |
| a | block-max side symbol (3-valued on ceil-form) + half | (bit 0 reserved, must be 0) | **NOT ADOPTED** -> Appendix A.1 | +10,642 B (+0.057%), worse on 40/40 experts [M CR fga vs fg] | table 1,211-3,402 rows [I] |
| b | K table sets chosen by k-means | (none) | **NOT ADOPTED as a compression feature** -> Appendix A.2; the set mechanism is kept (K <= 16, v1 writes K = 2) only to hold the plain-form fallback set (and later MTP sets) | K=2 on top of a: -2,087 B; a+K2 on c+d+e: +8,585 B [M CR] | none per symbol |

**v1 default profile (byte-exact): CEIL | LENC | TAIL | DIGEST (flags 0xB4) with anchor rule g0.** Measured equivalent fg0dt = 16,506,880.0 B = **0.87800 of raw, -2.144% vs WNR2** (CI -2.334..-1.992), 40/40 byte-exact [M CR g0]; with this spec's layout (64-B header, group base offsets) about 16,509,300 B = 0.8781 (-2.13%) [I]. Capacity vs WEX1 +13.9% instead of +11.2% (WNR2 held-out 0.89943); container (7,680 records) about 126.8 GB instead of 129.9 GB [I].

**v1 value-exact profile: v1 + CZ (flags 0xB6)**, only after section 5.6's tests. Measured nearest: fg0cde 0.85282 (-4.661%) and fg0cd 0.85275 (-4.669%) [M CR g0]; fg0cdt (CZ + tail) was not run; expected about 15,994,400 B = 0.8507 (-4.87%) [I: fg0cd - 40,199 B tail + 2,452 B layout].

### 0.2 Changes from DRAFT 0.1

1. New feature **f (ceil-form)**: exact encoder rule, per-matrix gate, per-matrix fallback to a plain-form table set, decoder inverse and its position in the GPU kernel (sections 5.2, 6, 7.2).
2. New feature **g (anchor)**: the record byte `median[m]` becomes `anchor[m]`; the v1 encoder writes a constant per (set, matrix type) stored in the model (rule g0); decoders are unchanged (section 5.3).
3. **a and b re-decided** on top of f+g: both cost bytes or gain nothing, a also needs a 4-13x larger table: moved to Appendix A. Model layout 1 (BM7H) is reserved and rejected; record flag bit 0 is reserved.
4. **Start-state bits**: TAIL + DIGEST is the default; HASH remains as a mutually exclusive alternative (section 5.8).
5. **Hash domain**: every hash/digest covers the CODED domain (the symbols the streams carry, before the ceil inverse), so the scale kernel can hash without knowing which blocks were transformed (section 5.1).
6. **Sign bypass** added as optional model layout 2 + SIGNPLANE record flag (section 5.9).
7. WNM3 tables are per matrix type inside each set (as WNR2 and as measured), not pooled across types as DRAFT 0.1 assumed. Every set carries its own scale CDFs, a `form` byte (plain or ceil) and `anchor0[3]`.
8. LENC predictor replaced by the one measured on real ceil-form records (25 scale bins per type, per-record fitted slope), still with DRAFT 0.1's GPU-friendly per-group base offsets.
9. Record header re-laid (64 B kept): `ceil_mask`, `record_digest`; owner fields moved by 2 B.
10. Expected sizes replaced by real-record measurements (section 10); canonical-zero safety evidence, tooling breakages and the required 5090 tests written out (section 5.6); new test plan and ordered rollout (sections 11-12).

---

## 1. Scope, terms, constants

- Scope: one record = one expert = 3 matrices m = 0 gate w1, 1 down w2, 2 up w3 (unchanged matrix order, unchanged DECODED slab layout: scales of w1, w2, w3, then packed FP4 weights of w1, w2, w3) [R REF:184].
- rANS is exactly WNR2's [R REF:206-208, 228, 240]: 32-bit state, `L = 2^23`, precision 14 bits (total 16384), byte renormalisation, 4-B big-endian initial state in `[L, 256L)`, every symbol frequency >= 1, encoder `x_max = ((L >> 14) << 8) * freq`. Decoder step: `slot = x & 16383`; symbol s with `cum[s] <= slot < cum[s+1]`; `x = freq[s]*(x >> 14) + slot - cum[s]`; renormalise.
- `align4096(n)`, `pad4(n)`: n rounded up to a multiple of 4096 / 4. Integers little-endian unless stated (initial rANS state big-endian, as WNR2).
- Block = 32 consecutive FP4 nibbles of a row with one UE8M0 scale. `count_m % 32 == 0` is enforced by `wex_entry_raw_sizes` [R REF:184]; streams never cross a matrix [R REF:205]; with Cw a power of 2 >= 512 every weight stream holds whole blocks [R REF:209] (checked on all 708 matrices of 236 records [M RF refute_count.py:222]).
- Magnitude of a nibble q: `q & 7`; sign: `q >> 3`. FP4 E2M1 magnitudes 0..7 = values 0, 0.5, 1, 1.5, 2, 3, 4, 6.
- **Coded domain**: the scale bytes `cs` and nibbles `cq` that the record's streams (and, with TAIL, the start states; with SIGNPLANE, the sign plane) carry. **Output domain**: the decoded slab. Output = `T_m^-1(coded)` per matrix (identity when ceil is off for m); with CZ the output equals the canonicalised source (section 5.6).
- Chunks `Cw` (512..16384) and `Cs` (256..16384), streams `Ws_m = ceil(count_m / Cw)`, `Ss_m = ceil(scount_m / Cs)`, global stream index g (scale streams of m0..m2, then weight streams of m0..m2), groups of 32 streams per matrix and kind: unchanged from DRAFT 0.1 section 1. V4.1: ngS = 69, ngW = 540.

Constants (DRAFT 0.1 values kept unless marked new):

| name | value | meaning |
|---|---|---|
| WEX_HEADER_NEURAL_RANS3 | 32 | header flag: WNR3 records + a WNM3 model before the table |
| WEX_HEADER_CANONICAL_ZERO | 64 | header flag: some records may be value-exact |
| WEX_CODEC_NEURAL_RANS3 | 5 | entry codec |
| WEX_NR3_RECORD_MAGIC / MODEL_MAGIC / DESCRIPTOR_MAGIC | 0x33524E57 / 0x334D4E57 / 0x33444E57 | 'WNR3' / 'WNM3' / 'WND3' |
| WEX_NR3_DESCRIPTOR_OFFSET | 64 | |
| WEX_NR3_HEADER_BYTES | 64 | |
| WEX_NR3_MAX_MODEL_BYTES | 1,048,576 | |
| WEX_NR3_MAX_SETS | 16 | |
| WEX_NR3_GROUP | 32 | streams per length group |
| WEX_NR3_HASH_SEED | 0x33524E57 | H32 seed |
| WEX_NR3_HASH_MASK | 0x7FFFFF | 23 bits (HASH) |
| WEX_NR3_TAIL_W (new) | 5 | weight-stream tail symbols (4 bits each in layout 0, 3 in layout 2) |
| WEX_NR3_TAIL_S (new) | 2 | scale-stream tail bytes |
| WEX_NR3_LEN_BINS (new) | 25 | length-predictor scale bins, dbin = clamp(cs - anchor, -12, 12) + 12 |
| WEX_CUDA_FEATURE_NR3 | 16 | helper feature bit |

---

## 2. Container (WEX3 version 3) - unchanged rules

DRAFT 0.1 section 2 stays normative (header page with WND3 descriptor at byte 64, header flags 32 and 64, entry codec 5, record placement at the slab end, DEDUP owner/bind rule, why no new version or magic). Changes:

- **Header flags**: unchanged. Ceil-form, tail seeding, digest and sign plane are byte-exact record features and need NO header flag. The WNR3 builder writes 37 (SCALE_TILES|DEDUP|NEURAL_RANS3), or 101 with canonical zero. 32 requires version 3 and excludes 8/16; 64 requires 32.
- **Entry flags for codec 5**: bits 0-7 = 5; **bits 8-15 mirror record flag bits 0-7** (DRAFT 0.1: bits 8-11); bits 16-31 = 0. Same caveat as DRAFT 0.1: no code path may read `flags >> 8` as scale_pages for codec 5 (**OPEN-K**, section 13 item 2).
- `wex_validate_entry` for codec 5: DRAFT 0.1 rules 1, 2, 5, 6, plus: bit 8 (reserved) = 0; bit 14 (SIGNPLANE) set <=> WND3 layout == 2; bit 10 (LENC) => WND3 features has LENPRED; the minimum size bound uses the record's minimum stream count as in DRAFT 0.1.
- WND3 `model_summary` = `layout | K << 8 | features << 16` (unchanged meaning; layout now 0 or 2).
- Old readers fail closed exactly as in DRAFT 0.1 section 2.5 (unknown header flag 32, unknown codec 5; `repack_v4_wex.py` accepts v3 flags only up to 15 [R REF:303]); the first point is **OPEN-K** (section 13 item 1).
- Model region: v1 builder K = 2, layout 0, LENPRED -> 57,736 B -> `table_offset` 65,536, the same as WNR2's [R REF:190] (section 3.5).

---

## 3. WNM3 model blob (revised)

### 3.1 Header (64 B)

| off | type | field | rule |
|---|---|---|---|
| 0 | u32 | magic | 0x334D4E57 'WNM3' |
| 4 | u16 | version | 1 (DRAFT 0.1's WNM3 was never built; this layout supersedes it) |
| 6 | u16 | precision | 14 |
| 8 | u16 | layout | 0 BASE16 (16-symbol rows); 2 SIGN8 (8-symbol magnitude rows, records carry a sign plane); 1 reserved (DRAFT 0.1 BM7H, rejected) |
| 10 | u16 | K | number of sets, 1..16 |
| 12 | u16 | full_contexts | 567 |
| 14 | u16 | rows | 259 |
| 16 | u16 | matrices | 3 |
| 18 | u16 | alphabet | 16 (layout 0) or 8 (layout 2) |
| 20 | u32 | features | bit 0 LENPRED; other bits 0 |
| 24 | u32 | index_off | 64 |
| 28 | u32 | sets_off | index_off + pad4(2 * 567) = 1,200 |
| 32 | u32 | set_bytes | pad4(8 + 3 * rows * (alphabet + 1) * 2 + 3 * 257 * 2) |
| 36 | u32 | lenpred_off | LENPRED: sets_off + K * set_bytes; else 0 |
| 40 | u32 | total_bytes | size of the blob |
| 44 | u32 | index_kind | 1 (canonical structural index; only value accepted) |
| 48 | u32[4] | reserved | 0 |

### 3.2 Sections
1. **index** `u16[567]`, full context -> row (0xFFFF unreachable), must equal the canonical layout-0 index of DRAFT 0.1 section 3.3: `row = srel*37 + (start ? 36 : tri(pm, mx))`, 259 rows, SHA-256 of the u16-LE index `e0aebac7d1b9b257737b92d1065bde74243f7f8f617a1619b077f6e6639d0738` [M, team 3 `wnr3_ref.py`]. Ceil-form needs no new rows: coded blocks have max 0..7 like plain blocks, and pm <= mx always holds [I]. Layout 2 uses the same index (contexts use only `q & 7`).
2. **sets**: K sets at `sets_off + k * set_bytes`, each:

| off in set | type | field | rule |
|---|---|---|---|
| 0 | u8 | form | 0 = plain (tables for untransformed data), 1 = ceil (tables for ceil-form data) |
| 1 | u8 | role | 0 routed, 1 MTP (informational; other values rejected) |
| 2 | u8[3] | anchor0[t] | training srel anchor per matrix type t (section 5.3) |
| 5 | u8[3] | reserved | 0 |
| 8 | u16 | weight CDFs `[3][rows][alphabet+1]` | per matrix type t, per row: `cum[0] = 0 < ... < cum[alphabet] = 16384` |
| 8 + 3*rows*(alphabet+1)*2 | u16 | scale CDFs `[3][257]` | as WNM2 [R REF:232] |

3. **lenpred** (LENPRED only): `u32 etab[K][3][25]`, expected coded bits of one 32-weight block of set k, matrix type t and scale bin dbin, in 1/256 bit, each <= 32 * 14 * 256 = 114,688.

Set selection: record field `set_id[m]` (< K) selects set k for matrix m, and the decoder uses table `sets[k].weights[m]`, `sets[k].scales[m]`, `etab[k][m]`. Only the selected table of one matrix is ever in shared memory (section 7.3).

### 3.3 Validation (`wex_nr3_model_open`)
Reject unless: magic, version 1, precision 14, layout in {0, 2} with alphabet 16 / 8, full_contexts 567, rows 259, 1 <= K <= 16, matrices 3, no unknown feature bits, index_kind 1, every offset and `set_bytes` equal to its formula, `total_bytes` equal to the formula and to the buffer size, padding/reserved zero, index == canonical, every CDF strictly increasing from 0 to 16384, every `form` in {0, 1}, every `role` in {0, 1}, every `etab` <= 114,688.

### 3.4 Training (reference trainer, deterministic)
- Counts use exactly the decode contexts in the coded domain of each set's form.
- **v1 sets** (builder default when the container's even-block share is >= 1%): set 0 form 1 (ceil), trained on `T(training matrices)` (section 5.2) of every training matrix that passes the gate; set 1 form 0 (plain), trained on the same training matrices untransformed. Below 1% even share (PTQ checkpoints: 0.0034-0.0049% [M CO other_amd_r1.json, other_rh_glm53.json]) the builder writes K = 1, form 0 and never sets CEIL [I policy: the transform then gains ~0, measured -0.0001..-0.0004% of weight bits [M CO]].
- Quantisation per row: `wex_nr2_quantize` (1 + floor(count*(16384-n)/total), largest remainders, ties to the lower symbol, empty rows near-uniform) [R REF:237].
- `anchor0[t]`: section 5.3. `etab`: round(256 x mean coded bits of a block of (set, type, dbin)) on the training data.
- MTP sets (role 1) only if the container stores MTP records (**open**, section 13.2).

### 3.5 Sizes [I, computed from 3.1-3.2]

| layout | K | LENPRED | bytes | table_offset |
|---|---|---|---|---|
| 0 | 1 | yes | 29,468 | 36,864 |
| 0 | 2 (v1) | yes | 57,736 | 65,536 |
| 2 | 2 | yes | 32,872 | 36,864 |

(WNM2: 59,408 B, table_offset 65,536 [R REF:190].) The blob is stored once per container. Model id = FNV-1a 32 over the blob (0 -> 1), registry rules as DRAFT 0.1 section 3.7.

---

## 4. WNR3 record (revised)

### 4.1 Layout
`[0, 64)` header; `[64, 64 + LS)` length section; scale payload (SP B); sign plane (SG B, SIGNPLANE only: `SG = sum_m count_m / 8`, 4,423,680 B on V4.1); weight payload (WP B); zero padding. `total_bytes == align4096(64 + LS + SP + SG + WP)`, padding all zero.

### 4.2 Header (64 B)

| off | type | field | rule |
|---|---|---|---|
| 0 | u32 | magic | 'WNR3' |
| 4 | u32 | model_id | registered WNM3 id == WND3 model_id |
| 8 | u32 | flags | bit 0 reserved (0); 1 CZ; 2 LENC; 3 HASH; 4 CEIL; 5 TAIL; 6 SIGNPLANE; 7 DIGEST; bits 8-31 = 0; == entry flags bits 8-15 |
| 12 | u32 | LS | length section bytes |
| 16 | u32 | SP | scale payload bytes |
| 20 | u32 | WP | weight payload bytes |
| 24 | u32 | Cw | power of 2, 512..16384 |
| 28 | u32 | Cs | power of 2, 256..16384 |
| 32 | u8[3] | anchor[m] | srel anchor of matrix m (was `median[m]`; section 5.3) |
| 35 | u8[3] | set_id[m] | < K |
| 38 | u8 | ceil_mask | bit m set = matrix m is in ceil form; bits 3-7 = 0 |
| 39 | u8 | reserved | 0 |
| 40 | u16 | owner_layer | == bind layer |
| 42 | u16 | owner_expert | == bind expert |
| 44 | u32 | cz_count | CZ: number of code-8 nibbles mapped to 0 (>= 1); else 0 |
| 48 | u32 | record_digest | DIGEST: section 5.8; else 0 |
| 52 | u32[2] | reserved | 0 |
| 60 | u32 | header_check | `H32(bytes 0..59, seed 0x33524E57)` (DRAFT 0.1) |

Wire struct: `'<8I8B2H5I'` = 64 B.

**Flag constraints** (all checked by every decoder):
- bit 0 == 0.
- CEIL <=> `ceil_mask != 0`; for every m: `((ceil_mask >> m) & 1) == sets[set_id[m]].form`.
- HASH and TAIL never both. TAIL => DIGEST.
- SIGNPLANE <=> model layout 2; SIGNPLANE => (HASH or DIGEST); SIGNPLANE => not CZ (section 5.9).
- CZ => header flag 64 and `cz_count >= 1`.
- LENC => model LENPRED.
- DIGEST = 0 => `record_digest == 0`.

v1 builder flags: 0xB4 (CEIL|LENC|TAIL|DIGEST) if at least one matrix passes the ceil gate, else 0xA4; with CZ 0xB6 / 0xA6.

### 4.3 Length section
- **LENC = 0**: u16 lengths of the S scale streams then of the W weight streams in g order, zero-padded to a multiple of 4 (WNR2 table).
- **LENC = 1**: `u16 sbase[3]`, `u16 alpha[3]` (12 B), then the scale part, then the weight part, each laid out as in DRAFT 0.1 section 4.3 (`u32 base_off[ng]`, `u8 width[ng]`, pad4, `u32 words[sum(width)]`, residual i of a group at bits `[i*w, i*w+w)` LSB-first, zigzag). Checks as DRAFT 0.1 (width <= 17, `base_off[0] == 0`, group continuity, part totals == SP / WP, unused slots 0, every length in `[4, min(65535, 2n + 8)]` with n = coded symbols of the stream).
- **Prediction** (integer, identical on CPU and GPU; the predictor measured on real records [C `<RESEARCH-PC>\team4\ceil-ref\wnr_common.py:573-599, 771-783`]):
  - scale stream of matrix m: `pred = sbase[m]` (the encoder writes the median scale-stream length of m).
  - weight stream j of matrix m, set k = `set_id[m]`: `Q = sum over the stream's blocks b of etab[k][m][dbin_b]`, `dbin_b = clamp(cs_b - anchor[m], -12, 12) + 12` on the CODED scales; `pred = 4 + ((Q * alpha[m] + 2^25) >> 26)` (u64). The encoder fits `alpha[m]` in 0..65535 to minimise the sum of |residual| (coarse search step 4 within +-64 of the least-squares value, then +-4) [C `wnr_common.py:582-599`].
  - `len = pred + unzigzag(z)`.
- Measured (ref-impl layout: widths + packed residuals, no base offsets): 38,880 -> 13,121 B mean on fg0dt, residual std 3.80 B (scales) / 7.20 B (weights) [M CR]. This spec adds `4 * (ngS + ngW)` = 2,436 B of base offsets (V4.1) so each warp computes its offsets alone (DRAFT 0.1 section 4.3 rationale): about 15,570 B [I].

### 4.4 Size bounds
Unchanged from DRAFT 0.1 for layout 0 without side symbols: a stream of n coded symbols is <= ceil(14n/8) + 4 < 2n + 8 [I].

---

## 5. Codec

### 5.1 Coded domain, hash input, identity
- All integrity data (HASH h23, DIGEST, header_check) is computed on the CODED domain. Reason: the scale kernel decodes scale streams before any weight block is known, so it cannot know which scales will be decremented; hashing coded scales keeps the scale kernel independent of ceil-form [I]. Hashing the coded domain was run end to end on 17 real records (decode exact, inverse byte-exact) [M RF e2e_real.json].
- Per-stream hash input: 16-B identity then the stream's coded bytes (weight stream: n/2 bytes of coded nibbles, low nibble = even symbol, including tail symbols and, with SIGNPLANE, the sign bits as bit 3 of each nibble; scale stream: n coded bytes):
  ```
  ident[0] = model_id
  ident[1] = owner_layer << 16 | owner_expert
  ident[2] = g | ((ceil_mask >> m) & 1) << 31        (g = global stream index)
  ident[3] = (flags & 0xFF) | m << 8 | set_id[m] << 16 | anchor[m] << 24
  h_g = H32(ident || coded_bytes, seed 0x33524E57)   (MurmurHash3_x86_32, DRAFT 0.1 section 5.6)
  ```
- H32 and its verification value 0xB0F57EE3 are unchanged (DRAFT 0.1 section 5.6 [S, M]). DRAFT 0.1's stream test vectors remain valid as H32 function vectors; record-level vectors with the new ident/flags must be regenerated (**open**, test T5).

### 5.2 Ceil-form (feature f)

**Observation** [M MV, RF]: in all 260,997,120 blocks checked (160 sample + 70 extra routed experts over all 40 layers + 6 MTP experts) the block max magnitude is 6 or 7, never below; 30.65% of routed blocks (sample 30.647%, extra 30.646%; w1/w2/w3 30.31/31.12/30.51%; per expert 22.8-31.9%) and 22.15% of MTP blocks have max 7 and no magnitude 1 or 3 [M RF refute_an.json, MV]. Under i.i.d. magnitudes the share would be 1.9e-4 [M MV]. The artefact is already in DeepSeek's upstream checkpoint (8 REAP-pruned upstream experts 27.8-31.6%, all 13 upstream items 29.415%, kept experts byte-identical to the sample) and in every QAT-native MXFP4 checkpoint measured (V4-Flash 25.98%, V4-Pro 23.89%, gpt-oss-20b 21.94%, gpt-oss-120b 21.36%, AMD Quark gpt-oss-120b 21.50%), but not in PTQ checkpoints (0.0034-0.0049%) [M CO]. Mechanism (sources read, export path inferred): DeepSeek's `fast_round_scale` and triton_kernels' default ROUND_UP are ceil rules (s = 2^ceil(log2(amax/6))), so a first pass yields block max 3.0/4.0/6.0 x s; re-quantising with ceil or OCP floor moves exactly the max-3.0 blocks to s/2 with every code doubled, value-exact [C/S CO, sim_double.json]. OCP MX v1.0 section 6.3 specifies floor but allows other algorithms [S CO]. So T below rewrites each such block into the form the ceil rule produced: same values, cheaper to code.

**Definitions** for one block (nibbles q_0..q_31, scale s, magnitudes `a_j = q_j & 7`, `M = max a_j`):
- `even` := `M == 7` and no `a_j` in {1, 3} (so every `a_j` is in {0, 2, 4, 5, 6, 7}).
- `H` (halve): 0->0, 2->1, 4->2, 5->3, 6->4, 7->5.  `D` (double): `D(a) = a + min(a, 2)`: 0->0, 1->2, 2->4, 3->5, 4->6, 5->7. `D(H(a)) = a` on {0,2,4,5,6,7}.

**Gate (per matrix m, encoder)**: matrix m may be coded in ceil form iff **no block of m (after CZ, if CZ is on) has M == 5**. This is the exact bijection condition [M RF edge_tests2.json: exhaustive over all 65,536 4-nibble blocks x scales {0, 1, 120, 253, 254, 255}: 0 collisions and 0 round-trip failures over 328,800 pairs without max-5 blocks; 32,208 collisions when max-5 blocks are allowed]. Blocks with M in 0..4 (including all-zero blocks) are legal and pass through unchanged. (The two measured codecs used a stricter record-level gate "no block with M < 6 and no even block with s >= 254" [C `wnr_common.py:217-265`]; on V4.1 both gates give identical records because 0 blocks have M < 6 and the largest scale of an even block is 124 routed / 125 MTP [M RF].)

**Encoder transform T_m** (only if the gate passes for m):
```
for every block b of matrix m:
    if even(b) and s_b <= 254:  q_j' = (q_j & 8) | H(q_j & 7)   for j = 0..31;   s_b' = s_b + 1
    else:                       q_j' = q_j;                      s_b' = s_b
```
An even block with s_b = 255 is left unchanged (it keeps M = 7, so the decoder leaves it alone; this removes the only overflow without a fallback [M RF edge_tests2.json: 0 overflow, 0 collisions]). Coded scale 255 can occur (from s_b = 254) and is a legal symbol; the coded form is never dequantised.

**Decoder inverse T_m^-1** (only if bit m of `ceil_mask` is set):
```
for every decoded block b of matrix m (coded nibbles q', coded scale cs_b):
    M' = max(q_j' & 7)
    if M' == 5:  if cs_b == 0: INVALID
                 q_j = (q_j' & 8) | D(q_j' & 7);   s_b = cs_b - 1
    else:        q_j = q_j';  s_b = cs_b
```
Why it is exact [I, with the exhaustive check above]: transformed blocks have coded max H(7) = 5; untransformed blocks of a gated matrix have max in {0..4, 6, 7}; so `M' == 5` selects exactly the transformed blocks, and D, `- 1` undo H, `+ 1`. The sign bit is never touched (sign plane unchanged on 708/708 matrices [M RF]). The `cs_b == 0` test can only fire on corrupted data (a transformed scale is >= 1) and closes the SWAR-borrow hazard noted in the prototype (`wnr2dec.cu:208-211` uses a plain u32 subtraction) [C CG].

**Per-matrix fallback**: a matrix that fails the gate is coded plain: `ceil_mask` bit m = 0 and `set_id[m]` = a set with form 0. Fallback must NOT use the ceil-trained tables: a plain record coded with the ceil model cost +397,312 B vs the plain WNR2 model on the same expert (17,285,120 vs 16,887,808 B) [M CR test_ceil.json], +11.7..+12.7% of raw in the estimate [M RF]. Expected fallbacks on V4.1: 0 of 7,680 records (0 of 230 routed records checked; 95% upper bound 100 records by the rule of three) [M/I RF]. A record with no gated matrix has CEIL = 0.

**Where the encoder checks**: the build report prints per record `ceil_mask`, number of transformed blocks per matrix, number of max-5 blocks (gate), number of even blocks with scale 255 (skipped); the container report prints the fallback count. The CPU verify of every record (section 11 T10) covers T^-1.

**Measured gain** (real records, CR, 40 held-out experts, 0 fallbacks, 40/40 byte-exact):
- f alone: 16,578,764.8 B = 0.88182 (WNR2 16,909,926.4 = 0.89943): **-1.761% of raw**, CI -1.948..-1.632, median expert -1.656%; weights 3.7801 -> 3.7080 bit/w, scales 1.0610 -> 0.9755 bit/scale (both incl. flush) [M CR].
- Real minus code-length estimate: 5.1 B (CI -16.6..+26.2) [M CR]; the quoted -1.81% (MV) is the all-160 / odd-80 figure (-1.803% / -1.846%) of the same estimate [M CR]. Real flush is ~3.5 B/stream, cancels in deltas [M CR, T2A].
- In-sample GPU-prototype records (8 experts): 0.90227 -> 0.88155 (-2.07% of raw) [M CG encode_report.json].
- Attribution [M RF anchor_an.json, estimate P2]: about 0.27 points of the f-vs-WNR2 gain is the srel-anchor fix that f brings implicitly (the ceil-form median rarely flips: 2/690 matrices vs 43/690 in plain form); with fixed anchors on both forms the ceil gain is -1.59% of raw (P2/P4), -1.60% (P3). A non-ceil model with a 3-class side stream reaches 97% of that, but needs 3,402 rows and a class stream; ceil-form keeps the 259-row table and no extra stream [M RF].
- Transfer: with a V4.1-trained model, the cost of reusing it on V4-Flash drops from +4.19% (plain) to +0.30% (ceil) [M CO transfer_ds.json].

### 5.3 srel anchor (feature g)
- Plain form: the lower median flips 120 <-> 121 in 27 of 480 sample matrices (5.6%), shifting every srel label of the matrix; fixing the anchor to the lower value of the densest adjacent pair gives 3.78243 -> 3.77290 bit/w held-out (-0.25%) with one model [M T3S anchor_clusters.json] (team 3 quotes -0.257%).
- On ceil-form both rules flip once among the 120 test matrices: the median gives 120 instead of 121 on L24_E058 w1 (costs 217,350 B on that record); the densest pair gives 121 on L39_E156 w2, a near tie (pair sums 349,985 vs 350,076; costs 204,800-212,992 B by config). Net: densest pair vs median -853 B, CI -16,952..+15,062 [M CR].
- **Rule g0 (v1)**: `anchor[m] = sets[set_id[m]].anchor0[m]`, where the trainer sets `anchor0[t]` = the mode, over the set's training matrices of type t (coded domain of the set's form), of the densest-pair lower value `argmax_v (h[v] + h[v+1])` (first on ties). On V4.1 this is 120 for every type in both forms (ceil: 240/240 training matrices; plain: pair = 120 in 480/480 sample matrices) [M CR anchor_diag.py, T3S anchor_census]. Measured with 120: fg0 0.88152 (-1.791%, CI -1.992..-1.640), g0 vs f -6,028 B (CI -16,967..-459) [M CR g0].
- Decoders do not check `anchor[m]` against `anchor0` or against the decoded scales: the byte stays an encoder choice, so a later encoder may align per matrix (e.g. argmin of estimated cost over anchor0 +-1; not measured). WNR2 stores the byte per record and the documented decoders read it [R REF:200, 222-225]; whether any KAIRA decoder or checker re-derives it as the lower median of the decoded scales is **OPEN-K** (section 13.1 item 10): such a check would reject g0 records.
- Label offset does not matter on V4.1: the refuter's estimate gives the same 3.7081 bit/w for a fixed ceil anchor and for "pair(original) + 1" (= 121) [M RF]. It matters for MTP (median anchors make MTP layer 0 worse, +0.14% of raw; pair(original)+1 gains -1.25/-1.53/-1.40% on the 3 MTP layers) [M RF]: whether the mode rule is right for MTP is **open**.

### 5.4 Scale streams
Scale stream g of matrix m holds coded scales `cs` `[j*Cs, min((j+1)*Cs, scount_m))` in row-major order, order-0 with `sets[set_id[m]].scales[m]`. With TAIL the last `kt = min(2, n)` coded scales are not rANS-coded (section 5.8).

### 5.5 Weight streams and contexts
Weight stream j of matrix m holds coded nibbles `[j*Cw, min((j+1)*Cw, count_m))`, row-major, symbol 2i = low nibble of byte i [R REF:221]. Layout 0: the coded symbol is the 4-bit nibble; layout 2: the 3-bit magnitude (sign in the sign plane). Context: at block start `srel = clamp(cs_b - anchor[m], -3, 3) + 3`, `pm = mx = 8`; after symbol x: `pm = x & 7`, `mx = max(mx, x & 7)` (8 = none); `row = srel*37 + (start ? 36 : tri(pm, mx))` [R REF:222-224; DRAFT 0.1 3.3]. Contexts use the CODED scale and CODED nibbles. With TAIL the last `kw = min(5, n)` coded symbols of the stream are not rANS-coded and need no context. Per-record checks: CZ => no coded nibble (rANS-decoded or tail) equals 8; ceil checks as in 5.2.

### 5.6 Canonical zero (feature c, value-exact)

**Rule** (DRAFT 0.1 5.4, unchanged): with `--canonical-zero` the encoder maps every code 8 (-0) to 0 (+0) before counting, the gate, T and encoding; `cz_count` = mapped nibbles; CZ only if `cz_count >= 1`. T commutes with it (T keeps magnitude 0 and the sign bit): 0 failures on 708 matrices and on the exhaustive 4-nibble test [M RF]. Output = canonicalised source, equal in value, not in bytes.

**Size** [M CR]: c on top of f+g: -514,848 B (-2.738% of raw, CI -515,456..-514,249); fg0c 0.85411, fg0cd 0.85275, fg0cde 0.85282 (-4.661%, 40/40 value-exact). -0 is 5.8263% and +0 5.8255% of V4.1 weights [M MV] (the brief's "12.8%" is the V4 pilot's share of all zeros, RIC section 9); the sign of a zero is a fair coin worth 0.1165 bit/w.

**Kernel-level argument** (team 2; code read in the colibri dev tree, line ranges listed in `<RESEARCH-PC>\team2\zero-safety\scripts\zs_kern.c:4-19`; KAIRA's fork NOT checked, **OPEN-K**):
- Inputs: most CPU arms decode code 8 to +0.0f or int8 0 (`deepseek_v4.c:20123-20129` coli_e2m1_decode; `quant.h:1284`, `:1370`, `:1392`), so for them canonicalisation does not even change the decoded operands. The scalar `matmul_mxfp4` arm (`quant.h:1266-1267` mx4_lut), the default CUDA FP32 kernels (`backend_cuda_dsv4.cu:99` f4(), `:2250-2313` v4f_dot_fp4) and the Vulkan shader (`shaders/chain_moe_gv.comp:97-114`) decode -0.0f [C T2Z, RIC section 2].
- Arithmetic [I, IEEE 754 round-to-nearest]: every accumulator starts at +0; adding a zero of either sign to a nonzero value returns it unchanged, and (+0) + (-0) = +0, so a sum never differs between -0 and +0 weight operands; products with finite activations are zeros either way. The only possible difference is the sign of an exactly-zero output when an accumulator is seeded with the first product instead of +0; the negative-control kernel K12 (Vulkan first-term emulation) shows exactly that (256 sign-only mismatches on synthetic zero rows) while every real kernel shows none [M T2Z].
- **Measured, original vs canonicalised records, 0 differing output bits**:
  - CPU kernels copied verbatim (AVX2, scalar, FMA, int8 IDOT, rows16, batch, CUDA and Vulkan emulations): 350,208 outputs per kernel (14 kernels incl. negative control, 4 experts, 9 activation sets incl. all -0, all +0, denormals, inf, NaN, 1e30); 3,112,960 outputs per kernel for 6 kernels over all 160 experts; 4,608 synthetic zero-row outputs per kernel; positive control (one flipped weight) detected in every kernel [M T2Z run_full4_quick160.json].
  - colibri CUDA kernels on the RTX 3070 (sm_86, NVRTC 12.9): 10 kernels (mv_fp4_grouped<1,2,4,8>, mv_fp4_expert_major<4> T4, mv<4>, mv_rows<4,4>, mv_rows<4,16>, mm_batch<4> T4, v4f_dot_fp4): 14,008,320 outputs per kernel (56,033,280 for the 4-token ones), 0 value, 0 sign-only, 0 NaN-payload mismatches, 0 nondeterministic reruns; 184,320 synthetic zero-row outputs per kernel; positive control fires [M T2Z gpu_all160.json]. sm_86 mma f16/bf16 tensor-core tiles: 0 differing D on 20,000 random tiles; all-zero A with random zero signs gives D = +0 in 2,560,000/2,560,000 [M T2Z mma_signedzero.json]; tc_fp16_gemm_bitdiff 0 [M T2Z gpu_probe.log].
  - Data: 0 all-zero 32-blocks and 0 all-zero rows in w1 of the 40 experts probed (14,745,600 blocks); 13 all-zero groups of 8 [M T2Z zero_probe.json].
- **Untested**: the sm_120 tensor-core paths (DeepGEMM sm120 FP4; cuBLASLt MXFP8, where code 8 becomes E4M3 0x80), Vulkan on other drivers [RIC section 2]. colibri's top-k, argmax and Engram treat -0 as +0 [RIC section 2, team 2].

**Required before any CZ use (proposal from team 2; it would replace the logprob check of REF:150, REF:250 - user decision)**:
1. **RTX 5090 bit test**: for every expert-matmul kernel KAIRA can dispatch on sm_120 (default CUDA FP32 kernels, DeepGEMM sm120 FP4, cuBLASLt MXFP8, Vulkan if enabled), run original vs canonicalised slabs with identical activations (the 9 adversarial sets above + real activations), memcmp the outputs: 0 differing bytes required; positive and negative controls must fire. Which kernels KAIRA dispatches is **OPEN-K**.
2. **Per-step logits memcmp**: same prompts, same deterministic settings, byte-exact container vs CZ container; memcmp the logits buffer at every decode step (e.g. the 4 W3-style prompts x >= 256 tokens): 0 differing bytes at every step.
If both give 0, CZ is bit-identical by construction on that machine and build.

**Tooling that breaks with CZ** (team 2, RIC section 2; all outside the arithmetic):
- the V4.1 deploy gate: `sample_records.py` + `test_decode_batch` compare decoded slabs byte for byte with the WEX1 slab [R REF:331-334, 348-349] -> must compare against Z(WEX1) for CZ records and require that differences sit exactly on the code-8 nibbles and number `cz_count`;
- exact-decode tests (`test_nr2_cuda`, `test_expert_store_wex`) -> canonical-exact variants;
- whole-file sha256 and `.verified` markers of the W3 procedure [R REF:42-44, 82, 139, 216] -> a CZ container has its own sha256; the builder must also record sha256 of the canonicalised source slabs so a readback can be verified;
- mirror pairing (header page + table byte-identical [R REF:468]) -> only byte copies of the same CZ container pair;
- **irreversibility**: a CZ container cannot recreate the original bytes. Deleting the WEX1 container or the safetensors after building a CZ container is irreversible. Rule: keep WEX1 (and the safetensors) until tests 1-2 passed and the decision is recorded. (`check_container` already takes `canonical_zero 0|1` [R REF:285]; byte-exact WNR3 incl. ceil-form needs none of this.)
- Engine policy unchanged from DRAFT 0.1 8.3: `COLI_WEX_ALLOW_CANONICAL_ZERO` (default 0) refuses a flag-64 container.

### 5.7 Length coding (feature d)
Section 4.3. Measured on ceil-form records: -25,791 B (CI -26,080..-25,483), 40/40 experts improved; with CZ the table is 13,366 B (weight residual std 8.11) [M CR]. The predictor reads coded scales only, so it is computed before any ceil fix-up.

### 5.8 Start-state bits: TAIL + DIGEST (default) vs HASH

The 23 free bits of `x0` in `[L, 2L)` can carry either a per-stream hash (HASH, DRAFT 0.1 feature e) or raw data (TAIL). Measured and inferred properties:

| option | what x0 carries | bytes per V4.1 record | per-stream final-state check | silent single-bit flips (sign flips) | identity binding |
|---|---|---|---|---|---|
| today (WNR2) | nothing, x0 = L | 0 | `x == L` | accepted: w1 31/12,000 = 0.26%, w2 62/12,000 = 0.52% [M T2A corrupt_audit.json]; 392/69,120 = 0.57% [M T2A symperm_audit.json]; on ceil-form w1 0.342%, w2 0.579% (24,000 flips each) [M RF flip_ceil_big.json]; every accepted case is one weight with its sign flipped (f[s] == f[s^8], CDF starts 8192 apart) [M T2A, RF] | none (but a valid stream at a wrong index with real contexts is caught 24,000/24,000 [M T2A test H]) |
| HASH | `L + (h_g & 0x7FFFFF)` | **+1,352 B** (+0.0072%) [M CR fge vs fg] | `x == L + h23(decoded)` | closed: 104/104 detected [M team 2 audit, MV]; 0/9,000 accepted [M ans-prior-art, MV]; residual 2^-23 per corrupted stream [I] | per stream |
| TAIL alone | `L + sum_j tail_j << (b*j)`, 5 nibbles (b = 4) / 2 bytes (b = 8) | **-40,199 B** (-0.214%) [M CR fgt vs fg] | window `L <= x < L + 2^20` (weights) / `2^16` (scales): accepts ~2^-11 / ~2^-15 of random end states vs ~2^-31 for `x == L` [I] | stays open (state unchanged by a sign flip) [I]; rate not measured | none |
| **TAIL + DIGEST** (v1) | tail as above; 32-bit record digest in header word 48 | **-41,548 B vs HASH (-0.221%)** [M CR fgdt vs fgde, CI -41,618..-41,495; with g0: 0.87800 vs 0.88021]; digest 0 B (reserved word of the fixed 64-B header) | window as above | closed: the digest covers every decoded coded symbol; residual 2^-32 per corrupted record [I] | per record (ident in every h_g) |

**Recommendation: TAIL + DIGEST.** It is 41.5 KB (0.22% of raw) smaller per record than HASH, its residual false-accept rate for a corrupted record (2^-32) is lower than HASH's per-stream 2^-23, and it closes the sign-flip hole that today's check leaves open (0.26-0.58% of random single-bit flips, each silently flipping one weight's sign). What it gives up: localisation of the bad stream (irrelevant: the commit is all-or-nothing per record [R REF:431]) and some per-lane simplicity (a device-wide XOR and a compare in the commit). TAIL without DIGEST is forbidden (it is strictly weaker than today). HASH stays available for builds that want per-stream localisation (e.g. debugging); never both HASH and TAIL.

**TAIL definition** [C `wnr_common.py:62, 398-405, 675-676, 731-733`]: for a weight stream of n coded symbols, `kw = min(5, n)`, the last kw symbols `y_0..y_{kw-1}` (stream order) are not coded; `x0 = L + sum_j y_j << (b*j)` with b = 4 (layout 0) or 3 (layout 2, magnitudes; signs stay in the sign plane; layout-2 tail not measured); the encoder encodes the first n - kw symbols backwards from x0. The decoder decodes n - kw symbols, requires exact length consumption and `0 <= x - L < 2^(b*kw)`, and reads `y_j = ((x - L) >> (b*j)) & (2^b - 1)`. Scale streams: `ks = min(2, n)`, b = 8. Cost: the start state rises from L to at most L + 2^20, costing < 0.18 bit per stream [I]; the measured net saving is ~2.07 B per stream [M CR].

**DIGEST definition**: `record_digest = XOR over all S + W streams g of h_g` (full 32-bit H32 of section 5.1). XOR is order-independent, so lanes can fold it with warp shuffles and one `atomicXor` per warp into a per-record word that is zeroed together with the error word [I]. Swapped or misplaced streams change it because g and the identity are in every h_g [I]. Header damage is caught by `header_check` (which covers `record_digest`).

Other start-state uses measured but not adopted: Appendix A.4-A.6.

### 5.9 Sign-plane layout (feature s, optional, decode speed)
- Model layout 2 + record flag SIGNPLANE: rANS codes the 3-bit magnitude with 8-symbol CDFs (same 259 rows, same contexts); the sign of every weight is stored raw, one u32 per 32-block (bit j = sign of nibble j), fixed size, no lengths, as a separate plane between the scale and weight payloads (team 2 variant D [C `<RESEARCH-PC>\team2\gpu-proto\src\wnr2fmt.py:19-20, 192`]).
- Bytes: the sign given magnitude and context carries 0.9999990 bit, so the bypass costs 1.03e-6 bit/w [M T2G signent.json]; CFD 16,573,814 vs CF 16,573,694 B/record [M CG encode_report.json].
- Decode time on the RTX 3070 (best config per variant, final run, 41 paired rounds) [M CG best_final.txt]: 1 matrix A259 300.0 / D 280.6 us (-6.5%), CF5 304.1 / CFD5 276.5 (-9.1%); 1 record A259 590.8 / D 523.3 (-11.4%), CF5 565.2 / CFD5 540.7 (-4.3%); 8 records A259 4,171.8 / D 3,835.9 (-8.1%), CF5 4,185.1 / CFD5 3,848.2 (-8.1%). At the same launch config D/A259 is 0.935 (m1, nt32/lb), 0.708 (1 record, nt64/out4), 0.812 (8 records, nt32/out4) [M CG]. The layout needs the staged-store path: at nt32 with plain stores D took 7,663.6 us per 8 records, 1.31x A [M T2G final_table.txt:13]. Shared memory: 4,160 B (layout 2) vs 8,304 B (layout 0) per block [M CG].
- Two chains per stream (DB2/CFDB2) are faster again (CFDB2 3,307.5 us per 8 records, 413 us/record) but cost +0.36% of the record [M CG]: Appendix A.3.
- Constraints: SIGNPLANE => (DIGEST or HASH), because the plain sign bits are protected by no rANS invariant [RIC section 5]; SIGNPLANE => not CZ in v1: with a fixed plane the zero signs are still stored, which gives back c's whole 0.1165 bit/w [I]. T and T^-1 act on magnitudes only (CFD5/CFDB2 byte-exact on the GPU [M CG]).
- Decision: not the v1 default. On the PCIe path the decode is roughly free and bytes are what count [R brief]; adopt layout 2 only if a compressed VRAM tier needs decode throughput, after a 5090 measurement.

### 5.10 Reference encoder order
1. Load the source record (WEX1 slab); CZ mode: canonicalise.
2. Per matrix: gate (count max-5 blocks); choose `set_id[m]` (ceil set if the gate passes and the model has one, else the plain set); `ceil_mask`; apply T to gated matrices.
3. `anchor[m] = anchor0` of the chosen set (g0).
4. Contexts on coded data; per stream: forward pass for contexts and h_g; x0 (L, L + h23, or L + tail); encode backwards; flush 4 B.
5. Lengths (u16 table or LENC with fitted `sbase`, `alpha`); DIGEST = XOR of h_g.
6. Header (header_check last), padding. If `total_bytes >= aligned_raw`, store raw (flags 0).
7. Verify with the CPU reference decoder (coded domain, then T^-1, then byte compare with the source, or canonical compare for CZ) before writing [R REF:289].

---

## 6. Decoder order of operations (normative)

**Open (once per container)**: DRAFT 0.1 section 6 steps 1-5, with layout in {0, 2} and the set checks of 3.3.

**Per record** (CPU reference `wex_nr3_decode` and GPU helper; any failure -> status -1, destination untouched):
1. Entry rules (section 2); read `total_bytes` into the slab end.
2. Header: magic; `header_check`; model id; flags == entry bits 8-15; all flag constraints of 4.2 (incl. set form vs `ceil_mask`); chunks; `set_id < K`; owner == bind; reserved == 0; cz_count rule; size equalities; padding zero.
3. Geometry from the entry.
4. Length section, scale part (u16 table, or LENC with `sbase`); bounds; group continuity.
5. **Scale streams -> coded scales `cs`** (into a coded-scale buffer, never into the output); TAIL bytes from the final state; final-state rule (`x == L`, `x == L + h23`, or tail window); h_g into the digest accumulator.
6. LENC: weight lengths from `cs` (coded domain); bounds; continuity.
7. **Weight streams**, per block: contexts from `cs` and coded nibbles; decode 32 coded symbols (the stream's last block: 27 + 5 tail symbols with TAIL, available only after step 8's final state); CZ check; h_g over the coded words; **then** T^-1 for gated matrices: `M' == 5` -> D on the nibbles, final scale `cs_b - 1` (`cs_b == 0` -> invalid); else final scale = `cs_b`. Weights are written in output form; final scales go to the OUTPUT scale region, separate from `cs`.
8. Per stream: final-state rule; h_g into the digest.
9. DIGEST: accumulated XOR == `record_digest`.
10. Commit the 6 regions (output scales + weights) only if no error and the digest matches.

Order invariants: integrity checks run on the coded domain (steps 5, 7-9); T^-1 never feeds back into contexts, lengths or hashes; the scale fix-up of block b happens after block b's coded max is known and never before step 5 finished. The only dependencies are step 6 on step 5 and step 7 on step 5, as in WNR2 (the weight kernel already runs after the scale kernel because contexts need scales [R REF:435]).

---

## 7. GPU helper and kernels

### 7.1 Scale kernel
As `wex_cuda_nr2_scales` [R REF:425] plus: writes CODED scales to a coded-scale scratch buffer (`sum_m scount_m` B = 1,105,920 B on V4.1); per-lane H32; TAIL extraction; final-state rule; warp XOR + atomicXor of h_g (DIGEST); LENC scale part as DRAFT 0.1 7.1. Measured prototype scale-stream decode: ~45 us per record, identical for plain and ceil-form scale streams (orig nt32 m3 46.1 us, ceil 45.3 us) [M CG].

### 7.2 Weight kernel: where the ceil inverse sits
Per lane (one stream), relative to KAIRA's kernel (one thread per stream, 32-thread blocks per matrix, shared CDF rows, cp.async ring, one uint4 store per 32 symbols [R REF:426-427]; exact code positions **OPEN-K**):
1. Unchanged decode chain: contexts read `cs` (via `__ldg` today), 16-way (layout 0) or 8-way (layout 2) compare, renormalisation. Ceil-form adds nothing to the dependency chain.
2. **After the 32nd symbol of a block, before the block's uint4 store**: the block max is the running `mx` (already tracked for contexts; for the last block with TAIL, include the 5 tail nibbles); if the matrix is gated and `mx == 5`: apply `D` to the 8 nibbles of each of the 4 u32 register words with SWAR `f(m) = m + min(m, 2)` (sign bit kept) and set bit b of a per-lane block mask. Prototype: `wnr2dec.cu:183-188`, applied at `:310-315` [C CG]; cost CF3/CF0 median 0.995-0.999 (free) [M CG]. h_g is computed on the words before D.
3. **Final scales, deferred**: at stream end (or every 16 blocks for Cw > 2048) the lane writes its blocks' final scales (`cs_b - mask_b`, per-byte subtraction such as `__vsub4`, never a plain u32 subtraction) with aligned 16-B stores into the OUTPUT scale region. Every stream's output scales must land there: gated streams write all their scale words (full write) or the commit copies the unchanged words from `cs` and the lane writes only the changed ones (mask form); non-gated matrices are a plain copy of `cs` (by the lane or by the commit). Measured on the 3070 (own-cost vs the same streams with the inverse off, min/median/max over 8 configs): mask form CF5 m1 0.976/1.017/1.034, 1 record 0.986/1.007/1.022, 8 records 0.986/1.003/1.015; full 64-B write CF6 the same (8-record median 1.003); the literal per-block 1-byte store CF is expensive (1 record 1.012/1.157/1.205, 8 records 1.015/1.059/1.213) and is NOT allowed [M CG]. CF5 vs A259 at the same config: median 1.014 (m1), 1.003 (1 record), 0.988 (8 records) [M CG].
4. **No in-place patch of the coded scales**: the coded scales are a `const __restrict__` / `__ldg` input; patching them in place is undefined behaviour and makes a retry decode with wrong srel [C RF on `team2\gpu-proto\src\wnr2dec.cu:182, 237, 244`; C CG `wnr2dec.cu:218-223`]. The output region must be distinct from `cs`; this also keeps decodes idempotent.
5. Optional fusion (not implemented, not timed): the weight kernel writes only a per-block bitmap (138,240 B per V4.1 record) and the commit kernel, which already copies every scale byte [R REF:431], writes `cs - bit`. Saves the 1.05 MiB coded-scale copy per lane; to measure on the 5090.
6. TAIL: after n - 5 steps, check the window, extract tail nibbles, complete the last block (step 2), then hash and store.
7. DIGEST: finalise h_g, warp XOR, one atomicXor per warp; the commit kernel compares the per-record word with `record_digest` (batch path: one digest word per record next to the error words in page-locked scratch [R REF:446]; **OPEN-K**).
8. SIGNPLANE: read the block's u32 sign word, merge sign bits into the 4 output words before D (D keeps bit 3).

Prototype correctness: byte-exact against the ORIGINAL safetensors weight and scale bytes for all non-diagnostic variants: 352 checks of 138,240 weight streams + 138,240 scale groups each, 8 launch configs, single and multi-stream launches and the full scale -> weight pipeline, 0 bad streams; negative controls fail as expected (inverse off: 138,240/138,240 weight streams wrong; weight-only inverse: all 138,240 scale groups wrong) [M CG bench_final.json].

### 7.3 Shared-memory and register budget (the a/b decision input)

| table (one matrix, per block) | rows | bytes | fits the ~99 KB/block budget [I, DRAFT 0.1 7.3 assumption] | measured |
|---|---|---|---|---|
| WNR2 today (567 rows, 32 B/row) | 567 | 18,144 | yes | A: 18,160 B smem [M T2G] |
| **v1 layout 0** (f+g, any K) | 259 | 8,288 | yes | 8,304 B smem; ceil inverse +0 B, regs +0..7 (nt32/out4 73 -> 80, nt128/256 out4 40 -> 44, occupancy unchanged) [M CG] |
| v1 layout 2 (sign plane) | 259 | 4,144 | yes | 4,160 B smem [M CG] |
| a on ceil-form, uncompacted (3-valued side x half) | 3,402 | 108,864 | **no** | blob 349,046 B [M CR] |
| a on ceil-form, structural compaction (max 5/6/7 groups x half + start) | 1,211 | 38,752 | yes, 4.7x v1 [I] | a 33,184-B table (team 2 variant C) ran 1.9-2.3x slower at 32 threads, about A at its best config [M T2G final_table.txt:11, 23, 35] |
| b (K sets) | 259 per set | 8,288 (only the selected set) | yes | device table K x 3 x 8,288 B [I] |

KAIRA adds a 64-word cp.async ring per lane (8,192 B per 32 lanes) [R REF:427]: v1 layout 0 at 32 threads = 16,480 B per block, 6 blocks/SM by DRAFT 0.1's arithmetic [I]; to confirm with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on sm_120 (**OPEN-K**). Extra device memory: coded-scale buffer 1.05 MiB per lane, 8.4 MiB per batch-calling thread with 8 lanes on top of ~287 MiB [R REF:416] [I].

### 7.4 What the GPU catches
Group continuity, bounds, width, final-state rule (with HASH or the TAIL window), CZ symbol test, set-form check (host), `cs_b == 0` in a transformed block, digest compare (commit kernel), header check and padding (host). Any one sets the error word; the commit writes nothing [R REF:431].

### 7.5 5090 transfer caveats (from CG, inferred)
One V4.1 record is 540 warps: ~11.7 warps/SM on the 3070's 46 SMs but ~3.2 on the 5090's 170 SMs [R REF:432], i.e. the latency-bound regime of the prototype's 1-matrix case, where CF5 cost +1.7% (median) [M/I CG]. L2 size, store behaviour, sm_120 register allocation and KAIRA's own kernel (cp.async ring, 567-row CTX 0 layout today, separate 88-KB-smem scale kernel, commit kernel) differ; the prototype's absolute times (A259 ~520-590 us/record) do not transfer (KAIRA: 0.2 ms/record kernel-only on the 5090 [R brief]). Bytes transfer unchanged; on the PCIe path (0.28 ms/record, PCIe-bound [R brief]) -2.1% of bytes shortens uploads by ~2% [I].

---

## 8. Feature interaction matrix (ceil-form vs everything)

| other feature | interaction | rule | evidence |
|---|---|---|---|
| g anchor | anchors are computed on coded-domain scales of the set's form; ceil-form makes the median nearly stable (2/690 flips vs 43/690 plain) | g0: constant `anchor0` per set and type; fallback matrices use the plain set's anchor0 | [M CR, RF] |
| d LENC | predictor uses coded scales (before fix-up) and per-set `etab` trained on coded data | section 4.3 | -25,791 B on ceil-form [M CR] |
| t TAIL | tail symbols are coded symbols of the stream's last block; T^-1 of that block waits for the tail | 6 step 7, 7.2 item 6 | -40,199 B on ceil-form, 40/40 byte-exact [M CR] |
| DIGEST / e HASH | computed on coded symbols; ident carries the ceil bit of m and the flags | 5.1 | hash +1,352 B on ceil-form [M CR]; coded-domain hash e2e 17/17 [M RF] |
| c CZ | canonicalise before gate and T; commutes; CZ check valid in both domains | 5.6 | fg0c -2.738% of raw [M CR]; 0 commutation failures [M RF] |
| s SIGNPLANE | T acts on magnitudes, sign plane untouched | 5.9 | CFD5, CFDB2 byte-exact on GPU [M CG]; sign plane unchanged 708/708 [M RF] |
| a side symbol | if ever used: `bm3 = clip(max - 5, 0, 2)` on the coded block | not adopted | +10,642 B [M CR] |
| b K sets | sets carry `form`; fallback = plain set; no k-means | 3.2, 5.2 | fallback with ceil model +397,312 B [M CR] |
| canonical index | unchanged 259 rows (coded max 5 is an ordinary tri row) | 3.2 | [I] |
| DEDUP | T deterministic and bijective: equal source records give equal coded records | none | 0 duplicate matrices among 708 [M RF] |
| raw fallback (codec 0) | unchanged: a record not smaller than raw is stored raw | 5.10 step 6 | [R REF:186] |
| CPU decode path (`wex_cuda_decode_to_host` fallback) | must implement T^-1 | 6 | **OPEN-K** |
| deploy gate (byte compare with WEX1) | passes after T^-1 (byte-exact) | none | [M RF e2e_real.json, CG] |
| RAM-cache verify, mirrors | operate on record bytes / header page: unaffected | none | [R REF:216, 468] |
| MTP records (if stored) | own sets (role 1); routed tables expand MTP records (ratio 1.110 plain, 1.079 ceil) | **open** | [M RF] |

---

## 9. Engine, helper API, migration

DRAFT 0.1 section 8 stays (helper feature bit 16 and the bind-taking entry points `wex_cuda_decode_to_host3`, `_to_device3`, `_batch_to_device3`; engine guard; old entry points return 1 for codec 5; compatibility matrix; disk note). Additions:
- Device tables per (model, device): K sets x (3 x 259 x 32-B rows + scale direct tables 3 x 16384 x 5 B) + etab; ~0.55 MB at K = 2 [I].
- Workspace: + coded-scale buffer per lane (7.3); digest words next to error words.
- Tools: `wex_nr3_tools` exports gain per-matrix ceil statistics; `nr3lib.py` record struct `'<8I8B2H5I'`; `check_container` learns codec 5, applies T^-1 and compares byte-exact, or canonical-exact for CZ records; `sample_records.py` / `test_decode_batch` learn codec 5 (byte-exact compare unchanged for non-CZ records) (**OPEN-K**: these live in `lab\v41-gpu-20261107\w1\` and `lab\wex-prod-20261105\` [R REF:273, 332]).
- Build: same procedure as W1 (retrain, encode, CPU-verify every record, every 43rd GPU/CPU byte compare, readbacks, whole-file sha256, per-record rehash [R REF:286-291]) plus the ceil statistics of 5.2.

---

## 10. Expected sizes (V4.1, mean record)

Measured real records (CR, 40 held-out experts, g0 = constant anchor 120; ref-impl layout: 48-B header, length table without base offsets):

| config | mean total B | of raw | vs WNR2 (pp of raw) | capacity vs WEX1 [I] | 7,680 records [I] |
|---|---|---|---|---|---|
| WNR2 | 16,909,926.4 | 0.89943 | | +11.18% | 129.9 GB |
| f | 16,578,764.8 | 0.88182 | -1.761 | +13.40% | 127.3 GB |
| f g0 | 16,573,132.8 | 0.88152 | -1.791 | +13.44% | 127.3 GB |
| f g0 d | 16,546,611.2 | 0.88011 | -1.932 | +13.62% | 127.1 GB |
| f g0 d e | 16,548,556.8 | 0.88021 | -1.922 | +13.61% | 127.1 GB |
| **f g0 d t (v1)** | **16,506,880.0** | **0.87800** | **-2.144** | +13.90% | 126.8 GB |
| f g0 c | 16,057,856.0 | 0.85411 | -4.532 | +17.08% | 123.3 GB |
| f g0 c d | 16,032,153.6 | 0.85275 | -4.669 | +17.27% | 123.1 GB |
| f g0 c d e | 16,033,587.2 | 0.85282 | -4.661 | +17.26% | 123.1 GB |
| f g0 c d t (v1z) | not run | ~0.8507 [I] | ~-4.87 [I] | ~+17.5% | ~122.8 GB |

v1 parts [M CR fgdt g0]: header 48, length table 13,121.2, scales 134,356.1, weights 16,357,171.4, padding 2,183.4 B; 3.6976 bit/w, 0.9719 bit/scale (incl. flush). This spec's layout adds ~2,452 B (base offsets 2,436 + header 16) -> ~16,509,300 B = 0.8781 [I]; digest 0 B.

Value of the bytes (team 3's capacity-value model, estimate, assumes G2 + W2 not yet built [RIC section 6]): elasticity +0.65% tok/s per -1% bytes on U:, +1.09% on E: -> v1 about +1.4% (U:) / +2.3% (E:) vs WNR2, v1z about +3.2% / +5.3% [I]; an ABBA x2 A/B resolves about +-0.85% [RIC section 6].

---

## 11. Test plan

**T1 codec round trips** (CPU reference): 3 chunk pairs (512:256, 2048:512, 4096:2048) x flag sets {plain, CEIL, CEIL|LENC, CEIL|TAIL|DIGEST, CEIL|LENC|HASH, CEIL|LENC|TAIL|DIGEST, + CZ variants, + SIGNPLANE (layout 2, without CZ)} x ceil_mask in {0, 1, 2, 4, 7}. Synthetic matrices: even blocks (all 6 magnitudes, signs, -0), even blocks with scale 0, 1, 253, 254 (coded 255), 255 (skipped); blocks with max 0..4 in a gated matrix (pass through); a matrix with one max-5 block (gate fails -> plain set, ceil bit 0); all-zero blocks; all -0 (CZ); uniform random nibbles (raw fallback); scale count not a multiple of 4 (H32 tail); scale streams with n < 2 (tail = n); a transformed last block of a stream (tail inside it); partial length groups; streams near the 2n+8 bound. Exhaustive: the 65,536 4-nibble patterns x scales {0, 1, 120, 253, 254, 255}, padded to 32 nibbles, as RF edge_tests2. CPU decode byte-exact (canonical-exact for CZ).

**T2 corruption, each rejected with the slab untouched** (DRAFT 0.1's >= 30 cases, minus BM7, plus): flag bit 0 set; HASH|TAIL; TAIL without DIGEST; SIGNPLANE with CZ; `ceil_mask` bit flipped with header_check recomputed (form check); `set_id` changed to the other form's set with matching `ceil_mask` (must fail on digest/hash); transformed block with coded scale 0; `ceil_mask` bit 3 set; digest off by one bit; tail window violated by one; sign-plane bit flips (SIGNPLANE: digest must catch); LENC `alpha`/`sbase` changed (continuity).

**T3 misplaced data and silent acceptance**: DRAFT 0.1 T3 (bind, owner patch, swapped streams, DEDUP alias) with DIGEST; then re-run team 2's audit protocol (`audit_corrupt.py`, cases A-J, 8 experts x 1,500 streams, 12,000 trials per case and matrix, w1 and w2) on ceil-form records for {today's rule, HASH, TAIL alone, TAIL + DIGEST}: required 0 accepted for HASH and TAIL + DIGEST; report the TAIL-alone acceptance (not measured so far).

**T4 canonical zero**: DRAFT 0.1 T4, plus the two 5090 tests of 5.6 and the tooling changes (Z(WEX1) compare, sha256 of canonical slabs).

**T5 model**: WNM3 open rejects layout 1, alphabet/layout mismatch, form 2, role 2, K = 0 or 17, bad set_bytes, etab > bound, plus DRAFT 0.1 T5; regenerate the record-level H32 vectors.

**T6 container**: DRAFT 0.1 T6 with entry bits 8-15.

**T7 GPU** (sm_86 and sm_120): byte-exact vs the CPU reference and vs the original source on 30+ records via host and device paths; batch sizes 1, 2, 3, 4, 6, 8, 10, 16; negative controls (inverse off; weight-only inverse) must fail everywhere (CG style); retry idempotence: decode the same device record twice -> identical output; compute-sanitizer memcheck/racecheck 0 errors (final-scale stores); a corrupted record inside a batch -> -1 and untouched, others 0.

**T8 fuzzing**: >= 300 mutations per flag set; differential CPU vs GPU status.

**T9 sizes and speed**: reproduce CR on the same 40 held-out experts with the spec layout (target: within +-0.02 pp of raw of CR's fgdt g0 + 2,452 B); fg0cdt measured for the first time; kernel-only us/record at 1 and 8 concurrent records on the 5090 with KAIRA's kernel: gate WNR3 <= WNR2 + 5% (DRAFT 0.1 7.3), inverse own-cost <= +2% (CG measured +0.3..+1.7% median).

**T10 build checks**: all records CPU-verified; every 43rd GPU/CPU byte compare; readbacks; sha256; per-record rehash; ceil statistics (max-5 blocks, skipped scale-255 even blocks, fallback matrices) equal between build report and `check_container`.

**T11 inference**: byte-exact WNR3 -> short token-identity run (it is byte-identical to WEX1 after decode); CZ -> 5.6 tests 1-2.

**T12 generality (optional)**: gate and gain on V4-Flash, V4-Pro, gpt-oss (CO data in `<RESEARCH-PC>\team4-data\`), and a synthetic single-pass ceil checkpoint (expected ~22% max-5 blocks -> gate fails -> plain set) [I CO sim_double.json].

---

## 12. Ordered rollout

1. **R0 prerequisites**: resolve the OPEN-K items of section 13; keep WEX1 and the safetensors; decide the disk for a ~126.8 GB container (U: had 122.6 GB free at 17:11 [R REF:139]).
2. **R1 CPU reference** (`wex_neural_rans3.h`, tools): f, g0, d, t + DIGEST, e as alternative; T1-T6, T8 pass; T9 bytes reproduce CR.
3. **R2 GPU helper** (feature 16): layout 0 kernels with the deferred scale write and separate output; T7 on the 3070, then the 5090; T9 speed gate on the 5090.
4. **R3 64-record V4.1 sample** (`sample_records.py`, seed 20261107): deploy gate (byte compare with WEX1 after T^-1) with `test_decode_batch3`, CPU/GPU agreement, corruption statuses.
5. **R4 full byte-exact build** (flags 0xB4, K = 2, layout 0): T10; 0 expected fallbacks; sha256 + `.verified`.
6. **R5 engine integration and A/B** vs WNR2 on the same drive (token identity, tok/s; expected ~+1.4% on U: by the model [I]).
7. **R6 optional sign-plane layout** only if a compressed VRAM tier needs decode speed (5090 measurement first).
8. **R7 canonical zero (separate decision)**: 5090 bit test + per-step logits memcmp; tooling changes; CZ container next to the byte-exact one; never delete WEX1/safetensors before the decision.

---

## 13. Open items

### 13.1 Needs the KAIRA sources (not on this PC)
1. `wex_validate_header` (v3) rejects unknown flag bits (REF:174 says only "any of flags 1|2|4|8|16").
2. No code path reads `entry.flags >> 8` as scale_pages for codecs other than 1 (entry bits 8-15).
3. KAIRA weight kernel: where the uint4 store and the running `mx` are, how the scale buffer is read (`__ldg`/`__restrict__`), real smem/registers/blocks per SM on sm_86 and sm_120, and whether the 567-row CTX 0 layout is replaced by the 259-row index.
4. Scale kernel: blocks per matrix (LENC), where it writes scales (needs the coded-scale buffer).
5. Commit kernel `wex_cuda_commit_regions` and the batch scratch (`WexCudaBatchHost`): room for a digest word per record and a separate final-scale region or bitmap.
6. Registry room for the model kind (WNM2/WNM3).
7. Which expert-matmul kernels KAIRA dispatches on sm_120 (for the CZ bit test), and whether team 2's colibri line numbers (`deepseek_v4.c:20123-20129`, `quant.h:1266-1404`, `backend_cuda_dsv4.cu:99, 2250-2313`, `chain_moe_gv.comp:97-114`) match KAIRA's fork.
8. Code paths of `sample_records.py`, `precheck_records.py`, `test_decode_batch.cu`, `check_container.c`, `build_nr2_full.py` to extend for codec 5, T^-1 and Z(WEX1) compares.
9. Helper lane memory: room for +1.05 MiB per lane.
10. Whether `wex_nr2_parse`, the CUDA plan or `check_container` re-check `median[m]` against the decoded scales (g0 writes a constant instead).

### 13.2 Other open questions
- GPU cost of per-lane H32 (HASH or DIGEST) and of the digest atomics: not measured anywhere.
- TAIL-alone silent acceptance rate: not measured (T3).
- fg0cdt and layout-2 tail: not run.
- Commit-kernel fusion of the bitmap: not implemented.
- Constant anchor (g0) on models other than V4.1 and on MTP: measured on V4.1 only (1 flip in 120 test matrices per rule) [M CR].
- MTP records: whether the container stores them (384 routed MTP experts); they need own sets [M RF].
- Shared experts were not checked against the ceil invariant [RF].
- A real single-pass-ceil checkpoint was not measured (gate failure predicted from simulation) [CO].
- 3070 timings are a proxy for the 5090 (7.5).

---

## 14. Sources
- Measurements: CR, CG, RF, CO, MV, T2A, T2Z, T2G, T3S, ANS (table at the top); team 3 DRAFT 0.1 and `wnr3_ref.py` / `lenres.json` (`<RESEARCH-PC>\team3\wnr3-spec\`).
- Reference: `D:\Desktop\WNR2-REFERENCE.md` (lines cited as REF:n) and `...\research\wnr2-brief.md`.
- Code read: `<RESEARCH-PC>\team4\ceil-ref\wnr_common.py` (transform 212-265, tail 62/398-405/675-676/731-733, LENC 499-599/771-789); `<RESEARCH-PC>\team4\ceil-gpu\src\wnr2dec.cu` (as cited by CG); `<RESEARCH-PC>\team2\gpu-proto\src\wnr2fmt.py:12-22`; `<RESEARCH-PC>\team2\zero-safety\scripts\zs_kern.c:4-19` (colibri source map).
- [S] MurmurHash3 / SMHasher (DRAFT 0.1 section 13); OCP MX v1.0 section 6.3 and DeepSeek `inference/kernel.py` (via CO and RIC sources).

---

## Appendix A. Measured, not adopted

**A.1 Feature a (block-max side symbol + block half) on ceil-form.** On ceil-form the block max takes 3 values (5, 6, 7), so the side symbol becomes `bm3 = clip(max - 5, 0, 2)`, one 3-symbol rANS step per block, contexts x6 (3,402 per type). Real records: fga vs fg +10,642 B (CI +10,135..+11,524), improved on 0 of 40 experts; side symbols cost 0.0332 bit/w; fgad -1.847% vs fgd -1.905% [M CR]. Estimate on top of ceil-form: +0.062% [M MV mxceil2]. Table: 3,402 rows uncompacted (108,864 B, does not fit), 1,211 rows compacted (38,752 B, 4.7x v1) [I]; a team 2 table of 33 KB ran 1.9-2.3x slower at 32 threads [M T2G]. On the plain form a gave -0.20% of the record [RIC section 7]; ceil-form absorbs it (the max-7 bit was a proxy for the doubled blocks) [M MV]. Verdict: costs bytes and speed. Reserved: record flag bit 0, model layout 1.

**A.2 Feature b (K k-means table sets).** fgaK2 vs fga -2,087 B (CI -4,564..-601, 27/40 improved), fgaK2 -1.722% vs fg -1.764%; fgaK2cde vs fgcde +8,585 B (CI +6,900..+9,625) [M CR]. Online-research team: K tables add ~0 on ceil-form (relayed in the task brief; RIC section 7 gives -0.02..-0.04%); plain-form K = 2..4 gains (-0.26..-0.28%) were mostly the anchor flip [M T3S: K2 "by flip flag" equals K2 EM, 3.77262 bit/w]. Shared memory: none (one set in smem); but no gain. The set mechanism stays only for the plain fallback set (and possible MTP sets).

**A.3 Two rANS chains per stream (DB2/CFDB2).** Fastest measured decode (CFDB2 431.1 us/record, 3,307.5 us per 8 records on the 3070) but +0.36% of the record (CFDB2 0.88477 vs CF 0.88155) and +8-10 registers at nt128/256 (occupancy 40 -> 36/32 warps) [M CG]. Not adopted: bytes dominate on the PCIe path.

**A.4 Split start state** (e.g. 12-bit hash + 2 tail nibbles): ~-0.12% estimated [RIC section 3], not measured; dominated by TAIL + DIGEST.

**A.5 Small initial state** (x0 = 2^k + h): -0.184..-0.188% of raw on 720 streams (k = 8 / 0), 24/24 round trips [M ANS flushx.json]; uses the same start-state slot as TAIL, measured only on estimates; TAIL is measured on real records (-0.214%).

**A.6 Alphabet order 2*mag + sign** (instead of codes 0..15): silent single-bit-flip acceptance 0.567% -> 0.058% at identical bits (3.759073 bit/w) [M T2A symperm_audit.json]. Made unnecessary by DIGEST/HASH; would also change the CDF order the GPU compare relies on [I].

**A.7 Extra contexts on ceil-form** (held-out estimates, flush 3.5 B): full pm2 -0.0005% (42 KB table); 6-value pm2 bucket -0.0004%; position in block -0.019% (185 KB table) [M MV mxceil2]; row/column, running-mean, phase contexts ~0 (online-research team, relayed in the task brief; RIC section 7: learned context tree and row/column classes 0.002-0.005 bit/w). srel alignment on top of ceil-form -0.003 bit/w (online-research team) is about what g0 captures: -6,028 B = -0.0014 bit/w [M CR; I conversion].

**A.8 Non-ceil side-information model** (3-class side + half + fixed anchor + scale|class): -1.807% / -1.780% / -1.827% of raw (P2/P3/P4), 97% of the clean ceil gain; ceil-form adds -0.055% on top; needs 3,402 rows and a class stream decoded before scales [M RF anchor_an.json]. Ceil-form is preferred for simplicity at equal or better size.

## Appendix B. Worked example of T (one block)

Source block (scale s = 120, magnitudes of 32 nibbles, signs arbitrary): {7, 6, 0, 2, 4, 5, 0, 7, ... } using only {0, 2, 4, 5, 6, 7}, max 7 -> even. Values: magnitudes {0, 2, 4, 5, 6, 7} = {0, 1, 2, 3, 4, 6} x 2^(120-127).
T: magnitudes -> {5, 4, 0, 1, 2, 3, 0, 5, ...}, scale 121. Values: {3, 2, 0, 0.5, 1, 1.5, 0, 3} x 2^(121-127) = the same numbers (block max 3.0 at scale 121: the ceil rule's form).
Decoder: coded max 5 -> D: {7, 6, 0, 2, 4, 5, 0, 7, ...}, scale 121 - 1 = 120: the source bytes.
A non-even block {7, 3, 1, 6, ...} (contains 1 and 3) or {6, 4, 2, ...} (max 6) is not transformed; its coded max is 6 or 7, so the decoder leaves it alone.
