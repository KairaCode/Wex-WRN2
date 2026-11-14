# WNR3 specification (FINAL v1, 2026-11-10)

**Codec version:** WNR3 v1 (FINAL), frozen on 2026-11-10. **Status:** stable, in production. **Change log:** "Document history" at the end. The lossless expert-record codec of the WEX formats, successor of WNR2; record codec 5 of WEX3 and WEX4 containers.

**Inputs:** the owner's decisions of 2026-11-10 (below); drafts `Wex-WRN2\research\wnr3-20261107\team4\spec-update\WNR3-DRAFT-v2.1.md` (DRAFT 0.2.1), `Wex-WRN2\research\wnr3-20261107\team3\wnr3-spec\WNR3-DRAFT.md` (DRAFT 0.1), `team4\spec-update\`, `Wex-WRN2\docs\reports\wnr3-research-measurements-20261107.md`; 5090 results `Wex-WRN2\research\wnr3-5090-20261107\RESULTS.md`, `zero\ZERO-RESULTS.md`, `ceil-kaira\`, `g0\`; the engine source of that time in the maintainer's tree (`src/colibri/c/wex_neural_rans.h`, `wex_format.h`, `wex_decode.c/.h`, `wex_cuda.cuh`, `tools/wex_cuda_bridge.cu`, `expert_store_wex.c`, `deepseek_v4.c`). Line numbers `v4:NNNN` refer to that version of `src/colibri/c/deepseek_v4.c`, which is not in this repository.

**Status:** in production since 2026-11-10 as the record codec of DeepSeek V4.1 REAP-192 (WNR3-CZ, codec 5); since 2026-11-11 those records sit in a WEX4 container ([WEX4-SPEC.md](WEX4-SPEC.md)). The code is in `c/` and in the engine. What runs today: [STATUS.md](../STATUS.md).

**Gates** (section 7.2, all on 2026-11-10): G1 to G6 passed, G5 as the real-stream audit of [WNR3-VERIFY.md](reports/WNR3-VERIFY.md) section 2 instead of the protocol in the gate table; G7 passed through a lab switch instead of subset containers (erratum 8); the full container of G8 was built and read back (123,306,192,896 B, 7,680/7,680 records). Switch record: [WNR3-PRODUCTION-20261110.md](reports/WNR3-PRODUCTION-20261110.md).

**History of the implementation (2026-11-10):** written on the engine branch `wnr3-20261110` (commits for the CPU codec 5; the GPU decoder, helper feature 16, engine and builder; review fixes; the lab-only `V4_LAB_CANON_ZERO`). Gate evidence: G3 `research/wnr3-20261110/g3_result.json`; G2b/G4/G6 `research/wnr3-20261110/mutants/gpu_run1.log` and WNR3-VERIFY sections 1 and 3; G5 WNR3-VERIFY section 2. After the switch WEX1 and all WNR2 copies of V4.1 were deleted.

Paths in this file: `Wex-WRN2\...` is this repository; `lab\...`, `app\...` are relative to the maintainer's application tree (`<TREE>`); `<worktree>` is its worktree of branch wnr3-20261110. Files in this repo: `c/` (codec, decoders, store), `c/tools/wex_cuda_bridge.cu` (helper), `c/tests/test_wex_format.c`, `c/tests/test_expert_store_wex.c`, `c/tools/build_nr3.py` and `c/tools/wex_nr3_tools.c` (builder, encoder DLL), `c/tests/` (engine/GPU tests), `integration/engine/deepseek_v4.wnr3.wex-excerpt.c` (engine hooks), `research/wnr3-20261110/` (verification scripts and results).

### Errata (normative: where they and the sections below disagree, the errata and the code win)

The sections below are the spec as frozen before implementation; each erratum names the sections it corrects. They were not folded into those sections so that the frozen text stays as it was.

1. **GPU ceil inverse: in-place per-block scale fix-up, no 64-block emask** (sections 9.4, 8). The weight kernel decrements the block's scale byte in place right after the block's D inverse (`scales[block] = scale - 1`), before the block's uint4 store. This is safe for the same reasons as the planned group flush: the block's scale was already read for its context, `next_scale` is `scales[block + 1]` (read before the write), and the scale ranges of different weight streams are disjoint. No `emask`, no `wex_cuda_ceil_fix_scales`, no group flush; any legal `Cw` works the same way. `scale == 0` in a max-5 block sets error bit 4. ptxas: 0 spill stores/loads.
2. **`wex_nr3_owner_ok` takes the container header flags** (`(entry, record, layer, expert, header_flags)`, section 9.2): besides the owner (bytes 28..31) it requires the record's CZ bit (byte 27 bit 3) to equal header flag 64. Every decode site passes the flags: engine host slab, GPU-direct fill, batch completion, the standalone `wex` store, the tools.
3. **The CZ refusal is one shared function**, `coli_wex_container_refused(header_flags, who)` in `wex_decode.c` (sections 4.4, 9.7), called by the engine store (`deepseek_v4.c`) and by the standalone store (`expert_store_wex.c`, `COLI_EXPERT_STORE=wex`). Same messages as section 4.4.
4. **RAM cache release** (sections 4.8, 9.7): only a *rejected* record (decode status -1, owner or profile failure) leaves the cache of compressed records. Status 1 (helper unavailable) and mirror-target failures keep it, because they say nothing about the record's bytes. The planned "`status == 0` instead of `hit || status == 0`" would have dropped good records on status 1.
5. **Header flags 32|4 (WNR3 + DEDUP) are invalid** (section 2.2 said the builder never sets DEDUP; the validator now enforces it).
6. **Builder modes** (section 9.8): `build_nr3.py DLL WEX1 bins OUT.bin (--experts L:E,... | --g3 RECORDS.json)` and `build_nr3.py DLL WEX1 container OUT.wex --layers ... [--max-experts N]`, model options `--model FILE | --train L:E,... | --train-lab80 [--save-model FILE]`, `--cz`. There is no `--full` / `--allow-full` / `--fuzz` mode: a full container is container mode with all 40 layers, and its free-space check is the raw sum + 1 GiB (~135.5 GiB for V4.1), to be relaxed to about 125 GiB before a full build (WNR3-VERIFY section 5). The CPU/GPU mutation differential lives in `c/tests/test_wex_nr3_gpu.cu`, not in the builder.
7. **Engine test:** `c/tests/test_v4_wex_nr3.c` runs the engine's own store (`deepseek_v4.c`) on WNR3 and WNR3-CZ containers through a CPU stand-in helper (`c/tests/fake_wex_cuda.c` -> `fake_wex_cuda.dll`, features 1|4|16, and 32 since 2026-11-11).
8. **G7** is replaced by the lab switch `V4_LAB_CANON_ZERO=1` on a WEX1 run (no subset containers): every FP4 -0 nibble of every expert slab becomes +0 right after the read, so a WEX1 run computes on what a WNR3-CZ container decodes to. Script: `research/wnr3-20261110/g7lab/`. Run on 2026-11-10: PASS (WNR3-VERIFY, section G7-lab result).

---

## 0. Decisions and what this spec does with them

| # | owner decision (2026-11-10) | in this spec |
|---|---|---|
| 1 | **ceil-form + rule g** (not the fixed g0 = 120 anchor) | sections 4.2, 4.3; per-record anchor byte computed by rule g on the coded scales; the decoder reads the byte (unchanged from WNR2) |
| 2 | **canonical zero** (every FP4 -0 stored as +0, value-exact) | section 4.4; container header flag 64, record flag CZ; every gate compares against canonicalized WEX1 (section 7) |
| 3 | **per-stream integrity hash** `x0 = L + h23`, MurmurHash3 of stream identity + decoded symbols; NOT TAIL+DIGEST | section 4.5 |

**Deliberately NOT in WNR3** (in the drafts, not decided): coded length table (LENC), TAIL/DIGEST, sign plane (layout 2), WNM3 model blob, K table sets / plain fallback set, block-max side symbol, 64-B header, header_check word. WNR3 therefore keeps WNR2's record skeleton (32-B header, u16 stream lengths, 4-B big-endian flush, WNM2 model blob), which is exactly the skeleton the 5090 CFG measurement used. Any of the dropped features later means a new record magic (WNR4) and a new codec id. *(When this was written the next free id was 6. WEX4 has since given 6 to WBP1, so a WNR4 record would be codec 7: [WEX4-SPEC.md](WEX4-SPEC.md) sections 7 and 12.2.)*

**Measured basis** (all on V4.1 REAP-192 MXFP4, raw record 18,800,640 B, 40 held-out experts, model trained on 80 others):

| quantity | value | source |
|---|---|---|
| WNR2 (median anchor), same 40 | 16,909,926.4 B = 0.89943 | RESULTS.md 1.2 (Model A) |
| CFG = ceil-form + rule g, WNR2 skeleton | **16,578,252.8 B = 0.88179** (-1.764% of raw) | RESULTS.md 1.2, `ceil-kaira/ceil_kaira_records.json` |
| CFG max over the 12 largest production records | 17,985,536 B (prod WNR2 18,321,408) | RESULTS.md 1.2 / 2.2 |
| CFG GPU batch decode (lab kernel, 5090) | 0.380-0.382 ms/record batch-16 (WNR2 helper 0.372-0.385) | RESULTS.md 1.2 |
| hash x0 cost | +1,352 B/record (+0.0072%) | DRAFT 0.2.1 row e (CR fge vs fg; measured with FNV/lowbias32, cost depends only on x0 spread, transfers) |
| canonical zero on top of ceil+anchor | -514,848 B/record (-2.738% of raw) | DRAFT 0.2.1 row c (CR fgc vs fg, g0 anchor basis) |
| CZ bit test, default CUDA paths | 0 differing bits / 83,841,024 outputs | `zero/ZERO-RESULTS.md` |

Expected WNR3 record (estimate, section 6): **~0.8819 of raw without CZ, ~0.8545 with CZ** (the owner's "~0.853" figure included LENC, -0.137%, which is not in WNR3).

---

## 1. Scope and terms

- One record = one routed expert = matrices m = 0 gate w1, 1 down w2, 2 up w3. Decoded slab layout unchanged: scales w1, w2, w3, then packed FP4 weights w1, w2, w3 (`wex_entry_raw_sizes`).
- Block = 32 consecutive FP4 nibbles of a row with one UE8M0 scale. Nibble q: magnitude `q & 7`, sign `q >> 3`. E2M1 magnitudes 0..7 = 0, 0.5, 1, 1.5, 2, 3, 4, 6. Symbol 2i is the low nibble of byte i.
- rANS exactly as WNR2: 32-bit state, `L = 2^23` (`WEX_NR_L`), 14-bit CDFs, byte renormalisation, 4-B big-endian initial state in `[L, 256L)`, every stream ends exactly at its u16 length.
- **Coded domain**: the scales `cs` and nibbles `cq` that the streams carry (after canonical zero and ceil-form). **Output domain**: the decoded slab = source (non-CZ) or Z(source) (CZ).
- **Z(x)**: x with every weight nibble 0x8 replaced by 0x0 (scales untouched).
- Global stream index g: scale streams of m0, m1, m2, then weight streams of m0, m1, m2 (= the CPU `stream` counter in `wex_nr2_decode` and the index into the GPU offsets table).
- V4.1 chunks stay `Cw = 2048`, `Cs = 512` (stored per record, any legal pair works).

---

## 2. Identifiers and versioning

### 2.1 Constants

| name | value | file | meaning |
|---|---|---|---|
| `WEX_HEADER_NEURAL_RANS3` | 32 | wex_format.h | container holds WNR3 records + one WNM2 model (WNMD descriptor at byte 64, as WNR2) |
| `WEX_HEADER_CANONICAL_ZERO` | 64 | wex_format.h | every WNR3 record is value-exact (CZ); requires 32 |
| `WEX_ENTRY_NEURAL_RANS3` | 5 | wex_format.h | entry codec; `entry->flags == 5` exactly (no bits 8-31) |
| `WEX_NR3_MAGIC` | `0x33524E57` ("WNR3") | wex_neural_rans.h | record magic |
| `WEX_NR3_FLAG_CEIL_MASK` | `0x07` (bit m = matrix m in ceil form) | wex_neural_rans.h | record flags |
| `WEX_NR3_FLAG_CZ` | `0x08` | wex_neural_rans.h | record flags |
| `WEX_NR3_HASH_SEED` | `0x33524E57` | wex_neural_rans.h | MurmurHash3 seed (= magic) |
| `WEX_NR3_HASH_MASK` | `0x007FFFFF` | wex_neural_rans.h | 23 bits |
| `WEX_CUDA_FEATURE_NR3` | 16 | wex_cuda_bridge.cu | helper accepts codec 5 in decode_to_host / _to_device / _batch_to_device |

Model: **no new model format**. A WNR3 container stores a WNM2 blob (`WEX_NR2_MODEL_MAGIC`, version 1, 59,408 B, FNV-1a id, WNMD descriptor), trained on coded-domain data (section 5). The model registries (engine `coli_wex_register_model`, helper `wex_cuda_register_model`) are unchanged and shared by WNR2 and WNR3.

Helper ABI stays 1; with WNR3 `wex_cuda_helper_features()` returned 31 (1|2|4|8|16) and there were no new entry points. *(The bridge in `c/tools/wex_cuda_bridge.cu` has since gained bit 32, `WEX_CUDA_FEATURE_BATCH_HOST` = `wex_cuda_decode_batch_to_host`, added for the qwen38 store, and returns 63.)*

### 2.2 Container rules (WEX3 version 3)

- Header page, entry table, record placement (`wex_record_read_offset = aligned_raw - total_bytes`), mirror rule (byte-identical header page + table) unchanged.
- v3 accepted header flags become `1|2|4|8|16|32|64`. New rules: 32 excludes 8 and 16; 64 requires 32.
- WNR3 builder writes header flags **33** (SCALE_TILES|NEURAL_RANS3), or **97** with CZ. **No DEDUP**: the builder never aliases records (V4.1 has 0 duplicate matrices), so a record's owner is always its own (layer, expert) and `wex_validate_table` already rejects aliases without flag 4.
- Entries: codec 0 (raw, record not smaller than aligned raw; stores the SOURCE bytes, also in a CZ build), or codec 5. Codec 4 entries are invalid in a flag-32 container (header rule: 32 excludes 16).
- `wex_validate_entry` for codec 5 = the codec-4 rule with flag 32 instead of 16: header v3 with flag 32, `total_bytes` 4096-aligned, `>= align4096(32 + 6 * streams at chunk 16384)`, `< aligned_raw`.
- WNMD descriptor: `wex_nr2_model_location` accepts flag 16 **or** 32 (same descriptor, same blob size).

### 2.3 Versioning and compatibility

WNR2 containers keep decoding unchanged: codec 4 takes exactly today's code path (record magic WNR2, `x0 = L`, no inverse, no hash); the WNR2 encoder output stays byte-identical (golden test, section 10). Future record changes get a new magic and codec; reserved record-flag bits must be 0 and are rejected (fail closed).

In this table "deployed" is the engine and the helper installed before the switch of 2026-11-10 (no WNR3 code, helper features 15), and "new" is the WNR3 code, which is what is installed since.

| reader \ container | WEX1 / WNR2 (flag 16, codec 4) | WNR3 (flag 32, codec 5) | WNR3 CZ (flags 32+64) |
|---|---|---|---|
| deployed engine + deployed helper (features 15) | OK | refused at `wex_validate_header` (unknown flag 32): "[v4_wex] Invalid or incompatible container", safetensors | same |
| new engine + deployed helper | OK | refused by the guard (needs helper feature 16), safetensors | same |
| new engine + new helper | OK, bit-identical to today | OK | OK unless `DSV4_CUDA_MOE_MASKED` != 0 (refused, section 4.4) |
| new helper + deployed engine | OK (the engine never sends codec 5) | n/a | n/a |
| `c/tools/repack_v4_wex.py` | refuses flags > 15 (unchanged) | refuses | refuses |

---

## 3. Record layout

`[0, 32)` header; `[32, 32 + 2(S+W))` u16 stream lengths (scale streams then weight streams, g order); scale payload (SP B); weight payload (WP B); zero padding to `total_bytes = align4096(32 + 2(S+W) + SP + WP)`. Identical to WNR2 except header bytes 0..3 and 27..31.

| off | type | field | rule |
|---|---|---|---|
| 0 | u32 | magic | `0x33524E57` "WNR3" |
| 4 | u32 | model_id | != 0, == WNMD id, registered |
| 8 | u32 | SP | scale payload bytes == sum of scale stream lengths |
| 12 | u32 | WP | weight payload bytes == sum of weight stream lengths |
| 16 | u32 | Cw | power of 2, 512..16384 |
| 20 | u32 | Cs | power of 2, 256..16384 |
| 24 | u8[3] | anchor[m] | srel anchor of matrix m, rule g on the CODED scales (4.3); decoders read it, never re-derive it |
| 27 | u8 | flags | bits 0-2 CEIL[m]; bit 3 CZ; **bits 4-7 must be 0** |
| 28 | u32 | owner | `layer << 16 \| expert` of the entry the record belongs to |

Parse (`wex_nr3_parse`) checks everything `wex_nr2_parse` checks (chunks, geometry, every length >= 4, length sums == SP/WP, `total == align4096(used)`, zero padding) plus magic WNR3 and `flags & 0xF0 == 0`. It does not check the owner (the caller does, 4.7) and does not check CZ against the container flag (the engine does at open; `check_container`/builder per record).

Wire struct (Python): `'<6I3BBI'` = 32 B.

---

## 4. Codec

### 4.1 Streams and contexts (unchanged from WNR2)

Scale stream: coded scales `[j*Cs, min((j+1)*Cs, scount_m))`, order-0 with the model's scale CDF of m. Weight stream: coded nibbles `[j*Cw, min((j+1)*Cw, count_m))`, context at block start `srel = clamp(cs_b - anchor[m], -3, 3) + 3`, `pm = mx = 8`, after symbol x `pm = x & 7`, `mx = max(mx, x & 7)`, `ctx = (srel*9 + pm)*9 + mx` (567 contexts). Contexts always use CODED scales and CODED nibbles. Streams never cross a matrix; `count_m % 32 == 0` so every weight stream holds whole blocks and `n/2` whole bytes.

### 4.2 Ceil-form (decision 1)

Definitions for one block (nibbles q_0..q_31, scale s, `a_j = q_j & 7`, `M = max a_j`):
- `even` := `M == 7` and no `a_j` in {1, 3}.
- `H` (halve) 0->0, 2->1, 4->2, 5->3, 6->4, 7->5. `D(a) = a + min(a, 2)`: 0->0, 1->2, 2->4, 3->5, 4->6, 5->7. `D(H(a)) = a` on {0,2,4,5,6,7}.

**Gate (encoder, per matrix m):** m is coded in ceil form (CEIL[m] = 1) iff **no block of m has M == 5** (evaluated after CZ). That is the exact bijection condition (exhaustive 4-nibble test, RF edge_tests2.json). Blocks with M in 0..4 are legal and untouched. A matrix that fails the gate is coded plain with the same model (CEIL[m] = 0), reported by the builder; expected count on V4.1: 0 (0 blocks with M < 6 in 260,997,120 checked). No plain fallback table set exists in WNR3 (cost if it ever happens: about +2.1% of that record's raw, CR test_ceil.json; still lossless).

**Transform T_m** (CEIL[m] = 1):
```
for every block b of m:
    if even(b) and s_b <= 254:  q_j' = (q_j & 8) | H(q_j & 7);  s_b' = s_b + 1
    else:                       unchanged
```
An even block with `s_b == 255` stays unchanged (keeps M = 7). Coded scale 255 (from 254) is a legal symbol.

**Inverse T_m^-1** (decoder, CEIL[m] = 1), per decoded block (coded nibbles q', coded scale cs_b):
```
M' = max(q_j' & 7)
if M' == 5:  if cs_b == 0: INVALID (error)
             q_j = (q_j' & 8) | D(q_j' & 7);  s_b = cs_b - 1
else:        unchanged
```
`M'` in 0..4 is legal and passes unchanged (the lab's NumPy `ceil_inverse` and team-4's `wnr_common.py` assert `M >= 6`; a conforming decoder must not). CEIL[m] = 0: identity. The sign bit is never touched; T commutes with Z.

### 4.3 Rule g anchor (decision 1)

`anchor[m] = argmax over v in 0..254 of (h[v] + h[v+1])`, h = histogram of matrix m's CODED scales, first maximum wins (strict `>` scan from v = 0), exactly the lab's `wex_nr2_median` variant `g0_anchor == -2` in `research\wnr3-5090-20261107\g0\wex_neural_rans_g0.h`. Used identically by the trainer (counts) and the encoder; written to header bytes 24..26; decoders only read it (`wex_neural_rans.h` decoder and `wex_cuda.cuh` `plan.median[m]` today). Evidence: rule g never worse than median on 40/40 held-out experts and removes g0's tail blow-up (L39_E120: g0 20,975,616 B, rule g 18,411,520 B plain / 17,985,536 B ceil-form).

### 4.4 Canonical zero (decision 2)

- **Rule (encoder):** in a CZ build every weight nibble 0x8 is replaced by 0x0 before the gate, T, training counts and encoding. Scales are never touched. Every record of a CZ container gets flag CZ and the container gets header flag 64 (a non-CZ build sets neither).
- **Decoder:** a CZ record must not decode any coded nibble 0x8 (rANS-decoded symbol 8): invalid (CPU -1, GPU error bit 8). Checked in the coded domain (T keeps magnitude 0 and the sign bit, so it is the same in the output domain).
- **Output:** Z(source): equal in value, not in bytes.
- **Where it applies:** only routed-expert records inside a WNR3 container with header flag 64, consumed by the CUDA V4/V4.1 engine (`deepseek_v4.c`, the only WEX consumer). Not to scales, not to WNR2 containers, not to experts the engine takes from safetensors (absent entries, fallbacks: they keep their -0, value-equal), not to shared experts/attention/MTP (not in the container), not to Vulkan (`deepseek_v41.c` has no WEX code).
- **Proof state:** default CUDA paths bit-identical (83,841,024 outputs, 0 differing bits; CPU FP4 path decodes code 8 as +0 anyway). NOT proven: the masked DeepGEMM path `dg_moe<256>` (`DSV4_CUDA_MOE_MASKED=1`, non-deterministic and emits -0 itself) and an end-to-end per-step logits memcmp. Hence:
  - engine policy: a flag-64 container is refused when `DSV4_CUDA_MOE_MASKED` is set and != 0 ("[v4_wex] value-exact (canonical-zero) container refused: DSV4_CUDA_MOE_MASKED is not proven value-exact"); otherwise it logs "[v4_wex] value-exact (canonical-zero) container".
  - deployment gate G7 (section 7.2): per-step logits memcmp before any CZ container replaces the WNR2 one.
- **Irreversibility:** a CZ container cannot reproduce the WEX1 bytes. Keep `U:\kaira_wex\v41-reap\v4_flash_experts.wex` (WEX1) and the safetensors until G7 passed and the owner decided. *2026-11-10: G7 passed and WEX1 was deleted after the switch; the safetensors are now the only byte-exact source (WEX1 can be regenerated from them, [WNR3-PRODUCTION-20261110.md](reports/WNR3-PRODUCTION-20261110.md)).*

### 4.5 Per-stream integrity hash (decision 3)

**Function H32 = MurmurHash3_x86_32** (Appleby, public domain): `c1 = 0xcc9e2d51`, `c2 = 0x1b873593`; per 4-byte little-endian block `k *= c1; k = rotl(k,15); k *= c2; h ^= k; h = rotl(h,13); h = h*5 + 0xe6546b64`; tail of 1-3 bytes `k = t0 | t1<<8 | t2<<16; k *= c1; k = rotl(k,15); k *= c2; h ^= k`; then `h ^= len`; `fmix32: h ^= h>>16; h *= 0x85ebca6b; h ^= h>>13; h *= 0xc2b2ae35; h ^= h>>16`. Seed `0x33524E57`.

**What is hashed, per stream g** (key = 16-B identity || data, `len = 16 + nbytes`):
```
ident[0] = model_id                     (record bytes 4..7)
ident[1] = owner                        (record bytes 28..31)
ident[2] = g                            (global stream index, section 1)
ident[3] = u32 LE of record bytes 24..27 = anchor[0] | anchor[1]<<8 | anchor[2]<<16 | flags<<24
data     = scale stream: its n decoded CODED scale bytes
           weight stream: its n/2 decoded CODED packed bytes (low nibble = even symbol), before T^-1
h23      = H32(ident || data, seed 0x33524E57) & 0x7FFFFF
```
Each ident word is one MurmurHash block (LE), so the data starts block-aligned. The ident covers every header field the decode depends on except chunks/sizes/lengths, which the structural checks cover (a wrong length shifts stream boundaries).

**Initial / final state:** the encoder starts each stream at `x0 = L + h23` (in `[L, 2L)`, inside the legal `[L, 256L)`) and encodes backwards as today, then flushes 4 B. The decoder keeps the initial-state check `[L, 256L)` and requires, after consuming exactly the stream's length, **`final state == L + h23(decoded data)`**; WNR2 (codec 4) keeps `final == L`. Every WNR3 stream is hashed; there is no flag to turn it off (a flag bit would be a single point that disables the protection).

**Strength:** a corrupted stream passes with probability ~2^-23; a stream decoded at another index, under another owner, anchor, flag set or model fails (ident). Today's WNR2 lets 0.26-0.63% of single bit flips through (each a single sign flip, f[s] == f[s^8]); with any 23-bit x0 hash 104/104 sign flips were caught and 0/9,000 accepted (team 2, other hash functions). **MurmurHash3 detection is to be re-measured (G5).**

**Test vectors** (computed 2026-11-10 with a pure-Python MurmurHash3 that reproduces the SMHasher verification value; must be reproduced by the C and CUDA code):

| case | value |
|---|---|
| SMHasher Murmur3A verification (keys 0..255, seed 256-i, hash of the hash table with seed 0) | `0xB0F57EE3` |
| `H32("", 0)` / `H32("", 0x33524E57)` / `H32(bytes 00..0F, 0x33524E57)` | `0x00000000` / `0xE7F741F1` / `0x31CE0919` |
| weight stream: model_id `0x299DC586` (lab model_CFG_rule_g.wnm2), owner L39 E120 = `0x00270078`, g = 2165, bytes 24..27 = `78 79 78 0F` (anchors 120,121,120, flags CEIL 7 + CZ) -> ident[3] `0x0F787978`; data 1024 B `data[i] = (37i + 11) & 0x77` | H32 `0x126485B2`, h23 `0x6485B2`, x0 `0x00E485B2` |
| scale stream: same record, g = 719, data `7F 7E 80 7F 7D` (5 B, exercises the tail) | H32 `0x8A4976D8`, h23 `0x4976D8`, x0 `0x00C976D8` |

### 4.6 Encoder order (normative)

Per record (source = the WEX1 slab of (layer, expert)):
1. Copy the source slab. CZ build: apply Z, count mapped nibbles.
2. Per matrix: gate (count blocks with M == 5); if it passes set CEIL[m] and apply T (count transformed blocks and even blocks skipped at s = 255).
3. `anchor[m]` = rule g on the coded scales.
4. Header bytes 24..31 (anchors, flags, owner) are now fixed. For each stream in g order: forward pass for contexts; `h23` over ident + the stream's coded data; encode backwards from `x0 = L + h23`; flush 4 B; u16 length (<= 65535).
5. Header, zero padding. If `total_bytes >= aligned_raw`, write the entry raw (codec 0, source bytes).
6. Verify with the CPU reference decoder (section 4.7) against the source (non-CZ) or Z(source) (CZ) before the record is written.

### 4.7 Decoder order (normative, CPU reference and GPU)

Any failure: status -1, destination untouched (CPU decodes into scratch and memcpys on success; GPU commit kernel copies only when the error word is 0, both as today).
1. Entry rules (codec 5, sizes); read the record at the slab end. Caller (engine/store): **owner check** `record bytes 28..31 == layer << 16 | expert` before decoding (catches a record served for the wrong expert, e.g. by a RAM-cache or table bug).
2. Parse (section 3), model id registered.
3. Scale streams -> coded scales (CPU: into the scratch scale region; GPU: into the raw buffer's scale region); per stream: exact length, `final == L + h23(stream's coded scale bytes)`.
4. Weight streams, per block: decode 32 symbols with contexts from the coded scale and coded nibbles; CZ record: no symbol 8; per stream: `final == L + h23(stream's coded packed bytes)`.
5. T^-1 for every CEIL[m] matrix: blocks with coded max 5 get D and `scale - 1` (`cs == 0` -> invalid). GPU: in the weight kernel after the block's hash words are folded, before the block's uint4 store; the scale decrements are written **in place** into the same lane's own scale bytes after those blocks are decoded (section 9.4). CPU: one pass over the scratch slab after all streams decoded.
6. Commit.

Order invariants: hashes, contexts and CZ checks use the coded domain; T^-1 never feeds back into contexts or hashes.

### 4.8 Error reporting path

| layer | report |
|---|---|
| CPU `wex_nr2_decode` / `wex_decode_nr2_record` (codec 5) | returns -1 (hash, CZ, ceil underflow, structural all -1); slab untouched |
| GPU kernels | `atomicOr(error, bit)`: 1 structural (today), 2 hash mismatch, 4 ceil `cs == 0`, 8 CZ nibble 8 |
| helper `wex_cuda_bridge.cu` | any nonzero error word -> -1 (single, device, batch status[i]) as today; additionally the first time a decode fails with bits 2/4/8 the helper prints once per process `[wex] WNR3 record failed its integrity check (error 0x%x); rejected` |
| engine wrappers `wex_decode.c` | unchanged (-1 passes through; host path prints its existing "CUDA reconstruction failed ... using CPU fallback" once, then the CPU decoder rejects the same record -> -1) |
| engine `deepseek_v4.c` | -1 -> existing failure handling: GPU-direct fill not published, `v4_wex_direct_failures`++, expert served by the host path / safetensors; mode 0 falls back to safetensors. **Change:** a RAM-cache slot whose record failed to decode is dropped (today a cache hit that fails stays cached and fails again every time, section 9.7) |
| owner mismatch | engine: same as a decode failure (counted in `v4_wex_direct_failures`), record never decoded |

---

## 5. Model (WNM2 blob, retrained)

- **Format unchanged** (WNM2, 567 weight contexts x 3 matrices, 3 scale CDFs, 59,408 B, FNV-1a id). No WNM3.
- **Retrain: yes, always.** The WNR2 production model (trained on plain data with median anchors) is lossless on WNR3 records but loses the gain (an anchor misaligned by one step cost +8.33% of raw in the lab control). The trainer counts in the coded domain of the profile being built: source -> Z (CZ build) -> per-matrix gate + T -> rule-g anchors on the coded scales -> `wex_nr2_count_anchored` (the current `wex_nr2_count` with the anchor passed in) -> `wex_nr2_model_build` (quantisation unchanged). Pass 1 of the builder counts every present record (as the W1 build did; 7,680 on V4.1), pass 2 encodes with that model.
- A CZ model and a non-CZ model differ (the CZ model gives symbol 8 the minimum frequency 1/16384, cost ~0.0001 bit/w); each container embeds its own.
- Lab models are **test fixtures only**: `ceil-kaira/model_CFG_rule_g.wnm2` (id `0x299DC586`, ceil + rule g, no CZ, 80 experts) and `model_CF_median.wnm2` (id `0xC2A18E9C`, ceil + median anchor; ids in `ceil_kaira_records.json`).

---

## 6. Expected sizes (V4.1, per record, estimates from the measured basis)

| profile | mean record | of raw | container (7,680) | source |
|---|---|---|---|---|
| production WNR2 | 16,914,680 B | 0.8997 | 129.9 GB | build W1 |
| WNR3 without CZ | ~16,579,600 B | ~0.8819 | ~127.3 GB | CFG 16,578,252.8 + hash 1,352 |
| WNR3 with CZ (v1 profile) | ~16,064,800 B | ~0.8545 | ~123.4 GB | minus 514,848 (CR, g0 basis; never measured with rule g) |

Max record: CFG 17,985,536 B on the 12 largest production records; the population max under WNR3 is unmeasured (RAM-cache slot size and helper workspaces already size themselves from the largest record / aligned raw at open). The production model is trained on 7,680 records, the lab on 80: the full-build numbers can differ slightly. **A full container does not fit any drive today (U: ~13, C: ~58, D: ~92, E: ~89 GB free): no full encode in this plan.** *(State before the build. The full container was built on 2026-11-10 after space was freed: 123,306,192,896 B, 0.854 of WEX1, [WNR3-PRODUCTION-20261110.md](reports/WNR3-PRODUCTION-20261110.md).)*

---

## 7. Compatibility checks and gates

### 7.1 Canonicalized-WEX1 rule

WNR3 non-CZ records decode byte-exact to the WEX1 slab. WNR3 CZ records decode to Z(WEX1). Every gate or tool that compares decoded bytes with WEX1 must, for a CZ record, compare with Z(WEX1) and additionally check that `decoded XOR WEX1` is nonzero only in nibbles where WEX1 holds 0x8, with exactly as many such nibbles as WEX1 has 0x8 nibbles. Affected:
- the builder's per-record CPU verify (4.6 step 6) and its readback of every 43rd record;
- `check_container`-style validation (`canonical_zero` argument 1 = compare with Z(source));
- test bins for `test_decode_batch.exe` (bin layout u32 model_bytes + blob + u32 count + per record {72-B entry, u32 raw_bytes, raw slab, record}): for CZ records the raw slab field holds **Z(WEX1)**; `test_decode_batch.cu` itself needs no change (it does not inspect the codec);
- sha256 of decoded slabs / `.verified` markers: the builder records sha256 of Z(source) per record and for the whole container its own file sha256;
- mirror pairing: only byte copies of the same CZ container pair (unchanged rule).

### 7.2 Gates (each names what it unlocks)

| gate | what | pass | unlocks |
|---|---|---|---|
| G1 | CPU unit tests (`test_wex_format`, `test_expert_store_wex`, section 10) | all pass, incl. hash vectors and WNR2 golden | P2 |
| G2 | **WNR2 unchanged**: (a) WNR2 encoder golden: FNV-1a of `wex_nr2_encode` output on fixed synthetic inputs equals the value recorded on master before any change; (b) new helper on the 64-record V4.1 WNR2 sample (`sample_records.py` from the WNR2 container on U:/E: + WEX1 U:, seed 20261107) | (a) equal; (b) 512/512 byte-equal, corruption statuses `[0,0,-1,0,-1,0]`, batch-16 p50 within +-3% of the deployed helper (sha cceb4fb5...) measured in the same locked session | helper merge |
| G3 | conformance with the lab: new encoder, non-CZ, with the lab model `model_CFG_rule_g.wnm2`, on the 40 held-out + 12 tail experts | anchors == lab rule-g anchors per matrix; mean total within `16,578,252.8 + 1,352 +- 1,500 B`; 52/52 CPU byte-exact vs WEX1; 0 gate failures | GPU work |
| G4 | GPU WNR3 (5090, heavy.lock): bins of G3's 52 records non-CZ and CZ (CZ model trained on the same 80) | all batch sizes byte-exact vs WEX1 / Z(WEX1); corruption statuses as G2; CPU vs GPU differential on >= 300 mutated records (same status, same output when 0); ptxas 0 spill stores/loads; batch-16 p50 <= WNR2 + 5% on the same GPU | engine work |
| G5 | integrity audit: single-bit flips in weight and scale payloads of 2 experts, 3,000 flips per matrix (w1, w2) and form, WNR2-rule vs WNR3 | WNR3 accepted 0 (expected <= 0.01); WNR2 rate reported for comparison | CZ/hash sign-off |
| G6 | CZ correctness on G4's bins | decoded == Z(WEX1); differences vs WEX1 exactly on the 0x8 nibbles; per-record count == source count | G7 |
| G7 | **end-to-end** (engine built from the worktree, run outside app/, heavy.lock, the application not running): two subset containers built from WEX1 with only layers {0, 13, 26, 39} present (other entries all-zero -> safetensors), A = WNR3 non-CZ, B = WNR3 CZ (~12.4 GB each, on D:); 4 W3-style prompts x >= 256 greedy tokens, `V4_LOGITS_DUMP` | A-vs-A identical first (else inconclusive, report); then A-vs-B logits files byte-identical at every step; A tokens == a WNR2 run's tokens | owner decision on a full CZ build |
| G8 | full build + W3-style procedure (all records CPU-verified, every 43rd GPU/CPU compare, readbacks, sha256, 64-record GPU check, A/B speed) | as W1 | **owner decision** (disk space) |

---

## 8. Not tested / open (carried into the plan)

- MurmurHash3 detection rates (G5); GPU cost of the hash (G4 timing).
- In-place scale fix-up (section 9.4) differs from the lab kernel (separate coded-scale buffer); timing re-measured in G4. *(Done: per-block in-place decrement, no emask; WNR3-CZ batch-16 0.366-0.373 ms/record vs WNR2 0.376-0.377, WNR3-VERIFY section 3.)*
- Lab ceil kernel handles `Cw <= 2048` only (64-bit block mask, 4 uint4); the implemented per-block write has no block mask, so any legal Cw works; `Cw > 2048` is tested on CPU/GPU in G1/G4 with synthetic records only.
- Rule g and CZ on the full 7,680-record population; WNR3 CZ size measured only via the g0 basis.
- Masked DeepGEMM path with CZ (refused by policy, 4.4); Vulkan never reads WEX.
- MTP experts: not in the container.

---

## 9. Implementation plan (engine source, in worktree `<worktree>`)

*Historical: sections 9 to 11 are the plan as written before the code, with line numbers of the engine source of that time. The work was done on 2026-11-10; the code is in `c/`, and the errata at the top of this file say where it differs from the plan.*

P0: `git -C <TREE> worktree add <worktree> -b wnr3-20261110 master`. Before any edit, record the WNR2 golden (G2a) with master code. Commits stay local, never pushed. Builds from a fresh copy of the worktree (tests may rebuild `backend_loader.o` as HIP in place; grep `COLI_HIP_RUNTIME_DIR=0` before any install). Never touch the live `src\`, `app\` and the application's configuration file.

### 9.1 `wex_neural_rans.h` (CPU codec, shared helpers)

- Constants of 2.1. `#ifdef __CUDACC__ #define WEX_NR_HD __host__ __device__ #else #define WEX_NR_HD #endif`.
- `WEX_NR_HD static inline uint32_t wex_nr3_h32_block(uint32_t h, uint32_t k)`, `wex_nr3_h32_tail(uint32_t h, uint32_t k)`, `wex_nr3_h32_final(uint32_t h, uint32_t len)` (MurmurHash3 steps of 4.5; shared by CPU and the CUDA kernels).
- `static inline uint32_t wex_nr3_stream_h23(const uint32_t ident[4], const unsigned char *data, uint32_t n)`: ident blocks, LE data blocks, tail, `len = 16 + n`, mask.
- `WexNr2Record`: add `uint32_t magic, owner, word24; unsigned char flags;` (WNR2: magic WNR2, rest 0).
- Parse: rename the body of `wex_nr2_parse` to `wex_nrx_parse(record, total, layout, expected_model, magic, view)`; for WNR2 keep `record[27] == 0 && u32@28 == 0`, for WNR3 require `flags & 0xF0 == 0` and fill flags/owner/word24. Wrappers `wex_nr2_parse` (signature unchanged) and `wex_nr3_parse`.
- `wex_nr2_decode`: unchanged for magic WNR2. For WNR3: after each stream, `expect = L + wex_nr3_stream_h23({model, owner, stream, word24}, out_range, bytes)` and compare instead of `L`; in the weight loop reject `symbol == 8` when CZ; at the end call `wex_nr3_ceil_inverse(layout, flags & 7, scales_out, weights_out)`.
- `wex_nr3_anchor(const unsigned char *scales, uint32_t n)`: rule g (4.3), copy of the lab code.
- `wex_nr3_canonical_zero(unsigned char *weights, size_t bytes) -> uint64_t mapped`.
- `wex_nr3_ceil_forward(const WexNr2Layout*, unsigned char *const scales[3], unsigned char *const weights[3], uint32_t stats[3][3]) -> ceil_mask` (in place on the caller's copy; per-matrix gate, skip-255 rule; stats = transformed, max-5 blocks, skipped-255 per m).
- `wex_nr3_ceil_inverse(const WexNr2Layout*, uint32_t mask, scales, weights) -> 0 / -1` (`cs == 0` in a max-5 block -> -1).
- Count: body of `wex_nr2_count` -> `wex_nr2_count_anchored(layout, scales, weights, const unsigned char anchor[3], wc, sc)`; `wex_nr2_count` computes medians and calls it (results unchanged).
- Encode: body of `wex_nr2_encode` -> `wex_nrx_encode(model, layout, scales, weights, wchunk, schunk, const unsigned char anchor[3], uint32_t magic, uint32_t flags, uint32_t owner, out, cap, written)`; for WNR3 each stream starts at `L + h23` computed over the input range (scales `+first, count`; weights `+first/2, count/2`) and the header writes magic/anchors/flags/owner. `wex_nr2_encode` = medians, WNR2, 0, 0 (byte-identical, G2a). `wex_nr3_encode(model, layout, coded scales, coded weights, wchunk, schunk, flags, owner, ...)` = rule-g anchors from the coded scales, WNR3.

### 9.2 `wex_format.h`

- Constants 32, 64, 5; `static inline int wex_entry_nr_model(const WexEntry *e) { return e && (e->flags == WEX_ENTRY_NEURAL_RANS2 || e->flags == WEX_ENTRY_NEURAL_RANS3); }`.
- `wex_validate_header`: v3 mask adds 32|64; reject 32 with 8 or 16; reject 64 without 32.
- `wex_record_read_offset`: codec 5 like codec 4.
- `wex_nr2_model_location`: accept flag 16 or 32.
- `wex_validate_entry`: codec-5 branch (2.2).
- `wex_decode_nr2_record`: accept codec 4 or 5, parse with the magic of the codec (`wex_nr2_parse` / `wex_nr3_parse`), decode (inverse inside `wex_nr2_decode`).
- `wex_decode_record`: refuse codec 5 like codec 4 ("needs its model").
- `static inline int wex_nr3_owner_ok(const WexEntry *e, const unsigned char *record, uint32_t layer, uint32_t expert)`: 1 unless codec 5 and `u32@28 != layer << 16 | expert`.

### 9.3 `wex_decode.c` / `wex_decode.h`

- `wex_cuda_initialize`: remember `features() & 16` in a static `wex_cuda_nr3`.
- New `int coli_wex_cuda_nr3_available(void)` (declare in `wex_decode.h`): `wex_cuda_model_fn && wex_cuda_nr3`.
- `coli_wex_cuda_decode_to_device`: accept codec 5; return 1 when `!wex_cuda_nr3`.
- `coli_wex_decode_record`: codec 5 resolves its model like codec 4 (bytes 4..7); helper only when `in_helper && wex_cuda_nr3`; CPU fallback `wex_decode_nr2_record`.
- `coli_wex_cuda_decode_batch_to_device`: unchanged (callers pass codec 5 only when NR3 is available; the engine guard enforces it).

### 9.4 `wex_cuda.cuh`

- `WexCudaNr2Plan` += `uint32_t nr3, ceil_mask, cz, owner, word24`.
- `wex_cuda_plan_nr2`: accept codec 4 or 5 (`wex_entry_nr_model`), parse with the codec's magic, fill the new fields (0 for WNR2).
- Error bits `WEX_CUDA_ERR_STRUCT 1, _HASH 2, _CEIL 4, _CZ 8`.
- `wex_cuda_nr2_scales`: if `plan.nr3`, after the stream's last symbol hash `out[0..count)` (byte loads composed LE, alignment-free; the lane re-reads its own writes) with ident `{model_id, owner, stream, word24}`; compare `state == L + h23` (`ERR_HASH`), keep `s.pos != end` / initial-state errors as `ERR_STRUCT`. WNR2: unchanged code path (uniform branch on `plan.nr3`).
- `wex_cuda_nr2_weights<Lookup>`, if `plan.nr3` (uniform branch):
  - at stream open `h = seed` folded with the 4 ident words;
  - after the 32 symbols of a block: fold `word[0..3]` into h (before D); CZ: `t = w ^ 0x88888888; ((t - 0x11111111) & ~t & 0x88888888) != 0` on the 4 words -> `ERR_CZ`;
  - if `(plan.ceil_mask >> m) & 1` and `maximum == 5`: `scale == 0` -> `ERR_CEIL`; `word[i] = wex_cuda_ceil_dbl8(word[i])` (lab SWAR) and **`scales[block] = scale - 1` in place, right away** (the block's scale was already read for its context; `next_scale = scales[block + 1]` was read before; scale ranges of different weight streams are disjoint). The `scales` pointer must stay a plain pointer (no `__ldg` / `const __restrict__`).
  - *(As implemented. The frozen plan had a u64 `emask` flushed after every 64-block group with the lab `wex_cuda_ceil_fix_scales`; the per-block write is simpler, needs no flush and has no `Cw` limit.)*
  - stream end: `h = wex_nr3_h32_final(h, 16 + count/2)`; `state != L + (h & mask)` -> `ERR_HASH`.
  No workspace, plan-layout or commit change (the lab's separate coded-scale buffer `raw + sc_aligned` is not needed).

### 9.5 `c/tools/wex_cuda_bridge.cu`

- `WEX_CUDA_FEATURE_NR3 16` in `wex_cuda_helper_features()`; header comment.
- `wex_cuda_decode_to_host`, `wex_cuda_decode_to_device`, `wex_cuda_decode_batch_to_device`: replace `entry->flags==WEX_ENTRY_NEURAL_RANS2` by `wex_entry_nr_model(entry)` (3 sites).
- `wex_cuda_nr2_decode` / batch result: on a nonzero error word with bits 2/4/8, print the once-per-process line of 4.8 (static atomic flag).
- `wex_cuda_register_model`: unchanged.

### 9.6 `expert_store_wex.c`

- Model registration at open: `hdr->flags & (WEX_HEADER_NEURAL_RANS2 | WEX_HEADER_NEURAL_RANS3)` (v: line 652).
- `wex_load_slot`: before `coli_wex_decode_record`, `wex_nr3_owner_ok(entry, slot->slab + read_offset, layer, expert)` else fail like a decode failure.

### 9.7 `deepseek_v4.c` (line numbers of the plan's engine source)

- `v4_wex_load_manifest` (v4:13718-13738): model registration for flag 16 or 32; for 32 require `coli_wex_cuda_nr3_available()`, else "[v4_wex] WNR3 container needs kaira_wex_cuda.dll with WNR3 support (helper feature 16) next to the engine" and fail; for 64 the policy of 4.4 (`getenv("DSV4_CUDA_MOE_MASKED")`).
- `coli_v4_expert_store_wex_entry` (v4:14588): also accept codec 5 when `coli_wex_cuda_nr3_available()`.
- RAM cache sizing / prewarm filters (v4:14756, v4:14900): codec 3/4 -> 3/4/5.
- Batch filter (v4:19600): accept codec 4 or 5.
- Owner check (`wex_nr3_owner_ok`) before every decode of a record for key K: `v4_wex_fill_mirror` (v4:19354, after the RAM-cache hit or disk read; failure -> same path as status -1), the batch completion loop before `coli_wex_cuda_decode_batch_to_device` (v4:19701 region; a failing record is left out and gets `ok = 0`), `v4_read_expert_record` (v4:14468, before `coli_wex_decode_record`).
- RAM cache: `coli_v4_wex_crc_done` (v4:14866) also drops a READY slot when `filled == 0` (index cell cleared); callers pass `status == 0` instead of `hit || status == 0` (v4:19360 and the batch loop after v4:19701), so a cached record that fails its hash is re-read from disk next time.
- Open log: "[v4_wex] Container v3 active ..." gains the codec name (WNR2/WNR3) and "value-exact" for flag 64.

### 9.8 Encoder tool (new, in the worktree)

The WNR2 encoder lives only in the lab (`lab\wex-gpu-direct-20261019\nr2\encode_nr2.cu`, `build_nr2_full.py`, `nr2lib.py`, `check_container.c`). WNR3 gets a CPU-only tool in `src/colibri/c/tools/` (V4 CPU encode was 148 ms/expert; V4.1 ~0.25 s count+encode + ~0.5 s verify per record; 16 threads -> a 52-record sample in seconds, a full build ~8 min):
- `c/tools/wex_nr3_tools.c` -> `wex_nr3_tools.dll` (MSYS2 gcc `-O2 -shared`, includes `wex_format.h`): `wex_nr3_tools_abi()` = 1; `wex_nr3_prepare(entry, raw_slab, cz, coded_out, stats_out)` (copy, Z, gate + T, anchors, stats); `wex_nr3_count(entry, coded, wc, sc)`; `wex_nr3_build_model(wc, sc, blob)`; `wex_nr3_model_id(blob, &id)`; `wex_nr3_encode(blob, entry, coded, flags, owner, wchunk, schunk, out, cap, &written)` (padded record); `wex_nr3_verify(blob, entry, record, expected_slab, owner)` (CPU decode into scratch + memcmp; expected = source or Z(source)); `wex_nr3_check_container(header_page, model, entries, file_bytes)` (header/table/model validators of `wex_format.h`).
- `c/tools/build_nr3.py` (port of `build_nr2_full.py`, ctypes inline): source WEX1 (`U:\kaira_wex\v41-reap\v4_flash_experts.wex`), chunks 2048:512; modes `--bins` (list of (layer, expert) or count+seed -> `test_decode_batch` bin with WEX1 or Z(WEX1) as expected slab; optional `--model` to use a fixed blob, e.g. the lab CFG model for G3), `--subset-layers` (container with only those layers present, for G7), `--full` (refuses without `--allow-full` and a free-space check of 0.90 x source + margin). Every record CPU-verified before writing; build report JSON with per-record ceil_mask, gate failures, transformed/skipped counts, anchors, cz_count, sizes, sha256 of the expected slab; `.partial` then `os.replace`.
- Large outputs (bins, subset containers) go to `D:\kaira_wnr3_lab\` or an untracked worktree folder excluded in `.git/info/exclude`, never under the OneDrive-synced `lab\`.

### 9.9 What the lab code already gives

| lab item | reuse | limits to fix |
|---|---|---|
| `ceil-kaira/wex_cuda_ceil.cuh` (`wex_cuda_ceil_dbl8`, `wex_cuda_ceil_fix_scales`, emask logic in `wex_cuda_nr2_weights<1>`) | copy dbl8 and fix_scales verbatim; the emask insertion point (after the 32nd symbol, before the uint4 store) | inverts every matrix (no per-matrix bit), no `cs == 0` check, `Cw <= 2048` only, separate coded-scale buffer, no hash/CZ; 91 regs / 0 spills measured |
| `ceil-kaira/wex_cuda_bridge_ceil.cu`, `gen_ceil_cuda.py`, `build_ceil.cmd` | build command line (nvcc 13.4, sm_120, `-Xptxas -v`) | workspace enlargement not needed with in-place fix-up |
| `ceil-kaira/make_ceil_records.py` (`ceil_forward_packed`, `ceil_inverse_packed`) | independent NumPy reference for T / T^-1 in G3 (cross-check the C transform on the 52 records) | record-level `assert bmax >= 6`; inverse rejects nothing -> use only on V4.1 data |
| `g0/wex_neural_rans_g0.h` (rule g), `g0/g0_enc.c`, `g0/g0_dec.c`, `g0/t2_g0.py` (`Shards`, `picks`, `slab`, ENTRY struct) | rule g code -> `wex_nr3_anchor`; expert sampling (80 train even, 40 test `4L+1/4L+3`, 12 tail from `t1_record_sizes.json`) for G3/G4 | scratch copy of an older header |
| `ceil-kaira/model_CFG_rule_g.wnm2` (id 0x299DC586), `ceil_kaira_records.json` | G3 fixture and expected sizes per record | 80-expert model |
| `records_A.bin`, `records_CFG.bin`, `records_CFG_all52.bin` in a temporary folder (`<TMP>/data/g0bins`) (11 GB, still present 2026-11-10, ephemeral) | the WEX1 raw slabs of the 40 + 12 experts without re-reading the 144 GB container | may disappear; `build_nr3.py --bins` regenerates from WEX1 |
| `lab\wex-prod-20261105\test_decode_batch.cu` + `lab\v41-gpu-20261107\w1\build_helper.cmd` | GPU byte-exact + timing harness, unchanged (codec-agnostic) | none |
| `lab\v41-gpu-20261107\w1\sample_records.py` | G2 WNR2 regression sample | WNR2 only (fine for G2) |
| `zero/` (harness, `ZERO-RESULTS.md`) | CZ kernel evidence; rerun only if kernels change | masked path unproven |
| team3 `wnr3-spec/wnr3_ref.py` | independent Python MurmurHash3 (SMHasher check) to cross-check the C vectors | its stream vectors use another ident layout: use section 4.5's |
| team 2 `audit_corrupt.py` protocol | G5 design | other hash; re-run with the C decoder |

---

## 10. Tests to add

- `c/tests/test_wex_format.c`
  - `test_nr3_hash_vectors`: SMHasher `0xB0F57EE3` via `wex_nr3_h32_*`, the function vectors and both stream vectors of 4.5.
  - `test_nr2_golden`: FNV-1a of `wex_nr2_encode` output for 3 seeded synthetic records (values recorded on master in P0) -> G2a.
  - `test_nr3_codec`: chunk pairs 512:256, 2048:512, 4096:2048, 16384:16384; synthetic matrices with even blocks (ceil), even blocks at s = 254 (coded 255) and s = 255 (skipped), blocks with max 0..4, all-zero blocks, one matrix with a max-5 block (gate fails -> CEIL bit 0, others 1), -0 everywhere (CZ), scale counts not a multiple of 4 (hash tail), `Cw > 2048` (emask flush); CZ off/on; decode == source / Z(source); anchors == rule g of the coded scales; raw fallback for uniform random nibbles.
  - corruptions (each -1, slab untouched): a single-bit sign flip that the WNR2 rule accepts (search one on a WNR2 encode of the same data; WNR3 must reject), flags bits 4-7, a CEIL bit flipped, an anchor byte +1, owner changed, model id of another registered model, two equal-length streams swapped, a crafted CZ record containing symbol 8, a crafted ceil record with a max-5 block at coded scale 0, a WNR3 record under a codec-4 entry and a WNR2 record under a codec-5 entry, truncation, nonzero padding.
  - header/entry rules: flag 32 accepted (v3), 32 with 16 or 8 refused, 64 without 32 refused, codec 5 without flag 32 refused, `wex_nr2_model_location` with flag 32.
- `c/tests/test_expert_store_wex.c` `test_nr3_container`: flags 33 and 97, model registered at open, exact (and canonical-exact) decode through the store, foreign model refused, owner mismatch (record bytes of another expert copied into an entry) refused, WNR2 `test_nr2_container` unchanged.
- GPU (lab-style, not in `make test`): `test_decode_batch.exe` with the new helper on G2's WNR2 sample and G4's WNR3 bins; a mutation runner in `build_nr3.py --fuzz` writing >= 300 corrupted records per profile for the CPU/GPU differential; `nvcc -Xptxas -v` log kept with the build.
- Tool tests: `build_nr3.py --bins` on 4 records with `--model` = lab CFG reproduces the lab per-record sizes + hash cost (G3 smoke); `--subset-layers` container passes `wex_nr3_check_container`.

---

## 11. Rollout order

1. **P0** worktree + golden.
2. **P1** 9.1 + 9.2 + G1 tests (CPU only).
3. **P2** 9.3 + 9.6 + store test; 9.8 tool; **G3** (CPU, no lock needed; reads U: WEX1 or the lab bins).
4. **P3** 9.4 + 9.5; helper built from a fresh workspace; **G2b, G4, G5, G6** on the 5090 (heavy.lock via `lab\wnr3-5090-20261107\lock.sh`, only with no application `deepseek_v4.exe` running; poll every 2 min up to 3 h, else report not run).
5. **P4** 9.7; engine built from a fresh workspace; **G7** (two ~12.4 GB subset containers on D:, deleted after the run unless the owner keeps them).
6. **P5** (owner decision) free ~124-128 GB, full build (G8), deploy like W3. Until then nothing in `app\` changes and the live container stays WNR2. *Done 2026-11-10 (G7 via the lab switch instead of subset containers; [WNR3-PRODUCTION-20261110.md](reports/WNR3-PRODUCTION-20261110.md)).*

---

## Document history

| Date | Change |
|---|---|
| 2026-11-10 | FINAL v1: frozen, implemented in the engine (branch wnr3-20261110) and put in production for DeepSeek V4.1 |
| 2026-11-11 | The WNR3 records moved, unchanged, into WEX4 containers ([WEX4-SPEC.md](WEX4-SPEC.md)) |
| 2026-11-13 | Version header; the implementation notes are named errata. No format change |
