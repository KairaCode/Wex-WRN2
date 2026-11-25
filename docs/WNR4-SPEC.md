# WNR4 specification (v0.1, updated 2026-11-25)

**Status (2026-11-25):** WNR4 (WEX4 codec 8, WNM4 model blob kind 3) defines four record kinds — **0 = MX** (MXFP4), **1 = NV** (NVFP4), **2 = GG** (GGUF 2-bit/4-bit slices), and **3 = I4** (int4-g64) — implemented on CPU and CUDA in `wnr4/` (helper ABI 6). Whole-model WEX4+WNR4 containers have been built for DeepSeek V4.1 REAP-192 (MX, 122.87 GB, 0.851), Qwen3.8-Flash-Next MXFP4 (MX, 56.85 GB, 0.886), Qwen3.8-Flash-Next INT4-g64 (I4, 53.29 GB, 0.784, verified 24,576/24,576 byte-exact), and Qwen3.8-Flash-Next UD-IQ1_M (GG, 33.38 GB, 0.957, verified 24,576/24,576 byte-exact). Engine-side CPU/CUDA decode for kind MX (codec 8, helper feature 64, `wex4_wnr4_experts` opt-in) is implemented in the maintainer's DeepSeek engine tree; the independent Go (`impl/go-wex4`) and Rust (`impl/rust-wex4`) readers decode kinds GG and I4. [WEX4-SPEC.md](WEX4-SPEC.md) section 7.3 specifies codec 8 and blob kind 3 for non-EXPERT records (GG, I4); formalising a `required_features` bit for EXPERT records (MX, NV) and adding codec-8 conformance vectors to `wex4/vectors/` remain open (section 8). WNR3 (codec 5) is unchanged; codec 7 stays reserved. The name WNR4 was used before for a smaller proposal, the "WNR4 record" of WEX4-SPEC section 12.2 (three changes to WNR3's stream skeleton, not adopted); this specification replaces it. Measurements and design rationale: [research/wnr4-20261121](../research/wnr4-20261121/README.md), [research/wnr4-gg-20261121](../research/wnr4-gg-20261121/README.md), [research/wnr4-i4-20261122](../research/wnr4-i4-20261122/README.md), and [reports/wnr4-research-20261121.md](reports/wnr4-research-20261121.md).

### Summary of WNR4 kinds

| Kind | Name | Target weights | WNM4 blob size | Scale model | Typical ratio (vs raw) | RTX 5090 GPU decode (kernel) | Independent readers |
|---:|---|---|---:|---|---|---:|---|
| 0 | **MX** | MXFP4 slabs (1 UE8M0 / 32 nibbles) | 59,408 B (`pb=1`) or 290,744 B (`pb=5`) | order-0 UE8M0 + rule-g class | **0.849–0.886** (V4.1 0.851, Q38 0.886) | **188–209 GB/s** (0.19 ms @ G=32) | C (`wnr4.h`), CUDA (`wnr4_cuda.cuh`), DeepSeek engine (feat 64) |
| 1 | **NV** | NVFP4 slabs (1 E4M3 / 16 nibbles) | 6.52 MB (`pb=1`) or 6.95 MB (`pb=5`) | row-above delta + up-left/up-right | **0.924** (Q38 NVFP4) | **140–149 GB/s** (0.25 ms @ G=32) | C (`wnr4.h`), CUDA (`wnr4_cuda.cuh`) |
| 2 | **GG** | GGUF routed experts (`Q2_0`, `IQ4_NL`; others verbatim) | 36,644 B | f16 scale high/low byte classes | **0.957** (UD-IQ1_M, 33.38 GB) | **133.0 GB/s** whole / 57.0 coded (G=32) | C, CUDA, Python (`wnr4gg.py`), **Go**, **Rust** |
| 3 | **I4** | int4-g64 experts (signed int4 + f32 scale / 64) | 196,688 B | FP8 block-scaled lattice `lat(k, B)` + escape | **0.784** (Q38 int4, 53.29 GB) | **156.9 GB/s** (G=32) / **193.2 GB/s** (G=16) | C, CUDA, Python (`wnr4i4.py`), **Go**, **Rust** |

**One paragraph.** WNR4 is a lossless record codec for MXFP4 and NVFP4 routed-expert slabs, built so that a GPU can decode many records in one launch at tens of GB/s. It keeps WNR3's weight model (block-scale class relative to a rule-g anchor, previous magnitude, running maximum; 16-symbol 14-bit tables; canonical zero; ceil form for MXFP4), optionally refined by the nibble's position in the last four of its block, codes NVFP4 scales as a delta from the scale above with the up-left/up-right deltas as context, and changes the skeleton: a record is a list of *tiles*; a tile is 32 rANS coders (*lanes*) that share one 16-bit word stream, decoded by one warp in lockstep; the last 4 nibbles of each lane ride in its initial state. Integrity moves from a 23-bit hash per 1 KB stream to a 32-bit hash per tile plus checked end states and an exact word count. Kind 2 (GG, section 5) puts the same skeleton under another model, for the 18-byte blocks of the GGUF Q2_0 and IQ4_NL slices of a GGUF routed-expert record; kind 3 (I4, section 6) for the signed int4 codes and f32 group scales of int4-g64 experts, the scales coded as indices into a lattice that an FP8 block-scaled source produces.

## 1. Terms

- **Slab:** the decoded expert, as WEX3/WEX4 lay it out: per matrix m (0 gate w1, 1 down w2, 2 up w3) `scount[m]` scale bytes at `soff[m]`, then packed nibbles at `woff[m]` (two per byte, symbol 2i in the low nibble). `scols[m]` = scales per row.
- **Kind:** 0 = MX (MXFP4: one UE8M0 scale per 32 nibbles), 1 = NV (NVFP4: one E4M3 scale per 16 nibbles). `B` = 32 or 16 nibbles per block, `spc` = 1 or 2 scales per cell. 2 = GG (GGUF expert slices): it has its own geometry, model blob and phases (section 5) and shares with MX and NV the rANS below, the record header of section 3 and, from section 4, the words, the tail in the state, the hash and the end states. 3 = I4 (int4-g64 experts): likewise its own geometry, model blob and phases (section 6). The other terms of this section and sections 2 to 4 are those of kinds 0 and 1.
- **Cell:** 32 consecutive nibbles of a matrix (16 bytes) with their `spc` scales. Matrix m has `cells[m] = scount[m] / spc` cells; `scount[m] % spc == 0`, `scount[m] % scols[m] == 0` and `cells[m] < 2^27` are required, and the six regions of the slab (three scale ranges, three nibble ranges) must not overlap.
- **Coded domain:** as WNR3 section 1: canonical zero when flag CZ (every nibble 8 coded as 0, the decoder refuses a decoded 8), and for MX the WNR3 ceil form per matrix (flag bit m; gate and transform exactly WNR3 section 4.2). NV never uses the ceil form.
- **Class** of a scale s in matrix m: `clamp(s - anchor[m], -R, R) + R`, R = 3 (MX, 7 classes) or 6 (NV, 13 classes). An encoder writes `anchor[m]` = rule g (WNR3 4.3) on the coded scales of m; a decoder takes the stored byte as it is (any value 0..255, as WNR3 erratum 14).
- **rANS:** 32-bit state x, `L = 2^16`, precision 14 (`T = 16384`), 16-bit little-endian words. Decode of one symbol with table `cdf`: `slot = x & (T-1)`, `s` = largest symbol with `cdf[s] <= slot`, `x = (cdf[s+1]-cdf[s]) * (x >> 14) + slot - cdf[s]`; then, if `x < L`, `x = x << 16 | next word`. One word at most per symbol.

## 2. Model blob (WNM4)

Little-endian. Header 8 x u32: `0x344D4E57` ("WNM4"), version 1, kind, 14, `wctx = 81 * classes * pb` (classes 7 MX, 13 NV), `sctx` (1 MX, 4160 NV), `pb` (position buckets, 1 or 5), 0. Then `3 * wctx` weight tables of 17 u16 (`cdf[0] = 0 .. cdf[16] = 16384`, strictly increasing), row `m * wctx + ((cls * 9 + pm) * 9 + mx) * pb + b`, with `b = 0` when pb = 1, else `b = 0` for block positions `p < B - 4` and `b = p - B + 5` (1..4) for the last four; then `3 * sctx` scale tables of 257 u16, row `m * sctx + ctx`. Size: MX 59,408 B (pb 1) or 290,744 B (pb 5); NV 6,522,158 B (pb 1) or 6,951,782 B (pb 5). Id = FNV-1a of the whole blob, 0 replaced by 1. Training (not normative: any tables decode): the reference trainer counts the coded domain with matrix-wide contexts (the NV scale context as in section 4 but with the row above available everywhere except the matrix's first row, and all 32 nibbles of every cell, the tail nibbles included), then quantizes by WNR3's rule (`wex_nr2_quantize`: every symbol at least 1, remainders by size). A WEX4 container would carry it as a blob of kind 3 (WEX4-SPEC 5.1 and 7.3, draft: MX and NV records still need a required feature bit), the kind of every WNM4 blob. The WNM4 model of kind 2 (GG) has another layout: section 5.2.

## 3. Record

| bytes | field |
|---|---|
| 0..3 | magic `0x34524E57` ("WNR4") |
| 4..7 | model id (!= 0, the WNM4 id) |
| 8..11 | owner `layer << 16 \| expert` |
| 12 | kind |
| 13 | flags: bits 0-2 ceil form of matrix m (MX only), bit 3 CZ, bits 4-7 zero |
| 14..15 | G (u16), cells per lane per tile, 1..4096 |
| 16..18 | anchor[3] |
| 19 | 0 |
| 20..23 | ntiles = sum over m of `ceil(cells[m] / (32 G))` |
| 24..27 | payload bytes P |
| 28..63 | zero |
| 64.. | `ntiles + 1` u32 tile offsets into the payload: `off[0] = 0`, `off[ntiles] = P`, every tile `off[t+1] - off[t] >= 128` and even |
| then | payload: per tile 32 u32 initial states (lane order), then its u16 words |
| then | zero padding to `total = align4096(64 + 4 (ntiles+1) + P)` |

Tiles run matrix 0, then 1, then 2; tile t of matrix m covers cells `[c0, c0 + n)`, `c0 = (t - first tile of m) * 32 G`, `n = min(32 G, cells[m] - c0)`. Kind 2 keeps this header and changes the rest of the layout: section 5.3.

## 4. Tile

Lane l, `ns = n * spc` scales. Lane l owns `ls = ceil((ns - l) / 32)` scales (0 if `l >= ns`) and `lc = ceil((n - l) / 32)` cells (0 if `l >= n`).

**Phase 1, scales.** Steps k = 0, 1, ..: lane l decodes scale `i = c0*spc + 32 k + l` of matrix m when `32 k + l < ns`, lanes in ascending order. MX: row `m` (order 0), the symbol is the scale. NV: when `scols[m] >= 33` and `i - scols[m] >= c0*spc` (the row above lies inside the tile, hence was decoded at an earlier step), `up = s[i - cols]`, `dul = clamp(s[i-cols-1] - up, -3, 3) + 3` if the scale is not the first of its row and `i - cols - 1 >= c0*spc`, else 7, `dur = clamp(s[i-cols+1] - up, -3, 3) + 3` if it is not the last of its row, else 7; `ctx = (1 + up / 4) * 64 + dul * 8 + dur` and the symbol is `(s - up) mod 256`. Otherwise ctx 63 and the symbol is s.

**Phase 2, cells.** For rounds j = 0, 1, ..: lane l owns cell `c0 + 32 j + l` when `32 j + l < n`; for nibble k = 0..31 every lane with a cell decodes its k-th nibble, lanes in ascending order. At `k % B == 0`: `cls` from the scale of that block (decoded in phase 1), `pm = mx = 8`. Row as in section 2 with `p = k % B`; after symbol y: `pm = y & 7`, `mx = pm` if `mx == 8` else `max(mx, pm)`.

**Tail in the state.** For a lane `l >= 2` that owns at least one cell, nibbles 28..31 of its last cell (round `(n - l - 1) / 32`) are not coded: the lane skips them in the lockstep (no decode, no renormalisation) and they are `(x_final - L)` read as 4 nibbles (bits 0-3 = nibble 28). The decoder requires `L <= x_final < L + 2^16`; the cell is finished (hash, CZ check, ceil inverse, output) after the last step.

**Words.** At each symbol position (k of phase 1, (j, k) of phase 2) the lanes that renormalise take the next words of the tile in ascending lane order. A decoder must never read past the tile's words, and must have used exactly all of them at the end.

**Output.** Scales go to the slab in phase 1. At the end of each cell (MX with ceil bit m): if the cell's coded maximum magnitude is 5, the scale must be > 0 (else invalid), the nibbles get `D(a) = a + min(a, 2)` (sign kept) and the scale becomes `s - 1`. CZ: a decoded symbol 8 is invalid.

**Hash and end states.** Lane hash = MurmurHash3_x86_32 blocks (seed `0x34524E57`) over the ident words `model_id, owner, t, l, kind | flags << 8 | G << 16, anchor0 | anchor1 << 8 | anchor2 << 16`, then one block per scale lane l decodes in phase 1, in step order (the scale's byte value as the block), then per cell of the lane, in round order, 4 blocks (the coded cell's 16 bytes as 4 LE words, before the ceil inverse), finalised with length `24 + 4 * ls + 16 * lc`. Tile hash H = XOR of the 32 lane hashes. The encoder starts lane 0 at `L + (H & 0xFFFF)`, lane 1 at `L + (H >> 16)`, lanes 2..31 at `L + tail` (its 4 tail nibbles, or 0 for a lane without cells), encodes every coded symbol in exact reverse decode order, and stores the 32 final states. The decoder requires every initial state `>= L`, lanes 0 and 1 ending at their hash values (whether or not they own cells), lanes 2..31 without cells ending at L, tail lanes ending in `[L, L + 2^16)`, and the word count exact. Any failure: the record is invalid (fail closed: the caller must not use what was written). Only valid / invalid is normative; which error bits a decoder reports for an invalid record is not.

## 5. Kind 2: GG, GGUF expert slices

Draft, experimental, like the rest of this document. Reference codec: `wnr4/wnr4_gg.h` (header only, C99, built on `wnr4/wnr4.h`).

### 5.1 Input and geometry

The input is a GGUF routed-expert record as `wex4/gguf2wex4.py` writes it ([WEX4-SPEC.md](WEX4-SPEC.md) section 14.1): the expert's slices of those of the gate, gate_up, up and down tensors that its layer has, in that order, back to back from offset 0. The record does not describe its slices: the decoder is given the geometry, `nseg` slices (1 to 4), each with its ggml type and its size in bytes (at least 1), and must be given exactly the geometry the encoder had.

- A slice of ggml type 42 (Q2_0, **subtype 0**) or 20 (IQ4_NL, **subtype 1**) is **coded**. Its size must be a multiple of 18; it is `blocks = bytes / 18` blocks, `blocks < 2^27`, each of 18 bytes: the scale d (f16, little endian: byte 0 is the **low byte**, byte 1 the **high byte**), then 16 code bytes.
- Every other slice is **verbatim**: stored in the record as it is. Its ggml type does not enter the record.
- The slices together are at most `2^32 - 1` bytes, the verbatim ones at most `2^31 - 1`. The record decodes to the sum of the slices.
- **Nibble** k (0..31) of a block is bits `4 (k & 1)` to `4 (k & 1) + 3` of code byte `k / 2`: low nibble first, whatever the ggml type means by those bits (the 2-bit codes of Q2_0 are coded two at a time).

Tiles cover the coded slices in slice order: slice s has `ceil(blocks[s] / (32 G))` tiles, and tile t of slice s covers blocks `[c0, c0 + n)` of s, `c0 = (t - first tile of s) * 32 G`, `n = min(32 G, blocks[s] - c0)`. G is blocks per lane per tile. Block i of a tile is block `c0 + i` of its slice; lane l owns blocks `i = l, l + 32, l + 64, ..` below n, that is `lc = ceil((n - l) / 32)` blocks (0 if `l >= n`), each with its one scale.

### 5.2 Model blob (WNM4 kind 2)

Little-endian. Header 8 x u32: `0x344D4E57` ("WNM4"), version 1, kind 2, 14, classes 32, subtypes 2, 0, 0. Then, for subtype 0 (Q2_0) and then subtype 1 (IQ4_NL):

| bytes | content |
|---|---|
| 256 | class map: one byte per value of a high byte, each below 32 |
| 514 | high-byte table: 257 u16 |
| 32 x 514 | low-byte tables of classes 0 to 31: 257 u16 each |
| 32 x 34 | nibble tables of classes 0 to 31: 17 u16 each |

That is 18,306 bytes per subtype and **36,644 bytes** in all. Every table is a 14-bit cumulative table as in section 2 (`cdf[0] = 0`, last entry 16384, strictly increasing). Id = FNV-1a of the whole blob, 0 replaced by 1. The **class** of a block is `map[u][its high byte]` for the subtype u of its slice; a decoder takes the map as stored (any value below 32).

Training (not normative: any map and tables that pass the checks decode). The reference trainer (`wnr4_gg_count`, `wnr4_gg_model_build`) counts, per subtype, over every block of the training slices (the tail nibbles included): the high bytes, the low bytes per value of the high byte, and the nibbles per value of the high byte. **Class map rule:** the 31 most frequent high bytes (count above 0; between equal counts the smaller byte first) get classes 1 to 31 in that order, every other value class 0. The low-byte and nibble counts of the high bytes of one class are added together, and every table is quantized by WNR3's rule as in section 2; a table with no counts is the near-uniform one that rule gives.

### 5.3 Record

The header of section 3, with:

| bytes | field |
|---|---|
| 12 | kind 2 |
| 13 | flags: 0 |
| 14..15 | G, 1..4096 |
| 16..19 | 0 (no anchors) |
| 20..23 | ntiles = sum over the coded slices of `ceil(blocks / (32 G))`; 0 when no slice is coded |
| 64.. | `ntiles + 1` u32 tile offsets into the payload, as section 3 |
| then | **verbatim area**: the verbatim slices in slice order, back to back, zero-padded to `V` = their total size rounded up to a multiple of 4 |
| then | payload P: per tile 32 u32 initial states (lane order), then its u16 words, as section 3 |
| then | zero padding to `total = align4096(64 + 4 (ntiles + 1) + V + P)` |

Owner (bytes 8..11) is the value the container binds the record to (in WEX4, the record entry's owner: WEX4-SPEC section 7.3). Decoding copies the verbatim slices to their offsets and decodes the tiles into the coded slices.

### 5.4 Tile

Lane l, u = the subtype of the tile's slice, rANS as section 1 (256-symbol tables in phases 1 and 2, 16-symbol tables in phase 3).

**Phase 1, high bytes.** Steps k = 0, 1, ..: lane l decodes the high byte of block `32 k + l` when `32 k + l < n`, lanes in ascending order, with the high-byte table of u (order 0).

**Phase 2, low bytes.** The same steps: lane l decodes the low byte of block `32 k + l` with the low-byte table of that block's class (its high byte was decoded in phase 1).

**Phase 3, code bytes.** For rounds j = 0, 1, ..: lane l owns block `32 j + l` when `32 j + l < n`; for nibble k = 0..31 every lane with a block decodes its k-th nibble, lanes in ascending order, with the nibble table of its block's class. There is no other context.

**Tail in the state** as section 4: for a lane `l >= 2` that owns at least one block, nibbles 28..31 of its last block (round `(n - l - 1) / 32`), which are code bytes 14 and 15, are not coded; they are `x_final - L` (bits 0-7 code byte 14, bits 8-15 code byte 15), and the decoder requires `L <= x_final < L + 2^16`. **Words** as section 4, the symbol positions being the steps of phases 1 and 2 and the (j, k) of phase 3.

### 5.5 Hash and end states

As section 4, with three changes:

- The ident words are `model_id, owner, t, l, 2 | G << 16` (kind | flags << 8 | G << 16 with flags 0) and, in place of the anchor word, **the ggml type of the tile's slice** (42 or 20). t is the tile's index in the record.
- After them, per lane: one block per high byte the lane decodes in phase 1, in step order (the byte as the block); one block per low byte in phase 2, in step order; then per block of the lane, in round order, 4 blocks: its 16 code bytes as 4 little-endian words, the tail bytes included.
- The lane hash is finalised with length `hlen = 24 + 8 x scales + 16 x cells`, where scales = cells = `lc` (one scale per block, hashed as two 4-byte blocks).

Tile hash H = XOR of the 32 lane hashes. The encoder starts lane 0 at `L + (H & 0xFFFF)`, lane 1 at `L + (H >> 16)`, lanes 2..31 at `L + tail` (code byte 14 | code byte 15 << 8 of the lane's last block, or 0 for a lane without blocks), encodes every coded symbol in exact reverse decode order, and stores the 32 final states. **End-state checks:** every initial state `>= L`; no read past the tile's words and exactly all of them used; lanes 0 and 1 end at their hash values; lanes 2..31 without blocks end at L; tail lanes end in `[L, L + 2^16)`. Kind 2 has no canonical zero and no ceil form. Any failure: the record is invalid (fail closed); only valid / invalid is normative.

**Record checks.** Besides the tiles a decoder requires: magic; model id not 0 and equal to the id of the model it was given; owner equal to the expected owner; kind 2; flags 0; bytes 16..19 and 28..63 zero; G in 1..4096; ntiles equal to the count from the geometry; `total` a multiple of 4096 and equal to `align4096(64 + 4 (ntiles + 1) + V + P)`; the padding of the verbatim area and after the payload zero; `off[0] = 0`, every tile `off[t+1] - off[t] >= 128` and even, `off[ntiles] = P`.

**Not covered by the tile hash: the verbatim slices.** Their bytes enter no tile hash (the zero padding is checked byte by byte), so a verbatim byte changed on disk decodes without error. In a WEX4 container the record's SHA-256 covers them, and a reader must check it at the record's first read (WEX4-SPEC sections 7.3 and 8.2).

**First check** (informative; one run of each figure; logs [results/first-check.log](../research/wnr4-gg-20261121/results/first-check.log) and [results/gpu-test.log](../research/wnr4-gg-20261121/results/gpu-test.log) in [research/wnr4-gg-20261121](../research/wnr4-gg-20261121/README.md)). A model trained on the 48 even-numbered of 96 sampled experts of Qwen3.8-Flash-Next UD-IQ1_M (layers 0, 6, 12, 24, 36 and 47, experts 0-15); the 48 odd-numbered experts were coded and all decoded back byte-exact. Only the down slice is coded (Q2_0, 460,800 bytes of each record: of 1,305,600 in the 40 records whose gate and up, stored verbatim, are ggml type 16, of 1,510,400 in the 8 with type 22). At G = 128 the coded records take 0.955 of the 4096-aligned raw record (0.9546; 0.9531 on the 40 records with type-16 gate and up, 0.9610 on the 8 with type 22); at G = 32 they take 0.9553. The prototype CUDA decoder (section 7, test `wnr4/tests/test_wnr4_gg_gpu.cu`; RTX 5090, records already in device memory, L2 flushed before each run, kernel only, median of 20 runs) decodes the coded slices of a batch of 32 of the 96 sampled records (the test trains its own model, on the even-numbered real records and the even records of two synthetic groups it cuts from them, and codes all 96) at 57.0 GB/s at G = 32 against 16.8 GB/s at G = 128 (3.4 times), and one record alone in 0.224 ms against 0.861 ms. This is why `wex4/wnr4gg.py` packs at G = 32 by default: about 0.07 points of ratio for 3.4 times the GPU decode speed.

**Whole model and a second pack** (informative; scripts and logs in [research/wnr4-gg-20261121](../research/wnr4-gg-20261121/README.md)). All 24,576 routed-expert records of Qwen3.8-Flash-Next UD-IQ1_M (48 layers x 512 experts, 34,812,723,200 bytes of experts) were packed into a RAW container by `wex4/gguf2wex4.py pack --experts-only`, each record checked equal to its GGUF slices. A model trained on every 8th record (3,072 records) then coded them at G = 32 with `wex4/wnr4gg.py pack`: 33,354,760,192 bytes of records against 34,839,986,176 raw (**0.9574**), 1.49 GB (4.3%) less on disk, and `wex4/wnr4gg.py verify --source` decoded all 24,576 on the CPU identical to the RAW container. Only the down slices are coded; the IQ2 gate and up slices are stored verbatim. A second GGUF quantization of a Qwen3.8-Flash-Next variant with 256 experts per layer, labelled IQ1_M (12,288 experts, 25.146 GB), has gate and up slices of other types (IQ3_XXS, IQ2_S, IQ3_S, IQ4_XS; about 59% of its bytes, all verbatim under GG) and an IQ4_NL down slice in 39 of its 48 layers. On 288 held-out records of a 480-record sample its coded records take **0.9763** of raw (0.9780 in the layers with an IQ4_NL down slice, 0.9669 in those with Q2_0); all 480 decode byte-identical, and the CPU decoder runs at 1.69 GB/s on 16 threads of a Ryzen 9 7950X3D in the logged run (1.88 GB/s in an earlier run whose log is not in the folder). The engine that runs that pack reads expert records from NVMe at about 4.7 GB/s on one read thread, faster than that decode, and even with free decode the 2.4% fewer bytes give at most about +2% tok/s, below the A/B noise (1.4-2.8% per-arm spread; the read speed and the noise are engine measurements, not in this repository): no gain when the experts are read from disk, and it is not integrated.

## 6. Kind 3: I4, int4-g64 experts

Draft, experimental, like the rest of this document. Reference codec: `wnr4/wnr4_i4.h` (header only, C99, built on `wnr4/wnr4.h`).

### 6.1 Input and geometry

The input is an int4-g64 expert record of 1 to 3 matrices. Matrix m has `rows[m]` rows (at least 1) and `cols[m]` inputs per row, a multiple of 64 (at least 64). A weight is a signed int4 code stored as a **nibble** `v + 8` (the codec takes any nibble 0..15), and each row is cut into **groups** of 64 consecutive inputs with one f32 **scale** each (the weight is `(nibble - 8) x scale`; the codec does not dequantize). Matrix m has `gc[m] = cols[m] / 64` groups per row and `cells[m] = rows[m] x gc[m]` groups, numbered row-major: group i is row `i / gc[m]`, column group `i % gc[m]`.

The record is the codes of matrix 0, 1, .. (32 bytes per group, groups in order), then the scales of matrix 0, 1, .. (4 bytes per group, little endian, groups in order), back to back from offset 0: `36 x (sum of cells)` bytes, at most `2^32 - 1`. The record does not describe its matrices: the decoder is given the geometry (`nmat`, the rows and cols, the packing) and must be given exactly the geometry the encoder had; only the packing is also in the record (6.3).

- **Packing** of a group's 32 code bytes. Element e (0..63: input `64 (i % gc) + e` of the row) is, in **planar** packing (P = 0), bits `4 (e / 32)` to `4 (e / 32) + 3` of byte `e % 32` (byte k holds elements k and k + 32); in **pairs** packing (P = 1), bits `4 (e % 2)` to `4 (e % 2) + 3` of byte `e / 2`. The packing changes the coded size, never what decodes.
- **Base blocks.** Matrix m is cut into blocks of 128 rows x 2 column groups (128 inputs): group i lies in block `bbase[m] + (row / 128) x nbc[m] + (i % gc[m]) / 2`, where `nbc[m] = ceil(gc[m] / 2)` and `bbase[m]` is the number of blocks of the matrices before m. The record holds one f32 **base** B per block, `nbases` = the sum over m of `ceil(rows[m] / 128) x nbc[m]` of them (6.3).
- **Lattice.** For k = 0..126, `E4M3(k)` is the value of the FP8 E4M3 code k (`k x 2^-9` for k < 8, else `(1 + (k & 7) / 8) x 2^((k >> 3) - 7)`), and `lat(k, B) = fl(fl(E4M3(k) x B) / 7)`: an IEEE 754 binary32 multiply, then a binary32 division by 7, each rounded to nearest even, subnormal results kept (a decoder must not flush them to zero). A scale is coded as a lattice index k (0..126), standing for `lat(k, B)` under its block's base, or as the escape index 127 followed by its 4 bytes. The lattice is what an int4 converter of an FP8 block-scaled checkpoint (one scale per 128 x 128 block) produces when it writes a group's scale as amax / 7, amax being the group's largest `|E4M3 x B|`; any other f32 value (NaN, infinity, negative values and -0 among them) escapes, so every record codes losslessly, only less compactly.

Tiles cover the matrices in order: matrix m has `ceil(cells[m] / (32 G))` tiles, and tile t of matrix m covers groups `[c0, c0 + n)` of m, as section 3 says of cells. G is groups per lane per tile; lane l owns groups `c0 + l + 32 j` below `c0 + n`, that is `lc = ceil((n - l) / 32)` groups (0 if `l >= n`), each with its scale.

### 6.2 Model blob (WNM4 kind 3)

Little-endian. Header 8 x u32: `0x344D4E57` ("WNM4"), version 1, kind 3, 14, 900, 68, 0, 0. Then `3 x 900` nibble tables of 17 u16, row `m x 900 + (cls x 10 + pm) x 10 + mx` (6.4), then `3 x 68` scale tables of 257 u16, row `m x 68 + r`: r = 0..63 the index contexts, 64..67 the escape bytes (high byte first). Every table is a 14-bit cumulative table as in section 2. That is **196,688 bytes**. Id = FNV-1a of the whole blob, 0 replaced by 1. The tables of matrix slot m serve matrix m of a record; a record of fewer than 3 matrices leaves the last slots unused.

Training (not normative: any tables that pass the checks decode). The reference trainer (`wnr4_i4_count`, `wnr4_i4_model_build`) codes each training record as the encoder would, with tiles of G = 32, and counts every symbol, the tail nibbles included; the escape-byte tables also count the 4 bytes of every scale, escaped or not, so they fit scales that miss the lattice. Every table is quantized by WNR3's rule as in section 2. The reference encoder (not normative either) takes as `anchor[m]` the median of `s >> 21` over the scales s of matrix m, and as each block's base the candidate under which most of its scales lie on the lattice, the candidates putting the block's largest scale on the codes 96 to 126; a block where nothing fits gets base 0.

### 6.3 Record

The header of section 3, with:

| bytes | field |
|---|---|
| 12 | kind 3 |
| 13 | flags: bit 0 the packing P (1 = pairs), bits 1-7 zero |
| 14..15 | G, 1..4096 |
| 16..19 | 0 |
| 20..23 | ntiles = sum over m of `ceil(cells[m] / (32 G))` |
| 28..33 | `anchor[0..2]`, three u16 (0 for m >= nmat) |
| 34..63 | zero |
| 64.. | `ntiles + 1` u32 tile offsets into the payload, as section 3 |
| then | **base area**: `nbases` u32, the bases of the blocks in block order, each 0 or a positive normal finite f32 (`0x00800000 <= B < 0x7F800000`) |
| then | payload P: per tile 32 u32 initial states (lane order), then its u16 words, as section 3 |
| then | zero padding to `total = align4096(64 + 4 (ntiles + 1) + 4 nbases + P)` |

Owner (bytes 8..11) as 5.3. A decoder takes the anchors as stored (any value of a matrix the record has).

### 6.4 Tile

Lane l, rANS as section 1 (256-symbol tables in phase 1, 16-symbol tables in phase 2).

**Phase 1, scales.** Steps k = 0, 1, ..: lane l decodes the scale of group `i = c0 + 32 k + l` when `32 k + l < n`. First, every such lane in ascending order decodes the **index symbol**: the reference of group i is group `u = i - d`, with `d = gc` when `gc >= 32`, else `d = ceil(32 / gc) x gc` (the group `d / gc` rows above, decoded at an earlier step), if `u >= c0`, else there is none; `kref` is the index of the reference (126 when it escaped), or 120 without a reference; the table is scale row `(32 if there is a reference, else 0) + kref / 4` of matrix m; after symbol y, `k = (y + kref) mod 256`, and `k > 127` is invalid. Then, for b = 0, 1, 2, 3, every lane whose k is 127, in ascending order, decodes byte `3 - b` of its scale with scale row `64 + b`. A scale with `k <= 126` is `lat(k, B)` under the base of its block.

**Phase 2, groups.** For rounds j = 0, 1, ..: lane l owns group `c0 + 32 j + l` when `32 j + l < n`; for element e = 0..63 every lane with a group decodes its nibble e, lanes in ascending order. Row as in 6.2, with `cls = clamp((s >> 21) - anchor[m], -4, 4) + 4`, s the 32 bits of the group's scale (decoded in phase 1) as an unsigned integer, and `pm = mx = 9` at e = 0; after symbol y: `a = |y - 8|`, `pm = a`, `mx = a` if `mx == 9`, else `max(mx, a)`.

**Tail in the state** as section 4: for a lane `l >= 2` that owns at least one group, elements 60..63 of its last group (round `(n - l - 1) / 32`) are not coded; they are `x_final - L` (bits 0-3 element 60, .., bits 12-15 element 63), and the decoder requires `L <= x_final < L + 2^16`. **Words** as section 4, the symbol positions being, at each step of phase 1, the index symbol and then the four escape bytes, and the (j, e) of phase 2.

### 6.5 Hash and end states

As section 4, with three changes:

- The ident words are `model_id, owner, t, l, 3 | P << 8 | G << 16` and, in place of the anchor word, `m | anchor[m] << 16`.
- After them, per lane: one block per scale the lane decodes in phase 1, in step order (the scale's 32 bits); then per group of the lane, in round order, 8 blocks: its 32 code bytes, in the record's packing, as 8 little-endian words, the tail included.
- The lane hash is finalised with length `hlen = 24 + 4 x lc + 32 x lc`.

Tile hash H = XOR of the 32 lane hashes. The encoder starts lane 0 at `L + (H & 0xFFFF)`, lane 1 at `L + (H >> 16)`, lanes 2..31 at `L + tail` (element 60 | element 61 << 4 | element 62 << 8 | element 63 << 12 of the lane's last group, or 0 for a lane without groups), encodes every coded symbol in exact reverse decode order, and stores the 32 final states. End-state checks and word count as 5.5. Kind 3 has no canonical zero and no ceil form. Any failure: the record is invalid (fail closed); only valid / invalid is normative.

**Record checks.** Besides the tiles a decoder requires: magic; model id not 0 and equal to the id of the model it was given; owner equal to the expected owner; kind 3; flags equal to the packing of the geometry; bytes 16..19 and 34..63 zero; the anchors of matrices the geometry does not have zero; G in 1..4096; ntiles equal to the count from the geometry; `total` a multiple of 4096 and equal to `align4096(64 + 4 (ntiles + 1) + 4 nbases + P)`; every base 0 or a positive normal finite f32; the padding after the payload zero; `off[0] = 0`, every tile `off[t+1] - off[t] >= 128` and even, `off[ntiles] = P`.

**What the tile hash covers.** Every decoded byte, scale or code, enters a tile hash: a kind-3 record has no verbatim bytes. Two changes of the record can still decode to the same bytes, and the CPU tests count them: a base of a block whose scales all escape (it is never read) changed to another valid base, and an anchor changed so that no class changes.

**First check** (informative; one run of each figure; logs and the bench in [research/wnr4-i4-20261122](../research/wnr4-i4-20261122/README.md)). Qwen3.8-Flash-Next, int4-g64 export (3 matrices: gate and up 640 x 2560, down 2560 x 640, planar; 2,764,800 bytes per record): a model trained on 32 sampled records coded 256 other records, sampled across all 48 layers, every one decoded back byte-exact on 1 and on 8 threads; every scale of them lay on the lattice (no escape). At G = 32 the coded records take **0.78333** of raw (4096-aligned), at G = 64 0.78270, at G = 16 0.78474; zstd -19 gives 0.8074 on the same 256 records (unpadded). CPU (Ryzen 9 7950X3D, gcc 16.2 -O2 -march=haswell, G = 32): decode 84 MB/s on 1 thread and 447 MB/s on 8, encode 93 MB/s on 1 thread. GPU (RTX 5090, CUDA 13.4, test `wnr4/tests/test_wnr4_i4_gpu.cu`; records already in device memory, L2 flushed before each run, kernel only, median of 20): batches of 32 records decode at 156.9 GB/s at G = 32, 193.2 GB/s at G = 16 and 79.8 GB/s at G = 64; one record alone in 0.474 ms, 0.241 ms and 1.035 ms. A second export, Inkling-2-Mini (int4-g64 converted from an NVFP4 checkpoint, 3 matrices 1152 x 2304, 1152 x 2304, 2304 x 1152), does not fit the lattice: 90% of its scales escape, and 256 held-out records take 0.88139 of raw, more than zstd -19 (0.8406). Kind 3 is meant for exports of FP8 block-scaled checkpoints.

**Whole model** (one run): all 24,576 records of that export (48 layers x 512 experts, 67,947,724,800 bytes of records), with a model trained on every 8th record (3,072), packed by `wex4/wnr4i4.py pack` at G = 32: 53,246,078,976 bytes of records (**0.7836** of raw), a 53,294,047,232-byte container against 67,995,492,352 bytes RAW (14.7 GB, 21.6% less), and `wex4/wnr4i4.py verify --source` decoded all 24,576 records on the CPU identical to the RAW container (a Ryzen 7 9700X: pack 64 MB/s, verify 242 MB/s on 4 threads).

## 7. Decoders

These are implementation notes; only sections 1-6 are normative.

- CPU reference: `wnr4/wnr4.h`: `wnr4_decode` (one record, OpenMP over its tiles) and `wnr4_decode_batch` (every tile of every record of a batch in one parallel loop). Tool API: `wnr4/tools/wnr4_tools.c`.
- CUDA: `wnr4/wnr4_cuda.cuh`. One warp per tile; a block holds 4, 8 or 16 tiles of the same matrix (chosen from the batch's tile count); the matrix's weight rows (compact order: only the 37 reachable (pm, mx) states per class) in shared memory or, as an option, read through L1; a 64-word window of the tile's word stream kept in two registers per lane and slid with one coalesced load per 32 words consumed, so a renormalising lane takes its word by shuffle; the decoded symbol's start and end read from its row by two 16-bit loads after a 16-way comparison; no `break` in the symbol loop, so the warp collectives stay on their convergent fast path; NVFP4 scales through a two-level table (a 16-value summary row, then the 16-value block row: two 32-byte loads per scale instead of an 8-step binary search); scales of phase 2 read back from the destination after `__syncwarp`; one launch for all tiles of a batch; output written straight to per-record, per-matrix destination pointers (device memory). Errors: per-record bit set (1 structure, 2 hash, 4 ceil, 8 CZ).
- Helper prototype: `wnr4/tools/wnr4_cuda_bridge.cu` (`wnr4_cuda_register_model`; `wnr4_cuda_batch_submit` / `wnr4_cuda_batch_wait` on a caller stream or a per-slot stream, up to 4 batches in flight per thread and device, a 5th before a wait refused with -4 so no verdict is dropped; `wnr4_cuda_decode_batch` synchronous; `wnr4_cuda_release_thread`). Records come from page-locked host memory; a record that fails reports -1 and its destination may hold partial data (the caller does not publish that slot); CUDA or memory errors and an unregistered model report 1.
- Kind 2, CPU reference: `wnr4/wnr4_gg.h`: `wnr4_gg_decode` (one record: the verbatim slices, then OpenMP over its tiles), `wnr4_gg_encode`, the trainer of section 5.2. Tool API: `wnr4/tools/wnr4_gg_tools.c` (a geometry is passed as `nseg`, the ggml types and the byte sizes). CPU tests: `wnr4/tests/test_wnr4_gg.c` (run by `scripts/build_wnr4_check.sh`). CUDA: `wnr4/wnr4_gg_cuda.cuh` (one warp per tile, as for MX and NV; the verbatim slices copied from the record's device copy), with the helper entry points `wnr4_cuda_gg_register_model`, `wnr4_cuda_gg_batch_submit` and `wnr4_cuda_gg_decode_batch` in `wnr4/tools/wnr4_cuda_bridge.cu`, whose `wnr4_cuda_abi()` is 5 since they were added (it was 4; the MX and NV entry points are unchanged); one geometry per batch. GPU test: `wnr4/tests/test_wnr4_gg_gpu.cu` (built with `-arch=sm_120`, run on an RTX 5090; not in CI).
- Kind 3, CPU reference: `wnr4/wnr4_i4.h`: `wnr4_i4_decode` (one record, OpenMP over its tiles), `wnr4_i4_encode`, the trainer of section 6.2. Tool API: `wnr4/tools/wnr4_i4_tools.c` (a geometry is passed as `nmat`, the rows, the cols and the packing). CPU tests: `wnr4/tests/test_wnr4_i4.c` (run by `scripts/build_wnr4_check.sh`). CUDA: `wnr4/wnr4_i4_cuda.cuh` (one warp per tile, as for the other kinds; the matrix's nibble rows in shared memory in compact order, the 46 reachable (pm, mx) states per class; the scale rows read through L1, phase 1 being 1/65 of the symbols; the tile's lattice indices, the contexts of later steps, in shared memory, 32 G bytes per warp, so the CUDA decoder takes G <= 64 and a record of a larger G is refused with -1, for the CPU to decode; the lattice with `__fmul_rn` and `__fdiv_rn`, so it must not be built with `--use_fast_math` or `-ftz=true`), with the helper entry points `wnr4_cuda_i4_register_model`, `wnr4_cuda_i4_batch_submit` and `wnr4_cuda_i4_decode_batch`, whose `wnr4_cuda_abi()` is 6 since they were added (it was 5; the other entry points are unchanged); one geometry per batch, destinations 4-byte aligned. GPU test: `wnr4/tests/test_wnr4_i4_gpu.cu` (built with `-arch=sm_120`, run on an RTX 5090; not in CI).
- Independent decoders of kinds 2 and 3, written from sections 1 to 6 and WEX4-SPEC 7.3 with the reference code closed (2026-11-23): the Go reader (`impl/go-wex4`: `wnr4.go`, `wnr4_gg.go`, `wnr4_i4.go`) and the Rust reader (`impl/rust-wex4/src/wnr4/`), written apart from each other. Both are CPU, one thread, called by their whole-file checks for WEX4 codec 8; what the text left open is part F of `impl/go-wex4/SPEC-NOTES.md` and part G of `impl/rust-wex4/SPEC-NOTES.md`. They decode the codec-8 files of `impl/go-wex4/testdata` to the content digests of their RAW sources and agree with each other on mutated copies of them (`impl/rust-wex4/differential.py`). Kinds 0 and 1 have no independent decoder.

## 8. Open

WEX4 codec 8 and blob kind 3 are defined in WEX4-SPEC section 7.3 for records that are neither EXPERT nor VALUE_EXACT (covering kind 2 in GGUF records and kind 3 in int4-g64 expert records). Kinds 0 (MX) and 1 (NV) in an EXPERT record still need a formal `required_features` bit in WEX4-SPEC section 10, rule 6 (the maintainer's DeepSeek engine tree currently gates MX codec-8 EXPERT records behind an explicit per-TU opt-in `wex4_wnr4_experts` and CUDA helper feature bit 64). Also open: conformance vectors in `wex4/vectors/` (waiting for the required feature bit to be frozen), independent Go/Rust decoders for kinds 0 and 1, a public container writer for MX/NV, and libFuzzer harnesses for `wnr4/`.

Informative notes (updated 2026-11-25):

- **Engine integration status.** On 2026-11-24, CPU and CUDA decode for kind 0 (MX) was integrated into the maintainer's DeepSeek V4 engine tree (`wex_decode.c`, `wex_cuda_bridge.cu` feature 64, `deepseek_v4.c`), and whole-model WNR4-MX containers were built for DeepSeek V4.1 REAP-192 (122.87 GB, 0.851) and Qwen3.8 MXFP4 (56.85 GB, 0.886). On Qwen3.8 MXFP4, RAW stays the active read container because dual-NVMe decode is not disk-bound (see the 2026-11-21 analysis in [reports/wnr4-research-20261121.md](reports/wnr4-research-20261121.md#81-engine-integration-analysed-not-done) section 8.1). Kinds GG and I4 are not read directly by an inference engine during disk-bound decode (at NVMe speeds, decoding every miss on CPU or via a GPU round trip costs more than the bytes saved); they serve as compact, byte-exact storage and distribution containers.
- **Kind 3 and other int4 exports.** Kind 3 (section 6, 2026-11-22) is the int4-g64 candidate of the report's section 8.3 ([reports/wnr4-research-20261121.md](reports/wnr4-research-20261121.md#83-int4-g64-prototype)) with a per-tile hash, the tail in the state, a CUDA decoder and a container writer (`wex4/wnr4i4.py`), and with one scale model only, the lattice of an FP8 block-scaled source plus an escape. The prototype also had a palette mode and a lattice of NVFP4 products, with which it took 0.8264 of raw on Inkling-2-Mini, whose scales kind 3 escapes (0.88139); a kind for such exports would need one of them. Sage-S2.1-Flash (pairs packing, 0.7855 in the prototype) was not run again with kind 3.
- **IQ2 blocks in kind GG.** GG codes Q2_0 and IQ4_NL only. The IQ2 codes are grid indices and signs, close to maximum entropy; a per-type pre-study with an earlier prototype coder still estimates that coding the IQ2 gate and up slices too would save about 1.1 GB more on UD-IQ1_M. Not done; figures in [reports/wnr4-research-20261121.md](reports/wnr4-research-20261121.md#82-kind-gg-gguf-2-bit-experts) section 8.2.
