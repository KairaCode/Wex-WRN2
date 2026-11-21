# WNR4 specification (DRAFT 0.1, 2026-11-21)

**Status:** research prototype. Not in any production container, engine or production helper (a prototype helper is in `wnr4/tools`). [WEX4-SPEC.md](WEX4-SPEC.md) assigns it, as a draft, codec 8 and blob kind 3 (its section 7.3), for records that are neither EXPERT nor VALUE_EXACT; so far that means GGUF routed-expert records coded with kind 2 (GG, section 5), which was added to this draft on the same day. WNR3 (codec 5) is unchanged; so is WNR3-NV, the NVFP4 variant of WNR3 on the not yet public `nvfp4-wnr3` branch, which proposes codec 7 for itself (WEX4-SPEC still lists 7 as reserved for the next expert codec). The name WNR4 was used before for a smaller proposal, the "WNR4 record" of WEX4-SPEC section 12.2 (three changes to WNR3's stream skeleton, not adopted); this draft replaces it. Measurements and the reasoning behind every choice: [research/wnr4-20261121](../research/wnr4-20261121/README.md) and [reports/wnr4-research-20261121.md](reports/wnr4-research-20261121.md).

**One paragraph.** WNR4 is a lossless record codec for MXFP4 and NVFP4 routed-expert slabs, built so that a GPU can decode many records in one launch at tens of GB/s. It keeps WNR3's weight model (block-scale class relative to a rule-g anchor, previous magnitude, running maximum; 16-symbol 14-bit tables; canonical zero; ceil form for MXFP4), optionally refined by the nibble's position in the last four of its block, codes NVFP4 scales as a delta from the scale above with the up-left/up-right deltas as context, and changes the skeleton: a record is a list of *tiles*; a tile is 32 rANS coders (*lanes*) that share one 16-bit word stream, decoded by one warp in lockstep; the last 4 nibbles of each lane ride in its initial state. Integrity moves from a 23-bit hash per 1 KB stream to a 32-bit hash per tile plus checked end states and an exact word count. Kind 2 (GG, section 5) puts the same skeleton under another model, for the 18-byte blocks of the GGUF Q2_0 and IQ4_NL slices of a GGUF routed-expert record.

## 1. Terms

- **Slab:** the decoded expert, as WEX3/WEX4 lay it out: per matrix m (0 gate w1, 1 down w2, 2 up w3) `scount[m]` scale bytes at `soff[m]`, then packed nibbles at `woff[m]` (two per byte, symbol 2i in the low nibble). `scols[m]` = scales per row.
- **Kind:** 0 = MX (MXFP4: one UE8M0 scale per 32 nibbles), 1 = NV (NVFP4: one E4M3 scale per 16 nibbles). `B` = 32 or 16 nibbles per block, `spc` = 1 or 2 scales per cell. 2 = GG (GGUF expert slices): it has its own geometry, model blob and phases (section 5) and shares with MX and NV the rANS below, the record header of section 3 and, from section 4, the words, the tail in the state, the hash and the end states. The other terms of this section and sections 2 to 4 are those of kinds 0 and 1.
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

**First check** (informative; one run of each figure; logs [results/first-check.log](../research/wnr4-gg-20261121/results/first-check.log) and [results/gpu-test.log](../research/wnr4-gg-20261121/results/gpu-test.log) in [research/wnr4-gg-20261121](../research/wnr4-gg-20261121/README.md)). A model trained on the 48 even-numbered of 96 sampled experts of Qwen3.8-Flash-Next UD-IQ1_M (layers 0, 6, 12, 24, 36 and 47, experts 0-15); the 48 odd-numbered experts were coded and all decoded back byte-exact. Only the down slice is coded (Q2_0, 460,800 bytes of each record: of 1,305,600 in the 40 records whose gate and up, stored verbatim, are ggml type 16, of 1,510,400 in the 8 with type 22). At G = 128 the coded records take 0.955 of the 4096-aligned raw record (0.9546; 0.9531 on the 40 records with type-16 gate and up, 0.9610 on the 8 with type 22); at G = 32 they take 0.9553. The prototype CUDA decoder (section 6, test `wnr4/tests/test_wnr4_gg_gpu.cu`; RTX 5090, records already in device memory, L2 flushed before each run, kernel only, median of 20 runs) decodes the coded slices of a batch of 32 of the 96 sampled records (the test trains its own model, on the even-numbered real records and the even records of two synthetic groups it cuts from them, and codes all 96) at 57.0 GB/s at G = 32 against 16.8 GB/s at G = 128 (3.4 times), and one record alone in 0.224 ms against 0.861 ms. This is why `wex4/wnr4gg.py` packs at G = 32 by default: about 0.07 points of ratio for 3.4 times the GPU decode speed.

**Whole model and a second pack** (informative; scripts and logs in [research/wnr4-gg-20261121](../research/wnr4-gg-20261121/README.md)). All 24,576 routed-expert records of Qwen3.8-Flash-Next UD-IQ1_M (48 layers x 512 experts, 34,812,723,200 bytes of experts) were packed into a RAW container by `wex4/gguf2wex4.py pack --experts-only`, each record checked equal to its GGUF slices. A model trained on every 8th record (3,072 records) then coded them at G = 32 with `wex4/wnr4gg.py pack`: 33,354,760,192 bytes of records against 34,839,986,176 raw (**0.9574**), 1.49 GB (4.3%) less on disk, and `wex4/wnr4gg.py verify --source` decoded all 24,576 on the CPU identical to the RAW container. Only the down slices are coded; the IQ2 gate and up slices are stored verbatim. A second GGUF quantization of a Qwen3.8-Flash-Next variant with 256 experts per layer, labelled IQ1_M (12,288 experts, 25.146 GB), has gate and up slices of other types (IQ3_XXS, IQ2_S, IQ3_S, IQ4_XS; about 59% of its bytes, all verbatim under GG) and an IQ4_NL down slice in 39 of its 48 layers. On 288 held-out records of a 480-record sample its coded records take **0.9763** of raw (0.9780 in the layers with an IQ4_NL down slice, 0.9669 in those with Q2_0); all 480 decode byte-identical, and the CPU decoder runs at 1.69 GB/s on 16 threads of a Ryzen 9 7950X3D in the logged run (1.88 GB/s in an earlier run whose log is not in the folder). The engine that runs that pack reads expert records from NVMe at about 4.7 GB/s on one read thread, faster than that decode, and even with free decode the 2.4% fewer bytes give at most about +2% tok/s, below the A/B noise (1.4-2.8% per-arm spread; the read speed and the noise are engine measurements, not in this repository): no gain when the experts are read from disk, and it is not integrated.

## 6. Decoders

These are implementation notes; only sections 1-5 are normative.

- CPU reference: `wnr4/wnr4.h`: `wnr4_decode` (one record, OpenMP over its tiles) and `wnr4_decode_batch` (every tile of every record of a batch in one parallel loop). Tool API: `wnr4/tools/wnr4_tools.c`.
- CUDA: `wnr4/wnr4_cuda.cuh`. One warp per tile; a block holds 4, 8 or 16 tiles of the same matrix (chosen from the batch's tile count); the matrix's weight rows (compact order: only the 37 reachable (pm, mx) states per class) in shared memory or, as an option, read through L1; a 64-word window of the tile's word stream kept in two registers per lane and slid with one coalesced load per 32 words consumed, so a renormalising lane takes its word by shuffle; the decoded symbol's start and end read from its row by two 16-bit loads after a 16-way comparison; no `break` in the symbol loop, so the warp collectives stay on their convergent fast path; NVFP4 scales through a two-level table (a 16-value summary row, then the 16-value block row: two 32-byte loads per scale instead of an 8-step binary search); scales of phase 2 read back from the destination after `__syncwarp`; one launch for all tiles of a batch; output written straight to per-record, per-matrix destination pointers (device memory). Errors: per-record bit set (1 structure, 2 hash, 4 ceil, 8 CZ).
- Helper prototype: `wnr4/tools/wnr4_cuda_bridge.cu` (`wnr4_cuda_register_model`; `wnr4_cuda_batch_submit` / `wnr4_cuda_batch_wait` on a caller stream or a per-slot stream, up to 4 batches in flight per thread and device, a 5th before a wait refused with -4 so no verdict is dropped; `wnr4_cuda_decode_batch` synchronous; `wnr4_cuda_release_thread`). Records come from page-locked host memory; a record that fails reports -1 and its destination may hold partial data (the caller does not publish that slot); CUDA or memory errors and an unregistered model report 1.
- Kind 2, CPU reference: `wnr4/wnr4_gg.h`: `wnr4_gg_decode` (one record: the verbatim slices, then OpenMP over its tiles), `wnr4_gg_encode`, the trainer of section 5.2. Tool API: `wnr4/tools/wnr4_gg_tools.c` (a geometry is passed as `nseg`, the ggml types and the byte sizes). CPU tests: `wnr4/tests/test_wnr4_gg.c` (run by `scripts/build_wnr4_check.sh`). CUDA: `wnr4/wnr4_gg_cuda.cuh` (one warp per tile, as for MX and NV; the verbatim slices copied from the record's device copy), with the helper entry points `wnr4_cuda_gg_register_model`, `wnr4_cuda_gg_batch_submit` and `wnr4_cuda_gg_decode_batch` in `wnr4/tools/wnr4_cuda_bridge.cu`, whose `wnr4_cuda_abi()` is 5 since they were added (it was 4; the MX and NV entry points are unchanged); one geometry per batch. GPU test: `wnr4/tests/test_wnr4_gg_gpu.cu` (built with `-arch=sm_120`, run on an RTX 5090; not in CI).

## 7. Open

WEX4 codec 8 and blob kind 3 are assigned as a draft (WEX4-SPEC section 7.3) for records that are neither EXPERT nor VALUE_EXACT, which covers kind 2 in the GGUF records of WEX4-SPEC section 14.1. Kinds 0 and 1 in an EXPERT record still need EXPERT to allow codec 8, hence a required feature bit (WEX4-SPEC section 10, rule 6). Also open: conformance vectors (they wait until the WEX4 draft is adopted), production helper and engine integration, fuzzing of the parser beyond the tests of `wnr4/tests` and the research folder.

Informative, as of 2026-11-21:

- **No engine integrates WNR4.** For kinds MX and NV this was decided on 2026-11-21 by analysis only, without engine code: on Qwen3.8-Flash-Next MXFP4 the RAM slots WNR4 frees are worth less than the A/B can resolve, and on DeepSeek V4.1 REAP-192 the estimated gain (+1.5 to +3% tok/s, G = 32 only) is within the run noise. The figures and the steps that would reopen the question: [reports/wnr4-research-20261121.md](reports/wnr4-research-20261121.md#81-engine-integration-analysed-not-done) section 8.1. The GG containers of section 5.5 are not in any engine either.
- **int4-g64 as a candidate kind** (signed int4 with one f32 scale per 64 inputs along the row, no zero point). A CPU prototype on this draft's skeleton (32-lane tiles, one 16-bit word stream, the scales in phase 1, the nibbles in phase 2) is not in this repository. On 768 held-out experts (256 from each of three int4 exports: Qwen3.8-Flash-Next, Sage-S2.1-Flash, Inkling-2-Mini) it takes 0.7829, 0.7855 and 0.8264 of raw against 0.8074, 0.8039 and 0.8406 for zstd -19, all round trips byte-exact; per-tile hash, tail in the state, GPU kernel and container writer are not done. Figures: [reports/wnr4-research-20261121.md](reports/wnr4-research-20261121.md#83-int4-g64-prototype) section 8.3.
- **IQ2 blocks in kind GG.** GG codes Q2_0 and IQ4_NL only. The IQ2 codes are grid indices and signs, close to maximum entropy; a per-type pre-study with an earlier prototype coder still estimates that coding the IQ2 gate and up slices too would save about 1.1 GB more on UD-IQ1_M. Not done; figures in [reports/wnr4-research-20261121.md](reports/wnr4-research-20261121.md#82-kind-gg-gguf-2-bit-experts) section 8.2.
