# WNR4 specification (DRAFT 0.1, 2026-11-21)

**Status:** research prototype. Not in any container, engine or helper. WNR3 (codec 5) and WNR3-NV (codec 7) are unchanged. Measurements and the reasoning behind every choice: [research/wnr4-20261121](../research/wnr4-20261121/README.md) and [wnr4-research.md](wnr4-research.md).

**One paragraph.** WNR4 is a lossless record codec for MXFP4 and NVFP4 routed-expert slabs, built so that a GPU can decode many records in one launch at tens of GB/s. It keeps WNR3's weight model (block-scale class relative to a rule-g anchor, previous magnitude, running maximum; 16-symbol 14-bit tables; canonical zero; ceil form for MXFP4), optionally refined by the nibble's position in the last four of its block, codes NVFP4 scales as a delta from the scale above with the up-left/up-right deltas as context, and changes the skeleton: a record is a list of *tiles*; a tile is 32 rANS coders (*lanes*) that share one 16-bit word stream, decoded by one warp in lockstep; the last 4 nibbles of each lane ride in its initial state. Integrity moves from a 23-bit hash per 1 KB stream to a 32-bit hash per tile plus checked end states and an exact word count.

## 1. Terms

- **Slab:** the decoded expert, as WEX3/WEX4 lay it out: per matrix m (0 gate w1, 1 down w2, 2 up w3) `scount[m]` scale bytes at `soff[m]`, then packed nibbles at `woff[m]` (two per byte, symbol 2i in the low nibble). `scols[m]` = scales per row.
- **Kind:** 0 = MX (MXFP4: one UE8M0 scale per 32 nibbles), 1 = NV (NVFP4: one E4M3 scale per 16 nibbles). `B` = 32 or 16 nibbles per block, `spc` = 1 or 2 scales per cell.
- **Cell:** 32 consecutive nibbles of a matrix (16 bytes) with their `spc` scales. Matrix m has `cells[m] = scount[m] / spc` cells; `scount[m] % spc == 0`, `scount[m] % scols[m] == 0` and `cells[m] < 2^27` are required, and the six regions of the slab (three scale ranges, three nibble ranges) must not overlap.
- **Coded domain:** as WNR3 section 1: canonical zero when flag CZ (every nibble 8 coded as 0, the decoder refuses a decoded 8), and for MX the WNR3 ceil form per matrix (flag bit m; gate and transform exactly WNR3 section 4.2). NV never uses the ceil form.
- **Class** of a scale s in matrix m: `clamp(s - anchor[m], -R, R) + R`, R = 3 (MX, 7 classes) or 6 (NV, 13 classes). An encoder writes `anchor[m]` = rule g (WNR3 4.3) on the coded scales of m; a decoder takes the stored byte as it is (any value 0..255, as WNR3 erratum 14).
- **rANS:** 32-bit state x, `L = 2^16`, precision 14 (`T = 16384`), 16-bit little-endian words. Decode of one symbol with table `cdf`: `slot = x & (T-1)`, `s` = largest symbol with `cdf[s] <= slot`, `x = (cdf[s+1]-cdf[s]) * (x >> 14) + slot - cdf[s]`; then, if `x < L`, `x = x << 16 | next word`. One word at most per symbol.

## 2. Model blob (WNM4)

Little-endian. Header 8 x u32: `0x344D4E57` ("WNM4"), version 1, kind, 14, `wctx = 81 * classes * pb` (classes 7 MX, 13 NV), `sctx` (1 MX, 4160 NV), `pb` (position buckets, 1 or 5), 0. Then `3 * wctx` weight tables of 17 u16 (`cdf[0] = 0 .. cdf[16] = 16384`, strictly increasing), row `m * wctx + ((cls * 9 + pm) * 9 + mx) * pb + b`, with `b = 0` when pb = 1, else `b = 0` for block positions `p < B - 4` and `b = p - B + 5` (1..4) for the last four; then `3 * sctx` scale tables of 257 u16, row `m * sctx + ctx`. Size: MX 59,408 B (pb 1) or 290,744 B (pb 5); NV 6,522,158 B (pb 1) or 6,951,782 B (pb 5). Id = FNV-1a of the whole blob, 0 replaced by 1. Training (not normative: any tables decode): the reference trainer counts the coded domain with matrix-wide contexts (the NV scale context as in section 4 but with the row above available everywhere except the matrix's first row, and all 32 nibbles of every cell, the tail nibbles included), then quantizes by WNR3's rule (`wex_nr2_quantize`: every symbol at least 1, remainders by size). A WEX4 container would carry it as a blob of a new kind (proposed 3).

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

Tiles run matrix 0, then 1, then 2; tile t of matrix m covers cells `[c0, c0 + n)`, `c0 = (t - first tile of m) * 32 G`, `n = min(32 G, cells[m] - c0)`.

## 4. Tile

Lane l, `ns = n * spc` scales. Lane l owns `ls = ceil((ns - l) / 32)` scales (0 if `l >= ns`) and `lc = ceil((n - l) / 32)` cells (0 if `l >= n`).

**Phase 1, scales.** Steps k = 0, 1, ..: lane l decodes scale `i = c0*spc + 32 k + l` of matrix m when `32 k + l < ns`, lanes in ascending order. MX: row `m` (order 0), the symbol is the scale. NV: when `scols[m] >= 33` and `i - scols[m] >= c0*spc` (the row above lies inside the tile, hence was decoded at an earlier step), `up = s[i - cols]`, `dul = clamp(s[i-cols-1] - up, -3, 3) + 3` if the scale is not the first of its row and `i - cols - 1 >= c0*spc`, else 7, `dur = clamp(s[i-cols+1] - up, -3, 3) + 3` if it is not the last of its row, else 7; `ctx = (1 + up / 4) * 64 + dul * 8 + dur` and the symbol is `(s - up) mod 256`. Otherwise ctx 63 and the symbol is s.

**Phase 2, cells.** For rounds j = 0, 1, ..: lane l owns cell `c0 + 32 j + l` when `32 j + l < n`; for nibble k = 0..31 every lane with a cell decodes its k-th nibble, lanes in ascending order. At `k % B == 0`: `cls` from the scale of that block (decoded in phase 1), `pm = mx = 8`. Row as in section 2 with `p = k % B`; after symbol y: `pm = y & 7`, `mx = pm` if `mx == 8` else `max(mx, pm)`.

**Tail in the state.** For a lane `l >= 2` that owns at least one cell, nibbles 28..31 of its last cell (round `(n - l - 1) / 32`) are not coded: the lane skips them in the lockstep (no decode, no renormalisation) and they are `(x_final - L)` read as 4 nibbles (bits 0-3 = nibble 28). The decoder requires `L <= x_final < L + 2^16`; the cell is finished (hash, CZ check, ceil inverse, output) after the last step.

**Words.** At each symbol position (k of phase 1, (j, k) of phase 2) the lanes that renormalise take the next words of the tile in ascending lane order. A decoder must never read past the tile's words, and must have used exactly all of them at the end.

**Output.** Scales go to the slab in phase 1. At the end of each cell (MX with ceil bit m): if the cell's coded maximum magnitude is 5, the scale must be > 0 (else invalid), the nibbles get `D(a) = a + min(a, 2)` (sign kept) and the scale becomes `s - 1`. CZ: a decoded symbol 8 is invalid.

**Hash and end states.** Lane hash = MurmurHash3_x86_32 blocks (seed `0x34524E57`) over the ident words `model_id, owner, t, l, kind | flags << 8 | G << 16, anchor0 | anchor1 << 8 | anchor2 << 16`, then one block per scale lane l decodes in phase 1, in step order (the scale's byte value as the block), then per cell of the lane, in round order, 4 blocks (the coded cell's 16 bytes as 4 LE words, before the ceil inverse), finalised with length `24 + 4 * ls + 16 * lc`. Tile hash H = XOR of the 32 lane hashes. The encoder starts lane 0 at `L + (H & 0xFFFF)`, lane 1 at `L + (H >> 16)`, lanes 2..31 at `L + tail` (its 4 tail nibbles, or 0 for a lane without cells), encodes every coded symbol in exact reverse decode order, and stores the 32 final states. The decoder requires every initial state `>= L`, lanes 0 and 1 ending at their hash values (whether or not they own cells), lanes 2..31 without cells ending at L, tail lanes ending in `[L, L + 2^16)`, and the word count exact. Any failure: the record is invalid (fail closed: the caller must not use what was written). Only valid / invalid is normative; which error bits a decoder reports for an invalid record is not.

## 5. Decoders

These are implementation notes; only sections 1-4 are normative.

- CPU reference: `wnr4/wnr4.h`: `wnr4_decode` (one record, OpenMP over its tiles) and `wnr4_decode_batch` (every tile of every record of a batch in one parallel loop). Tool API: `wnr4/tools/wnr4_tools.c`.
- CUDA: `wnr4/wnr4_cuda.cuh`. One warp per tile; a block holds 4, 8 or 16 tiles of the same matrix (chosen from the batch's tile count); the matrix's weight rows (compact order: only the 37 reachable (pm, mx) states per class) in shared memory or, as an option, read through L1; a 64-word window of the tile's word stream kept in two registers per lane and slid with one coalesced load per 32 words consumed, so a renormalising lane takes its word by shuffle; the decoded symbol's start and end read from its row by two 16-bit loads after a 16-way comparison; no `break` in the symbol loop, so the warp collectives stay on their convergent fast path; NVFP4 scales through a two-level table (a 16-value summary row, then the 16-value block row: two 32-byte loads per scale instead of an 8-step binary search); scales of phase 2 read back from the destination after `__syncwarp`; one launch for all tiles of a batch; output written straight to per-record, per-matrix destination pointers (device memory). Errors: per-record bit set (1 structure, 2 hash, 4 ceil, 8 CZ).
- Helper prototype: `wnr4/tools/wnr4_cuda_bridge.cu` (`wnr4_cuda_register_model`; `wnr4_cuda_batch_submit` / `wnr4_cuda_batch_wait` on a caller stream or a per-slot stream, up to 4 batches in flight per thread and device, a 5th before a wait refused with -4 so no verdict is dropped; `wnr4_cuda_decode_batch` synchronous; `wnr4_cuda_release_thread`). Records come from page-locked host memory; a record that fails reports -1 and its destination may hold partial data (the caller does not publish that slot); CUDA or memory errors and an unregistered model report 1.

## 6. Open

New WEX4 codec id (proposed 8, required feature bit), blob kind (3), conformance vectors, helper entry points and engine integration, fuzzing of the parser beyond the tests in the research folder.
