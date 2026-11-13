# WNR2 brief for research agents (from WNR2-REFERENCE.md, state 2026-11-07)

Full reference (80 KB, read with offset/limit if you need detail): `D:\Desktop\WNR2-REFERENCE.md`.

## What WNR2 is
- Lossless entropy coding of MXFP4 MoE expert records (OCP MX: FP4 E2M1 weights, 1 UE8M0/E8M0 scale per 32 consecutive weights along the input dimension). KAIRA's own engine (a private fork of colibri) offloads MoE experts from NVMe -> RAM -> VRAM; one record = one expert = 3 matrices (gate w1, down w2, up w3).
- Target model: DeepSeek V4.1 Flash REAP-192 MXFP4: 40 layers x 192 experts = 7680 records, hidden 5120, moe_inter 2304, top-6. Raw record 18,800,640 B = 17,694,720 B FP4 weights + 1,105,920 B scales. Previous model V4: 43 x 256 experts, 13.4 MB raw record.
- Container: WEX3 (4096-aligned header page, 72-B entry table, records contiguous, each padded to 4096). One shared probability model "WNM2" (59,408 B) per container; records reference it by FNV-1a id.
- Weights: rANS (32-bit state, L = 2^23, byte renormalization, 14-bit CDFs) with a STATIC shared model of 567 contexts per matrix type (3 matrix types): context = (srel*9 + pm)*9 + mx where srel = clamp(block_scale - median_scale_of_matrix, -3, 3)+3 (7 values), pm = magnitude (x&7) of the previous symbol in the 32-block (8 = none), mx = max magnitude so far in the block (8 = none). 16-symbol CDF per context. Symbols are row-major 4-bit codes.
- Scales: order-0 rANS, one 256-symbol CDF per matrix type.
- Streams: independent rANS streams that never cross a matrix; chunk = 2048 weights / 512 scales per stream on V4.1 (19,440 streams per record, u16 lengths). One GPU thread decodes one stream (latency-bound, ~200 cycles/symbol/lane); a block serves one matrix; CDF rows in shared memory, branch-free 16-way compare; payload ring refilled with cp.async.
- No per-record checksum: corruption is caught only by rANS structural invariants (initial state range, exact stream length, final state == L).
- WNR2z variant: maps FP4 -0 (code 8) to +0 before coding (value-exact, not byte-exact): -0 is 12.8% of weights, saves ~0.128 bit/w; not production (needs logprob check).

## Measured numbers (prefer these)
- V4.1: average record 16,914,680 B = 0.8997 of raw (-10.03%); weights 3.781 bit/weight; scales 1.066 bit/scale; all-in 3.824 bit/w; stream-length overhead 38,880 B/record; padding ~2 KB/record.
- V4: 3.670 bit/w weights, 1.021 bit/scale, -12.78%.
- Encoding V4.1 records with the OLD V4 model gave 0.941-0.950 of raw vs 0.8997 with a retrained model -> model specialisation matters a lot.
- Order-0 entropy: FP4 nibbles 3.866 bit (order-0 saves at most -3.4% on weights); scales 0.98 bit.
- Held-out cross-entropy experiments on V4 (shared model): scale+previous symbol 3.684 bit/w; + block max 3.665; + position in block 3.660. Rows, columns, the upper neighbour row and gate<->up correlation gave NOTHING.
- Generic codecs on decoded MXFP4 slabs (ratio whole / weights / scales): zstd-3 0.9225/0.968/0.189; zstd-19 0.917; deflate-6 0.920; lzma-6 0.920; WNR2 0.866 (V4).
- Chunk size vs size: 512 weights/stream +2.1%, 1024 +0.9%, 2048 +0.28%, vs 4096 baseline.
- GPU decode (RTX 5090, WDDM, 1 copy engine, PCIe ~48.6 GiB/s pinned): kernel-only one record 0.2 ms at 2048:512; 8 concurrent records 102.8 GiB/s decoded; batch of 8 RAM->VRAM 0.28 ms/record, PCIe-bound (floor ~0.22 ms); raw WEX1 slab upload ~0.26 ms. So on the PCIe path the decode is roughly free; the format's value is CAPACITY (+11.1% records in a fixed RAM/VRAM/disk budget) and fewer NVMe bytes.
- CPU decode (reference): ~0.3 s/expert (~44 MB/s), unusable; SIMD rANS 0.5-1.4 GB/s/core published.
- System: disk-bound MoE offload; machine has 32 GB RAM; NVMe drives 1.6-10.7 GB/s; V4.1 needs ~214 expert misses/token x 18.8 MB. Simulation: WNR2 cuts bytes 13-16% but disk reads only 1-2%; a COMPRESSED VRAM tier (decode from VRAM on use) was estimated at ~19-20% fewer reads.
- Decision rule: WNR2 becomes the speed format only with >= +3% tok/s (95% CI excluding 0) and identical tokens; otherwise it stays a capacity format. Any change must stay LOSSLESS (byte-exact preferred; value-exact needs a logprob check).

## Already researched / ruled out (do NOT re-propose unless you have genuinely new evidence; say so explicitly)
- ZipServ (fixed 3-bit code on BF16, not MXFP4); DFloat11 and ECF8 (gains come from memory/batch, not decode speed); ZipMoE (CPU decode NVMe->RAM); nvCOMP Decompression Engine (B200/B300/GB200/GB300 only); arXiv 2606.15789 and snuri00/vram-compression (2-3x slower than cuBLAS); Recoil (>90 GB/s on 2080 Ti, reference point); ZipNN (claims FP4 bits near-uniform, conflicts with measured 3.67 bit/w).
- Fused rANS-in-GEMV (needs ~2 TB/s decoded, ~19x short); green contexts; GPUDirect Storage on Windows GeForce; IoRing as a throughput lever; MPS on Windows; HOBBIT (lossy substitution); W-TinyLFU; DirectStorage/GDeflate (~-8% only); Palette3 (lossy 3-bit).
- Cache-policy and prefetch work (F1 gating, LRU vs LFU, S3-FIFO, FlashMoE +21% hit claim) is tracked separately: mention only if it needs a FORMAT change.

## What "improve the format" means here
Any change to the on-disk/in-memory record format, codec, model or container that, losslessly:
1. lowers bytes per record (bit/weight or bit/scale) — every -1% is +1.1% capacity;
2. keeps or improves GPU decode speed (must stay ~0.2 ms/record or better; enabling decode-from-VRAM for a compressed VRAM tier needs high decode throughput);
3. improves robustness (integrity checks) or engineering (alignment, layout, per-matrix pipelining, multi-model/per-layer models) at negligible cost.
Quantify against the numbers above whenever possible.
