# Literature notes for WNR4 (2026-11-21)

Survey done by a web-research pass on 2026-11-21; numbers as the sources state them, "unverified" where only a snippet or no number was found. No source reports results on sm_120 / RTX 5090.

## GPU rANS / interleaved ANS

- **DietGPU** (Johnson, facebookresearch/dietgpu, README and `GpuANSDecode.cuh`): 32-bit state (kept below 2^31), 16-bit renormalisation words, L = 2^15, 9-11 probability bits, 256-symbol alphabet, decode table of 2^probBits 4-byte entries in shared memory. One warp decodes one 4 KiB block; the 32 lanes share one word stream: `read = state < L`, `__ballot_sync`, `__popc` of the lower lanes gives each lane its word, the pointer advances by the vote count. No per-lane final-state check (optional checksum). 250-410 GB/s (ANS codec) on an A100. **WNR4 uses the same shared-stream renormalisation**, with a context model instead of one table per block, and checks every lane's final state.
- **Recoil** (Lin et al., ICPP 2023, arXiv 2306.12141): one 32-way interleaved rANS stream (32-bit state, L = 2^16, 16-bit words, up to 16 probability bits); decoder-side split points stored as metadata (renormalisation-point states and symbol indices), so the encoded stream is the same for any parallelism. 90+ GB/s kernel time on an RTX 2080 Ti, 11+ GB/s on a 16-core Xeon. Overhead +0.03% on enwik9 at n = 16.
- **multians** (Weissenberger and Schmidt, ICPP 2019): tANS self-synchronisation, no partitioning; overhead too large at high precision according to Recoil. GB/s unverified.
- **Giesen, "Interleaved entropy coders"** (arXiv 1402.3392): N coders can share one buffer with no metadata if encoder and decoder visit them in the same order; the decoder ends in the encoder's initial state. CPU: 8-way SSE4.1 564-573 MiB/s (i7-2600K).
- **ryg_rans**: rans_word_sse41 8-way with 16-bit I/O 520-551 MB/s (i7-2600K).
- **nvCOMP ANS**: no primary GB/s found (one blog: 181-190 GB/s on B200, direction unverified).

## GPU Huffman / LUT decoders

- Weissenberger and Schmidt GPU Huffman (ICPP 2018), Yamamoto et al. gap arrays (ICPP 2020): GB/s unverified.
- Rivera et al. (arXiv 2201.09118), V100, 16-bit cuSZ codes: optimised gap-array 84.7-123.9 GB/s, optimised self-sync 63.1-101.9 GB/s.
- **DFloat11** (arXiv 2504.11651, NeurIPS 2025): Huffman on BF16 exponents, hierarchical 256-entry LUTs in shared memory, two-phase decode with prefix sum; 67.6-69.5% of BF16 size; "up to 20.97x" faster than nvCOMP ANS; no absolute GB/s in the text.
- **ECF8** (arXiv 2510.02676): FP8 exponents, Huffman with 256-entry hierarchical LUTs, tested up to the RTX 5090; 9.8-14.8% saving on LLMs; no decode GB/s.
- For a 16-symbol alphabet at ~3.6-3.8 bit/symbol the integer code lengths of Huffman would cost a large part of an 8-15% gain (inference); rANS keeps fractional costs.

## Lossless compression of low-bit weights

- ZipNN (arXiv 2411.05239): exponent Huffman for BF16 (~33%); no 4-bit result.
- Intel, Heilper and Singer (arXiv 2508.19263): FP4 values found essentially incompressible with Huffman; only NVFP4 scale factors compress.
- Tan et al., ISCA 2026 (arXiv 2606.15789): rANS on DietGPU's kernel with per-lane substreams and a 4-byte offset per tile, shared codebook per layer, 12 probability bits; INT4 (not MXFP4) entropy far below ours; no context modelling, no GB/s.
- EntroLLM, Huff-LLM, ZipLLM, NeuZip, ZipServ: gains come from the quantiser, from BF16 exponents, from deduplication across models or from lossy modes; nothing context-models MXFP4/NVFP4 nibbles. **Gap found by the survey: no published work models FP4 nibbles with scale classes as WNR3/WNR4 do.**

## Hardware decompression

- Blackwell Decompression Engine (nvCOMP FAQ): B200, B300, GB200, GB300 only; Snappy, LZ4, Deflate, GZip; up to 600 GB/s. Nothing for GeForce/sm_120, no ANS.
- GDeflate / DirectStorage on GeForce runs on the SMs; third-party figures ~15-33 GB/s (unverified).

## Transfer model (nvCOMP blog, Sakharnykh et al., 2020)

"Optimizing data transfer using lossless compression with nvCOMP": effective throughput `B (K-1) / (T_comp + T_transfer + T_decomp)` for an all-gather; compression only helps with fast GPU compress/decompress; overlap of copies and kernels left as an exercise. Applied to expert loads (our derivation): with link bandwidth W (NVMe 6-14 GB/s, PCIe), stored/raw ratio q and decode throughput X,
- **sequential** read then decode: compressed wins iff `q/W + 1/X < 1/W`, i.e. `X > W / (1 - q)`. For q = 0.88: X > 8.3 W (50 GB/s on U: at 6 GB/s, 117 GB/s for 14 GB/s striped);
- **pipelined** (decode of batch i overlaps the read of batch i+1): time per byte `max(q/W, 1/X)`, so compressed wins iff `X > W` and then saves the full `1 - q` of read time. This is why WNR4 is built for batched, overlapped decode and why the >= 40 GB/s target of the engine-side analysis is the right order of magnitude.

## Further papers and posts read on 2026-11-21

- **ENEC** (Yang et al., arXiv 2604.03298, 2026): lossless BF16/FP16/FP32 weights on Ascend NPUs; no entropy coder: exponent remapping by frequency plus block fixed-length bit packing, offsets by prefix sums. 188-336 GB/s decompression on Ascend 910B2; GPU port (A800, Qwen-32B) ratio 1.35 at about 421 GB/s, against a DietGPU-like rANS at ratio 1.47 and about 271 GB/s. FP4/MXFP4/NVFP4 not addressed. Lesson for us: fixed-length packing trades ratio for speed; on FP4 nibbles (3.6-3.9 bit entropy of 4) a fixed-length code gains nothing, so it does not apply.
- **FMT** (arXiv 2611.23108, 2026): lossless 8-bit images in a fixed number of parallel steps; rANS with interleaved lanes whose state is carried across steps (one lane per GPU thread), giving each step its own lanes would make files 13% larger. WNR4 already does the same (a tile's 32 lanes carry their state across the scale phase and the cell rounds). The gains of FMT come from spatial prediction, which FP4 weights do not have (mutual information across rows, experts and layers is ~0).
- **DFloat11** (arXiv 2504.11651 v2): kernel details above; no absolute GB/s in the text; A100 used for the latency breakdown.
- **NVIDIA Dynamic Memory Compression**, **Google TurboQuant**, **MIT/NVIDIA QDA**: lossy (KV-cache compression, vector quantization, automated quantization recipes). They change the model's numbers, so they are out of scope for a lossless expert codec; they could only matter upstream, as a different quantized checkpoint.
- **ZipNN** (github.com/zipnn/zipnn): byte grouping + Huffman/zstd for BF16/FP32 exponents; for FP4 the bytes are already dense nibbles, so its exponent split has nothing to act on.

## MoE offloading and model I/O papers (read 2026-11-21, one reader + one checker per paper)

None of the nine compresses weights losslessly, so none changes the WNR4 format; their value is in when and how a codec like WNR4 pays off in an offloading engine.

| paper | what it is | what carries over |
|---|---|---|
| Edge0 (arXiv 2609.18063) | SSD-streamed MoE engine (mmap, page cache, double-buffered slots), a "prerouter" predicts layer N+1's routing for token t+1 and uses it as the routing (lossy, int4 g64 + distilled LoRA); 35B MoE at ~20 tok/s in ~3 GiB on a 24 GB Mac mini | overlapping expert reads with compute is the lever (blocked time per step 244.0 -> 101.9 ms, decode +80-84%); a per-load fixed cost (their fit 1.17 ms + 1.33 ms x cold fraction) favours fewer, larger reads; prediction over-fetch (+16% bytes at K = 4) can eat the gain. The routing replacement is lossy: not for a lossless path |
| MoE-CORE (2610.01950) | host-to-NPU expert offloading scheduler: per-layer non-uniform decode caches from routing traces, a frequency/EMA/recency/router-score eviction, cross-layer prefetch with the next layer's router | the miss cost model `t = tau_launch + s / beta` (fixed + bytes over bandwidth): bytes matter for 18.8 MB V4.1 records, the fixed term for 2.6 MB Qwen3.8 records; prefetch count bounded by the overlap window and free slots; non-uniform per-layer slots; its approximate substitution is lossy |
| ExpertFlow (2410.17954, DAC '26) | single-GPU offloading with an all-layer routing predictor run before the first MoE layer, token rebatching, predictive per-layer cache (hit ratio 91.96%, up to 61.15 points over LRU on Switch models) | an all-layer lookahead gives the longest prefetch horizon, i.e. the batches a batched GPU decoder needs; orthogonal to quantization/codecs |
| AdaOffload (2611.10219) | multi-GPU hybrid CPU/GPU expert execution with an online fetch-vs-CPU-run split chosen from a profiled cost model | its cost is `max(compute, transfer)`: with WNR4 the transfer term becomes compressed bytes / min(NVMe, decode), our pipelining condition; no SSD tier, no compression |
| ExFold (2608.24938) | training-free folding of excluded experts into retained ones (lossy), up to 2.45x TPOT | fewer distinct experts per step cuts reads directly, but lossy: a separate mode, not part of WNR4 |
| GMS / Dynamo (2609.25451) | restart of inference replicas: GPU-memory service keeps model allocations alive across process death, shadow engines | stable virtual addresses for VRAM slots keep captured CUDA graphs valid; relevant if WNR4 decodes straight into graph-referenced expert slots |
| Checkpoint/restore I/O (2512.24511) | liburing vs POSIX, O_DIRECT vs buffered, aggregation for multi-GB checkpoints on Lustre | pre-allocated aligned buffers, batched submissions matched to queue depth, no allocation in the per-read path |
| LayerCheck (2609.27193), MoSEs (2511.06237) | training-time selective checkpointing; continual-learning adapters | nothing for WNR4 or expert streaming |

Reading for WNR4: the codec saves 7-15% of bytes, so it helps only on the bytes term of a miss and only when decode runs behind the reads. All the engine-side levers these papers measure (lookahead prediction, overlap, coalescing, cache policy) also create the batches that WNR4's one-launch-per-batch decoder needs; without them a single 2.6 MB miss is dominated by fixed costs, compressed or not.
- **GAAC** (Guo, George, Joshi, arXiv 2610.38210, 2026): lossless BF16 KV-cache compression, exponents only (sign and mantissa raw), groups of 8 exponents with a 2-bit mode tag (3-bit, 4-bit or raw offset from a calibrated per-channel base), reconstructed inside a fused attention kernel; 1.405-1.407x on BF16 KV, attention 1.19-1.27x vs FlashAttention-2 at 64K context on a GB10, end-to-end W4A16 decode step 1.04-1.15x; FP8 KV only 1.09-1.30x. Not about weights, no entropy coder, no storage tier. Lessons it confirms: decode-first can cost more than the bytes it saves (930.8 us of exponent decode on a 64K cache, hence the fusion), and a grouped mode tag with a raw fallback keeps outliers cheap. For WNR4 the analogue of "decode hidden under the consumer" is decode hidden under the NVMe read.
