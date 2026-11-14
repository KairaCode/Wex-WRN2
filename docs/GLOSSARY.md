# Glossary

The words a first-time reader meets in this repository, grouped by subject. Every definition names the file it was checked against; paths are relative to the repository root. Numbers are quoted with the conditions their source gives.

Back to the [documentation index](README.md).

## The data

**Routed expert.** One feed-forward expert of a Mixture-of-Experts layer, chosen per token by the layer's router. It is three matrices: w1 (gate), w2 (down) and w3 (up). DeepSeek V4.1 REAP-192 has 40 layers x 192 = 7,680 of them, Qwen3.8-Flash-Next 48 x 512 = 24,576. Only routed experts are in the containers: shared experts, attention and the 384 MTP experts of V4.1 stay in the safetensors. *Source: `docs/WNR3-SPEC.md` sections 1 and 8; `docs/reports/wnr2-history-20261110.md` section 2 (geometry); `docs/reports/QWEN38-MXFP4-20261111.md` ("Containers"); `docs/WEX4-SPEC.md` section 12.3.*

**MXFP4.** The weight format of these experts: weights are 4-bit FP4 codes in blocks of 32 along a row, and each block has one 8-bit E8M0 scale. *Source: `docs/WNR3-SPEC.md` section 1.*

**FP4 (E2M1).** One 4-bit weight code, a nibble: bit 3 is the sign and bits 0 to 2 pick the magnitude from 0, 0.5, 1, 1.5, 2, 3, 4, 6. Two nibbles share a byte, the even one in the low half. Code 8 is -0 and code 0 is +0. *Source: `docs/WNR3-SPEC.md` section 1.*

**E8M0 (also written UE8M0).** The scale of a block of 32 weights: one byte `s`, and a weight's value is its FP4 value times 2^(s - 127). The C comments say UE8M0 and the safetensors dtype name is `F8_E8M0`; it is the same byte. *Source: `c/tools/repack_q38_mxfp4_wex.py` (function that rebuilds the values); `c/wex_neural_rans.h`; `docs/WEX4-SPEC.md` section 5.4.*

**Slab.** The decoded form of one expert in memory, and the content of a raw record: the scales of w1, w2, w3, then the packed FP4 weights of w1, w2, w3. A V4.1 slab is 18,800,640 B (17,694,720 B of weights and 1,105,920 B of scales), a Qwen3.8 slab 2,611,200 B (stored as a 2,613,248 B record, padded to 4096). *Source: `c/wex_format.h` (`WexEntry`, `aligned_expert_bytes`); `docs/WNR2-REFERENCE.md` section 5 and `docs/reports/wnr2-history-20261110.md` section 2; `docs/reports/QWEN38-MXFP4-20261111.md`.*

**Record.** What the container stores for one expert and what the engine fetches with one unbuffered read: a multiple of 4096 bytes at a 4096-aligned offset. A record is either the slab itself (raw, codec 0) or a coded form that decodes to it. *Source: `docs/WEX4-SPEC.md` section 2; `docs/reports/colibri-integration-20261109.md` section 1.1.*

**REAP-192.** The DeepSeek V4.1 checkpoint this work uses (DeepSeek V4.1 Flash 1017 REAP-192 MXFP4): 192 routed experts in each of 40 layers, where DeepSeek V4 Flash 0731 had 256 in each of 43. One planning note describes REAP as having pruned the cold experts; no file in this repository expands the acronym. *Source: `docs/reports/wnr2-history-20261110.md` section 2 (geometry); `research/v41-gpu-20261109/PLAN.md` (expected impact of G2).*

## Containers

**Container (WEX1 to WEX4).** The file layout around the records. WEX1, WEX2 and WEX3 share one layout: a 4096-byte header page, a table with one 72-byte entry per (layer, expert), then the records. WEX4 replaces the header and the table with a hashed header page and a hashed index and leaves the records as they are. The container version and the record codec are numbered separately; the full table of ids is in the root [README](../README.md) ("Format family"). The name is expanded in one place only, the October architecture page: "Windows Expert eXtension". *Source: `c/wex_format.h`; `c/wex4.h`; `docs/WEX4-SPEC.md` sections 3 to 5; `docs/reports/wex-architecture-20261018.it.md` (title).*

**Record codec (also "entry codec").** How one record is coded, a number in the record's table entry: 0 raw, 1 scale tiles (WEX2), 2 palette3 (lossy, never used in production), 3 WNR1, 4 WNR2, 5 WNR3, 6 WBP1 (WEX4 only). *Source: `c/wex_format.h` (`WEX_ENTRY_*`); `c/wex4.h` (`WEX4_CODEC_*`).*

**Root.** The SHA-256 of the first 4064 bytes of a WEX4 header page, stored at byte 4064. The header page holds the hash of the index and the index holds a SHA-256 of every record and blob, so these 32 bytes identify the whole file. Two copies are the same container when their roots are equal, and an engine can be told the expected root with `COLI_WEX_ROOT` ("pinning"). *Source: `docs/WEX4-SPEC.md` sections 4, 8 and 13.3.*

**Root pin (`COLI_WEX_ROOT`).** The expected root of a WEX4 container, 64 hex digits, given to the engine in its environment. A container with another root does not open, and under a pin a container that is not WEX4 is refused too (it has no root). The pin is what turns the hashes into protection against substitution: without it they detect damage only. *Source: `docs/WEX4-SPEC.md` sections 8 and 13.3.*

**`wex4_verify`.** The checker `c/tools/wex4_verify.c`: one read-only pass over a WEX4 file on several threads that checks the header page, the index and every blob and record against its SHA-256, then prints the line to pin, `COLI_WEX_ROOT=<root>`. It hashes and does not decode; `wex4/wex4.py verify` also decodes. *Source: `docs/WEX4-SPEC.md` section 15.1.*

**Conformance vectors.** The files of `wex4/vectors/`, each with the verdict, error class and root a reader must give (`MANIFEST.tsv`), for testing a reader written from the specification. *Source: `docs/WEX4-SPEC.md` section 17; `wex4/vectors/README.md`.*

**WBP1.** Record codec 6 of WEX4: order-0 rANS over byte planes, for tensors that are not MXFP4 experts (FP8, BF16, F32, integers). It is in `c/wex4.h` and tested; no production file holds such a record and no engine calls it. *Source: `docs/WEX4-SPEC.md` section 7.2; `STATUS.md` ("Not measured, not run or not verified").*

## Record codecs for MXFP4 experts

**rANS.** Range asymmetric numeral systems, the entropy coder of WNR1, WNR2, WNR3 and WBP1. Here: a 32-bit state, lower bound L = 2^23, 14-bit cumulative frequency tables (12-bit in the retired WNR1), byte-wise renormalisation. *Source: `docs/WNR3-SPEC.md` section 1; `docs/WNR2-REFERENCE.md` section 6.*

**Stream, chunk (Cw, Cs).** A record's symbols are cut into independent rANS streams so that a GPU decodes them in parallel: at most Cw weight symbols or Cs scale symbols per stream, never across a matrix. The two chunk sizes are stored in every record. V4.1 uses 2048:512, which gives 19,440 streams per record. *Source: `docs/WNR2-REFERENCE.md` section 6 and `docs/reports/wnr2-history-20261110.md` section 2.*

**WNM2 (and WNMD).** The probability model of a WNR2 or WNR3 container: a 59,408-byte table of cumulative frequencies, one 16-symbol table for each of 567 contexts of each of the 3 matrices, plus 3 scale tables. It is built from symbol counts over the experts; no neural network is involved, although code names say "neural". Every record names its model by a 32-bit FNV-1a id. WNMD is the 24-byte descriptor in a WEX3 header page that says where the model is; in WEX4 the model is a blob. *Source: `docs/WNR2-REFERENCE.md` sections 4, 6 and 7; `c/wex_neural_rans.h`.*

**WNR1, WNR2, WNR3.** Three generations of lossless rANS record codecs (codecs 3, 4, 5). WNR1 carried a model in every record and is retired. WNR2 uses the container's WNM2 model, with 567 weight contexts and u16 stream lengths; it has no checksum, and 1.8-1.9% of random single-bit flips in a whole real record were accepted on the CPU. WNR3 keeps WNR2's record skeleton and model format and adds ceil-form, rule-g anchors, optional canonical zero, a hash in every stream and the record's owner. The repository name spells the second one "WRN2"; every constant and file says WNR2. *Source: `docs/WNR2-REFERENCE.md` sections 4 and 7; `docs/WNR3-SPEC.md` sections 0 and 3; `docs/reports/WNR3-VERIFY.md` (flip audit).*

**Ceil-form.** A reversible rewrite WNR3 applies per matrix before coding. A block whose largest magnitude code is 7 and that uses neither code 1 nor code 3 is stored with its magnitudes halved and its scale raised by one (unless the scale is already 255); the decoder undoes this for every block whose coded maximum is 5. The encoder uses it for a matrix only when no block of that matrix already has maximum 5, which is what makes it exactly reversible. *Source: `docs/WNR3-SPEC.md` section 4.2.*

**Rule g (anchor).** How WNR3 picks the anchor of a matrix, the scale value that the weight context measures a block's scale against: the value `v` with the most scales equal to `v` or `v + 1`. It replaces WNR2's median, is stored in record bytes 24 to 26, and decoders read it and never recompute it. Ceil-form and rule g together measured 0.88179 of raw against 0.89943 for WNR2 on 40 held-out V4.1 experts. *Source: `docs/WNR3-SPEC.md` sections 0 and 4.3.*

**Canonical zero (CZ, "WNR3-CZ"), value-exact.** An option of a WNR3 build: every FP4 -0 is stored as +0. The record is then **value-exact**: it decodes to the source values but not to the source bytes (the opposite is byte-exact, and canonical zero is the only value-exact case). The container carries header flag 64 (in WEX4 the VALUE_EXACT bits), and the original bytes cannot be rebuilt from it. It is worth 2.74% of raw on V4.1. The engine refuses such a container on the masked DeepGEMM path (`DSV4_CUDA_MOE_MASKED`), which is not proven value-exact. The Quark export of Qwen3.8 has no -0, so the option gives nothing there. *Source: `docs/WNR3-SPEC.md` section 4.4; `docs/WEX4-SPEC.md` sections 2 and 12.3; `docs/reports/QWEN38-MXFP4-20261111.md`.*

**Stream hash (h23).** WNR3's integrity check: 23 bits of a MurmurHash3 over a stream's decoded symbols and its identity (model id, owner, stream index, anchors and flags) are folded into the stream's start state, and the decoder requires that exact final state. A damaged or misplaced stream passes with probability about 2^-23. Measured: 0 of 2,231,336 single-bit flips accepted on the GPU, 0 of 3,000 random ones on the CPU. *Source: `docs/WNR3-SPEC.md` section 4.5; `docs/reports/WNR3-VERIFY.md` section 2.*

**Owner.** The slot a record belongs to, `layer << 16 | expert`, written in WNR3 record bytes 28 to 31 and repeated in a WEX4 index entry. The engine compares it before decoding, so a record served for the wrong expert is refused. *Source: `docs/WNR3-SPEC.md` sections 3 and 4.7; `docs/WEX4-SPEC.md` sections 5.2 and 7.1.*

## Reading and decoding

**Helper DLL (`kaira_wex_cuda.dll`) and its feature bits.** The optional CUDA library that decodes compressed records, built from `c/tools/wex_cuda_bridge.cu` with the kernels of `c/wex_cuda.cuh`. The engine loads it the first time a record needs it and falls back to the CPU decoder, or refuses the container, without it. ABI 1 means it has `wex_cuda_decode_to_host`. The feature bits are what `wex_cuda_helper_features()` returns, one bit per optional ability; today's bridge returns 63.

| bit | ability |
|---|---|
| 1 | decode one record straight into caller-owned GPU buffers |
| 2 | page-lock the caller's host buffers |
| 4 | WNR2 records, with a registered WNM2 model |
| 8 | a batch of records decoded into GPU buffers in one call |
| 16 | WNR3 records in the single-record and batch-to-GPU entry points |
| 32 | a batch of WNR2 or WNR3 records decoded into host buffers in one call |

*Source: `c/tools/wex_cuda_bridge.cu` (header comment and `wex_cuda_helper_features`); `c/wex_decode.h`.*

**GPU-direct, mode 0, 1, 2.** `COLI_WEX_GPU_DIRECT` in the DeepSeek engine. Mode 0 reads a record into a host slab, decodes it there and uploads the slab. Mode 1 sends the compressed record to the GPU and decodes it straight into the expert's buffers in VRAM, so the decoded slab never passes through host RAM. Mode 2 also lets the next-layer prefetcher do that. Raw records never go direct. *Source: `docs/reports/colibri-integration-20261109.md` sections 1.2 and 5.5.*

**Level A, B, C.** The three steps in which the container hooks are ported into `deepseek_v4.c`: A is mode 0 and is enough to read any container; B is the store side of GPU-direct and the RAM cache of compressed records; C is the GPU side of GPU-direct. Only level A has a patch for stock colibri. *Source: `docs/reports/colibri-integration-20261109.md` section 5.*

**Mirror, replica.** Four different things carry the word "mirror". (1) A **container replica**: a byte copy of the container on another drive, in a folder named by `COLI_WEX_EXTRA_MIRRORS`, accepted only when it matches the primary (header page and entry table; for WEX4, the root). (2) A **shard mirror**: a folder with copies of the safetensors shards, named by `COLI_MODEL_MIRROR`. (3) A **VRAM mirror**: the copy of one expert's weights and scales that the DeepSeek engine keeps on the GPU. (4) The **source mirror**: the `c/` folder of this repository, which the maintainer keeps identical to `src/colibri/c` of the application tree (`sync-map.tsv`, the `mirror` line of `check_repo.py`). When a page says "U: with a full mirror on C:" it means (1). *Source: `docs/reports/colibri-integration-20261109.md` sections 1.2, 5.1 (hook A6), 5.3 and 5.5; `docs/WEX4-SPEC.md` section 8.3.*

**Read split, striped read.** With a replica, `COLI_WEX_WEIGHTS=38,62` sends 38% of the experts to the primary and 62% to the replica; a fixed hash of (layer, expert) decides, so one expert always comes from the same drive. A striped read (`V4_WEX_STRIPE`, `COLI_WEX_STRIPE`) instead cuts one record in two and reads the parts from both drives at once; it measured slower on V4.1 and is off. *Source: `docs/reports/colibri-integration-20261109.md` section 12 and the box at its top; `STATUS.md`.*

**Tier (RAM tier, VRAM tier).** Where an expert that is not read from disk at this token is kept: decoded in host RAM slots, or resident on the GPU. An expert in neither tier is a miss and costs a read. *Source: `docs/reports/colibri-integration-20261109.md` section 1.2; `docs/reports/QWEN38-MXFP4-20261111.md`.*

**Integrity lane, first-read SHA-256.** The engine's check of a record's stored bytes against the SHA-256 in the WEX4 index, the first time a process reads that record (`COLI_WEX_VERIFY`; on by default in the DeepSeek engine when the root is pinned, off by default in qwen38). A record that fails stays refused for the rest of the process. `V4_WEX_SHA_ASYNC=1` moves the hashing of the DeepSeek engine onto a pool of 4 threads (the hash pool). "W06" is the label, in the engine's code comments, of the rule that a pinned root refuses a file that is not WEX4. *Source: `docs/WEX4-SPEC.md` section 8.2; `docs/COLIBRI-INTEGRATION.md` sections 5 and 7.*

**Sub-reads (`V4_WEX_SUBREADS`).** The DeepSeek engine reads each drive's part of a record as N overlapped reads instead of one, and with a mirror cuts the record across both drives; N = 4 in the `v41gpu` profile. *Source: `docs/COLIBRI-INTEGRATION.md` section 5; `docs/BENCHMARKS.md`.*

**Tier prewarm (`COLI_V4_TIER_PREWARM_AHEAD`, `V4_PREWARM_BATCH`).** At start-up, with `COLI_V4_TIER_PREWARM=1`, the DeepSeek engine fills the VRAM tier with the experts it used most. By default it does so one record at a time; `COLI_V4_TIER_PREWARM_AHEAD=N` keeps N reads in flight, and `V4_PREWARM_BATCH=1` reads and decodes batches of 8 records through the GPU-direct batch path, two batches deep. *Source: `docs/COLIBRI-INTEGRATION.md` section 5; `docs/BENCHMARKS.md`.*

**`wexio_probe`.** `c/tools/wexio_probe.py`: a read-only drive probe (Windows) that reads a container at random 4096-aligned offsets with N reads in flight, per request size, from each drive alone and from both, and prints GB/s and latency. *Source: its header comment; `docs/BENCHMARKS.md` section "Drives".*

**Arena.** The qwen38 engine's VRAM tier with slots of each expert's exact size (`QT_ARENA=1`) instead of padded slots; by the research's count it holds 9,049 experts instead of 7,178 at the same charged budget. Engine code, not container code. *Source: `docs/reports/QWEN38-MXFP4-20261111.md` ("After the install").*

**Prefix store.** The qwen38 engine's cache of the model state after the system prompt, reused by later turns; in one control run it held up to 1089 MiB and reused 65-75% of later turns' prompts. Engine code, not container code. The "prefix-checkpoint slots" of the DeepSeek V4.1 RAM plan (210 MiB each) are a different mechanism of that engine, which this repository does not describe further. *Source: `docs/reports/QWEN38-MXFP4-20261111.md`; `STATUS.md` (Qwen3.8 and low-RAM sections).*

## Engine and host

**colibri.** The inference engine these formats plug into: C sources in one folder (`c/` upstream, `src/colibri/c` in the maintainer's tree), one executable per model family. `deepseek_v4` and `qwen38` are the two that read these containers. "Stock" or "upstream" colibri is the public engine without the maintainer's changes. *Source: `docs/reports/colibri-integration-20261109.md` section 2.*

**U: and C:.** The two NVMe drives of the test machine, U: on PCIe 4 and C: on PCIe 5, each holding a full copy of the production containers; the engines split reads 38/62 between them. *Source: `docs/BENCHMARKS.md`.*

**Application and its profiles (`v41gpu`, `qwen38`).** The maintainer's application ships and runs colibri: a host program, the engine binaries under `app\`, and model profiles in its configuration file. Its source tree also carries everything under `c/`; in paths it is written `<TREE>`. A profile is one entry of that configuration file: a model, its engine and the environment variables passed to it, among them the container path, the replica folder and the read split. *Source: `docs/reports/colibri-integration-20261109.md` sections 5.8 and 8.2; `CONTRIBUTING.md` section 1.*

## Words in the measurements

**TTFT and decode tok/s.** TTFT is the time to first token: from sending the prompt to the first generated token, on the client's clock. It grows with the prompt length, so it is always quoted with the prompt. Decode tok/s is the generated tokens per second counted from the first token, so the time spent on the prompt (prefill) is in TTFT and not in it. *Source: `docs/reports/v41-reap-baseline-m0-20261107.md` (setup, and section 2 for what a run records).*

**ABBA.** The run order A1 B1 B2 A2 of two arms A and B, each run in a fresh process, so that a slow drift of the machine does not favour one arm. "ABBA x2" is two such sets, four pairs. "One pair" is one A and one B and gives no interval. *Source: `docs/reports/colibri-integration-20261109.md` section 12; `docs/reports/wnr2-history-20261110.md` section 1 (measurement protocol) and "W3 final".*

**Bit-identical, identical tokens.** Two runs produced the same logits file byte for byte, or the same token ids. These are statements about correctness and are kept apart from speed. *Source: `CONTRIBUTING.md` section 5.*

**[M], [I], [R].** Tags in the 2026-11-07 to 2026-11-09 plans and reports: measured here, inferred, read in the code. The Italian reports of 2026-11-05 use [misurato], [simulazione], [pubblicato], [inferenza]. A run marked `*` in `docs/reports/wnr2-history-20261110.md` (section 12) is contaminated: other work was running on the machine. *Source: `research/v41-gpu-20261109/PLAN.md` (header); `docs/reports/wnr2-cache-decode-cpu-wddm-20261105.it.md`; `docs/WNR2-REFERENCE.md`.*

**Lane, gate and goal labels.** Short labels name work packages and acceptance gates. The same letter can mean different things in different files:

| label | where | meaning |
|---|---|---|
| M0 | V4.1 GPU plan | CPU-only baseline and drive probe |
| G1, G2, G3 | V4.1 GPU plan | V4.1 on the GPU tiers; G2 is the routed experts through the GPU pipeline with VRAM mirrors |
| P1 | V4.1 GPU plan | prefill changes |
| W1 | V4.1 GPU plan | the build of the V4.1 WNR2 container |
| W2 | V4.1 GPU plan | one batched decode per layer (`V4_WEX_BATCH`) |
| W3 | V4.1 GPU plan | the A/B of WEX1 against WNR2 on V4.1 |
| F1 | V4.1 GPU plan | confidence-gated prefetch |
| D1 | V4.1 GPU plan | a second drive |
| H1 | V4.1 GPU plan | the host passing the container variables for V4.1 |
| G1 to G8 | WNR3 specification, section 7.2 | acceptance gates of WNR3; G7 is the end-to-end proof of canonical zero, G8 the full container |
| M1 to M4 | WEX4 specification, section 13.5 | migration gates of WEX4 |
| G1 to G7 | WEX4 specification, section 1.1 | design goals of WEX4 |

*Source: `research/v41-gpu-20261109/PLAN.md`; `docs/WNR3-SPEC.md` section 7.2; `docs/WEX4-SPEC.md` sections 1.1 and 13.5.*
