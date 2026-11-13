---
name: wnr2-complete
description: "Everything about WNR2 (KAIRA neural-rANS expert container): format, encoder, decoders, helper, engine integration, containers, measurements, decisions, open work"
metadata:
  node_type: memory
  type: project
  origin: KAIRA private memory note (snapshot)
  modified: 2026-11-07T16:17:28.409Z
---

# WNR2 complete reference (state at 2026-11-07; old snapshot kept so the kit's REF:<line> citations stay valid. Current version: docs/WNR2-REFERENCE.md at the repo root)

**Why:** WNR2 work is spread over three lab folders, two reports, the private Wex-WRN2 repo and six memory notes, and those sources disagree on key numbers. This file is the single place that says what WNR2 is, where every container and model sits, what was measured under which conditions, and what the user allowed.

**How to apply:**
- Before any WNR2/WEX task, read sections 1 (rules: no upstream, no deploy or kaira.json edit without OK, Opus agents, heavy.lock, no benchmarks during builds), 2 (disk state) and 3 (open work: W2 -> W3 order, G2 first). They come first because the file is larger than one Read (~80 KB); read sections 4-15 with offset/limit when needed (format/codec 4-7, build 8, decoders/helper 9-10, engine 11, measurements 12, timeline 13, contradictions 14, sources 15). Runs marked * are contaminated.
- Re-check disk free space and the existence of the files before relying on them.
- Prefer measured V4.1 numbers (-10.03%, +11.1%) over older V4 estimates.
- After any format change, update KAIRA src first, then re-sync Wex-WRN2.

Related: private KAIRA memory notes (not in this repo).

## 1. User decisions and rules

- **2026-10-20 04:09**: the user approved installing ONLY the WEX1-compatible runtime.
  - Lab engine 7d139737, backend d05fd43b (race fix) and helper cac5348d went into app\colibri-optimized; kaira.json unchanged.
  - Backup: lab\app-before-wexgd-20261020-040912. Rollback: copy back deepseek_v4.exe + coli_cuda_dsv4_dg.dll and delete kaira_wex_cuda.dll.
  - Switching KAIRA to WNR2, GPU-direct or the RAM cache via kaira.json was NOT approved. Verdict: WNR2 is a SPACE/capacity option.
- **2026-11-05**: the user chose to port WEX to Qwen3.8 (lever there: C:+U: striping, not compression; the WEX-Q8 container on C: needs OK on space; report lab\wex-prod-20261105\REPORT.md).
  - Then V4 was replaced by DeepSeek V4.1 Flash 1017 REAP-192 MXFP4 (patrickbdevaney/...), not "Thor", because Thor's K-quant experts are unreadable by the engine and by WNR2.
  - Deletion of the old V4 WNR2 container was approved.
- **2026-11-06**: containers go WEX1 raw on D:\kaira_wex and WNR2 on E:\kaira_wex.
  - All V4.1 + WEX/WNR2 agents run on Opus (user rule).
  - Never commit, push, deploy to app\ or edit app\kaira.json without explicit OK.
  - The user manages the Defender exclusions; Claude does not change security settings.
  - No benchmark while a W1 encode or any build runs; take heavy.lock for big I/O.
- **heavy.lock**: `<KAIRA>\lab\v41-gpu-20261107\heavy.lock`.
  - Take it by creating the file exclusively (New-Item fails while it exists) and writing `<label> <ISO time>` into it. `lab\v41-gpu-20261107\w1\take_lock.ps1 -Label '...'` does this, retrying every 60 s for up to 90 min (prints TAKEN or TIMEOUT).
  - Release it by deleting the file.
- **2026-11-07 ~00:47-01:10**: V4.1 installed into app\ from src, including kaira_wex_cuda.dll built in ws-int. Backup lab\app-before-v41-full-20261107-004555, hash list lab\install-v41-full-20261107.txt; coli_cuda.dll not replaced; kaira.json unchanged.
- **2026-11-07 ~02:00**: the user approved:
  - copying the WEX1 to U:\kaira_wex\v41-reap (use only when .verified exists);
  - moving the V4 checkpoint to H: with a junction;
  - for W3, putting both containers on U: if space allows (copy under heavy.lock, verify size + sha256, write .verified), otherwise reporting the drive confound.
- **2026-11-07 ~03:30**: KAIRA-only decision. WEX1/WNR2 code must NEVER go upstream to JustVugg/colibri, and WEX/WNR2 hunks are stripped from upstream PRs.
  - Code covered: expert_store_wex.c, wex_decode.c/.h, tools\wex_cuda_bridge.cu, wex_cuda.cuh, repack_v4_wex.py, the nr2 encoder.
  - It lives in the PRIVATE repo https://github.com/KairaCode/Wex-WRN2 (clone <HOME>\Wex-WRN2, initial commit a74b279 synced from KAIRA master 88ddb07, created ~16:10).
  - Pushed: main = a74b279 tracks origin/main (https://github.com/KairaCode/Wex-WRN2.git) and is in sync with KAIRA 88ddb07 (checked 2026-11-07 17:11). src\wex_cuda.cuh is byte-identical to KAIRA's; models\ holds v4-flash-0731.wnm2 and v41-reap192.wnm2.
  - KAIRA src\colibri\c stays the live copy; re-sync the repo after format changes.
  - Reason: beating safetensors is KAIRA's own asset (WEX1 on U: ~1.12 vs safetensors ~0.99 tok/s).
- **Install rule**: a WNR2 deploy needs the new engine AND a helper with feature 4 together. With an old helper, WNR2 went to the CPU at ~0.29 s/expert; since 2026-11-05 the engine refuses the container instead. Ask before installing anything.
- **Decision rules**:
  - Nov 5: if < ~70 avoidable NVMe misses/token and the batch pipeline does not go under 0.22 ms, WEX1 stays. Round 2: if disk is not really the bottleneck, WEX1 stays.
  - PLAN 3.2 for V4.1: WNR2 becomes the speed format only if >= +3% tok/s with the 95% CI excluding 0 AND token ids identical in every prompt of every run; otherwise WEX1 stays in production and WNR2 is a capacity format.
  - Any token difference stops the A/B and is treated as a bug.
- **Measurement protocol** (PLAN 3.1):
  - Quiet machine: no KAIRA/engine, sub-agents, builds or copies; Defender exclusions active; high-performance power plan.
  - Record hashes of the exe, both DSV4 DLLs and kaira_wex_cuda.dll; pin COLI_DSV4_DLL.
  - Lossless settings: V4_SKIP_COLD=0 (kaira.json forces 1 = lossy), V4_TOPK unset, V4_MTP=0, temperature 0.
  - Prompts: lab\v41-gpu-20261107\ab\prompts.jsonl (warm-up + code1, code2, chat_it1, chat_it2, reason1, reason2, ctx2k, short1; 256 tokens; no system prompt). Fresh process per run, ABBA x2.
- **W1 design choices**: retrain the WNM2 instead of reusing V4's; use 2048:512 directly; `--min-free-gib 5` to protect E:.
- **Not to do**:
  - fused rANS-in-GEMV (needs ~2.07 TB/s decoded vs 102.8 GiB/s measured, ~19x short, 1.16x GDDR7 1.79 TB/s);
  - green contexts (contention is not on SMs); GDS on Windows GeForce; IoRing as a throughput lever; MPS on Windows;
  - HOBBIT low-precision substitution (not lossless); W-TinyLFU; DirectStorage/GDeflate (-8%).
  - Prefetch must be gated, not removed. The per-record WNR1 helper path should be retired in favour of the per-layer batch.
- **Literature verdicts** (report "Sfruttare WNR2 e WEX3"; do not redo this research):
  - ZipServ: fixed 3-bit code on BF16, not for MXFP4.
  - DFloat11, ECF8: gains come from memory or batch size, not decode speed.
  - ZipMoE: CPU decode for NVMe->RAM; abstract says 6.76x, body 42.49x.
  - nvCOMP Decompression Engine: only on B200/B300/GB200/GB300; Windows not confirmed.
  - arXiv 2606.15789 and snuri00/vram-compression: 2-3x slower than cuBLAS.
  - Recoil: >90 GB/s on a 2080 Ti (reference point, not a target).
  - ZipNN: claims FP4 E2M1 bits are near-uniform, which conflicts with our measured 3.67 bit/w.

## 2. Containers and files on disk

| Path | Content / model | Size | Date | Status |
|---|---|---|---|---|
| E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex | V4.1 REAP-192 WNR2 (see header below), model 0xab4261bd | 129,905,360,896 B (120.98 GiB) | 2026-11-07 02:42 | EXISTS, only copy (no U: copy) |
| D:\kaira_wex\v41-reap\v4_flash_experts.wex | V4.1 WEX1 original | 144,389,472,256 B | 2026-11-07 00:59:50 | exists; kept until the workflow ends; D: free 61.4 GB |
| U:\kaira_wex\v41-reap\v4_flash_experts.wex (+ .verified, 73 B) | byte copy of the V4.1 WEX1, sha256 4C055238D9011CE25D5C53A5EA61EC32BD8D1CF2886247194A13E7F53E7FA549 | 144,389,472,256 B | copied 02:23:12-02:34:10 2026-11-07 | exists; the reference container for REAP runs |
| E:\kaira_lab\wnr2-full\v4_flash_experts.nr2.wex | V4-0731 full WNR2, model 0x49435c70, sha256 c0c91294...2dd388, 43x256, aligned 13,369,344, records from 860,160, chunks 4096:2048 | 128,368,873,472 B | 2026-10-20 | DELETED (E: copy ~2026-11-05 with OK; U: copy by the user 2026-10-21); E:\kaira_lab is empty |
| (V4 WEX1) v4_flash_experts.wex | V4-0731 WEX1 | 147,170,537,472 B (137.06 GiB) | built 2026-10-17 | DELETED with the V4 weights (gone by 2026-11-05) |
| Q:\kaira-temp\U_kaira_build_wexgd-20261019 | 10 files: pilot-v1.wex (WEX1) 9,198,907,392; pilot-v3-neural512.wex (WNR1) 8,326,713,344; pilot-v3-nr2.wex (WNR2, 688 V4 experts, model 0x2e0ca289, sha256 e5e04723...) 8,031,371,264 (2026-10-20 00:31); pilot-v3-nr2z.wex (WNR2z, 688 V4 experts, model 0xd4fa45dc, sha256 2ef03ead...) 7,754,739,712 (2026-10-20 00:32); build.json, mirror.json, cdf-uniform-fallback.bin, verify_cpu.dll, wex_neural_encoder.dll, Oct-19 kaira_wex_cuda.dll (195,072 B). links.json and manifest.json are one level up, in Q:\kaira-temp | 33,312,321,406 B (33.3 GB) | moved from U:\kaira_build\wexgd-20261019 (nr2\temp-move.log: 10 files, nr2z included) | exists; U:\kaira_build empty; restore: `lab\wex-gpu-direct-20261019\nr2\temp_move.py restore Q:/kaira-temp` |
| U:\kaira_lab\wnr2-samples | records64_4096_2048.bin 1,600,768,792; records64_2048_512.bin 1,603,336,984 (64 verified V4 experts each: model + entries + raw + records); 16-expert rechunk inputs records_512 403,983,576 / _1024 401,755,352 / _1024_256 401,960,152 / _2048 400,641,240 / _2048_2048 400,641,240 / _2048_256 400,891,096 / _4096 400,112,856 | ~6.0 GB | 2026-11-05 | exists; V4 WNR2 samples (the other V4 WNR2 data left are the pilot WNR2/WNR2z containers in Q:\kaira-temp) |
| U:\kaira_lab (older) | wex-neural-lossless-20261018 (~171 MB: sample-deep-256/512/1024.wex 24,723,456 / 23,986,176 / 23,617,536; sample-scale-only-*; sample-v2.wex 25,415,680); wex-validation-20261019-155553 (215 MB); wex-weight-20261018 (~45 MB: sample-v2.wex, sample-v3-palette3.wex 19,128,320); wex3-integrated-20261019 and wex3-optimized-integrated-20261019 (empty); predictor_eval (374 MB, 25 files, 2026-10-13; sim_more.json) | ~0.8 GB | 2026-10-13..19 | exists |

- **Model blobs** (all 59,408 B):

| model | id | sha256 / notes | locations |
|---|---|---|---|
| V4.1 REAP | 0xab4261bd (2873254333) | f83fee82734219358d0f1290245298d44e2808d7abb48c9c0a742cd6dd855e3e; trained on all 7680 experts | embedded in the container; `lab\v41-gpu-20261107\w1\model-full-nr2.wnm2`; `<HOME>\Wex-WRN2\models\v41-reap192.wnm2` |
| V4 full | 0x49435c70 | 2ddd4220...2d49 | `lab\wex-gpu-direct-20261019\nr2\model-full-nr2.wnm2`; `Wex-WRN2\models\v4-flash-0731.wnm2` (2026-10-20 01:51) |
| pilot | 0x2e0ca289 | | model-nr2.wnm2 |
| pilot WNR2z | 0xd4fa45dc | | model-nr2z.wnm2 |

  Training counts are stored next to them: counts-nr2.npz, counts-full-nr2.npz (weight[3,567,16], scale[3,256]).
- **V4.1 WNR2 header**:
  - WEX3 v3, 40 layers x 192 experts, aligned_expert_bytes 18,800,640, flags 21;
  - WNMD at 64 (bytes 59,408, offset 4096, id 0xab4261bd); table_offset 65536; records start at 618,496;
  - uuid 01f11564c51f44599b9885b727ad2937; sha256 3f703fa4d993ed46d2873676c71be07cf11c11c7baef3e0bce6428c30921eb9d.
- **Geometry**:
  - V4.1: 40 layers x 192 experts = 7680, hidden 5120, moe_inter 2304, top-6. A record holds 3 matrices 5120x2304 FP4 = 17,694,720 B of weights + 1,105,920 B of scales = 18,800,640 B raw. At 2048:512 that is 19,440 streams (17,280 weight + 2,160 scale) = 38,880 B of lengths.
  - V4: 43x256 = 11,008 experts, 13,369,344 B (12.75 MiB) decoded per expert.
- **Drives** (M0 probe, 2026-11-07, GB/s rand 18.8 MB QD8):

| drive | model | link | GB/s | QD1 p50 | free space |
|---|---|---|---|---|---|
| D: | Sabrent SB-2130-1TB | PCIe 3.0 x2 (max x4 gen4) | 1.61 | 12.2 ms | 61.4 GB = 57.2 GiB |
| U: | SK hynix HFS001TEJ9X162N | gen4 x4 | 6.58 | 3.2 ms | 279 GB at 02:34; 122.6 GB = 114.2 GiB (other files on U: changed at 16:04) |
| C: | Crucial T700 | gen5 x4 | 10.66 | 2.3 ms | |
| E: | ADATA SX8200PNP | gen3 x4 | 2.29 | 8.3 ms | 135.2 GiB before W1, 15.3 GB after (w1.md); 105.9 GB = 98.6 GiB |

  - Free space checked 2026-11-07 17:11 (Get-PSDrive, decimal GB; `df -h` shows GiB, so E: "99G" / U: "115G" / D: "58G" are the SAME values, not a drop). Re-check before use. U: is ~7 GB short of the ~130 GB (121 GiB) a WNR2 copy needs for the W3 same-drive control.

  - E: also holds engram shard 48 (E:\models\DeepSeek-V4.1-Flash-1017-engram\model-00048-of-00048.safetensors, 101,537,926,640 B), so check space before writing to E:.
  - Each letter is its own physical disk.
  - The V4 checkpoint was moved U:\Models\DeepSeek-V4-Flash-0731 -> H:\models\DeepSeek-V4-Flash-0731 with a junction (02:00-02:23 2026-11-07, 152 files, 107 hashed). Engram shard 47 stays at C:\models\DeepSeek-V4.1-Flash-1017-engram.

## 3. Open work / next steps

- **G2** (prerequisite): route V4.1 decode MoE through moe_token_pipeline (VRAM mirrors, vram_first lanes, v4_np_predict prefetch, copy worker, promotion, async attach, WEX GPU-direct + RAM record cache).
  - The V4.1 router must avoid V4_ROUTER_AVX2 (is_v41 guard) and coli_v4_gpu_route; add dsv4_cuda_set_act_group (group 32).
  - Test: 256 greedy tokens with COLI_WEX_GPU_DIRECT=0 vs =2 + COLI_WEX_RAM_CACHE_GB=4 must be token-identical.
  - Expected VRAM tier: ~1,030-1,110 mirrors of 17.93 MiB (14-16% of 7,680). The backend reserve or mirror budget must also cover the helper's own thread_local lanes (~287 MiB per batch-calling thread on V4.1, outside backend accounting; see section 10, helper memory footprint).
- **W2** (NOT in src: grep finds V4_WEX_BATCH/decode_batch only in wex_decode.c/.h and the bridge). Batched WNR2 decode per layer behind env V4_WEX_BATCH=1 (default off; only with COLI_WEX_GPU_DIRECT>=1 and coli_wex_cuda_batch_available()).
  1. In moe_token_pipeline after routing (v4:9210-9229), collect WNR2 misses of coli_v4_gpu_expert_peek_routed that are not in host RAM (via coli_v4_expert_store_wex_entry, v4:13964).
  2. Split v4_gpu_mirror_fill_unlocked (v4:17963-18004) into reserve(N) (one mutex hold + one dsv4_cuda_stream_drain) and publish(i).
  3. Fetch records from the RAM cache or with N overlapped ReadFile into a ring of 8 pinned staging buffers (~132 MB).
  4. Make one batch call; publish status 0; on -1/1 release and fall back per record.
  - Budget the helper lanes (8 x 2 x 17.93 MiB per calling thread) in the VRAM reserve, and restore the caller's CUDA device after each helper call (the helper leaves it set to `device`).
  - The prefetch twin in v4_np_worker comes after F1.
  - Expected: -0.6 to -0.9 ms per decode, -35 to -90 ms/token at 60-110 decodes/token.
  - Tests: batch vs per-record byte-identical; V4_WEX_BATCH 0 vs 1 token-identical; counters for batches/token, records/batch, ms/batch.
- **W3** (not run; no ab/results.md or REPORT.md): WEX1 vs WNR2 A/B on V4.1 at equal RAM BYTES (slots x 18.8 MB + cache GB within 1%).
  - The arms differ only in COLI_WEX_PATH. Common settings: `--memory-gb 16`, COLI_WEX_GPU_DIRECT=2, COLI_WEX_RAM_CACHE_GB=4, DSV4_CUDA_VRAM_RESERVE_MB=4096, V4_COPY_WORKER=1, V4_PROMOTE_VRAM=2, V4_ASYNC_ATTACH=1, DSV4_CUDA_STREAM_PRIORITY=1, V4_WEX_BATCH=1, same F1 and V4_GPU_HEAD.
  - Add a 'wexprof' line next to moeprof (VRAM hits, RAM hits, NVMe reads/bytes, WNR2 demand/prefetch decodes, decode ms). Use +11.1% capacity, not +14%.
  - D: (1.61) vs E: (2.29 GB/s) differ by 42%, above the 10% threshold, so a same-drive control on U: is mandatory. It needs a ~130 GB WNR2 copy verified against sha256 3f703fa4... plus a .verified marker. U: had only 122.6 GB (114.2 GiB) free at 17:11 on 2026-11-07, ~7 GB short, so space must be freed first.
- **F1**: confidence-gated prefetch with a per-token byte budget (V4_NP_MIN_CONF, V4_NP_TOKEN_MB, conf in V4NpItem v4:7229, npprof gated=, histogram under V4_PREDICT_STATS=1). It exists as lab F1.patch (adds tests\test_v4_np_queue.c), is not in src and is not measured. Prefetch must stay.
- **Before any WNR2 deploy**: run test_decode_batch with the APP helper (app\colibri-optimized\kaira_wex_cuda.dll, sha a7977200...) on V4.1 records; only the W1 lab build (cceb4fb5...) has passed that check.
- **Cache policy candidates** (Report "WNR2 cache decode CPU e WDDM", next to F1):
  - Measured LRU hit rate 70% at 1530 vs IRM-static 45%, so "LRU beats LFU" is not proven. Replay offline: LRU with a frequency tie-break, S3-FIFO-style probation for prefetches, decay. FlashMoE claims +21% hit.
  - Proposed gating rule: depth-1 (top-8) promotes to VRAM only at high confidence; depth-2 (top-6) goes only to RAM probation; a per-token byte budget; a per-layer cap, more generous on deep layers.
- **WDDM candidates** (same report): cudaDeviceScheduleSpin on the decode thread and blocking sync on the copy thread; HAGS untested; cudaMemcpyBatchAsync cannot be captured in a CUDA graph; graphs must cope with expert addresses that change every token.
- **Other**:
  - H1: host must pass COLI_WEX_PATH for V4.1; the stale app\kaira.json COLI_WEX_PATH (top-level env and v4 profile).
  - Residual bug from upstream #2668: a failed mirror upload leaves the expert registered without data, so it always runs on CPU (~10-line fix after W2).
  - D1: second drive / D: link fix (hardware decision, needs OK; could make D: 3-4x faster).
  - WNR2z logprob check.
  - Event-pipelined batch per lane (target 0.22 ms/record); pacing of decode bursts; more threads per record (the kernel is under-occupied); CUDA Graphs per layer.
  - Per-token counters on the 32 GB machine (priority 1).
  - Compressed WNR2 RAM/VRAM tier with fixed slots + byte-capped LRU (fixed slots recommended).
  - A private backend upload stream (upload() waits 3x).
  - CPU SIMD decode only behind the 2 GB/s-on-3-cores gate.
  - A direct path for WEX1/V2 raw records (never started).
  - A load-model option in build_nr2_full.py; repack_v4_wex.py cannot build or validate WNR2.
  - The qwen38 WEX port is not in src.
  - Re-sync Wex-WRN2 after changes.

## 4. What WNR2 is

WNR2 is KAIRA's lossless entropy coding of MXFP4 MoE expert records (FP4 E2M1 weights + UE8M0 scales). It is entry codec 4 inside a WEX3 container (header magic WEX3, version 3, header flag NEURAL_RANS2 = 16). One shared probability model (WNM2, 59,408 B) is stored once per container, and every record names it by its FNV-1a id. Weights use a 567-context rANS per matrix and scales an order-0 rANS. All CDFs are 14-bit.
- Naming: the user and the private repo spell it "WRN2" (KairaCode/Wex-WRN2), but every code constant, file and doc says WNR2: record magic 'WNR2', model 'WNM2', descriptor 'WNMD'. WNR2z is the canonical-zero variant. WNR1 is the older per-record-model codec (codec 3).
- The name says "neural", but the WNM2 model is a count-based context table. The trainer counts symbols and quantizes them to CDFs. No neural net is involved (see section 14).
- Container family in 3 lines:
  - **WEX1**: raw aligned MXFP4 records (one per expert, 1 UE8M0 byte per 32 weights) read unbuffered (FILE_FLAG_NO_BUFFERING). No decode step.
  - **WEX2**: UE8M0 scales packed in 256-value tiles (u8 base, 0..8 bits, residuals LSB-first, a constant tile is 2 B; prefix `<III>` = magic 0x31534357 + decoded scale count + stream bytes; entry flags = (scale_pages<<8)|1). FP4 weights are unchanged and read zero-copy to slab + scale_bytes_raw - scale_pages*4096.
  - **WEX3**: Palette3 (codec 2, 3-bit, 8 medoid levels instead of 15 FP4 values; LOSSY; only with `--format-version 3 --weight-codec palette3 --max-weight-relative-error`; never used in production), WNR1/WNR2 lossless rANS, and cross-layer SHA-256 DEDUP (entries share file_offset).

## 5. WEX container format (wex_format.h)

- All fields are little-endian. WEX_ALIGN = 4096: the header page, the entry table and every record offset/size are multiples of 4096.
- Magics: WEX1 0x57455831 (version 1, flags must be 0); WEX2 0x57455832 (version 2, only SCALE_TILES); WEX3 0x57455833 (version 3, any of flags 1|2|4|8|16). WNR2 exists only in WEX3.
- **WexHeader**, 40 B packed inside a 4096-B page: `{u32 magic, u32 version, u32 num_layers, u32 num_experts, u64 aligned_expert_bytes (max DECODED 4096-aligned slab), u64 table_offset, u32 flags, u32 reserved=0}`.
  - V2/V3 put a 16-B random build UUID at bytes 40..55. It matches primary and mirror copies and is not a checksum.
  - In WNR2 containers the WNMD descriptor sits at byte 64.
- **Header flags**: SCALE_TILES 1, PALETTE3 2, DEDUP 4, NEURAL_RANS 8 (WNR1), NEURAL_RANS2 16 ("WNR2 records + a WNM2 model before the table").
- **WexEntry**, 72 B: `{u64 file_offset, u32 total_bytes (aligned ON-DISK size), u32 scale_bytes, u32 weight_bytes (DECODED layout), u16 scale_shapes[3][2], u16 weight_shapes[3][2], u32 scale_offsets[3], u32 weight_offsets[3], u32 flags}`.
  - flags: codec in bits 0-7 (CODEC_MASK 0xff), scale_pages in bits 8-31.
  - An all-zero entry means absent/dense, and the engine uses safetensors for that expert.
  - Static asserts enforce 40/72 B.
- **Entry codecs**: 0 raw, 1 SCALE_TILES, 2 PALETTE3 (lossy), 3 NEURAL_RANS (WNR1), 4 NEURAL_RANS2 (WNR2).
- **Matrix order**: m=0 gate w1, m=1 down w2, m=2 up w3. The decoded slab holds the scales of w1,w2,w3, then the packed FP4 weights of w1,w2,w3. wex_entry_raw_sizes checks scale cols*16 == weight cols and that offsets are contiguous.
- **Record placement**: for codecs 2/3/4, `wex_record_read_offset = aligned_raw - total_bytes`, so the compressed record is read into the END of the decoded slab.
- **wex_validate_entry**, codec 4: requires header version 3 with flag 16, and total_bytes 4096-aligned, >= align(32 + 6*streams at max chunk 16384) and < aligned_raw. A record that is not smaller than raw must be stored raw (flags 0).
- **wex_validate_table** (v3): physical records contiguous; aliases only with DEDUP and byte-identical 72-B entries; the last record ends at EOF; max aligned raw == aligned_expert_bytes.
- **Layout written by the WNR2 builder**:
  - header page (WEX3 v3, uuid, WNMD at 64);
  - WNM2 model at offset 4096, zero-padded to 65536 (= table_offset = 4096 + align(59408) = 4096 + 61440);
  - 4096-aligned entry table;
  - records contiguous to EOF.
  - Header flags written: 21 (SCALE_TILES|DEDUP|NEURAL_RANS2).

## 6. WNR2 record and model descriptor

- **Descriptor 'WNMD'** (WEX_NR2_DESCRIPTOR_MAGIC 0x444d4e57) at header byte 64 (WEX_NR2_DESCRIPTOR_OFFSET): `{u32 magic, u32 bytes (must equal WEX_NR2_MODEL_BYTES 59408), u64 offset, u32 model id (non-zero), u32 0}`.
  - The model lies in [offset, offset+bytes); the offset is 4096-aligned, >= 4096, and sits between the header page and the table.
  - wex_nr2_model_location requires magic WEX3, version 3 and flag 16.
- **Record**: a 32-B header `{u32 magic 'WNR2' 0x32524e57, u32 model_id, u32 scale_payload_bytes, u32 weight_payload_bytes, u32 weight_chunk, u32 scale_chunk, u8 median[3] (bytes 24..26), u8 0 (27), u32 0 (28)}`. The header is followed by:
  - the u16 lengths of every scale stream, then of every weight stream;
  - the scale payload;
  - the weight payload;
  - zero padding to 4096 (canonical, checked).
- **Streams** never cross a matrix: matrix m has ceil(count_m/chunk) streams.
  - Each stream is a 4-B big-endian initial state in [L, 256*L) followed by byte-renormalization bytes (state = state<<8 | byte while state < L).
  - It must end exactly at its length with final state == WEX_NR_L = 1<<23.
  - Stream length is >= 4 and <= 65535 (u16).
- **Chunks**: both are powers of 2. Weight chunk 512..16384 (WEX_NR2_MIN_WEIGHT_CHUNK..WEX_NR2_MAX_CHUNK), scale chunk 256..16384 (WEX_NR2_MIN_SCALE_CHUNK..MAX).
  - Chunks are stored per record, so any legal value works without a format change: V4 used 4096:2048, V4.1 uses 2048:512.
  - The encoder validates chunks at encode_nr2.cu:60-62. The decoders read them per record at wex_neural_rans.h:229-232 and wex_cuda.cuh:404-405.
- Using u16 lengths instead of u32 offsets and long streams cut stream overhead from 2.94% (WNR1) to 0.1%.
- **Integrity model**: WNR2 records carry NO checksum or CRC. Corruption is caught only by structural rANS invariants (wex_nr2_parse, wex_neural_rans.h:213-262; wex_cuda.cuh:280-283):
  - initial state in [L, 256L); each stream consumes exactly its u16 length with final state == L;
  - length sums equal the payload sizes; record size == align4096(used bytes); zero padding.
  - That catches almost everything (300 fuzz cases, 0 accepted) but guarantees nothing. Whole-file integrity rests only on the sha256 values and the .verified markers; the only extra runtime check is COLI_WEX_RAM_CACHE_VERIFY=1 (64-bit checksum on RAM-cache slots).

## 7. Codec and WNM2 model (wex_neural_rans.h)

- **Weight context**:
  - Symbols are row-major 4-bit FP4 codes; symbol 2k is in the low nibble.
  - For each 32-weight block with UE8M0 scale s, at block start: srel = clamp(s - median_m, -3, 3) + 3 (7 values), previous magnitude pm = 8 (none), block max mx = 8 (none).
  - After symbol x: pm = x&7, mx = max(mx, x&7).
  - context = (srel*9 + pm)*9 + mx, giving WEX_NR2_CONTEXTS = 567 contexts per matrix (7x9x9), each a 16-symbol CDF.
  - median_m is the lower median of the matrix's scales (wex_nr2_median), stored per record.
  - Contexts depend only on FP4 nibble and E8M0 statistics; geometry comes from each WexEntry.
- **Scales**: order-0 rANS, one 256-symbol CDF per matrix.
- **CDFs**: integers, precision WEX_NR2_PRECISION = 14 (total 16384), every symbol frequency >= 1.
- **WNM2 blob** = WEX_NR2_MODEL_BYTES 59,408:
  - 32-B header `{u32 'WNM2' 0x324d4e57, version 1, precision 14, contexts 567, matrices 3, 0, 0, 0}`;
  - weight CDFs 3*567*17 u16 = 57,834 B (WEX_NR2_WCDF_BYTES);
  - scale CDFs 3*257 u16 = 1,542 B (WEX_NR2_SCDF_BYTES).
  - wex_nr2_model_open validates the header and every CDF.
- **Model id**: FNV-1a 32-bit over the whole blob (offset 2166136261, prime 16777619; 0 is mapped to 1). Every record repeats the id.
- **Training (deterministic)**:
  - wex_nr2_count accumulates counts[3][567][16] for weights and counts[3][256] for scales, using exactly the decode contexts.
  - wex_nr2_quantize gives each symbol 1 + floor(count*(16384-symbols)/total) and hands the remaining slots to the largest remainders (ties go to the lower symbol). Empty counts give a near-uniform CDF.
  - wex_nr2_model_build writes the blob.
- **Reference encoder wex_nr2_encode**:
  - A forward pass computes the contexts, then each stream is encoded backwards (wex_nr2_put, x_max = ((L>>14)<<8)*freq) and 4 state bytes are flushed.
  - Order: scales m0..m2, then weights m0..m2.
  - Scratch per stream is bounded by 2*WEX_NR2_MAX_CHUNK+8.
  - CPU reference functions: wex_nr2_encode/put/flush, wex_nr2_decode/decode_symbol/stream_open, wex_nr2_count, wex_nr2_quantize, wex_nr2_model_build, wex_nr2_median (wex_neural_rans.h:264-517; decode_symbol 273, stream_open 291, decode 302-350, median 355, encode 450-517).
- **Why WNR2 (2026-10-20 design)**:
  - WNR1 byte budget (687 experts, % of raw): weights 3.681 bit/weight = 86.6%; rANS flush 4 B per 512 symbols 1.47%; u32 offsets 1.47%; scale tiles (1.24 bit/scale) 0.91%; per-record CDF + header + padding 0.05%. So 2.94% was parallelism overhead.
  - Held-out cross-entropy (shared model): scale+previous (WNR1) 3.684 bit/w; + block max 3.665; + position 3.660. Rows, columns, the upper neighbour and gate<->up gave nothing. Entropy-coded scales reach 0.97 bit.
- **WNR2z** (`--canonical-zero`, wex_nr2_canonical_zero):
  - FP4 code 8 (-0) is mapped to 0 (+0) before counting and encoding. Contexts are unchanged (8&7 == 0&7).
  - Value-exact but NOT byte-exact. +/-0 is 12.8% of FP4 weights and carries 0.128 bit/weight of no information.
  - Needs a logprob comparison before production; tokens were identical in 4 runs only.
- **WNR1, for contrast** (codec 3, header flag 8, magic 'WNR1' 0x31524e57, L = 2^23):
  - Per-record model: a 256-entry u16 scale->class map plus up to 256 classes x 17 contexts x 17 u16 CDFs (12-bit total 4096; WEX_NR_CDF_BYTES 17*17*2).
  - Context = previous symbol, reset every 32 weights. Chunks of 256/512/1024 symbols with u32 stream offsets; scales as a tile stream.
  - Each WNR1 record stores its own chunk (256, 512 or 1024) at byte 20, read by wex_nr_parse (wex_neural_rans.h:34-35). The pilot used 512 (pilot-v3-neural512). WEX_NR_CHUNK_SYMBOLS 1024 (wex_neural_rans.h:12) is only the largest chunk, used by wex_validate_entry (wex_format.h:225) for the minimum-size bound; no contradiction.
  - Its GPU kernel wex_cuda_neural_rans_weights uses `__shared__ unsigned char input_cache[32768]`.
  - The pilot uniform fallback CDF is cdf-uniform-fallback.bin.
- **Generic codecs measured** (16 decoded MXFP4 slabs, 213.9 MB; ratio whole / FP4 weights / UE8M0 scales):

| codec | whole | weights | scales |
|---|---|---|---|
| zstd-3 | 0.9225 | 0.968 | 0.189 |
| zstd-19 | 0.917 | 0.965 | 0.135 |
| deflate-6 (GDeflate/DirectStorage family) | 0.920 | 0.968 | 0.158 |
| lzma-6 | 0.920 | 0.969 | 0.132 |
| WNR2 | 0.866 | 0.917 (3.67 bit) | 1.02 bit/scale |

  - Order-0 entropy: FP4 nibbles 3.866 bits (an order-0 coder saves at most -3.4% on weights); scales 0.98 bits.
  - The 567-context model gives 2.5x the weight saving of order-0 or generic codecs. GDeflate reaches only ~-8% vs -13.4% for WNR2, so DirectStorage is not worth it.

## 8. Encoder and build procedure

- **Lab home**: `lab\wex-gpu-direct-20261019\nr2\`: encode_nr2.cu, build_nr2.py (pilot), build_nr2_full.py, nr2lib.py, check_container.c, check_all.py, test_nr2_cuda.cu, test_nr2_tools.py, build_tools.cmd, build_test_nr2.cmd, run_test_nr2.cmd, dump_exports.cmd, temp_move.py (lab only), src-before-nr2\ (pre-WNR2 source snapshot, the rollback reference).
  - The V4.1 helpers sample_records.py and precheck_records.py are NOT in lab nr2: they are in `lab\v41-gpu-20261107\w1\` (build script build_helper.cmd).
  - `<HOME>\Wex-WRN2\nr2` has the nr2 sources and cmd scripts (no temp_move.py) plus the V4.1 helpers, where the build script is named build_helper_v41.cmd.
- **wex_nr2_tools.dll**, built from encode_nr2.cu. It is lab only and not linked into the engine.
  - ABI: wex_nr2_tools_abi = 1.
  - Exports: wex_nr2_count_gpu, wex_nr2_build_model, wex_nr2_model_id, wex_nr2_encode_gpu, wex_nr2_verify, wex_nr2_verify_slab, wex_nr2_check_container, wex_nr2_count_cpu, wex_nr2_encode_cpu, wex_nr2_canonical_zero, wex_nr2_release.
  - Return codes: 0 ok, -1 invalid, 1 CUDA failed. One process-wide workspace, serialized by a mutex.
  - GPU output is byte-identical to the CPU reference. Geometry comes from each WexEntry (encode_nr2.cu:44-79).
- **Build commands**:
  - wex_nr2_tools.dll: `nr2\build_tools.cmd` (nvcc CUDA v13.4, `-O3 -std=c++17 -arch=sm_120`, VS 18 BuildTools vcvars64, `-I src/colibri/c`).
  - Helper: `lab\wex-gpu-direct-20261019\build_helper.cmd` (`nvcc -O3 -arch=sm_120 -Xcompiler /O2,/EHsc,/arch:AVX2 -shared tools/wex_cuda_bridge.cu`).
  - Test: `nr2\build_test_nr2.cmd` builds test_nr2_cuda.exe.
- **nr2lib.py**: ctypes bindings (DLL_PATH = HERE/'wex_nr2_tools.dll') plus wire structs: HEADER `'<IIIIQQII'` (40 B), ENTRY `'<QIII6H6H3I3II'` (72 B), MODEL_BYTES 59408, WNR2_MAGIC 0x32524e57, WNMD_MAGIC 0x444d4e57, WEIGHT_BINS 3*567*16, SCALE_BINS 3*256, fnv1a().
- **check_container.c**: an independent MSYS2-gcc checker. Usage: `check_container <wnr2.wex> <source-v1.wex> <canonical_zero 0|1> [stride]`.
- **build_nr2_full.py procedure**:
  1. The source must be a raw WEX1 container (magic V1, version 1, flags 0, absent entries all-zero).
  2. Pass 1: GPU training counts over every present expert build the WNM2. It ALWAYS retrains; there is no load-model option.
  3. Pass 2: each expert is GPU-encoded, padded to 4096 and verified with the CPU reference (wex_nr2_verify) in a thread pool before it is written. A record that is not smaller than raw is stored raw (flags 0). Every 43rd expert is also CPU-encoded and byte-compared.
  4. Final checks: wex_nr2_check_container, a slab-path readback of every 43rd expert plus the last and all fallbacks, a whole-file sha256 and a per-record rehash.
  5. Output goes to `<out>.partial`, then os.replace.
- **build_nr2_full.py options**:
  - `--source`, `--out` (both required);
  - `--weight-chunk` (default 4096), `--scale-chunk` (default 2048);
  - `--canonical-zero`;
  - `--verify-workers` (4), `--crosscheck-every` (43);
  - `--overwrite`;
  - `--min-free-gib` (2.0; aborts and deletes the partial file below it);
  - `--report-dir` (default: the nr2 folder; added 2026-11-07 so V4.1 reports do not overwrite the V4 build-full-nr2.json / model-full-nr2.wnm2 / counts-full-nr2.npz).
  - Up-front free-space check: 0.90 x source size + margin. Outputs: build-<suffix>.json, model-<suffix>.wnm2, counts-<suffix>.npz.
- **tools\repack_v4_wex.py does NOT write WNR2.**
  - `--weight-codec` accepts native|palette3|neural-rans. neural-rans writes WNR1 (codec 3) and needs `--neural-cdf`, `--neural-encoder-dll` and `--neural-chunk-symbols` 256/512/1024 (default 512).
  - Its validate_container allows v3 header flags only up to 15, so it would reject a WNR2 container (flag 16). WNR2 only comes from the lab nr2 builder.
  - It writes a unique .tmp file and then os.replace.
- **Tool timings** (test_nr2_tools, 2026-10-20, V4 records):

| step | GPU | CPU |
|---|---|---|
| count | 1.51 ms/expert warm (median 1.55) | 48.0 ms |
| encode | 4.82 ms warm (median 8.84) | 148.1 ms |
| CPU verify (reference decode) | | median 317 ms |

  All checks passed: real, synthetic, canonical-zero, own-model, uniform-model, adversarial.
- **Record size vs chunk** (weight/scale, average of 3 real V4 experts): 512/256 11,914,581 B; 1024/512 11,777,365; 2048/16384 11,702,272; 4096/2048 11,664,043; 8192/4096 11,650,389; 16384/16384 11,636,053.
  - With a uniform model a record is 13,410,304 B, larger than raw, so it would fall back to raw.
  - Adversarial input: record 44,609,536 B, max stream 28,676 B, within the bound 2n+8 = 32,776.
- **Pilot build** (build_nr2.py, 688 experts): training 3.3 s, encode 57 s.
- **V4 full build** (2026-10-20): 779 s total for 147 GB (training 53.7 s, encode 453.7 s, validation 272 s; "13 minutes"). 11,008/11,008 experts verified on CPU during the build and again on GPU (check_full.json).
- **V4.1 W1 build** (2026-11-07), exact command:
  `python -I lab\wex-gpu-direct-20261019\nr2\build_nr2_full.py --source U:\kaira_wex\v41-reap\v4_flash_experts.wex --out E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex --weight-chunk 2048 --scale-chunk 512 --verify-workers 12 --min-free-gib 5 --report-dir lab\v41-gpu-20261107\w1`
  - Ran 02:34:57-02:47:01 = 723.2 s under heavy.lock (taken 02:34:33, released 02:47:55). Exit 0.
  - Phases: training 62.05 s (GPU count calls 22.39 s); encode with CPU verify on 12 workers 373.35 s (GPU encode calls 100.33 s); readback + whole-file hash 287.81 s.
  - Build verification:
    - all 7680 records exact under the CPU reference wex_nr2_verify;
    - 179 GPU records byte-equal to the CPU encoder (every 43rd);
    - wex_validate_header + wex_validate_table + wex_nr2_model_location + wex_nr2_model_open OK, and the id matches the descriptor;
    - 180 slab-path readbacks;
    - all 7680 on-disk records rehashed equal;
    - 7680/7680 records are WNR2, 0 raw fallbacks, 0 negative-zero codes mapped (canonical_zero false).
  - 64-record GPU check:
    - w1\sample_records.py (seed 20261107) pairs each record from E: with the same expert's slab from the original WEX1 on D:, asserting geometry and model id.
    - test_decode_batch.exe (source in lab\wex-prod-20261105) ran with a kaira_wex_cuda.dll built from clean src at 79aff2b (wex_cuda_bridge.cu sha a346652f..., wex_cuda.cuh b6a6563f..., wex_neural_rans.h d875e1d1...).
    - Batch sizes 1,2,3,4,6,8,10,16 = 512 decodes, all byte-equal; ALL OK, rc=0.
    - Corruption statuses [0,0,-1,0,-1,0]: a damaged payload and an invalid entry return -1 with the destination untouched.
    - Scratch records64.bin is 2,284,530,456 B.
  - Retraining paid off: in the pre-check, 8 V4.1 experts encoded with the old V4 model came out at 0.941-0.950 of raw, vs 0.8997 with the V4.1 model. Any embedded model is lossless, so a V4 model would have been valid, just larger.
  - W1 files in lab\v41-gpu-20261107\w1:
    - build-v41-nr2.log (15,428 B);
    - build-full-nr2.json (923,467 B, per-record rows);
    - model-full-nr2.wnm2 (59,408);
    - counts-full-nr2.npz (37,633);
    - verify64.log, sample_records.py;
    - precheck_records.py (its single-call timing loop needs >= 16 records and failed on 8 by design);
    - build_helper.cmd and take_lock.ps1;
    - build\kaira_wex_cuda.dll (387,072 B) + .lib/.exp, and build\test_decode_batch.exe (208,384 B).
    - build_helper.log shows a harmless "vswhere.exe non riconosciuto" warning.
- **Re-run the GPU check**:
  1. `python -I lab\v41-gpu-20261107\w1\sample_records.py NR2.wex WEX1.wex OUT.bin [count=64] [seed=20261107]`
  2. `lab\v41-gpu-20261107\w1\build\test_decode_batch.exe HELPER.dll OUT.bin` (source lab\wex-prod-20261105\test_decode_batch.cu; helper and test built by w1\build_helper.cmd, nvcc 13.4 -arch=sm_120).
  - Bin layout: u32 model_bytes + WNM2 blob + u32 count + per record (72-B entry, u32 raw_bytes, raw slab, record of entry.total_bytes).
  - records64.bin (2.28 GB) was written to a temporary folder (`<TMP>`), so it is ephemeral and probably gone; regenerate it.
  - "failed:2" in verify64.log is the two deliberate corruptions, not a failure.

## 9. CPU decoder

- **wex_decode_nr2_record**(entry, slab, capacity, ..., model) in wex_format.h:480-507:
  - validates sizes, the read offset, and wex_nr2_parse with the model id;
  - decodes into malloc scratch and memcpy's to the slab only on success.
  - The generic wex_decode_record refuses codec 4 (returns -1, "needs its model").
- **wex_nr2_decode** decodes scales stream by stream (binary search in the 256-symbol CDF), then weights with the exact context rule.
- **Speed**: ~0.29-0.32 s per V4 expert (288-317 ms median), the same order as WNR1 (~0.3 s), about 44 MB/s.
  - Keeping up with a 6.6 GiB/s disk would need ~7.6 GiB/s of output (~11-12 of 16 cores at ~0.7 GB/s/core). Published SIMD rANS runs 0.5-1.4 GB/s/core; Recoil reaches 11+ GB/s on a 16-core Xeon W-3245.
  - With 13 inference threads, ~3 cores would be left and would need ~2.7 GB/s/core, which is not reachable. No rANS measurement exists on the 7950X3D.
  - CPU decode also loses the PCIe saving, because the raw 12.75 MiB slab goes to VRAM.
- **Table sizes**: a CDF-only table is 567 x 32 B = 18 KB (fits L2). A 16384-entry slot->symbol LUT per context would be 9.3 MB (does not fit L1/L2).
- **Gate**: CPU decode is worth doing only if the container stays on a single ~1.8 GiB/s drive AND a SIMD 567-context rANS microbench exceeds 2 GB/s on 3 cores; otherwise discard it.
- **Model registry** (wex_decode.c:97-154), coli_wex_register_model(blob, bytes, &id):
  - validates with wex_nr2_model_open and keeps a private copy; at most WEX_NR2_MAX_MODELS = 16 per process, never freed;
  - the same id with different bytes returns -1 (hash collision); re-registering an identical model is a no-op;
  - on Windows it also calls the helper's wex_cuda_register_model and records `in_helper`.
- **coli_wex_decode_record** (host-slab path, wex_decode.c:257-285):
  - For codec 4 it reads the model id at record bytes 4..7; an unknown id returns -1.
  - For codec 3, or codec 4 with the model registered in the helper, it tries the helper's wex_cuda_decode_to_host first. It prints once "[wex] CUDA reconstruction failed/unavailable (%d); using CPU fallback." and then falls back to the CPU decoder.
- coli_wex_decode_record was moved from expert_store_wex.c to wex_decode.c on 2026-10-18 23:39 so test_deepseek_v4 links.
  - Makefile.deepseek-v4 adds wex_decode.o to V4_WIN_EXTRA_OBJS; the Makefile adds it to V4_TEST_LINK_OBJS, test_expert_store_wex and build/segment.

## 10. GPU helper kaira_wex_cuda.dll

- **Source**: src\colibri\c\tools\wex_cuda_bridge.cu, kernels in wex_cuda.cuh. ABI 1 (wex_cuda_helper_abi returns 1; provides wex_cuda_decode_to_host).
- **Loading** (wex_decode.c:13-94, Windows only, lazy pthread_once):
  - COLI_WEX_CUDA_LIB gives an absolute helper path; "0" forces the CPU.
  - Default: kaira_wex_cuda.dll next to the exe, loaded with LoadLibraryExA (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|DEFAULT_DIRS).
  - Requires abi == 1 and decode_to_host. Optional symbols are resolved by feature bits.
  - Logs "[wex] Neural CUDA reconstruction enabled: <path>" or "[wex] Neural CUDA helper unavailable; using the CPU decoder." The DLL is kept until exit.
  - Non-Windows builds report every CUDA path as unavailable.
- **Feature bits** (wex_cuda_helper_features, bridge :290-296):

| bit | name | enables |
|---|---|---|
| 1 | DEVICE | wex_cuda_decode_to_device |
| 2 | REGISTER | wex_cuda_host_register / _unregister (cudaHostRegisterPortable) |
| 4 | NR2 | wex_cuda_register_model; decode_to_host and decode_to_device accept codec 4 |
| 8 | BATCH | wex_cuda_decode_batch_to_device |

  - Current src returns 15. The 2026-10-20 helper (sha cac5348d...) reported 7, with no batch support.
- **Exported API**:
  - `unsigned wex_cuda_helper_abi(void)`; `unsigned wex_cuda_helper_features(void)`;
  - `int wex_cuda_decode_to_host(const WexEntry*, unsigned char *slab, size_t cap)`;
  - `int wex_cuda_decode_to_device(const WexEntry*, const unsigned char *record, int device, unsigned char *const *dst_scale, unsigned char *const *dst_weight)`;
  - `int wex_cuda_register_model(const unsigned char *blob, size_t bytes)`: 0 ok/already; -1 invalid, or another blob with the same id; 1 table of 16 full or no memory;
  - `int wex_cuda_decode_batch_to_device(int n, const WexEntry *const*, const unsigned char *const*, int device, unsigned char *const *dst_scale, unsigned char *const *dst_weight, int *status)` (bridge :409);
  - `int wex_cuda_host_register(void*, size_t)`; `int wex_cuda_host_unregister(void*)`.
- **Engine wrappers** (wex_decode.h:24-57):
  - coli_wex_decode_record; coli_wex_cuda_decode_count; coli_wex_cuda_device_available (feature 1);
  - coli_wex_cuda_decode_to_device: 0 ok, 1 unavailable, -1 invalid; codec 4 returns 1 unless feature 4 is present;
  - coli_wex_cuda_host_register/unregister (1 when unavailable); coli_wex_cuda_device_decode_count; coli_wex_register_model;
  - coli_wex_cuda_nr2_available (feature 4); coli_wex_cuda_batch_available (features 8 and 4);
  - coli_wex_cuda_decode_batch_to_device (wex_decode.c:198-219): returns the count of non-zero statuses; without feature 8 every status is 1 and the caller falls back per record.
- **decode_to_device semantics**:
  - It validates, uploads, decodes into private scratch, checks the error word, commits the 6 regions, then synchronizes.
  - An invalid record returns -1 and leaves the destination untouched. Scratch follows the current device.
  - Backend and helper runtimes share the CUDA primary context, so device pointers pass between them (tested).
  - A WNR2 record without a registered model returns 1.
  - Every decode entry point calls cudaSetDevice(device) (bridge :307, :346, :440) and never restores the caller's previous device, so the calling thread is left on `device`. Harmless on the single 5090 today; a trap for W2 integration.
- **Helper memory footprint** (wex_cuda_bridge.cu:23-90, 384-399, 409-500): workspaces are thread_local, allocated with cudaMalloc OUTSIDE the backend's VRAM accounting, and freed only when the thread exits.
  - Batch path: up to 8 lanes per calling thread, each holding device_record + device_raw at the largest aligned record. On V4.1: 8 x 2 x 17.93 MiB = ~287 MiB per thread (~16 expert mirrors).
  - Single-record path: another 2 x aligned of VRAM per thread, plus pinned host_raw on the host path.
  - Each ensure() requires free >= 2*bytes + tiles*4 + lut + 8 MiB.
  - The G2/W2 mirror budget (~1,030-1,110) or DSV4_CUDA_VRAM_RESERVE_MB must cover the helper lanes (WDDM over-commit risk).
- **Helper model registry**: process-global, WEX_CUDA_NR2_MAX_MODELS = 16, WEX_CUDA_NR2_MAX_DEVICES = 64. Device tables are built lazily per (model, device) and kept for the process lifetime.
  - Layout (one cudaMalloc): blob (align256 = 59,648 B) + weight rows [3*567] x 32 B + scale_direct [3][16384] u32 (slot -> (freq-1) | (slot-start)<<14) + scale_symbol [3][16384] B.
  - Weight buckets [3*567][1<<10] u32 (~6.97 MB) exist only for lookup 0. With lookup 1 the tables total ~360 KB (359,936 B).
- **CUDA decoder** (bit-identical to wex_nr2_decode):
  - Host side: wex_cuda_plan_nr2 validates with wex_nr2_parse; wex_cuda_nr2_offsets prefix-sums the u16 lengths into u32 offsets (streams+1).
  - **Scale kernel** wex_cuda_nr2_scales: one thread per stream, 32 threads/block, direct tables in dynamic shared memory. WEX_CUDA_NR2_SCALE_SMEM = 16384*5 + ring 8192 = 90,112 B ("88 KB", opt-in via cudaFuncSetAttribute).
  - **Weight kernel** wex_cuda_nr2_weights<Lookup>: one thread per stream, the matrix's 567 CDF rows in shared memory, branch-free 16-way compare (PRMT sign trick), one uint4 store per 32 symbols. Each block serves one matrix.
  - **Payload ring**: a 64-word (WEX_CUDA_NR2_RING_WORDS) shared ring per lane, refilled by cp.async at warp-uniform points, with WEX_CUDA_NR2_RING_AHEAD = 40 words ahead. Renormalization is branch-free (0/1/2 bytes per symbol). Compared with register prefetch, the scale kernel went 193 -> 75 us and the weight kernel 300 -> 253 us per expert.
  - **Errors**: any violation (bad initial state, read past end, early end, final state != L) sets *error.
  - **WEX_CUDA_NR2_LOOKUP** (default 1): 1 = shared CDF rows + 16-way compare (~253 us/expert at chunk 4096); 0 = global coarse buckets (WEX_CUDA_NR2_BUCKET_BITS 10) + exact walk (~925 us; the 7 MB table misses L1).
  - **Model table kernels**: wex_cuda_nr2_build_rows / _build_scales / _build_buckets.
  - **Commit** wex_cuda_commit_regions (grid 64x6, 256 threads): copies the 3 scale + 3 weight regions into the caller's buffers ONLY if *error == 0, on the same stream, so the destination never holds partial or corrupt data. In the raw buffer, weights start at align16(scale_bytes).
  - **Occupancy**: a single 11 MiB V4 record launches only 12 scale blocks + 192 weight blocks of 32 threads (about 1 warp/SM on 170 SMs). It is latency-bound at ~200 cycles/symbol/lane.
- **Single-record pipeline** (wex_cuda_nr2_decode): a thread-local workspace with two non-blocking streams.
  - `stream` uploads the offsets, then the record head (header + lengths + scale payload) and records event `head`, then the weight payload and event `data`.
  - `aux` waits on `head`, memsets the error word and runs the scale kernel; it waits on `data` and runs the weight kernel, then the commit (device path) or a D2H copy to pinned host_raw (host path), then the error D2H and a sync.
  - Code comment: on WDDM each copy-engine/kernel switch costs ~15-35 us.
  - ensure() needs free VRAM >= 2*bytes + 4*tiles + lut + 8 MiB.
  - Cost: ~14 CUDA calls and 2 syncs per decode.
- **October bridge optimization** (2026-10-19):
  - The record goes straight from the caller's slab to CUDA: no second pinned host buffer (saves 12.75 MiB per thread on V4) and no duplicate validation.
  - Output stays in private scratch until kernels, return copy and error check finish.
  - Scripts: lab\wex_neural\run_bridge_optimization.py, verify_bridge_optimization.py, validate_integrated_optimization.py.
  - Direct-path cost was estimated from code, not measured: ~2 x 12.75 MiB VRAM workspace per decoding thread plus a LUT, and two page-locked 12.75 MiB RAM buffers.
- **Batch API** (feature 8; committed in 9a631c8):
  - Spreads n records round-robin over up to WEX_CUDA_BATCH_LANES = 8 thread-local lanes. Each lane has its own copy `stream` and kernel `aux` stream plus events head/data/consumed (`consumed` orders a lane's next record after its previous kernels).
  - Offsets and error words live in page-locked scratch (WexCudaBatchHost). Workspaces grow per call. ONE host wait per batch.
  - Destinations: record i, matrix m -> dst_scale[3*i+m] / dst_weight[3*i+m].
  - status[i]: 0 written; -1 invalid record or destination (untouched); 1 CUDA/resources unavailable or model not registered (data correct or untouched, never partial). If queueing fails at record i, i and every later record report 1.
- **Deployed helper**: `app\colibri-optimized\kaira_wex_cuda.dll`, 387,072 B, dated 2026-11-07 00:47:44, sha256 prefix a79772004426c38f. It was built in ws-int for the V4.1 install. Its binary contains the export names wex_cuda_register_model and wex_cuda_decode_batch_to_device, so features 4 and 8 are installed.
  - Full sha256: deployed a79772004426c38fe2ed1196d27cb4e17a52512c0368d5bb71b86b2e3286b049. The helper that passed the V4.1 64-record check is a DIFFERENT binary: `lab\v41-gpu-20261107\w1\build\kaira_wex_cuda.dll`, 387,072 B, built 02:04 from clean 79aff2b, sha256 cceb4fb573b3b3e7b71843bf85a2b87cf9efdd94155bfc5deec717cce949e782.
  - Same size, different hash (a PE timestamp alone could explain it), but the deployed helper has never been run against V4.1 WNR2 records.

## 11. Engine integration (deepseek_v4.c only)

WEX/WNR2 integration exists only in deepseek_v4.c and the standalone store expert_store_wex.c. qwen38_core.h has no WEX code, and deepseek_v41.c is CPU/Vulkan only.

- **Container open** (v4:13255-13354):
  - Path: COLI_WEX_PATH, default `<model_dir>\v4_flash_experts.wex`. Opened with FILE_FLAG_NO_BUFFERING|OVERLAPPED|SEQUENTIAL_SCAN.
  - v4_wex_load_manifest validates header and table.
  - Logs "[v4_wex] Container v%u active: <path> (L layers, E experts, N decoded bytes/expert, direct_io=on)" or "[v4_wex] Invalid or incompatible container: %s; using safetensors".
- **WNR2 guard** (v4:13126-13183, added 2026-11-05):
  - For header flag 16 the engine reads the WNM2 at the descriptor, registers it and requires registered id == descriptor id.
  - If !coli_wex_cuda_nr2_available() it prints "[v4_wex] WNR2 container needs kaira_wex_cuda.dll with WNR2 support (helper feature 4) next to the engine" and REFUSES the container, so the engine stays on safetensors.
  - Before 2026-11-05 it silently fell back to the CPU decoder at ~0.3 s/expert.
  - The standalone expert_store_wex.c has NO such guard: it registers the model and can fall back to the CPU.
- **Mirrors**:
  - COLI_MODEL_MIRROR (';' or ',' list) plus COLI_WEX_EXTRA_MIRRORS (dirs holding only the container) give up to 3 WEX replicas (v4:13289-13352).
  - A replica is accepted only if size, the 4096-B header page and the table are byte-identical ("Matching mirror container active" / "Ignoring mismatched mirror"). WNR2 replicas re-register the same model, which is a no-op.
  - Read split: with 1 mirror, the dual-SSD split coli_st_expert_route / COLI_DISK_WEIGHTS / the startup probe v4_mirror_probe_bw (v4:371-404). With more mirrors, or COLI_WEX_WEIGHTS=<primary>,<mirror1>[,...], a deterministic Knuth hash (layer*E+expert)*2654435761 over the weights decides (equal weights when unset).
  - Store-side 58/42 split: hv = (h>>16)&255, mirror if hv >= mirror_cut_256. v4_wex_pick keeps each expert on the same drive.
- **Mode 0** (host slab, GPU-direct off), v4_read_expert_record (v4:13866-13911):
  - Reads total_bytes into slot->slab + wex_record_read_offset, then coli_wex_decode_record.
  - Counters: v4_direct_reads / payload_bytes / fallbacks. A failure falls back to safetensors.
  - For WNR2 this costs 3 PCIe trips: H2D record, D2H slab, H2D again at attach. That is ~38 MB (38.8 MB measured) per expert vs 13.4 MB for WEX1.
- **COLI_WEX_GPU_DIRECT** (atoi, default 0; v4:18488-18574):
  - **1**: decode lanes (v4_job_run) send VRAM-first misses that are not in RAM straight from disk -> page-locked per-lane staging -> helper decode into the fp4 VRAM mirror (GPU-only view).
  - **2**: also the next-layer prefetcher v4_np_worker. Guesses with rank < COLI_WEX_GPU_DIRECT_PREFETCH_RANK (default 6; <0 -> 0) go to VRAM via coli_v4_gpu_expert_prefetch_wex (no routing credit); weaker guesses keep the RAM prefetch.
  - Requires helper feature 1 AND the backend's dsv4_cuda_wex_direct_available (dsv4_cuda_tensor_alloc_fp4 + dsv4_cuda_tensor_fp4_buffers + stream drain, resolved via GetProcAddress). Otherwise it logs that it is "staying on the host slab path".
- **Store side** (v4:14011-14034): coli_v4_expert_store_wex_entry returns a record only if:
  - its codec is 3, or 4 with coli_wex_cuda_nr2_available();
  - it fits the staging buffer;
  - RAM neither holds nor is reading it (else ram_skip).
  - coli_v4_expert_store_wex_read does an overlapped read from the routed mirror and adds to disk_sec.
- **Staging and shapes**: one _aligned_malloc(align4096(record)) per loader thread, page-locked via coli_wex_cuda_host_register (pageable still works).
  - Mirror tensor order gate/down/up = w1/w2/w3. Shape check: rows == weight_shapes[m][0], cols == weight_shapes[m][1]*2.
  - Views are COLI_TENSOR_FP4_NATIVE_BLOCK with UE8M0 scales, block_columns 32.
- **Demand chain** (one record per call; PLAN line numbers at 79aff2b): v4_job_run -> coli_v4_gpu_expert_attach_wex (v4:18450-18510, ~18528 now) -> v4_gpu_expert_attach_cached_fill (v4:18092) -> v4_wex_fill_mirror (v4:18372-18400) -> coli_wex_cuda_decode_to_device (v4:18397). The current uncommitted tree differs (e.g. crc_get at 14162, RAM cache use at 18477-18484).
- **Mode-2 race protection**:
  - V4GpuMirrorFill reserves the entry (busy) under the mutex; the producer reads disk and decodes outside the mutex (the read was moved inside the producer after a duplicate-read bug), then publishes or frees and wakes waiters. Busy entries are neither victims nor hits, and same-key requesters wait.
  - Epochs: every peek/attach/publish marks the entry with cache->epoch. The MoE layer advances it via coli_v4_gpu_mirror_step after compute + stream drain. The direct producer never evicts a current-epoch entry.
  - Guard (patch_guard.py): prefetch never takes the 16 round-robin transient slots. In mode 2 the host path also avoids current-epoch victims. A direct load with no free mirror is a refusal (status 2), not a failure.
  - Counters: "v4_gpu_mirror_epochs steps= guard_skips= direct_refusals= pool= pool_fills= pool_hits=". moeprof gained cpu_layers.
- **Prefetch pool** (patch_pool.py; COLI_WEX_GPU_DIRECT_POOL, default 32, clamped 0..256):
  - No new VRAM: the pool marks the coldest loaded, non-busy, non-current-epoch resident mirrors, which lose the RAM-priority bit.
  - Only prefetch loads colder than every resident write into it; a full pool is reused LRU.
  - Host path and lane direct loads never evict pool entries (victim_range, LRU and victim_safe skip them). Modes 0/1 never mark entries.
  - Test: coli_v4_test_gpu_pool_slot in test_v4_gpu_priority.
- **Compressed-record RAM cache** (coli_v4_wex_crc_*, v4:14073-14303):
  - Enabled by COLI_WEX_RAM_CACHE_GB > 0 AND COLI_WEX_GPU_DIRECT >= 1 AND an open container.
  - Fixed slots, each the size of the largest codec 3/4 record (11.76 MiB on V4). 4096-aligned arena, page-locked unless COLI_WEX_RAM_CACHE_PIN=0. LRU by clock; leased slots are never evicted ("refused" counter).
  - Used only by v4_wex_fill_mirror: a VRAM miss decodes straight from the RAM record, and disk-read records stay cached.
  - COLI_WEX_RAM_CACHE_VERIFY=1 adds a 64-bit checksum on fill and on every hit (diagnostic, slower).
  - coli_v4_wex_crc_prewarm (called at v4:15287 after the VRAM and RAM-slab tiers, 8 OpenMP threads) loads the most-used records that neither tier holds, then zeroes hit/miss counters.
  - Logs "[v4_wex] RAM cache of compressed records: N slots x M MiB = G GiB, page-locked=P"; exit lines "v4_wex_ram_cache slots= slot_mib= hits= misses= evictions= refused= prewarmed= pinned=" and "v4_wex_ram_cache_verify checks= bad=".
- **Other knobs and diagnostics**:
  - WEX_READ_RETRY=1: one retry after Sleep(16) on a failed read (opt-in).
  - DSV4_CUDA_STREAM_PRIORITY=1 (backend_cuda_dsv4.cu, added 2026-11-05): the inference streams get the highest priority. It only trims tails.
  - Exit line "v4_wex_gpu_direct reads= payload_bytes= ram_skips= failures= device_decodes=" (v4:15066); start log "[v4_wex] GPU-direct Neural-rANS reconstruction enabled ...".
- **Backend upload race** (pre-existing, also in production, latent with WEX1):
  - Bug: upload() and the pageable branch of dsv4_cuda_tensor_refill_fp4 used synchronous cudaMemcpy on the legacy stream while the compute streams are cudaStreamNonBlocking. MoE kernels could read a fresh mirror before its scales landed (only mirror creations from RAM during decode, at most 16 per process).
  - Fix lab\wex-gpu-direct-20261019\backend-upload-race-fix.patch (43 lines; REPORT says 42, likely before the error-path nit fix): cudaMemcpyAsync on c->stream + cudaStreamSynchronize (error paths too); the pageable refill is always drained.
  - Probable cause of the C4 divergence, not proven (needs an upload() probe or nsys).
  - Still open: upload() waits 3x per new mirror, sometimes holding cache->mutex. Legacy-stream copies remain in dsv4_cuda_kv_create, expert_set_create, expert_bank_upload_tp2 and expert_set_upload_hash (not called).
- **CUDA Graphs**: the backend exports dsv4_cuda_graph_begin / _launch (backend_loader_dsv4.c wraps them), but deepseek_v4.c never calls them.
- **Standalone store expert_store_wex.c** (backend 'wex'):
  - With an engine it delegates to coli_v4_expert_store_open_planned.
  - Standalone it opens COLI_WEX_PATH or `<model_dir>\v4_flash_experts.wex` and registers the WNR2 model ("invalid WNR2 model descriptor" / "invalid or unreadable WNR2 model"). It checks the COLI_MODEL_MIRROR replica, uses 24 slots/layer and COLI_DISK_WEIGHTS default 58/42.
- **Tests**:
  - tests\test_wex_format.c test_nr2_codec: round trip with 3 chunk sizes; 14 corruptions rejected with the slab untouched; descriptor and header flag. test_wex_format also has 9,216 tile combos (9 widths x 4 variants x 256 counts, test_wex_format.c:128-135). The "30,000 streams" figure is a separate comparison against the previous decoder reported in docs\wex-v2.md:41, not part of test_wex_format.c.
  - tests\test_expert_store_wex.c test_nr2_container: model registration at open, exact decode, corrupt/foreign model refused, entries invalid without the header flag; encodes at 4096:2048.
  - nr2\test_nr2_cuda: 30 records exact via host and device with buffers from another runtime; 70 malformed rejected; 300 fuzz cases (0 accepted); WNR1 regression OK; compute-sanitizer 0 errors.
  - check_all.py: 688/688 exact for WNR2 and WNR2z.
- **V4.1 today**: V4.1 runs CPU-only in deepseek_v4.c.
  - Logs: "v41_moe path=cpu-union experts=192 hidden=5120 moe=2304 topk=6 (GPU route/bank and the decode prefetch pipeline are V4-only)" and "v41_attn path=cpu".
  - On V4.1 today WNR2 can run only in mode 0: host slab via v4_read_expert_record -> coli_wex_decode_record -> helper wex_cuda_decode_to_host, plus a D2H of 18.8 MB per record. UNTESTED. The WNR2 guard (v4:13151-13172) checks only coli_wex_cuda_nr2_available(), with no is_v41 or GPU-engine condition, and the helper loads lazily whatever DSV4_CUDA is set to.
  - GPU-direct, the RAM record cache and VRAM mirrors need G2. An M0-style WNR2 CPU arm is possible now.
  - On CPU MoE, WEX1 is read with "v4_ssd_io mode=wex-direct-aligned fallback=buffered-pread".
- **Host and profile**:
  - src\host\ColibriFamily.cs V41EnvNames treats WEX, V4_* and DSV4_* as V4-only and drops them for V4.1, so KAIRA cannot pass COLI_WEX_PATH to V4.1 yet (item H1). Runs pass it directly (M0: COLI_WEX_PATH=D:\kaira_wex\v41-reap\v4_flash_experts.wex).
  - app\kaira.json (the only live config; app\colibri-optimized has no kaira.json) still sets COLI_EXPERT_STORE=wex and COLI_WEX_PATH=C:\models\DeepSeek-V4-Flash-0731\v4_flash_experts.wex twice: in the top-level colibri.env default block (lines 17-18) and in the models[id=v4] profile (lines 123-124). That file no longer exists.
- **Machine**: RTX 5090 under WDDM: asyncEngineCount=1 (a single copy engine, so H2D and D2H serialize), tccDriver=0, CUDA driver/runtime 13050, 170 SMs. Ryzen 9 7950X3D.

## 12. Measurements

**Sizes**

| Date | What | Result |
|---|---|---|
| 2026-10-18 | WEX2 on 2 layer-10 V4 experts | 26,746,880 -> 25,415,680 B (-4.977%); CPU tile decoder 1,119.932 -> 63.789 us/expert (17.6x, 7950X3D, GCC 16.2 -O3 x86-64-v3); scales ~6.25% of volume; never built full |
| 2026-10-20 | Pilot, 688 of 11,008 V4 experts (6.25%) | WEX1 9,198,907,392; WNR1 -9.48% (3.681 bpw, 11.54 MiB/expert, full projected ~124.1 GiB, never built); WNR2 -12.69% vs WEX1 / -3.55% vs WNR1 (3.674 bpw, avg record 11,672,255 B = lengths 13 KB + scales 101 KB + weights 11.556 MB); WNR2z -15.70% / -6.87% (3.546 bpw, 1,106,956,429 -0 codes mapped, 11.27 MB/expert) |
| 2026-10-20 | V4 full WNR2 | 11,008/11,008 WNR2, 0 fallbacks; avg record 11,661,338 B vs raw 13,369,344 (header 32, lengths 13,056, scale 100,330, weight 11,545,863, padding 2,058); 3.6703 bit/w, 1.0206 bit/scale, 3.707 all-in; -12.775% (128,368,873,472 vs 147,170,537,472) |
| 2026-11-07 | V4.1 W1 | avg record 16,914,680 B = 0.8997 of raw 18,800,640 (header 32, lengths 38,880, scale 147,386.4, weight 16,726,327.8, padding 2,053.9); 3.781 bit/w (V4: 3.670), 1.066 bit/scale, 3.824 all-in; records 129,904,742,400 vs 144,388,915,200; saved 14,484,111,360 B = -10.031%; capacity per fixed tier 1/0.8997 = +11.1% records; finer chunks cost ~0.3 points [inferred] |

**GPU decode**

| Date | What | Conditions | Result |
|---|---|---|---|
| 2026-10-19 | WNR1 helper, 14 real records, median of 20 | | host decode + upload 2604 us; direct pageable 1034 us; direct pinned 819 us |
| 2026-10-20 | test_nr2_cuda (helper features 7) | pinned staging, median per V4 expert | decode_to_device: WNR2 chunk 4096 569.3 us, chunk 8192 822.4 us, WNR1 514.1 us. Kernel only: H2D 203.6, scales 75.2, weights shared-rows 252.2 (total 321.3) vs buckets 878.4 (total 948.1) |
| 2026-10-20 | check_all.py (decode_to_host) | | pilot nr2 688/688 exact, mean 1.998 ms; nr2z 688/688 (vs canonicalized raw), 3.859 ms; full V4 11,008/11,008 exact, mean 3.66 ms, 40.3 s decode, 187.7 s total |
| 2026-11-05 | helper single call | one thread | ~922-1095 decodes/s = ~1.0 ms/expert = ~0.22 copy + ~0.39 kernel + ~0.4 launch/sync |
| 2026-11-05 | bench_nr2_kernel | record already in VRAM | 1 stream 0.387 ms (32.2 GiB/s decoded); 2: 0.199 (62.6); 4: 0.143 (86.9); 8: 0.121 (102.8 GiB/s decoded = 89.5 compressed) |
| 2026-11-05 | bench_nr2_batch | V4 11.04 MiB records, RAM->VRAM incl. PCIe, one sync per batch, 40 rounds | batch 1/2/4/6/8 = 0.625/0.497/0.349/0.303/0.281 ms/record; 105 slabs byte-identical; batch 8 = -72% vs helper; PCIe-bound at 6-8 (floor 0.22 ms), parity with a raw WEX1 slab (~0.26-0.27 ms) |
| 2026-11-05 | bench_nr2_rechunk | 16 V4 experts, single-record ms / 8-concurrent ms | weight 4096: size 1.000, 192 blocks, 0.437/0.137 (90.9 GiB/s); 2048: 1.0028, 384, 0.264/0.132 (94.4); 1024: 1.0088, 768, 0.227/0.136; 512: 1.0208, 1536, 0.224/0.146. weight:scale 2048:2048 +0.28% 0.261/0.126; **2048:512 +0.34% 0.1998/0.1254 (99.3 GiB/s), -54% vs 4096:2048**; 2048:256 +0.42% 0.186/0.139; 1024:256 +0.99% 0.188/0.132. Floor ~0.19-0.22 ms |
| 2026-11-05 | test_decode_batch, 64 V4 records | 512 verified each, ALL OK | 2048:512 (11.08 MiB): single 0.734, b1 0.544, b2 0.435, b4 0.406, b6 0.434, b8 0.339, b10 0.403, b16 0.382. 4096:2048 (11.04 MiB): single 0.848, b1 0.762, b2 0.607, b4 0.407, b6 0.403, b8 0.399, b10 0.431, b16 0.352 |
| 2026-11-07 | verify64, V4.1 records (16.14 MiB) | NOT a benchmark: GPU shared with outside work, 13-98% busy, 10-24 GB used | single 0.759; b1 0.718; b2 0.541; b4 0.449; b6 0.424; b8 0.402; b10 0.394; b16 0.378 ms/record (PLAN predicted ~0.4-0.55) |

**Transfers, contention and launch overhead**

| Date | What | Conditions | Result |
|---|---|---|---|
| 2026-11-05 | h2d_bw | pinned | 11 MiB (11,534,336 B) 48.60 GiB/s = 0.221 ms per copy with sync; 53.42 GiB/s back to back. Raw 12.75 MiB ~0.25-0.256 ms, so a compressed record saves only ~0.03 ms of PCIe. At 48.6 GiB/s (52.2 GB/s) an 18.8 MB V4.1 raw record is ~0.36 ms and a 16.14 MiB V4.1 WNR2 record ~0.32 ms (PLAN.md:441 says 0.39, same arithmetic slip) |
| 2026-11-05 | bench_prio | fg = 40 short kernels + sync, p50/p90 us | WNR2 decode bg: idle 397/473, same priority 511/710, high-priority chain 500/583. H2D-only 11 MB: 422/833, 561/1100, 544/1136. Kernels only: 399/699, 507/638, 506/608. Plain H2D alone inflates the chain ~33%, WNR2 decodes ~29%. Priority: p50 -2%, p90 -18%, p99 -13%. Contention is PCIe + command fetch, not SMs |
| 2026-11-05 | bench_nr2_batch-contend | 650 decodes/s (~118/token at 5.5 tok/s), fg p50/p90 us | idle 381.4/440.2; singles 412.6 (+8%)/526.3 (+20%); batches of 6 384.4 (+1%)/675.0 (+53%). Bursts need pacing |
| 2026-11-05 | wddm_graph | 40-kernel chain + sync | short kernels: streams 342.0/462.1 us vs one CUDA graph 35.9/56.4; heavier: 402.3 vs 186.3 (~8.5 us/kernel via streams vs ~0.9 in a graph) |

**Disk and simulation**

| Date | What | Conditions | Result |
|---|---|---|---|
| 2026-11-05 | record_read_probe on E: | unbuffered, 8 threads | kept run sim\record-read-e-t8-swap.json: WEX1-sized 12.45 MiB 1.619 GiB/s, 7.51 ms/expert; WNR2 11.08 MiB 1.613 GiB/s, 6.71 ms; time ratio 0.893 = byte ratio (11.08/12.45). Discarded reverse-order run record-read-e-t8.json (WNR2 1.995 vs WEX1 1.329 GiB/s; mean 11.05 / 12.66 MiB; local_measurements_2.md:41 mixes its MiB values into the kept run); a third run failed with an unexplained read error |
| 2026-11-05 | Simulation sim\sim_tiers.py | IRM from .coli_usage, 1.277e9 selections (~258/token), calibrated x0.626 on the October replay | Oct replay: 45.8 demand reads/token at 1530 VRAM experts (70% hit), 43.9 at 1700, 41.0 at 2000; live: 9-15 demand + ~71 prefetch reads, ~1.1 GB/token. Demand reads/token (MB), WEX1 / WNR2 / +compressed VRAM: CV1530 4GB 82.8 (1108) / 81.9 (955) / 77.2 (900, 8.4 decodes); 8GB 76.5 (1023) / 74.8 (872) / 70.5 (822); 16GB 65.4 (874) / 62.5-63.2 (728-737) / 58.8 (686); CV1950 8GB 68.5 (915) / 66.9 (780) / 62.1 (724, 9.2). WNR2 cuts bytes ~13-16% but reads only 1-2%; compressed VRAM ~19-20%. V4 on 32 GB would be disk-bound at ~0.7-1.1 GB/token |
| 2026-11-05 | Disk arithmetic | | at 6.6 GiB/s a WNR2 miss (11.04 MiB) costs 1.63 ms vs WEX1 (12.75 MiB) 1.89 ms, ~0.25 ms saved; each avoided miss is ~+0.14% at 182 ms/token; +10% needs ~70 avoided misses/token. Prefetch ~71 reads x ~11.7 MB = ~0.83 GB/token vs WNR2 saving ~0.15 GB/token: halving prefetch is worth ~58 ms/token, WNR2 ~20 ms/token. At 118 decodes/token the old helper costs ~118 ms/token vs ~33 ms with batch [inference]. Report "Sfruttare WNR2 e WEX3": delta-T ~= 0.25 ms x avoided NVMe misses + (+0.034 to -0.02 ms) x RAM hits |
| 2026-11-07 | Disk floor per V4.1 token | ~214 misses x 18.8 MB, ~4.0 GB/token [inferred] | safetensors on U: 0.61-0.70 s; WEX1 on D: 2.5-2.7 s; WEX1 on U: 0.61 s; WNR2 on E: ~1.5 s |
| 2026-11-05 | Qwen3.8 striping | file probe | C: 3.34, U: 2.23, C:+U: 6.60 GiB/s; U:+O: 1.81 vs U: 4.06 alone. Qwen FP8 entropy 6.53 bits/byte, so lossless saves at most 18%; CPU decode ~3 ms/expert; a fixed-length FP8 exponent code saves ~6.7% |

- **Simulation caveats** (Report "WNR2 cache decode CPU e WDDM", sections 1/3): one calibration point only (57 slots/layer, ~33 GB); the IRM model overestimates VRAM misses ~2x (141.7 vs 68.9); the 0.626 factor is applied to reads but not to VRAM misses, so the calibrated columns are mutually inconsistent; the V4 routing dump v4_moe_dump.bin is deleted, so the replay cannot be redone.

**Inference A/B**

| Date | What | Conditions | Result |
|---|---|---|---|
| 2026-10-19 | Pilot A/B 1 | A1 B1 C1 C2 B2 A2; 4 prompts x 64 tok greedy | 6/6 identical; mode 1: 46 device decodes/run, 0 failures, 1178 ram_skips; ~3.8-3.9 tok/s (contaminated); wait ~135, attach ~40, GPU ~17 ms/token |
| 2026-10-19 | Pilot mode 2 debugging | first cut, then epochs+guard (diag-e) | First cut diverged (D1=D2 prompt 2 token 23; 2938 reads vs 1550 decodes, fixed to 713=713; D3 prompt 1 token 42; D4 prompt 1 token 19; D5 identical 4.03). diag-e: A3 WEX1 3.951, E1 3.847*, E2 3.932*, C3 3.902*, E3 3.931, E4 3.924 (* contaminated), all identical; 46-47 decodes, ~412-416 refusals, guard_skips 5, cpu_layers 0 |
| 2026-10-19 | diag-f with pool | | A4 3.968*, F1 3.973, F2 4.055, C4 (mode 1) 4.018, F3 4.059, F4 4.019*; all identical; mode 2: 430-433 decodes vs 43 in mode 1; pool fills 393-395, uses 911-917; ram_skips ~1180 -> ~700; bytes A4 407.34 GB, C4 405.44, F 405.33 |
| 2026-10-20 | diag-g pilot | all contaminated (other desktop apps running) | A5 WEX1 4.03 (407.24 GB); N1 WNR1 mode2+pool 4.11 (431 loads, 12.08 MB); R1/R2 WNR2 4.18/4.14 (432/434 loads, 11.65 MB); Z1 WNR2z 4.17 (421, 11.24 MB, 403.32 GB); R0 WNR2 mode 0 4.11. Identical tokens |
| 2026-10-20 | V4 full A/B | U: only, no WEX mirror, identical tokens | WEX1 A1 5.573, A2 5.641*, A3 5.624 (407 GB); WNR2 mode 0 W0 4.814, W0b 4.833* (354.9 GB, -12.9% bytes) = **-14%**; mode 2 ~4.4 on 3 prompts = **~-20%** (stopped at prompt 4 by the NVMe thermal guard, SK Hynix sensor 75 C). Per token A1 vs W0: wait 76 vs 71, attach 35.6 vs 54.6, GPU 13.9 vs 15.5 ms |
| 2026-10-20 | Equal-RAM A/B with RAM cache | RAM_GB=30, U:, COLI_V4_EXPERT_SLOTS=16 + 12.86 GiB cache (1119 x 11.76 MiB) | WEX1 mode 0 5.62/5.63* (A7)/5.61* (A8); WNR2+cache mode 2 C3 5.446, C4 5.429 (diverged prompt 0 token 7), V2 5.462* = **~-3%** (the average includes contaminated V2); V1 +VERIFY 4.948* (19,456 checks, 0 bad); C5 PREFETCH_RANK=3 4.306*; C6 mode 1 3.935*; W3 mode 2 without cache 3.428* (contamination flags from full-crc3/4/5.log). C3: 19,051 hits, 11,107 misses, 30,158 device decodes (~118/token, mostly prefetch), 129.4 GB payload, ram_skips 17,573. MoE 113.5 vs 129.8 ms/token, but hc 27.9 vs 14.4 and attn 43.0 vs 34.6 (GPU contention). 6 slots break batch prefill |
| 2026-10-20 | Race-fix validation (full-fix1) | C4 config | R1/R2/R3 5.401/5.440/5.392*, identical; A9 WEX1 5.580* (flags from full-fix1.log). Base rate ~1 divergence per 5 runs, so this is not proof. DG sha 1510983d (runs), later d05fd43b after an error-path nit |
| 2026-10-20 | Deploy A/B (deploy-ab2) | real recipe, WEX1 | old a19c81cf/DG 026534d5: A1 7.463*, A2 7.755; new 7d139737/DG d05fd43b: B1 7.619*, B2 7.779; 4/4 identical |
| 2026-11-07 | M0 REAP CPU baseline | DSV4_CUDA=0, RAM_GB=16, CTX=4096, warm-up 16 tok + code1/chat_it1 64 tok, binary sha e215651b, 03:10-03:22 | R0a safetensors on U: 0.976/1.008 tok/s (TTFT 14.3/12.8 s); R0b WEX1 on D: 0.468/0.480 (34.1/29.4 s); R0c WEX1 on U: 1.110/1.137 (11.8/10.7 s); hit rate 34.6-37.1%. WEX1 on U: is about +13% vs safetensors (code1 1.110/0.976 = +13.7%, chat_it1 1.137/1.008 = +12.8%) and 2.4x D:. R0c: 9 slots/layer, 6.30 GiB cache + 6.65 GiB dense, prewarm 360 experts in 1.3 s, v4_direct reads=26129, payload 491,241,922,560 B, 0 fallbacks. **No WNR2 arm yet.** |

(* = contaminated run)

## 13. Timeline

- **2026-10-17**: V4 WEX1 container built (147,170,537,472 B), production.
- **2026-10-18**: WEX2 scale tiles + fast CPU decoder, Palette3 and WNR1 samples, architecture doc. wex_decode.c split at 23:39; tests (test_deepseek_v4, test_wex_format, test_expert_store_wex) green at 23:50.
- **2026-10-19**: GPU-direct modes 1/2, bridge optimization, epochs, guard, prefetch pool (lab\wex-gpu-direct-20261019).
- **2026-10-20**:
  - WNR2 codec + WNM2 + GPU encoder; pilot containers; full V4 WNR2 (128.37 GB, 779 s); full A/B (-14% / ~-20%);
  - compressed RAM cache (~-3%); backend upload race fix;
  - WEX1-only runtime deployed at 04:09.
- **2026-10-21**: the user deleted the U: copy of the V4 WNR2.
- **2026-11-05**:
  - V4 weights found gone, machine at 32 GB RAM; reports "Sfruttare WNR2 e WEX3" and "WNR2 cache decode CPU e WDDM";
  - batch/kernel/rechunk/H2D/WDDM/prio benches; batch helper (feature 8); stream-priority knob; WNR2-without-helper guard;
  - V4 WNR2 E: container deleted with OK; V4.1 REAP-192 chosen.
- **2026-11-06**: handoff. The V4-0731 checkpoint (rev 7872f01, 166.9 GB) was re-downloaded to U:\models in the evening.
- **2026-11-07**:
  - 00:47 batch-capable helper installed; ~00:59 V4.1 WEX1 repack on D: (40x192 = 7680 records of 18,800,640 B, 0 compressed/dedup/neural); ~01:10 V4.1 install;
  - 02:00-02:23 V4 checkpoint moved to H:; 02:23-02:34 verified U: WEX1 copy;
  - 02:34:57-02:47:01 W1 WNR2 build; 03:10-03:22 M0 baseline;
  - ~03:30 KAIRA-only decision; ~16:10 Wex-WRN2 repo; 16:33 F1.patch.
- **Git** (KAIRA master, newest first): 88ddb07 (port upstream #2625/#2623/#2639 + cc_buf race fix); 79aff2b (PLAN base); 0007c89 (V4.1 REAP-192 in V4 engine, CPU); 9a631c8 (qwen38/WEX: Mente route tap, warmstart, row-wise tier prefill, batched WNR2 helper); a94cfdd (25 upstream fixes). WNR2 core sources exist since baseline 2ed0ab9 (2026-10-26).
- **Uncommitted**: Makefile, backend_cuda_dsv4.cu/.h, backend_loader_dsv4.c, deepseek_v4.c, dsv4.def, tests\test_v41_on_v4_gpu_refusal.c, plus untracked tests\test_dsv4_fp8_block32_cuda.c (G1 appears applied).

## 14. Contradictions between sources

- **wex-gpu-direct-lab.md**: its description and first paragraph say "NOT deployed", but its body and the MEMORY.md index say engine/backend/helper were DEPLOYED on 2026-10-20 04:09. Resolution: the runtime binaries were deployed on WEX1; WNR2, GPU-direct and the RAM cache were never enabled. The same file also says E:\kaira_lab\wnr2-full is "the only copy now", but it was deleted later.
- **Savings**: -12.69%/-12.7% (pilot), -12.78% (full V4), -13.4% (Nov report vs zstd), "13-14%" and "14-16%" (Nov), and -12.8% expected for V4.1, but V4.1 measured -10.03%.
  - Capacity: ~14% (Oct, PLAN 3.2), +14.7% (sim), "15% in piu" (Nov REPORT), ~10% (COMPARISON), but V4.1 measured +11.1%.
  - m0.md assumed ~16.4 MB records; actual 16,914,680 B. PLAN W1 estimated ~126 GB (117.7 GiB) of output; actual 129.9 GB (120.98 GiB).
- **"3.94 tok/s"**: the Nov reports and PLAN F1 call it V4 "without prefetch". The source run C6 (3.935*) is WNR2 + cache in MODE 1 (no direct prefetch to VRAM, RAM prefetch still on), and it is a CONTAMINATED run. Also, "mode 2 ~-20%" (W2 ~4.4, interrupted) vs W3 mode 2 without cache 3.428* (-39% vs 5.62), also from a contaminated run.
- **The equal-RAM -3% A/B** ran on a 64 GB machine that was not disk-bound, so it does not transfer to 32 GB. Its average also includes the contaminated run V2 5.462*.
- **GB/token vs tok/s** (Report "WNR2 cache decode CPU e WDDM", open): 0.6-1.15 GB/token read from E: at 1.6-2.0 GiB/s does not fit the measured 7.1 tok/s.
- **Single-record decode time**: 569 us (2026-10-20, pinned) vs ~1.0 ms helper wall (2026-11-05) vs kernel-only 0.387 (bench_nr2_kernel) or 0.437 (rechunk baseline) vs 0.734/0.759 single calls. These measure different things (kernel vs wall, idle vs shared GPU), and the reports admit it is unclear whether the October A/B used the slow helper path. bench_nr2_batch V4 0.625/0.281 vs test_decode_batch. The research note gpu_entropy_decoders.md still uses a stale ~0.6 ms per record (= ~19 GB/s compressed) that the Nov report corrects.
- **w1.md** quotes "V4 4096:2048 records 0.734 / 0.339 ms", but the 2026-11-05 logs show those numbers for the 2048:512 file; 4096:2048 measured 0.848/0.399.
- **V4 record size**: 11,672,255 B (pilot, 11.13 MiB), 11.65-11.66 MB (diag-g/full; 128,368,873,472/11,008 = 11.12 MiB), 11.67 MB (arch doc). The Nov notes use 11.04 MiB (11.05 in the probe), which is low. wex-qwen38 says "11 MiB"; the V4 WEX1 expert is 13.4 MB = 12.75 MiB.
- **Feature 8 deployment**: v41-reap192-port.md (2026-11-06) says the batch helper was "built in lab only, not deployed"; v41-port-done says the ws-int helper was installed on 2026-11-07; PLAN said "[I]". A string check of the installed DLL confirms features 4+8. Memory also calls the October helper "feature bit 4", while the code has bits 1/2/4/8 and the README says "bits 4 and 8".
- **Pilot containers**: the old memory figure of ~33 GB was right (33.3 GB, 10 files including nr2z, now in Q:\kaira-temp\U_kaira_build_wexgd-20261019; U:\kaira_build is empty). Only the REPORT's ~17.5 GB is stale.
- **Stale REPORT "Limiti"**: it says no disk has space for the full WNR2 (estimate ~128.5 GB, -18.7 GB; WNR2z ~124.1 GB, ~16 min), yet it was built the same day on E:. The Nov 5 REPORT says E:\kaira_lab\wnr2-full remains; it was deleted later.
- **Arch doc** docs\wex-architettura-completa.md contradicts itself:
  - its 2026-10-18 body is stale: the code block (:55-58) lists header flags only up to 8, line 119 says no v2/v3 full container exists, and it says kaira_wex_cuda.dll is absent from app;
  - its 20 October WNR2 section (:216-245) names WEX_HEADER_NEURAL_RANS2 and reports the 128.37 GB full WNR2 container;
  - names U: "Samsung 970 Gen4" (M0: SK hynix HFS001TEJ9X162N), and expert_store_wex.c comments call U: Samsung 970;
  - its rANS bullets (reset every 32 weights, 16-state CDF) describe WNR1;
  - says the CDFs are "addestrate da una rete neurale", but they are counts.
- **V4 weights**: the 2026-11-06 handoff says they are gone; upstream-prs-20261105.md says V4-0731 was re-downloaded on the evening of 2026-11-06; PLAN moves it to H:. No V4 WNR2 container exists either way.
- **W1 source**: PLAN said D:, but U: was used (6.6 vs 1.6 GB/s); the 64-record check read D:, so both copies are covered.
- **Free space**: E: 15.3 GB after W1 (w1.md) vs 105.9 GB at 17:11 on 2026-11-07; U: 279 GB at 02:34 vs 122.6 GB at 17:11. Mind units: `df -h` prints GiB (E: 99G = 105.9 GB).
- **zipmoe note**: its break-even of 55.7 GiB/s (B/r) is wrong. Serial break-even is ~380 GiB/s (B/(1-r)); pipelined it is D > B, ~49 GiB/s.
- **qwen38**: memory says WEX was ported to Qwen3.8 (decision 2026-11-05), but qwen38_core.h has no WEX code.
- **MEMORY.md** indexed wnr2-complete.md before this file existed. docs\handoff\memory\MEMORY.md lacks 4 index lines (wnr2-complete, wex-formats-own-repo, v41-port-done-20261107, v41-handoff-20261106); the other handoff memory files are byte-identical to the live ones.

## 15. Source index

- **Code**: `<KAIRA>\src\colibri\c\` wex_format.h, wex_neural_rans.h, wex_decode.c/.h, wex_cuda.cuh, tools\wex_cuda_bridge.cu, tools\repack_v4_wex.py, deepseek_v4.c, expert_store_wex.c, backend_cuda_dsv4.cu, tests\test_wex_format.c, tests\test_expert_store_wex.c; src\host\ColibriFamily.cs; app\kaira.json.
  - Also in src\colibri\c\tools\: wex_neural_rans.py (WNR1 Python encoder + reference used by repack_v4_wex.py: chunks 256/512/1024, wex_nr_encode_gpu), bench_wex_cuda.cu and bench_wex_decode.c (WNR1/WEX2 benches).
- **October lab**: `lab\wex-gpu-direct-20261019\`
  - REPORT.md, COMPARISON.md;
  - wex-gpu-direct.patch (16 files vs src-before\ with SHA256SUMS), wex-gpu-direct.pre-crc.patch, backend-upload-race-fix.patch, src-crc\;
  - nr2\ (incl. build-full-nr2.json, check_full.json, test-nr2.log, build\test_nr2_cuda.json, temp-move.log, run_test_nr2.cmd, dump_exports.cmd, src-before-nr2\ = pre-WNR2 source snapshot for rollback);
  - patch_*.py (attach, reserve, read_inside, epoch, guard, pool), run_ab.py, run_ab2.py, run_plan.py, compare_tokens.py;
  - result dirs inference\, diag-d3..g\, full-ab*\, full-crc3/4/5\, full-fix1\, deploy-ab2.
- **November lab**: `lab\wex-prod-20261105\`
  - REPORT.md, reports\ ("Sfruttare WNR2 e WEX3.md", "WNR2 cache decode CPU e WDDM.md"), research_notes\ (local_measurements.md, local_measurements_2.md, cpu_entropy_decoders.md, "Sfruttare WNR2 e WEX3\gpu_entropy_decoders.md" with the stale 0.6 ms / 19 GB/s figure);
  - v41-port-plan.md and v41-port-gap.md (where the V4.1 W items came from);
  - bench_nr2_{batch,kernel,rechunk}.cu, h2d_bw.cu, wddm_graph.cu, bench_prio.cu, test_decode_batch.cu, rechunk.py, reencode_bin.py, sim\, *.json and *.log results.
- **V4.1 lab**: `lab\v41-gpu-20261107\` PLAN.md (W1/W2/W3/F1/G2/D1/H1, 3.1/3.2), w1.md + w1\, m0.md + m0\ (drive_probe.log, run_m0.out.log, runs\), moves.log, F1.patch, laneG-g1.patch, P1.patch. Also `lab\v41-port-20261106\` (repack-wex1-20261107.log, phase1-carryforward.md, HANDOFF.md).
- **Private repo**: `<HOME>\Wex-WRN2\`
  - README.md;
  - src\, tools\, nr2\, models\, bench\, tests\;
  - docs\ (wex-architettura-completa.md, wex-v2.md, the two Nov reports, v41-reap-wnr2-build.md = w1.md, v41-reap-baseline-m0.md = m0.md).
  - Build scripts carry KAIRA absolute paths (MSYS2 ucrt64, CUDA 13.4, MSVC 18).
- **Memory**: private KAIRA memory notes (not in this repo).
