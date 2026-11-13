# WNR2 history: decisions, drives, measurements and timeline (2026-10-20 to 2026-11-10)

> **Dated report, taken out of [WNR2-REFERENCE.md](../WNR2-REFERENCE.md) on 2026-11-13 with its text and section numbers unchanged.** Sections 4 to 10 (the WEX1-WEX3 container, the WNR2 record, the WNM2 model, the encoder, the decoders and the helper API) stayed in the reference; a "section 4" to "section 10" below means that page. The rules quoted in section 1 are those of the private period of this work (the owner's operating rules of the lab, a private repository); they do not apply to the repository as published. WNR2 itself was superseded by WNR3 on 2026-11-10 and no WNR2 container exists any more.

## How the original page was read


**Successor:** WNR3 (codec 5, header flags 32/64) is specified in [WNR3-SPEC.md](../WNR3-SPEC.md) and verified in [WNR3-VERIFY.md](WNR3-VERIFY.md); status in the section "WNR3 implemented (2026-11-10)" at the end of this file. WNR2 containers, the WNM2 model format and the WNR2 code path are unchanged by WNR3. **Since 2026-11-10 WNR3 is in production and every WNR2 container (U:, C:, E:) and WEX1 are deleted** ([WNR3-PRODUCTION-20261110.md](WNR3-PRODUCTION-20261110.md)); the disk tables in this file are history. Since 2026-11-11 the WNR3 records sit in a WEX4 container ([WEX4-SPEC.md](../WEX4-SPEC.md)) and Qwen3.8 reads MXFP4 records from WEX4 containers too; what runs today is in [STATUS.md](../../STATUS.md). This file remains the reference for the WNR2 record, the WNM2 model, the encoder and the helper API (sections 4 to 10). Section 11 describes the engine integration with line numbers at KAIRA cfb7ea0, and section 12 holds the measurements up to 2026-11-09. Its rules, disk tables, open work and timeline (sections 1 to 3 and 13 to 15, and the dated sections at the end) are the record of 2026-10-20 to 2026-11-10.

**Why this page exists:** WNR2 work was spread over three lab folders, two reports and this repository, and those sources disagreed on key numbers. This file is the single place that says what WNR2 is, where every container and model sat, what was measured under which conditions, and which decisions the owner took.

**How to read it:**
- Sections 1 (rules of the period), 2 (disk state) and 3 (open work) are the record of 2026-10-20 to 2026-11-10. The reference is sections 4-15: format and codec 4-7, build 8, decoders and helper 9-10, engine 11, measurements 12, timeline 13, contradictions 14, sources 15. Runs marked * are contaminated.
- The dated sections at the end (from '## Containers on disk, update 2026-11-08') are the newest; where they disagree with sections 2-13, they win.
- Paths: lab\, app\ and src\host paths are relative to the KAIRA tree (`<KAIRA>`) and are not in the Wex-WRN2 repo; copies of some lab results are under research\ there (e.g. research\v41-gpu-20261109\REPORT.md, research\v41-w3-20261109\results.md, research\v41-ab-20261108\ab-samedrive\results.md).
- Re-check disk free space and the existence of the files before relying on them.
- Prefer measured V4.1 numbers (-10.03%, +11.1%) over older V4 estimates.
- After any format change, update KAIRA src first, then re-sync Wex-WRN2.

## 1. Owner decisions and rules

- **2026-10-20 04:09**: the owner approved installing ONLY the WEX1-compatible runtime.
  - Lab engine 7d139737, backend d05fd43b (race fix) and helper cac5348d went into app\colibri-optimized; kaira.json unchanged.
  - Backup: lab\app-before-wexgd-20261020-040912. Rollback: copy back deepseek_v4.exe + coli_cuda_dsv4_dg.dll and delete kaira_wex_cuda.dll.
  - Switching KAIRA to WNR2, GPU-direct or the RAM cache via kaira.json was NOT approved. Verdict: WNR2 is a SPACE/capacity option. (History: superseded on 2026-11-09, see below.)
- **2026-11-05**: the owner chose to port WEX to Qwen3.8 (lever there: C:+U: striping, not compression; the WEX-Q8 container on C: needs OK on space; report lab\wex-prod-20261105\REPORT.md).
  - Then V4 was replaced by DeepSeek V4.1 Flash 1017 REAP-192 MXFP4 (patrickbdevaney/...), not "Thor", because Thor's K-quant experts are unreadable by the engine and by WNR2.
  - Deletion of the old V4 WNR2 container was approved.
- **2026-11-06**: containers go WEX1 raw on D:\kaira_wex and WNR2 on E:\kaira_wex.
  - Never commit, push, deploy to app\ or edit app\kaira.json without explicit OK.
  - Defender exclusions are set by the owner; no script of this work changes security settings.
  - No benchmark while a W1 encode or any build runs; take heavy.lock for big I/O.
- **heavy.lock**: `<KAIRA>\lab\v41-gpu-20261107\heavy.lock`.
  - Take it by creating the file exclusively (New-Item fails while it exists) and writing `<label> <ISO time>` into it. `lab\v41-gpu-20261107\w1\take_lock.ps1 -Label '...'` does this, retrying every 60 s for up to 90 min (prints TAKEN or TIMEOUT).
  - Release it by deleting the file.
- **2026-11-07 ~00:47-01:10**: V4.1 installed into app\ from src, including kaira_wex_cuda.dll built in ws-int. Backup lab\app-before-v41-full-20261107-004555, hash list lab\install-v41-full-20261107.txt; coli_cuda.dll not replaced; kaira.json unchanged.
- **2026-11-07 ~02:00**: the owner approved:
  - copying the WEX1 to U:\kaira_wex\v41-reap (use only when .verified exists);
  - moving the V4 checkpoint to H: with a junction;
  - for W3, putting both containers on U: if space allows (copy under heavy.lock, verify size + sha256, write .verified), otherwise reporting the drive confound.
- **2026-11-07 ~03:30**: KAIRA-only decision. WEX1/WNR2 code must NEVER go upstream to JustVugg/colibri, and WEX/WNR2 hunks are stripped from upstream PRs.
  - Code covered: expert_store_wex.c, wex_decode.c/.h, c\tools\wex_cuda_bridge.cu, wex_cuda.cuh, repack_v4_wex.py, the nr2 encoder.
  - It lives in the PRIVATE repo https://github.com/KairaCode/Wex-WRN2 (local clone `<Wex-WRN2>`, initial commit a74b279 synced from KAIRA master 88ddb07, created ~16:10).
  - Initial commit a74b279 (2026-11-07 16:10, from KAIRA 88ddb07). On 2026-11-09 main = origin/main = c39a650, 10 commits later. src\* (now c\) and the shared tools (wex_cuda_bridge.cu, repack_v4_wex.py, wex_neural_rans.py, benches) are byte-identical to KAIRA master cfb7ea0; models\ holds v4-flash-0731.wnm2 and v41-reap192.wnm2.
  - KAIRA src\colibri\c stays the live copy; re-sync the repo after format changes.
  - Reason: beating safetensors is KAIRA's own asset (WEX1 on U: ~1.12 vs safetensors ~0.99 tok/s).
  - The no-upstream rule applies to the public JustVugg/colibri repo.
- **2026-11-08 21:15**: container rule (replaces the +3% rule of PLAN 3.2): keep WNR2 if the token ids are identical and it is not slower.
- **2026-11-09**: the owner approved the U:+C: mirror (05:00), the cfb7ea0 install (06:05), profile v41gpu with WNR2 + GPU-direct 2 + U:+C: mirror (06:17), and v41gpu as the active model (16:47). At 17:08 the owner switched the active model to qwen38 in the UI.
- **Install rule**: a WNR2 deploy needs the new engine AND a helper with feature 4 together. With an old helper, WNR2 went to the CPU at ~0.29 s/expert; since 2026-11-05 the engine refuses the container instead. Ask before installing anything.
- **Decision rules**:
  - Nov 5: if < ~70 avoidable NVMe misses/token and the batch pipeline does not go under 0.22 ms, WEX1 stays. Round 2: if disk is not really the bottleneck, WEX1 stays.
  - PLAN 3.2 for V4.1: WNR2 becomes the speed format only if >= +3% tok/s with the 95% CI excluding 0 AND token ids identical in every prompt of every run; otherwise WEX1 stays in production and WNR2 is a capacity format. Replaced on 2026-11-08 21:15 by the "identical tokens, not slower" rule (above).
  - Any token difference stops the A/B and is treated as a bug.
- **Measurement protocol** (PLAN 3.1):
  - Quiet machine: no KAIRA/engine, builds or copies; Defender exclusions active; high-performance power plan.
  - Record hashes of the exe, both DSV4 DLLs and kaira_wex_cuda.dll; pin COLI_DSV4_DLL.
  - Lossless settings: V4_SKIP_COLD=0 (kaira.json v4_skip_cold is false, line 824; the host also forces 0 for V4.1 on deepseek_v4.exe, Engine.cs:616), V4_TOPK unset, V4_MTP=0, temperature 0.
  - Prompts: lab\v41-gpu-20261107\ab\prompts.jsonl (warm-up + code1, code2, chat_it1, chat_it2, reason1, reason2, ctx2k, short1; 256 tokens; no system prompt). Fresh process per run, ABBA x2.
- **W1 design choices**: retrain the WNM2 instead of reusing V4's; use 2048:512 directly; `--min-free-gib 5` to protect E:.
- **Not to do**:
  - fused rANS-in-GEMV (needs ~2.07 TB/s decoded vs 102.8 GiB/s measured, ~19x short, 1.16x GDDR7 1.79 TB/s);
  - green contexts (contention is not on SMs); GDS on Windows GeForce; IoRing as a throughput lever; MPS on Windows;
  - HOBBIT low-precision substitution (not lossless); W-TinyLFU; DirectStorage/GDeflate (-8%).
  - Prefetch must be gated, not removed. The per-record WNR1 helper path should be retired in favour of the per-layer batch.
- **Literature verdicts** (report "Sfruttare WNR2 e WEX3", now [reports/wnr2-wex3-when-compression-pays-20261105.it.md](wnr2-wex3-when-compression-pays-20261105.it.md); do not redo this research):
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
| E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex | V4.1 REAP-192 WNR2 (see header below), model 0xab4261bd | 129,905,360,896 B (120.98 GiB) | 2026-11-07 02:42 | EXISTS (the original). Verified copies: U:\kaira_wex\v41-reap (primary in v41gpu, since 2026-11-08 17:58:49) and C:\kaira_wex\v41-reap (mirror, since 2026-11-09 05:20:12). All three are 129,905,360,896 B, sha256 3F703FA4...EB9D. |
| D:\kaira_wex\v41-reap\v4_flash_experts.wex | V4.1 WEX1 original | 144,389,472,256 B | 2026-11-07 00:59:50 | DELETED 2026-11-09 05:02:30, after the U: copy was re-verified (sha256 4C055238...); D: free went 61.4 -> 205.8 GB. The WEX1 on U: is now the only copy. |
| U:\kaira_wex\v41-reap\v4_flash_experts.wex (+ .verified, 73 B) | byte copy of the V4.1 WEX1, sha256 4C055238D9011CE25D5C53A5EA61EC32BD8D1CF2886247194A13E7F53E7FA549 | 144,389,472,256 B | copied 02:23:12-02:34:10 2026-11-07 | exists; the reference container for REAP runs |
| E:\kaira_lab\wnr2-full\v4_flash_experts.nr2.wex | V4-0731 full WNR2, model 0x49435c70, sha256 c0c91294...2dd388, 43x256, aligned 13,369,344, records from 860,160, chunks 4096:2048 | 128,368,873,472 B | 2026-10-20 | DELETED (E: copy ~2026-11-05 with OK; U: copy by the owner 2026-10-21); E:\kaira_lab is empty |
| (V4 WEX1) v4_flash_experts.wex | V4-0731 WEX1 | 147,170,537,472 B (137.06 GiB) | built 2026-10-17 | DELETED with the V4 weights (gone by 2026-11-05) |
| Q:\kaira-temp\U_kaira_build_wexgd-20261019 | 10 files: pilot-v1.wex (WEX1) 9,198,907,392; pilot-v3-neural512.wex (WNR1) 8,326,713,344; pilot-v3-nr2.wex (WNR2, 688 V4 experts, model 0x2e0ca289, sha256 e5e04723...) 8,031,371,264 (2026-10-20 00:31); pilot-v3-nr2z.wex (WNR2z, 688 V4 experts, model 0xd4fa45dc, sha256 2ef03ead...) 7,754,739,712 (2026-10-20 00:32); build.json, mirror.json, cdf-uniform-fallback.bin, verify_cpu.dll, wex_neural_encoder.dll, Oct-19 kaira_wex_cuda.dll (195,072 B). links.json and manifest.json are one level up, in Q:\kaira-temp | 33,312,321,406 B (33.3 GB) | moved from U:\kaira_build\wexgd-20261019 (lab\wex-gpu-direct-20261019\nr2\temp-move.log: 10 files, nr2z included) | exists; U:\kaira_build empty; restore: `lab\wex-gpu-direct-20261019\nr2\temp_move.py restore Q:/kaira-temp` |
| U:\kaira_lab\wnr2-samples | records64_4096_2048.bin 1,600,768,792; records64_2048_512.bin 1,603,336,984 (64 verified V4 experts each: model + entries + raw + records); 16-expert rechunk inputs records_512 403,983,576 / _1024 401,755,352 / _1024_256 401,960,152 / _2048 400,641,240 / _2048_2048 400,641,240 / _2048_256 400,891,096 / _4096 400,112,856 | ~6.0 GB | 2026-11-05 | junction to H:\kaira_lab\wnr2-samples (moved 2026-11-08 17:51-17:52); V4 WNR2 samples (the other V4 WNR2 data left are the pilot WNR2/WNR2z containers in Q:\kaira-temp) |
| U:\kaira_lab (older) | wex-neural-lossless-20261018 (~171 MB: sample-deep-256/512/1024.wex 24,723,456 / 23,986,176 / 23,617,536; sample-scale-only-*; sample-v2.wex 25,415,680); wex-validation-20261019-155553 (215 MB); wex-weight-20261018 (~45 MB: sample-v2.wex, sample-v3-palette3.wex 19,128,320); wex3-integrated-20261019 and wex3-optimized-integrated-20261019 (empty); predictor_eval (374 MB, 25 files, 2026-10-13; sim_more.json) | ~0.8 GB | 2026-10-13..19 | exists |

- **Model blobs** (all 59,408 B):

| model | id | sha256 / notes | locations |
|---|---|---|---|
| V4.1 REAP | 0xab4261bd (2873254333) | f83fee82734219358d0f1290245298d44e2808d7abb48c9c0a742cd6dd855e3e; trained on all 7680 experts | embedded in the container; `lab\v41-gpu-20261107\w1\model-full-nr2.wnm2`; `<Wex-WRN2>\models\v41-reap192.wnm2` |
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
| D: | Sabrent SB-2130-1TB | PCIe 3.0 x2 (max x4 gen4) | 1.61 | 12.2 ms | 91.9 GB (11-07 17:11: 61.4 GB = 57.2 GiB) |
| U: | SK hynix HFS001TEJ9X162N | gen4 x4 | 6.58 | 3.2 ms | 14.6 GB (11-07: 279 GB at 02:34; 122.6 GB = 114.2 GiB at 17:11, other files on U: changed at 16:04) |
| C: | Crucial T700 | gen5 x4 | 10.66 | 2.3 ms | 30.4 GB, falling during the qwen38 pagefile incident |
| E: | ADATA SX8200PNP | gen3 x4 | 2.29 | 8.3 ms | 89.4 GB (11-07: 135.2 GiB before W1, 15.3 GB after (w1.md); 105.9 GB = 98.6 GiB at 17:11) |
| H: | Seagate ST3000DM001 | SATA HDD | | | holds the V4-0731 checkpoint (behind the U:\models\DeepSeek-V4-Flash-0731 junction), lfm2.5, mellum2.1 and wnr2-samples |

  - Free space 2026-11-09 ~17:50 (Get-PSDrive, decimal GB): C: 30.4 (falling during the qwen38 pagefile incident), D: 91.9, E: 89.4, U: 14.6. Re-check before use. For the 11-07 values, `df -h` shows GiB, so E: "99G" / U: "115G" / D: "58G" were the SAME values, not a drop. The 'U: ~7 GB short' note is obsolete: the U: WNR2 copy exists since 2026-11-08.

  - E: also holds engram shard 48 (E:\models\DeepSeek-V4.1-Flash-1017-engram\model-00048-of-00048.safetensors, 101,537,926,640 B), so check space before writing to E:.
  - Each letter is its own physical disk.
  - The V4 checkpoint was moved U:\Models\DeepSeek-V4-Flash-0731 -> H:\models\DeepSeek-V4-Flash-0731 with a junction (02:00-02:23 2026-11-07, 152 files, 107 hashed). Engram shard 47 moved to D:\models\DeepSeek-V4.1-Flash-1017-engram (D: is PCIe 3.0 x2) on 2026-11-09 05:02-05:10. A junction is left at the old C: path, and COLI_MODEL_DIRS (kaira.json:567,593) still names it.

## 3. Open work / next steps

- **G2**: DONE in cfb7ea0 (installed 2026-11-09 06:05). V4.1 decode MoE runs through moe_token_pipeline. GPU-direct 0 vs 2 is token-identical (REPORT.md:37-39). Measured 934-1,028 VRAM mirrors plus 360 RAM slots at 16 GB (REPORT.md:55). The original plan: route V4.1 decode MoE through moe_token_pipeline (VRAM mirrors, vram_first lanes, v4_np_predict prefetch, copy worker, promotion, async attach, WEX GPU-direct + RAM record cache).
  - The V4.1 router must avoid V4_ROUTER_AVX2 (is_v41 guard) and coli_v4_gpu_route; add dsv4_cuda_set_act_group (group 32).
  - Test: 256 greedy tokens with COLI_WEX_GPU_DIRECT=0 vs =2 + COLI_WEX_RAM_CACHE_GB=4 must be token-identical.
  - Expected VRAM tier: ~1,030-1,110 mirrors of 17.93 MiB (13.4-14.5% of 7,680); measured 934-1,028 (REPORT.md:55). The backend reserve or mirror budget must also cover the helper's own thread_local lanes (~287 MiB per batch-calling thread on V4.1, outside backend accounting; see section 10, helper memory footprint).
- **W2**: DONE in cfb7ea0, opt-in V4_WEX_BATCH=1 (getenv at deepseek_v4.c:19538; the batch block in moe_token_pipeline at :9732), which v41gpu sets (kaira.json:592). +0.5% vs per-record decode (CI +0.34..+0.70%), logits bit-identical, batches of only ~1.56-1.83 records. Still open: VRAM of the batch lanes outside the reserve (~2 GB [I]) and no unit tests for the overlapped read, the RAM cache inside a batch and the feature-8 guard (REPORT.md:80,85). Default off; only with COLI_WEX_GPU_DIRECT>=1 and coli_wex_cuda_batch_available(). The design (line numbers at cfb7ea0):
  1. In moe_token_pipeline (v4:9491; batch block 9732), collect WNR2 misses of coli_v4_gpu_expert_peek_routed that are not in host RAM (via coli_v4_expert_store_wex_entry, v4:14579).
  2. Split v4_gpu_mirror_fill_unlocked (v4:18798) into reserve(N) (one mutex hold + one dsv4_cuda_stream_drain) and publish(i).
  3. Fetch records from the RAM cache or with N overlapped ReadFile into a ring of 8 pinned staging buffers (~132 MB).
  4. Make one batch call; publish status 0; on -1/1 release and fall back per record.
  - Budget the helper lanes (8 x 2 x 17.93 MiB per calling thread) in the VRAM reserve, and restore the caller's CUDA device after each helper call (the helper leaves it set to `device`).
  - The prefetch twin in v4_np_worker comes after F1.
  - Expected: -0.6 to -0.9 ms per decode, -35 to -90 ms/token at 60-110 decodes/token.
  - Tests: batch vs per-record byte-identical; V4_WEX_BATCH 0 vs 1 token-identical; counters for batches/token, records/batch, ms/batch.
- **W3**: DONE 2026-11-09 01:50-05:00 (lab\v41-gpu-20261107\ab\results.md). Both containers on U:, --memory-gb 16, ABBA x2, tokens and logits bit-identical, WNR2 -1.80% +/- 0.65%. Deviation from the plan: no COLI_WEX_RAM_CACHE_GB (ab\results.md:98). The compressed-cache arm (pilot C, run_w3c.sh) and the U:+C: mirror arm were not measured. The plan was a WEX1 vs WNR2 A/B on V4.1 at equal RAM BYTES (slots x 18.8 MB + cache GB within 1%).
  - The arms differ only in COLI_WEX_PATH. Planned common settings: `--memory-gb 16`, COLI_WEX_GPU_DIRECT=2, COLI_WEX_RAM_CACHE_GB=4, DSV4_CUDA_VRAM_RESERVE_MB=4096, V4_COPY_WORKER=1, V4_PROMOTE_VRAM=2, V4_ASYNC_ATTACH=1, DSV4_CUDA_STREAM_PRIORITY=1, V4_WEX_BATCH=1, same F1 and V4_GPU_HEAD.
  - Add a 'wexprof' line next to moeprof (VRAM hits, RAM hits, NVMe reads/bytes, WNR2 demand/prefetch decodes, decode ms). Use +11.1% capacity, not +14%.
  - D: (1.61) vs E: (2.29 GB/s) differ by 42%, above the 10% threshold, so a same-drive control on U: is mandatory. It needs a ~130 GB WNR2 copy verified against sha256 3f703fa4... plus a .verified marker. The U: space note is obsolete: the WNR2 copy on U: was made on 2026-11-08.
- **F1**: in src since cfb7ea0 (V4NpItem.conf at deepseek_v4.c:7351; V4_NP_MIN_CONF / V4_NP_TOKEN_MB at :7440-7441), defaults off. Confidence-gated prefetch with a per-token byte budget (npprof gated=, histogram under V4_PREDICT_STATS=1). Thresholds tau1=0.10 and tau2=0.15 were chosen offline (ab\results.md:117-131). The F1 grid (f1\run_f1.sh) is not measured. Prefetch must stay.
- **WNR2 deploy helper check**: done. Since 2026-11-09 06:17 the app helper is the W1 build cceb4fb5, the binary that passed test_decode_batch on V4.1 records (verify64.log ALL OK).
- **Cache policy candidates** (Report "WNR2 cache decode CPU e WDDM", now [reports/wnr2-cache-decode-cpu-wddm-20261105.it.md](wnr2-cache-decode-cpu-wddm-20261105.it.md), next to F1):
  - Measured LRU hit rate 70% at 1530 vs IRM-static 45%, so "LRU beats LFU" is not proven. Replay offline: LRU with a frequency tie-break, S3-FIFO-style probation for prefetches, decay. FlashMoE claims +21% hit.
  - Proposed gating rule: depth-1 (top-8) promotes to VRAM only at high confidence; depth-2 (top-6) goes only to RAM probation; a per-token byte budget; a per-layer cap, more generous on deep layers.
- **WDDM candidates** (same report): cudaDeviceScheduleSpin on the decode thread and blocking sync on the copy thread; HAGS untested; cudaMemcpyBatchAsync cannot be captured in a CUDA graph; graphs must cope with expert addresses that change every token.
- **Other**:
  - H1: done through COLI_FAMILY_ENGINE=deepseek_v4 in the v41gpu profile. The stale V4 COLI_WEX_PATH entries are gone from kaira.json. Still open: the outdated preflight warning (DeepSeekV41Preflight.cs:243-246 at cfb7ea0, 241-244 in the working tree) that tells v41gpu to remove COLI_WEX_PATH.
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
  - The qwen38 WEX port is not in src. *(Done 2026-11-11 with MXFP4 experts: `c/qwen38_wex.h`, [QWEN38-MXFP4-20261111.md](QWEN38-MXFP4-20261111.md).)*
  - Re-sync Wex-WRN2 after changes.

## 11. Engine integration (deepseek_v4.c only)

WEX/WNR2 integration exists only in deepseek_v4.c and the standalone store expert_store_wex.c. qwen38_core.h has no WEX code, and deepseek_v41.c is CPU/Vulkan only. *(State at cfb7ea0. Since 2026-11-11 the qwen38 engine has its own container store, `c/qwen38_wex.h`.)*

Line numbers `v4:NNNN` in this section are for KAIRA master cfb7ea0 (`git show cfb7ea0:src/colibri/c/deepseek_v4.c`); the uncommitted working tree is shifted by about +140.

- **Container open** (v4:13821-13918; COLI_WEX_PATH getenv 13823, logs 13848-13853):
  - Path: COLI_WEX_PATH, default `<model_dir>\v4_flash_experts.wex`. Opened with FILE_FLAG_NO_BUFFERING|OVERLAPPED|SEQUENTIAL_SCAN.
  - v4_wex_load_manifest validates header and table.
  - Logs "[v4_wex] Container v%u active: <path> (L layers, E experts, N decoded bytes/expert, direct_io=on)" or "[v4_wex] Invalid or incompatible container: %s; using safetensors".
- **WNR2 guard** (in v4_wex_load_manifest, v4:13693-13750; message at 13736; added 2026-11-05):
  - For header flag 16 the engine reads the WNM2 at the descriptor, registers it and requires registered id == descriptor id.
  - If !coli_wex_cuda_nr2_available() it prints "[v4_wex] WNR2 container needs kaira_wex_cuda.dll with WNR2 support (helper feature 4) next to the engine" and REFUSES the container, so the engine stays on safetensors.
  - Before 2026-11-05 it silently fell back to the CPU decoder at ~0.3 s/expert.
  - The standalone expert_store_wex.c has NO such guard: it registers the model and can fall back to the CPU.
- **Mirrors**:
  - COLI_MODEL_MIRROR (';' or ',' list) plus COLI_WEX_EXTRA_MIRRORS (dirs holding only the container) give up to 3 WEX replicas (v4:13860-13913).
  - A replica is accepted only if size, the 4096-B header page and the table are byte-identical ("Matching mirror container active" / "Ignoring mismatched mirror"). WNR2 replicas re-register the same model, which is a no-op.
  - Read split: with 1 mirror, the dual-SSD split coli_st_expert_route / COLI_DISK_WEIGHTS / the startup probe v4_mirror_probe_bw (v4:374-401; v4_expert_route 351-357; the pick is v4_wex_pick, v4:14395). With more mirrors, or COLI_WEX_WEIGHTS=<primary>,<mirror1>[,...], a deterministic Knuth hash (layer*E+expert)*2654435761 over the weights decides (equal weights when unset). NOTE (2026-11-09): the 1-mirror dual-SSD route is active only when COLI_MODEL_MIRROR registers shard copies; with a container-only mirror (COLI_WEX_EXTRA_MIRRORS) set COLI_WEX_WEIGHTS, or every read goes to the primary.
  - Store-side 58/42 split: hv = (h>>16)&255, mirror if hv >= mirror_cut_256. v4_wex_pick keeps each expert on the same drive.
- **Mode 0** (host slab, GPU-direct off), v4_read_expert_record (v4:14434-14550):
  - Reads total_bytes into slot->slab + wex_record_read_offset, then coli_wex_decode_record.
  - Counters: v4_direct_reads / payload_bytes / fallbacks. A failure falls back to safetensors.
  - For WNR2 this costs 3 PCIe trips: H2D record, D2H slab, H2D again at attach. That is ~38 MB (38.8 MB measured) per expert vs 13.4 MB for WEX1.
- **COLI_WEX_GPU_DIRECT** (atoi, default 0; v4_wex_gpu_direct_enabled v4:19363-19380, getenv 19366; _PREFETCH_RANK getenv 19390):
  - **1**: decode lanes (v4_job_run) send VRAM-first misses that are not in RAM straight from disk -> page-locked per-lane staging -> helper decode into the fp4 VRAM mirror (GPU-only view).
  - **2**: also the next-layer prefetcher v4_np_worker. Guesses with rank < COLI_WEX_GPU_DIRECT_PREFETCH_RANK (default 6; <0 -> 0) go to VRAM via coli_v4_gpu_expert_prefetch_wex (no routing credit); weaker guesses keep the RAM prefetch.
  - Requires helper feature 1 AND the backend's dsv4_cuda_wex_direct_available (dsv4_cuda_tensor_alloc_fp4 + dsv4_cuda_tensor_fp4_buffers + stream drain, resolved via GetProcAddress). Otherwise it logs that it is "staying on the host slab path".
- **Store side** (coli_v4_expert_store_wex_entry v4:14579-14599, coli_v4_expert_store_wex_read 14601-14628): coli_v4_expert_store_wex_entry returns a record only if:
  - its codec is 3, or 4 with coli_wex_cuda_nr2_available();
  - it fits the staging buffer;
  - RAM neither holds nor is reading it (else ram_skip).
  - coli_v4_expert_store_wex_read does an overlapped read from the routed mirror and adds to disk_sec.
- **Staging and shapes**: one _aligned_malloc(align4096(record)) per loader thread, page-locked via coli_wex_cuda_host_register (pageable still works).
  - Mirror tensor order gate/down/up = w1/w2/w3. Shape check: rows == weight_shapes[m][0], cols == weight_shapes[m][1]*2.
  - Views are COLI_TENSOR_FP4_NATIVE_BLOCK with UE8M0 scales, block_columns 32.
- **Demand chain** (one record per call): v4_job_run (call at v4:6756) -> coli_v4_gpu_expert_attach_wex (v4:19403) -> v4_gpu_expert_attach_cached_fill (v4:18938) -> v4_wex_fill_mirror (v4:19337-19361; RAM cache coli_v4_wex_crc_get at 19347) -> coli_wex_cuda_decode_to_device (v4:19354). With V4_WEX_BATCH=1 (v4:19538) a layer's misses go through one batch call instead (section 3, W2).
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
- **Compressed-record RAM cache** (coli_v4_wex_crc_*, v4:14745-14950; v4_wex_crc 14745, crc_get 14805, prewarm 14890):
  - Enabled by COLI_WEX_RAM_CACHE_GB > 0 AND COLI_WEX_GPU_DIRECT >= 1 AND an open container.
  - Fixed slots, each the size of the largest codec 3/4 record (11.76 MiB on V4). 4096-aligned arena, page-locked unless COLI_WEX_RAM_CACHE_PIN=0. LRU by clock; leased slots are never evicted ("refused" counter).
  - Used only by v4_wex_fill_mirror: a VRAM miss decodes straight from the RAM record, and disk-read records stay cached.
  - COLI_WEX_RAM_CACHE_VERIFY=1 adds a 64-bit checksum on fill and on every hit (diagnostic, slower).
  - coli_v4_wex_crc_prewarm (called at v4:15963 after the VRAM and RAM-slab tiers, 8 OpenMP threads) loads the most-used records that neither tier holds, then zeroes hit/miss counters.
  - Logs "[v4_wex] RAM cache of compressed records: N slots x M MiB = G GiB, page-locked=P"; exit lines "v4_wex_ram_cache slots= slot_mib= hits= misses= evictions= refused= prewarmed= pinned=" and "v4_wex_ram_cache_verify checks= bad=".
- **Other knobs and diagnostics**:
  - WEX_READ_RETRY=1: one retry after Sleep(16) on a failed read (opt-in).
  - DSV4_CUDA_STREAM_PRIORITY=1 (backend_cuda_dsv4.cu, added 2026-11-05): the inference streams get the highest priority. It only trims tails.
  - Exit line "v4_wex_gpu_direct reads= payload_bytes= ram_skips= failures= device_decodes=" (v4:15742); start log "[v4_wex] GPU-direct Neural-rANS reconstruction enabled ...".
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
  - c\tests\test_wex_format.c test_nr2_codec: round trip with 3 chunk sizes; 14 corruptions rejected with the slab untouched; descriptor and header flag. test_wex_format also has 9,216 tile combos (9 widths x 4 variants x 256 counts, test_wex_format.c:128-135). The "30,000 streams" figure is a separate comparison against the previous decoder reported in docs\reports\wex2-scale-tiles-20261018.it.md:41, not part of test_wex_format.c.
  - c\tests\test_expert_store_wex.c test_nr2_container: model registration at open, exact decode, corrupt/foreign model refused, entries invalid without the header flag; encodes at 4096:2048.
  - nr2\test_nr2_cuda: 30 records exact via host and device with buffers from another runtime; 70 malformed rejected; 300 fuzz cases (0 accepted); WNR1 regression OK; compute-sanitizer 0 errors.
  - check_all.py: 688/688 exact for WNR2 and WNR2z.
- **V4.1 at cfb7ea0** (installed 2026-11-09; later builds and what runs today: [STATUS.md](../../STATUS.md)): decode MoE runs through moe_token_pipeline with VRAM mirrors, WEX GPU-direct mode 2, W2 batches and the exact GPU head (V4_GPU_HEAD=1). The attention core and prefill MoE stay on the CPU.
  - Logs: "v41_moe path=cpu-union ... (every row without expert mirrors, else prefill and multi-slot rows; the GPU route and the prefill bank are V4-only)" (v4:11241) and "v41_attn path=cpu projections=gpu".
  - WNR2 GPU-direct is tested on V4.1 (W3: 8 runs; mirror verify: 41,271 decodes, 0 failures). In mode 0 (host slab via v4_read_expert_record -> coli_wex_decode_record -> helper wex_cuda_decode_to_host, plus a D2H of 18.8 MB per record) a full V4.1 run is still untested. The WNR2 guard (v4:13736) checks only coli_wex_cuda_nr2_available(), with no is_v41 or GPU-engine condition, and the helper loads lazily whatever DSV4_CUDA is set to.
  - On CPU MoE, WEX1 is read with "v4_ssd_io mode=wex-direct-aligned fallback=buffered-pread".
- **Host and profile**:
  - src\host\ColibriFamily.cs V41EnvNames still drops WEX/V4_*/DSV4_* for a V4.1 profile that runs deepseek_v41.exe. With COLI_FAMILY_ENGINE=deepseek_v4 (RunsOnV4Engine, ColibriFamily.cs:75-82), Engine.ApplyProfileEnv (Engine.cs:64-79) passes the whole profile env to deepseek_v4.exe, so H1 is solved for v41gpu. (M0 passed COLI_WEX_PATH=D:\kaira_wex\v41-reap\v4_flash_experts.wex directly.)
  - app\kaira.json (the only live config) no longer sets COLI_EXPERT_STORE or the stale V4 COLI_WEX_PATH; those were last seen in lab\app-before-v41-full-20261107-004555\kaira.json:17-18,123-124. The only WEX settings are in the v41gpu profile: COLI_FAMILY_ENGINE=deepseek_v4 (588), COLI_WEX_PATH=U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex (589), COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap (590), COLI_WEX_WEIGHTS=38,62 (591), V4_WEX_BATCH=1 (592), COLI_WEX_GPU_DIRECT=2 (620).
- **Machine**: RTX 5090 under WDDM: asyncEngineCount=1 (a single copy engine, so H2D and D2H serialize), tccDriver=0, CUDA driver/runtime 13050, 170 SMs. Ryzen 9 7950X3D.

## 12. Measurements

**Sizes**

| Date | What | Result |
|---|---|---|
| 2026-10-18 | WEX2 on 2 layer-10 V4 experts | 26,746,880 -> 25,415,680 B (-4.977%); CPU tile decoder 1,119.932 -> 63.789 us/expert (17.6x, 7950X3D, GCC 16.2 -O3 x86-64-v3); scales ~6.25% of volume; never built full |
| 2026-10-20 | Pilot, 688 of 11,008 V4 experts (6.25%) | WEX1 9,198,907,392; WNR1 -9.48% (3.681 bpw, 11.54 MiB/expert, full projected ~124.1 GiB, never built); WNR2 -12.69% vs WEX1 / -3.55% vs WNR1 (3.674 bpw, avg record 11,672,255 B = lengths 13 KB + scales 101 KB + weights 11.556 MB); WNR2z -15.70% / -6.87% (3.546 bpw, 1,106,956,429 -0 codes mapped, 11.27 MB/expert) |
| 2026-10-20 | V4 full WNR2 | 11,008/11,008 WNR2, 0 fallbacks; avg record 11,661,338 B vs raw 13,369,344 (header 32, lengths 13,056, scale 100,330, weight 11,545,863, padding 2,058); 3.6703 bit/w, 1.0206 bit/scale, 3.707 all-in; -12.775% (128,368,873,472 vs 147,170,537,472) |
| 2026-11-07 | V4.1 W1 | avg record 16,914,680 B = 0.8997 of raw 18,800,640 (header 32, lengths 38,880, scale 147,386.4, weight 16,726,327.8, padding 2,053.9); 3.781 bit/w (V4: 3.670), 1.066 bit/scale, 3.824 all-in; records 129,904,742,400 vs 144,388,915,200 (saved 14,484,172,800 B); whole files 129,905,360,896 vs 144,389,472,256 (saved 14,484,111,360 B); both -10.031%; capacity per fixed tier 1/0.8997 = +11.1% records; finer chunks cost ~0.3 points [inferred] |

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
| 2026-11-05 | h2d_bw | pinned | 11 MiB (11,534,336 B) 48.60 GiB/s = 0.221 ms per copy with sync; 53.42 GiB/s back to back. Raw 12.75 MiB ~0.25-0.256 ms, so a compressed record saves only ~0.03 ms of PCIe. At 48.6 GiB/s (52.2 GB/s) an 18.8 MB V4.1 raw record is ~0.36 ms and a 16.14 MiB V4.1 WNR2 record ~0.32 ms (PLAN.md:445 says 0.39, same arithmetic slip) |
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
- **2026-10-21**: the owner deleted the U: copy of the V4 WNR2.
- **2026-11-05**:
  - V4 weights found gone, machine at 32 GB RAM; reports "Sfruttare WNR2 e WEX3" and "WNR2 cache decode CPU e WDDM";
  - batch/kernel/rechunk/H2D/WDDM/prio benches; batch helper (feature 8); stream-priority knob; WNR2-without-helper guard;
  - V4 WNR2 E: container deleted with OK; V4.1 REAP-192 chosen.
- **2026-11-06**: the V4-0731 checkpoint (rev 7872f01, 166.9 GB) was re-downloaded to U:\models in the evening.
- **2026-11-07**:
  - 00:47 batch-capable helper installed; ~00:59 V4.1 WEX1 repack on D: (40x192 = 7680 records of 18,800,640 B, 0 compressed/dedup/neural); ~01:10 V4.1 install;
  - 02:00-02:23 V4 checkpoint moved to H:; 02:23-02:34 verified U: WEX1 copy;
  - 02:34:57-02:47:01 W1 WNR2 build; 03:10-03:22 M0 baseline;
  - ~03:30 KAIRA-only decision; ~16:10 Wex-WRN2 repo; 16:33 F1.patch.
- **Git** (KAIRA master, newest first; local only, no remote): cfb7ea0 (V4.1 GPU tiers G1/G2/G3, P1, F1 off, W2 opt-in); 6be34da (upstream dev fixes); 88ddb07 (port upstream #2625/#2623/#2639 + cc_buf race fix); 79aff2b (PLAN base); 0007c89 (V4.1 REAP-192 in V4 engine, CPU); 9a631c8 (qwen38/WEX: Mente route tap, warmstart, row-wise tier prefill, batched WNR2 helper); a94cfdd (25 upstream fixes). WNR2 core sources exist since baseline 2ed0ab9 (2026-10-26).
- **Uncommitted** (2026-11-09 evening): the working tree has new uncommitted edits (qwen38/qwen36 CUDA fix, deepseek_v4.c, Makefiles, src\host RAM guard) and does not match the installed build.

## 14. Contradictions between sources

- **The October GPU-direct lab note**: its first paragraph says "NOT deployed", but its body says engine/backend/helper were DEPLOYED on 2026-10-20 04:09. Resolution: the runtime binaries were deployed on WEX1; WNR2, GPU-direct and the RAM cache were never enabled. The same note also says E:\kaira_lab\wnr2-full is "the only copy now", but it was deleted later.
- **Savings**: -12.69%/-12.7% (pilot), -12.78% (full V4), -13.4% (Nov report vs zstd), "13-14%" and "14-16%" (Nov), and -12.8% expected for V4.1, but V4.1 measured -10.03%.
  - Capacity: ~14% (Oct, PLAN 3.2), +14.7% (sim), "15% in piu" (Nov REPORT), ~10% (COMPARISON), but V4.1 measured +11.1%.
  - m0.md assumed ~16.4 MB records; actual 16,914,680 B. PLAN W1 estimated ~126 GB (117.7 GiB) of output; actual 129.9 GB (120.98 GiB).
- **"3.94 tok/s"**: the Nov reports and PLAN F1 call it V4 "without prefetch". The source run C6 (3.935*) is WNR2 + cache in MODE 1 (no direct prefetch to VRAM, RAM prefetch still on), and it is a CONTAMINATED run. Also, "mode 2 ~-20%" (W2 ~4.4, interrupted) vs W3 mode 2 without cache 3.428* (-39% vs 5.62), also from a contaminated run.
- **The equal-RAM -3% A/B** ran on a 64 GB machine that was not disk-bound, so it does not transfer to 32 GB. Its average also includes the contaminated run V2 5.462*.
- **GB/token vs tok/s** (Report "WNR2 cache decode CPU e WDDM", open): 0.6-1.15 GB/token read from E: at 1.6-2.0 GiB/s does not fit the measured 7.1 tok/s.
- **Single-record decode time**: 569 us (2026-10-20, pinned) vs ~1.0 ms helper wall (2026-11-05) vs kernel-only 0.387 (bench_nr2_kernel) or 0.437 (rechunk baseline) vs 0.734/0.759 single calls. These measure different things (kernel vs wall, idle vs shared GPU), and the reports admit it is unclear whether the October A/B used the slow helper path. bench_nr2_batch V4 0.625/0.281 vs test_decode_batch. The research note gpu_entropy_decoders.md still uses a stale ~0.6 ms per record (= ~19 GB/s compressed) that the Nov report corrects.
- **w1.md** quotes "V4 4096:2048 records 0.734 / 0.339 ms", but the 2026-11-05 logs show those numbers for the 2048:512 file; 4096:2048 measured 0.848/0.399.
- **V4 record size**: 11,672,255 B (pilot, 11.13 MiB), 11.65-11.66 MB (diag-g/full; 128,368,873,472/11,008 = 11.12 MiB), 11.67 MB (arch doc). The Nov notes use 11.04 MiB (11.05 in the probe), which is low. wex-qwen38 says "11 MiB"; the V4 WEX1 expert is 13.4 MB = 12.75 MiB.
- **Feature 8 deployment**: the port note of 2026-11-06 says the batch helper was "built in lab only, not deployed"; the note written after the port says the ws-int helper was installed on 2026-11-07; PLAN said "[I]". A string check of the installed DLL confirms features 4+8. An older note also calls the October helper "feature bit 4", while the code has bits 1/2/4/8 and the README says "bits 4 and 8".
- **Pilot containers**: the older figure of ~33 GB was right (33.3 GB, 10 files including nr2z, now in Q:\kaira-temp\U_kaira_build_wexgd-20261019; U:\kaira_build is empty). Only the REPORT's ~17.5 GB is stale.
- **Stale REPORT "Limiti"**: it says no disk has space for the full WNR2 (estimate ~128.5 GB, -18.7 GB; WNR2z ~124.1 GB, ~16 min), yet it was built the same day on E:. The Nov 5 REPORT says E:\kaira_lab\wnr2-full remains; it was deleted later.
- **Arch doc** docs\reports\wex-architecture-20261018.it.md contradicts itself:
  - its 2026-10-18 body is stale: the code block (:55-58) lists header flags only up to 8, line 119 says no v2/v3 full container exists, and it says kaira_wex_cuda.dll is absent from app;
  - its 20 October WNR2 section (:216-245) names WEX_HEADER_NEURAL_RANS2 and reports the 128.37 GB full WNR2 container;
  - names U: "Samsung 970 Gen4" (M0: SK hynix HFS001TEJ9X162N), and expert_store_wex.c comments call U: Samsung 970;
  - its rANS bullets (reset every 32 weights, 16-state CDF) describe WNR1;
  - says the CDFs are "addestrate da una rete neurale", but they are counts.
- **V4 weights**: the lab note of 2026-11-06 says they are gone; a later note says V4-0731 was re-downloaded on the evening of 2026-11-06; PLAN moves it to H:. No V4 WNR2 container exists either way.
- **W1 source**: PLAN said D:, but U: was used (6.6 vs 1.6 GB/s); the 64-record check read D:, so both copies are covered.
- **Free space**: E: 15.3 GB after W1 (w1.md) vs 105.9 GB at 17:11 on 2026-11-07; U: 279 GB at 02:34 vs 122.6 GB at 17:11. Mind units: `df -h` prints GiB (E: 99G = 105.9 GB).
- **zipmoe note**: its break-even of 55.7 GiB/s (B/r) is wrong. Serial break-even is ~380 GiB/s (B/(1-r)); pipelined it is D > B, ~49 GiB/s.
- **qwen38**: an earlier note says WEX was ported to Qwen3.8 (decision 2026-11-05), but qwen38_core.h has no WEX code. *(True on 2026-11-09. Since 2026-11-11 the qwen38 engine reads MXFP4 experts through `c/qwen38_wex.h`.)*

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
- **Private repo**: `<Wex-WRN2>\`
  - README.md;
  - c\ (mirror of KAIRA src\colibri\c, with tools\ and tests\), wex4\, scripts\, nr2\, models\, bench\, research\, integration\, fixtures\;
  - docs\ (COLIBRI-INTEGRATION.md, WNR2-REFERENCE.md) and docs\reports\ (wex-architecture-20261018.it.md, wex2-scale-tiles-20261018.it.md, wnr2-wex3-when-compression-pays-20261105.it.md, wnr2-cache-decode-cpu-wddm-20261105.it.md, v41-reap-wnr2-build-20261107.md = w1.md, v41-reap-baseline-m0-20261107.md = m0.md, kaira-status-20261109.it.md, wnr3-research-20261107.it.md and wnr3-research-measurements-20261107.md (were research-wnr3-20261107\), wnr4-entropy-engine-evaluation-20261108.it.md, wex-prod-20261105.it.md, wnr2-reads-vs-ram-cache-20261109.md).
  - The `.cmd` build scripts take `VCVARS`, `NVCC` and `ARCH` from the environment; only the default of `VCVARS` is an absolute MSVC 18 path.

## Containers on disk, update 2026-11-08
- WNR2 for V4.1 REAP copied E: -> U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex (sha256 3F703FA4..., .verified written). It now sits next to WEX1 on the same drive, so a same-drive WEX1 vs WNR2 A/B is possible. The E: copy is kept.
- To free U: (owner's OK), lfm2.5-24b-a2b, mellum2.1-12b-a2.5b and U:\kaira_lab\wnr2-samples were moved to H: and junctions left in place. Every file was sha256-verified. Script and log: lab\v41-gpu-20261107\free_u_and_copy_wnr2.ps1 and moves.log.
- U: free afterwards: 14.6 GB.

## Owner decision 2026-11-08: keep WNR2
- Same-drive A/B on U: (G2 GPU build, `--memory-gb 13`, bf16 head streamed from U:, `lab\v41-gpu-20261107\ab-samedrive-20261108`). In the first A1 vs B1 pair, WNR2 is about +1.7-3.6% tok/s per prompt and TTFT on the 2k-context prompt drops from 527 to 500 s.
- The owner chose to keep WNR2 if the token ids are identical and it is not slower: WNR2 is the V4.1 container even below the +3% rule, as long as the token ids are identical and it is not slower. The rule is recorded in the PLAN.md UPDATE.
- Done on 2026-11-09 with the owner's OK: the KAIRA profile v41gpu uses WNR2 with the U:+C: mirror, and the D: copy of WEX1 was deleted. WEX1 on U: (144 GB) stays until a short ABBA confirms the mirror; removing it needs the owner's OK.
- ABBA final (2026-11-08 20:33-21:44, `--memory-gb 13`, G2 build deepseek_v4-fix3.exe 6493e16e): WNR2 vs WEX1 decode is +2.04% (paired, 95% CI +1.24..+2.85, 8/8 pairs positive; prompt-unit CI +0.77..+3.32). TTFT -4.4..-6.9%.
- Tokens and logits are bit-identical in all 4 runs (sha256 4300486e...).
- Under the owner's rule this result (+2.04%, --memory-gb 13, bf16 head streamed from U:, one ABBA) would have adopted WNR2. It was superseded by W3 on 11-09 (16 GB, ABBA x2): WNR2 -1.80% +/- 0.65%, so by the rule WEX1 won on a single drive. The owner nevertheless chose WNR2 striped over U:+C: (38/62) for the v41gpu profile (installed 06:17). That configuration has only an unpaired sanity run (code1 3.04, chat_it1 2.66 tok/s). The short ABBA against WEX1 U:-only is still open. Results: `lab\v41-gpu-20261107\ab-samedrive-20261108\results.md`. About 2/3 of the gain comes from less U: contention on the streamed bf16 head at 13 GB.

## U:+C: mirror set up (2026-11-09 05:00-05:20, approved by the owner)
- The WEX1 copy on U: was re-verified (sha256 4C055238...). The duplicate WEX1 on D: was then deleted, freeing 144 GB.
- Engram shard 47 moved from C:\models\DeepSeek-V4.1-Flash-1017-engram to D:\models\... with a junction left at the old C: path. 4 files, sha256 identical. kaira.json and COLI_MODEL_DIRS are unchanged.
- WNR2 copied to C:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex (sha256 3F703FA4..., .verified).
- WNR2 copies now: U: (primary), C: (mirror), E: (original).
- Engine env for the mirror: COLI_WEX_PATH=U:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex, COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap, COLI_WEX_WEIGHTS=38,62. COLI_DISK_WEIGHTS would NOT split a container-only mirror. Verified on 2026-11-09: see 'Deployed 2026-11-09' below.
- Free space afterwards: C: 34.5 GB, D: 104.3 GB, U: 14.6 GB.
- Script: lab\v41-gpu-20261107\mirror_uc_setup.ps1. Log: moves.log.

## W3 final (2026-11-09 01:50-05:00, integrated build, 16 GB, same drive U:)
- WNR2 vs WEX1 decode: -1.80% +/- 0.65% (4 ABBA pairs). Tokens and logits are bit-identical in all 8 runs.
- Under the owner's rule ("not slower") WEX1 stays the container. WNR2 remains a capacity format.
- Cause: GPU-direct records bypass the RAM tier, so WNR2 makes 11% more reads. Its rANS decode also shares the GPU and copy engine with attention (+8.5 ms).
- Next test: WNR2 with the U:+C: mirror (and the compressed RAM cache) against WEX1 U:-only. Notes are in lab\wnr2-fastbench-20261109.
- The GPU workflow report is saved at lab\v41-gpu-20261107\REPORT.md. G1+G2+G3 are done:
  - code1 1.11 -> 2.11-2.27 tok/s;
  - chat_it1 1.14 -> 1.67-1.70 tok/s;
  - P1 ctx2k TTFT -23%;
  - W2 +0.5% (opt-in);
  - F1 code done, not measured.
- At the time of REPORT.md (05:20) nothing was committed or installed. Later that morning the tree was committed (6be34da, cfb7ea0 at 05:25) and installed (06:05), as described below.

- Commits (2026-11-09, local only, not pushed): 6be34da upstream dev fixes; cfb7ea0 V4.1 on the GPU tiers (G1/G2/G3, P1, F1 off, W2 opt-in). The owner's src/host RAM-guard edits remain uncommitted.

## Deployed 2026-11-09 (approved by the owner, then the mirror in kaira.json, also approved)
- app\colibri-optimized now has the files built from cfb7ea0:
  - deepseek_v4.exe 7a6f3e3f;
  - coli_cuda_dsv4_dg.dll b1b7cdb5;
  - coli_cuda_dsv4.dll 5483b6a7;
  - mcp_server.py.
  coli_cuda.dll was not replaced. Backup and rollback: lab\app-before-v41-gpu-20261109-060501 (rollback.ps1).
- kaira.json has a new profile "v41gpu". COLI_FAMILY_ENGINE=deepseek_v4 is the host switch that makes V4.1 run on deepseek_v4.exe; without it the host filters out the WEX/DSV4 env vars. At deploy time (06:05) active_model was still v4. It became v41gpu at 16:47 and qwen38 at 17:08 (see the last entry).
- Verified: launched the way the host launches it; 32 greedy tokens identical to W3; KAIRA.exe --selftest-dsv41 passes 63/63.
- Mirror switch (06:17): helper kaira_wex_cuda.dll a7977200 -> cceb4fb5 (the W1 build that W3 used). In v41gpu: COLI_WEX_PATH now points to WNR2 on U:, plus COLI_WEX_EXTRA_MIRRORS=C:\kaira_wex\v41-reap, COLI_WEX_WEIGHTS=38,62 and V4_WEX_BATCH=1.
  - Use COLI_WEX_WEIGHTS, not COLI_DISK_WEIGHTS. Without COLI_WEX_WEIGHTS a single container-only mirror is routed through coli_st_expert_route, which is inactive unless COLI_MODEL_MIRROR registers shard copies, so every read silently goes to the primary. The host also strips COLI_DISK_WEIGHTS when mirror_dir is null.
  - Backup: lab\app-before-mirror-20261109-061724.
- Known issues:
  - the host preflight warns on every v41gpu start to remove COLI_WEX_PATH (outdated text, harmless);
  - profile v4 now runs the new exe and has had no real V4 smoke yet;
  - coli doctor reports a RAM false alarm;
  - unfixed backend risk R1 (REPORT.md section 6): host_pinned registers whole VirtualQuery regions (backend_cuda_dsv4.cu:3449); v41gpu runs with V4_COPY_WORKER=1 and pinning on by default. REAP was clean in testing, but the bug is live. R2 V4 mirror-cache bug also unfixed (one-line fix, needs OK).
- Mirror verified (2026-11-09 06:23, host-equivalent launch, RAM auto 16):
  - log lines: "Matching mirror container active: C:\...nr2.wex" and "read split over 2 drives: 38% 62%";
  - OS disk counters during the prompts: C: 61.1% / U: 38.9%;
  - GPU-direct: 41,271 device decodes, 0 failures;
  - token ids identical to the first 32 (reason1) and 128 (code1, chat_it1) ids of all 8 W3 runs, WEX1 and WNR2.
- Sanity tok/s (not an A/B, 128 tok, ctx 32768, 16 OMP threads vs 13 in W3): code1 3.04 vs W3 WEX1 2.27 (+34%), chat_it1 2.66 vs 1.67 (+59%).
- Problems:
  - free system RAM fell to about 1 GB after the prompts;
  - U: heats fast (+21-22 C in about 2.6 min: 36->57 and 38->60 C);
  - a proper short ABBA against WEX1 U:-only is still to do. *(Done 2026-11-10: +49%, NOTE-5; see "WNR3 in production" below.)*
- 2026-11-09 06:00: the owner postponed the remaining tests. At 17:17 the fast-bench validation (lab\wnr2-fastbench-20261109\validate: replay ABBA, 3 variants, short ABBA WEX1 vs WNR2 on U:) was resumed and holds heavy.lock; check its results before planning the short ABBA mirror vs WEX1, which is still not scripted (use COLI_WEX_WEIGHTS=38,62).
- 2026-11-09 16:47: active_model set to v41gpu with the owner's approval (backup lab\app-before-active-v41gpu-20261109-164704). At 17:08 the owner switched the active model to qwen38 in the UI (app\kaira.json:869), so KAIRA now starts qwen38, not V4.1. v41gpu (WNR2 + U:+C: mirror) stays configured but inactive. Same afternoon: the 11-07 qwen38.exe/qwen36.exe (HIP builds) were replaced at 17:07 by CUDA rebuilds from artifacts\qwen-cuda-fix-20261109 (rollback: lab\app-before-qwen-cuda-fix-20261109-170614; the -170740 folder already holds the new builds). The first CUDA qwen38 run exhausted RAM and the C: pagefile; a fix is in progress, so do not start qwen38 until it lands.
- 2026-11-09 evening: fast-bench results (research/wnr2-fastbench-20261109, VALIDATION.md):
  - The short ABBA reproduces W3's verdict in ~18 min. 16 GB, 3 prompts x 128 tok: WNR2 U:-only -2.69% (fixed-effects CI +-0.38, pair-means +-1.55), against W3 cut the same way at -2.44%. Token ids and logits are identical. After review the default became 7 prompts x 64 tok (W3 cut that way: -1.89%). The tool now has VRAM/RAM/quiet-machine gates, a stall guard, and needs both pair means on the verdict's side.
  - The trace replay is only a byte-checked drive exerciser. On a shared trace it ranks WNR2 9.5% faster, the wrong sign, because it measures bytes and its WNR2 scheduling differs from the engine's. Do not use it to compare formats.
  - Engine counters, hypothesis only: in the engine, WNR2 reads queue deeper and each read is slower. Read latency 15.7-16.0 ms against 13.7 ms per read, 5.5-5.7 outstanding against 4.4, +0.3-0.9% bytes, +12% records. Env-only tests to try: V4_NEXT_PREFETCH_THREADS 4 vs 8, COLI_WEX_GPU_DIRECT_PREFETCH_RANK, COLI_WEX_RAM_CACHE_GB > 0.
  - WNR2 + U:+C: mirror vs WEX1 U:: NOT MEASURED yet *(measured 2026-11-10: +49%, one pair)*. Two attempts found the PC busy with the owner's desktop apps, other engines and builds. Driver: mirror-abba/run_mirror_abba.sh, about 30-35 min on a free PC. The owner's tests were stopped on 11-10 at 00:12.
- 2026-11-10 00:04: active_model is v41gpu again (WNR2 + U:+C: mirror). qwen38 was removed from app\.

## WNR3 implemented (2026-11-10)

- WNR3 = WNR2 record skeleton + ceil-form with rule-g anchors + canonical zero (optional, header flag 64) + a per-stream MurmurHash3 integrity hash (final state L + h23). Entry codec 5, header flag 32, record magic WNR3, owner in record bytes 28..31, helper feature 16 (features 31). Spec: [WNR3-SPEC.md](../WNR3-SPEC.md) (its "Implementation notes" list where the code deviates, e.g. the in-place per-block scale fix-up instead of the 64-block emask).
- Code: KAIRA branch wnr3-20261110 (b7a49ff, fcbe820, 8ef9645, cade51d on master 3b604bb), not merged, not deployed *(both done later that day, next section)*. Synced here on branch wnr3-20261110: c\ (then src\), c\tools\wex_cuda_bridge.cu, c/tests/test_wex_format.c, c/tests/test_expert_store_wex.c, the files then in nr3\ (c\tools\build_nr3.py, c\tools\wex_nr3_tools.c, c\tests\test_v4_wex_nr3.c, c\tests\fake_wex_cuda.c, c\tests\test_wex_nr3_gpu.cu), integration/engine/deepseek_v4.wnr3.wex-excerpt.c, research\wnr3-20261110\.
- Measured on 103 real V4.1 records ([WNR3-VERIFY.md](WNR3-VERIFY.md)): WNR3-CZ 0.8545 of raw on the 40 held-out (WNR2 0.8994, WNR3 without CZ 0.8818); value-exact 103/103 on CPU and GPU; 0 wrong single-bit flips accepted (WNR2: 1.8-1.9% of random flips over a whole record, 0.13-0.43% in the exhaustive GPU regions); GPU decode at parity (batch-16 0.366-0.373 vs 0.376-0.377 ms/record). Full container estimate ~123.4 GB = 114.9 GiB.
- NOT done: G7 (end-to-end per-step logits with V4_LAB_CANON_ZERO, ~55 min under heavy.lock) and the full container build (disk plan in WNR3-VERIFY section 5, owner decision). Production stays WNR2 (U:+C: mirror). *(Superseded the same day: G7 passed, container built, WNR3 in production; next section.)*

## WNR3 in production (2026-11-10, later)

- G7 PASSED; full WNR3-CZ container built (123,306,192,896 B, 0.949 of WNR2, 7,680/7,680 read back exact); merged into KAIRA master (89bdc30).
- KAIRA runs V4.1 REAP-192 from WNR3 on U: with a full C: mirror (38/62, GPU-direct 2, batch 1). Long turn (1,886 tokens): 2.92-2.96 tok/s vs WNR2 2.86 the same morning; short run 2.7-3.3 tok/s, logits bit-identical to the WNR2 reference.
- WEX1 and every WNR2 copy (U:, C:, E:) were deleted. A WNR2 rollback needs WEX1 regenerated from the safetensors and WNR2 rebuilt from it. Details: [WNR3-PRODUCTION-20261110.md](WNR3-PRODUCTION-20261110.md).
