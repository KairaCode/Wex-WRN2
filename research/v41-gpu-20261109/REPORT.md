# DeepSeek V4.1 REAP-192 on the GPU tiers: final report (2026-11-09 05:20)

> Snapshot 2026-11-09 05:20. Later the same day: committed as cfb7ea0 (05:25), installed in app\ (06:05, backup lab\app-before-v41-gpu-20261109-060501), profile v41gpu switched to WNR2 + U:+C: mirror (06:17). R8 now holds only for the stale preflight warning (DeepSeekV41Preflight.cs:241-244).

## 1. Bottom line
- **(a) GPU tiers: done.** G1 (dense FP8 with 32x32 blocks), G2 (MoE through moe_token_pipeline) and G3 (exact GPU head) are in src. Measured decode vs the M0 CPU baseline (R0c, WEX1 on U:):
  - code1: 1.110 -> 2.110-2.268 tok/s (+90 to +104%).
  - chat_it1: 1.137 -> 1.673-1.695 tok/s (+47 to +49%).
  - Against a CPU run in the same session: 1.55-2.04x.
  - The plan expected 1.1-2.5 tok/s.
- **(b) W2 batched WNR2: done, opt-in** (V4_WEX_BATCH=1). Only +0.5% (CI +0.34 to +0.70%) over per-record decode, with bit-identical logits.
- **(c) W3 A/B: done** (full ABBA x2, same drive U:, equal RAM). Tokens and logits are bit-identical, but WNR2 is -1.80% ± 0.65%.
  - Under the user's 11-08 rule (identical tokens and not slower), WEX1 stays the container.
  - WNR2 stays a capacity format (-10% disk).
- **(d) F1 prefetch gating: code done, not measured.** Defaults are off. Thresholds were picked offline: τ1=0.10, τ2=0.15.
- **P1 prefill: done, far below plan.** ctx2k (2,276 tokens) TTFT went 523 -> 404 s (-22.7%); the plan expected 5-6x. Prefill is now bound by CPU attention plus the CPU MoE union.
- **V4 outputs are byte-identical** as far as provable without V4 weights.
- **Nothing is committed and nothing is installed.** app\ still runs the 11-07 CPU-only V4.1 engine.

## 2. Today cut and follow-ups
| Package | Status | Planned | Measured / evidence |
|---|---|---|---|
| M0 | done | — | m0.md §6 (R0a/b/c, run by the main session after the lane could not run them); drive probe done |
| G1 | done | -100 to -190 ms/token | -143 ms/token (-15.6%) |
| G2 | done | 1.1-2.5 tok/s | 1.53-2.27 tok/s (W3 WEX1 arm). Done criterion "GPU-direct 0 vs 2 token-identical" checked by me: see below |
| P1 | done | TTFT 5-6x | 1.29x. Chunk invariance on the fixture passes |
| M0 decprof | done | — | in G1 |
| W1 | done | -12.8% bytes | -10.03% (129.9 vs 144.4 GB); 7680/7680 records exact; 64 records byte-identical through the GPU batch decode |
| G3 | done | -25 to -30 ms/token | -39 ms/token (+8%); 403/403 head calls bit-identical; 1.23 GiB VRAM |
| F1 code | done | — | unit tests pass; defaults off |
| Integration | done | — | 5.1 gates green, REAP 5.2 run |
| W2 | done | — | ≈ -3 ms/token (batches of only 1.56 / 1.83 records) |
| W3 | done | — | WNR2 -1.80% ± 0.65% |
| F1 grid | NOT run | — | cancelled when the main session took heavy.lock at 05:00 |
| W3 pilot C | NOT run | — | same cancellation |
| D1 | superseded by user decisions | — | WEX1 copied to U: on 11-07; D: copy deleted 05:02 today; WNR2 U:+C: mirror being built now (approved by the user, 2026-11-09); effect not measured |
| G8 / S1 / R1 / G4-G7 | not started | — | deferred in the plan |

GPU-direct 0 vs 2 check: the greedy text of w0/code1/chat_it1 is equal across:
- G2 A2/A3 and integ I (GPU-direct 0);
- G2 B/B3 and W3 A1/B1 (GPU-direct 2).

## 3. Numbers vs M0
**TTFT.**
- Short prompts: 11.8/10.7 s (M0) -> 9.7-12.0 s.
- ctx2k: CPU 519.7 s, GPU before P1 523.4 s, GPU with P1 404.5 s.

**Decode token (W3 WEX1, 534 ms).**
- Expert wait 312 ms (58%, the disk).
- MoE GPU compute 94 ms.
- Attention 74 ms.
- mHC 21 ms, attach 6 ms, head about 1.5 ms.
- For comparison, the CPU split: attention 140, FFN 716 ms.

**Disk and caches.**
- 80.7 reads and 1.52 GB per forward.
- 934-1,028 VRAM mirrors plus 360 RAM slots.
- Board VRAM peak 27.6 of 32.6 GB.

**Predictor.** Recall 76.4% at k=8. Precision 57.3% at depth 1, 73.4% at depth 2.

**RAM.** GPU arms: working set 15.4 GB, private bytes 42 GB. Lowest available RAM on the machine: 2.7 GB.

## 4. What I verified today on the final tree
- src is byte-identical to ws-ab\c (W3's build) and ws-w\c (W2 final). It differs from ws-integ only in W2's 4 files, and the backend .cu is unchanged since the integration's DLL bit-compare.
- **deepseek-v4-tiny-check + v41-on-v4-check** in ws-final: exit 0, 60 PASS, 0 FAIL. 19 warnings, the same count as W2's build.
- **V4 engine, old (79aff2b, ws-m0 exe e215651b) vs final (671886f0), CPU, V4 tiny fixture:** PASS 36/36 (6 cases x 6 variants), logits byte-identical.
- **The same with the GPU tier opened** (DG DLL): PASS 36/36. Caveat: oracle mode uploads nothing, so this covers the open and routing paths only. The GPU kernels are covered by the DLL bit-compare.
- **p5 (V4.1 CPU vs ref_vendor.json):** PASS. Greedy 8/8, max row rel 1.67e-5.
- **fixture_g2 (V4.1 tiny GPU vs CPU):** both DLLs, plain plus prewarm 8 and 33. 32 tokens identical, logits diff 0, 186 MoE checks with rel_l2 0, cpu_layers 0.
- **Python tests of the non-lane edits** all pass: mcp_server, experiment_manifest, glm_usage_path, tool_args_literal, windows_macro_names, vulkan_engine_counts, and openai_server (462 tests).
- **heavy.lock** is held by the main session's mirror_uc_setup.ps1: PID 17032, Robocopy live, WNR2 E:->C: copy started 05:10. It is live, not stale, and I left it alone. heavy.lock.stale-0447 is only an inert record.
- **Git index** is empty: nothing staged.

## 5. Not done or not verified
**Measurements never run:**
- F1 grid.
- W3 pilot C (WNR2 RAM_GB 12 plus a 4 GB compressed RAM cache, vs WEX1 at 16 GB).
- The U:+C: mirror.
- P1's pool and chunk effects separately (arm Q changed both at once); the plan's 800-token run (replaced by the 2,276-token one); RAM use at long context.
- The per-token wexprof line from the plan (it does not exist; totals print at exit only).
- VRAM used by W2's batch lanes (about 2 GB [I], outside the reserve accounting).
- Protocol 3.1 conditions: power plan was Balanced and Defender exclusions were unverified. Only W3 used ABBA; G1, G2, G3 and the integration ran one run per arm.

**Tests not run or weak:**
- dsv4-cuda-test (Linux make target) never ran; the same tests ran through the loader.
- No unit tests for these W2 paths: overlapped read issue/wait, the compressed RAM record cache inside a batch, the feature-8 guard, the twin turned off under F1 gating.
- No unit test for F1's WEX-direct branch.
- The 128-block arm of gpu_refusal never reaches a real backend init.
- The 64 GiB probe in the head test reads past a host array on a GPU with 64 GB or more.
- The prefill_chunk test runs past the fixture's max positions and accepts any exit code.

**Claims resting on an argument:**
- REAP GPU vs CPU greedy tokens diverge early on 3 of 4 prompts:
  - code1 at token 5 (margin 0.17%), chat_it1 at token 8 (6.5%), ctx2k at token 5 (2.4%).
  - G2 justified this with one alternative CPU build as a noise floor, and code1 sits at 2.05x that floor.
  - ctx2k has no noise floor.
  - Per-layer checks ran on w0 only.
- The P1 log prints pool=host even on the WEX1 store, where the reviewer says the pool is a no-op. I did not verify this.

## 6. Risks
- **R1. Backend host_pinned bug.** It registers the whole VirtualQuery region (cu:3449). It affects V4 too and is unfixed.
  - Tiny fixture with 8 slots: 22 of 30 runs crash with the copy worker on, and 8 of 30 give wrong logits with it off. With DSV4_CUDA_PIN_HOST=0, both are 0 of 30.
  - ok() also leaves the sticky CUDA error set.
  - The V4.1 profile sets V4_COPY_WORKER=1 and DSV4_CUDA_PIN_HOST=1. REAP was clean only because 18.8 MB slabs are their own regions.
- **R2. V4 latent mirror-cache bug** (added entry at resident_count=found). Unfixed; it needs OK, and the fix is one line.
- **R3. Long prompts.** About 400 s TTFT at 2.3k tokens, and no V4.1 prefix reuse (S1), so every KAIRA turn re-prefills.
- **R4. RAM pressure.** The user removed the host RAM guard on purpose. The engine silently budgets min(--memory-gb, available RAM) at open.
- **R5. V4 changes not behind is_v41.** Outputs are identical, but timing and work change:
  - release() broadcasts on every lease release;
  - the VRAM reserve is re-read on every miss;
  - F1's predictor routes max(K,topk) deep even with gating off (deepseek_v4.c:8191), so V4 depth 2 now routes 6 deep instead of 4.
- **R6. Smaller items:**
  - g_act_group is never reset at init or shutdown;
  - `if(!said++)` at cu:3115 overflows after 2^31 calls;
  - the GPU head upload ignores the VRAM reserve;
  - route-tap prefill frames over 128 rows are dropped;
  - W2 frees VRAM mirrors when its staging ring runs short (169-195 per run), and its batch failures are not counted;
  - 3 missing-field-initializer warnings in the W2 test hook.
- **R7. The environment changed after every measurement:**
  - WEX1 on D: is deleted;
  - engram shard 47 is now on D: (PCIe 3.0 x2), reached through a junction at the old C: path;
  - WNR2 is being copied to C:.
  - Re-baseline before the next A/B.
- **R8. KAIRA cannot use the container as configured:**
  - kaira-profile-v41-reap.json has no COLI_WEX_PATH, so KAIRA would read safetensors, a path never measured on the GPU;
  - DeepSeekV41Preflight.cs:241-244 still warns that no V4.1 WEX container exists (stale);
  - V4_PREFILL_CHUNK=128 no longer affects V4.1, which uses V41_PREFILL_CHUNK (default 1024).

## 7. Repository state
- HEAD is 88ddb07 (a main-session commit; it does not touch deepseek_v4.c).
- 26 modified files (+2,902 / -429) and 7 new test files, in three groups:
  1. **V4.1 GPU lane work:**
     - Makefile, Makefile.deepseek-v4, backend_cuda_dsv4.cu/.h, backend_loader_dsv4.c, dsv4.def;
     - deepseek_v4.c (+1,987), deepseek_v4_internal.h;
     - tests: v41_on_v4 gpu_refusal / moe / schema / union_slots, np_queue, parity harness;
     - new tests: fp8_block32_cuda, head_cuda, prefill_pool, prefill_chunk.
  2. **Main-session upstream ports (11-08 16:39):**
     - backend_vulkan.c, colibri.c, experiment_manifest.py, mcp_server.py, openai_server.py and their tests;
     - 3 untracked tests.
  3. **The user's host RAM-guard removal:** src\host\*.cs.
- The lane patch files are records only; src is the source of truth.
- **app\:** no lane touched it. Engine and DLLs date from 11-07 00:46. KAIRA.exe was rebuilt 11-08 01:03, kaira.json and the logs changed when KAIRA ran at 01:17, and openai_server.py changed 11-08 17:11.

## 8. Leftovers (delete only with OK)
- lab\v41-port-20261106\ws-rev-* (12 review workspaces): 19.3 GB.
- ws-g, ws-integ, ws-w, ws-c, ws-d, ws-gv2, ws-final: 9.2 GB.
- Keep: ws-ab (the measured build), ws-m0 (79aff2b baseline exe), ws-int (DLL recipes).
- Temporary dumps: about 13 GB (w3dump 6.4, g2dump* 2.5, abdump 1.1, g3dump 0.8, integdump 0.5).
- The original W1 container on E:: 130 GB. E: has 15 GB free.
- The workspaces sit under OneDrive.

## 9. Next steps
1. Wait for mirror_uc_setup.ps1 to finish. It must write the C:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex.verified marker and log "lock released" in moves.log.
2. Re-baseline WEX1 in the new layout (run_w3.sh, arm A only) and compare with W3's 1.898 tok/s.
3. Under protocol 3.1 (ABBA x2, --memory-gb 16), measure WEX1 on U: against WNR2 striped over U:+C: (with V4_WEX_BATCH=1), plus pilot C (ab\run_w3c.sh). Apply the user's rule.
4. Run the F1 grid (f1\run_f1.sh) on the chosen container. Keep a setting only at ≥ +3% with the CI excluding 0.
5. With OK, the fixes that touch V4 or the backend:
   - R1: pin per allocation, and clear the sticky error. Repro: r3_race41.py with COLI_V4_EXPERT_SLOTS=8.
   - R2: the one-line V4 mirror-cache fix.
   - R6: reset g_act_group, fix said++.
   - Gate F1's deep route on gating or stats.
   - Test hygiene items from section 5.
6. Host and profile: add COLI_WEX_PATH (and the mirror list); fix the stale preflight warning; set V41_PREFILL_CHUNK explicitly; keep V4_GPU_HEAD=1, DSV4_CUDA=1 and RESERVE 4096.
7. Next speed levers: disk striping, F1, then R1-plan (drop the host dense copies, about +6.6 GB of RAM). For long prompts: S1, then G8, then P1b. Later: G5-G7.

## 10. Needs the user's OK
- **Commit**, as three separate commits (lane work / upstream ports / host guard).
- **Install into app\:**
  - back up app\ first;
  - build exe plus both DSV4 DLLs from the committed tree, or ws-ab after a hash check;
  - add kaira_wex_cuda.dll only if WNR2 is chosen.
- **app\kaira.json V4.1 profile:** COLI_WEX_PATH and mirrors, V41_PREFILL_CHUNK, V4_WEX_BATCH only with WNR2, F1 gates off. This needs the host fix in step 6, or the preflight warns wrongly.
- **Disk layout:** whether to keep the E: copy of WNR2, and whether engram 47 stays on the slow D: link.
- **The V4-touching fixes** in step 5.
- **Deleting the leftovers** (about 30 GB in OneDrive plus 13 GB of temporary dumps).

## Open issues (from the final agent)
- F1 grid and W3 pilot C never run, and neither has the U:+C: WNR2 mirror; every REAP number predates today's file moves (engram 47 now on D: through a junction).
- Backend host_pinned whole-region registration bug (V4 and V4.1), plus the sticky CUDA error in ok(); the V4.1 profile has copy worker and pin host on. Fix needs user OK.
- V4 latent mirror-cache bug (resident_count=found) is unfixed; the fix is one line and needs user OK.
- V4 timing/work changes not behind is_v41: release() broadcast, VRAM reserve re-read on every miss, F1 deep route (depth 2 routes 6 deep with prefetch on).
- KAIRA profile lacks COLI_WEX_PATH; the preflight warning in DeepSeekV41Preflight.cs:241-244 is stale; V4_PREFILL_CHUNK no longer affects V4.1.
- dsv4-cuda-test (Linux make target) never ran; W2 batch internals and F1's WEX-direct branch have no unit test; the gpu_refusal 128-block arm has no real coverage.
- GPU-vs-CPU early REAP token flips accepted against a single noise-floor sample; ctx2k has none.
- Nothing committed, nothing installed: commit split, app install, kaira.json profile, disk layout and about 43 GB of leftovers all need the user's OK.
