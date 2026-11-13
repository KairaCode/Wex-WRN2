# DeepSeek V4.1 REAP-192 on the GPU tiers of deepseek_v4.c: plan (2026-11-07)

> **UPDATE 2026-11-07 ~02:00 from the main session (user-approved file moves; read this first).**
> - M0 found D: on a PCIe 3.0 x2 link (1.6 GB/s); U: reads at 6.6 GB/s. The user approved:
>   - a copy of the WEX1 container to `U:\kaira_wex\v41-reap\v4_flash_experts.wex`. It is ready only when `U:\kaira_wex\v41-reap\v4_flash_experts.wex.verified` exists (it holds the sha256). Until then use the D: copy. The D: copy stays until this workflow ends.
>   - the old V4 checkpoint `U:\Models\DeepSeek-V4-Flash-0731` moves to `H:\models\DeepSeek-V4-Flash-0731`. A junction keeps the U: path working, but reads come from the H: HDD.
> - The main session does these copies under `heavy.lock` (label "main-session file moves"), so wait for the lock as usual.
> - Once the `.verified` marker exists, REAP GPU runs (integration, W2) should prefer the U: WEX1 and record which container they used.
> - **W3 (A/B):** put both containers on the same drive, U:. If `U:\kaira_wex\v41-reap\` has no verified WNR2 copy, copy the W1 container from E: to U: yourself under heavy.lock (U: should have about 269 GB free after the moves; check first). Verify it by size and sha256, write a `<name>.verified` marker, then run the A/B on U:. If space or time does not allow it, run it as planned and report the drive confound.
> - The engram shard 47 stays at `C:\models\DeepSeek-V4.1-Flash-1017-engram` for this whole workflow. It moves to D: only after the workflow ends, so keep `COLI_MODEL_DIRS` as given.
>
> **UPDATE 2026-11-07 ~03:30: M0 REAP CPU baseline is now MEASURED:** see `m0.md` section 6. The reference is WEX1 on U: at ~1.12 tok/s decode and ~11 s TTFT (30-token prompts); safetensors on U: run at ~0.99 tok/s, and WEX1 on D: at ~0.47 tok/s. Use the U: WEX1 for every REAP run and compare against those numbers. Also: **WEX1/WNR2 code is KAIRA-only and never goes upstream** (user decision).
>
> **UPDATE 2026-11-07 ~21:00, RAM rule override (main session):** with the user's normal apps open, the machine sits at about 14 GB of available RAM, so the 18 GB rule in the prompts can never be met. For every REAP run, proceed when available RAM is at least 15 GB with `--memory-gb 16`. If it stays below 15 GB for 10 minutes, run every arm of the comparison (CPU reference and GPU, or WEX1 and WNR2) at `--memory-gb 13`, and report that you did. The heavy.lock rule is unchanged. **Confirmed by the user on 2026-11-08: 18 GB of free RAM will never be available.** Never wait for 18 GB and never skip a run for lack of it. Use this rule, and keep the same `--memory-gb` in both arms of any A/B.

> **UPDATE 2026-11-08 21:15, the user's decision on the container:** the first same-drive A/B on U: (`ab-samedrive-20261108`, A1 vs B1) shows WNR2 about +2-3% faster than WEX1. Owner decision 2026-11-08 21:15: keep WNR2 if token ids are identical and it is not slower. WNR2 is KAIRA's V4.1 container even below the +3% threshold of section 3.2, provided the full ABBA shows identical token ids and WNR2 is not slower. W3 and the final report must apply this rule, not the old "+3% with a CI excluding 0".
>
> **UPDATE 2026-11-07 ~17:00, second upstream review** (`RIEPILOGO-REVIEW-20261107b.md`):
> - **Integration:** `coli_v4_gpu_fp8_matmul_batch` still has no 512-row cap per call. Add it when integrating P1.
> - P1's bit-identity test must be relaxed once the prefill MoE moves to the GPU (G8).
> - **Possible "P1b":** layer-major V4.1 prefill, as in upstream e1219b30d. It works if the attention list and candidate mask are kept per row for the whole segment. Upstream measured disk reads 98.9 → 44.3 GB on a 4k prompt. Evaluate it after P1.
> - Upstream `cuda_dsv4c.cuh` (#2673) is the reference for G5/G6. It needs KAIRA's FP8/FP4 quantization added.
>
> **UPDATE 2026-11-07 ~02:30: findings of the upstream PR review** (`<HOME>\colibri_pr_work\review\RIEPILOGO-REVIEW-20261107.md`, which has the details).
> - **G2, mandatory:** `V4_ROUTER_AVX2=1` is on in every KAIRA profile, and the AVX2 router accepts E=192. Without an `is_v41` guard, V4.1 would be routed with V4's normalisation. Guard it (V4.1 → its own router) and add a test.
> - **P1:** the estimate at line ~397 is wrong. A 1024-row `matmul_batch` takes about 160 MiB of scratch, not about 40 MB. Cap each call at 512 rows (the chunk may stay larger if it is split per call).
> - **G3:** reuse the exact GPU head of upstream #2667 (logits bit-identical to the CPU head; opt-in), and its `matvec_grouped` format guard in lane A.
> - **Avoid** porting upstream #2665/#2666/#2670 here: they rewrite the same functions as G1/G2/W2, and #2666 makes engine open fail when shared-expert mirrors are missing.
> - **Residual KAIRA bug (from #2668):** a failed mirror upload leaves the expert registered with no data, and that expert then always runs on the CPU. The fix is about 10 lines; do it after W2, or in G2 if you are touching that code anyway.

This is a plan. No source was edited to write it. Base: git `master` at 79aff2b, clean `src/`. Never commit, push, deploy to `app\` or edit `app\kaira.json` without the user's explicit OK.

Line anchors are `file:line` under `src\colibri\c\` unless a path is given (`cu` = `backend_cuda_dsv4.cu`, `ld` = `backend_loader_dsv4.c`, `v4` = `deepseek_v4.c`). Tags: **[M]** measured, **[R]** read in the code today, **[I]** inferred or estimated, to be checked.

Sources read: `lab\v41-port-20261106\HANDOFF.md` and `phase1-carryforward.md`; `lab\wex-prod-20261105\` (REPORT, the v41 plan and gap, both reports); the memory notes in `docs\handoff\memory\`; the REAP smoke and WEX1 repack logs; and the code below. Three read-only sub-agents mapped the CUDA backend, the WEX/WNR2 path and the MoE/prefetch path. I re-checked the anchors I build on.

---

## 0. Where we start, and the one fact that sets the order

**State [R/M].**
- V4.1 runs CPU-only because the whole CUDA tier refuses to open: `coli_v4_gpu_engine_open` returns early when `fp8_block != 128` (v4:16533-16539). As a result there are no dense mirrors, no expert mirrors, no attach and no prefetch.
- The REAP smoke (`lab\v41-port-20261106\smoke-reap-fixed-20261107.log`, `--memory-gb 16`, a 13-position run on a cold cache) read safetensors from U:, not the WEX1.
  - Hit rate 10.9%: 1,612 misses, 30.3 GB read.
  - TTFT 9.67 s for 9 prompt tokens, then 1.44 s/token.
  - RAM tier: 9 slots per layer (6.30 GiB), plus 6.65 GiB of resident dense.

**The fact [I, from those numbers].** V4.1 decode and prefill are **disk-bound** today.
- Each miss is an 18.8 MB record. About 214 misses per decode token is about 4 GB/token, which is about 1.4 s at 2.5-3 GB/s. CPU compute is the smaller term.
- So the GPU work pays in two ways:
  - it moves the CPU compute off the critical path (dense ~6.5 GB of FP8 per token, experts ~4.5 GB of FP4 per token);
  - more importantly, it adds a ~19 GiB VRAM expert tier, about 1,050-1,200 mirrors, on top of the RAM tier, together with V4's prefetch, copy worker and promotion machinery.
- Everything after that is about bytes from NVMe: prefetch budget, WNR2 capacity, a second drive.

**Prefill is worse than decode [R + I].**
- V4.1 prefill runs chunk-major with chunks clamped to 128 rows (`v4_prefill_chunk_width`, v4:21878-21886; `v41_target_batch`, v4:21894-21934).
- Every 128-row chunk sweeps all 40 layers. A 128-row union touches ~150-190 of the 192 experts per layer, so a chunk reads ~100+ GB of experts [I]. That puts a 1k-token prompt at about 5 minutes of TTFT on one NVMe [I].
- V4 avoids this with a layer-major prefill and the GPU bank.

---

## 1. Work packages, ranked

Effort: **S** < 2 h, **M** about half a day, **L** more than a day (Opus agent plus review). Speed numbers are [I] unless marked; M0 replaces them with measurements.

### M0. Baseline and a per-token profile for V4.1 (S; enabler; first)

- **Goal.** Get warm numbers instead of the cold 13-position smoke, and see where V4.1 time goes per token.
- **Code (S, about 20 lines, v4 only).**
  - `decprof`/`attnprof`/`moeprof` (`DSV4_DECODE_PROF`, v4:9883-9935) never run for V4.1, because `block_token_pipeline` returns at v4:9806-9808 before them.
  - Add the same timers around the two sublayer calls in `coli_v41_block_rows` (attention v4:~6301, FFN v4:~6317), under the existing env check. Print `decprof` at layer 0 from the V4.1 branch at v4:9806.
  - V4 is untouched: the branch is only on `is_v41`.
- **Runs (no code; quiet machine, see 3.1):**
  - R0a: CPU, safetensors from U:.
  - R0b: CPU, `COLI_WEX_PATH=D:\kaira_wex\v41-reap\v4_flash_experts.wex` (this container has never been used by a run yet).
  - Each run: `--memory-gb 16`, the same 4 prompts, 1 warm-up prompt plus 128 greedy tokens. Record the PROF line (`v4_prof_emit`, v4:24200, serve mode), `v4_tokens`, `v4_direct`, `decprof`.
  - Drive probe on C: D: E: U: with the engine's own pattern: 8 threads, aligned unbuffered reads of 18.8 MB and of ~16.4 MB (the WNR2 record size), 30 s cold per drive. Write the script to `lab\v41-gpu-20261107\drive_probe.py`; model it on `v4_mirror_probe_bw`, v4:371-404.
- **Output.** Warm s/token, misses/token, GB/token, effective GB/s per drive. These decide D1, the A/B setup (W3) and the F1 grid.
- **Test.** `make PYTHON=python deepseek-v4-tiny-check` stays green (21 PASS plus `v41-on-v4-check`).

### G1. Open the GPU tier for V4.1; dense FP8 with 32x32 blocks on the GPU (M; goal a; today)

- **Goal.**
  - All FP8 dense linears run on the GPU through the existing per-call "dense-matvec" tier: wq_a, wq_b, wkv, wo_a, wo_b, and the shared expert w1/w2/w3. That is 162 MB per layer, 6.48 GB for 40 layers.
  - V4.1's attention *core* stays on the CPU: compressor, shared KV, indexer, candidate mask, sparse attend, and the FP4 qdq of the latent and the index keys. V4.1 decode already reaches the GPU through `ColiTensorView.gpu`:
    - `v41_fp8_rows` (v4:3168) calls `coli_fp8_matvec_ref` (GPU branch v4:27545) for one row, or `coli_fp8_matmul_batch_ref` (GPU branch v4:~27838) for several.
- **Numerics (D-1).** The CUDA dense kernels take the activation raw (cu:2717 `matvec_grouped`, 2761 `matvec_pair`, 2893, 2940). On V4 that is the production behaviour; keep it.
  - For V4.1, quantize **on the host** before upload with the existing `coli_fp8_activation_qdq_ref(out, scales, in, columns, block)`. Its output is already dequantized (`e4m3 * scale`, v4:26761-26815).
  - Feeding that into the raw-input kernel gives exactly the vendor's `act_quant(x, 32)` input. Only the order of the sums differs from the CPU.
  - No kernel-side activation change is needed for dense.
- **Backend (lane A), about 50 lines of .cu plus about 12 of glue:**
  1. Add `int blk;` to `Dsv4CudaTensor` (cu:53). `upload()` sets it to 128. The host never reads the struct, so the ABI is unchanged.
  2. Add `template<int B>` to the scale-index expressions in `mv` (cu:108), `mv_warp8` (cu:118-133), `mv_pair_warp8` (cu:174-193), `mv_rows` (cu:208), `mm_batch` (cu:300) and `mm_batch_w` (cu:395). Dispatch on `t->blk` in `run_mv` (cu:271; the gate `(I&127)==0 && I<=8192` stays, and every V4.1 size passes it), `launch_mv1` (cu:2752), `matvec_pair` (cu:2761), `proj_batch` (cu:2893) and `matmul_batch` (cu:2940). Also cover the grouped per-group scale offset used by `matvec_grouped`.
  3. Add a new export, `int dsv4_cuda_upload_fp8_ex(Dsv4CudaTensor **t, const void *w, const void *scale_e8m0, int O, int I, int device, int bf16, int blk)`. It sizes scales as `ceil(O/blk) * ceil(I/blk)` and skips the DeepGEMM pack (cu:2111-2113, 529) when `blk != 128`. Do not edit the existing `upload_fp8`/`upload_fp8_bf16` (cu:2118/2119).
     - Copy with `cudaMemcpyAsync` on `c->stream` plus a drain, the same as the Oct-20 upload-race fix.
  4. Add a prototype in `backend_cuda_dsv4.h` and a line in `dsv4.def`. In `ld`, add an optional `GetProcAddress` (ld:579-614, not `RESOLVE`) and a wrapper; `dsv4_cuda_fp8_block_available()` is true when the symbol exists.
     - Both DLLs build from the same `.cu`/`.def` (Makefile:1065, 1078). Production loads `coli_cuda_dsv4_dg.dll` first (ld:441), so test both.
- **Engine (lane B), about 60 lines of v4:**
  - At v4:16533-16539, move the `fp8_block != 128` check after `dsv4_cuda_init` (v4:16543). Refuse, with the same log, only when `!dsv4_cuda_fp8_block_available()`.
  - In `v4_gpu_upload_fp8_fmt` (v4:16634):
    - replace `columns % 128` (v4:16650) with `% block`, where `block = weights->plan.fp8_block`;
    - re-encode the E8M0 grid at `ceil(rows/block) * ceil(cols/block)` (v4:16670), since resident scales are one f32 per block;
    - call `_ex` when `block != 128`.
  - In `v4_gpu_upload_fp8_ref` (v4:16774), refuse `block != 128` explicitly. It is only used by the V4 indexer fast path.
  - In `coli_v4_gpu_layer_upload` (v4:16875):
    - skip the attention-batch block (v4:16904-16975) when `weights->plan.is_v41`. Its compressor/indexer/mHC shapes are V4's, and uploading them only wastes VRAM;
    - leave the gate/bias uploads alone; they refuse E=192 with a one-time log (v4:16836, 16867).
  - In the four wrappers `coli_v4_gpu_fp8_matvec` (v4:16998), `coli_v4_gpu_fp8_matvec_pair` (v4:17007), `coli_v4_gpu_matvec_grouped` (v4:17127) and `coli_v4_gpu_fp8_matmul_batch` (v4:17140): when `w->block_columns != 128`, host-qdq each input row per block into a static scratch, then call the backend.
    - Decode and prefill are single-threaded, as these wrappers already assume.
    - Per-group qdq of the grouped input is the same as qdq per 32 of the whole row, because the group width 4096 is a multiple of 32.
  - V4.1 wo_a (v4:3632-3650) sets `view.gpu = NULL` per group, so wo_a stays on the CPU. For `rows == 1 && wo_a.gpu`, call `coli_v4_gpu_matvec_grouped(&wo_a, oa, att, groups)` and then `v41_round(oa, ...)`, the same bf16 output rounding as `v41_fp8_rows`. Keep the per-group CPU loop as the fallback.
  - Update the `v41_attn path=cpu` log (v4:3379) to say that the projections go to the GPU when mirrors exist.
- **What stays byte-identical for V4.** The B=128 instantiations are the same integer expressions. The old exports are untouched, and `blk` defaults to 128. Every engine branch keys on `block_columns != 128` or `is_v41`. Prove it with the bit-compare in the test plan; do not assume it.
- **Expected impact [I].**
  - CPU dense FP8 today is ~6.5 GB per token through AVX2, about 130-220 ms.
  - On the GPU it becomes ~280 round trips × ~60-100 µs under WDDM, plus ~4 ms of kernels, about 20-30 ms. V4 measured 4.6 + 5.7 + 7 ms for qa, qb and wo_a over 43 layers.
  - Net: **-100 to -190 ms/token**. That is 10-15% of today's 1.44 s, because the disk dominates.
  - Prefill: the dense matmul for 128-1024 rows goes from CPU-bound (13 TFLOP per 1k tokens on the CPU [I]) to seconds.
- **Tests.**
  - *Backend unit*, new `tests\test_dsv4_fp8_block32_cuda.c` (add it to the `dsv4-cuda-test` rule, Makefile:1255):
    - random E4M3 weights and E8M0 32x32 scales at the V4.1 shapes 1280x5120, 32768x1280, 5120x8192, 2304x5120, 5120x2304, and grouped 8x(1024x4096);
    - the input is host-qdq'd per 32;
    - compare against `coli_fp8_matvec_ref` on the CPU: relative error ≤ 1e-5, since only the order of the sums differs.
  - *Prefill path*: the same inputs through `matmul_batch` at 1, 7, 128 and 1024 rows.
  - *V4 bit-compare*: a lab harness writes the outputs of `matvec`, `matvec_pair`, `matvec_grouped`, `matmul_batch`, `wo_decode`, `expert_group` and `dsv4_cuda_moe` at V4 shapes, under `COLI_DSV4_DLL` set to the old and then the new DLL (generic and DG). `cmp` must show identical bytes. Then run `dsv4-fused6-test` (Makefile:1266), `dsv4-cuda-loader-test` (Makefile:1092) and `tests\test_backend_loader_dsv4_parity.py`.
  - *Engine*:
    - `tests\test_v41_on_v4_gpu_refusal.c` asserts the refusal. Change it to: refused with the old DLL, tier up with the new one (it runs under `v41-on-v4-check`, Makefile:2693).
    - `deepseek-v4-tiny-check` must stay green.
    - The tiny V4.1 fixture (`lab\v41-port-20261106\fixtures\v41-tiny`, dim 128) with `DSV4_CUDA=1` against `DSV4_CUDA=0`: greedy tokens identical over 32 steps, and log the top-2 logit margin at every step where they diverge. This fixture exercises the generic (non-fast) kernel paths.
  - *REAP smoke* on the GPU: the same invocation as the fixed smoke, plus `COLI_WEX_PATH`. Check that the `v4_gpu layer=N uploaded=` lines total about 6.5 GB, and that the first 32 greedy tokens match a CPU run of the same prompt (margin log as above).

### G2. V4.1 decode MoE through `moe_token_pipeline` (M-L; goal a; today; the big decode win)

- **Goal.** A decode token of V4.1 uses V4's GPU MoE machinery: VRAM mirror cache, loader lanes with `vram_first`, `v4_np_predict` prefetch, copy worker, promotion, async attach, and WEX GPU-direct with the RAM record cache.
  - (b), (c) and (d) all live on this path. None of them can be tested on V4.1 until this lands. V4 weights are gone, so V4.1 is the only real target.
- **Why the reuse works [R].**
  - `moe_token_pipeline` (v4:8990) takes E, H and topk from config (v4:8999-9001), and has no 256/4096/2048 literal in its body (scanned v4:8990-9710).
  - The mirror cache is shape-generic: the record is `elements/2 + elements/32`, i.e. 18.8 MB for V4.1 (v4:17950).
  - `dsv4_cuda_expert_group` (cu:3013) takes H and I from the tensors.
  - `dsv4_cuda_moe` (cu:3186) skips its fused path unless `H==4096 && I==2048` (cu:3201) and falls through to the batched path.
- **Engine (lane B), about 60 lines of v4:**
  1. Add `const int *tokens` to `V41BlockContext` (v4:6337-6344) and to `v41_block` (v4:6368). The four callers already hold the tokens: `block_token_impl` (v4:6502), `block_token_pipeline` (v4:9806, `token`), `coli_v4_block_window_batch_ref` (v4:10704) and `coli_v4_block_window_rows_ref` (v4:10897).
  2. In `v41_ffn_sublayer` (v4:6360), when `rows == 1 && !c->positions && c->tokens && c->experts->gpu`, return `moe_token_pipeline(branches, weights, config, experts, normalized, tokens[0])`.
     - That needs a forward declaration. If the two sit in different `COLI_V4_UNIT_*` blocks, add an internal non-static wrapper in `deepseek_v4_internal.h`.
     - Multi-slot decode (`positions != NULL`) and prefill keep the CPU union (v4:10614).
  3. Router (v4:9058-9090), when `is_v41`:
     - no `cpu_exact` (`V4_ROUTER_AVX2` uses V4's normalisation, `v4_np_route.h:62`);
     - no `coli_v4_gpu_route` (it refuses E≠256 at v4:17243);
     - call `coli_v41_route_bf16_logits(route_weights, indices, input, raw_gate, bias, NULL, n, d, topk, scale, NULL)` (v4:15971; `+1e-20` normalisation per vendor `model.py:826-827`).
  4. Routed-expert and shared-expert compute on the GPU, where `gpu_compute` is set (v4:~9323) and at the calls `dsv4_cuda_moe` (v4:~9495/9515) and `dsv4_cuda_expert_group` (v4:~9525, 9620). When `is_v41`:
     - require `dsv4_cuda_act_group_available()`;
     - pass a host-qdq'd copy of `input` (per `fp8_block`), which is the vendor `act_quant(x, 32)` for both the FP4 routed and the FP8 shared FC1 input, instead of the raw `input`;
     - otherwise keep today's CPU fallback.
     - Leave the `coli_fp4_act_block == 128` guards at v4:27373 and 27639 unchanged: the union's CPU FP4 path stays CPU.
  5. `V4_NP_ONLINE` turns itself off for E≠256 (v4:7800). The learned gates file is rejected on its dimensions (`v4_np_gates.h:22-29`), so the predictor falls back to the model's own next-layer gate (v4:7814). No change is needed; log it once.
- **Backend (lane A), about 25 lines.**
  - The FC2 input (the SwiGLU output) and the shared-expert intermediate are quantized on the device by `fp8_sim`, which uses groups of 128 (cu:513). Today's launch sites:
    - `expert_group`: cu:~3049 (batched), ~3054 (per-expert);
    - `dsv4_cuda_moe` batched path: cu:3229-3241 and 3268;
    - the shared expert: cu:3237/3288;
    - `dsv4_cuda_expert_fp8`: cu:3058ff.
  - Make it `template<int G>`, add an optional export `void dsv4_cuda_set_act_group(int g)` (default 128), and switch only those launches to the global group.
  - V4 never calls the setter, so G stays 128 and the code is identical.
- **Not done here (stays on the CPU, small cost).**
  - Router GPU (G4): about 2 MB of bf16 per layer, ~3 ms/token on the CPU.
  - mHC on the GPU (G5): ~157 MB/token of f32 mixes, ~5-10 ms [I].
  - Engram: 2 layers, rows read on the host.
  - Attention core.
- **Expected impact [I].**
  - VRAM budget (section 4.1): ~1,050-1,200 mirrors (14-16% of the 7,680 experts), on top of 335 RAM slots at 16 GB, so ~19% of all experts are cached against 4.7% today. Prefetch overlaps disk with compute.
  - V4 reached 72% VRAM hits with 14% of its experts in VRAM [M, Oct]. REAP pruned the cold half of the experts, so routing is probably flatter [I]: assume 45-65% total hits warm.
  - That gives ~85-130 misses/token × 18.8 MB = 1.6-2.4 GB/token, i.e. 0.4-0.9 s at 2.5-4 GB/s on one drive.
  - **G1+G2: about 1.1-2.5 tok/s against 0.69 tok/s now (1.6-3.5x)**. NVMe stays the bound; the M0 and post-G2 counters will pin the number.
- **Tests.**
  - *Backend*: an `expert_group` test at E=192 shapes (H=5120, I=2304, count 6) with a host-qdq'd FC1 input and G=32, against the CPU `coli_fp4` path with `coli_fp4_act_block=32`; relative error ≤ 1e-5. Run the V4 bit-compare again (G=128).
  - *Engine*: run the tiny fixture with `DSV4_CUDA=1` as in G1, and check that `moeprof` shows GPU compute.
  - *REAP*: 256 greedy tokens on the GPU, with
    - (i) `COLI_WEX_GPU_DIRECT=0` (host slabs, attach),
    - (ii) `=2` with `COLI_WEX_RAM_CACHE_GB=4`;
    - tokens identical between (i) and (ii), since the same bytes go through the same kernels. Compare against a CPU run (margin log).
  - *Counters*: `moeprof`, `attachprof` (`V4_ATTACH_PROF=1`), `promoteprof` and `copyprof` now print for V4.1, because decode goes through the pipeline.
  - *V4*: `deepseek-v4-tiny-check`, unit tests, `dsv4-fused6-test`, bit-compare.

### P1. V4.1 prefill: borrow the whole cache, and chunks larger than 128 (S-M; TTFT; today)

- **Goal.** One expert sweep per *large* chunk instead of one per 128 rows.
- **Engine (lane C), about 30 lines of v4, all keyed on `is_v41`:**
  1. Add `coli_v4_expert_store_prefill_pool_host(store, layer)` next to `coli_v4_expert_store_prefill_pool` (v4:14886-14893), without the `store->gpu` early return.
     - V4.1 prefill MoE stays a CPU union on host slots even with the GPU tier up. V4's function, with its GPU no-op, is untouched.
  2. In `v41_target_batch` (v4:21894-21934):
     - call it with `layer_id` before each `coli_v4_block_window_batch_ref` (v4:21918), and with `-1` on every return path and at the end;
     - pass `use_prefill_pool` through from `target_batch_impl`, which returns into `v41_target_batch` at v4:21976 before V4's pool call at v4:21993.
     - Victims come from `hot_oldest_pool_victim` (v4:14471), which keeps a 6-slot floor per other layer.
  3. Chunk width for V4.1: `V41_PREFILL_CHUNK`, default 1024, clamp [1, 4096]; V4 keeps `v4_prefill_chunk_width` (v4:21878). The union already splits rows by `V4_EXPERT_BATCH_MAX=128` (v4:10104).
     - With the borrowed pool (335 slots ≥ 192, one layer's union), each expert is read once per layer per 1024-row chunk instead of once per 128 rows.
     - Chunk-major order (v4:21888-21893) is kept, so the cross-layer state rules are unchanged; a larger chunk is closer to the vendor, which runs the whole prompt in one forward.
- **Expected impact [I].**
  - A 1k-token prompt goes from ~8 sweeps (~5 min on one NVMe) to ~1 sweep: ~45-60 s of disk, with ~20-30 s of CPU MoE compute overlapping it. **TTFT ~5-6x better.**
  - The next prefill lever is the GPU bank for E=192 (G8, deferred).
  - RAM: activations and attention scratch at 1024 rows are ~0.5-1 GB (q is rows × 32768 × 4 B = 128 MB, plus att, scores and the hc buffers).
- **Tests.**
  - Tiny fixture with `V41_ATTN_FP32=1` and `V41_PREFILL_CHUNK` = 1, 3, 128 and 4096: identical greedy tokens, and logits within 1e-6 against `ref_vendor.json`.
  - `tests\test_v41_on_v4_union_slots` (in `v41-on-v4-check`).
  - REAP: an 800-token prompt at chunk 128 against 1024. The tokens must match (or the margin is logged); measure TTFT and `v4_direct` bytes.
  - V4: `deepseek-v4-tiny-check` (V4 never enters `v41_target_batch`).

### W1. Build the V4.1 WNR2 container (S; data; today, in the background)

- **Can the V4-trained CDF model be reused?** Valid, but retrain [R].
  - The WNM2 contexts depend only on FP4 nibble and E8M0 statistics: `srel(scale-median,7) x prev magnitude 9 x block max 9 = 567` per matrix, with 3 matrices fixed (`wex_neural_rans.h:121-141`).
  - Geometry comes from each `WexEntry` (`lab\wex-gpu-direct-20261019\nr2\encode_nr2.cu:44-79`), and the model is embedded in the container, so any model is lossless.
  - However, `build_nr2_full.py` always retrains (54 s on V4) and has no load option. Retraining costs nothing and fits REAP's experts, so there is no reason to reuse.
  - `tools\repack_v4_wex.py` writes only WNR1 (codec 3), not WNR2.
- **Command** (GPU encoder `wex_nr2_tools.dll`, already built):
  `python -I lab\wex-gpu-direct-20261019\nr2\build_nr2_full.py --source D:\kaira_wex\v41-reap\v4_flash_experts.wex --out E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex --weight-chunk 2048 --scale-chunk 512`
  - The chunks are 2048:512 directly: the encoder validates them (`encode_nr2.cu:58-60`), the decoder reads the sizes per record (`wex_neural_rans.h:229-232`, `wex_cuda.cuh:404-405`), and `test_decode_batch` at 2048:512 printed ALL OK.
- **Cost [I from V4: 779 s for 147 GB].** About 13-16 min.
  - Output ~126 GB (117.7 GiB). E: has 136 GiB free; the script's free-space check (0.9 × source + 2 GiB ≈ 123 GiB) passes with ~13 GiB to spare.
  - It reads D: and loads the GPU, so **do not run it while M0 or any benchmark is running**.
- **Test.**
  - The script's own readback and hash validation, plus the CPU cross-check every 43rd expert.
  - Then 64 random records: decode through `coli_wex_cuda_decode_batch_to_device` (lab `test_decode_batch.exe`) and compare bytes with the same records in the WEX1 container.
  - Report bytes saved; about 12.8% is expected [I].

### G3. GPU head for V4.1 (S; goal a; today if time)

- **Goal.**
  - Take the head off the CPU: a bf16 matvec over 129280 × 5120, i.e. 1.32 GB read per token, ~25-35 ms [I].
  - `coli_v4_gpu_head_argmax` (v4:20415) refuses V4.1 (v4:20429-20433). Its kernel is DeepGEMM-only, with `constexpr I=4096` (cu:2951/2971).
- **Design.**
  - Upload the head with `dsv4_cuda_upload_bf16` (as at v4:20436), run the generic `dsv4_cuda_matvec` (the `mv_bf16` branch in cu:2717), download the 517 KB of logits and do the argmax and sampling on the host.
  - Opt-in `V4_GPU_HEAD=1`, as for V4. No new kernel.
  - Costs 1.23 GiB of VRAM, about 70 mirrors.
- **Expected.** -25 to -30 ms/token [I]; worth it only if M0 shows head time.
- **Test.** Tiny fixture and REAP: argmax equal to the CPU head on 256 steps, with the margin logged on a mismatch.

### F1. Confidence-gated prefetch with a per-token byte budget (M; goal d; code today, measure after G2)

- **Goal.** Keep the prefetch, which V4 needs (3.94 tok/s without it, ~5.5 with it [M, Oct]), but stop the speculative bytes. On V4 those were ~71 reads, ~0.83 GB/token, more than half of all bytes [I, sim]. A V4.1 record costs 1.4x more.
- **Engine (lane D), about 80 lines of v4, disjoint regions:**
  - **Confidence: none today [R].** The predicted route weights are computed in `v4_np_router_worker` (v4:7763/7765), but `v4_np_publish_ex` (v4:7455) and `v4_np_issue_ex` (v4:7378) carry only the ids.
    - Add `float conf` to `V4NpItem` (v4:7229). It is the predicted normalized route weight of that expert at its depth; depth 2 keeps its own weight.
  - **Threshold.** `V4_NP_MIN_CONF` (default 0 = today). In `v4_np_worker` (v4:7320-7365), skip the WEX-direct prefetch (v4:7341), the promote (v4:7356) and the attach (v4:7359) when `conf < threshold`. A depth-2 item may still do a RAM lookup.
  - **Byte budget.** `V4_NP_TOKEN_MB` (default 0 = unlimited). Keep a per-forward counter keyed by `item.stamp` (the `v4_np_forward` count) and reset it when the stamp changes.
    - Charge it in `v4_np_worker` before the WEX-direct, promote or attach. When over budget, count it in `promoteprof refused` (`v4_promote_counts[1]`, v4:18594) and in a new `npprof gated=` counter.
    - The existing `V4_PROMOTE_INFLIGHT_MB` cap (v4:18606-18612) stays.
  - **Stats.** Under `V4_PREDICT_STATS=1` (v4:7281), log a used/unused histogram by confidence bucket. That is what picks the threshold.
- **V4.** Defaults off, so behaviour is byte-identical.
- **Expected [I].** +5-15% only when M0 and the post-G2 counters show the disk is the bottleneck and prefetch precision is low. Otherwise about zero.
- **Tests.**
  - A unit test of the budget and stamp reset (a small C test on the queue/worker helpers, or `COLI_V4_TEST_HOOKS`).
  - REAP: tokens identical across every (threshold, budget) arm, since prefetch must never change the outputs. Protocol in 3.3.

### W2. Batched WNR2 decode per layer (M; goal b; after G2 and F1)

- **Goal.** Replace the per-record helper on the demand path, ~1.0 ms per record [M], with one `coli_wex_cuda_decode_batch_to_device` per layer: 0.28-0.30 ms per record at batch 6-8 [M, V4 11 MiB records]; ~0.4-0.55 ms expected for V4.1's ~16.4 MB records [I].
- **Where [R].**
  - The demand chain today: `v4_job_run` → `coli_v4_gpu_expert_attach_wex` (v4:18450-18510) → `v4_gpu_expert_attach_cached_fill` (v4:18092) → `v4_wex_fill_mirror` (v4:18372-18400) → `coli_wex_cuda_decode_to_device` (v4:18397), one record per call.
  - Wrapper: `wex_decode.h:53-57`, implemented at `wex_decode.c:198-219`. Bridge export: `tools\wex_cuda_bridge.cu:409`, 8 thread-local lanes, workspaces grown per call; feature bit 8 at `:290-296`. Feature 8 is in the deployed helper too [I].
- **Design (env `V4_WEX_BATCH=1`, default off; only with `COLI_WEX_GPU_DIRECT>=1` and `coli_wex_cuda_batch_available()`).**
  1. In `moe_token_pipeline`, after routing and before the lane preload (v4:9210-9229): for each `expert_ids[i]`, if `coli_v4_gpu_expert_peek_routed` misses and `coli_v4_expert_store_wex_entry` (v4:13964) says WNR2 and not in the host RAM tier, add it to the batch.
  2. Split `v4_gpu_mirror_fill_unlocked` (v4:17963-18004) into reserve(N) and publish(i). Reserve N mirrors under one cache-mutex hold, with one `dsv4_cuda_stream_drain`.
  3. Get the N records:
     - from the compressed RAM record cache (v4:14055-14099),
     - or with N overlapped `ReadFile` calls on the WEX handle into a ring of 8 page-locked staging buffers (~132 MB pinned), waiting on all of them.
  4. Make one batch call, then publish each `status==0` mirror. Statuses -1 and 1 release the reservation and leave that expert to today's per-record path.
  5. The lanes then hit VRAM at their `vram_first` peek (v4:~6638), so no lane logic changes.
  6. The prefetch twin (`v4_np_worker` draining same-layer items, v4:7320-7390) comes afterwards, because it edits F1's worker.
- **V4.** Off by default, so identical.
- **Expected [I].** About -0.6 to -0.9 ms per WNR2 decode. At ~60-110 WNR2 decodes per token on the WNR2 arm, that is -35 to -90 ms/token. It is a necessary condition for the WNR2 arm, not a gain over WEX1.
- **Tests.**
  - The lab `test_decode_batch` on V4.1 records: byte-identical, and corrupt or invalid records give status -1 with the destination untouched.
  - REAP with the WNR2 container: `V4_WEX_BATCH=0` against `1` gives identical tokens. Counters: batches/token, records/batch, ms/batch.

### W3. A/B of WEX1 against WNR2 at equal RAM (S code + M wall time; goal c; after W1, W2, G2)

- Protocol in 3.2. Code is only the counters: per-token VRAM hits, RAM hits, NVMe reads and bytes, WNR2 decodes (demand/prefetch) and decode ms, in one `wexprof` line next to `moeprof`.

### D1. A second drive for WEX1 (S; data; **needs the user's OK**; only if M0 says the disk dominates)

- `COLI_MODEL_MIRROR` / `COLI_WEX_EXTRA_MIRRORS` accept up to 3 replicas of a byte-identical container (v4:13244-13300), with the split from `COLI_DISK_WEIGHTS` or the startup probe (v4:371-404). Correction 2026-11-09: that split applies only when `COLI_MODEL_MIRROR` registers shard copies. A container-only mirror is split by `COLI_WEX_WEIGHTS` (e.g. `38,62` for U:+C:, primary first); without it every read silently goes to the primary. Check for `[v4_wex] read split over 2 drives: 38% 62%`.
- Copying the WEX1 to `U:\kaira_wex\v41-reap\` (144 GB; U: has ~258 GiB free) gives D:+U: striping. On Qwen files, C:+U: measured 6.6 GiB/s against 3.34 and 2.23 GiB/s alone [M].
- **Expected.** The disk term ÷1.5-1.9, i.e. **~+40-80% tok/s** if disk-bound [I]. This is likely the biggest single decode lever after G2.
- Caveat: the Oct V4.1 mirrors on D:/O: were slower, but O: is a 1 GB/s USB disk. Only the probe decides.

### Deferred (not today), roughly in value order

| ID | What | Why later / size |
|---|---|---|
| S1 | V4.1 snapshots / prefix reuse (refused at v4:12225-12237) | Every KAIRA turn re-prefills the whole conversation: the top usability item after P1. Needs the V4.1 attention state (comp, keys, picks, mask, engram history) in the snapshot. M-L |
| G8 | GPU bank for V4.1 prefill MoE (E=192) | The bank path has 256/4096/2048 at v4:18804-19326 (`valid[256]`, `missing[256]`, `bank_create(256,4096,2048)`) and cu:3716/3719 `const H=4096,I=2048`. It moves ~20-30 s of CPU per 1k prompt tokens off the CPU [I]. M-L |
| R1 | Drop the host copies of GPU-resident dense and head for V4.1 | +6.5-7.8 GB of RAM for experts (~350-420 slots). Precedent: Vulkan `v4c_drop_layer` (v4:21174-21260). Every CPU fallback must then be impossible or re-read. M |
| K1 | V4.1 compressed KV and index keys stored as bf16, later FP4 | `comp`/`keys` are fp32 (v4:2193): ~6.4 GB of host RAM at 1M context. bf16 is lossless if the stored values are the fp4/e4m3 qdq'd ones [I, verify]. S-M |
| G4 | GPU router E=192 | `route_top6*` take E and `min_sum` (cu:523, 608, 1869-1885), guards at cu:2714/3562/3633/3662/3681, engine at v4:16836/16867/17185/17243. ~3 ms/token. S-M |
| G5 | GPU mHC for V4.1 (pre_mix carry, eps 1e-20, H≠4096) | `v4_gpu_mhc_refuses_v41` (v4:19725), `hidden!=4096` (v4:19743, 19792, 20283, 20358), eps `1e-6f` (cu:497, 2479, 2487). Needs the engram handoff (4.3). ~5-10 ms. M |
| G6 | V4.1 attention on the GPU | New semantics (ratio 1/2 without APE or overlap, shared KV, FP4 latent, two-level indexer): new kernels, not a generalization. Matters at long context (4.2). L |
| G7 | CUDA Graphs on decode | 36 µs vs 342 µs per 40-kernel chain [M], but per-call round trips need a device-resident chain first (after G5/G6). M |
| PR1 | Refit of the V4.1 predictor gates | Needs a dump collection (~1 h) and a ridge fit; V4 gained 79.6 → 84.1 R@8 [M]. M |
| H1 | Host passes `COLI_WEX_PATH` for V4.1 | The host drops it for the V4.1 family (`src\host\ColibriFamily.cs:106` [I]). Needed before KAIRA can use D:/E:. S, src/host |
| DS | DSpark/MTP for V4.1 | Already planned as a later phase (WP-10). L |
| DG | DeepGEMM shapes for V4.1 | `K/512` breaks at I=2304 (cu:535-546); the generic path first. M |

---

## 2. Lanes: what runs in parallel, what is sequential

Every lane works in its own workspace (`python -I lab\v41-port-20261106\ws.py sync|make|run <ws> ...`), makes a patch against 79aff2b into `lab\v41-gpu-20261107\<lane>.patch`, and **never edits `src\` directly**. An integrator applies the patches in the order A → B → C → D (`git apply --3way` in an integration workspace), runs the full regression (5.1), and touches `src\` only when the user says so.

| Lane | Packages | Files and regions | Can start | Parallel with |
|---|---|---|---|---|
| A, backend | G1-bk, then G2-bk (later G4-bk) | `backend_cuda_dsv4.cu/.h`, `dsv4.def`, `backend_loader_dsv4.c`, `tests\test_dsv4_*cuda*`, `Makefile:1255` | now | B, C, D, E |
| B, engine GPU + block | M0 code, G1-en, G2-en, G3 | v4:3379, 3632-3655; 6270-6420; 8990-9640; 9796-9810; 16528-17200; 20415-20455; `tests\test_v41_on_v4_gpu_refusal.c` | now; it codes against the exports fixed in 2.1 and runs on the GPU once A has a DLL | A, C, D, E |
| C, engine prefill | P1 | v4:14880-14905 (new function), 21876-21976 | now | A, B, D, E |
| D, engine prefetch | F1 | v4:7180-7470, 7726-7790, 18594-18640 | now (code); measure after B | A, B, C, E |
| E, data and measurement | M0 runs, drive probe, W1 build, later D1 | `lab\v41-gpu-20261107\` only, E: (and U: for D1) | now | everyone, but **no benchmark while W1 or any build runs** |

**Sequential, because they share code or need a result first:**
- G2-en after G1-en (the tier must open); both are in lane B.
- W2 after G2 (it edits `moe_token_pipeline` v4:9210-9229, next to B's router and compute edits) and after F1 (its prefetch twin edits `v4_np_worker`). W2 also needs W1 to test.
- W3 after W1 + W2 + G2.
- F1's measurement and D1's decision after the M0 numbers and a working G2.
- G5 before G7. S1 needs the V4.1 attention-state layout fixed, so do it after P1.

### 2.1 Interface fixed up front (so A and B can work at once)

```c
/* backend_cuda_dsv4.h — new, optional exports (old ones untouched) */
int  dsv4_cuda_upload_fp8_ex(Dsv4CudaTensor **t, const void *weight, const void *scale_e8m0,
                             int rows, int columns, int device, int bf16_rounded, int block);
void dsv4_cuda_set_act_group(int group);          /* 128 default; 32 for V4.1 */
int  dsv4_cuda_fp8_block_available(void);          /* loader: _ex resolved */
int  dsv4_cuda_act_group_available(void);          /* loader: setter resolved */
```

Scale layout for `_ex`: raw E8M0 bytes, row-major, `ceil(rows/block) x ceil(columns/block)`. Dense activations are quantized by the engine (host) for `block != 128`; the device quantizes only the expert and shared-expert intermediates, with the global group.

---

## 3. Measurement protocols

### 3.1 Common rules (every run in M0, W3, F1)

- **Quiet machine.**
  - KAIRA and its engine are closed. No sub-agent, build, W1 encode, copy or indexing is running. Gemma or other models are off.
  - Check that Defender exclusions are active (the user runs `lab\v41-port-20261106\defender-exclusions.ps1` as admin; we may not do it).
  - High-performance power plan.
  - `nvidia-smi` shows no other compute process. Log free RAM and free VRAM at start.
  - Same boot session for all arms of one comparison.
  - WEX reads are unbuffered (`FILE_FLAG_NO_BUFFERING`, v4:13225), so the page cache does not need flushing; safetensors reads (R0a) do, so run R0a first after a reboot or note it.
- **Fixed binaries.** Record the hashes of the exe, both DSV4 DLLs and `kaira_wex_cuda.dll`. Pin `COLI_DSV4_DLL`.
- **Lossless settings.** `V4_SKIP_COLD=0` (kaira.json forces it to 1 otherwise; lossy), `V4_TOPK` unset, `V4_MTP=0`, temperature 0.
- **Prompt set.** `lab\v41-gpu-20261107\ab\prompts.jsonl`, 8 prompts:
  - 2 code, 2 Italian chat, 2 reasoning, 1 with a ~2k-token context, 1 short;
  - the KAIRA system prompt is **not** used (P1/S1 make TTFT a separate question);
  - 1 warm-up prompt is discarded; 256 tokens each.
- **Per prompt, record:**
  - decode tok/s, TTFT, the PROF line (v4:24200);
  - `decprof`/`moeprof`/`attachprof`/`promoteprof`/`copyprof` (`DSV4_DECODE_PROF=1`, `V4_ATTACH_PROF=1`; check once that they cost < 1%);
  - `v4_tokens`, `v4_direct`;
  - the new `wexprof`/`npprof` lines;
  - the generated token ids.
- **Statistics.**
  - Each run is a fresh process (cold VRAM/RAM caches), then the warm-up, then the 8 prompts in sequence, so the cache warms as in real use.
  - Arms run in ABBA order, twice (8 runs).
  - Report the per-prompt paired difference and the mean ± 95% CI over the 4 pairs, plus the median and p90 token latency.

### 3.2 WEX1 vs WNR2 at equal RAM on V4.1 (goal c)

- **The arms differ only in `COLI_WEX_PATH`.**
  - A = `D:\kaira_wex\v41-reap\v4_flash_experts.wex`
  - B = `E:\kaira_wex\v41-reap\v4_flash_experts.nr2.wex`
- **Identical in both arms:**
  - `--memory-gb 16` (fixes the RAM tier), `COLI_WEX_GPU_DIRECT=2`, `COLI_WEX_RAM_CACHE_GB=4`;
  - `DSV4_CUDA_VRAM_RESERVE_MB=4096`, `DSV4_CUDA_EXPERT_MIRRORS` default;
  - `V4_COPY_WORKER=1 V4_PROMOTE_VRAM=2 V4_ASYNC_ATTACH=1 DSV4_CUDA_STREAM_PRIORITY=1`;
  - `V4_WEX_BATCH=1` (a no-op on WEX1), the same F1 settings, the same `V4_GPU_HEAD`.
- **Equal RAM means equal bytes, not equal experts.**
  - Check from the startup lines (`ram_tiers ... target_cache=`, the RAM-cache size line) that slots × 18.8 MB plus the cache GB match within 1%.
  - WNR2 then holds ~14% more records in the same 4 GB. That capacity is the mechanism under test.
- **Token identity.** WNR2 is lossless, and decoded bytes are identical to WEX1, which W1 verifies. The same kernels run on the same bytes, so **token ids must be identical in every prompt of every run**.
  - Any difference stops the A/B and is a bug. The Oct C4 divergence was an upload race.
- **The drive confound.**
  - D: and E: are different drives, and E: also serves engram shard 48 (the layer-14 table), so the WNR2 arm shares E:.
  - Before the A/B, run the drive probe (M0) on D: and E: at the arms' record sizes, and record the engram time (`v41_engram` share of the per-layer timer).
  - If D: and E: differ by > 10%, add a **same-drive control on U:** run sequentially, so each arm is on U: in turn (copy, run, delete; ~13-15 min per copy). It needs the user's OK for the ~270 GB of writes.
- **Decision rule** (from `reports\WNR2 cache decode CPU e WDDM.md`):
  - WNR2 becomes the speed format only if it is ≥ +3% with the CI excluding 0, and the tokens are identical.
  - Otherwise WEX1 stays in production and WNR2 is a capacity format (-12.8% disk).
  - Report where the time went (disk ms, decode ms, wait ms per token) so the result can be explained, not just stated.

### 3.3 Prefetch gating (goal d)

1. **Offline pick.** One baseline run (WEX1, F1 defaults off) with `V4_PREDICT_STATS=1`. It gives recall and precision of the V4.1 predictor (model-gate fallback) at depths 1 and 2, and the used/unused histogram by confidence bucket. Pick 2 thresholds at about the 50% and 70% precision points.
2. **Grid** (WEX1, fixed RAM as in 3.2), against the baseline (threshold 0, budget ∞), ABBA per arm, 2 repetitions:
   - threshold τ ∈ {0, τ1, τ2} × budget ∈ {∞, 8, 16 experts/token, i.e. ~150 and ~300 MB}.
   - Do not run all 9 combinations: run τ alone, the budget alone, then the best pair.
3. **Metrics:**
   - tok/s, p90 token latency;
   - prefetches issued per token, precision (used within the next forward), wasted MB/token;
   - demand misses/token, wait ms/token, `promoteprof refused`, `npprof gated`.
4. **Tokens identical across all arms** (prefetch may not change outputs).
5. **Keep** a setting only if tok/s ≥ +3% with the CI excluding 0. Otherwise the defaults stay off.

---

## 4. Risks, and what to defer

### 4.1 VRAM budget (RTX 5090, 32 GiB)

| Item | GiB | Note |
|---|---|---|
| WDDM, desktop, CUDA context | ~1.2-1.5 | [I]; measure with nvidia-smi before open |
| Dense FP8 mirrors (8 tensors × 40 layers) | 6.04 | 162 MB/layer [R shapes]; scales at 32x32 are +0.1% |
| Head bf16 (G3, opt-in) | 1.23 | 129280 × 5120 × 2 |
| Reserve `DSV4_CUDA_VRAM_RESERVE_MB` | 4.0 | kaira.json value; default 2.8 with the MoE batch, else 0.6 (v4:18128-18134) |
| Work buffers, activations, prefill batch | ~0.3-0.8 | 1024-row `matmul_batch` in/out ~40 MB; more at a 4096-row chunk |
| **Expert mirrors** | **~18-19.5** | 17.93 MiB each, so **~1,030-1,110** (≈1,150-1,200 without G3 or with a 2.8 GB reserve); 14-16% of 7,680 experts |

- **Risk: WDDM over-commit.** WDDM pages to system memory silently instead of failing, and speed collapses.
  - Keep the reserve. The growth guard re-probes free VRAM every 64 misses (v4:18150-18167).
  - Watch "Shared GPU memory" in Task Manager during the first REAP GPU run.
- The 40 × 192 experts (144 GB) never fit; the plan assumes a cache.

### 4.2 1M context

- The V4.1 attention state is on the host in fp32 (`comp [cap][512]`, `keys [cap][128]`, v4:2193).
  - kv sources are layers 2, 8, 14 (ratio 2) and 20 (ratio 1), so the rows at N tokens are 2.5N.
  - At N=1M: comp 5.1 GB + keys 1.3 GB ≈ **6.4 GB of RAM** [I]. Even with the 16 GB engine budget, that comes out of the expert cache.
- CPU indexer scoring grows linearly: ~10-30 ms/token at 128k, ~100-250 ms at 1M [I].
- **Mitigation:**
  - K1 (bf16, later FP4 storage) and G6 later.
  - Until then, make the resource plan count this KV. Carry-forward WP-4b: it still counts compressed KV on every ratio>0 layer.
  - Suggest to the user a 128k cap on the GPU profile, keeping 1M allowed with the RAM cost stated. **This is the user's decision.**

### 4.3 Engram and the host streams

- Today mHC stays on the CPU for V4.1: `V4_MHC_CHAIN` is off for V4.1 (v4:22105) and GPU mHC is refused (v4:19725). So the residual streams are host memory at layers 1 and 14, and **there is no handoff in this plan**.
- When G5 moves the streams to the device, `engram_gate` (`deepseek_engram.h:362`) needs:
  - either a D2H plus H2D of hc × H × 4 = 80 KB at layers 1 and 14 (two round trips per token, ~0.1-0.2 ms under WDDM),
  - or a small CUDA gate fed by the host-computed kv rows. Compute the kv early; it depends only on the tokens.
- Engram tables are read on C: and E: (~48 rows of 264 B per token, cache of 65,536 rows). E: is shared with the WNR2 container: see the confound in 3.2.

### 4.4 WDDM, PCIe and the single copy engine

- `asyncEngineCount=1` [M]: mirror uploads (18.8 MB, ~0.39 ms each at 48.6 GiB/s [M]) serialize with every activation H2D/D2H of the per-call tier.
  - Background H2D alone slowed a 40-kernel chain by ~33% [M].
  - G1+G2 add ~280-400 synchronous round trips per token.
  - Mitigations already in the tree: `DSV4_CUDA_STREAM_PRIORITY=1` (p90 -18% [M]), and the copy worker's gate.
  - The real fix is a device-resident chain plus CUDA Graphs (G5/G6/G7): deferred.
- The DeepGEMM DLL is loaded first on sm_12x (ld:441). Every V4.1 test must run with both DLLs, because production uses `coli_cuda_dsv4_dg.dll`.

### 4.5 Numerics

- GPU sums run in a different order from the CPU. V4.1 on the GPU will **not** be bit-identical to the verified CPU path, which is the same situation as V4's GPU tier.
- Parity uses tolerances plus top-2 margin logging, not equality.
- The fp32 parity mode (`V41_ATTN_FP32=1`, v4:16303-16310) remains a CPU-only tool.
- **Arms of one configuration** (GPU-direct 0/2, batch on/off, WEX1/WNR2, prefetch arms) **must** be token-identical.

### 4.6 RAM (32 GB)

- The host keeps the dense copies after upload (no drop in `coli_v4_gpu_layer_upload`, v4:16875-16996), so G1 frees no RAM. R1 is the follow-up.
- P1's 1024-row chunks add ~0.5-1 GB of scratch during prefill.
- Sub-agents and KAIRA running alongside swing the budget (Oct: 25-49 GB swings [M]): **no benchmark while other sessions run**.

### 4.7 Process

- One file (v4, 29.5k lines) is edited by three lanes in disjoint regions. Patches are merged by one integrator, and the full regression runs after each merge.
- A forgotten duplicate unit copy is a known trap (phase-1 MISS-02), so grep each changed function name for a second definition.

---

## 5. Test and regression matrix

### 5.1 Gates before any merge (every lane)

1. `make PYTHON=python deepseek-v4-tiny-check`: 21 PASS, plus `v41-on-v4-check` (Makefile:825-841, 2693-2699).
2. `tests\test_deepseek_v4` (unit) and the `test_v41_on_v4_*` binaries (config 125 checks, schema 44, forward refusal, union slots, gpu refusal as updated in G1).
3. `make dsv4-cuda-loader-test`, `make dsv4-fused6-test`, `make dsv4-cuda-test`.
   - Record today's status first: the attention-batch test calls a stub (cu:2641), and the MoE-batch test is DG-only.
   - Then run `python -I tests\test_backend_loader_dsv4_parity.py`.
4. V4 bit-compare: the old DLL against the new one, generic and DG, at V4 shapes (G1 test plan). `cmp` must be identical.
5. Tiny V4.1 fixture: CPU (`DSV4_CUDA=0`) against `ref_vendor.json` (unchanged), and GPU against CPU (tolerance and margin).

### 5.2 REAP on the GPU (after G1, after G2, after each later package)

- Use the invocation of `smoke-reap-fixed-20261107.log`, plus `COLI_MODEL_DIRS='C:/models/DeepSeek-V4.1-Flash-1017-engram;E:/models/DeepSeek-V4.1-Flash-1017-engram'`, `COLI_WEX_PATH=D:\kaira_wex\v41-reap\v4_flash_experts.wex`, `DSV4_CUDA=1`, `--memory-gb 16`.
- Logs to `lab\v41-gpu-20261107\smoke-*.log`. Expect:
  - `v4_gpu tier=dense-matvec`, no `fp8-block-32-unsupported`;
  - per-layer `uploaded=` totals ~6.5 GB;
  - `moeprof` lines;
  - the first 32 greedy tokens equal to the CPU run, or the margin is logged.

---

## 6. The "today" cut (2026-11-07)

Ordered by speed-up per hour; lanes A-E run in parallel, Opus agents only.

| # | Package | Lane | Effort | Expected | Done when |
|---|---|---|---|---|---|
| 1 | M0 runs + drive probe (before any build or W1) | E | S | numbers for everything below | warm s/token, misses/token, GB/s per drive written to `lab\v41-gpu-20261107\m0.md` |
| 2 | G1 dense FP8 block 32 on the GPU | A + B | M | -100 to -190 ms/token; prefill dense off the CPU | 5.1 green; REAP smoke on the GPU |
| 3 | G2 V4.1 decode MoE on the pipeline (VRAM mirrors, prefetch, copy worker, GPU-direct) | A + B | M-L | **~1.1-2.5 tok/s against 0.69** [I] | 5.1 green; 256-token REAP run; GPU-direct 0 vs 2 token-identical |
| 4 | P1 prefill pool + 1024-row chunks | C | S-M | TTFT ~5-6x better on 1k prompts [I] | fixture chunk invariance; REAP 800-token TTFT |
| 5 | M0 code (V4.1 decprof) | B | S | measurement | `decprof` printed for V4.1 |
| 6 | W1 WNR2 container 2048:512 (after M0, while lanes code) | E | S (15 min wall) | enables (b)/(c); ~-12.8% bytes | build JSON OK; 64 records byte-identical to WEX1 |
| 7 | G3 GPU head (if time) | B | S | -25 to -30 ms/token | argmax equal over 256 steps |
| 8 | F1 code (measured tomorrow, after G2 numbers) | D | M | +5-15% if disk-bound [I] | unit test; defaults off, V4 identical |

**Tomorrow:** W2 (batched WNR2 per layer), then W3 (A/B), F1 measurement, the D1 decision (needs OK), then G8 / S1 / R1 by the measured bottleneck.

**Realistic end-of-day result:** V4.1 decoding on the GPU tiers at roughly 1-2.5 tok/s, still NVMe-bound. Prefill TTFT is several times shorter. Every V4 gate stays green. The WNR2 container is built and verified. Nothing is committed or deployed.

---

Not done in this pass: I did not run any build, test or benchmark. Every tok/s figure is an estimate from the measured numbers cited. These are unverified: the warm hit rate of REAP routing, D:'s read speed, the CPU split of the 1.44 s/token, and whether the stored V4.1 latent is exactly representable in bf16 (K1).
