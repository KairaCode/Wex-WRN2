# Checklist: GPU paths that need the KAIRA / colibri build (not covered by this kit)

WNR3 draft 5.6 ("Required before any CZ use") and 11 T4/T7/T11. The kit's `run_zero_bittest.py` covers the default
CUDA FP32 kernels **as copied from the colibri dev tree** (team 2), an FP16 tensor-core GEMM and, experimentally, a
raw sm_120 block-scaled FP4 MMA. Everything below runs only inside KAIRA's build. Fill in each `Result:` line.
CZ = canonical-zero container (every E2M1 code 8 = -0 stored as 0 = +0; flags 0x1E in the draft); "byte-exact
container" = the WEX1/WNR3 container without CZ.

## 0. Preconditions
- [ ] KAIRA commit / build flags / CUDA toolkit / driver / GPU recorded. Result:
- [ ] Which expert-matmul kernels KAIRA dispatches on sm_120 for V4.1 MXFP4 experts (prefill and decode, batch 1 and
      >1): list them (this is OPEN-K item 7). Result:
- [ ] KAIRA's fork of `backend_cuda_dsv4.cu`: diff the FP4 paths (`f4()` :99, `mv`/`mv_rows`/`mv_fp4_grouped` :100-143,
      `mv_fp4_expert_major` :156-190, `mm_batch` :226-243, `v4f_*` :2242-2313 in the dev tree) against
      `tests/gpu/src_zero/zs_gpu_kernels.cu`. If they differ, copy the fork's device code into that file (keep the
      kernel names) and re-run `run_zero_bittest.py`. Result:
- [ ] Two containers of the same experts: byte-exact and CZ; the CZ decode differs from the byte-exact decode exactly on
      the code-8 nibbles and their number equals `cz_count` (tooling check of 5.6). Result:

## 1. RTX 5090 bit test, kernel by kernel (draft 5.6 test 1)
For every kernel of 0.: same activations for both containers (the 9 adversarial sets of the kit - gauss, mixed +-0,
all -0, all +0, one nonzero rest -0, 10% denormals, inf, NaN, 1e30 - plus real activations captured from a prompt),
memcmp of the output buffers: **0 differing bytes required**. Positive control (one weight nibble changed -> outputs
differ) and negative control (a kernel or setting that is known to propagate the sign of zero, e.g. the kit's
`negctl_seeded`, or a first-term-seeded reduction -> sign-only mismatches on all-zero rows) must fire. Rerun twice:
identical outputs (otherwise the kernel is nondeterministic and the memcmp is meaningless).
- [ ] Default CUDA FP32 kernels as built by KAIRA (sm_120). Result:
- [ ] **DeepGEMM sm120 FP4** path: E2M1 operands go to the tensor core (`mma ... kind::mxf4` / `kind::f8f6f4`, block
      scales ue8m0). Check whether weights are repacked (and whether the repack maps code 8 to 0 already). Compare the
      kit's EXPERIMENTAL `sm120_mxf4_EXPERIMENTAL` numbers (raw MMA, original vs canonical A) with the real GEMM.
      Result:
- [ ] **cuBLASLt MXFP8 path (`tc_fp4_matvec`)**: FP4 -> E4M3 conversion turns code 8 into E4M3 0x80 (-0) instead of
      0x00; check the conversion and memcmp the matvec outputs, for every block-scale mode used. Result:
- [ ] **Vulkan** (`shaders/chain_moe_gv.comp:97-114` decodes -0.0f), if KAIRA can enable it on this machine: same
      test on this driver. Result:
- [ ] Any fused path (fused6 / v4f_moe_kernel at V4 shapes, MTP experts, shared experts if FP4). Result:

## 2. End-to-end per-step logits memcmp (draft 5.6 test 2, T11)
- [ ] Deterministic settings fixed and recorded (greedy / fixed seeds, same batch composition, same kernel selection,
      CUDA graphs and split-K settings identical, no autotuning between runs). Result:
- [ ] Same prompts on both containers: the 4 W3-style prompts x >= 256 decode tokens (prefill included); dump the
      logits buffer at EVERY step (raw float bytes), memcmp per step: **0 differing bytes at every step**. Result:
- [ ] Negative control: byte-exact container run twice -> 0 differing bytes (determinism of the setup). Result:
- [ ] Positive control: one weight nibble changed in one expert that is routed in the prompt -> logits differ from that
      step on. Result:
- [ ] Token identity of the generated text (byte-exact WNR3 vs WEX1, CZ vs byte-exact). Result:

## 3. Other T7 items not in this kit
- [ ] GPU decode byte-exact on 30+ records via host and device paths with the full v1 profile (CEIL|LENC|HASH), batch
      sizes 1, 2, 3, 4, 6, 8, 10, 16 (the kit: 8 records, CEIL without LENC/HASH in gpu_bench; v1 only experimental).
- [ ] Retry idempotence (decode the same device record twice), a corrupted record inside a batch -> -1 and untouched.
- [ ] `compute-sanitizer --tool memcheck` and `--tool racecheck`: 0 errors on the decoders (final-scale stores).
- [ ] T9 speed with KAIRA's kernel on the 5090: us/record at 1 and 8 concurrent records, with HASH; gates WNR3 <=
      WNR2 + 5%, inverse own-cost <= +2%; same-config ratios over all launch configs.

If every item of 1 and 2 gives 0 differing bytes, CZ is bit-identical by construction on this machine and build
(draft 5.6); otherwise record the first differing kernel / step and stop.
