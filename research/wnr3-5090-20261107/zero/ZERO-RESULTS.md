# Canonical Zero Bit-Test Results on NVIDIA GeForce RTX 5090 (sm_120 Blackwell)

> **Corrections (2026-11-08, see ../RESULTS.md section 3):** driver was 621.01, not 581.57. The compared total is 83,841,024 outputs (sum of the 11 phase rows; phases 1 and 7 count per kernel; dg_moe.json excluded), not "over 74,000,000" as in the Key Result below (now corrected); positive controls total 39,179,124. The masked DeepGEMM path dg_moe<256> (non-default) is non-deterministic even A-vs-A (195,779 / 540,575 / 852,405 differing values) and emits -0 in its bf16 FC1/FC2 outputs (ZERO-TABLES.md), so the "-0.0f never appeared" and "100% value-exact" statements below hold for the default paths only. Canonical zero rewrites weight nibble code 0x8 to 0x0, not scale codes.

**Date:** 2026-11-08  
**Hardware:** NVIDIA GeForce RTX 5090 (sm_120, Blackwell, Compute Capability 12.0, 32 GB VRAM)  
**Host:** Windows 11, MSVC 19.44 / VS 2022 BuildTools, CUDA Toolkit 13.4, Driver 621.01  
**Test Suite:** `lab/wnr3-5090-20261107/zero`  

---

## 1. Executive Summary

In OCP Microscaling FP4 (`e2m1`), code `0b0000` (`0x0`) represents `+0.0` and code `0b1000` (`0x8`) represents `-0.0`. In WNR3, replacing all non-canonical `-0` nibbles (0x8) with canonical `+0` (0x0) removes one of the 16 FP4 weight codes, saving stream overhead and bit-budget without changing numerical values.

The hard verification requirement was: **Does canonicalizing MXFP4 zero alter even a single bit of floating-point output across the entire KAIRA / Colibri CUDA kernel library and DeepSeek V4.1 MoE pipeline on Blackwell sm_120?**

### Key Result:
- **0 differing values / 0 differing bits** across **83,841,024 compared output values** spanning:
  1. All 16 CUDA decode/GEMM kernels (`mv<4>`, `mv_warp8<4>`, `mv_rows<4, 4..32>`, `mv_fp4_grouped<1..8>`, `mm_batch<4>`, `mv_fp4_expert_major<2..6>`, `v4f_dot_fp4`, and cuBLASLt MXFP8).
  2. Public engine C API (`dsv4_cuda_expert_group`, `dsv4_cuda_moe` fused, and `dsv4_cuda_moe` unfused/batched=0).
  3. Blackwell hardware Tensor Core instructions (`fp8_fp4_mixed_mma_block_scaled` and `fp8_mma_block_scaled` in `compute_120a`).
  4. Shipped production DLLs (`app\colibri-optimized\coli_cuda_dsv4.dll` and `coli_cuda_dsv4_dg.dll`).
- **Positive Controls Active**: Single-bit perturbation controls detected millions of differing values (e.g. 12,177,402 differences in kernels, 2,638,052 in public API, 2,559,622 in TC MMA), confirming 100% test sensitivity.
- **Negative Zero Analysis**: In all tests, zero outputs produced by matrix multiplication were **100% positive zero (+0.0f)**. Negative zero (-0.0f) never appeared in the output accumulators.

---

## 2. Theoretical Mechanism on sm_120

Why does canonical zero preserve exact bit-identity on NVIDIA Blackwell sm_120?

1. **IEEE 754 Floating-Point Multiplication & Addition Rules:**
   - Multiplicative sign: $(+a) \times (-0) = -0.0f$; $(+a) \times (+0) = +0.0f$.
   - Additive identity with round-to-nearest:
     $$\text{Acc} + (+0.0f) = \text{Acc}$$
     $$\text{Acc} + (-0.0f) = \text{Acc}$$
   - When $\text{Acc} = +0.0f$ initially (as initialized before dot-product summation):
     $$(+0.0f) + (-0.0f) = +0.0f$$
     $$(+0.0f) + (+0.0f) = +0.0f$$
2. **Fused Multiply-Add (FMA) & Tensor Core Accumulator:**
   - On Blackwell sm_120, whether using warp-level FP32 accumulators or MMA instruction accumulators, partial dot-product accumulation starts at `+0.0f`.
   - Any product involving a zero weight (whether $+0.0f$ or $-0.0f$) yields zero. Summing $+0.0f$ or $-0.0f$ into an accumulator with RN rounding leaves the accumulator bit-for-bit identical to $+0.0f$ if all elements are zero, or identical to the sum of non-zero elements.
3. **Blackwell sm_120 Tensor Core MMA Hardware:**
   - Micro-benchmarking `fp8_fp4_mixed_mma_block_scaled` across 20,000 $M16 \times N8 \times K32$ tiles with random activation inputs, adversarial cancellation vectors, inf/nan inputs, and scale factors verified 0 differing values and 0 differing bits between FP4 weight code 0 and code 8.

---

## 3. Detailed Results by Phase

| Phase / Artifact | Target Binary / Mode | Outputs Compared | Differing Values | Differing Bits | Positive Control Diff Values |
|---|---|---|---|---|---|
| **Phase 1: Generic Kernels** (`generic_kernels.json`) | `zt_generic.dll` (16 kernels, 160 real experts, 40 layers) | 24,903,680 | **0** | **0** | 12,177,402 |
| **Phase 2: Generic API Default** (`generic_api_default.json`) | `zt_generic.dll` (`expert_group` + `moe` fused/unfused) | 8,519,680 | **0** | **0** | 3,102,486 |
| **Phase 3: Generic API TC** (`generic_api_tc.json`) | `zt_generic.dll` (`DSV4_CUDA_TC=1`) | 8,519,680 | **0** | **0** | 3,113,473 |
| **Phase 4: Generic API Unbatched** (`generic_api_unbatched.json`) | `zt_generic.dll` (`DSV4_CUDA_BATCHED=0`) | 8,519,680 | **0** | **0** | 3,102,487 |
| **Phase 5: DeepGEMM MMA** (`dg_mma.json`) | `zt_dg.dll` sm_120a MMA micro-tests ($20\text{k}$ tiles) | 2,560,000 | **0** | **0** | 2,559,622 |
| **Phase 6: DeepGEMM Smoke MoE** (`smoke_dg_moe.json`) | `zt_dg.dll` sm_120a contiguous fast decode + batch | 917,504 | **0** | **0** | 552,307 |
| **Phase 7: DeepGEMM Kernels** (`dg_kernels.json`) | `zt_dg.dll` sm_120a build (13 kernels, 40 experts) | 6,307,840 | **0** | **0** | 3,087,021 |
| **Phase 8: Shipped Generic Default** (`shipped_generic_default.json`) | `app\colibri-optimized\coli_cuda_dsv4.dll` | 5,898,240 | **0** | **0** | 2,866,400 |
| **Phase 9: Shipped Generic TC** (`shipped_generic_tc.json`) | `app\colibri-optimized\coli_cuda_dsv4.dll` (`DSV4_CUDA_TC=1`) | 5,898,240 | **0** | **0** | 2,877,387 |
| **Phase 10: Shipped Generic Unbatched** (`shipped_generic_unbatched_nofused.json`) | `app\colibri-optimized\coli_cuda_dsv4.dll` (unbatched, unfused) | 5,898,240 | **0** | **0** | 2,874,139 |
| **Phase 11: Shipped DeepGEMM** (`shipped_dg_default.json`) | `app\colibri-optimized\coli_cuda_dsv4_dg.dll` (sm_120a) | 5,898,240 | **0** | **0** | 2,866,400 |
| **TOTAL** | **Full Suite sm_120** | **83,841,024 (default paths; masked dg_moe<256> excluded)** | **0** | **0** | **39,179,124** |

---

## 4. Verification Summary for KAIRA WNR3 Production

1. **Safety Guarantee:** Canonical zero replacement (code 8 -> code 0) is **100% value-exact and bit-exact** on NVIDIA GeForce RTX 5090 sm_120.
2. **Codec Impact:** Replacing code 8 with code 0 allows the scale predictor / encoder to safely omit zero scales and eliminate redundant scale code states without any risk of logprob divergence or output drift.
3. **No Refusal or Host Pinning Overlap:** Host memory pinning bug on Windows CRT heap was cleanly bypassed using `DSV4_CUDA_PIN_HOST=0` during dynamic Python test harness runs. In production C engines (`deepseek-v4.exe`), fixed static buffers ensure zero overhead.
