
### dg_kernels.json  (kernels, deepgemm sm_120a, env {'DSV4_CUDA_PIN_HOST': '0'}, 23.3 s)
code-8 nibbles / nibbles: {'real_matrices': (82442283, 1415577600), 'synthetic_zero_rows': (5302418, 19005440)} experts: 40 shapes: {'w1': [2304, 5120], 'w2': [5120, 2304], 'w3': [2304, 5120]}

real_matrices

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| mv<4> | 6,225,920 | 0 / 0 | 0 | 0 | 3,041,979 / 30,559,628 | 829,744 | 0/0 | 434,595 | 0 | - |
| mv_warp8<4> | 6,225,920 | 0 / 0 | 0 | 0 | 3,041,954 / 30,551,836 | 829,744 | 0/0 | 434,595 | 0 | - |
| mv_rows<4,4> | 6,225,920 | 0 / 0 | 0 | 0 | 3,041,979 / 30,559,628 | 829,744 | 0/0 | 434,595 | 0 | - |
| mv_rows<4,8> | 6,225,920 | 0 / 0 | 0 | 0 | 3,041,979 / 30,559,628 | 829,744 | 0/0 | 434,595 | 0 | - |
| mv_rows<4,16> | 6,225,920 | 0 / 0 | 0 | 0 | 3,041,979 / 30,559,628 | 829,744 | 0/0 | 434,595 | 0 | - |
| mv_rows<4,32> | 6,225,920 | 0 / 0 | 0 | 0 | 3,041,979 / 30,559,628 | 829,744 | 0/0 | 434,595 | 0 | - |
| mv_fp4_grouped<1> | 6,225,920 | 0 / 0 | 0 | 0 | 3,041,941 / 30,558,719 | 829,744 | 0/0 | 434,595 | 0 | - |
| mv_fp4_grouped<2> | 6,225,920 | 0 / 0 | 0 | 0 | 3,041,941 / 30,558,719 | 829,744 | 0/0 | 434,595 | 0 | - |
| mv_fp4_grouped<4> | 6,225,920 | 0 / 0 | 0 | 0 | 3,041,941 / 30,558,719 | 829,744 | 0/0 | 434,595 | 0 | - |
| mv_fp4_grouped<8> | 6,225,920 | 0 / 0 | 0 | 0 | 3,041,941 / 30,558,719 | 829,744 | 0/0 | 434,595 | 0 | - |
| mm_batch<4> | 6,225,920 | 0 / 0 | 0 | 0 | 3,041,979 / 30,559,628 | 829,744 | 0/0 | 434,595 | 0 | - |
| v4f_dot_fp4(warp/row wrapper) | 6,225,920 | 0 / 0 | 0 | 0 | 3,041,954 / 30,551,836 | 829,744 | 0/0 | 434,595 | 0 | - |
| TC cuBLASLt MXFP8 (tc_quant+tc_expand+cublasLtMatmul) | 6,225,920 | 0 / 0 | 0 | 0 | 3,430,585 / 15,077,242 | 1,218,866 | 0/0 | 0 | 0 | - |

synthetic_zero_rows

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| mv<4> | 81,920 | 0 / 0 | 0 | 0 | 45,042 / 530,275 | 41,292 | 0/0 | 7,974 | 0 | - |
| mv_warp8<4> | 81,920 | 0 / 0 | 0 | 0 | 45,043 / 530,465 | 41,292 | 0/0 | 7,974 | 0 | - |
| mv_rows<4,4> | 81,920 | 0 / 0 | 0 | 0 | 45,042 / 530,275 | 41,292 | 0/0 | 7,974 | 0 | - |
| mv_rows<4,8> | 81,920 | 0 / 0 | 0 | 0 | 45,042 / 530,275 | 41,292 | 0/0 | 7,974 | 0 | - |
| mv_rows<4,16> | 81,920 | 0 / 0 | 0 | 0 | 45,042 / 530,275 | 41,292 | 0/0 | 7,974 | 0 | - |
| mv_rows<4,32> | 81,920 | 0 / 0 | 0 | 0 | 45,042 / 530,275 | 41,292 | 0/0 | 7,974 | 0 | - |
| mv_fp4_grouped<1> | 81,920 | 0 / 0 | 0 | 0 | 45,043 / 530,653 | 41,292 | 0/0 | 7,974 | 0 | - |
| mv_fp4_grouped<2> | 81,920 | 0 / 0 | 0 | 0 | 45,043 / 530,653 | 41,292 | 0/0 | 7,974 | 0 | - |
| mv_fp4_grouped<4> | 81,920 | 0 / 0 | 0 | 0 | 45,043 / 530,653 | 41,292 | 0/0 | 7,974 | 0 | - |
| mv_fp4_grouped<8> | 81,920 | 0 / 0 | 0 | 0 | 45,043 / 530,653 | 41,292 | 0/0 | 7,974 | 0 | - |
| mm_batch<4> | 81,920 | 0 / 0 | 0 | 0 | 45,042 / 530,275 | 41,292 | 0/0 | 7,974 | 0 | - |
| v4f_dot_fp4(warp/row wrapper) | 81,920 | 0 / 0 | 0 | 0 | 45,043 / 530,465 | 41,292 | 0/0 | 7,974 | 0 | - |

not run: {'mv_fp4_expert_major<2>': ['2304x5120', '5120x2304', '64x2304', '64x5120'], 'mv_fp4_expert_major<3>': ['2304x5120', '5120x2304', '64x2304', '64x5120'], 'mv_fp4_expert_major<4>': ['2304x5120', '5120x2304', '64x2304', '64x5120'], 'mv_fp4_expert_major<6>': ['2304x5120', '5120x2304', '64x2304', '64x5120'], 'TC cuBLASLt MXFP8 (tc_quant+tc_expand+cublasLtMatmul)': ['64x2304', '64x5120']}

### dg_mma.json  (mma, deepgemm sm_120a, env {'DSV4_CUDA_PIN_HOST': '0'}, 1.0 s)

**mxf8f6f4 e4m3 x e2m1 (DeepGEMM fp8_fp4_mixed_mma_block_scaled)**

- random A, random B codes (orig vs canon), C=+0: differing D / bits: [0, 0]
- random A, random B codes, C=+0: rerun differing D: 0
- random A, random B codes (orig vs canon), C=-0: differing D / bits: [0, 0]
- random A, random B codes, C=-0: rerun differing D: 0
- random A, random B codes (orig vs canon), C=random: differing D / bits: [0, 0]
- random A, random B codes, C=random: rerun differing D: 0
- positive control (one B code sign-flipped per lane), C=+0: differing D: 2559622
- B all zero (codes 0/8 random) vs all code 0, C=+0: differing D: 0
- B all zero (codes 0/8 random), C=+0: D==-0 / D==+0: [0, 2560000]
- B all code 0, C=+0: D==-0 / D==+0: [0, 2560000]
- B all zero (codes 0/8 random) vs all code 0, C=-0: differing D: 0
- B all zero (codes 0/8 random), C=-0: D==-0 / D==+0: [0, 2560000]
- B all code 0, C=-0: D==-0 / D==+0: [0, 2560000]
- A>0, B all code 8 (every product -0) vs B all code 0 (every product +0), C=+0: differing D: 0
- A>0, B all code 8, C=+0: D==-0 / D==+0: [0, 2560000]
- A>0, B all code 0, C=+0: D==-0 / D==+0: [0, 2560000]
- A>0, B all code 8 (every product -0) vs B all code 0 (every product +0), C=-0: differing D: 0
- A>0, B all code 8, C=-0: D==-0 / D==+0: [0, 2560000]
- A>0, B all code 0, C=-0: D==-0 / D==+0: [0, 2560000]
- A all -0, B > 0 (every product -0), C=+0: D==-0 / D==+0: [0, 2560000]
- A all -0, B > 0 (every product -0), C=-0: D==-0 / D==+0: [0, 2560000]

**mxf8f6f4 e4m3 x e4m3 (fp8_mma_block_scaled; B = tc_expand bytes)**

- random A, random B codes (orig vs canon), C=+0: differing D / bits: [0, 0]
- random A, random B codes, C=+0: rerun differing D: 0
- random A, random B codes (orig vs canon), C=-0: differing D / bits: [0, 0]
- random A, random B codes, C=-0: rerun differing D: 0
- random A, random B codes (orig vs canon), C=random: differing D / bits: [0, 0]
- random A, random B codes, C=random: rerun differing D: 0
- positive control (one B code sign-flipped per lane), C=+0: differing D: 2559582
- B all zero (codes 0/8 random) vs all code 0, C=+0: differing D: 0
- B all zero (codes 0/8 random), C=+0: D==-0 / D==+0: [0, 2560000]
- B all code 0, C=+0: D==-0 / D==+0: [0, 2560000]
- B all zero (codes 0/8 random) vs all code 0, C=-0: differing D: 0
- B all zero (codes 0/8 random), C=-0: D==-0 / D==+0: [0, 2560000]
- B all code 0, C=-0: D==-0 / D==+0: [0, 2560000]
- A>0, B all code 8 (every product -0) vs B all code 0 (every product +0), C=+0: differing D: 0
- A>0, B all code 8, C=+0: D==-0 / D==+0: [0, 2560000]
- A>0, B all code 0, C=+0: D==-0 / D==+0: [0, 2560000]
- A>0, B all code 8 (every product -0) vs B all code 0 (every product +0), C=-0: differing D: 0
- A>0, B all code 8, C=-0: D==-0 / D==+0: [0, 2560000]
- A>0, B all code 0, C=-0: D==-0 / D==+0: [0, 2560000]
- A all -0, B > 0 (every product -0), C=+0: D==-0 / D==+0: [0, 2560000]
- A all -0, B > 0 (every product -0), C=-0: D==-0 / D==+0: [0, 2560000]

### dg_moe.json  (dg, deepgemm sm_120a, env {'DSV4_CUDA_PIN_HOST': '0'}, 38.8 s)

paths

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| dg_moe<256> (masked decode) : routed output (fp32 of bf16 sum) | 2,621,440 | 533757 / 2192253 | 0 | 540575 | 2,049,739 / 7,617,984 | 491,699 | 0/0 | 0 | 0 | gauss:41237, gauss*0.01:41627, laplace:36854, gauss_25%+0_25%-0:41064, one_nonzero_rest-0:41848, inf_at_100:48403, nan_at_200:32334, gauss*1e30:36260, -row0:41836, adv_all_products-0(row r0):43438, adv_exact_cancel_then-0(row r0):45112, denormals_10%:38080, adv_all_products-0(row r1):45664 |
| dg_moe<256> (masked decode) : FC1 GEMM output (bf16, 6 routes x [gate|up] 4096) | 15,728,640 | 204497 / 1716525 | 0 | 195779 | 7,442,905 / 23,197,760 | 3,075,924 | 8/8 | 0 | 0 | gauss:18944, gauss*0.01:22013, laplace:14592, gauss_25%+0_25%-0:17600, one_nonzero_rest-0:14932, inf_at_100:17920, nan_at_200:9152, gauss*1e30:15232, -row0:14400, adv_exact_cancel_then-0(row r0):16448, denormals_10%:15424, adv_all_products-0(row r0):14208, adv_all_products-0(row r1):13632 |
| dg_moe<256> (masked decode) : FC2 GEMM output (bf16, 6 routes x 4096) | 15,728,640 | 846380 / 5126900 | 9 | 852405 | 11,552,805 / 41,991,216 | 3,209,521 | 26/22 | 0 | 0 | gauss:70843, gauss*0.01:69734, laplace:62526, gauss_25%+0_25%-0:61570, one_nonzero_rest-0:61681, inf_at_100:77725, nan_at_200:50523, gauss*1e30:58805, -row0:65737, adv_all_products-0(row r0):70646, adv_exact_cancel_then-0(row r0):61328, denormals_10%:61716, adv_all_products-0(row r1):73546 |
| dg_moe_contiguous_fast<256> (decode default) : routed output (fp32 of bf16 sum) | 2,621,440 | 0 / 0 | 0 | 0 | 2,032,215 / 7,121,499 | 491,694 | 0/0 | 0 | 0 | - |
| dg_moe_contiguous_fast<256> (decode default) : FC1 GEMM output (bf16, 6 routes x [gate|up] 4096) | 15,728,640 | 0 / 0 | 0 | 0 | 7,357,999 / 21,758,713 | 3,075,953 | 0/0 | 0 | 0 | - |
| dg_moe_contiguous_fast<256> (decode default) : FC2 GEMM output (bf16, 6 routes x 4096) | 15,728,640 | 0 / 0 | 0 | 0 | 11,646,241 / 40,024,410 | 2,951,295 | 0/0 | 0 | 0 | - |
| dg_moe_batch_contiguous grouped (host ids, prefill default) : routed output (fp32 of bf16 sum) | 2,621,440 | 0 / 0 | 0 | 0 | 2,032,215 / 7,121,499 | 491,694 | 0/0 | 0 | 0 | - |
| dg_moe_batch_contiguous legacy replicated : routed output (fp32 of bf16 sum) | 2,621,440 | 0 / 0 | 0 | 0 | 2,032,215 / 7,121,499 | 491,694 | 0/0 | 0 | 0 | - |

### generic_api_default.json  (api, generic sm_120, env {'DSV4_CUDA_PIN_HOST': '0'}, 85.3 s)

paths

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| dsv4_cuda_expert_group(4 experts, H=5120, I=2304) | 3,276,800 | 0 / 0 | 0 | 0 | 2,638,052 / 13,892,293 | 614,801 | 0/0 | 0 | 0 | - |
| dsv4_cuda_moe fused v4f_moe_kernel (6 x 4096/2048, cropped) | 2,621,440 | 0 / 0 | 0 | 0 | 228,348 / 520,398 | 491,520 | 0/0 | 0 | 0 | - |
| dsv4_cuda_moe launches path (FUSED6=0, mv_fp4_grouped<8>; 6 x 4096/2048, cropped) | 2,621,440 | 0 / 0 | 0 | 0 | 236,086 / 541,056 | 491,523 | 0/0 | 0 | 0 | - |

### generic_api_tc.json  (api, generic sm_120, env {'DSV4_CUDA_PIN_HOST': '0'}, 81.5 s)

paths

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| dsv4_cuda_expert_group(4 experts, H=5120, I=2304) | 3,276,800 | 0 / 0 | 0 | 0 | 2,649,039 / 14,237,895 | 614,825 | 0/0 | 0 | 0 | - |
| dsv4_cuda_moe fused v4f_moe_kernel (6 x 4096/2048, cropped) | 2,621,440 | 0 / 0 | 0 | 0 | 228,348 / 520,398 | 491,520 | 0/0 | 0 | 0 | - |
| dsv4_cuda_moe launches path (FUSED6=0, mv_fp4_grouped<8>; 6 x 4096/2048, cropped) | 2,621,440 | 0 / 0 | 0 | 0 | 236,086 / 541,056 | 491,523 | 0/0 | 0 | 0 | - |

### generic_api_unbatched.json  (api, generic sm_120, env {'DSV4_CUDA_PIN_HOST': '0'}, 77.7 s)

paths

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| dsv4_cuda_expert_group(4 experts, H=5120, I=2304) | 3,276,800 | 0 / 0 | 0 | 0 | 2,638,053 / 13,892,247 | 614,802 | 0/0 | 0 | 0 | - |
| dsv4_cuda_moe fused v4f_moe_kernel (6 x 4096/2048, cropped) | 2,621,440 | 0 / 0 | 0 | 0 | 228,348 / 520,398 | 491,520 | 0/0 | 0 | 0 | - |
| dsv4_cuda_moe launches path (FUSED6=0, mv_fp4_grouped<8>; 6 x 4096/2048, cropped) | 2,621,440 | 0 / 0 | 0 | 0 | 236,086 / 541,056 | 491,523 | 0/0 | 0 | 0 | - |

### generic_kernels.json  (kernels, generic sm_120, env {}, 88.1 s)
code-8 nibbles / nibbles: {'real_matrices': (329904941, 5662310400), 'synthetic_zero_rows': (5305414, 19005440)} experts: 160 shapes: {'w1': [2304, 5120], 'w2': [5120, 2304], 'w3': [2304, 5120]}

real_matrices

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| mv<4> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,402 / 121,979,648 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_warp8<4> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,165 / 121,933,268 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_rows<4,4> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,402 / 121,979,648 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_rows<4,8> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,402 / 121,979,648 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_rows<4,16> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,402 / 121,979,648 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_rows<4,32> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,402 / 121,979,648 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_fp4_grouped<1> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,137 / 121,955,016 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_fp4_grouped<2> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,137 / 121,955,016 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_fp4_grouped<4> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,137 / 121,955,016 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_fp4_grouped<8> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,137 / 121,955,016 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mm_batch<4> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,402 / 121,979,648 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_fp4_expert_major<2> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,084 / 121,973,998 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_fp4_expert_major<3> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,084 / 121,973,998 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_fp4_expert_major<4> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,084 / 121,973,998 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| mv_fp4_expert_major<6> | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,084 / 121,973,998 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| v4f_dot_fp4(warp/row wrapper) | 24,903,680 | 0 / 0 | 0 | 0 | 12,177,165 / 121,933,268 | 3,317,449 | 0/0 | 1,738,483 | 0 | - |
| TC cuBLASLt MXFP8 (tc_quant+tc_expand+cublasLtMatmul) | 24,903,680 | 0 / 0 | 0 | 0 | 13,733,284 / 59,668,693 | 4,873,931 | 0/0 | 0 | 0 | - |

synthetic_zero_rows

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| mv<4> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,166 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_warp8<4> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 531,821 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_rows<4,4> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,166 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_rows<4,8> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,166 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_rows<4,16> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,166 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_rows<4,32> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,166 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_fp4_grouped<1> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,020 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_fp4_grouped<2> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,020 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_fp4_grouped<4> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,020 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_fp4_grouped<8> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,020 | 41,296 | 0/0 | 7,974 | 0 | - |
| mm_batch<4> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,166 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_fp4_expert_major<2> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,191 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_fp4_expert_major<3> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,191 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_fp4_expert_major<4> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,191 | 41,296 | 0/0 | 7,974 | 0 | - |
| mv_fp4_expert_major<6> | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 532,191 | 41,296 | 0/0 | 7,974 | 0 | - |
| v4f_dot_fp4(warp/row wrapper) | 81,920 | 0 / 0 | 0 | 0 | 45,811 / 531,821 | 41,296 | 0/0 | 7,974 | 0 | - |

not run: {'TC cuBLASLt MXFP8 (tc_quant+tc_expand+cublasLtMatmul)': ['64x2304', '64x5120']}

### shipped_dg_default.json  (shipped, shipped app DLL, env {'DSV4_CUDA_PIN_HOST': '0'}, 60.0 s)

paths

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| dsv4_cuda_expert_group(4 experts, H=5120, I=2304) | 3,276,800 | 0 / 0 | 0 | 0 | 2,638,052 / 13,892,293 | 614,801 | 0/0 | 0 | 0 | - |
| dsv4_cuda_moe fused v4f_moe_kernel (6 x 4096/2048, cropped) | 2,621,440 | 0 / 0 | 0 | 0 | 228,348 / 520,398 | 491,520 | 0/0 | 0 | 0 | - |

### shipped_generic_default.json  (shipped, shipped app DLL, env {'DSV4_CUDA_PIN_HOST': '0'}, 56.1 s)

paths

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| dsv4_cuda_expert_group(4 experts, H=5120, I=2304) | 3,276,800 | 0 / 0 | 0 | 0 | 2,638,052 / 13,892,293 | 614,801 | 0/0 | 0 | 0 | - |
| dsv4_cuda_moe fused v4f_moe_kernel (6 x 4096/2048, cropped) | 2,621,440 | 0 / 0 | 0 | 0 | 228,348 / 520,398 | 491,520 | 0/0 | 0 | 0 | - |

### shipped_generic_tc.json  (shipped, shipped app DLL, env {'DSV4_CUDA_PIN_HOST': '0', 'DSV4_CUDA_TC': '1'}, 63.4 s)

paths

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| dsv4_cuda_expert_group(4 experts, H=5120, I=2304) | 3,276,800 | 0 / 0 | 0 | 0 | 2,649,039 / 14,237,895 | 614,825 | 0/0 | 0 | 0 | - |
| dsv4_cuda_moe fused v4f_moe_kernel (6 x 4096/2048, cropped) | 2,621,440 | 0 / 0 | 0 | 0 | 228,348 / 520,398 | 491,520 | 0/0 | 0 | 0 | - |

### shipped_generic_unbatched_nofused.json  (shipped, shipped app DLL, env {'DSV4_CUDA_BATCHED': '0', 'DSV4_CUDA_MOE_FUSED6': '0', 'DSV4_CUDA_PIN_HOST': '0'}, 52.5 s)

paths

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| dsv4_cuda_expert_group(4 experts, H=5120, I=2304) | 3,276,800 | 0 / 0 | 0 | 0 | 2,638,053 / 13,892,247 | 614,802 | 0/0 | 0 | 0 | - |
| dsv4_cuda_moe launches path (FUSED6=0, mv_fp4_grouped<8>; 6 x 4096/2048, cropped) | 2,621,440 | 0 / 0 | 0 | 0 | 228,428 / 519,714 | 491,520 | 0/0 | 0 | 0 | - |

### smoke_dg_moe.json  (dg, deepgemm sm_120a, env {'DSV4_CUDA_PIN_HOST': '0'}, 1.0 s)

paths

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| dg_moe<256> (masked decode) : routed output (fp32 of bf16 sum) | 65,536 | 0 / 0 | 0 | 0 | 50,000 / 167,105 | 12,288 | 0/0 | 0 | 0 | - |
| dg_moe<256> (masked decode) : FC1 GEMM output (bf16, 6 routes x [gate|up] 4096) | 393,216 | 0 / 0 | 0 | 0 | 170,956 / 484,605 | 77,254 | 0/0 | 0 | 0 | - |
| dg_moe<256> (masked decode) : FC2 GEMM output (bf16, 6 routes x 4096) | 393,216 | 0 / 0 | 0 | 0 | 281,351 / 916,969 | 73,781 | 0/0 | 0 | 0 | - |
| dg_moe_contiguous_fast<256> (decode default) : routed output (fp32 of bf16 sum) | 65,536 | 0 / 0 | 0 | 0 | 50,000 / 167,105 | 12,288 | 0/0 | 0 | 0 | - |
| dg_moe_contiguous_fast<256> (decode default) : FC1 GEMM output (bf16, 6 routes x [gate|up] 4096) | 393,216 | 0 / 0 | 0 | 0 | 170,956 / 484,605 | 77,254 | 0/0 | 0 | 0 | - |
| dg_moe_contiguous_fast<256> (decode default) : FC2 GEMM output (bf16, 6 routes x 4096) | 393,216 | 0 / 0 | 0 | 0 | 281,351 / 916,969 | 73,781 | 0/0 | 0 | 0 | - |
| dg_moe_batch_contiguous grouped (host ids, prefill default) : routed output (fp32 of bf16 sum) | 65,536 | 0 / 0 | 0 | 0 | 50,000 / 167,105 | 12,288 | 0/0 | 0 | 0 | - |
| dg_moe_batch_contiguous legacy replicated : routed output (fp32 of bf16 sum) | 65,536 | 0 / 0 | 0 | 0 | 50,000 / 167,105 | 12,288 | 0/0 | 0 | 0 | - |

### smoke_generic_kernels.json  (kernels, generic sm_120, env {}, 0.9 s)
code-8 nibbles / nibbles: {'real_matrices': (2069142, 35389440), 'synthetic_zero_rows': (132506, 475136)} experts: 1 shapes: {'w1': [2304, 5120], 'w2': [5120, 2304], 'w3': [2304, 5120]}

real_matrices

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| mv<4> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,317 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_warp8<4> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,508 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_rows<4,4> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,317 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_rows<4,8> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,317 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_rows<4,16> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,317 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_rows<4,32> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,317 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_fp4_grouped<1> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,762 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_fp4_grouped<2> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,762 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_fp4_grouped<4> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,762 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_fp4_grouped<8> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,762 | 20,829 | 0/0 | 10,793 | 0 | - |
| mm_batch<4> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,317 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_fp4_expert_major<2> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,587 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_fp4_expert_major<3> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,587 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_fp4_expert_major<4> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,587 | 20,829 | 0/0 | 10,793 | 0 | - |
| mv_fp4_expert_major<6> | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,587 | 20,829 | 0/0 | 10,793 | 0 | - |
| v4f_dot_fp4(warp/row wrapper) | 155,648 | 0 / 0 | 0 | 0 | 75,464 / 807,508 | 20,829 | 0/0 | 10,793 | 0 | - |
| TC cuBLASLt MXFP8 (tc_quant+tc_expand+cublasLtMatmul) | 155,648 | 0 / 0 | 0 | 0 | 85,192 / 375,301 | 30,557 | 0/0 | 0 | 0 | - |

synthetic_zero_rows

| kernel / path | outputs compared (per version) | orig vs canon: differing values / bits | of which zero-sign only | rerun (A vs A) differing | positive control: differing values / bits | exact-zero outputs (A) | -0 outputs A/B | NaN outputs (A) | unwritten | canon diffs by act set |
|---|---|---|---|---|---|---|---|---|---|---|
| mv<4> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,145 | 1,033 | 0/0 | 198 | 0 | - |
| mv_warp8<4> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,130 | 1,033 | 0/0 | 198 | 0 | - |
| mv_rows<4,4> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,145 | 1,033 | 0/0 | 198 | 0 | - |
| mv_rows<4,8> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,145 | 1,033 | 0/0 | 198 | 0 | - |
| mv_rows<4,16> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,145 | 1,033 | 0/0 | 198 | 0 | - |
| mv_rows<4,32> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,145 | 1,033 | 0/0 | 198 | 0 | - |
| mv_fp4_grouped<1> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,147 | 1,033 | 0/0 | 198 | 0 | - |
| mv_fp4_grouped<2> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,147 | 1,033 | 0/0 | 198 | 0 | - |
| mv_fp4_grouped<4> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,147 | 1,033 | 0/0 | 198 | 0 | - |
| mv_fp4_grouped<8> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,147 | 1,033 | 0/0 | 198 | 0 | - |
| mm_batch<4> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,145 | 1,033 | 0/0 | 198 | 0 | - |
| mv_fp4_expert_major<2> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,145 | 1,033 | 0/0 | 198 | 0 | - |
| mv_fp4_expert_major<3> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,145 | 1,033 | 0/0 | 198 | 0 | - |
| mv_fp4_expert_major<4> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,145 | 1,033 | 0/0 | 198 | 0 | - |
| mv_fp4_expert_major<6> | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,145 | 1,033 | 0/0 | 198 | 0 | - |
| v4f_dot_fp4(warp/row wrapper) | 2,048 | 0 / 0 | 0 | 0 | 972 / 13,130 | 1,033 | 0/0 | 198 | 0 | - |

not run: {'TC cuBLASLt MXFP8 (tc_quant+tc_expand+cublasLtMatmul)': ['64x2304', '64x5120']}
